/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edsculpt
 */

#include <limits>

#include "BLI_array.hh"
#include "BLI_index_mask.hh"
#include "BLI_listbase.hh"
#include "BLI_math_base.hh"
#include "BLI_math_geom.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_offset_indices.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "DNA_image_types.h"
#include "DNA_mesh_types.h"
#include "DNA_multires_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "BKE_attribute.hh"
#include "BKE_context.hh"
#include "BKE_image.hh"
#include "BKE_lib_id.hh"
#include "BKE_main.hh"
#include "BKE_mesh.hh"
#include "BKE_mesh_tangent.hh"
#include "BKE_multiresolution.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_report.hh"
#include "BKE_sculpt_layer.hh"

#include "DEG_depsgraph.hh"

#include "IMB_imbuf.hh"
#include "IMB_imbuf_types.hh"
#include "IMB_partial_update.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "atomic_ops.h"

#include "sculpt_multires.hh"

namespace blender::ed::sculpt_paint::multires {

namespace {

static const EnumPropertyItem multires_bake_type_items[] = {
    {MULTIRES_BAKE_NORMALS, "NORMALS", 0, "Normals", "Bake tangent-space normals"},
    {MULTIRES_BAKE_DISPLACEMENT, "DISPLACEMENT", 0, "Displacement", "Bake height displacement"},
    {MULTIRES_BAKE_VECTOR_DISPLACEMENT,
     "VECTOR_DISPLACEMENT",
     0,
     "Vector Displacement",
     "Bake vector displacement"},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem multires_bake_space_items[] = {
    {MULTIRES_BAKE_SPACE_TANGENT, "TANGENT", 0, "Tangent", "Tangent space"},
    {MULTIRES_BAKE_SPACE_OBJECT, "OBJECT", 0, "Object", "Object space"},
    {0, nullptr, 0, nullptr, nullptr},
};

constexpr int BUCKET_SIZE = 32;
constexpr int BUCKET_AREA = BUCKET_SIZE * BUCKET_SIZE;

constexpr int32_t EMPTY_SLOT = std::numeric_limits<int32_t>::max();

struct BakeSettings {
  MultiresBakeType type = MULTIRES_BAKE_NORMALS;
  MultiresBakeSpace space = MULTIRES_BAKE_SPACE_TANGENT;
  bool use_normalize = false;
  int margin = 16;

  bool need_low_normals() const
  {
    return type != MULTIRES_BAKE_VECTOR_DISPLACEMENT || space == MULTIRES_BAKE_SPACE_TANGENT;
  }
  bool need_low_tangents() const
  {
    return type == MULTIRES_BAKE_NORMALS ||
           (type == MULTIRES_BAKE_VECTOR_DISPLACEMENT && space == MULTIRES_BAKE_SPACE_TANGENT);
  }
  bool need_high_normals() const
  {
    return type == MULTIRES_BAKE_NORMALS;
  }
};

struct LowCage {
  Span<float3> positions;
  Span<float3> vert_normals;
  Span<float3> face_normals;
  Span<bool> sharp;
  Span<float4> tangents;
  Span<float2> uv;
  Span<int> corner_verts;
  OffsetIndices<int> faces;
  int per_side = 1;
  int faces_per_base = 1;

  Mesh *mesh = nullptr;
  Array<float3> positions_storage;
  Array<float3> vert_normals_storage;
  Array<float3> face_normals_storage;
  Array<float4> tangents_storage;
  Array<float2> uv_storage;
  Array<bool> sharp_storage;

  LowCage() = default;
  LowCage(const LowCage &) = delete;
  LowCage &operator=(const LowCage &) = delete;
  ~LowCage()
  {
    if (mesh != nullptr) {
      BKE_id_free(nullptr, mesh);
    }
  }
};

struct HighMesh {
  Array<float3> positions;
  Array<float3> vert_normals;
  Array<int> corner_verts;
  int virt_per_side = 1;
  int top_per_side = 1;
  int faces_per_virtual_grid = 1;
};

struct BakedTile {
  ImBuf *ibuf = nullptr;
  Array<char> mask;
  Array<float> displacement;
};

static uchar unit_float_to_byte(const float v)
{
  return static_cast<uchar>(math::clamp(int(v * 255.0f + 0.5f), 0, 255));
}

static bool is_sculpt(const Object &ob)
{
  return (ob.mode & OB_MODE_SCULPT) && ob.runtime->sculpt_session;
}

template<typename T> static T interp_tri(const T &a, const T &b, const T &c, const float3 &w)
{
  return a * w.x + b * w.y + c * w.z;
}

static float3 bilinear(const float3 &p0,
                       const float3 &p1,
                       const float3 &p2,
                       const float3 &p3,
                       const float fu,
                       const float fv)
{
  const float w0 = (1.0f - fu) * (1.0f - fv);
  const float w1 = fu * (1.0f - fv);
  const float w2 = fu * fv;
  const float w3 = (1.0f - fu) * fv;
  return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
}

static void calc_area_normals(const Span<float3> positions,
                              const OffsetIndices<int> faces,
                              const Span<int> corner_verts,
                              Array<float3> &r_vert_normals,
                              Array<float3> &r_face_normals)
{
  r_face_normals.reinitialize(faces.size());
  threading::parallel_for(faces.index_range(), 1024, [&](const IndexRange range) {
    for (const int f : range) {
      const IndexRange face = faces[f];
      float3 n(0.0f);
      for (const int i : IndexRange(face.size())) {
        const float3 &p0 = positions[corner_verts[face[i]]];
        const float3 &p1 = positions[corner_verts[face[(i + 1) % face.size()]]];
        n.x += (p0.y - p1.y) * (p0.z + p1.z);
        n.y += (p0.z - p1.z) * (p0.x + p1.x);
        n.z += (p0.x - p1.x) * (p0.y + p1.y);
      }
      r_face_normals[f] = n;
    }
  });

  r_vert_normals.reinitialize(positions.size());
  r_vert_normals.as_mutable_span().fill(float3(0.0f));
  for (const int f : faces.index_range()) {
    for (const int c : faces[f]) {
      r_vert_normals[corner_verts[c]] += r_face_normals[f];
    }
  }

  threading::parallel_for(r_vert_normals.index_range(), 4096, [&](const IndexRange range) {
    for (const int i : range) {
      r_vert_normals[i] = math::normalize(r_vert_normals[i]);
    }
  });
  threading::parallel_for(r_face_normals.index_range(), 4096, [&](const IndexRange range) {
    for (const int i : range) {
      r_face_normals[i] = math::normalize(r_face_normals[i]);
    }
  });
}

static void expand_vert_attr_lvl1_linear(const Span<int2> base_edges,
                                         const OffsetIndices<int> base_faces,
                                         const Span<int> base_corner_verts,
                                         const Span<float3> in_attr,
                                         MutableSpan<float3> out_attr)
{
  const int verts_num = in_attr.size();
  const int edges_num = base_edges.size();
  out_attr.slice(0, verts_num).copy_from(in_attr);
  for (const int edge_idx : base_edges.index_range()) {
    const int2 edge = base_edges[edge_idx];
    out_attr[verts_num + edge_idx] = (in_attr[edge[0]] + in_attr[edge[1]]) * 0.5f;
  }
  for (const int face_idx : base_faces.index_range()) {
    const IndexRange face = base_faces[face_idx];
    float3 sum(0.0f);
    for (const int c : face) {
      sum += in_attr[base_corner_verts[c]];
    }
    out_attr[verts_num + edges_num + face_idx] = sum / float(face.size());
  }
}

template<typename T>
static void expand_corner_attr_lvl1_linear(const OffsetIndices<int> base_faces,
                                           const Span<T> in_attr,
                                           MutableSpan<T> out_attr)
{
  for (const int face_idx : base_faces.index_range()) {
    const IndexRange face = base_faces[face_idx];
    const int face_start = face.start();
    const int corners_num = face.size();
    T center(0.0f);
    for (const int corner : face) {
      center += in_attr[corner];
    }
    center = center * (1.0f / float(corners_num));

    T *out = &out_attr[face_start * 4];
    if (corners_num == 4) {
      const T &v0 = in_attr[face_start + 0];
      const T &v1 = in_attr[face_start + 1];
      const T &v2 = in_attr[face_start + 2];
      const T &v3 = in_attr[face_start + 3];
      const T m0 = (v0 + v1) * 0.5f;
      const T m1 = (v1 + v2) * 0.5f;
      const T m2 = (v2 + v3) * 0.5f;
      const T m3 = (v3 + v0) * 0.5f;

      out[0] = v0;
      out[1] = m0;
      out[2] = center;
      out[3] = m3;
      out[4] = m0;
      out[5] = v1;
      out[6] = m1;
      out[7] = center;
      out[8] = m3;
      out[9] = center;
      out[10] = m2;
      out[11] = v3;
      out[12] = center;
      out[13] = m1;
      out[14] = v2;
      out[15] = m2;
    }
    else {
      for (int corner = 0; corner < corners_num; corner++) {
        const int next_k = (corner + 1) % corners_num;
        const int prev_k = (corner + corners_num - 1) % corners_num;
        const T &v_curr = in_attr[face_start + corner];
        const T &v_next = in_attr[face_start + next_k];
        const T &v_prev = in_attr[face_start + prev_k];

        out[corner * 4 + 0] = v_curr;
        out[corner * 4 + 1] = (v_curr + v_next) * 0.5f;
        out[corner * 4 + 2] = center;
        out[corner * 4 + 3] = (v_prev + v_curr) * 0.5f;
      }
    }
  }
}

static int64_t tiled_index(const int2 xy, const int tiles_across)
{
  const int tile_x = xy.x / BUCKET_SIZE;
  const int tile_y = xy.y / BUCKET_SIZE;
  const int local = ((xy.y & (BUCKET_SIZE - 1)) * BUCKET_SIZE) | (xy.x & (BUCKET_SIZE - 1));
  return int64_t(tile_y * tiles_across + tile_x) * BUCKET_AREA + local;
}

static int2 bucket_slot_xy(const int bucket, const int slot, const int tiles_across)
{
  return int2((bucket % tiles_across) * BUCKET_SIZE + (slot & (BUCKET_SIZE - 1)),
              (bucket / tiles_across) * BUCKET_SIZE + (slot >> 5));
}

static int3 low_tri_corners(const OffsetIndices<int> faces, const int tri)
{
  const IndexRange face = faces[tri >> 1];
  if ((tri & 1) == 0) {
    return int3(face[0], face[1], face[2]);
  }
  return int3(face[0], face[2], face[3]);
}

static void claim_slot(int32_t &slot, const int32_t tri)
{
  int32_t old = atomic_load_int32(&slot);
  while (tri < old) {
    const int32_t prev = atomic_cas_int32(&slot, old, tri);
    if (prev == old) {
      break;
    }
    old = prev;
  }
}

static void rasterize_low_cage(const LowCage &cage,
                               const float2 uv_offset,
                               const int width,
                               const int height,
                               const int tiles_across,
                               MutableSpan<int32_t> r_tri_tiled)
{
  const int low_tris_num = cage.faces.size() * 2;
  const float inv_w = 1.0f / float(width);
  const float inv_h = 1.0f / float(height);

  threading::parallel_for(IndexRange(low_tris_num), 256, [&](const IndexRange range) {
    for (const int tri : range) {
      const int3 c = low_tri_corners(cage.faces, tri);
      const float2 uv0 = cage.uv[c.x] - uv_offset;
      const float2 uv1 = cage.uv[c.y] - uv_offset;
      const float2 uv2 = cage.uv[c.z] - uv_offset;

      const float2 p0(uv0.x * float(width) - 0.5f, uv0.y * float(height) - 0.5f);
      const float2 p1(uv1.x * float(width) - 0.5f, uv1.y * float(height) - 0.5f);
      const float2 p2(uv2.x * float(width) - 0.5f, uv2.y * float(height) - 0.5f);

      const float area2 = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
      if (math::abs(area2) < 1e-8f) {
        continue;
      }

      int x0 = int(math::floor(math::min(math::min(p0.x, p1.x), p2.x)));
      int x1 = int(math::ceil(math::max(math::max(p0.x, p1.x), p2.x)));
      int y0 = int(math::floor(math::min(math::min(p0.y, p1.y), p2.y)));
      int y1 = int(math::ceil(math::max(math::max(p0.y, p1.y), p2.y)));
      x0 = math::max(x0, 0);
      y0 = math::max(y0, 0);
      x1 = math::min(x1, width);
      y1 = math::min(y1, height);

      for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
          const float2 st((float(x) + 0.5f) * inv_w, (float(y) + 0.5f) * inv_h);
          const float3 w = math::barycentric_weights(uv0, uv1, uv2, st);
          if (w.x < 0.0f || w.y < 0.0f || w.z < 0.0f) {
            continue;
          }
          claim_slot(r_tri_tiled[tiled_index(int2(x, y), tiles_across)], tri);
        }
      }
    }
  });
}

static int gather_bucket_samples(const Span<int32_t> bucket_tris,
                                 const int bucket,
                                 const int tiles_across,
                                 const int width,
                                 const int height,
                                 const LowCage &cage,
                                 const float2 uv_offset,
                                 MutableSpan<int64_t> r_pixels,
                                 MutableSpan<int> r_tris,
                                 MutableSpan<float3> r_weights)
{
  const float2 inv_size(1.0f / float(width), 1.0f / float(height));
  int count = 0;
  for (int slot = 0; slot < BUCKET_AREA; slot++) {
    const int tri = bucket_tris[slot];
    if (tri == EMPTY_SLOT) {
      continue;
    }
    const int2 xy = bucket_slot_xy(bucket, slot, tiles_across);
    const int3 c = low_tri_corners(cage.faces, tri);
    const float2 uv0 = cage.uv[c.x] - uv_offset;
    const float2 uv1 = cage.uv[c.y] - uv_offset;
    const float2 uv2 = cage.uv[c.z] - uv_offset;
    const float2 st = (float2(xy) + float2(0.5f)) * inv_size;

    r_pixels[count] = int64_t(xy.y) * width + xy.x;
    r_tris[count] = tri;
    r_weights[count] = math::barycentric_weights(uv0, uv1, uv2, st);
    count++;
  }
  return count;
}

static void calc_high_sample_coords(const LowCage &cage,
                                    const Span<int> tris,
                                    const Span<float3> weights,
                                    MutableSpan<int> r_base_faces,
                                    MutableSpan<float2> r_st)
{
  const float inv_low_per_side = 1.0f / float(cage.per_side);
  for (const int i : tris.index_range()) {
    const int tri = tris[i];
    const int low_face = tri >> 1;
    const float3 &w = weights[i];

    const float2 st_low = (tri & 1) == 0 ? float2(w.y + w.z, w.z) : float2(w.y, w.y + w.z);

    const int base_face = low_face / cage.faces_per_base;
    const int low_rem = low_face % cage.faces_per_base;
    const int row_low = low_rem / cage.per_side;
    const int col_low = low_rem % cage.per_side;
    r_base_faces[i] = base_face;
    r_st[i] = (float2(float(col_low), float(row_low)) + st_low) * inv_low_per_side;
  }
}

static void sample_high_attr(const HighMesh &high,
                             const Span<float3> top_attr,
                             const Span<int> base_faces,
                             const Span<float2> st_coords,
                             MutableSpan<float3> r_out)
{
  const int virt_per_side = high.virt_per_side;
  const int top_per_side = high.top_per_side;
  for (const int i : base_faces.index_range()) {
    const float s = math::clamp(st_coords[i].x, 0.0f, 1.0f);
    const float t = math::clamp(st_coords[i].y, 0.0f, 1.0f);
    const int col_virt = math::min(int(s * virt_per_side), virt_per_side - 1);
    const int row_virt = math::min(int(t * virt_per_side), virt_per_side - 1);
    const float s_virt = s * virt_per_side - col_virt;
    const float t_virt = t * virt_per_side - row_virt;
    const int col_top = math::min(int(s_virt * top_per_side), top_per_side - 1);
    const int row_top = math::min(int(t_virt * top_per_side), top_per_side - 1);
    const float fs = s_virt * top_per_side - col_top;
    const float ft = t_virt * top_per_side - row_top;
    const int64_t virt_face = int64_t(base_faces[i]) * virt_per_side * virt_per_side +
                              int64_t(row_virt) * virt_per_side + col_virt;
    const int64_t sub = virt_face * high.faces_per_virtual_grid + int64_t(row_top) * top_per_side +
                        col_top;
    const int v0 = high.corner_verts[sub * 4 + 0];
    const int v1 = high.corner_verts[sub * 4 + 1];
    const int v2 = high.corner_verts[sub * 4 + 2];
    const int v3 = high.corner_verts[sub * 4 + 3];

    r_out[i] = bilinear(top_attr[v0], top_attr[v1], top_attr[v2], top_attr[v3], fs, ft);
  }
}

static void calc_low_positions(const LowCage &cage,
                               const Span<int> tris,
                               const Span<float3> weights,
                               MutableSpan<float3> r_positions)
{
  for (const int i : tris.index_range()) {
    const int3 c = low_tri_corners(cage.faces, tris[i]);
    r_positions[i] = interp_tri(cage.positions[cage.corner_verts[c.x]],
                                cage.positions[cage.corner_verts[c.y]],
                                cage.positions[cage.corner_verts[c.z]],
                                weights[i]);
  }
}

static void calc_low_normals(const LowCage &cage,
                             const Span<int> tris,
                             const Span<float3> weights,
                             MutableSpan<float3> r_normals)
{
  for (const int i : tris.index_range()) {
    const int tri = tris[i];
    const int low_face = tri >> 1;
    if (cage.sharp[low_face]) {
      r_normals[i] = cage.face_normals[low_face];
    }
    else {
      const int3 c = low_tri_corners(cage.faces, tri);
      r_normals[i] = math::normalize(interp_tri(cage.vert_normals[cage.corner_verts[c.x]],
                                                cage.vert_normals[cage.corner_verts[c.y]],
                                                cage.vert_normals[cage.corner_verts[c.z]],
                                                weights[i]));
    }
  }
}

static void calc_low_tangents(const LowCage &cage,
                              const Span<int> tris,
                              const Span<float3> weights,
                              MutableSpan<float4> r_tangents)
{
  for (const int i : tris.index_range()) {
    const int3 c = low_tri_corners(cage.faces, tris[i]);
    r_tangents[i] = interp_tri(
        cage.tangents[c.x], cage.tangents[c.y], cage.tangents[c.z], weights[i]);
  }
}

static void bake_tangent_normals(const Span<float3> low_normals,
                                 const Span<float4> low_tangents,
                                 const Span<float3> high_normals,
                                 MutableSpan<float3> r_baked)
{
  for (const int i : low_normals.index_range()) {
    const float3 &low_normal = low_normals[i];
    const float4 &low_tangent = low_tangents[i];

    float3 T = low_tangent.xyz();
    T -= low_normal * math::dot(low_normal, T);
    T = math::normalize(T);
    const float sign = (low_tangent.w < 0.0f) ? -1.0f : 1.0f;
    const float3 B = sign * math::cross(low_normal, T);

    const float3 &high_normal = high_normals[i];
    const float3 tangent_normal(
        math::dot(T, high_normal), math::dot(B, high_normal), math::dot(low_normal, high_normal));
    r_baked[i] = math::normalize(tangent_normal) * 0.5f + float3(0.5f);
  }
}

static void calc_scalar_displacements(const Span<float3> low_positions,
                                      const Span<float3> low_normals,
                                      const Span<float3> high_positions,
                                      MutableSpan<float> r_heights)
{
  for (const int i : low_positions.index_range()) {
    r_heights[i] = math::dot(low_normals[i], high_positions[i] - low_positions[i]);
  }
}

static void calc_vector_displacements(const Span<float3> low_positions,
                                      const Span<float3> high_positions,
                                      MutableSpan<float3> r_displacements)
{
  for (const int i : low_positions.index_range()) {
    r_displacements[i] = high_positions[i] - low_positions[i];
  }
}

static void transform_vector_displacements_to_tangent(const Span<float3> low_normals,
                                                      const Span<float4> low_tangents,
                                                      MutableSpan<float3> r_displacements)
{
  for (const int i : low_normals.index_range()) {
    const float3 &low_normal = low_normals[i];
    const float4 &low_tangent = low_tangents[i];

    float3 T = low_tangent.xyz();
    T -= low_normal * math::dot(low_normal, T);
    T = math::normalize(T);
    const float sign = (low_tangent.w < 0.0f) ? -1.0f : 1.0f;
    const float3 B = sign * math::cross(low_normal, T);

    const float3 d = r_displacements[i];
    r_displacements[i] = float3(math::dot(T, d), math::dot(low_normal, d), math::dot(B, d));
  }
}

static void store_pixels(const MutableSpan<float4> float_px,
                         const MutableSpan<uchar4> byte_px,
                         const Span<int64_t> pixels,
                         const Span<float3> values)
{
  if (!float_px.is_empty()) {
    for (const int i : pixels.index_range()) {
      const float3 &v = values[i];
      float_px[pixels[i]] = float4(v.x, v.y, v.z, 1.0f);
    }
  }
  if (!byte_px.is_empty()) {
    for (const int i : pixels.index_range()) {
      const float3 &v = values[i];
      byte_px[pixels[i]] = uchar4(
          unit_float_to_byte(v.x), unit_float_to_byte(v.y), unit_float_to_byte(v.z), 255);
    }
  }
}

static void store_scalar_pixels(const MutableSpan<float4> float_px,
                                const MutableSpan<uchar4> byte_px,
                                const Span<int64_t> pixels,
                                const Span<float> values)
{
  if (!float_px.is_empty()) {
    for (const int i : pixels.index_range()) {
      const float v = values[i];
      float_px[pixels[i]] = float4(v, v, v, 1.0f);
    }
  }
  if (!byte_px.is_empty()) {
    for (const int i : pixels.index_range()) {
      const uchar b = unit_float_to_byte(values[i]);
      byte_px[pixels[i]] = uchar4(b, b, b, 255);
    }
  }
}

static void store_raw_displacements(MutableSpan<float> disp_px,
                                    const Span<int64_t> pixels,
                                    const Span<float> heights,
                                    float2 &r_minmax)
{
  float min_h = r_minmax.x;
  float max_h = r_minmax.y;
  for (const int i : pixels.index_range()) {
    const float h = heights[i];
    disp_px[pixels[i]] = h;
    min_h = math::min(min_h, h);
    max_h = math::max(max_h, h);
  }
  r_minmax = float2(min_h, max_h);
}

static void normalize_tile_displacement(ImBuf &ibuf,
                                        const Span<float> displacement,
                                        const Span<char> mask,
                                        const float max_distance)
{
  const int64_t pixels_num = int64_t(ibuf.x) * ibuf.y;
  MutableSpan<float4> float_px;
  MutableSpan<uchar4> byte_px;
  if (float *fp = ibuf.float_data_for_write()) {
    float_px = MutableSpan(reinterpret_cast<float4 *>(fp), pixels_num);
  }
  if (uchar *bp = ibuf.byte_data_for_write()) {
    byte_px = MutableSpan(reinterpret_cast<uchar4 *>(bp), pixels_num);
  }

  const bool zero_range = max_distance <= 1e-5f;
  const float inv_denom = zero_range ? 0.0f : 1.0f / (2.0f * max_distance);

  threading::parallel_for(IndexRange(pixels_num), 4096, [&](const IndexRange range) {
    if (!float_px.is_empty()) {
      for (const int64_t i : range) {
        if (mask[i] == FILTER_MASK_USED) {
          const float norm = zero_range ? 0.5f : (displacement[i] + max_distance) * inv_denom;
          float_px[i] = float4(norm, norm, norm, 1.0f);
        }
      }
    }
    if (!byte_px.is_empty()) {
      for (const int64_t i : range) {
        if (mask[i] == FILTER_MASK_USED) {
          const float norm = zero_range ? 0.5f : (displacement[i] + max_distance) * inv_denom;
          const uchar b = unit_float_to_byte(norm);
          byte_px[i] = uchar4(b, b, b, 255);
        }
      }
    }
  });
}

static void finish_baked_tile(Image &image, ImBuf &ibuf, Array<char> &mask, const int margin)
{
  if (margin > 0) {
    IMB_filter_extend(&ibuf, mask.data(), margin);
  }
  IMB_partial_update_mark_full(&ibuf);
  IMB_mark_dirty(&ibuf);
  BKE_image_release_ibuf(&image, &ibuf, nullptr);
}

static bool bake_tile(const BakeSettings &settings,
                      const LowCage &cage,
                      const HighMesh &high,
                      Image &image,
                      ImageTile &tile,
                      Vector<BakedTile> &r_deferred_tiles,
                      float2 &r_height_minmax)
{
  ImageUser iuser;
  BKE_imageuser_default(&iuser);
  iuser.tile = tile.tile_number;

  ImBuf *ibuf = BKE_image_acquire_ibuf(&image, &iuser, nullptr);
  if (ibuf == nullptr || ibuf->x <= 0 || ibuf->y <= 0) {
    if (ibuf) {
      BKE_image_release_ibuf(&image, ibuf, nullptr);
    }
    return false;
  }
  const int W = ibuf->x;
  const int H = ibuf->y;
  const int64_t pixels_num = int64_t(W) * H;

  float2 uv_offset(0.0f, 0.0f);
  BKE_image_get_tile_uv(&image, tile.tile_number, &uv_offset.x);

  MutableSpan<float4> float_px;
  MutableSpan<uchar4> byte_px;
  if (float *fp = ibuf->float_data_for_write()) {
    float_px = MutableSpan(reinterpret_cast<float4 *>(fp), pixels_num);
  }
  if (uchar *bp = ibuf->byte_data_for_write()) {
    byte_px = MutableSpan(reinterpret_cast<uchar4 *>(bp), pixels_num);
  }

  const int tiles_across = (W + BUCKET_SIZE - 1) / BUCKET_SIZE;
  const int tiles_down = (H + BUCKET_SIZE - 1) / BUCKET_SIZE;
  const int buckets_num = tiles_across * tiles_down;

  Array<int32_t> tri_tiled(int64_t(buckets_num) * BUCKET_AREA, EMPTY_SLOT);
  rasterize_low_cage(cage, uv_offset, W, H, tiles_across, tri_tiled);

  Array<char> mask(pixels_num, FILTER_MASK_NULL);

  const bool defer_normalize = (settings.type == MULTIRES_BAKE_DISPLACEMENT) &&
                               settings.use_normalize;
  Array<float> displacement;
  Array<float2> bucket_minmax;
  if (defer_normalize) {
    displacement.reinitialize(pixels_num);
    bucket_minmax.reinitialize(buckets_num);
    bucket_minmax.as_mutable_span().fill(float2(FLT_MAX, -FLT_MAX));
  }

  threading::parallel_for(IndexRange(buckets_num), 8, [&](const IndexRange range) {
    for (const int bucket : range) {
      const Span<int32_t> bucket_tris = tri_tiled.as_span().slice(int64_t(bucket) * BUCKET_AREA,
                                                                  BUCKET_AREA);
      Array<int64_t, BUCKET_AREA> pixels_buf(BUCKET_AREA);
      Array<int, BUCKET_AREA> tris_buf(BUCKET_AREA);
      Array<float3, BUCKET_AREA> weights_buf(BUCKET_AREA);

      const int count = gather_bucket_samples(bucket_tris,
                                              bucket,
                                              tiles_across,
                                              W,
                                              H,
                                              cage,
                                              uv_offset,
                                              pixels_buf,
                                              tris_buf,
                                              weights_buf);
      if (count == 0) {
        continue;
      }

      const Span<int64_t> pixels = pixels_buf.as_span().take_front(count);
      const Span<int> tris = tris_buf.as_span().take_front(count);
      const Span<float3> weights = weights_buf.as_span().take_front(count);

      for (const int64_t pixel : pixels) {
        mask[pixel] = FILTER_MASK_USED;
      }

      Array<int, BUCKET_AREA> high_base_faces(count);
      Array<float2, BUCKET_AREA> high_st(count);
      calc_high_sample_coords(cage, tris, weights, high_base_faces, high_st);

      switch (settings.type) {
        case MULTIRES_BAKE_NORMALS: {
          Array<float3, BUCKET_AREA> low_normals(count);
          Array<float4, BUCKET_AREA> low_tangents(count);
          Array<float3, BUCKET_AREA> high_normals(count);
          Array<float3, BUCKET_AREA> baked(count);

          calc_low_normals(cage, tris, weights, low_normals);
          calc_low_tangents(cage, tris, weights, low_tangents);
          sample_high_attr(high, high.vert_normals, high_base_faces, high_st, high_normals);
          bake_tangent_normals(low_normals, low_tangents, high_normals, baked);
          store_pixels(float_px, byte_px, pixels, baked);
          break;
        }
        case MULTIRES_BAKE_DISPLACEMENT: {
          Array<float3, BUCKET_AREA> low_positions(count);
          Array<float3, BUCKET_AREA> low_normals(count);
          Array<float3, BUCKET_AREA> high_positions(count);
          Array<float, BUCKET_AREA> heights(count);

          calc_low_positions(cage, tris, weights, low_positions);
          calc_low_normals(cage, tris, weights, low_normals);
          sample_high_attr(high, high.positions, high_base_faces, high_st, high_positions);
          calc_scalar_displacements(low_positions, low_normals, high_positions, heights);
          if (defer_normalize) {
            store_raw_displacements(
                displacement.as_mutable_span(), pixels, heights, bucket_minmax[bucket]);
          }
          else {
            store_scalar_pixels(float_px, byte_px, pixels, heights);
          }
          break;
        }
        case MULTIRES_BAKE_VECTOR_DISPLACEMENT: {
          Array<float3, BUCKET_AREA> low_positions(count);
          Array<float3, BUCKET_AREA> high_positions(count);
          Array<float3, BUCKET_AREA> displacements(count);

          calc_low_positions(cage, tris, weights, low_positions);
          sample_high_attr(high, high.positions, high_base_faces, high_st, high_positions);
          calc_vector_displacements(low_positions, high_positions, displacements);
          if (settings.space == MULTIRES_BAKE_SPACE_TANGENT) {
            Array<float3, BUCKET_AREA> low_normals(count);
            Array<float4, BUCKET_AREA> low_tangents(count);
            calc_low_normals(cage, tris, weights, low_normals);
            calc_low_tangents(cage, tris, weights, low_tangents);
            transform_vector_displacements_to_tangent(low_normals, low_tangents, displacements);
          }
          store_pixels(float_px, byte_px, pixels, displacements);
          break;
        }
      }
    }
  });

  if (defer_normalize) {
    for (const float2 &mm : bucket_minmax) {
      r_height_minmax.x = math::min(r_height_minmax.x, mm.x);
      r_height_minmax.y = math::max(r_height_minmax.y, mm.y);
    }
    r_deferred_tiles.append({ibuf, std::move(mask), std::move(displacement)});
  }
  else {
    finish_baked_tile(image, *ibuf, mask, settings.margin);
  }

  return true;
}

static void build_low_cage_base(const Mesh &base_mesh,
                                const Multiresolution &multires,
                                const Span<float2> base_uv,
                                const BakeSettings &settings,
                                LowCage &r_cage)
{
  const bke::AttributeAccessor attributes = base_mesh.attributes();

  Array<float3> base_eval_positions(base_mesh.verts_num);
  bke::sculpt_layer::calc_eval_positions(base_mesh, base_eval_positions.as_mutable_span());

  Array<float3> base_vert_normals;
  Array<float3> base_face_normals;
  VArraySpan<bool> sharp;
  Array<Array<float4>> tangent_data;

  if (settings.need_low_normals()) {
    calc_area_normals(base_eval_positions,
                      base_mesh.faces(),
                      base_mesh.corner_verts(),
                      base_vert_normals,
                      base_face_normals);

    sharp = *attributes.lookup_or_default<bool>("sharp_face", bke::AttrDomain::Face, false);

    if (settings.need_low_tangents()) {
      tangent_data = bke::mesh::calc_uv_tangents(base_eval_positions,
                                                 base_mesh.faces(),
                                                 base_mesh.corner_verts(),
                                                 base_mesh.corner_tris(),
                                                 base_mesh.corner_tri_faces(),
                                                 sharp,
                                                 base_vert_normals,
                                                 base_face_normals,
                                                 base_mesh.corner_normals(),
                                                 {base_uv});
    }
  }

  if (multires.base_all_quads) {
    r_cage.positions_storage = std::move(base_eval_positions);
    r_cage.positions = r_cage.positions_storage.as_span();
    r_cage.uv = base_uv;
    r_cage.corner_verts = base_mesh.corner_verts();
    r_cage.faces = base_mesh.faces();

    if (settings.need_low_normals()) {
      r_cage.vert_normals_storage = std::move(base_vert_normals);
      r_cage.face_normals_storage = std::move(base_face_normals);
      r_cage.sharp_storage.reinitialize(base_mesh.faces_num);
      r_cage.sharp_storage.as_mutable_span().copy_from(sharp);

      r_cage.vert_normals = r_cage.vert_normals_storage.as_span();
      r_cage.face_normals = r_cage.face_normals_storage.as_span();
      r_cage.sharp = r_cage.sharp_storage.as_span();

      if (settings.need_low_tangents()) {
        r_cage.tangents_storage = std::move(tangent_data[0]);
        r_cage.tangents = r_cage.tangents_storage.as_span();
      }
    }
    return;
  }

  const Mesh &base1 = multires.runtime->ensure_base_mesh(base_mesh, 1, false);
  const OffsetIndices<int> base_faces = base_mesh.faces();

  r_cage.positions_storage.reinitialize(base1.verts_num);
  expand_vert_attr_lvl1_linear(base_mesh.edges(),
                               base_faces,
                               base_mesh.corner_verts(),
                               base_eval_positions,
                               r_cage.positions_storage.as_mutable_span());

  r_cage.uv_storage.reinitialize(base1.corners_num);
  expand_corner_attr_lvl1_linear(base_faces, base_uv, r_cage.uv_storage.as_mutable_span());

  r_cage.positions = r_cage.positions_storage.as_span();
  r_cage.uv = r_cage.uv_storage.as_span();
  r_cage.corner_verts = base1.corner_verts();
  r_cage.faces = base1.faces();

  if (settings.need_low_normals()) {
    r_cage.vert_normals_storage.reinitialize(base1.verts_num);
    expand_vert_attr_lvl1_linear(base_mesh.edges(),
                                 base_faces,
                                 base_mesh.corner_verts(),
                                 base_vert_normals,
                                 r_cage.vert_normals_storage.as_mutable_span());

    r_cage.face_normals_storage.reinitialize(base1.faces_num);
    r_cage.sharp_storage.reinitialize(base1.faces_num);
    for (const int f : base_faces.index_range()) {
      const IndexRange face = base_faces[f];
      r_cage.face_normals_storage.as_mutable_span()
          .slice(face.start(), face.size())
          .fill(base_face_normals[f]);
      r_cage.sharp_storage.as_mutable_span().slice(face.start(), face.size()).fill(sharp[f]);
    }

    r_cage.vert_normals = r_cage.vert_normals_storage.as_span();
    r_cage.face_normals = r_cage.face_normals_storage.as_span();
    r_cage.sharp = r_cage.sharp_storage.as_span();

    if (settings.need_low_tangents()) {
      r_cage.tangents_storage.reinitialize(base1.corners_num);
      expand_corner_attr_lvl1_linear(
          base_faces, tangent_data[0].as_span(), r_cage.tangents_storage.as_mutable_span());
      r_cage.tangents = r_cage.tangents_storage.as_span();
    }
  }
}

static void build_low_cage_level(const Mesh &base_mesh,
                                 const Multiresolution &multires,
                                 const int bake_lvl,
                                 const bool in_sculpt,
                                 const BakeSettings &settings,
                                 LowCage &r_cage)
{
  if (in_sculpt && bake_lvl == multires.curr_level) {
    r_cage.mesh = Multiresolution::eval_to_mesh_in_sculpt(base_mesh, bake_lvl);
  }
  else {
    r_cage.mesh = Multiresolution::eval_to_mesh(base_mesh, bake_lvl);
  }

  const Mesh &low_mesh = *r_cage.mesh;
  if (low_mesh.sculpt_layers_len > 0) {
    r_cage.positions_storage.reinitialize(low_mesh.verts_num);
    bke::sculpt_layer::calc_eval_positions(low_mesh, r_cage.positions_storage.as_mutable_span());
    r_cage.positions = r_cage.positions_storage.as_span();
  }
  else {
    r_cage.positions = low_mesh.vert_positions();
  }
  r_cage.corner_verts = low_mesh.corner_verts();
  r_cage.faces = low_mesh.faces();

  const StringRef uv_name = low_mesh.active_uv_map_name();
  const bke::AttributeAccessor low_attributes = low_mesh.attributes();
  const VArray<float2> low_uv = *low_attributes.lookup<float2>(uv_name, bke::AttrDomain::Corner);
  if (low_uv.is_span()) {
    r_cage.uv = low_uv.get_internal_span();
  }
  else {
    r_cage.uv_storage.reinitialize(low_uv.size());
    low_uv.materialize(r_cage.uv_storage.as_mutable_span());
    r_cage.uv = r_cage.uv_storage.as_span();
  }

  if (!settings.need_low_normals()) {
    return;
  }

  calc_area_normals(r_cage.positions,
                    r_cage.faces,
                    r_cage.corner_verts,
                    r_cage.vert_normals_storage,
                    r_cage.face_normals_storage);
  r_cage.vert_normals = r_cage.vert_normals_storage.as_span();
  r_cage.face_normals = r_cage.face_normals_storage.as_span();

  r_cage.sharp_storage.reinitialize(r_cage.faces.size());
  r_cage.sharp_storage.as_mutable_span().fill(true);
  r_cage.sharp = r_cage.sharp_storage.as_span();

  if (settings.need_low_tangents()) {
    Array<Array<float4>> tangent_data = bke::mesh::calc_uv_tangents(r_cage.positions,
                                                                    r_cage.faces,
                                                                    r_cage.corner_verts,
                                                                    low_mesh.corner_tris(),
                                                                    low_mesh.corner_tri_faces(),
                                                                    r_cage.sharp,
                                                                    r_cage.vert_normals,
                                                                    r_cage.face_normals,
                                                                    low_mesh.corner_normals(),
                                                                    {r_cage.uv});

    r_cage.tangents_storage = std::move(tangent_data[0]);
    r_cage.tangents = r_cage.tangents_storage.as_span();
  }
}

static void build_low_cage(const Mesh &base_mesh,
                           const Multiresolution &multires,
                           const int bake_lvl,
                           const bool in_sculpt,
                           const Span<float2> base_uv,
                           const BakeSettings &settings,
                           LowCage &r_cage)
{
  const int eff_bake_lvl = multires.base_all_quads ? bake_lvl : std::max(0, bake_lvl - 1);
  r_cage.per_side = 1 << eff_bake_lvl;
  r_cage.faces_per_base = r_cage.per_side * r_cage.per_side;

  if (bake_lvl == 0) {
    build_low_cage_base(base_mesh, multires, base_uv, settings, r_cage);
  }
  else {
    build_low_cage_level(base_mesh, multires, bake_lvl, in_sculpt, settings, r_cage);
  }
}

static HighMesh build_high_mesh(const Mesh &base_mesh,
                                const Multiresolution &multires,
                                const bool need_normals)
{
  HighMesh high;

  const int top_lvl = multires.top_level;
  const int top_base_lvl = multires.base_level(top_lvl);
  const int rel_lvl = top_lvl - top_base_lvl;
  const int eff_top_lvl = multires.base_all_quads ? top_lvl : (top_lvl - 1);
  const int eff_top_base_lvl = multires.base_all_quads ? top_base_lvl :
                                                         std::max(0, top_base_lvl - 1);
  high.virt_per_side = 1 << eff_top_base_lvl;
  high.top_per_side = 1 << (eff_top_lvl - eff_top_base_lvl);
  high.faces_per_virtual_grid = high.top_per_side * high.top_per_side;

  const bke::multiresolution::BaseTopology top_base = multires.get_base_topology(base_mesh,
                                                                                 top_lvl);

  const Span<float3> top_base_positions = multires.top_base_positions();
  high.positions.reinitialize(multires.top_verts_num());
  bke::multiresolution::calc_top_eval_positions(
      base_mesh, top_base_positions, high.positions.as_mutable_span());

  const int64_t top_subfaces = top_base.corners_num * Multiresolution::faces_per_corner(rel_lvl);
  high.corner_verts.reinitialize(top_subfaces * 4);

  if (need_normals) {
    bke::pbvh::Tree top_pbvh = bke::pbvh::Tree::from_stored_nodes_multires(top_lvl, base_mesh);
    const Span<bke::pbvh::MultiresNode> top_nodes = top_pbvh.nodes<bke::pbvh::MultiresNode>();
    const bke::multiresolution::OffsetsSpan top_offsets_span(top_pbvh.offsets);

    bke::multiresolution::calc_corner_verts(
        top_base, rel_lvl, top_offsets_span, high.corner_verts.as_mutable_span());

    high.vert_normals.reinitialize(multires.top_verts_num());
    IndexMaskMemory memory;
    const IndexMask leaf_nodes = bke::pbvh::all_leaf_nodes(top_pbvh, memory);
    bke::pbvh::calc_normals_multires(top_base,
                                     top_offsets_span,
                                     rel_lvl,
                                     high.corner_verts.as_span(),
                                     high.positions.as_span(),
                                     top_nodes,
                                     leaf_nodes,
                                     high.vert_normals.as_mutable_span());
  }
  else {
    const bke::multiresolution::Offsets top_offsets = Multiresolution::build_table_offsets(
        multires, top_base, top_lvl);
    const bke::multiresolution::OffsetsSpan top_offsets_span(top_offsets);
    bke::multiresolution::calc_corner_verts(
        top_base, rel_lvl, top_offsets_span, high.corner_verts.as_mutable_span());
  }

  return high;
}

static wmOperatorStatus multires_bake_exec(bContext *C, wmOperator *op)
{
  Main *bmain = CTX_data_main(C);
  Scene *scene = CTX_data_scene(C);
  Object *ob = CTX_data_active_object(C);

  const Mesh &base_mesh = *id_cast<const Mesh *>(ob->data);

  if (BKE_object_sculpt_use_dyntopo(ob) || (base_mesh.flag & ME_SCULPT_DYNAMIC_TOPOLOGY)) {
    BKE_report(op->reports, RPT_ERROR, "Cannot bake while Dynamic Topology is active.");
    return OPERATOR_CANCELLED;
  }

  if (base_mesh.multires == nullptr) {
    BKE_report(op->reports, RPT_ERROR, "Mesh has no multiresolution data");
    return OPERATOR_CANCELLED;
  }

  const Multiresolution &multires = *base_mesh.multires;

  if (!bke::multiresolution::base_topology_matches(base_mesh)) {
    return OPERATOR_CANCELLED;
  }

  Sculpt *sculpt = (scene && scene->toolsettings) ? scene->toolsettings->sculpt : nullptr;

  BakeSettings settings;
  if (sculpt) {
    settings.type = sculpt->multires_bake.type;
    settings.space = sculpt->multires_bake.space;
    settings.use_normalize = bool(sculpt->multires_bake.use_normalize);
    settings.margin = sculpt->multires_bake.margin;
  }
  if (RNA_struct_property_is_set(op->ptr, "type")) {
    settings.type = static_cast<MultiresBakeType>(RNA_enum_get(op->ptr, "type"));
  }
  if (RNA_struct_property_is_set(op->ptr, "space")) {
    settings.space = static_cast<MultiresBakeSpace>(RNA_enum_get(op->ptr, "space"));
  }
  if (RNA_struct_property_is_set(op->ptr, "use_normalize")) {
    settings.use_normalize = RNA_boolean_get(op->ptr, "use_normalize");
  }
  if (RNA_struct_property_is_set(op->ptr, "margin")) {
    settings.margin = RNA_int_get(op->ptr, "margin");
  }

  char image_name[MAX_ID_NAME - 2];
  RNA_string_get(op->ptr, "image", image_name);
  Image *image = nullptr;
  if (image_name[0] != '\0') {
    image = id_cast<Image *>(BKE_libblock_find_name(bmain, ID_IM, image_name));
    if (image == nullptr) {
      BKE_report(op->reports, RPT_ERROR, "Image not found");
      return OPERATOR_CANCELLED;
    }
  }
  else if (sculpt && sculpt->multires_bake.image) {
    image = sculpt->multires_bake.image;
  }
  else {
    BKE_report(op->reports, RPT_ERROR, "No image specified");
    return OPERATOR_CANCELLED;
  }

  const bool in_sculpt = is_sculpt(*ob);
  const int bake_lvl = math::clamp(
      in_sculpt ? multires.sculpt_level : multires.view_level, 0, multires.top_level);
  if (bake_lvl >= multires.top_level) {
    BKE_report(op->reports, RPT_ERROR, "Bake level equals top level, nothing to bake");
    return OPERATOR_CANCELLED;
  }

  const StringRef uv_name = base_mesh.active_uv_map_name();
  if (uv_name.is_empty()) {
    BKE_report(op->reports, RPT_ERROR, "Base mesh has no active UV map");
    return OPERATOR_CANCELLED;
  }
  const bke::AttributeAccessor attributes = base_mesh.attributes();
  const VArraySpan<float2> base_uv_attr =
      attributes.lookup<float2>(uv_name, bke::AttrDomain::Corner).varray;

  if (base_uv_attr.is_empty()) {
    BKE_report(op->reports, RPT_ERROR, "Active UV layer is empty");
    return OPERATOR_CANCELLED;
  }

  LowCage cage;
  build_low_cage(base_mesh, multires, bake_lvl, in_sculpt, base_uv_attr, settings, cage);

  const HighMesh high = build_high_mesh(base_mesh, multires, settings.need_high_normals());

  bool ok = false;
  Vector<BakedTile> deferred_tiles;
  float2 height_minmax(FLT_MAX, -FLT_MAX);

  for (ImageTile &tile : image->tiles) {
    ok = bake_tile(settings, cage, high, *image, tile, deferred_tiles, height_minmax) || ok;
  }

  if (!deferred_tiles.is_empty()) {
    const float max_distance = (height_minmax.x <= height_minmax.y) ?
                                   math::max(math::abs(height_minmax.x),
                                             math::abs(height_minmax.y)) :
                                   0.0f;
    for (BakedTile &bt : deferred_tiles) {
      normalize_tile_displacement(*bt.ibuf, bt.displacement, bt.mask, max_distance);
      finish_baked_tile(*image, *bt.ibuf, bt.mask, settings.margin);
    }
  }

  if (!ok) {
    BKE_report(op->reports, RPT_ERROR, "No image tiles could be baked");
    return OPERATOR_CANCELLED;
  }

  DEG_id_tag_update(&image->id, 0);
  WM_event_add_notifier(C, NC_IMAGE | ND_DRAW, image);
  return OPERATOR_FINISHED;
}

static bool multires_bake_poll(bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  if (ob == nullptr || ob->data == nullptr) {
    return false;
  }
  if (BKE_object_sculpt_use_dyntopo(ob)) {
    return false;
  }
  const Mesh &mesh = *id_cast<const Mesh *>(ob->data);

  if (mesh.multires == nullptr || (mesh.flag & ME_SCULPT_DYNAMIC_TOPOLOGY)) {
    return false;
  }

  return bke::multiresolution::base_counts_match(mesh);
}

}  // namespace

static void multires_bake_properties(wmOperatorType *ot)
{
  RNA_def_enum(ot->srna,
               "type",
               multires_bake_type_items,
               MULTIRES_BAKE_NORMALS,
               "Bake Type",
               "Type of data to bake from the multiresolution mesh");
  RNA_def_enum(ot->srna,
               "space",
               multires_bake_space_items,
               MULTIRES_BAKE_SPACE_TANGENT,
               "Space",
               "Orientation space for vector displacement");
  RNA_def_boolean(ot->srna,
                  "use_normalize",
                  false,
                  "Normalize",
                  "Normalize height values into [0, 1] range (recommended for 8-bit images)");
  RNA_def_string(
      ot->srna, "image", nullptr, MAX_ID_NAME - 2, "Image", "Image datablock to bake into");
  RNA_def_int(ot->srna,
              "margin",
              16,
              0,
              64,
              "Margin",
              "Dilation radius in pixels for the margin pass",
              0,
              64);
}

void SCULPT_OT_multires_bake(wmOperatorType *ot)
{
  ot->name = "Multires Bake";
  ot->idname = "SCULPT_OT_multires_bake";
  ot->description = "Bake data from the multires top level to an image";
  ot->exec = multires_bake_exec;
  ot->poll = multires_bake_poll;
  ot->flag = OPTYPE_REGISTER;

  multires_bake_properties(ot);
}

}  // namespace blender::ed::sculpt_paint::multires
