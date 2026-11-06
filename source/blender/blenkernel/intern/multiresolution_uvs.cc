/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 *
 */

#include "DNA_mesh_types.h"

#include "BKE_mesh_mapping.hh"
#include "BKE_multiresolution.hh"

#include "BLI_array.hh"
#include "BLI_math_vector.hh"
#include "BLI_offset_indices.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "multiresolution.hh"

#include <algorithm>

namespace blender::bke::multiresolution::uv {

using BaseTopology = bke::multiresolution::BaseTopology;

static int uv_total_faces(const int lvl, const int faces_num, const int corners_num)
{
  if (lvl == 0) {
    return faces_num;
  }
  return corners_num * (1 << (2 * (lvl - 1)));
}

enum class UVVertType : uint8_t {
  Interior,
  Boundary,
  Pinned,
};

struct UVBaseTopology {
  int verts_num = 0;
  int edges_num = 0;

  Vector<int2> edges;
  Array<int> corner_verts;
  Array<int> corner_edges;

  Array<int> vert_to_edge_offsets;
  Array<int> vert_to_edge_indices;
  GroupedSpan<int> vert_to_edge_map;

  Array<int> vert_to_corner_offsets;
  Array<int> vert_to_corner_indices;
  GroupedSpan<int> vert_to_corner_map;

  Array<int> edge_to_corner_offsets;
  Array<int> edge_to_corner_indices;
  GroupedSpan<int> edge_to_corner_map;

  Array<UVVertType> vert_types;
};

static UVBaseTopology build_uv_base_topology(const BaseTopology &base, const Span<float2> base_uvs)
{
  UVBaseTopology topo;

  topo.corner_verts.reinitialize(base.corners_num);
  topo.verts_num = bke::mesh::calc_uv_verts(
      base.faces, base.corner_verts, base.vert_to_corner_map, base_uvs, false, topo.corner_verts);

  topo.corner_edges.reinitialize(base.corners_num);
  topo.edges.reserve(base.edges_num);
  Array<int> mesh_edge_uv_edge(base.edges_num, -1);

  for (const int face_idx : IndexRange(base.faces_num)) {
    const IndexRange face_range = base.faces[face_idx];
    for (int k = 0; k < face_range.size(); k++) {
      const int c_curr = face_range.start() + k;
      const int c_next = face_range.start() + (k + 1) % face_range.size();
      const int edge_idx = base.corner_edges[c_curr];

      const int uv_v0 = topo.corner_verts[c_curr];
      const int uv_v1 = topo.corner_verts[c_next];
      const int2 uv_pair = base.face_edge_reversed[c_curr] ? int2(uv_v1, uv_v0) :
                                                             int2(uv_v0, uv_v1);

      const int existing_ue = mesh_edge_uv_edge[edge_idx];
      if (existing_ue != -1 && topo.edges[existing_ue] == uv_pair) {
        topo.corner_edges[c_curr] = existing_ue;
        continue;
      }

      const int ue_id = topo.edges.size();
      topo.edges.append(uv_pair);
      topo.corner_edges[c_curr] = ue_id;

      if (existing_ue == -1) {
        const Span<int> adj_faces = base.edge_to_face_map[edge_idx];
        if (adj_faces.size() == 2 && adj_faces[0] != adj_faces[1]) {
          mesh_edge_uv_edge[edge_idx] = ue_id;
        }
      }
    }
  }
  topo.edges_num = topo.edges.size();

  topo.vert_to_edge_map = bke::mesh::build_vert_to_edge_map(
      topo.edges, topo.verts_num, topo.vert_to_edge_offsets, topo.vert_to_edge_indices);
  topo.vert_to_corner_map = bke::mesh::build_vert_to_corner_map(
      topo.corner_verts, topo.verts_num, topo.vert_to_corner_offsets, topo.vert_to_corner_indices);
  topo.edge_to_corner_map = bke::mesh::build_edge_to_corner_map(
      topo.corner_edges, topo.edges_num, topo.edge_to_corner_offsets, topo.edge_to_corner_indices);

  topo.vert_types.reinitialize(topo.verts_num);
  threading::parallel_for(IndexRange(topo.verts_num), 1024, [&](const IndexRange range) {
    for (const int uv_id : range) {
      const Span<int> edges = topo.vert_to_edge_map[uv_id];
      const Span<int> corners = topo.vert_to_corner_map[uv_id];

      int bnd_edges_num = 0;
      for (const int ue : edges) {
        if (topo.edge_to_corner_map[ue].size() == 1) {
          bnd_edges_num++;
        }
      }

      if (bnd_edges_num == 0 && edges.size() == corners.size()) {
        topo.vert_types[uv_id] = UVVertType::Interior;
      }
      else if (bnd_edges_num == 2 && edges.size() == corners.size() + 1) {
        topo.vert_types[uv_id] = UVVertType::Boundary;
      }
      else {
        topo.vert_types[uv_id] = UVVertType::Pinned;
      }
    }
  });

  return topo;
}

static void subdivide_uv_grids_lvl0(const BaseTopology &base,
                                    const Span<float2> base_uvs,
                                    MutableSpan<float2> r_face_points)
{
  const OffsetIndices<int> faces = base.faces;
  threading::parallel_for(IndexRange(base.faces_num), 2048, [&](const IndexRange range) {
    for (const int face_idx : range) {
      const IndexRange face_range = faces[face_idx];
      float2 sum(0.0f);
      for (const float2 &uv : base_uvs.slice(face_range)) {
        sum += uv;
      }
      r_face_points[face_idx] = sum / float(face_range.size());
    }
  });
}

static void subdivide_uv_borders_lvl0(const BaseTopology &base,
                                      const UVBaseTopology &topo,
                                      const Span<float2> base_uvs,
                                      const Span<float2> face_points,
                                      MutableSpan<float2> r_edge_points)
{
  threading::parallel_for(IndexRange(topo.edges_num), 2048, [&](const IndexRange range) {
    for (const int ue : range) {
      const int2 edge = topo.edges[ue];
      const float2 v0 = base_uvs[topo.vert_to_corner_map[edge[0]][0]];
      const float2 v1 = base_uvs[topo.vert_to_corner_map[edge[1]][0]];
      const Span<int> edge_corners = topo.edge_to_corner_map[ue];

      if (edge_corners.size() == 2) {
        const float2 fa = face_points[base.corner_to_face_map[edge_corners[0]]];
        const float2 fb = face_points[base.corner_to_face_map[edge_corners[1]]];
        r_edge_points[ue] = (v0 + v1 + fa + fb) * 0.25f;
      }
      else {
        r_edge_points[ue] = (v0 + v1) * 0.5f;
      }
    }
  });
}

static void subdivide_uv_base_lvl0(const BaseTopology &base,
                                   const UVBaseTopology &topo,
                                   const Span<float2> base_uvs,
                                   const Span<float2> face_points,
                                   MutableSpan<float2> r_vert_points)
{
  threading::parallel_for(IndexRange(topo.verts_num), 1024, [&](const IndexRange range) {
    for (const int i : range) {
      const Span<int> vert_corners = topo.vert_to_corner_map[i];
      const float2 P = base_uvs[vert_corners[0]];

      if (topo.vert_types[i] == UVVertType::Pinned) {
        r_vert_points[i] = P;
        continue;
      }

      if (topo.vert_types[i] == UVVertType::Boundary) {
        r_vert_points[i] = P;
        continue;
      }

      const Span<int> vert_edges = topo.vert_to_edge_map[i];
      float2 sum_edges(0.0f);
      for (const int ue : vert_edges) {
        const int2 edge = topo.edges[ue];
        const int other = (edge[0] == i) ? edge[1] : edge[0];
        sum_edges += base_uvs[topo.vert_to_corner_map[other][0]];
      }

      float2 sum_faces(0.0f);
      for (const int corner : vert_corners) {
        sum_faces += face_points[base.corner_to_face_map[corner]];
      }

      const float n = float(vert_edges.size());
      r_vert_points[i] = (sum_faces + sum_edges + P * (n * (n - 2.0f))) / (n * n);
    }
  });
}

static void emit_uv_corners_lvl1(const BaseTopology &base,
                                 const UVBaseTopology &topo,
                                 const Span<float2> vert_points,
                                 const Span<float2> edge_points,
                                 const Span<float2> face_points,
                                 MutableSpan<float2> r_corners)
{
  threading::parallel_for(IndexRange(base.faces_num), 1024, [&](const IndexRange range) {
    for (const int face_idx : range) {
      const IndexRange face_range = base.faces[face_idx];
      const int face_start = face_range.start();
      const int face_size = face_range.size();
      float2 *out = &r_corners[face_start * 4];

      if (face_size == 4) {
        const float2 v0 = vert_points[topo.corner_verts[face_start + 0]];
        const float2 v1 = vert_points[topo.corner_verts[face_start + 1]];
        const float2 v2 = vert_points[topo.corner_verts[face_start + 2]];
        const float2 v3 = vert_points[topo.corner_verts[face_start + 3]];

        const float2 m0 = edge_points[topo.corner_edges[face_start + 0]];
        const float2 m1 = edge_points[topo.corner_edges[face_start + 1]];
        const float2 m2 = edge_points[topo.corner_edges[face_start + 2]];
        const float2 m3 = edge_points[topo.corner_edges[face_start + 3]];
        const float2 c = face_points[face_idx];

        out[0] = v0;
        out[1] = m0;
        out[2] = c;
        out[3] = m3;

        out[4] = m0;
        out[5] = v1;
        out[6] = m1;
        out[7] = c;

        out[8] = m3;
        out[9] = c;
        out[10] = m2;
        out[11] = v3;

        out[12] = c;
        out[13] = m1;
        out[14] = v2;
        out[15] = m2;
      }
      else {
        for (int corner = 0; corner < face_size; corner++) {
          const int prev_corner = (corner + face_size - 1) % face_size;
          out[corner * 4 + 0] = vert_points[topo.corner_verts[face_start + corner]];
          out[corner * 4 + 1] = edge_points[topo.corner_edges[face_start + corner]];
          out[corner * 4 + 2] = face_points[face_idx];
          out[corner * 4 + 3] = edge_points[topo.corner_edges[face_start + prev_corner]];
        }
      }
    }
  });
}

static void fill_border_side_uv(const int size,
                                const UVBaseTopology &topo,
                                const BaseTopology &base,
                                const int corner,
                                const Span<float2> in_face_uv,
                                const Span<float2> in_edge_uv,
                                const MutableSpan<float2> side)
{
  const int face_idx = corner >> 2;
  const int local_edge = corner & 3;
  const int corner_start = corner & ~3;

  extract_grid_inner_border(
      size, face_idx * size * size, local_edge, false, in_face_uv, side.slice(1, size));

  const int prev_corner = corner_start + ((local_edge + 3) & 3);
  const int prev_ue = topo.corner_edges[prev_corner];
  const int prev_pt_idx = base.face_edge_reversed[prev_corner] ? 0 : size - 1;
  side[0] = in_edge_uv[prev_ue * size + prev_pt_idx];

  const int next_corner = corner_start + ((local_edge + 1) & 3);
  const int next_ue = topo.corner_edges[next_corner];
  const int next_pt_idx = base.face_edge_reversed[next_corner] ? size - 1 : 0;
  side[size + 1] = in_edge_uv[next_ue * size + next_pt_idx];

  if (base.face_edge_reversed[corner]) {
    std::reverse(side.begin(), side.end());
  }
}

static void subdivide_uv_grids(const int lvl,
                               const UVBaseTopology &topo,
                               const BaseTopology &base,
                               const Span<float2> in_vert_uv,
                               const Span<float2> in_edge_uv,
                               const Span<float2> in_face_uv,
                               MutableSpan<float2> out_face_uv)
{
  if (lvl == 1) {
    threading::parallel_for(IndexRange(base.faces_num), 1024, [&](const IndexRange range) {
      for (const int face_idx : range) {
        const float2 c = in_face_uv[face_idx];

        const int corner_offset = face_idx * 4;
        const float2 v_bl = in_vert_uv[topo.corner_verts[corner_offset + 0]];
        const float2 v_br = in_vert_uv[topo.corner_verts[corner_offset + 1]];
        const float2 v_tr = in_vert_uv[topo.corner_verts[corner_offset + 2]];
        const float2 v_tl = in_vert_uv[topo.corner_verts[corner_offset + 3]];

        const float2 v_b = in_edge_uv[topo.corner_edges[corner_offset + 0]];
        const float2 v_r = in_edge_uv[topo.corner_edges[corner_offset + 1]];
        const float2 v_t = in_edge_uv[topo.corner_edges[corner_offset + 2]];
        const float2 v_l = in_edge_uv[topo.corner_edges[corner_offset + 3]];

        const MutableSpan<float2> out_grid = out_face_uv.slice(face_idx * 9, 9);
        const float2 f0 = v_bl + v_b + v_l + c;
        const float2 f1 = v_b + v_br + c + v_r;
        const float2 f2 = v_l + c + v_tl + v_t;
        const float2 f3 = c + v_r + v_t + v_tr;

        out_grid[0] = f0 * 0.25f;
        out_grid[1] = (v_b + c + (f0 + f1) * 0.25f) * 0.25f;
        out_grid[2] = f1 * 0.25f;
        out_grid[3] = (v_l + c + (f0 + f2) * 0.25f) * 0.25f;
        out_grid[4] = ((f0 + f1 + f2 + f3) * 0.0625f + (v_b + v_l + v_r + v_t) * 0.25f +
                       c * 2.0f) *
                      0.25f;
        out_grid[5] = (c + v_r + (f1 + f3) * 0.25f) * 0.25f;
        out_grid[6] = f2 * 0.25f;
        out_grid[7] = (c + v_t + (f2 + f3) * 0.25f) * 0.25f;
        out_grid[8] = f3 * 0.25f;
      }
    });
    return;
  }

  const int in_size = Multiresolution::grid_size(lvl);
  const int out_size = Multiresolution::grid_size(lvl + 1);
  const int in_size_sq = in_size * in_size;
  const int out_size_sq = out_size * out_size;
  const int grain_size = std::max(1, 4096 / out_size_sq);

  threading::parallel_for(IndexRange(base.faces_num), grain_size, [&](const IndexRange range) {
    Array<float2> bot_buf(in_size + 2);
    Array<float2> top_buf(in_size + 2);
    Array<float2> left_buf(in_size + 2);
    Array<float2> right_buf(in_size + 2);

    for (const int face_idx : range) {
      const Span<float2> grid = in_face_uv.slice(face_idx * in_size_sq, in_size_sq);
      MutableSpan<float2> out_grid = out_face_uv.slice(face_idx * out_size_sq, out_size_sq);

      const int corner_offset = face_idx * 4;
      const float2 v_bl = in_vert_uv[topo.corner_verts[corner_offset + 0]];
      const float2 v_br = in_vert_uv[topo.corner_verts[corner_offset + 1]];
      const float2 v_tr = in_vert_uv[topo.corner_verts[corner_offset + 2]];
      const float2 v_tl = in_vert_uv[topo.corner_verts[corner_offset + 3]];

      const int ue_b = topo.corner_edges[corner_offset + 0];
      const int ue_r = topo.corner_edges[corner_offset + 1];
      const int ue_t = topo.corner_edges[corner_offset + 2];
      const int ue_l = topo.corner_edges[corner_offset + 3];

      fill_border(v_bl,
                  v_br,
                  base.face_edge_reversed[corner_offset + 0],
                  in_edge_uv.slice(ue_b * in_size, in_size),
                  bot_buf.as_mutable_span());
      fill_border(v_br,
                  v_tr,
                  base.face_edge_reversed[corner_offset + 1],
                  in_edge_uv.slice(ue_r * in_size, in_size),
                  right_buf.as_mutable_span());
      fill_border(v_tl,
                  v_tr,
                  !base.face_edge_reversed[corner_offset + 2],
                  in_edge_uv.slice(ue_t * in_size, in_size),
                  top_buf.as_mutable_span());
      fill_border(v_bl,
                  v_tl,
                  !base.face_edge_reversed[corner_offset + 3],
                  in_edge_uv.slice(ue_l * in_size, in_size),
                  left_buf.as_mutable_span());

      for (int row = 0, row_offset = 0; row < in_size; row++, row_offset += in_size) {
        const Span<float2> in_bot = (row == 0) ? bot_buf.as_span().slice(1, in_size) :
                                                 grid.slice(row_offset - in_size, in_size);
        const Span<float2> in_mid = grid.slice(row_offset, in_size);
        const Span<float2> in_top = (row == in_size - 1) ?
                                        top_buf.as_span().slice(1, in_size) :
                                        grid.slice(row_offset + in_size, in_size);

        MutableSpan<float2> out_even = out_grid.slice(2 * row * out_size, out_size);
        MutableSpan<float2> out_odd = out_grid.slice((2 * row + 1) * out_size, out_size);

        run_row(in_size,
                row,
                left_buf.as_span(),
                right_buf.as_span(),
                in_bot,
                in_mid,
                in_top,
                out_even,
                out_odd);
      }

      int j = (in_size - 1) * in_size;
      int idx = (out_size - 1) * out_size;

      float2 m0 = left_buf[in_size];
      float2 t0 = left_buf[in_size + 1];
      float2 m1 = grid[j];
      float2 t1 = top_buf[1];

      for (int col = 0; col < in_size - 1; col++) {
        const float2 m2 = grid[j + 1];
        const float2 t2 = top_buf[col + 2];
        out_grid[idx] = (m0 + m1 + t0 + t1) * 0.25f;
        out_grid[idx + 1] = ((m1 + t1) * 0.375f) + ((m0 + m2 + t0 + t2) * 0.0625f);
        m0 = m1;
        t0 = t1;
        m1 = m2;
        t1 = t2;
        j += 1;
        idx += 2;
      }

      const float2 m_last = right_buf[in_size];
      const float2 t_last = right_buf[in_size + 1];
      out_grid[idx] = (m0 + m1 + t0 + t1) * 0.25f;
      out_grid[idx + 1] = ((m1 + t1) * 0.375f) + ((m0 + m_last + t0 + t_last) * 0.0625f);
      out_grid[idx + 2] = (m1 + m_last + t1 + t_last) * 0.25f;
    }
  });
}

static void subdivide_uv_borders(const int lvl,
                                 const UVBaseTopology &topo,
                                 const BaseTopology &base,
                                 const Span<float2> in_vert_uv,
                                 const Span<float2> in_edge_uv,
                                 const Span<float2> in_face_uv,
                                 MutableSpan<float2> out_edge_uv)
{
  const int in_size = Multiresolution::grid_size(lvl);
  const int out_size = Multiresolution::grid_size(lvl + 1);

  threading::parallel_for(IndexRange(topo.edges_num), 1024, [&](const IndexRange range) {
    Array<float2> top_side(in_size + 2);
    Array<float2> bot_side(in_size + 2);

    for (const int ue : range) {
      const int2 edge = topo.edges[ue];
      const Span<float2> inner_edge = in_edge_uv.slice(ue * in_size, in_size);
      MutableSpan<float2> r_edge_out = out_edge_uv.slice(ue * out_size, out_size);

      const float2 v0 = in_vert_uv[edge[0]];
      const float2 v1 = in_vert_uv[edge[1]];
      const Span<int> edge_corners = topo.edge_to_corner_map[ue];

      if (edge_corners.size() == 1) {
        float2 mid_prev = v0;
        for (const int i : IndexRange(in_size)) {
          const float2 mid_curr = inner_edge[i];
          r_edge_out[i * 2] = (mid_prev + mid_curr) * 0.5f;
          r_edge_out[i * 2 + 1] = mid_curr;
          mid_prev = mid_curr;
        }
        r_edge_out[2 * in_size] = (mid_prev + v1) * 0.5f;
        continue;
      }

      fill_border_side_uv(in_size, topo, base, edge_corners[0], in_face_uv, in_edge_uv, top_side);
      fill_border_side_uv(in_size, topo, base, edge_corners[1], in_face_uv, in_edge_uv, bot_side);

      float2 mid_prev = v0;
      for (int k = 0; k < in_size; k++) {
        const float2 mid_curr = inner_edge[k];
        const float2 mid_next = (k == in_size - 1) ? v1 : inner_edge[k + 1];

        const float2 t0 = top_side[k], t1 = top_side[k + 1], t2 = top_side[k + 2];
        const float2 b0 = bot_side[k], b1 = bot_side[k + 1], b2 = bot_side[k + 2];

        r_edge_out[2 * k] = ((mid_prev + mid_curr) * 0.375f) + ((t0 + t1 + b0 + b1) * 0.0625f);
        r_edge_out[2 * k + 1] = ((t0 + t2 + b0 + b2) * 1.0f +
                                 (t1 + b1 + mid_prev + mid_next) * 6.0f + mid_curr * 36.0f) *
                                (1.0f / 64.0f);

        mid_prev = mid_curr;
      }

      r_edge_out[2 * in_size] = ((mid_prev + v1) * 0.375f) +
                                ((top_side[in_size] + top_side[in_size + 1] + bot_side[in_size] +
                                  bot_side[in_size + 1]) *
                                 0.0625f);
    }
  });
}

static void subdivide_uv_base(const int lvl,
                              const UVBaseTopology &topo,
                              const Span<float2> in_vert_uv,
                              const Span<float2> in_edge_uv,
                              const Span<float2> in_face_uv,
                              MutableSpan<float2> out_vert_uv)
{
  const int in_size = Multiresolution::grid_size(lvl);

  threading::parallel_for(IndexRange(topo.verts_num), 1024, [&](const IndexRange range) {
    for (const int i : range) {
      const float2 P = in_vert_uv[i];

      if (topo.vert_types[i] == UVVertType::Pinned) {
        out_vert_uv[i] = P;
        continue;
      }

      if (topo.vert_types[i] == UVVertType::Boundary) {
        out_vert_uv[i] = P;
        continue;
      }

      const Span<int> vert_edges = topo.vert_to_edge_map[i];
      float2 sum_edges(0.0f);
      for (const int ue : vert_edges) {
        const int2 edge = topo.edges[ue];
        const int pos = (edge[0] == i) ? 0 : in_size - 1;
        sum_edges += in_edge_uv[ue * in_size + pos];
      }

      const Span<int> v_corners = topo.vert_to_corner_map[i];
      float2 sum_faces(0.0f);
      for (const int corner : v_corners) {
        const int face_idx = corner >> 2;
        const int local_idx = corner & 3;
        const int row = local_idx < 2 ? 0 : in_size - 1;
        const int col = (local_idx == 0 || local_idx == 3) ? 0 : in_size - 1;
        sum_faces += in_face_uv[face_idx * in_size * in_size + row * in_size + col];
      }

      const float n = float(v_corners.size());
      const float d = 4.0f * n * n;
      out_vert_uv[i] = (sum_faces + 6.0f * sum_edges + (d - 7.0f * n) * P) / d;
    }
  });
}

constexpr int MAX_STRIDE = 33;

template<bool VirtualOrder>
static inline void emit_tile_subfaces(const Span<float2> vertex_grid,
                                      const int row_stride,
                                      const int subfaces_per_side,
                                      const int out_row_stride,
                                      float2 *r_output)
{
  if constexpr (VirtualOrder) {
    int output_index = 0;
    for (int local_row = 0; local_row < subfaces_per_side; local_row++) {
      const int row_offset = int64_t(local_row) * row_stride;
      const int next_row_offset = int64_t(local_row + 1) * row_stride;
      for (int local_col = 0; local_col < subfaces_per_side; local_col++) {
        r_output[output_index++] = vertex_grid[row_offset + local_col];
        r_output[output_index++] = vertex_grid[row_offset + local_col + 1];
        r_output[output_index++] = vertex_grid[next_row_offset + local_col + 1];
        r_output[output_index++] = vertex_grid[next_row_offset + local_col];
      }
    }
  }
  else {
    for (int local_row = 0; local_row < subfaces_per_side; local_row++) {
      const int row_offset = int64_t(local_row) * row_stride;
      const int next_row_offset = int64_t(local_row + 1) * row_stride;
      float2 *row_out = r_output + int64_t(local_row) * out_row_stride;
      int output_index = 0;
      for (int local_col = 0; local_col < subfaces_per_side; local_col++) {
        row_out[output_index++] = vertex_grid[row_offset + local_col];
        row_out[output_index++] = vertex_grid[row_offset + local_col + 1];
        row_out[output_index++] = vertex_grid[next_row_offset + local_col + 1];
        row_out[output_index++] = vertex_grid[next_row_offset + local_col];
      }
    }
  }
}

template<bool WholeFace, bool VirtualOrder = true>
static void emit_uv_corners_impl(const UVBaseTopology &topo,
                                 const BaseTopology &base,
                                 const int target_lvl,
                                 const int base_lvl,
                                 const Span<float2> vert_uv,
                                 const Span<float2> edge_uv,
                                 const Span<float2> face_uv,
                                 MutableSpan<float2> r_corners)
{
  const int grid_size = Multiresolution::grid_size(target_lvl);
  const int grid_size_sq = grid_size * grid_size;
  const int tile_shift = target_lvl - base_lvl;
  const int subfaces_per_side = 1 << target_lvl;
  const int tiles_per_side = 1 << base_lvl;
  const int subfaces_per_tile = 1 << tile_shift;
  const int out_row_stride = (WholeFace || VirtualOrder) ? (subfaces_per_tile * 4) :
                                                           (subfaces_per_side * 4);
  const int total_tiles = base.faces_num << (2 * base_lvl);
  const int grain_size = std::max(1, 4096 / (subfaces_per_tile * subfaces_per_tile));

  threading::parallel_for(IndexRange(total_tiles), grain_size, [&](const IndexRange range) {
    const int stride = subfaces_per_tile + 1;
    const int sink = stride * stride;
    Array<float2, MAX_STRIDE * MAX_STRIDE + 1> window(sink + 1);

    for (const int tile : range) {
      const int face_index = WholeFace ? tile : (tile >> (2 * base_lvl));
      const int tile_index = WholeFace ? 0 : (tile & ((tiles_per_side * tiles_per_side) - 1));
      const int tile_row = WholeFace ? 0 : ((tile_index >> base_lvl) << tile_shift);
      const int tile_col = WholeFace ? 0 : ((tile_index & (tiles_per_side - 1)) << tile_shift);

      const Span<float2> face_grid = face_uv.slice(face_index * grid_size_sq, grid_size_sq);
      float2 *tile_output =
          (WholeFace || VirtualOrder) ?
              &r_corners[int64_t(tile) * subfaces_per_tile * subfaces_per_tile * 4] :
              &r_corners[((int64_t(face_index) * subfaces_per_side + tile_row) *
                              subfaces_per_side +
                          tile_col) *
                         4];

      const bool touch_bottom = WholeFace || tile_row == 0;
      const bool touch_top = WholeFace || tile_row + subfaces_per_tile == subfaces_per_side;
      const bool touch_left = WholeFace || tile_col == 0;
      const bool touch_right = WholeFace || tile_col + subfaces_per_tile == subfaces_per_side;

      if (!WholeFace && !(touch_bottom | touch_top | touch_left | touch_right)) {
        const int tile_offset = (tile_row - 1) * grid_size + (tile_col - 1);
        emit_tile_subfaces<VirtualOrder>(
            face_grid.slice(tile_offset, face_grid.size() - tile_offset),
            grid_size,
            subfaces_per_tile,
            out_row_stride,
            tile_output);
        continue;
      }

      const int corner_start = face_index * 4;
      const Span<float2> side_grids[4] = {
          edge_uv.slice(topo.corner_edges[corner_start + 0] * grid_size, grid_size),
          edge_uv.slice(topo.corner_edges[corner_start + 1] * grid_size, grid_size),
          edge_uv.slice(topo.corner_edges[corner_start + 2] * grid_size, grid_size),
          edge_uv.slice(topo.corner_edges[corner_start + 3] * grid_size, grid_size),
      };
      const int side_steps[4] = {
          base.face_edge_reversed[corner_start + 0] ? -1 : 1,
          base.face_edge_reversed[corner_start + 1] ? -1 : 1,
          !base.face_edge_reversed[corner_start + 2] ? -1 : 1,
          !base.face_edge_reversed[corner_start + 3] ? -1 : 1,
      };

      const int row_start = WholeFace ? 1 : std::max(1, tile_row);
      const int row_end = WholeFace ?
                              subfaces_per_side - 1 :
                              std::min(subfaces_per_side - 1, tile_row + subfaces_per_tile);
      const int col_start = WholeFace ? 1 : std::max(1, tile_col);
      const int col_end = WholeFace ?
                              subfaces_per_side - 1 :
                              std::min(subfaces_per_side - 1, tile_col + subfaces_per_tile);
      const int columns_num = col_end - col_start + 1;
      const int rows_num = row_end - row_start + 1;

      MutableSpan<float2> window_span = window.as_mutable_span();
      for (int grid_row = row_start; grid_row <= row_end; grid_row++) {
        window_span.slice((grid_row - tile_row) * stride + (col_start - tile_col), columns_num)
            .copy_from(face_grid.slice((grid_row - 1) * grid_size + (col_start - 1), columns_num));
      }

      const int dst_first[4] = {col_start - tile_col,
                                (row_start - tile_row) * stride + subfaces_per_tile,
                                subfaces_per_tile * stride + (col_start - tile_col),
                                (row_start - tile_row) * stride};
      const int dst_step[4] = {1, stride, 1, stride};
      const int src_first[4] = {col_start - 1, row_start - 1, col_start - 1, row_start - 1};
      const int count[4] = {touch_bottom ? columns_num : 0,
                            touch_right ? rows_num : 0,
                            touch_top ? columns_num : 0,
                            touch_left ? rows_num : 0};
      for (int side_index = 0; side_index < 4; side_index++) {
        const int step = side_steps[side_index];
        int src_index = (step == 1) ? src_first[side_index] :
                                      (grid_size - 1) - src_first[side_index];
        int dst_index = dst_first[side_index];
        for (int sample_index = 0; sample_index < count[side_index]; sample_index++) {
          window_span[dst_index] = side_grids[side_index][src_index];
          src_index += step;
          dst_index += dst_step[side_index];
        }
      }

      const int corner_dst[4] = {
          (touch_bottom & touch_left) ? 0 : sink,
          (touch_bottom & touch_right) ? subfaces_per_tile : sink,
          (touch_top & touch_right) ? subfaces_per_tile * stride + subfaces_per_tile : sink,
          (touch_top & touch_left) ? subfaces_per_tile * stride : sink};
      for (int corner_index = 0; corner_index < 4; corner_index++) {
        window[corner_dst[corner_index]] = vert_uv[topo.corner_verts[corner_start + corner_index]];
      }

      emit_tile_subfaces<WholeFace || VirtualOrder>(
          window_span.as_span(), stride, subfaces_per_tile, out_row_stride, tile_output);
    }
  });
}

static void emit_uv_corners(const UVBaseTopology &topo,
                            const BaseTopology &base,
                            const int target_lvl,
                            const int base_lvl,
                            const Span<float2> vert_uv,
                            const Span<float2> edge_uv,
                            const Span<float2> face_uv,
                            const bool virtual_base_order,
                            MutableSpan<float2> r_corners)
{
  if (base_lvl == 0) {
    emit_uv_corners_impl<true, true>(
        topo, base, target_lvl, base_lvl, vert_uv, edge_uv, face_uv, r_corners);
  }
  else if (virtual_base_order) {
    emit_uv_corners_impl<false, true>(
        topo, base, target_lvl, base_lvl, vert_uv, edge_uv, face_uv, r_corners);
  }
  else {
    emit_uv_corners_impl<false, false>(
        topo, base, target_lvl, base_lvl, vert_uv, edge_uv, face_uv, r_corners);
  }
}

static void calc_smooth_uvs_impl(const BaseTopology &base,
                                 const Span<float2> base_uvs,
                                 const int target_lvl,
                                 const int base_lvl,
                                 const bool virtual_base_order,
                                 MutableSpan<float2> r_uvs)
{
  const UVBaseTopology topo = build_uv_base_topology(base, base_uvs);

  Array<float2> face_uv_0(base.faces_num);
  subdivide_uv_grids_lvl0(base, base_uvs, face_uv_0);

  Array<float2> edge_uv_0(topo.edges_num);
  subdivide_uv_borders_lvl0(base, topo, base_uvs, face_uv_0, edge_uv_0);

  Array<float2> vert_uv_0(topo.verts_num);
  subdivide_uv_base_lvl0(base, topo, base_uvs, face_uv_0, vert_uv_0);

  if (target_lvl == 1) {
    emit_uv_corners_lvl1(base, topo, vert_uv_0, edge_uv_0, face_uv_0, r_uvs);
    return;
  }

  Array<float2> vert_uv[2];
  Array<float2> edge_uv[2];
  Array<float2> face_uv[2];

  vert_uv[0] = std::move(vert_uv_0);
  edge_uv[0] = std::move(edge_uv_0);
  face_uv[0] = std::move(face_uv_0);

  int curr_buffer = 0;
  for (int lvl = 1; lvl < target_lvl; lvl++) {
    const int next_buffer = 1 - curr_buffer;
    const int next_edge_size = Multiresolution::grid_size(lvl + 1);

    vert_uv[next_buffer].reinitialize(topo.verts_num);
    edge_uv[next_buffer].reinitialize(topo.edges_num * next_edge_size);
    face_uv[next_buffer].reinitialize(base.faces_num * next_edge_size * next_edge_size);

    subdivide_uv_base(lvl,
                      topo,
                      vert_uv[curr_buffer],
                      edge_uv[curr_buffer],
                      face_uv[curr_buffer],
                      vert_uv[next_buffer]);

    subdivide_uv_borders(lvl,
                         topo,
                         base,
                         vert_uv[curr_buffer],
                         edge_uv[curr_buffer],
                         face_uv[curr_buffer],
                         edge_uv[next_buffer]);

    subdivide_uv_grids(lvl,
                       topo,
                       base,
                       vert_uv[curr_buffer],
                       edge_uv[curr_buffer],
                       face_uv[curr_buffer],
                       face_uv[next_buffer]);

    curr_buffer = next_buffer;
  }

  emit_uv_corners(topo,
                  base,
                  target_lvl,
                  base_lvl,
                  vert_uv[curr_buffer].as_span(),
                  edge_uv[curr_buffer].as_span(),
                  face_uv[curr_buffer].as_span(),
                  virtual_base_order,
                  r_uvs);
}

void calc_smooth_uvs(const Mesh &base_mesh,
                     const Span<float2> base_uvs,
                     const int target_lvl,
                     MutableSpan<float2> r_uvs,
                     const bool virtual_base_order)
{
  if (base_uvs.is_empty() || r_uvs.is_empty()) {
    return;
  }
  if (target_lvl <= 0) {
    r_uvs.copy_from(base_uvs);
    return;
  }
  const int base_lvl = base_mesh.multires->base_level(target_lvl);
  if (base_mesh.multires->base_all_quads || target_lvl == 1) {
    calc_smooth_uvs_impl(base_mesh, base_uvs, target_lvl, base_lvl, virtual_base_order, r_uvs);
    return;
  }

  Array<float2> uv_lvl1(uv_total_faces(1, base_mesh.faces_num, base_mesh.corners_num) * 4);
  calc_smooth_uvs_impl(base_mesh, base_uvs, 1, 0, true, uv_lvl1.as_mutable_span());
  const Mesh &base1 = base_mesh.multires->runtime->ensure_base_mesh(base_mesh, 1, false);
  calc_smooth_uvs_impl(
      base1, uv_lvl1.as_span(), target_lvl - 1, base_lvl - 1, virtual_base_order, r_uvs);
}

Span<float2> calc_smooth_uvs(const Mesh &base_mesh,
                             const Span<float2> base_uvs,
                             const int target_lvl,
                             Array<float2> &storage,
                             const bool virtual_base_order)
{
  if (target_lvl <= 0 || base_uvs.is_empty()) {
    return base_uvs;
  }
  storage.reinitialize(uv_total_faces(target_lvl, base_mesh.faces_num, base_mesh.corners_num) * 4);
  calc_smooth_uvs(base_mesh, base_uvs, target_lvl, storage.as_mutable_span(), virtual_base_order);
  return storage;
}

}  // namespace blender::bke::multiresolution::uv
