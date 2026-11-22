/* SPDX-FileCopyrightText: 2026 Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#include "BKE_multiresolution.hh"

#include "BLI_assert.hh"
#include "BLI_math_base.hh"
#include "BLI_math_vector.hh"
#include "BLI_structure_of_array_conversion.hh"
#include "BLI_task.hh"
#include "BLI_utildefines.hh"
#include "BLI_vector.hh"

#include "multiresolution.hh"

#include <array>
#include <cfloat>
#include <cmath>
#include <functional>
#include <numbers>
#include <utility>

namespace blender::bke::multiresolution {

namespace {

struct TangentPair {
  float3 t1 = float3(0.0f);
  float3 t2 = float3(0.0f);
};

struct OldNewTangents {
  TangentPair old_tangents;
  TangentPair new_tangents;
};

template<typename OffsetPolicy> struct Context {
  int grid_size;
  const BaseTopology &base_topo;
  OffsetPolicy vert_offsets;
  OffsetPolicy edge_offsets;
  OffsetPolicy face_offsets;
  Span<float3> old_smooth;
  Span<float3> new_smooth;
  Span<float3> old_positions;
  MutableSpan<float3> new_positions;
  float tangent_len_low;
  float inv_tangent_len_span;
};

BLI_INLINE float non_negative(const float x)
{
  return (x > 0.0f) ? x : 0.0f;
}

BLI_INLINE float smoothstep(const float edge0, const float inv_span, const float x)
{
  const float u = (x - edge0) * inv_span;
  const float t0 = (u > 0.0f) ? u : 0.0f;
  const float t = (t0 < 1.0f) ? t0 : 1.0f;
  return t * t * (3.0f - 2.0f * t);
}

constexpr int MAX_GRID_SIZE = 33;
constexpr int CHUNK_CAPACITY = 256;
constexpr int TANGENT_MAX_ELEMENT_VERTS = 31 * 31;

struct SoaVec3 {
  alignas(32) float x[CHUNK_CAPACITY];
  alignas(32) float y[CHUNK_CAPACITY];
  alignas(32) float z[CHUNK_CAPACITY];

  BLI_INLINE_METHOD void set(const int index, const float3 &v)
  {
    x[index] = v.x;
    y[index] = v.y;
    z[index] = v.z;
  }

  BLI_INLINE_METHOD float3 get(const int index) const
  {
    return float3(x[index], y[index], z[index]);
  }

  BLI_INLINE_METHOD void unpack_from(const Span<float3> src, const int dst_offset = 0)
  {
    const int count = int(src.size());
    if (count >= 16) {
      aos_to_soa(count, src.data(), x + dst_offset, y + dst_offset, z + dst_offset);
    }
    else {
      for (int i = 0; i < count; i++) {
        set(dst_offset + i, src[i]);
      }
    }
  }

  BLI_INLINE_METHOD void pack_to(const MutableSpan<float3> dst, const int src_offset = 0) const
  {
    const int count = int(dst.size());
    if (count >= 16) {
      soa_to_aos(count, x + src_offset, y + src_offset, z + src_offset, dst.data());
    }
    else {
      for (int i = 0; i < count; i++) {
        dst[i] = get(src_offset + i);
      }
    }
  }
};

struct ChunkData {
  SoaVec3 old_smooth;
  SoaVec3 new_smooth;
  SoaVec3 pos;
  SoaVec3 t1_old;
  SoaVec3 t2_old;
  SoaVec3 t1_new;
  SoaVec3 t2_new;

  BLI_INLINE_METHOD void unpack_tangents(const Span<float3> t1_old_src,
                                         const Span<float3> t2_old_src,
                                         const Span<float3> t1_new_src,
                                         const Span<float3> t2_new_src)
  {
    t1_old.unpack_from(t1_old_src);
    t2_old.unpack_from(t2_old_src);
    t1_new.unpack_from(t1_new_src);
    t2_new.unpack_from(t2_new_src);
  }
};

template<typename OffsetPolicy>
static void gather_slices_to_soa(const IndexRange range,
                                 const int verts_per_elem,
                                 const OffsetPolicy offsets,
                                 const Span<float3> src,
                                 SoaVec3 &r_soa)
{
  int dst_offset = 0;
  for (const int elem_idx : range) {
    r_soa.unpack_from(src.slice(offsets[elem_idx], verts_per_elem), dst_offset);
    dst_offset += verts_per_elem;
  }
}

template<typename OffsetPolicy>
static void scatter_soa_to_slices(const IndexRange range,
                                  const int verts_per_elem,
                                  const SoaVec3 &src_soa,
                                  const OffsetPolicy offsets,
                                  const MutableSpan<float3> r_dst)
{
  int src_offset = 0;
  for (const int elem_idx : range) {
    src_soa.pack_to(r_dst.slice(offsets[elem_idx], verts_per_elem), src_offset);
    src_offset += verts_per_elem;
  }
}

static void apply_tangent_displacement(const int count,
                                       ChunkData &soa,
                                       const float tangent_len_low,
                                       const float inv_tangent_len_span)
{
  BLI_assert(count >= 0 && count <= CHUNK_CAPACITY);
  const int n = math::min(count, CHUNK_CAPACITY);

  const float *__restrict t1_old_x = soa.t1_old.x;
  const float *__restrict t1_old_y = soa.t1_old.y;
  const float *__restrict t1_old_z = soa.t1_old.z;
  const float *__restrict t2_old_x = soa.t2_old.x;
  const float *__restrict t2_old_y = soa.t2_old.y;
  const float *__restrict t2_old_z = soa.t2_old.z;
  const float *__restrict t1_new_x = soa.t1_new.x;
  const float *__restrict t1_new_y = soa.t1_new.y;
  const float *__restrict t1_new_z = soa.t1_new.z;
  const float *__restrict t2_new_x = soa.t2_new.x;
  const float *__restrict t2_new_y = soa.t2_new.y;
  const float *__restrict t2_new_z = soa.t2_new.z;
  const float *__restrict old_smooth_x = soa.old_smooth.x;
  const float *__restrict old_smooth_y = soa.old_smooth.y;
  const float *__restrict old_smooth_z = soa.old_smooth.z;
  const float *__restrict new_smooth_x = soa.new_smooth.x;
  const float *__restrict new_smooth_y = soa.new_smooth.y;
  const float *__restrict new_smooth_z = soa.new_smooth.z;
  float *__restrict pos_x = soa.pos.x;
  float *__restrict pos_y = soa.pos.y;
  float *__restrict pos_z = soa.pos.z;

  int any_active = 0;

  for (int i = 0; i < n; i++) {
    const int has_detail = int(pos_x[i] != old_smooth_x[i]) | int(pos_y[i] != old_smooth_y[i]) |
                           int(pos_z[i] != old_smooth_z[i]);
    const int tangents_differ = int(t1_old_x[i] != t1_new_x[i]) | int(t1_old_y[i] != t1_new_y[i]) |
                                int(t1_old_z[i] != t1_new_z[i]) | int(t2_old_x[i] != t2_new_x[i]) |
                                int(t2_old_y[i] != t2_new_y[i]) | int(t2_old_z[i] != t2_new_z[i]);
    any_active |= (has_detail & tangents_differ);
  }
  if (any_active == 0) {
    for (int i = 0; i < n; i++) {
      pos_x[i] += new_smooth_x[i] - old_smooth_x[i];
      pos_y[i] += new_smooth_y[i] - old_smooth_y[i];
      pos_z[i] += new_smooth_z[i] - old_smooth_z[i];
    }
    return;
  }

  constexpr float max_len_sq = 1e36f;
  constexpr float new_len_eps = 1e-15f;
  constexpr float max_area_ratio = 1e24f;
  constexpr float sin_low = 1e-3f;
  constexpr float sin_low_sq = sin_low * sin_low;
  constexpr float sin_high = 5e-3f;
  constexpr float inv_sin_sq_span = 1.0f / (sin_high * sin_high - sin_low_sq);
  constexpr float max_amplification = 16.0f;
  constexpr float max_amplification_sq = max_amplification * max_amplification;

  for (int i = 0; i < n; i++) {
    const float old_t1_x = t1_old_x[i];
    const float old_t1_y = t1_old_y[i];
    const float old_t1_z = t1_old_z[i];
    const float old_t2_x = t2_old_x[i];
    const float old_t2_y = t2_old_y[i];
    const float old_t2_z = t2_old_z[i];

    const float new_t1_x = t1_new_x[i];
    const float new_t1_y = t1_new_y[i];
    const float new_t1_z = t1_new_z[i];
    const float new_t2_x = t2_new_x[i];
    const float new_t2_y = t2_new_y[i];
    const float new_t2_z = t2_new_z[i];

    const bool tangents_same = (old_t1_x == new_t1_x) && (old_t1_y == new_t1_y) &&
                               (old_t1_z == new_t1_z) && (old_t2_x == new_t2_x) &&
                               (old_t2_y == new_t2_y) && (old_t2_z == new_t2_z);

    const float old_len1_sq = old_t1_x * old_t1_x + old_t1_y * old_t1_y + old_t1_z * old_t1_z;
    const float old_len2_sq = old_t2_x * old_t2_x + old_t2_y * old_t2_y + old_t2_z * old_t2_z;
    const float new_len1_sq = new_t1_x * new_t1_x + new_t1_y * new_t1_y + new_t1_z * new_t1_z;
    const float new_len2_sq = new_t2_x * new_t2_x + new_t2_y * new_t2_y + new_t2_z * new_t2_z;

    const float old_len1 = std::sqrt(non_negative(old_len1_sq));
    const float old_len2 = std::sqrt(non_negative(old_len2_sq));
    const float new_len1 = std::sqrt(non_negative(new_len1_sq));
    const float new_len2 = std::sqrt(non_negative(new_len2_sq));

    const float min_old_len = math::min(old_len1, old_len2);
    bool old_valid = !tangents_same && (min_old_len > tangent_len_low) &&
                     (old_len1_sq <= max_len_sq) && (old_len2_sq <= max_len_sq) &&
                     (new_len1_sq <= max_len_sq) && (new_len2_sq <= max_len_sq);
    float old_weight = smoothstep(tangent_len_low, inv_tangent_len_span, min_old_len);

    const float safe_old_len1 = old_valid ? old_len1 : 1.0f;
    const float safe_old_len2 = old_valid ? old_len2 : 1.0f;
    const float inv_old_len_prod = 1.0f / (safe_old_len1 * safe_old_len2);

    const float old_unit_cross_x = (old_t1_y * old_t2_z - old_t1_z * old_t2_y) * inv_old_len_prod;
    const float old_unit_cross_y = (old_t1_z * old_t2_x - old_t1_x * old_t2_z) * inv_old_len_prod;
    const float old_unit_cross_z = (old_t1_x * old_t2_y - old_t1_y * old_t2_x) * inv_old_len_prod;

    const float sin_old_sq = old_unit_cross_x * old_unit_cross_x +
                             old_unit_cross_y * old_unit_cross_y +
                             old_unit_cross_z * old_unit_cross_z;
    old_valid = old_valid && (sin_old_sq > sin_low_sq) && (sin_old_sq <= 2.0f);
    const float safe_sin_old_sq = old_valid ? sin_old_sq : 1.0f;
    old_weight *= smoothstep(sin_low_sq, inv_sin_sq_span, safe_sin_old_sq);

    const float min_new_len = math::min(new_len1, new_len2);
    bool new_normal_valid = old_valid && (min_new_len > new_len_eps);
    const float safe_new_len1 = new_normal_valid ? new_len1 : 1.0f;
    const float safe_new_len2 = new_normal_valid ? new_len2 : 1.0f;
    const float new_len_prod = safe_new_len1 * safe_new_len2;
    const float inv_new_len_prod = 1.0f / new_len_prod;

    const float new_unit_cross_x = (new_t1_y * new_t2_z - new_t1_z * new_t2_y) * inv_new_len_prod;
    const float new_unit_cross_y = (new_t1_z * new_t2_x - new_t1_x * new_t2_z) * inv_new_len_prod;
    const float new_unit_cross_z = (new_t1_x * new_t2_y - new_t1_y * new_t2_x) * inv_new_len_prod;

    const float sin_new_sq = new_unit_cross_x * new_unit_cross_x +
                             new_unit_cross_y * new_unit_cross_y +
                             new_unit_cross_z * new_unit_cross_z;
    const float raw_len_prod_ratio = new_len_prod * inv_old_len_prod;
    new_normal_valid = new_normal_valid && (sin_new_sq > sin_low_sq) && (sin_new_sq <= 2.0f) &&
                       (raw_len_prod_ratio <= max_area_ratio);
    const float safe_sin_new_sq = new_normal_valid ? sin_new_sq : 1.0f;
    const float new_normal_weight = new_normal_valid ?
                                        smoothstep(sin_low_sq, inv_sin_sq_span, safe_sin_new_sq) :
                                        0.0f;

    const float sin_product = std::sqrt(non_negative(safe_sin_old_sq * safe_sin_new_sq));
    const float inv_sin_denom = 1.0f / (safe_sin_old_sq * sin_product);
    const float inv_dual_basis_scale = (sin_product * inv_sin_denom) * inv_old_len_prod;
    const float len_prod_ratio = new_normal_valid ? raw_len_prod_ratio : 0.0f;
    const float normal_scale = new_normal_weight *
                               std::sqrt(non_negative(len_prod_ratio * inv_sin_denom));

    const float orig_pos_x = pos_x[i];
    const float orig_pos_y = pos_y[i];
    const float orig_pos_z = pos_z[i];

    const float old_detail_x = orig_pos_x - old_smooth_x[i];
    const float old_detail_y = orig_pos_y - old_smooth_y[i];
    const float old_detail_z = orig_pos_z - old_smooth_z[i];

    const float dual_u_x = old_t2_y * old_unit_cross_z - old_t2_z * old_unit_cross_y;
    const float dual_u_y = old_t2_z * old_unit_cross_x - old_t2_x * old_unit_cross_z;
    const float dual_u_z = old_t2_x * old_unit_cross_y - old_t2_y * old_unit_cross_x;
    const float coord_u = (old_detail_x * dual_u_x + old_detail_y * dual_u_y +
                           old_detail_z * dual_u_z) *
                          inv_dual_basis_scale;

    const float dual_v_x = old_unit_cross_y * old_t1_z - old_unit_cross_z * old_t1_y;
    const float dual_v_y = old_unit_cross_z * old_t1_x - old_unit_cross_x * old_t1_z;
    const float dual_v_z = old_unit_cross_x * old_t1_y - old_unit_cross_y * old_t1_x;
    const float coord_v = (old_detail_x * dual_v_x + old_detail_y * dual_v_y +
                           old_detail_z * dual_v_z) *
                          inv_dual_basis_scale;

    const float normal_disp = (old_detail_x * old_unit_cross_x + old_detail_y * old_unit_cross_y +
                               old_detail_z * old_unit_cross_z) *
                              normal_scale;

    const float new_detail_x = coord_u * new_t1_x + coord_v * new_t2_x +
                               normal_disp * new_unit_cross_x;
    const float new_detail_y = coord_u * new_t1_y + coord_v * new_t2_y +
                               normal_disp * new_unit_cross_y;
    const float new_detail_z = coord_u * new_t1_z + coord_v * new_t2_z +
                               normal_disp * new_unit_cross_z;

    const float old_detail_len_sq = old_detail_x * old_detail_x + old_detail_y * old_detail_y +
                                    old_detail_z * old_detail_z;
    const float new_detail_len_sq = new_detail_x * new_detail_x + new_detail_y * new_detail_y +
                                    new_detail_z * new_detail_z;
    const bool transformed_valid = old_valid && (old_detail_len_sq >= FLT_MIN) &&
                                   (old_detail_len_sq <= FLT_MAX) &&
                                   (new_detail_len_sq <= FLT_MAX);
    const bool needs_clamp = transformed_valid &&
                             (new_detail_len_sq > max_amplification_sq * old_detail_len_sq);
    const float safe_old_detail_len_sq = needs_clamp ? old_detail_len_sq : 0.0f;
    const float safe_new_detail_len_sq = needs_clamp ? new_detail_len_sq : 1.0f;
    const float amplification_scale = needs_clamp ?
                                          (max_amplification *
                                           std::sqrt(non_negative(safe_old_detail_len_sq /
                                                                  safe_new_detail_len_sq))) :
                                          1.0f;
    const float final_scale = old_weight * amplification_scale;

    const float out_detail_x = transformed_valid ? (new_detail_x * final_scale) : 0.0f;
    const float out_detail_y = transformed_valid ? (new_detail_y * final_scale) : 0.0f;
    const float out_detail_z = transformed_valid ? (new_detail_z * final_scale) : 0.0f;

    const float smooth_dx = new_smooth_x[i] - old_smooth_x[i];
    const float smooth_dy = new_smooth_y[i] - old_smooth_y[i];
    const float smooth_dz = new_smooth_z[i] - old_smooth_z[i];

    pos_x[i] = tangents_same ? (orig_pos_x + smooth_dx) : (new_smooth_x[i] + out_detail_x);
    pos_y[i] = tangents_same ? (orig_pos_y + smooth_dy) : (new_smooth_y[i] + out_detail_y);
    pos_z[i] = tangents_same ? (orig_pos_z + smooth_dz) : (new_smooth_z[i] + out_detail_z);
  }
}

template<typename OffsetPolicy, typename EvalFn>
static void run_unit_tangent_chunks(const int count,
                                    const OffsetPolicy offsets,
                                    const Context<OffsetPolicy> &ctx,
                                    EvalFn &&eval_element)
{
  threading::parallel_for(IndexRange(count), 4096, [&](const IndexRange range) {
    ChunkData chunk_data;

    for (int64_t i = range.start(); i < range.one_after_last(); i += CHUNK_CAPACITY) {
      const IndexRange chunk_range(i,
                                   math::min<int64_t>(CHUNK_CAPACITY, range.one_after_last() - i));
      const int chunk_size = int(chunk_range.size());

      int offset = 0;
      for (const int element_idx : chunk_range) {
        const OldNewTangents tangents = eval_element(element_idx);
        chunk_data.t1_old.set(offset, tangents.old_tangents.t1);
        chunk_data.t2_old.set(offset, tangents.old_tangents.t2);
        chunk_data.t1_new.set(offset, tangents.new_tangents.t1);
        chunk_data.t2_new.set(offset, tangents.new_tangents.t2);
        offset++;
      }

      offset = 0;
      for (const int element_idx : chunk_range) {
        const int vert_idx = offsets[element_idx];
        chunk_data.old_smooth.set(offset, ctx.old_smooth[vert_idx]);
        chunk_data.new_smooth.set(offset, ctx.new_smooth[vert_idx]);
        chunk_data.pos.set(offset, ctx.old_positions[vert_idx]);
        offset++;
      }

      apply_tangent_displacement(
          chunk_size, chunk_data, ctx.tangent_len_low, ctx.inv_tangent_len_span);

      offset = 0;
      for (const int element_idx : chunk_range) {
        ctx.new_positions[offsets[element_idx]] = chunk_data.pos.get(offset);
        offset++;
      }
    }
  });
}

template<typename OffsetPolicy, typename ScratchT, typename EvalFn>
static void calc_tangent_chunks(const int count,
                                const int verts_per_element,
                                const OffsetPolicy offsets,
                                const Context<OffsetPolicy> &ctx,
                                EvalFn &&eval_element)
{
  const int grain_size = math::max(1, 4096 / verts_per_element);

  if (verts_per_element > CHUNK_CAPACITY) {
    BLI_assert(verts_per_element <= TANGENT_MAX_ELEMENT_VERTS);
    threading::parallel_for(IndexRange(count), grain_size, [&](const IndexRange range) {
      ScratchT scratch;
      std::array<float3, TANGENT_MAX_ELEMENT_VERTS> t1_old, t2_old, t1_new, t2_new;
      ChunkData chunk_data;

      for (const int element_idx : range) {
        eval_element(element_idx,
                     scratch,
                     MutableSpan<float3>(t1_old.data(), verts_per_element),
                     MutableSpan<float3>(t2_old.data(), verts_per_element),
                     MutableSpan<float3>(t1_new.data(), verts_per_element),
                     MutableSpan<float3>(t2_new.data(), verts_per_element));

        const int element_offset = offsets[element_idx];
        for (int i = 0; i < verts_per_element; i += CHUNK_CAPACITY) {
          const int chunk_size = math::min(CHUNK_CAPACITY, verts_per_element - i);
          const int vert_offset = element_offset + i;

          chunk_data.unpack_tangents(Span<float3>(t1_old.data() + i, chunk_size),
                                     Span<float3>(t2_old.data() + i, chunk_size),
                                     Span<float3>(t1_new.data() + i, chunk_size),
                                     Span<float3>(t2_new.data() + i, chunk_size));
          chunk_data.old_smooth.unpack_from(ctx.old_smooth.slice(vert_offset, chunk_size));
          chunk_data.new_smooth.unpack_from(ctx.new_smooth.slice(vert_offset, chunk_size));
          chunk_data.pos.unpack_from(ctx.old_positions.slice(vert_offset, chunk_size));

          apply_tangent_displacement(
              chunk_size, chunk_data, ctx.tangent_len_low, ctx.inv_tangent_len_span);

          chunk_data.pos.pack_to(ctx.new_positions.slice(vert_offset, chunk_size));
        }
      }
    });
    return;
  }

  const int elements_per_chunk = CHUNK_CAPACITY / verts_per_element;

  threading::parallel_for(IndexRange(count), grain_size, [&](const IndexRange range) {
    ScratchT scratch;
    std::array<float3, CHUNK_CAPACITY> t1_old, t2_old, t1_new, t2_new;
    ChunkData chunk_data;

    for (int64_t i = range.start(); i < range.one_after_last(); i += elements_per_chunk) {
      const IndexRange chunk_range(
          i, math::min<int64_t>(elements_per_chunk, range.one_after_last() - i));

      int offset = 0;
      for (const int element_idx : chunk_range) {
        eval_element(element_idx,
                     scratch,
                     MutableSpan<float3>(t1_old.data() + offset, verts_per_element),
                     MutableSpan<float3>(t2_old.data() + offset, verts_per_element),
                     MutableSpan<float3>(t1_new.data() + offset, verts_per_element),
                     MutableSpan<float3>(t2_new.data() + offset, verts_per_element));
        offset += verts_per_element;
      }

      const int chunk_verts = int(chunk_range.size()) * verts_per_element;
      chunk_data.unpack_tangents(Span<float3>(t1_old.data(), chunk_verts),
                                 Span<float3>(t2_old.data(), chunk_verts),
                                 Span<float3>(t1_new.data(), chunk_verts),
                                 Span<float3>(t2_new.data(), chunk_verts));
      gather_slices_to_soa(
          chunk_range, verts_per_element, offsets, ctx.old_smooth, chunk_data.old_smooth);
      gather_slices_to_soa(
          chunk_range, verts_per_element, offsets, ctx.new_smooth, chunk_data.new_smooth);
      gather_slices_to_soa(
          chunk_range, verts_per_element, offsets, ctx.old_positions, chunk_data.pos);

      apply_tangent_displacement(
          chunk_verts, chunk_data, ctx.tangent_len_low, ctx.inv_tangent_len_span);

      scatter_soa_to_slices(
          chunk_range, verts_per_element, chunk_data.pos, offsets, ctx.new_positions);
    }
  });
}

BLI_INLINE TangentPair calc_regular_interior_tangents(const float3 &right,
                                                      const float3 &left,
                                                      const float3 &top,
                                                      const float3 &bottom,
                                                      const float3 &top_right,
                                                      const float3 &top_left,
                                                      const float3 &bottom_left,
                                                      const float3 &bottom_right)
{
  constexpr float inv_12 = 1.0f / 12.0f;
  return {
      (4.0f * (right - left) + (top_right + bottom_right - top_left - bottom_left)) * inv_12,
      (4.0f * (top - bottom) + (top_left + top_right - bottom_left - bottom_right)) * inv_12,
  };
}

BLI_INLINE TangentPair calc_regular_boundary_tangents(const float3 &center,
                                                      const float3 &boundary_prev,
                                                      const float3 &boundary_next,
                                                      const float3 &interior_neighbor,
                                                      const float3 &interior_diag_prev,
                                                      const float3 &interior_diag_next)
{
  constexpr float inv_6 = 1.0f / 6.0f;
  return {
      0.5f * (boundary_next - boundary_prev),
      (4.0f * (interior_neighbor - center) + (interior_diag_prev - boundary_prev) +
       (interior_diag_next - boundary_next)) *
          inv_6,
  };
}

BLI_INLINE OldNewTangents calc_extraordinary_interior_tangents(const int center_index,
                                                               const Span<int> edge_ring_indices,
                                                               const Span<int> diag_ring_indices,
                                                               const Span<float3> old_smooth,
                                                               const Span<float3> new_smooth)
{
  const int valence = int(edge_ring_indices.size());
  if (valence < 2) {
    return {};
  }

  const float valence_f = float(valence);
  const float pi = std::numbers::pi_v<float>;
  const float cos_half_sector = math::cos(pi / valence_f);
  const float edge_weight = cos_half_sector + math::sqrt(cos_half_sector * cos_half_sector + 4.0f);
  const float unit_scale = 2.0f / (valence_f * (edge_weight + 2.0f * cos_half_sector));
  const float sector_angle = (2.0f * pi) / valence_f;

  const float3 old_center = old_smooth[center_index];
  const float3 new_center = new_smooth[center_index];

  float3 old_tangent_1(0.0f);
  float3 old_tangent_2(0.0f);
  float3 new_tangent_1(0.0f);
  float3 new_tangent_2(0.0f);

  for (int sector = 0; sector < valence; sector++) {
    const float edge_angle = sector_angle * float(sector);
    const float diag_angle = sector_angle * (float(sector) + 0.5f);
    const float edge_cos_weight = edge_weight * math::cos(edge_angle);
    const float edge_sin_weight = edge_weight * math::sin(edge_angle);
    const float diag_cos_weight = math::cos(diag_angle);
    const float diag_sin_weight = math::sin(diag_angle);

    const int edge_neighbor = edge_ring_indices[sector];
    const int diag_neighbor = diag_ring_indices[sector];
    const float3 old_edge_offset = old_smooth[edge_neighbor] - old_center;
    const float3 old_diag_offset = old_smooth[diag_neighbor] - old_center;
    const float3 new_edge_offset = new_smooth[edge_neighbor] - new_center;
    const float3 new_diag_offset = new_smooth[diag_neighbor] - new_center;

    old_tangent_1 += old_edge_offset * edge_cos_weight + old_diag_offset * diag_cos_weight;
    old_tangent_2 += old_edge_offset * edge_sin_weight + old_diag_offset * diag_sin_weight;
    new_tangent_1 += new_edge_offset * edge_cos_weight + new_diag_offset * diag_cos_weight;
    new_tangent_2 += new_edge_offset * edge_sin_weight + new_diag_offset * diag_sin_weight;
  }

  return {
      {old_tangent_1 * unit_scale, old_tangent_2 * unit_scale},
      {new_tangent_1 * unit_scale, new_tangent_2 * unit_scale},
  };
}

BLI_INLINE OldNewTangents calc_extraordinary_boundary_tangents(const int center_index,
                                                               const int end_boundary_edge_idx,
                                                               const Span<int> edge_ring_indices,
                                                               const Span<int> diag_ring_indices,
                                                               const Span<float3> old_smooth,
                                                               const Span<float3> new_smooth)
{
  const int sectors_num = edge_ring_indices.size();
  const float valence = float(sectors_num);
  const float pi = std::numbers::pi_v<float>;
  const float half_sector_angle = pi / (2.0f * valence);
  const float cos_half_sector = math::cos(half_sector_angle);
  const float sin_half_sector = math::sin(half_sector_angle);
  const float edge_weight = cos_half_sector + math::sqrt(cos_half_sector * cos_half_sector + 4.0f);
  const float subdominant_eigenvalue = (2.0f + cos_half_sector * edge_weight) * 0.125f;

  const float boundary_edge_weight = (4.0f * subdominant_eigenvalue * sin_half_sector -
                                      (8.0f * subdominant_eigenvalue - 1.0f) /
                                          (2.0f * sin_half_sector)) /
                                     (4.0f * subdominant_eigenvalue - 1.0f);
  const float unit_scale = 2.0f / (valence * (edge_weight + 2.0f * cos_half_sector));

  const float3 old_tangent_1 = 0.5f * (old_smooth[end_boundary_edge_idx] -
                                       old_smooth[edge_ring_indices[0]]);
  const float3 new_tangent_1 = 0.5f * (new_smooth[end_boundary_edge_idx] -
                                       new_smooth[edge_ring_indices[0]]);

  const float3 old_center = old_smooth[center_index];
  const float3 new_center = new_smooth[center_index];

  float3 old_tangent_2 = boundary_edge_weight *
                         (old_smooth[edge_ring_indices[0]] + old_smooth[end_boundary_edge_idx] -
                          2.0f * old_center);
  float3 new_tangent_2 = boundary_edge_weight *
                         (new_smooth[edge_ring_indices[0]] + new_smooth[end_boundary_edge_idx] -
                          2.0f * new_center);

  old_tangent_2 += (old_smooth[diag_ring_indices[0]] - old_center) * sin_half_sector;
  new_tangent_2 += (new_smooth[diag_ring_indices[0]] - new_center) * sin_half_sector;

  const float sector_step = pi / valence;
  for (int sector = 1; sector < sectors_num; sector++) {
    const float sector_f = float(sector);
    const float diag_sin_weight = math::sin(sector_step * (sector_f + 0.5f));
    const float edge_sin_weight = edge_weight * math::sin(sector_step * sector_f);
    old_tangent_2 += (old_smooth[diag_ring_indices[sector]] - old_center) * diag_sin_weight +
                     (old_smooth[edge_ring_indices[sector]] - old_center) * edge_sin_weight;
    new_tangent_2 += (new_smooth[diag_ring_indices[sector]] - new_center) * diag_sin_weight +
                     (new_smooth[edge_ring_indices[sector]] - new_center) * edge_sin_weight;
  }

  return {
      {old_tangent_1, old_tangent_2 * unit_scale},
      {new_tangent_1, new_tangent_2 * unit_scale},
  };
}

template<typename OffsetPolicy>
BLI_INLINE int gather_base_edge_neighbor_index(const int vert,
                                               const int edge,
                                               const int grid_size,
                                               const Span<int2> edges,
                                               const OffsetPolicy edge_offsets)
{
  if (grid_size == 1) {
    return edge_offsets[edge];
  }
  const int2 edge_verts = edges[edge];
  const int edge_pos = (edge_verts[0] == vert) ? 0 : (grid_size - 1);
  return edge_offsets[edge] + edge_pos;
}

template<typename OffsetPolicy>
BLI_INLINE void gather_base_corner_ring_indices(const int vert,
                                                const int corner,
                                                const Context<OffsetPolicy> &ctx,
                                                int &r_edge_neighbor_index,
                                                int &r_diag_neighbor_index)
{
  const BaseTopology &base_topo = ctx.base_topo;
  const int edge = base_topo.corner_edges[corner];
  const int face = base_topo.corner_to_face_map[corner];

  r_edge_neighbor_index = gather_base_edge_neighbor_index(
      vert, edge, ctx.grid_size, base_topo.edges, ctx.edge_offsets);

  if (ctx.grid_size == 1) {
    r_diag_neighbor_index = ctx.face_offsets[face];
    return;
  }

  const IndexRange face_range = base_topo.faces[face];
  BLI_assert(face_range.size() == 4);
  const int local_corner = corner - int(face_range.start());
  const int row = (local_corner < 2) ? 0 : (ctx.grid_size - 1);
  const int col = (local_corner == 0 || local_corner == 3) ? 0 : (ctx.grid_size - 1);
  r_diag_neighbor_index = ctx.face_offsets[face] + row * ctx.grid_size + col;
}

template<typename OffsetPolicy>
BLI_INLINE void gather_base_ring_indices(const int vert,
                                         const Span<int> vert_corners,
                                         const Context<OffsetPolicy> &ctx,
                                         const MutableSpan<int> r_edge_ring,
                                         const MutableSpan<int> r_diag_ring)
{
  for (const int sector : vert_corners.index_range()) {
    gather_base_corner_ring_indices(
        vert, vert_corners[sector], ctx, r_edge_ring[sector], r_diag_ring[sector]);
  }
}

template<typename OffsetPolicy>
void calc_tangents_base(const Context<OffsetPolicy> &ctx,
                        const GroupedSpan<int> cyclic_vert_to_corners)
{
  const BaseTopology &base_topo = ctx.base_topo;
  const OffsetPolicy vert_offsets = ctx.vert_offsets;
  const Span<int2> edges = base_topo.edges;
  const OffsetIndices<int> faces = base_topo.faces;
  const Span<int> corner_edges = base_topo.corner_edges;
  const Span<int> corner_to_face = base_topo.corner_to_face_map;
  const BitSpan boundary_verts = base_topo.boundary_verts;
  const bool has_cyclic_corners = !cyclic_vert_to_corners.is_empty();

  run_unit_tangent_chunks(
      base_topo.verts_num, vert_offsets, ctx, [&](const int vert_idx) -> OldNewTangents {
        if (!has_cyclic_corners) {
          return {};
        }
        const Span<int> vert_corners = cyclic_vert_to_corners[vert_idx];
        const int face_valence = int(vert_corners.size());
        if (face_valence == 0 || vert_corners[0] < 0) {
          return {};
        }

        const int vert_pos_idx = vert_offsets[vert_idx];

        if (boundary_verts[vert_idx]) {
          const int last_corner = vert_corners[face_valence - 1];
          const IndexRange last_face = faces[corner_to_face[last_corner]];
          const int prev_corner = (last_corner == last_face.start()) ? int(last_face.last()) :
                                                                       (last_corner - 1);
          const int end_boundary_edge_idx = gather_base_edge_neighbor_index(
              vert_idx, corner_edges[prev_corner], ctx.grid_size, edges, ctx.edge_offsets);

          if (face_valence == 1) {
            int edge_0, diag_0;
            gather_base_corner_ring_indices(vert_idx, vert_corners[0], ctx, edge_0, diag_0);
            auto eval = [&](const Span<float3> smooth) -> TangentPair {
              return {smooth[end_boundary_edge_idx] - smooth[edge_0],
                      smooth[diag_0] - smooth[vert_pos_idx]};
            };
            return {eval(ctx.old_smooth), eval(ctx.new_smooth)};
          }
          if (face_valence == 2) {
            std::array<int, 2> edge_ring, diag_ring;
            gather_base_ring_indices(vert_idx, vert_corners, ctx, edge_ring, diag_ring);
            auto eval = [&](const Span<float3> smooth) {
              return calc_regular_boundary_tangents(smooth[vert_pos_idx],
                                                    smooth[edge_ring[0]],
                                                    smooth[end_boundary_edge_idx],
                                                    smooth[edge_ring[1]],
                                                    smooth[diag_ring[0]],
                                                    smooth[diag_ring[1]]);
            };
            return {eval(ctx.old_smooth), eval(ctx.new_smooth)};
          }

          Vector<int, 16> edge_ring_indices(face_valence);
          Vector<int, 16> diag_ring_indices(face_valence);
          gather_base_ring_indices(
              vert_idx, vert_corners, ctx, edge_ring_indices, diag_ring_indices);
          return calc_extraordinary_boundary_tangents(vert_pos_idx,
                                                      end_boundary_edge_idx,
                                                      edge_ring_indices,
                                                      diag_ring_indices,
                                                      ctx.old_smooth,
                                                      ctx.new_smooth);
        }

        if (face_valence == 4) {
          std::array<int, 4> edge_ring, diag_ring;
          gather_base_ring_indices(vert_idx, vert_corners, ctx, edge_ring, diag_ring);
          auto eval = [&](const Span<float3> smooth) {
            return calc_regular_interior_tangents(smooth[edge_ring[0]],
                                                  smooth[edge_ring[2]],
                                                  smooth[edge_ring[1]],
                                                  smooth[edge_ring[3]],
                                                  smooth[diag_ring[0]],
                                                  smooth[diag_ring[1]],
                                                  smooth[diag_ring[2]],
                                                  smooth[diag_ring[3]]);
          };
          return {eval(ctx.old_smooth), eval(ctx.new_smooth)};
        }

        Vector<int, 16> edge_ring_indices(face_valence);
        Vector<int, 16> diag_ring_indices(face_valence);
        gather_base_ring_indices(
            vert_idx, vert_corners, ctx, edge_ring_indices, diag_ring_indices);
        return calc_extraordinary_interior_tangents(
            vert_pos_idx, edge_ring_indices, diag_ring_indices, ctx.old_smooth, ctx.new_smooth);
      });
}

struct EdgeFaceSide {
  int face;
  int local_edge;
  bool reversed;
};

BLI_INLINE EdgeFaceSide make_edge_face_side(const int edge_idx,
                                            const int face_idx,
                                            const BaseTopology &base_topo)
{
  const IndexRange face_range = base_topo.faces[face_idx];
  const int local_edge = int(base_topo.corner_edges.slice(face_range).first_index(edge_idx));
  const bool reversed = base_topo.face_edge_reversed[face_range.start() + local_edge];
  return {face_idx, local_edge, reversed};
}

BLI_INLINE std::pair<EdgeFaceSide, EdgeFaceSide> order_adjacent_faces_for_edge(
    const int edge_idx, const Span<int> adj_faces, const BaseTopology &base_topo)
{
  EdgeFaceSide left = make_edge_face_side(edge_idx, adj_faces[0], base_topo);
  EdgeFaceSide right = make_edge_face_side(edge_idx, adj_faces[1], base_topo);

  if (left.reversed && !right.reversed) {
    std::swap(left, right);
  }
  return {left, right};
}

template<typename OffsetPolicy>
BLI_INLINE std::array<int, 3> gather_border_side_indices_lvl1(const EdgeFaceSide &side,
                                                              const Context<OffsetPolicy> &ctx)
{
  const IndexRange face_range = ctx.base_topo.faces[side.face];
  const int face_corners_num = int(face_range.size());
  const Span<int> face_edges = ctx.base_topo.corner_edges.slice(face_range);
  const int prev_edge = face_edges[(side.local_edge + face_corners_num - 1) % face_corners_num];
  const int next_edge = face_edges[(side.local_edge + 1) % face_corners_num];
  const int start_adj_edge = side.reversed ? next_edge : prev_edge;
  const int end_adj_edge = side.reversed ? prev_edge : next_edge;
  return {ctx.edge_offsets[start_adj_edge],
          ctx.face_offsets[side.face],
          ctx.edge_offsets[end_adj_edge]};
}

struct EdgeBorderScratch {
  std::array<float3, MAX_GRID_SIZE> edge;
  std::array<float3, MAX_GRID_SIZE> left;
  std::array<float3, MAX_GRID_SIZE> right;
};

struct FaceBorderScratch {
  std::array<float3, MAX_GRID_SIZE> bottom;
  std::array<float3, MAX_GRID_SIZE> top;
  std::array<float3, MAX_GRID_SIZE> left;
  std::array<float3, MAX_GRID_SIZE> right;
};

BLI_INLINE void calc_regular_tangent_row(const int grid_size,
                                         const Span<float3> row_bottom,
                                         const Span<float3> row_mid,
                                         const Span<float3> row_top,
                                         const float3 &bottom_left_init,
                                         const float3 &mid_left_init,
                                         const float3 &top_left_init,
                                         const float3 &bottom_right_end,
                                         const float3 &mid_right_end,
                                         const float3 &top_right_end,
                                         const MutableSpan<float3> r_t1,
                                         const MutableSpan<float3> r_t2)
{
  constexpr float inv_12 = 1.0f / 12.0f;

  float3 bottom_left = bottom_left_init;
  float3 mid_left = mid_left_init;
  float3 top_left = top_left_init;
  float3 col_diff_left = top_left - bottom_left;

  float3 bottom_center = row_bottom[0];
  float3 mid_center = row_mid[0];
  float3 top_center = row_top[0];
  float3 col_diff_center = top_center - bottom_center;

  auto eval_cell = [&](const int col,
                       const float3 &bottom_right,
                       const float3 &mid_right,
                       const float3 &top_right) {
    const float3 col_diff_right = top_right - bottom_right;
    r_t1[col] = (4.0f * (mid_right - mid_left) +
                 ((top_right - top_left) + (bottom_right - bottom_left))) *
                inv_12;
    r_t2[col] = (4.0f * col_diff_center + (col_diff_left + col_diff_right)) * inv_12;

    bottom_left = bottom_center;
    mid_left = mid_center;
    top_left = top_center;
    col_diff_left = col_diff_center;
    bottom_center = bottom_right;
    mid_center = mid_right;
    top_center = top_right;
    col_diff_center = col_diff_right;
  };

  for (int col = 0; col < grid_size - 1; col++) {
    eval_cell(col, row_bottom[col + 1], row_mid[col + 1], row_top[col + 1]);
  }
  eval_cell(grid_size - 1, bottom_right_end, mid_right_end, top_right_end);
}

static void calc_edge_boundary_tangents(const int grid_size,
                                        const Span<float3> edge_buf,
                                        const Span<float3> side_buf,
                                        const MutableSpan<float3> r_t1,
                                        const MutableSpan<float3> r_t2)
{
  constexpr float inv_6 = 1.0f / 6.0f;
  float3 inward_left = side_buf[0] - edge_buf[0];
  float3 inward_center = side_buf[1] - edge_buf[1];
  for (int k = 0; k < grid_size; k++) {
    const float3 inward_right = side_buf[k + 2] - edge_buf[k + 2];
    r_t1[k] = 0.5f * (edge_buf[k + 2] - edge_buf[k]);
    r_t2[k] = (4.0f * inward_center + (inward_left + inward_right)) * inv_6;
    inward_left = inward_center;
    inward_center = inward_right;
  }
}

template<typename OffsetPolicy>
static void calc_edge_tangents(const int edge_idx,
                               const Context<OffsetPolicy> &ctx,
                               EdgeBorderScratch &scratch,
                               const MutableSpan<float3> r_t1_old,
                               const MutableSpan<float3> r_t2_old,
                               const MutableSpan<float3> r_t1_new,
                               const MutableSpan<float3> r_t2_new)
{
  const BaseTopology &base_topo = ctx.base_topo;
  const Span<int> adj_faces = base_topo.edge_to_face_map[edge_idx];
  if (adj_faces.size() != 1 && adj_faces.size() != 2) {
    r_t1_old.fill(float3(0.0f));
    r_t2_old.fill(float3(0.0f));
    r_t1_new.fill(float3(0.0f));
    r_t2_new.fill(float3(0.0f));
    return;
  }

  const int grid_size = ctx.grid_size;
  const int border_len = grid_size + 2;
  const MutableSpan<float3> edge_buf(scratch.edge.data(), border_len);
  const MutableSpan<float3> left_buf(scratch.left.data(), border_len);
  const MutableSpan<float3> right_buf(scratch.right.data(), border_len);

  const int2 edge = base_topo.edges[edge_idx];
  const int v0 = ctx.vert_offsets[edge[0]];
  const int v1 = ctx.vert_offsets[edge[1]];
  const int edge_offset = ctx.edge_offsets[edge_idx];

  auto fill_side = [&](const EdgeFaceSide &side,
                       const Span<float3> smooth,
                       const MutableSpan<float3> side_buf) {
    fill_border_side(grid_size,
                     base_topo.corner_edges,
                     base_topo.corner_verts,
                     base_topo.edges,
                     int(base_topo.faces[side.face].start()),
                     side.local_edge,
                     side.reversed,
                     ctx.face_offsets[side.face],
                     ctx.edge_offsets,
                     smooth,
                     side_buf);
  };

  if (adj_faces.size() == 1) {
    const EdgeFaceSide side = make_edge_face_side(edge_idx, adj_faces[0], base_topo);
    auto eval_smooth = [&](const Span<float3> smooth,
                           const MutableSpan<float3> r_t1,
                           const MutableSpan<float3> r_t2) {
      fill_border(smooth[v0], smooth[v1], false, smooth.slice(edge_offset, grid_size), edge_buf);
      fill_side(side, smooth, left_buf);
      calc_edge_boundary_tangents(grid_size, edge_buf, left_buf, r_t1, r_t2);
    };
    eval_smooth(ctx.old_smooth, r_t1_old, r_t2_old);
    eval_smooth(ctx.new_smooth, r_t1_new, r_t2_new);
  }
  else {
    const auto [left, right] = order_adjacent_faces_for_edge(edge_idx, adj_faces, base_topo);
    auto eval_smooth = [&](const Span<float3> smooth,
                           const MutableSpan<float3> r_t1,
                           const MutableSpan<float3> r_t2) {
      fill_border(smooth[v0], smooth[v1], false, smooth.slice(edge_offset, grid_size), edge_buf);
      fill_side(left, smooth, left_buf);
      fill_side(right, smooth, right_buf);
      calc_regular_tangent_row(grid_size,
                               right_buf.slice(1, grid_size),
                               edge_buf.slice(1, grid_size),
                               left_buf.slice(1, grid_size),
                               right_buf[0],
                               edge_buf[0],
                               left_buf[0],
                               right_buf[grid_size + 1],
                               edge_buf[grid_size + 1],
                               left_buf[grid_size + 1],
                               r_t1,
                               r_t2);
    };
    eval_smooth(ctx.old_smooth, r_t1_old, r_t2_old);
    eval_smooth(ctx.new_smooth, r_t1_new, r_t2_new);
  }
}

template<typename OffsetPolicy> void calc_tangents_borders(const Context<OffsetPolicy> &ctx)
{
  const BaseTopology &base_topo = ctx.base_topo;
  const OffsetPolicy vert_offsets = ctx.vert_offsets;
  const OffsetPolicy edge_offsets = ctx.edge_offsets;

  if (ctx.grid_size == 1) {
    const Span<int2> edges = base_topo.edges;
    const GroupedSpan<int> edge_to_faces = base_topo.edge_to_face_map;

    run_unit_tangent_chunks(
        base_topo.edges_num, edge_offsets, ctx, [&](const int edge_idx) -> OldNewTangents {
          const int edge_pos_idx = edge_offsets[edge_idx];
          const int2 edge = edges[edge_idx];
          const int start_vert_pos_idx = vert_offsets[edge[0]];
          const int end_vert_pos_idx = vert_offsets[edge[1]];
          const Span<int> adj_faces = edge_to_faces[edge_idx];

          if (adj_faces.size() == 1) {
            const EdgeFaceSide side = make_edge_face_side(edge_idx, adj_faces[0], base_topo);
            const std::array<int, 3> side_indices = gather_border_side_indices_lvl1(side, ctx);
            auto eval = [&](const Span<float3> smooth) {
              return calc_regular_boundary_tangents(smooth[edge_pos_idx],
                                                    smooth[start_vert_pos_idx],
                                                    smooth[end_vert_pos_idx],
                                                    smooth[side_indices[1]],
                                                    smooth[side_indices[0]],
                                                    smooth[side_indices[2]]);
            };
            return {eval(ctx.old_smooth), eval(ctx.new_smooth)};
          }

          if (adj_faces.size() == 2) {
            const auto [left,
                        right] = order_adjacent_faces_for_edge(edge_idx, adj_faces, base_topo);
            const std::array<int, 3> left_side = gather_border_side_indices_lvl1(left, ctx);
            const std::array<int, 3> right_side = gather_border_side_indices_lvl1(right, ctx);
            auto eval = [&](const Span<float3> smooth) {
              return calc_regular_interior_tangents(smooth[end_vert_pos_idx],
                                                    smooth[start_vert_pos_idx],
                                                    smooth[left_side[1]],
                                                    smooth[right_side[1]],
                                                    smooth[left_side[2]],
                                                    smooth[left_side[0]],
                                                    smooth[right_side[0]],
                                                    smooth[right_side[2]]);
            };
            return {eval(ctx.old_smooth), eval(ctx.new_smooth)};
          }

          return {};
        });
    return;
  }

  BLI_assert(ctx.grid_size <= 31 && ctx.grid_size <= TANGENT_MAX_ELEMENT_VERTS);
  calc_tangent_chunks<OffsetPolicy, EdgeBorderScratch>(
      base_topo.edges_num,
      ctx.grid_size,
      ctx.edge_offsets,
      ctx,
      [&](const int edge_idx,
          EdgeBorderScratch &scratch,
          const MutableSpan<float3> r_t1_old,
          const MutableSpan<float3> r_t2_old,
          const MutableSpan<float3> r_t1_new,
          const MutableSpan<float3> r_t2_new) {
        calc_edge_tangents(edge_idx, ctx, scratch, r_t1_old, r_t2_old, r_t1_new, r_t2_new);
      });
}

static void calc_grid_tangents(const int grid_size,
                               const Span<float3> grid,
                               const Span<float3> bottom_buf,
                               const Span<float3> top_buf,
                               const Span<float3> left_buf,
                               const Span<float3> right_buf,
                               const MutableSpan<float3> r_t1,
                               const MutableSpan<float3> r_t2)
{
  for (int row = 0, row_offset = 0; row < grid_size; row++, row_offset += grid_size) {
    const Span<float3> row_bottom = (row == 0) ? bottom_buf.slice(1, grid_size) :
                                                 grid.slice(row_offset - grid_size, grid_size);
    const Span<float3> row_mid = grid.slice(row_offset, grid_size);
    const Span<float3> row_top = (row == grid_size - 1) ?
                                     top_buf.slice(1, grid_size) :
                                     grid.slice(row_offset + grid_size, grid_size);
    calc_regular_tangent_row(grid_size,
                             row_bottom,
                             row_mid,
                             row_top,
                             left_buf[row],
                             left_buf[row + 1],
                             left_buf[row + 2],
                             right_buf[row],
                             right_buf[row + 1],
                             right_buf[row + 2],
                             r_t1.slice(row_offset, grid_size),
                             r_t2.slice(row_offset, grid_size));
  }
}

template<typename OffsetPolicy>
static void calc_face_tangents(const int face_idx,
                               const Context<OffsetPolicy> &ctx,
                               FaceBorderScratch &scratch,
                               const MutableSpan<float3> r_t1_old,
                               const MutableSpan<float3> r_t2_old,
                               const MutableSpan<float3> r_t1_new,
                               const MutableSpan<float3> r_t2_new)
{
  const BaseTopology &base_topo = ctx.base_topo;
  const IndexRange face_range = base_topo.faces[face_idx];
  BLI_assert(face_range.size() == 4);

  const Span<int> face_verts = base_topo.corner_verts.slice(face_range);
  const Span<int> face_edges = base_topo.corner_edges.slice(face_range);
  const int corner_offset = int(face_range.start());

  const int vert_bottom_left = ctx.vert_offsets[face_verts[0]];
  const int vert_bottom_right = ctx.vert_offsets[face_verts[1]];
  const int vert_top_right = ctx.vert_offsets[face_verts[2]];
  const int vert_top_left = ctx.vert_offsets[face_verts[3]];
  const int edge_bottom = ctx.edge_offsets[face_edges[0]];
  const int edge_right = ctx.edge_offsets[face_edges[1]];
  const int edge_top = ctx.edge_offsets[face_edges[2]];
  const int edge_left = ctx.edge_offsets[face_edges[3]];
  const bool reverse_bottom = base_topo.face_edge_reversed[corner_offset + 0];
  const bool reverse_right = base_topo.face_edge_reversed[corner_offset + 1];
  const bool reverse_top = !base_topo.face_edge_reversed[corner_offset + 2];
  const bool reverse_left = !base_topo.face_edge_reversed[corner_offset + 3];

  const int grid_size = ctx.grid_size;
  const int border_len = grid_size + 2;
  const int grid_size_sq = grid_size * grid_size;
  const int face_offset = ctx.face_offsets[face_idx];
  const MutableSpan<float3> bottom_buf(scratch.bottom.data(), border_len);
  const MutableSpan<float3> top_buf(scratch.top.data(), border_len);
  const MutableSpan<float3> left_buf(scratch.left.data(), border_len);
  const MutableSpan<float3> right_buf(scratch.right.data(), border_len);

  auto eval_smooth = [&](const Span<float3> smooth,
                         const MutableSpan<float3> r_t1,
                         const MutableSpan<float3> r_t2) {
    fill_border(smooth[vert_bottom_left],
                smooth[vert_bottom_right],
                reverse_bottom,
                smooth.slice(edge_bottom, grid_size),
                bottom_buf);
    fill_border(smooth[vert_bottom_right],
                smooth[vert_top_right],
                reverse_right,
                smooth.slice(edge_right, grid_size),
                right_buf);
    fill_border(smooth[vert_top_left],
                smooth[vert_top_right],
                reverse_top,
                smooth.slice(edge_top, grid_size),
                top_buf);
    fill_border(smooth[vert_bottom_left],
                smooth[vert_top_left],
                reverse_left,
                smooth.slice(edge_left, grid_size),
                left_buf);
    calc_grid_tangents(grid_size,
                       smooth.slice(face_offset, grid_size_sq),
                       bottom_buf,
                       top_buf,
                       left_buf,
                       right_buf,
                       r_t1,
                       r_t2);
  };

  eval_smooth(ctx.old_smooth, r_t1_old, r_t2_old);
  eval_smooth(ctx.new_smooth, r_t1_new, r_t2_new);
}

template<typename OffsetPolicy> void calc_tangents_grids(const Context<OffsetPolicy> &ctx)
{
  const BaseTopology &base_topo = ctx.base_topo;
  const OffsetPolicy vert_offsets = ctx.vert_offsets;
  const OffsetPolicy edge_offsets = ctx.edge_offsets;
  const OffsetPolicy face_offsets = ctx.face_offsets;

  if (ctx.grid_size == 1) {
    const OffsetIndices<int> faces = base_topo.faces;
    const Span<int> corner_verts = base_topo.corner_verts;
    const Span<int> corner_edges = base_topo.corner_edges;

    run_unit_tangent_chunks(
        base_topo.faces_num, face_offsets, ctx, [&](const int face_idx) -> OldNewTangents {
          const IndexRange face_range = faces[face_idx];
          const int face_corners_num = int(face_range.size());
          const int face_pos_idx = face_offsets[face_idx];
          const Span<int> face_verts = corner_verts.slice(face_range);
          const Span<int> face_edges = corner_edges.slice(face_range);

          if (face_corners_num == 4) {
            const int bottom_left = vert_offsets[face_verts[0]];
            const int bottom_right = vert_offsets[face_verts[1]];
            const int top_right = vert_offsets[face_verts[2]];
            const int top_left = vert_offsets[face_verts[3]];
            const int bottom = edge_offsets[face_edges[0]];
            const int right = edge_offsets[face_edges[1]];
            const int top = edge_offsets[face_edges[2]];
            const int left = edge_offsets[face_edges[3]];

            auto eval = [&](const Span<float3> smooth) {
              return calc_regular_interior_tangents(smooth[right],
                                                    smooth[left],
                                                    smooth[top],
                                                    smooth[bottom],
                                                    smooth[top_right],
                                                    smooth[top_left],
                                                    smooth[bottom_left],
                                                    smooth[bottom_right]);
            };
            return {eval(ctx.old_smooth), eval(ctx.new_smooth)};
          }

          if (face_corners_num >= 3) {
            Vector<int, 16> edge_ring_indices(face_corners_num);
            Vector<int, 16> diag_ring_indices(face_corners_num);
            for (int corner = 0; corner < face_corners_num; corner++) {
              edge_ring_indices[corner] = edge_offsets[face_edges[corner]];
              diag_ring_indices[corner] =
                  vert_offsets[face_verts[(corner + 1) % face_corners_num]];
            }
            return calc_extraordinary_interior_tangents(face_pos_idx,
                                                        edge_ring_indices,
                                                        diag_ring_indices,
                                                        ctx.old_smooth,
                                                        ctx.new_smooth);
          }

          return {};
        });
    return;
  }

  const int grid_size_sq = ctx.grid_size * ctx.grid_size;
  BLI_assert(ctx.grid_size <= 31 && grid_size_sq <= TANGENT_MAX_ELEMENT_VERTS);
  calc_tangent_chunks<OffsetPolicy, FaceBorderScratch>(
      base_topo.faces_num,
      grid_size_sq,
      ctx.face_offsets,
      ctx,
      [&](const int face_idx,
          FaceBorderScratch &scratch,
          const MutableSpan<float3> r_t1_old,
          const MutableSpan<float3> r_t2_old,
          const MutableSpan<float3> r_t1_new,
          const MutableSpan<float3> r_t2_new) {
        calc_face_tangents(face_idx, ctx, scratch, r_t1_old, r_t2_old, r_t1_new, r_t2_new);
      });
}

}  // namespace

template<typename OffsetPolicy>
static float calc_ref_edge_len(const Span<int2> base_edges,
                               const OffsetPolicy vert_offsets,
                               const Span<float3> positions,
                               const int delta_lvl)
{
  if (base_edges.is_empty()) {
    return 0.0f;
  }
  const double total_len = threading::parallel_reduce(
      base_edges.index_range(),
      4096,
      0.0,
      [&](const IndexRange range, const double init) {
        double sum = init;
        for (const int edge_idx : range) {
          const int2 edge = base_edges[edge_idx];
          sum += double(
              math::distance(positions[vert_offsets[edge[0]]], positions[vert_offsets[edge[1]]]));
        }
        return sum;
      },
      std::plus<double>());
  const float avg_edge_len = float(total_len / double(base_edges.size()));
  return std::ldexp(avg_edge_len, -delta_lvl);
}

template<typename OffsetPolicy>
static void propagate_tangent_space_internal(const BaseTopology &base_topo,
                                             const int delta_lvl,
                                             const OffsetPolicy vert_offsets,
                                             const OffsetPolicy edge_offsets,
                                             const OffsetPolicy face_offsets,
                                             const GroupedSpan<int> cyclic_vert_to_corners,
                                             const Span<float3> old_smooth_positions,
                                             const Span<float3> new_smooth_positions,
                                             const Span<float3> old_positions,
                                             const MutableSpan<float3> r_new_positions)
{
  BLI_assert(delta_lvl >= 1);
  BLI_assert(old_smooth_positions.size() == r_new_positions.size());
  BLI_assert(new_smooth_positions.size() == r_new_positions.size());
  BLI_assert(old_positions.size() == r_new_positions.size());

  const float ref_edge_len = calc_ref_edge_len(
      base_topo.edges, vert_offsets, old_smooth_positions, delta_lvl);
  const float tangent_len_low = math::max(1e-7f, 1e-4f * ref_edge_len);
  const float tangent_len_high = math::max(1e-6f, 1e-3f * ref_edge_len);
  const float tangent_len_span = tangent_len_high - tangent_len_low;
  const float inv_tangent_len_span = (tangent_len_span > 0.0f) ? (1.0f / tangent_len_span) : 0.0f;

  const Context<OffsetPolicy> ctx{
      Multiresolution::grid_size(delta_lvl),
      base_topo,
      vert_offsets,
      edge_offsets,
      face_offsets,
      old_smooth_positions,
      new_smooth_positions,
      old_positions,
      r_new_positions,
      tangent_len_low,
      inv_tangent_len_span,
  };

  calc_tangents_base(ctx, cyclic_vert_to_corners);
  calc_tangents_borders(ctx);
  calc_tangents_grids(ctx);
}

void propagate_tangent_space(const BaseTopology &base_topo,
                             const int delta_lvl,
                             const OffsetsSpan &offsets,
                             const GroupedSpan<int> cyclic_vert_to_corners,
                             const Span<float3> old_smooth_positions,
                             const Span<float3> new_smooth_positions,
                             const Span<float3> old_positions,
                             const MutableSpan<float3> r_new_positions)
{
  propagate_tangent_space_internal<TableOffset>(base_topo,
                                                delta_lvl,
                                                TableOffset{offsets.verts},
                                                TableOffset{offsets.edges},
                                                TableOffset{offsets.faces},
                                                cyclic_vert_to_corners,
                                                old_smooth_positions,
                                                new_smooth_positions,
                                                old_positions,
                                                r_new_positions);
}

void propagate_tangent_space_linear(const BaseTopology &base_topo,
                                    const int delta_lvl,
                                    const LinearOffset vert_offsets,
                                    const LinearOffset edge_offsets,
                                    const LinearOffset face_offsets,
                                    const GroupedSpan<int> cyclic_vert_to_corners,
                                    const Span<float3> old_smooth_positions,
                                    const Span<float3> new_smooth_positions,
                                    const Span<float3> old_positions,
                                    const MutableSpan<float3> r_new_positions)
{
  propagate_tangent_space_internal<LinearOffset>(base_topo,
                                                 delta_lvl,
                                                 vert_offsets,
                                                 edge_offsets,
                                                 face_offsets,
                                                 cyclic_vert_to_corners,
                                                 old_smooth_positions,
                                                 new_smooth_positions,
                                                 old_positions,
                                                 r_new_positions);
}

}  // namespace blender::bke::multiresolution
