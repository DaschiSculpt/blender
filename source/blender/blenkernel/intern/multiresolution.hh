/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BKE_multiresolution.hh"

#include "BLI_compiler_compat.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_offset_indices.hh"
#include "BLI_span.hh"

#include <algorithm>

namespace blender::bke::multiresolution {

struct LinearOffset {
  int start;
  int stride;

  BLI_INLINE_METHOD int operator[](const int index) const
  {
    return start + index * stride;
  }
};

struct TableOffset {
  Span<int> offsets;

  BLI_INLINE_METHOD int operator[](const int index) const
  {
    return offsets[index];
  }
};

inline void make_linear_policies(const int lvl,
                                 const BaseTopology &base_topo,
                                 LinearOffset &vert_policy,
                                 LinearOffset &edge_policy,
                                 LinearOffset &face_policy)
{
  const int grid_size = Multiresolution::grid_size(lvl);
  vert_policy = {0, 1};
  edge_policy = {base_topo.verts_num, grid_size};
  face_policy = {base_topo.verts_num + base_topo.edges_num * grid_size, grid_size * grid_size};
}

template<typename T>
void subdivision_kernel(const int out_col,
                        const T vsum0,
                        const T vsum1,
                        const T vsum2,
                        const T b0,
                        const T b1,
                        const T b2,
                        const T m0,
                        const T m1,
                        const T m2,
                        const T t0,
                        const T t1,
                        const T t2,
                        const MutableSpan<T> out_even,
                        const MutableSpan<T> out_odd)
{
  out_even[out_col] = (vsum0 + vsum1) * 0.25f;
  out_even[out_col + 1] = vsum1 * 0.375f + (vsum0 + vsum2) * 0.0625f;

  out_odd[out_col] = ((m0 + m1) * 0.375f) + ((b0 + b1 + t0 + t1) * 0.0625f);

  out_odd[out_col + 1] = ((b0 + b2 + t0 + t2) + (b1 + t1 + m0 + m2) * 6.0f + (m1) * 36.0f) *
                         (1.0f / 64.0f);
}

template<typename T>
void run_row(const int lower_size,
             const int row,
             const Span<T> left_border,
             const Span<T> right_border,
             const Span<T> row_bot,
             const Span<T> row_mid,
             const Span<T> row_top,
             const MutableSpan<T> out_even,
             const MutableSpan<T> out_odd)
{
  T b0 = left_border[row], m0 = left_border[row + 1], t0 = left_border[row + 2];
  T vsum0 = b0 + m0;

  T b1 = row_bot[0], m1 = row_mid[0], t1 = row_top[0];
  T vsum1 = b1 + m1;

  for (int col = 0; col < lower_size - 1; col++) {
    const T b2 = row_bot[col + 1];
    const T m2 = row_mid[col + 1];
    const T t2 = row_top[col + 1];
    const T vsum2 = b2 + m2;

    subdivision_kernel(
        col * 2, vsum0, vsum1, vsum2, b0, b1, b2, m0, m1, m2, t0, t1, t2, out_even, out_odd);

    b0 = b1;
    m0 = m1;
    t0 = t1;
    vsum0 = vsum1;
    b1 = b2;
    m1 = m2;
    t1 = t2;
    vsum1 = vsum2;
  }

  subdivision_kernel((lower_size - 1) * 2,
                     vsum0,
                     vsum1,
                     right_border[row] + right_border[row + 1],
                     b0,
                     b1,
                     right_border[row],
                     m0,
                     m1,
                     right_border[row + 1],
                     t0,
                     t1,
                     right_border[row + 2],
                     out_even,
                     out_odd);

  const T f1 = (row_bot[lower_size - 1] + right_border[row] + row_mid[lower_size - 1] +
                right_border[row + 1]) *
               0.25f;
  const T f2 = (row_mid[lower_size - 1] + right_border[row + 1] + row_top[lower_size - 1] +
                right_border[row + 2]) *
               0.25f;
  out_even[2 * lower_size] = f1;
  out_odd[2 * lower_size] = (f1 + f2 + row_mid[lower_size - 1] + right_border[row + 1]) * 0.25f;
}

template<typename T>
void fill_border(
    const T c1, const T c2, const bool reverse, const Span<T> border, const MutableSpan<T> buffer)
{
  int n = buffer.size() - 1;
  buffer[0] = c1;
  buffer[n] = c2;

  if (reverse) {
    for (int i = 0; i < n - 1; i++) {
      buffer[i + 1] = border[n - 2 - i];
    }
  }
  else {
    for (int i = 0; i < n - 1; i++) {
      buffer[i + 1] = border[i];
    }
  }
}

template<typename T>
void extract_grid_inner_border(const int size,
                               const int grid_offset,
                               const int local_edge,
                               const bool reverse,
                               const Span<T> attr,
                               const MutableSpan<T> r_inner_border)
{
  if (local_edge & 1) {
    const int offset = local_edge == 1 ? size - 1 : 0;

    for (int i = 0; i < size; i++) {
      r_inner_border[i] = attr[grid_offset + offset + i * size];
    }
  }
  else {
    const int offset = local_edge == 0 ? 0 : size * (size - 1);
    r_inner_border.copy_from(attr.slice(grid_offset + offset, size));
  }

  if ((local_edge >= 2) ^ reverse) {
    std::reverse(r_inner_border.begin(), r_inner_border.end());
  }
}

template<typename T, typename InPolicy>
void fill_border_side(const int size,
                      const Span<int> corner_edges,
                      const Span<int> corner_verts,
                      const Span<int2> edges,
                      const int corner_start,
                      const int local_edge,
                      const bool reverse,
                      const int face_offset,
                      const InPolicy in_edge_offsets,
                      const Span<T> in_attr,
                      const MutableSpan<T> side)
{
  extract_grid_inner_border(size, face_offset, local_edge, reverse, in_attr, side.slice(1, size));

  const int prev_edge_local = (local_edge + 3) & 3;
  const int prev_edge_idx = corner_edges[corner_start + prev_edge_local];
  const int vert_start = corner_verts[corner_start + local_edge];
  const int2 prev_edge = edges[prev_edge_idx];
  const int prev_pt_idx = (prev_edge[0] == vert_start) ? 0 : size - 1;

  const int next_edge_local = (local_edge + 1) & 3;
  const int next_edge_idx = corner_edges[corner_start + next_edge_local];
  const int vert_end = corner_verts[corner_start + next_edge_local];
  const int2 next_edge = edges[next_edge_idx];
  const int next_pt_idx = (next_edge[0] == vert_end) ? 0 : size - 1;

  const int start_slot = reverse ? (size + 1) : 0;
  const int end_slot = reverse ? 0 : (size + 1);
  side[start_slot] = in_attr[in_edge_offsets[prev_edge_idx] + prev_pt_idx];
  side[end_slot] = in_attr[in_edge_offsets[next_edge_idx] + next_pt_idx];
}

void propagate_tangent_space_linear(const BaseTopology &base_topo,
                                    int delta_lvl,
                                    LinearOffset vert_offsets,
                                    LinearOffset edge_offsets,
                                    LinearOffset face_offsets,
                                    GroupedSpan<int> cyclic_vert_to_corners,
                                    Span<float3> old_smooth_positions,
                                    Span<float3> new_smooth_positions,
                                    Span<float3> old_positions,
                                    MutableSpan<float3> r_new_positions);

}  // namespace blender::bke::multiresolution
