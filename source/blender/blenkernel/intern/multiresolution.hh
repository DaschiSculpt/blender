/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 *
 * Shared subdivision helpers for multiresolution: the exact 1D sliding-window row
 * stencil used for both positions (multiresolution.cc) and UVs (multiresolution_uvs.cc).
 * Keeping a single copy guarantees the two paths can't drift apart.
 *
 * More shared multires helpers (e.g. tangents) can live here in the future.
 */

#pragma once

#include <algorithm>

#include "BLI_span.hh"

namespace blender::bke::multiresolution {

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

}  // namespace blender::bke::multiresolution
