/* SPDX-FileCopyrightText: 2020 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edsculpt
 */
#include "sculpt_smooth.hh"

#include "BLI_enumerable_thread_specific.hh"
#include "BLI_math_base.hh"
#include "BLI_math_geom_c.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_c.hh"

#include "BKE_attribute_math.hh"
#include "BKE_mesh.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_subdiv_ccg.hh"

#include "mesh_brush_common.hh"
#include "sculpt_automask.hh"
#include "sculpt_boundary.hh"
#include "sculpt_color.hh"
#include "sculpt_face_set.hh"
#include "sculpt_hide.hh"
#include "sculpt_intern.hh"

#include "bmesh.hh"

#include <cstdlib>
//
namespace blender::ed::sculpt_paint::smooth {

template<typename T>
void neighbor_data_average_mesh_check_loose(const Span<T> src,
                                            const Span<int> verts,
                                            const GroupedSpan<int> vert_neighbors,
                                            const MutableSpan<T> dst)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(verts.size() == dst.size());
  BLI_assert(vert_neighbors.size() == dst.size());

  for (const int i : vert_neighbors.index_range()) {
    const Span<int> neighbors = vert_neighbors[i];
    if (neighbors.is_empty()) {
      dst[i] = src[verts[i]];
    }
    else {
      dst[i] = bke::attribute_math::mix_indices(src, neighbors);
    }
  }
}

template void neighbor_data_average_mesh_check_loose<float>(Span<float>,
                                                            Span<int>,
                                                            GroupedSpan<int>,
                                                            MutableSpan<float>);
template void neighbor_data_average_mesh_check_loose<float3>(Span<float3>,
                                                             Span<int>,
                                                             GroupedSpan<int>,
                                                             MutableSpan<float3>);

template<typename T>
void neighbor_data_average_mesh(const Span<T> src,
                                const GroupedSpan<int> vert_neighbors,
                                const MutableSpan<T> dst)
{
  PRF_scope(ProfileCategory::Editor);
  /* TODO: Use #bke::attribute_math::mix_groups when we can control the internal threading. */
  BLI_assert(vert_neighbors.size() == dst.size());

  for (const int i : vert_neighbors.index_range()) {
    dst[i] = bke::attribute_math::mix_indices(src, vert_neighbors[i]);
  }
}

template void neighbor_data_average_mesh<float>(Span<float>, GroupedSpan<int>, MutableSpan<float>);
template void neighbor_data_average_mesh<float3>(Span<float3>,
                                                 GroupedSpan<int>,
                                                 MutableSpan<float3>);
template void neighbor_data_average_mesh<float4>(Span<float4>,
                                                 GroupedSpan<int>,
                                                 MutableSpan<float4>);

template<typename T, bool UseHidden>
static void neighbor_data_average_vert_multires_impl(
    const bke::multiresolution::BaseTopology &base_topo,
    const int grid_size,
    const Span<int> verts,
    const Span<int> vert_offsets,
    const Span<int> edge_offsets,
    const Span<T> attr,
    const MutableSpan<T> dst,
    const Span<bool> hidden)
{
  BLI_assert(verts.size() == dst.size());
  const Span<int2> base_edges = base_topo.edges;
  const GroupedSpan<int> vert_to_edges = base_topo.vert_to_edge_map;

  for (const int i : verts.index_range()) {
    const int vert = verts[i];
    Span<int> edges = vert_to_edges[vert];
    const int self_idx = vert_offsets[vert];
    if constexpr (UseHidden) {
      if (hidden[self_idx]) {
        dst[i] = attr[self_idx];
        continue;
      }
    }
    if (edges.is_empty()) {
      dst[i] = attr[self_idx];
      continue;
    }
    if (!base_topo.boundary_verts.is_empty() && base_topo.boundary_verts[vert]) {
      if (edges.size() == 2) {
        dst[i] = attr[self_idx];
        continue;
      }
      T sum = T(0.0f);
      int count = 0;
      for (const int edge_idx : edges) {
        const bool is_boundary_edge = base_topo.boundary_edges.is_empty() ?
                                          false :
                                          bool(base_topo.boundary_edges[edge_idx]);
        if (!is_boundary_edge) {
          continue;
        }
        const int offset = base_edges[edge_idx][0] == vert ? 0 : grid_size - 1;
        const int neighbor_idx = edge_offsets[edge_idx] + offset;
        if constexpr (UseHidden) {
          if (hidden[neighbor_idx]) {
            continue;
          }
        }
        sum += attr[neighbor_idx];
        count++;
      }
      if (count == 0) {
        dst[i] = attr[self_idx];
      }
      else {
        dst[i] = sum / count;
      }
      continue;
    }
    if constexpr (!UseHidden) {
      T sum = T(0.0f);
      for (const int edge_idx : edges) {
        const int offset = base_edges[edge_idx][0] == vert ? 0 : grid_size - 1;
        sum += attr[edge_offsets[edge_idx] + offset];
      }
      dst[i] = sum / edges.size();
      continue;
    }
    T sum = T(0.0f);
    int count = 0;
    for (const int edge_idx : edges) {
      const int offset = base_edges[edge_idx][0] == vert ? 0 : grid_size - 1;
      const int neighbor_idx = edge_offsets[edge_idx] + offset;
      if (hidden[neighbor_idx]) {
        continue;
      }
      sum += attr[neighbor_idx];
      count++;
    }
    if (count == 0) {
      dst[i] = attr[self_idx];
    }
    else {
      dst[i] = sum / count;
    }
  }
}

template<typename T>
static void neighbor_data_average_vert_multires(
    const bke::multiresolution::BaseTopology &base_topo,
    const int grid_size,
    const Span<int> verts,
    const Span<int> vert_offsets,
    const Span<int> edge_offsets,
    const Span<T> attr,
    const MutableSpan<T> dst,
    const Span<bool> hidden)
{
  if (hidden.is_empty()) {
    neighbor_data_average_vert_multires_impl<T, false>(
        base_topo, grid_size, verts, vert_offsets, edge_offsets, attr, dst, {});
  }
  else {
    neighbor_data_average_vert_multires_impl<T, true>(
        base_topo, grid_size, verts, vert_offsets, edge_offsets, attr, dst, hidden);
  }
}

template<typename T, bool UseHidden>
static void neighbor_data_average_edge_multires_impl(
    const Multiresolution &multires,
    const bke::multiresolution::BaseTopology &base_topo,
    const Span<int> edges,
    const int edges_offset,
    const Span<int> vert_offsets,
    const Span<int> face_offsets,
    const Span<T> attr,
    const MutableSpan<T> dst,
    const Span<bool> hidden)
{
  const int grid_size = multires.grid_size();
  const Span<int2> base_edges = base_topo.edges;
  const OffsetIndices<int> base_faces = base_topo.faces;
  const Span<int> corner_edges = base_topo.corner_edges;
  const GroupedSpan<int> edge_to_faces = base_topo.edge_to_face_map;
  const BitSpan face_edge_reversed = base_topo.face_edge_reversed;

  Array<T> inner_border_a(grid_size);
  Array<T> inner_border_b(grid_size);
  int dst_offset = 0;
  int edge_offset = edges_offset;

  for (const int i : edges.index_range()) {
    const int edge_idx = edges[i];
    const int2 edge = base_edges[edge_idx];
    const int v0_idx = vert_offsets[edge[0]];
    const int v1_idx = vert_offsets[edge[1]];
    const T v0 = attr[v0_idx];
    const T v1 = attr[v1_idx];
    const Span<int> adj_faces = edge_to_faces[edge_idx];
    const bool is_boundary = (adj_faces.size() < 2);
    const Span<T> curr_edge_verts = attr.slice(edge_offset, grid_size);
    if (is_boundary) {
      if constexpr (!UseHidden) {
        for (int k = 0; k < grid_size; k++) {
          const T prev = (k == 0) ? v0 : curr_edge_verts[k - 1];
          const T next = (k == grid_size - 1) ? v1 : curr_edge_verts[k + 1];
          dst[dst_offset++] = (prev + next) * 0.5f;
        }
      }
      else {
        for (int k = 0; k < grid_size; k++) {
          if (hidden[edge_offset + k]) {
            dst[dst_offset++] = curr_edge_verts[k];
            continue;
          }
          const int prev_idx = (k == 0) ? v0_idx : edge_offset + k - 1;
          const int next_idx = (k == grid_size - 1) ? v1_idx : edge_offset + k + 1;
          const T prev = (k == 0) ? v0 : curr_edge_verts[k - 1];
          const T next = (k == grid_size - 1) ? v1 : curr_edge_verts[k + 1];
          T sum = T(0.0f);
          int count = 0;
          if (!hidden[prev_idx]) {
            sum += prev;
            count++;
          }
          if (!hidden[next_idx]) {
            sum += next;
            count++;
          }
          if (count == 0) {
            dst[dst_offset++] = curr_edge_verts[k];
          }
          else {
            dst[dst_offset++] = sum / count;
          }
        }
      }
      edge_offset += grid_size;
      continue;
    }

    int face_idx_a, local_a;
    bool reversed_a;
    {
      face_idx_a = adj_faces[0];
      const IndexRange face_range = base_faces[face_idx_a];
      const Span<int> face_edges = corner_edges.slice(face_range);

      local_a = face_edges.first_index(edge_idx);
      reversed_a = face_edge_reversed[face_range.start() + local_a];
      multires.grid_inner_border<T>(
          face_offsets[face_idx_a], local_a, reversed_a, attr, inner_border_a);
    }

    int face_idx_b, local_b;
    bool reversed_b;
    {
      face_idx_b = adj_faces[1];
      const IndexRange face_range = base_faces[face_idx_b];
      const Span<int> face_edges = corner_edges.slice(face_range);

      local_b = face_edges.first_index(edge_idx);
      reversed_b = face_edge_reversed[face_range.start() + local_b];
      multires.grid_inner_border<T>(
          face_offsets[face_idx_b], local_b, reversed_b, attr, inner_border_b);
    }

    if constexpr (!UseHidden) {
      for (int k = 0; k < grid_size; k++) {
        const T prev = (k == 0) ? v0 : curr_edge_verts[k - 1];
        const T next = (k == grid_size - 1) ? v1 : curr_edge_verts[k + 1];
        dst[dst_offset++] = (prev + next + inner_border_a[k] + inner_border_b[k]) * 0.25f;
      }
      edge_offset += grid_size;
      continue;
    }

    const int grid_offset_a = face_offsets[face_idx_a];
    const int grid_offset_b = face_offsets[face_idx_b];
    for (int k = 0; k < grid_size; k++) {
      if (hidden[edge_offset + k]) {
        dst[dst_offset++] = curr_edge_verts[k];
        continue;
      }
      const int prev_idx = (k == 0) ? v0_idx : edge_offset + k - 1;
      const int next_idx = (k == grid_size - 1) ? v1_idx : edge_offset + k + 1;
      const int idx_a = Multiresolution::grid_inner_border_index(
          grid_offset_a, local_a, reversed_a, grid_size, k);
      const int idx_b = Multiresolution::grid_inner_border_index(
          grid_offset_b, local_b, reversed_b, grid_size, k);
      const T prev = (k == 0) ? v0 : curr_edge_verts[k - 1];
      const T next = (k == grid_size - 1) ? v1 : curr_edge_verts[k + 1];
      T sum = T(0.0f);
      int count = 0;
      if (!hidden[prev_idx]) {
        sum += prev;
        count++;
      }
      if (!hidden[next_idx]) {
        sum += next;
        count++;
      }
      if (!hidden[idx_a]) {
        sum += inner_border_a[k];
        count++;
      }
      if (!hidden[idx_b]) {
        sum += inner_border_b[k];
        count++;
      }
      if (count == 0) {
        dst[dst_offset++] = curr_edge_verts[k];
      }
      else {
        dst[dst_offset++] = sum / count;
      }
    }

    edge_offset += grid_size;
  }
}

template<typename T>
static void neighbor_data_average_edge_multires(
    const Multiresolution &multires,
    const bke::multiresolution::BaseTopology &base_topo,
    const Span<int> edges,
    const int edges_offset,
    const Span<int> vert_offsets,
    const Span<int> face_offsets,
    const Span<T> attr,
    const MutableSpan<T> dst,
    const Span<bool> hidden)
{
  if (hidden.is_empty()) {
    neighbor_data_average_edge_multires_impl<T, false>(
        multires, base_topo, edges, edges_offset, vert_offsets, face_offsets, attr, dst, {});
  }
  else {
    neighbor_data_average_edge_multires_impl<T, true>(
        multires, base_topo, edges, edges_offset, vert_offsets, face_offsets, attr, dst, hidden);
  }
}

template<typename T>
BLI_INLINE T calc_inner_border_average_multires(
    const int r, const int c, const int grid_size, const Span<T> attr, const Span<int> sub_faces)
{
  const int face_idx_BL = (r * (grid_size + 1) + c) * 4;
  const int face_idx_TR = ((r + 1) * (grid_size + 1) + (c + 1)) * 4;

  const T pL = attr[sub_faces[face_idx_BL + 3]];
  const T pU = attr[sub_faces[face_idx_TR + 3]];
  const T pR = attr[sub_faces[face_idx_TR + 1]];
  const T pD = attr[sub_faces[face_idx_BL + 1]];

  return (pL + pU + pR + pD) * 0.25f;
}

template<typename T, bool UseHidden>
static void neighbor_data_average_face_multires_impl(
    const Multiresolution &multires,
    const bke::multiresolution::BaseTopology &base_topo,
    const Span<int> node_faces,
    const Span<int> edge_offsets,
    const Span<int> face_offsets,
    const Span<T> attr,
    const MutableSpan<T> dst,
    const Span<bool> hidden)
{
  const Span<int> corner_edges = base_topo.corner_edges;
  const OffsetIndices<int> base_faces = base_topo.faces;
  const int grid_size = multires.grid_size();

  if (grid_size == 1) {
    for (const int i : node_faces.index_range()) {
      const int face_idx = node_faces[i];
      const IndexRange face_range = base_faces[face_idx];
      const int corners_num = face_range.size();
      const int face_start = face_range.start();

      if constexpr (!UseHidden) {
        T sum = T(0.0f);
        for (const int corner : IndexRange(corners_num)) {
          const int edge_idx = corner_edges[face_start + corner];
          sum += attr[edge_offsets[edge_idx]];
        }
        dst[i] = sum / float(corners_num);
        continue;
      }
      if (hidden[face_offsets[face_idx]]) {
        dst[i] = attr[face_offsets[face_idx]];
        continue;
      }
      T sum = T(0.0f);
      int count = 0;
      for (const int corner : IndexRange(corners_num)) {
        const int edge_idx = corner_edges[face_start + corner];
        const int edge_mid_vert = edge_offsets[edge_idx];
        if (hidden[edge_mid_vert]) {
          continue;
        }
        sum += attr[edge_mid_vert];
        count++;
      }
      if (count == 0) {
        dst[i] = attr[face_offsets[face_idx]];
      }
      else {
        dst[i] = sum / float(count);
      }
    }
    return;
  }

  const Span<int> corner_verts = multires.corner_verts();
  const int faces_per_quad_grid = multires.faces_per_corner() * 4;
  const int verts_per_quad_grid = grid_size * grid_size;
  const int faces_offset = face_offsets[node_faces[0]];
  int dst_offset = 0;

  auto border_average_hidden =
      [&](const int r, const int c, const Span<int> sub_faces, const T fallback) -> T {
    const int face_idx_BL = (r * (grid_size + 1) + c) * 4;
    const int face_idx_TR = ((r + 1) * (grid_size + 1) + (c + 1)) * 4;
    const int idx_L = sub_faces[face_idx_BL + 3];
    const int idx_U = sub_faces[face_idx_TR + 3];
    const int idx_R = sub_faces[face_idx_TR + 1];
    const int idx_D = sub_faces[face_idx_BL + 1];
    T sum = T(0.0f);
    int count = 0;
    if (!hidden[idx_L]) {
      sum += attr[idx_L];
      count++;
    }
    if (!hidden[idx_U]) {
      sum += attr[idx_U];
      count++;
    }
    if (!hidden[idx_R]) {
      sum += attr[idx_R];
      count++;
    }
    if (!hidden[idx_D]) {
      sum += attr[idx_D];
      count++;
    }
    if (count == 0) {
      return fallback;
    }
    return sum / count;
  };

  for (const int i : node_faces.index_range()) {
    const int face_idx = node_faces[i];
    const int face_offset = faces_offset + i * verts_per_quad_grid;
    const Span<T> grid = attr.slice(face_offset, verts_per_quad_grid);
    const Span<int> sub_faces = corner_verts.slice(face_idx * faces_per_quad_grid * 4,
                                                   faces_per_quad_grid * 4);

    if constexpr (!UseHidden) {
      for (const int c : IndexRange(grid_size)) {
        dst[dst_offset++] = calc_inner_border_average_multires(0, c, grid_size, attr, sub_faces);
      }
      for (int r = 1; r < grid_size - 1; r++) {
        dst[dst_offset++] = calc_inner_border_average_multires(r, 0, grid_size, attr, sub_faces);
        for (int c = 1; c < grid_size - 1; c++) {
          const int grid_offset = r * grid_size + c;
          dst[dst_offset++] = (grid[grid_offset - 1] + grid[grid_offset + 1] +
                               grid[grid_offset - grid_size] + grid[grid_offset + grid_size]) *
                              0.25f;
        }
        dst[dst_offset++] = calc_inner_border_average_multires(
            r, grid_size - 1, grid_size, attr, sub_faces);
      }
      for (const int c : IndexRange(grid_size)) {
        dst[dst_offset++] = calc_inner_border_average_multires(
            grid_size - 1, c, grid_size, attr, sub_faces);
      }
      continue;
    }

    for (const int c : IndexRange(grid_size)) {
      const int self_idx = face_offset + c;
      if (hidden[self_idx]) {
        dst[dst_offset++] = grid[c];
        continue;
      }
      dst[dst_offset++] = border_average_hidden(0, c, sub_faces, grid[c]);
    }
    for (int r = 1; r < grid_size - 1; r++) {
      {
        const int self_idx = face_offset + r * grid_size;
        if (hidden[self_idx]) {
          dst[dst_offset++] = grid[r * grid_size];
        }
        else {
          dst[dst_offset++] = border_average_hidden(r, 0, sub_faces, grid[r * grid_size]);
        }
      }
      for (int c = 1; c < grid_size - 1; c++) {
        const int grid_offset = r * grid_size + c;
        const int self_idx = face_offset + grid_offset;
        if (hidden[self_idx]) {
          dst[dst_offset++] = grid[grid_offset];
          continue;
        }
        T sum = T(0.0f);
        int count = 0;
        if (!hidden[self_idx - 1]) {
          sum += grid[grid_offset - 1];
          count++;
        }
        if (!hidden[self_idx + 1]) {
          sum += grid[grid_offset + 1];
          count++;
        }
        if (!hidden[self_idx - grid_size]) {
          sum += grid[grid_offset - grid_size];
          count++;
        }
        if (!hidden[self_idx + grid_size]) {
          sum += grid[grid_offset + grid_size];
          count++;
        }
        if (count == 0) {
          dst[dst_offset++] = grid[grid_offset];
        }
        else {
          dst[dst_offset++] = sum / count;
        }
      }
      {
        const int self_idx = face_offset + r * grid_size + grid_size - 1;
        if (hidden[self_idx]) {
          dst[dst_offset++] = grid[r * grid_size + grid_size - 1];
        }
        else {
          dst[dst_offset++] = border_average_hidden(
              r, grid_size - 1, sub_faces, grid[r * grid_size + grid_size - 1]);
        }
      }
    }
    for (const int c : IndexRange(grid_size)) {
      const int self_idx = face_offset + (grid_size - 1) * grid_size + c;
      if (hidden[self_idx]) {
        dst[dst_offset++] = grid[(grid_size - 1) * grid_size + c];
        continue;
      }
      dst[dst_offset++] = border_average_hidden(
          grid_size - 1, c, sub_faces, grid[(grid_size - 1) * grid_size + c]);
    }
  }
}

template<typename T>
static void neighbor_data_average_face_multires(
    const Multiresolution &multires,
    const bke::multiresolution::BaseTopology &base_topo,
    const Span<int> node_faces,
    const Span<int> edge_offsets,
    const Span<int> face_offsets,
    const Span<T> attr,
    const MutableSpan<T> dst,
    const Span<bool> hidden = {})
{
  if (hidden.is_empty()) {
    neighbor_data_average_face_multires_impl<T, false>(
        multires, base_topo, node_faces, edge_offsets, face_offsets, attr, dst, {});
  }
  else {
    neighbor_data_average_face_multires_impl<T, true>(
        multires, base_topo, node_faces, edge_offsets, face_offsets, attr, dst, hidden);
  }
}

template<typename T>
void neighbor_data_average_node_multires(const Multiresolution &multires,
                                         const bke::multiresolution::BaseTopology &base_topo,
                                         const bke::pbvh::MultiresNode &node,
                                         const bke::multiresolution::OffsetsSpan &offsets,
                                         const Span<T> attr,
                                         const MutableSpan<T> dst,
                                         const Span<bool> hidden)
{
  const int grid_size = multires.grid_size();
  neighbor_data_average_vert_multires(base_topo,
                                      grid_size,
                                      node.verts(),
                                      offsets.verts,
                                      offsets.edges,
                                      attr,
                                      dst.slice(node.local_verts_range()),
                                      hidden);
  neighbor_data_average_edge_multires(multires,
                                      base_topo,
                                      node.edges(),
                                      node.edges_offset(),
                                      offsets.verts,
                                      offsets.faces,
                                      attr,
                                      dst.slice(node.local_edges_range(grid_size)),
                                      hidden);
  neighbor_data_average_face_multires(multires,
                                      base_topo,
                                      node.faces(),
                                      offsets.edges,
                                      offsets.faces,
                                      attr,
                                      dst.slice(node.local_faces_range(grid_size)),
                                      hidden);
}

template void neighbor_data_average_node_multires<float>(
    const Multiresolution &multires,
    const bke::multiresolution::BaseTopology &,
    const bke::pbvh::MultiresNode &,
    const bke::multiresolution::OffsetsSpan &,
    const Span<float>,
    const MutableSpan<float>,
    const Span<bool>);

template void neighbor_data_average_node_multires<float3>(
    const Multiresolution &multires,
    const bke::multiresolution::BaseTopology &,
    const bke::pbvh::MultiresNode &,
    const bke::multiresolution::OffsetsSpan &,
    const Span<float3>,
    const MutableSpan<float3>,
    const Span<bool>);

template void neighbor_data_average_node_multires<float4>(
    const Multiresolution &multires,
    const bke::multiresolution::BaseTopology &,
    const bke::pbvh::MultiresNode &,
    const bke::multiresolution::OffsetsSpan &,
    const Span<float4>,
    const MutableSpan<float4>,
    const Span<bool>);

static float3 average_positions(const CCGKey &key,
                                const Span<float3> positions,
                                const Span<SubdivCCGCoord> coords)
{
  PRF_scope(ProfileCategory::Editor);
  const float factor = math::rcp(float(coords.size()));
  float3 result(0);
  for (const SubdivCCGCoord coord : coords) {
    result += positions[coord.to_index(key)] * factor;
  }
  return result;
}

template<bool use_factors>
static void neighbor_position_average_interior_grids_impl(const OffsetIndices<int> faces,
                                                          const Span<int> corner_verts,
                                                          const BitSpan boundary_verts,
                                                          const Set<OrderedEdge> &boundary_edges,
                                                          const SubdivCCG &subdiv_ccg,
                                                          const Span<int> grids,
                                                          const Span<float> factors,
                                                          const MutableSpan<float3> new_positions)
{
  PRF_scope(ProfileCategory::Editor);
  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);
  const Span<float3> positions = subdiv_ccg.positions;

  BLI_assert(grids.size() * key.grid_area == new_positions.size());
  if constexpr (use_factors) {
    BLI_assert(new_positions.size() == factors.size());
  }

  for (const int i : grids.index_range()) {
    const int node_verts_start = i * key.grid_area;
    const int grid = grids[i];
    const IndexRange grid_range = bke::ccg::grid_range(key, grid);

    /* TODO: This loop could be optimized in the future by skipping unnecessary logic for
     * non-boundary grid vertices. */
    for (const int y : IndexRange(key.grid_size)) {
      for (const int x : IndexRange(key.grid_size)) {
        const int offset = CCG_grid_xy_to_index(key.grid_size, x, y);
        const int node_vert_index = node_verts_start + offset;
        const int vert = grid_range[offset];

        if constexpr (use_factors) {
          if (factors[node_vert_index] == 0.0f) {
            new_positions[node_vert_index] = positions[vert];
            continue;
          }
        }

        SubdivCCGCoord coord{};
        coord.grid_index = grid;
        coord.x = x;
        coord.y = y;

        SubdivCCGNeighbors neighbors;
        BKE_subdiv_ccg_neighbor_coords_get(subdiv_ccg, coord, false, neighbors);

        if (boundary::vert_is_boundary(
                faces, corner_verts, boundary_verts, boundary_edges, subdiv_ccg, coord))
        {
          if (neighbors.coords.size() == 2) {
            /* Do not include neighbors of corner vertices. */
            neighbors.coords.clear();
          }
          else {
            /* Only include other boundary vertices as neighbors of boundary vertices. */
            neighbors.coords.remove_if([&](const SubdivCCGCoord coord) {
              return !boundary::vert_is_boundary(
                  faces, corner_verts, boundary_verts, boundary_edges, subdiv_ccg, coord);
            });
          }
        }

        if (neighbors.coords.is_empty()) {
          new_positions[node_vert_index] = positions[vert];
        }
        else {
          new_positions[node_vert_index] = average_positions(key, positions, neighbors.coords);
        }
      }
    }
  }
}

void neighbor_position_average_interior_grids(const OffsetIndices<int> faces,
                                              const Span<int> corner_verts,
                                              const BitSpan boundary_verts,
                                              const Set<OrderedEdge> &boundary_edges,
                                              const SubdivCCG &subdiv_ccg,
                                              const Span<int> grids,
                                              const MutableSpan<float3> new_positions)
{
  neighbor_position_average_interior_grids_impl<false>(
      faces, corner_verts, boundary_verts, boundary_edges, subdiv_ccg, grids, {}, new_positions);
}

void neighbor_position_average_interior_grids(const OffsetIndices<int> faces,
                                              const Span<int> corner_verts,
                                              const BitSpan boundary_verts,
                                              const Set<OrderedEdge> &boundary_edges,
                                              const SubdivCCG &subdiv_ccg,
                                              const Span<int> grids,
                                              const Span<float> factors,
                                              const MutableSpan<float3> new_positions)
{
  neighbor_position_average_interior_grids_impl<true>(faces,
                                                      corner_verts,
                                                      boundary_verts,
                                                      boundary_edges,
                                                      subdiv_ccg,
                                                      grids,
                                                      factors,
                                                      new_positions);
}

template<typename T>
void average_data_grids(const SubdivCCG &subdiv_ccg,
                        const Span<T> src,
                        const Span<int> grids,
                        const MutableSpan<T> dst)
{
  PRF_scope(ProfileCategory::Editor);
  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);

  BLI_assert(grids.size() * key.grid_area == dst.size());

  for (const int i : grids.index_range()) {
    const int grid = grids[i];
    const int node_verts_start = i * key.grid_area;

    /* TODO: This loop could be optimized in the future by skipping unnecessary logic for
     * non-boundary grid vertices. */
    for (const int y : IndexRange(key.grid_size)) {
      for (const int x : IndexRange(key.grid_size)) {
        const int offset = CCG_grid_xy_to_index(key.grid_size, x, y);
        const int node_vert_index = node_verts_start + offset;

        SubdivCCGCoord coord{};
        coord.grid_index = grid;
        coord.x = x;
        coord.y = y;

        SubdivCCGNeighbors neighbors;
        BKE_subdiv_ccg_neighbor_coords_get(subdiv_ccg, coord, false, neighbors);

        T sum{};
        for (const SubdivCCGCoord neighbor : neighbors.coords) {
          sum += src[neighbor.to_index(key)];
        }
        dst[node_vert_index] = math::safe_divide(sum, float(neighbors.coords.size()));
      }
    }
  }
}

template<typename T>
void average_data_bmesh(const Span<T> src, const Set<BMVert *, 0> &verts, const MutableSpan<T> dst)
{
  PRF_scope(ProfileCategory::Editor);
  BMeshNeighborVerts neighbor_data;

  int i = 0;
  for (BMVert *vert : verts) {
    T sum{};
    const Span<BMVert *> neighbors = vert_neighbors_get_bmesh(*vert, neighbor_data);
    for (const BMVert *neighbor : neighbors) {
      sum += src[BM_elem_index_get(neighbor)];
    }
    dst[i] = math::safe_divide(sum, float(neighbors.size()));
    i++;
  }
}

template void average_data_grids<float>(const SubdivCCG &,
                                        Span<float>,
                                        Span<int>,
                                        MutableSpan<float>);
template void average_data_grids<float3>(const SubdivCCG &,
                                         Span<float3>,
                                         Span<int>,
                                         MutableSpan<float3>);
template void average_data_bmesh<float>(Span<float> src,
                                        const Set<BMVert *, 0> &,
                                        MutableSpan<float>);
template void average_data_bmesh<float3>(Span<float3> src,
                                         const Set<BMVert *, 0> &,
                                         MutableSpan<float3>);

static float3 average_positions(const Span<const BMVert *> verts)
{
  PRF_scope(ProfileCategory::Editor);
  const float factor = math::safe_rcp(float(verts.size()));
  float3 result(0);
  for (const BMVert *vert : verts) {
    result += float3(vert->co) * factor;
  }
  return result;
}

void neighbor_position_average_bmesh(const Set<BMVert *, 0> &verts,
                                     const MutableSpan<float3> new_positions)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(verts.size() == new_positions.size());
  BMeshNeighborVerts neighbor_data;

  int i = 0;
  for (BMVert *vert : verts) {
    const Span<BMVert *> neighbors = vert_neighbors_get_bmesh(*vert, neighbor_data);
    new_positions[i] = average_positions(neighbors);
    i++;
  }
}

template<bool use_factors>
static void neighbor_position_average_interior_bmesh_impl(const Set<BMVert *, 0> &verts,
                                                          const Span<float> factors,
                                                          const MutableSpan<float3> new_positions)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(verts.size() == new_positions.size());
  if constexpr (use_factors) {
    BLI_assert(new_positions.size() == factors.size());
  }
  BMeshNeighborVerts neighbor_data;

  int i = 0;
  for (BMVert *vert : verts) {
    if constexpr (use_factors) {
      if (factors[i] == 0.0f) {
        new_positions[i] = float3(vert->co);
        i++;
        continue;
      }
    }

    const Span<BMVert *> neighbors = vert_neighbors_get_interior_bmesh(*vert, neighbor_data);
    if (neighbors.is_empty()) {
      new_positions[i] = float3(vert->co);
    }
    else {
      new_positions[i] = average_positions(neighbors);
    }
    i++;
  }
}
void neighbor_position_average_interior_bmesh(const Set<BMVert *, 0> &verts,
                                              const Span<float> factors,
                                              const MutableSpan<float3> new_positions)
{
  neighbor_position_average_interior_bmesh_impl<true>(verts, factors, new_positions);
}

void neighbor_position_average_interior_bmesh(const Set<BMVert *, 0> &verts,
                                              const MutableSpan<float3> new_positions)
{
  neighbor_position_average_interior_bmesh_impl<false>(verts, {}, new_positions);
}

void bmesh_four_neighbor_average(float avg[3], const float3 &direction, const BMVert *v)
{
  float avg_co[3] = {0.0f, 0.0f, 0.0f};
  float tot_co = 0.0f;

  BMIter eiter;
  BMEdge *e;

  BM_ITER_ELEM (e, &eiter, const_cast<BMVert *>(v), BM_EDGES_OF_VERT) {
    if (BM_edge_is_boundary(e)) {
      copy_v3_v3(avg, v->co);
      return;
    }
    BMVert *v_other = (e->v1 == v) ? e->v2 : e->v1;
    float vec[3];
    sub_v3_v3v3(vec, v_other->co, v->co);
    madd_v3_v3fl(vec, v->no, -dot_v3v3(vec, v->no));
    normalize_v3(vec);

    /* fac is a measure of how orthogonal or parallel the edge is
     * relative to the direction. */
    float fac = dot_v3v3(vec, direction);
    fac = fac * fac - 0.5f;
    fac *= fac;
    madd_v3_v3fl(avg_co, v_other->co, fac);
    tot_co += fac;
  }

  /* In case vert has no Edge s. */
  if (tot_co > 0.0f) {
    mul_v3_v3fl(avg, avg_co, 1.0f / tot_co);

    /* Preserve volume. */
    float vec[3];
    sub_v3_v3(avg, v->co);
    mul_v3_v3fl(vec, v->no, dot_v3v3(avg, v->no));
    sub_v3_v3(avg, vec);
    add_v3_v3(avg, v->co);
  }
  else {
    zero_v3(avg);
  }
}

void neighbor_color_average(const OffsetIndices<int> faces,
                            const Span<int> corner_verts,
                            const GroupedSpan<int> vert_to_face_map,
                            const GSpan color_attribute,
                            const bke::AttrDomain color_domain,
                            const GroupedSpan<int> vert_neighbors,
                            const MutableSpan<float4> smooth_colors)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(vert_neighbors.size() == smooth_colors.size());

  for (const int i : vert_neighbors.index_range()) {
    float4 sum(0);
    const Span<int> neighbors = vert_neighbors[i];
    for (const int vert : neighbors) {
      sum += color::color_vert_get(
          faces, corner_verts, vert_to_face_map, color_attribute, color_domain, vert);
    }
    smooth_colors[i] = math::safe_divide(sum, float(neighbors.size()));
  }
}

/* HC Smooth Algorithm. */
/* From: Improved Laplacian Smoothing of Noisy Surface Meshes */

void surface_smooth_laplacian_step(const Span<float3> positions,
                                   const Span<float3> orig_positions,
                                   const Span<float3> average_positions,
                                   const float alpha,
                                   MutableSpan<float3> laplacian_disp,
                                   MutableSpan<float3> translations)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(positions.size() == orig_positions.size());
  BLI_assert(positions.size() == average_positions.size());
  BLI_assert(positions.size() == laplacian_disp.size());
  BLI_assert(positions.size() == translations.size());

  for (const int i : average_positions.index_range()) {
    const float3 weighted_o = orig_positions[i] * alpha;
    const float3 weighted_q = positions[i] * (1.0f - alpha);
    const float3 d = weighted_o + weighted_q;
    laplacian_disp[i] = average_positions[i] - d;
    translations[i] = average_positions[i] - positions[i];
  }
}

void surface_smooth_displace_step(const Span<float3> laplacian_disp,
                                  const Span<float3> average_laplacian_disp,
                                  const float beta,
                                  const MutableSpan<float3> translations)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(laplacian_disp.size() == average_laplacian_disp.size());
  BLI_assert(laplacian_disp.size() == translations.size());

  for (const int i : laplacian_disp.index_range()) {
    float3 b_current_vert = average_laplacian_disp[i] * (1.0f - beta);
    b_current_vert += laplacian_disp[i] * beta;
    translations[i] = -b_current_vert;
  }
}

static float3 translation_to_plane(const float3 &current_position,
                                   const float3 &normal,
                                   const float3 &smoothed_position)
{
  float4 plane;
  plane_from_point_normal_v3(plane, current_position, normal);

  float3 smooth_closest_plane;
  closest_to_plane_v3(smooth_closest_plane, plane, smoothed_position);

  return smooth_closest_plane - current_position;
}

static float3 calc_boundary_normal_corner(const float3 &current_position,
                                          const Span<float3> vert_positions,
                                          const Span<int> neighbors)
{
  PRF_scope(ProfileCategory::Editor);
  float3 normal(0);
  for (const int vert : neighbors) {
    const float3 to_neighbor = vert_positions[vert] - current_position;
    normal += math::normalize(to_neighbor);
  }
  return math::normalize(normal);
}

static float3 calc_boundary_normal_corner(const CCGKey &key,
                                          const Span<float3> positions,
                                          const float3 &current_position,
                                          const Span<SubdivCCGCoord> neighbors)
{
  PRF_scope(ProfileCategory::Editor);
  float3 normal(0);
  for (const SubdivCCGCoord &coord : neighbors) {
    const float3 to_neighbor = positions[coord.to_index(key)] - current_position;
    normal += math::normalize(to_neighbor);
  }
  return math::normalize(normal);
}

static float3 calc_boundary_normal_corner(const float3 &current_position,
                                          const Span<BMVert *> neighbors)
{
  PRF_scope(ProfileCategory::Editor);
  float3 normal(0);
  for (BMVert *vert : neighbors) {
    const float3 neighbor_pos = vert->co;
    const float3 to_neighbor = neighbor_pos - current_position;
    normal += math::normalize(to_neighbor);
  }
  return math::normalize(normal);
}

void calc_relaxed_translations_faces(const Span<float3> vert_positions,
                                     const Span<float3> vert_normals,
                                     const OffsetIndices<int> faces,
                                     const Span<int> corner_verts,
                                     const GroupedSpan<int> vert_to_face_map,
                                     const BitSpan boundary_verts,
                                     const Set<OrderedEdge> &boundary_edges,
                                     const Span<int> face_sets,
                                     const Span<bool> hide_poly,
                                     const bool filter_boundary_face_sets,
                                     const Span<int> verts,
                                     const Span<float> factors,
                                     const MutableSpan<float3> translations)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(verts.size() == factors.size());
  BLI_assert(verts.size() == translations.size());

  Vector<int> neighbors;

  for (const int i : verts.index_range()) {
    if (factors[i] == 0.0f) {
      translations[i] = float3(0);
      continue;
    }

    vert_neighbors_get_mesh(faces, corner_verts, vert_to_face_map, hide_poly, verts[i], neighbors);

    /* Don't modify corner vertices */
    if (neighbors.size() <= 2) {
      translations[i] = float3(0);
      continue;
    }

    const bool is_boundary = boundary_verts[verts[i]];
    if (is_boundary) {
      neighbors.remove_if(
          [&](const int vert) { return !boundary_edges.contains(OrderedEdge(vert, verts[i])); });
    }

    if (filter_boundary_face_sets) {
      neighbors.remove_if([&](const int vert) {
        return face_set::vert_has_unique_face_set(vert_to_face_map, face_sets, vert);
      });
    }

    if (neighbors.is_empty()) {
      translations[i] = float3(0);
      continue;
    }

    const float3 smoothed_position = bke::attribute_math::mix_indices(vert_positions, neighbors);

    /* Normal Calculation */
    float3 normal;
    if (is_boundary && neighbors.size() == 2) {
      normal = calc_boundary_normal_corner(vert_positions[verts[i]], vert_positions, neighbors);
      if (math::is_zero(normal)) {
        translations[i] = float3(0);
        continue;
      }
    }
    else {
      normal = vert_normals[verts[i]];
    }

    const float3 translation = translation_to_plane(
        vert_positions[verts[i]], normal, smoothed_position);

    translations[i] = translation * factors[i];
  }
}

void calc_relaxed_translations_multires(const Object &object,
                                        const Multiresolution &multires,
                                        const bke::multiresolution::BaseTopology &base_topo,
                                        const bke::pbvh::MultiresNode &node,
                                        const Span<int> face_sets,
                                        const bool filter_boundary_face_sets,
                                        const Span<float> factors,
                                        const MutableSpan<float3> translations)
{
  using MultiresCoord = bke::multiresolution::MultiresCoord;
  using ElemType = bke::multiresolution::ElemType;

  const bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  const int grid_size = multires.grid_size();
  const IndexRange node_range = node.range(grid_size);
  const Span<float3> positions = multires.positions();
  const Span<float3> node_positions = positions.slice(node_range);
  const Span<float3> node_normals = multires.normals().slice(node_range);
  const Span<bool> hidden = multires.hidden_verts();

  auto average_positions = [&](const Span<MultiresCoord> coords) -> float3 {
    const float factor = math::rcp(float(coords.size()));
    float3 result(0);
    for (const MultiresCoord coord : coords) {
      const int idx = coord.to_index(pbvh.offsets);
      result += positions[idx] * factor;
    }
    return result;
  };

  auto calc_boundary_normal_corner = [&](const float3 &current_position,
                                         const Span<MultiresCoord> neighbors) -> float3 {
    float3 normal(0);
    for (const MultiresCoord coord : neighbors) {
      const int idx = coord.to_index(pbvh.offsets);
      const float3 to_neighbor = positions[idx] - current_position;
      normal += math::normalize(to_neighbor);
    }
    return math::normalize(normal);
  };

  node.foreach_coord(grid_size, [&](const int node_vert, const MultiresCoord coord) {
    if (factors[node_vert] == 0.0f) {
      translations[node_vert] = float3(0);
      return;
    }

    Vector<MultiresCoord> neighbors;
    Multiresolution::calc_neighbors(coord, base_topo, grid_size, pbvh.offsets, hidden, neighbors);

    if (neighbors.size() <= 2) {
      translations[node_vert] = float3(0);
      return;
    }

    const bool is_boundary = boundary::vert_is_boundary(
        coord, base_topo.boundary_verts, base_topo.boundary_edges);

    if (is_boundary) {
      neighbors.remove_if([&](const MultiresCoord neighbor) {
        return !boundary::vert_is_boundary(
            neighbor, base_topo.boundary_verts, base_topo.boundary_edges);
      });
    }

    if (filter_boundary_face_sets) {
      neighbors.remove_if([&](const MultiresCoord neighbor) {
        return face_set::vert_has_unique_face_set(neighbor, base_topo, grid_size, face_sets);
      });
    }

    if (neighbors.is_empty()) {
      translations[node_vert] = float3(0);
      return;
    }

    const float3 smoothed_position = average_positions(neighbors);

    float3 normal;
    if (is_boundary && neighbors.size() == 2) {
      normal = calc_boundary_normal_corner(node_positions[node_vert], neighbors);
      if (math::is_zero(normal)) {
        translations[node_vert] = float3(0);
        return;
      }
    }
    else {
      normal = node_normals[node_vert];
    }

    const float3 translation = translation_to_plane(
        node_positions[node_vert], normal, smoothed_position);

    translations[node_vert] = translation * factors[node_vert];
  });
}

void calc_relaxed_translations_grids(const SubdivCCG &subdiv_ccg,
                                     const OffsetIndices<int> faces,
                                     const Span<int> corner_verts,
                                     const Span<int> face_sets,
                                     const GroupedSpan<int> vert_to_face_map,
                                     const BitSpan boundary_verts,
                                     const Set<OrderedEdge> &boundary_edges,
                                     const Span<int> grids,
                                     const bool filter_boundary_face_sets,
                                     const Span<float> factors,
                                     const MutableSpan<float3> translations)
{
  PRF_scope(ProfileCategory::Editor);
  const Span<float3> positions = subdiv_ccg.positions;
  const Span<float3> normals = subdiv_ccg.normals;
  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);

  for (const int i : grids.index_range()) {
    const IndexRange grid_range = bke::ccg::grid_range(key, grids[i]);
    const int node_start = i * key.grid_area;
    for (const int y : IndexRange(key.grid_size)) {
      for (const int x : IndexRange(key.grid_size)) {
        const int offset = CCG_grid_xy_to_index(key.grid_size, x, y);
        const int node_vert = node_start + offset;
        const int vert = grid_range[offset];
        if (factors[node_vert] == 0.0f) {
          translations[node_vert] = float3(0);
          continue;
        }

        SubdivCCGCoord coord{};
        coord.grid_index = grids[i];
        coord.x = x;
        coord.y = y;

        SubdivCCGNeighbors neighbor_storage;
        BKE_subdiv_ccg_neighbor_coords_get(subdiv_ccg, coord, false, neighbor_storage);
        SubdivCCGNeighborCoords &neighbors = neighbor_storage.coords;

        /* Don't modify corner vertices */
        if (neighbors.size() <= 2) {
          translations[node_vert] = float3(0);
          continue;
        }

        const bool is_boundary = boundary::vert_is_boundary(
            faces, corner_verts, boundary_verts, boundary_edges, subdiv_ccg, coord);

        if (is_boundary) {
          neighbors.remove_if([&](const SubdivCCGCoord neighbor) {
            return !boundary::vert_is_boundary(
                faces, corner_verts, boundary_verts, boundary_edges, subdiv_ccg, neighbor);
          });
        }

        if (filter_boundary_face_sets) {
          neighbors.remove_if([&](const SubdivCCGCoord neighbor) {
            return face_set::vert_has_unique_face_set(
                faces, corner_verts, vert_to_face_map, face_sets, subdiv_ccg, neighbor);
          });
        }

        if (neighbors.is_empty()) {
          translations[node_vert] = float3(0);
          continue;
        }

        const float3 smoothed_position = average_positions(key, positions, neighbors);

        /* Normal Calculation */
        float3 normal;
        if (is_boundary && neighbors.size() == 2) {
          normal = calc_boundary_normal_corner(key, positions, positions[vert], neighbors);
          if (math::is_zero(normal)) {
            translations[node_vert] = float3(0);
            continue;
          }
        }
        else {
          normal = normals[vert];
        }

        const float3 translation = translation_to_plane(
            positions[vert], normal, smoothed_position);

        translations[node_vert] = translation * factors[node_vert];
      }
    }
  }
}

void calc_relaxed_translations_bmesh(const Set<BMVert *, 0> &verts,
                                     const Span<float3> positions,
                                     const int face_set_offset,
                                     const bool filter_boundary_face_sets,
                                     const Span<float> factors,
                                     const MutableSpan<float3> translations)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(verts.size() == factors.size());
  BLI_assert(verts.size() == translations.size());

  BMeshNeighborVerts neighbors;

  int i = 0;
  for (BMVert *vert : verts) {
    if (factors[i] == 0.0f) {
      translations[i] = float3(0);
      i++;
      continue;
    }

    vert_neighbors_get_bmesh(*vert, neighbors);

    /* Don't modify corner vertices */
    if (neighbors.size() <= 2) {
      translations[i] = float3(0);
      i++;
      continue;
    }

    const bool is_boundary = BM_vert_is_boundary(vert);
    if (is_boundary) {
      neighbors.remove_if([&](const BMVert *vert) { return !BM_vert_is_boundary(vert); });
    }

    if (filter_boundary_face_sets) {
      neighbors.remove_if([&](const BMVert *vert) {
        return face_set::vert_has_unique_face_set(face_set_offset, *vert);
      });
    }

    if (neighbors.is_empty()) {
      translations[i] = float3(0);
      i++;
      continue;
    }

    const float3 smoothed_position = average_positions(neighbors);

    /* Normal Calculation */
    float3 normal;
    if (is_boundary && neighbors.size() == 2) {
      normal = calc_boundary_normal_corner(positions[i], neighbors);
      if (math::is_zero(normal)) {
        translations[i] = float3(0);
        i++;
        continue;
      }
    }
    else {
      normal = vert->no;
    }

    const float3 translation = translation_to_plane(positions[i], normal, smoothed_position);

    translations[i] = translation * factors[i];
    i++;
  }
}

void blur_geometry_data_array(const Object &object,
                              const int iterations,
                              const MutableSpan<float> data)
{
  PRF_scope(ProfileCategory::Editor);
  struct LocalData {
    Vector<int> vert_indices;
    Vector<int> neighbor_offsets;
    Vector<int> neighbor_data;
    Vector<float> new_factors;
  };
  const SculptSession &ss = *object.runtime->sculpt_session;
  const bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);

  threading::EnumerableThreadSpecific<LocalData> all_tls;
  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh: {
      const Span<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();

      const Mesh &mesh = *id_cast<const Mesh *>(object.data);
      const OffsetIndices faces = mesh.faces();
      const Span<int> corner_verts = mesh.corner_verts();
      const GroupedSpan<int> vert_to_face_map = mesh.vert_to_face_map();
      const bke::AttributeAccessor attributes = mesh.attributes();
      const VArraySpan<bool> hide_vert = *attributes.lookup<bool>(".hide_vert",
                                                                  bke::AttrDomain::Point);
      const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly",
                                                                  bke::AttrDomain::Face);
      for ([[maybe_unused]] const int _ : IndexRange(iterations)) {
        node_mask.foreach_index(
            [&](const int i) {
              LocalData &tls = all_tls.local();
              Vector<int, bke::pbvh::MESH_LEAF_LIMIT> index_data;
              const Span<int> verts = hide::node_visible_verts(nodes[i], hide_vert, index_data);

              const GroupedSpan<int> neighbors = calc_vert_neighbors(faces,
                                                                     corner_verts,
                                                                     vert_to_face_map,
                                                                     hide_poly,
                                                                     verts,
                                                                     tls.neighbor_offsets,
                                                                     tls.neighbor_data);

              Array<float, bke::pbvh::MESH_LEAF_LIMIT> new_factors(verts.size());
              smooth::neighbor_data_average_mesh(
                  data.as_span(), neighbors, new_factors.as_mutable_span());

              scatter_data_mesh(new_factors.as_span(), verts, data);
            },
            exec_mode::grain_size(1));
      }
      break;
    }
    case bke::pbvh::Type::Multires: {
      const Mesh &mesh = *id_cast<const Mesh *>(object.data);
      const Multiresolution &multires = *mesh.multires;
      const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(mesh);
      const int grid_size = multires.grid_size();
      const Span<bool> hidden = multires.hidden_verts();
      const Span<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
      for ([[maybe_unused]] const int _ : IndexRange(iterations)) {
        node_mask.foreach_index(
            [&](const int node_index) {
              const IndexRange node_range = nodes[node_index].range(grid_size);
              Array<float, bke::pbvh::MULTIRES_LEAF_LIMIT> new_factors(node_range.size());

              smooth::neighbor_data_average_node_multires(multires,
                                                          base_topo,
                                                          nodes[node_index],
                                                          pbvh.offsets,
                                                          data.as_span(),
                                                          new_factors.as_mutable_span(),
                                                          hidden);

              if (hidden.is_empty()) {
                data.slice(node_range).copy_from(new_factors);
              }
              else {
                const int node_offset = nodes[node_index].offset();
                for (const int vert : node_range) {
                  if (!hidden[vert]) {
                    data[vert] = new_factors[vert - node_offset];
                  }
                }
              }
            },
            exec_mode::grain_size(1));
      }
      break;
    }
    case bke::pbvh::Type::Grids: {
      const Span<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
      const SubdivCCG &subdiv_ccg = *ss.subdiv_ccg;
      const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);
      const BitGroupVector<> &grid_hidden = subdiv_ccg.grid_hidden;
      for ([[maybe_unused]] const int _ : IndexRange(iterations)) {
        node_mask.foreach_index(
            [&](const int node_index) {
              LocalData &tls = all_tls.local();
              const Span<int> grids = nodes[node_index].grids();
              const int grid_verts_num = key.grid_area * grids.size();

              tls.new_factors.resize(grid_verts_num);
              const MutableSpan<float> new_factors = tls.new_factors;
              smooth::average_data_grids(subdiv_ccg, data.as_span(), grids, new_factors);

              if (grid_hidden.is_empty()) {
                scatter_data_grids(subdiv_ccg, new_factors.as_span(), grids, data);
              }
              else {
                for (const int i : grids.index_range()) {
                  const int node_start = i * key.grid_area;
                  BKE_subdiv_ccg_foreach_visible_grid_vert(
                      key, grid_hidden, grids[i], [&](const int offset) {
                        data[i] = new_factors[node_start + offset];
                      });
                }
              }
            },
            exec_mode::grain_size(1));
      }
      break;
    }
    case bke::pbvh::Type::BMesh: {
      const Span<bke::pbvh::BMeshNode> nodes = pbvh.nodes<bke::pbvh::BMeshNode>();
      for ([[maybe_unused]] const int _ : IndexRange(iterations)) {
        node_mask.foreach_index(
            [&](const int node_index) {
              LocalData &tls = all_tls.local();
              const Set<BMVert *, 0> &verts = BKE_pbvh_bmesh_node_unique_verts(
                  const_cast<bke::pbvh::BMeshNode *>(&nodes[node_index]));

              tls.new_factors.resize(verts.size());
              const MutableSpan<float> new_factors = tls.new_factors;
              smooth::average_data_bmesh(data.as_span(), verts, new_factors);

              int i = 0;
              for (const BMVert *vert : verts) {
                if (BM_elem_flag_test(vert, BM_ELEM_HIDDEN)) {
                  i++;
                  continue;
                }
                data[BM_elem_index_get(vert)] = new_factors[i];
                i++;
              }
            },
            exec_mode::grain_size(1));
      }
      break;
    }
  }
}

}  // namespace blender::ed::sculpt_paint::smooth
