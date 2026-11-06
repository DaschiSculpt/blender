/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "DNA_mesh_types.h"
#include "DNA_multires_types.h"

#include <algorithm>
#include <array>
#include <xxhash.h>

#include "BLI_color_types.hh"
#include "BLI_math_vector.hh"
#include "BLI_offset_indices.hh"
#include "BLI_task.hh"

#include "BKE_attribute.hh"
#include "BKE_attribute_storage.hh"
#include "BKE_lib_id.hh"
#include "BKE_multiresolution.hh"
#include "BKE_sculpt_layer.hh"

#include "MEM_guardedalloc.h"

#include "attribute_storage_access.hh"

#include "multiresolution.hh"

namespace blender {

using Offsets = bke::multiresolution::Offsets;
using OffsetsSpan = bke::multiresolution::OffsetsSpan;
using MultiresCoord = bke::multiresolution::MultiresCoord;
using ElemType = bke::multiresolution::ElemType;
using BaseTopology = bke::multiresolution::BaseTopology;

using bke::multiresolution::extract_grid_inner_border;
using bke::multiresolution::fill_border;
using bke::multiresolution::run_row;

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

static bool all_faces_quads(const Mesh &base_mesh)
{
  const OffsetIndices<int> faces = base_mesh.faces();
  for (const int face_idx : IndexRange(base_mesh.faces_num)) {
    if (faces[face_idx].size() != 4) {
      return false;
    }
  }
  return true;
}

uint64_t calc_base_topology_hash(const Mesh &base_mesh)
{
  XXH3_state_t *state = XXH3_createState();
  XXH3_64bits_reset(state);

  const int verts_num = base_mesh.verts_num;
  const int edges_num = base_mesh.edges_num;
  const int faces_num = base_mesh.faces_num;
  const int corners_num = base_mesh.corners_num;
  XXH3_64bits_update(state, &verts_num, sizeof(verts_num));
  XXH3_64bits_update(state, &edges_num, sizeof(edges_num));
  XXH3_64bits_update(state, &faces_num, sizeof(faces_num));
  XXH3_64bits_update(state, &corners_num, sizeof(corners_num));

  const Span<int> face_offsets = base_mesh.face_offsets();
  if (!face_offsets.is_empty()) {
    XXH3_64bits_update(state, face_offsets.data(), face_offsets.size_in_bytes());
  }
  const Span<int> corner_verts = base_mesh.corner_verts();
  if (!corner_verts.is_empty()) {
    XXH3_64bits_update(state, corner_verts.data(), corner_verts.size_in_bytes());
  }
  const Span<int> corner_edges = base_mesh.corner_edges();
  if (!corner_edges.is_empty()) {
    XXH3_64bits_update(state, corner_edges.data(), corner_edges.size_in_bytes());
  }
  const Span<int2> edges = base_mesh.edges();
  if (!edges.is_empty()) {
    XXH3_64bits_update(state, edges.data(), edges.size_in_bytes());
  }

  const uint64_t hash = XXH3_64bits_digest(state);
  XXH3_freeState(state);
  return hash;
}

Multiresolution *BKE_multiresolution_create(const Mesh &base_mesh)
{
  Multiresolution *multires = MEM_new<Multiresolution>(__func__);

  multires->top_level = 0;
  multires->curr_level = 0;
  multires->sculpt_level = 0;
  multires->view_level = 0;
  multires->render_level = 0;

  multires->bvh_partitions_len = 0;
  multires->bvh_partitions = nullptr;

  new (&multires->top_attribute_storage.wrap()) bke::AttributeStorage();
  new (&multires->curr_attribute_storage.wrap()) bke::AttributeStorage();

  multires->runtime = MEM_new<bke::multiresolution::MultiresolutionRuntime>(__func__, base_mesh);

  multires->base_verts_num = base_mesh.verts_num;
  multires->base_edges_num = base_mesh.edges_num;
  multires->base_faces_num = base_mesh.faces_num;
  multires->base_corners_num = base_mesh.corners_num;
  multires->base_all_quads = all_faces_quads(base_mesh);
  multires->base_topology_hash = calc_base_topology_hash(base_mesh);

  return multires;
}

namespace bke {

namespace multiresolution {

static void make_linear_policies(const int lvl,
                                 const bke::multiresolution::BaseTopology &base_topo,
                                 LinearOffset &vert_policy,
                                 LinearOffset &edge_policy,
                                 LinearOffset &face_policy)
{
  const int grid_size = Multiresolution::grid_size(lvl);
  vert_policy = {0, 1};
  edge_policy = {base_topo.verts_num, grid_size};
  face_policy = {base_topo.verts_num + base_topo.edges_num * grid_size, grid_size * grid_size};
};

int MultiresCoord::to_index(const OffsetsSpan &offsets) const
{
  switch (type) {
    case ElemType::Vert:
      return offsets.verts[base_index];
    case ElemType::Edge:
      return offsets.edges[base_index] + local_coord;
    case ElemType::Face:
      return offsets.faces[base_index] + local_coord;
  }
  BLI_assert_unreachable();
  return -1;
}

template<typename OutputPolicy>
static void calc_corner_verts(const BaseTopology &base_topo,
                              const int virtual_lvl,
                              const OutputPolicy vert_offsets,
                              const OutputPolicy edge_offsets,
                              const OutputPolicy face_offsets,
                              const MutableSpan<int> r_corner_verts)
{
  const int faces_num = base_topo.faces_num;
  const OffsetIndices<int> base_faces = base_topo.faces;
  const Span<int> base_corner_verts = base_topo.corner_verts;
  const Span<int> base_corner_edges = base_topo.corner_edges;

  if (virtual_lvl == 1) {
    threading::parallel_for(IndexRange(faces_num), 4096, [&](const IndexRange range) {
      for (const int face_idx : range) {
        const IndexRange face_range = base_faces[face_idx];
        const int face_start = face_range.start();
        const int corners_num = face_range.size();
        const int face_center_vert = face_offsets[face_idx];

        int *out = &r_corner_verts[face_start * 4];

        if (corners_num == 4) {
          const int v0 = vert_offsets[base_corner_verts[face_start + 0]];
          const int v1 = vert_offsets[base_corner_verts[face_start + 1]];
          const int v2 = vert_offsets[base_corner_verts[face_start + 2]];
          const int v3 = vert_offsets[base_corner_verts[face_start + 3]];
          const int m0 = edge_offsets[base_corner_edges[face_start + 0]];
          const int m1 = edge_offsets[base_corner_edges[face_start + 1]];
          const int m2 = edge_offsets[base_corner_edges[face_start + 2]];
          const int m3 = edge_offsets[base_corner_edges[face_start + 3]];

          out[0] = v0;
          out[1] = m0;
          out[2] = face_center_vert;
          out[3] = m3;
          out[4] = m0;
          out[5] = v1;
          out[6] = m1;
          out[7] = face_center_vert;
          out[8] = m3;
          out[9] = face_center_vert;
          out[10] = m2;
          out[11] = v3;
          out[12] = face_center_vert;
          out[13] = m1;
          out[14] = v2;
          out[15] = m2;
        }
        else {
          for (int k = 0; k < corners_num; k++) {
            const int prev_k = (k + corners_num - 1) % corners_num;

            const int vert_corner = vert_offsets[base_corner_verts[face_start + k]];
            const int vert_edge_next = edge_offsets[base_corner_edges[face_start + k]];
            const int vert_edge_prev = edge_offsets[base_corner_edges[face_start + prev_k]];

            out[k * 4 + 0] = vert_corner;
            out[k * 4 + 1] = vert_edge_next;
            out[k * 4 + 2] = face_center_vert;
            out[k * 4 + 3] = vert_edge_prev;
          }
        }
      }
    });
    return;
  }
  const int size = Multiresolution::grid_size(virtual_lvl);
  const int faces_per_side = size + 1;
  const int faces_per_grid = faces_per_side * faces_per_side;
  const BitSpan face_edge_reversed = base_topo.face_edge_reversed;

  const int grain_size = std::max(1, 8192 / (size * size));

  threading::parallel_for(IndexRange(faces_num), grain_size, [&](const IndexRange range) {
    for (const int face_idx : range) {
      const IndexRange face_range = base_faces[face_idx];
      const Span<int> face_verts = base_corner_verts.slice(face_range);
      const Span<int> face_edges = base_corner_edges.slice(face_range);
      const int face_offset = face_offsets[face_idx];

      std::array<int, 4> border_offsets = {edge_offsets[face_edges[0]],
                                           edge_offsets[face_edges[1]],
                                           edge_offsets[face_edges[2]],
                                           edge_offsets[face_edges[3]]};

      std::array<int, 4> border_start;
      std::array<int, 4> border_step;

      for (int i = 0; i < 4; i++) {
        int reverse = (int)face_edge_reversed[face_range.start() + i] ^ (i >> 1);
        border_step[i] = 1 - 2 * reverse;
        border_start[i] = border_offsets[i] + (reverse * (size - 1));
      }

      int *out = &r_corner_verts[face_idx * faces_per_grid * 4];

      {
        out[0] = vert_offsets[face_verts[0]];
        out[1] = border_start[0];
        out[2] = face_offset;
        out[3] = border_start[3];
        out += 4;

        border_start[0] += border_step[0];

        for (int i = 1; i < faces_per_side - 1; i++) {
          out[0] = border_start[0] - border_step[0];
          out[1] = border_start[0];
          out[2] = face_offset + i;
          out[3] = face_offset + i - 1;
          out += 4;
          border_start[0] += border_step[0];
        }

        out[0] = border_start[0] - border_step[0];
        out[1] = vert_offsets[face_verts[1]];
        out[2] = border_start[1];
        out[3] = face_offset + size - 1;
        out += 4;
        border_start[1] += border_step[1];
        border_start[3] += border_step[3];
      }

      int row_curr = face_offset;
      int row_next = face_offset + size;

      for (int row = 1; row < faces_per_side - 1; row++) {
        out[0] = border_start[3] - border_step[3];
        out[1] = row_curr;
        out[2] = row_next;
        out[3] = border_start[3];
        out += 4;

        for (int col = 1; col < faces_per_side - 1; col++) {
          out[0] = row_curr + col - 1;
          out[1] = row_curr + col;
          out[2] = row_next + col;
          out[3] = row_next + col - 1;
          out += 4;
        }

        out[0] = row_curr + size - 1;
        out[1] = border_start[1] - border_step[1];
        out[2] = border_start[1];
        out[3] = row_next + size - 1;
        out += 4;

        border_start[1] += border_step[1];
        border_start[3] += border_step[3];
        row_curr += size;
        row_next += size;
      }

      {
        out[0] = border_start[3] - border_step[3];
        out[1] = row_curr;
        out[2] = border_start[2];
        out[3] = vert_offsets[face_verts[3]];
        out += 4;

        int border_top = border_start[2];

        for (int col = 1; col < faces_per_side - 1; col++) {
          out[0] = row_curr + col - 1;
          out[1] = row_curr + col;
          out[2] = border_top + border_step[2];
          out[3] = border_top;
          out += 4;
          border_top += border_step[2];
        }

        out[0] = row_curr + size - 1;
        out[1] = border_start[1] - border_step[1];
        out[2] = vert_offsets[face_verts[2]];
        out[3] = border_top;
      }
    }
  });
}

template<typename OutputPolicy>
static void calc_edges(const BaseTopology &base_topo,
                       const int virtual_lvl,
                       const OutputPolicy vert_offsets,
                       const OutputPolicy edge_offsets,
                       const OutputPolicy face_offsets,
                       const MutableSpan<int2> r_edges)
{
  const int size = Multiresolution::grid_size(virtual_lvl);
  const int faces_per_side = size + 1;
  const int edges_num = base_topo.edges_num;
  const int faces_num = base_topo.faces_num;
  const Span<int2> base_edges = base_topo.edges;

  const int edge_grain_size = std::max(1, 4096 / size);

  threading::parallel_for(IndexRange(edges_num), edge_grain_size, [&](const IndexRange range) {
    for (const int edge_idx : range) {
      const int2 &edge = base_edges[edge_idx];
      int offset = edge_idx * faces_per_side;
      int edge_offset = edge_offsets[edge_idx];

      r_edges[offset++] = {vert_offsets[edge[0]], edge_offset};

      for (int i = 1; i < faces_per_side - 1; i++) {
        r_edges[offset++] = {edge_offset, edge_offset + 1};
        edge_offset++;
      }
      r_edges[offset] = {edge_offset, vert_offsets[edge[1]]};
    }
  });

  const int subedges_per_face = 2 * faces_per_side * (faces_per_side - 1);
  const int face_edge_base_start = edges_num * faces_per_side;
  const OffsetIndices<int> base_faces = base_topo.faces;
  const Span<int> corner_verts = base_topo.corner_verts;
  const Span<int> corner_edges = base_topo.corner_edges;
  const BitSpan face_edge_reversed = base_topo.face_edge_reversed;

  const int face_grain_size = std::max(1, 4096 / (size * size));

  threading::parallel_for(IndexRange(faces_num), face_grain_size, [&](const IndexRange range) {
    for (const int face_idx : range) {
      const IndexRange face_range = base_faces[face_idx];
      const Span<int> face_verts = corner_verts.slice(face_range);
      const Span<int> face_edges = corner_edges.slice(face_range);

      std::array<int, 4> border_offsets = {edge_offsets[face_edges[0]],
                                           edge_offsets[face_edges[1]],
                                           edge_offsets[face_edges[2]],
                                           edge_offsets[face_edges[3]]};
      std::array<int, 4> border_start;
      std::array<int, 4> border_step;
      for (int i = 0; i < 4; i++) {
        int reverse = (int)face_edge_reversed[face_range.start() + i] ^ (i >> 1);
        border_step[i] = 1 - 2 * reverse;
        border_start[i] = border_offsets[i] + (reverse * (size - 1));
      }

      int offset = face_edge_base_start + face_idx * subedges_per_face;
      int face_offset = face_offsets[face_idx];

      {
        int edge_offset = border_start[0];

        for (int j = 0; j < size; j++) {
          r_edges[offset++] = {edge_offset, face_offset++};
          edge_offset += border_step[0];
        }
      }

      face_offset = face_offsets[face_idx];
      for (int row = 1; row < faces_per_side - 1; row++) {
        for (int j = 0; j < size; j++) {
          r_edges[offset++] = {face_offset, face_offset + size};
          face_offset++;
        }
      }

      {
        int edge_offset = border_start[2];
        for (int j = 0; j < size; j++) {
          r_edges[offset++] = {face_offset++, edge_offset};
          edge_offset += border_step[2];
        }
      }

      face_offset = face_offsets[face_idx];

      {
        int edge_offset = border_start[3];

        for (int j = 0; j < size; j++) {
          r_edges[offset++] = {edge_offset, face_offset};
          face_offset += size;
          edge_offset += border_step[3];
        }
      }

      for (int col = 1; col < faces_per_side - 1; col++) {
        face_offset = face_offsets[face_idx] + col - 1;
        for (int j = 0; j < size; j++) {
          r_edges[offset++] = {face_offset, face_offset + 1};
          face_offset += size;
        }
      }

      {
        face_offset = face_offsets[face_idx] + size - 1;
        int edge_offset = border_start[1];
        for (int j = 0; j < size; j++) {
          r_edges[offset++] = {face_offset, edge_offset};
          face_offset += size;
          edge_offset += border_step[1];
        }
      }
    }
  });
}

static void calc_corner_edges(const BaseTopology &base_topo,
                              const int lvl,
                              const MutableSpan<int> r_corner_edges)
{
  const int size = Multiresolution::grid_size(lvl);
  const int faces_per_side = size + 1;
  const int faces_per_grid = faces_per_side * faces_per_side;
  const int faces_num = base_topo.faces_num;
  const int edges_num = base_topo.edges_num;
  const OffsetIndices<int> base_faces = base_topo.faces;
  const Span<int> corner_edges = base_topo.corner_edges;
  const BitSpan face_edge_reversed = base_topo.face_edge_reversed;

  const int subedges_per_face = 2 * faces_per_side * size;
  const int face_edge_base_start = edges_num * faces_per_side;

  const int grain_size = std::max(1, 4096 / (size * size));

  threading::parallel_for(IndexRange(faces_num), grain_size, [&](const IndexRange range) {
    for (const int face_idx : range) {
      const IndexRange face_range = base_faces[face_idx];
      const Span<int> face_edges = corner_edges.slice(face_range);

      std::array<int, 4> border_start;
      std::array<int, 4> border_step;
      for (int i = 0; i < 4; i++) {
        const int reverse = (int)face_edge_reversed[face_range.start() + i] ^ (i >> 1);
        border_step[i] = 1 - 2 * reverse;
        border_start[i] = face_edges[i] * faces_per_side + reverse * size;
      }

      int *out = &r_corner_edges[face_idx * faces_per_grid * 4];

      const int face_edge_voffset = face_edge_base_start + face_idx * subedges_per_face;
      const int face_edge_hoffset = face_edge_voffset + subedges_per_face / 2;

      {
        int edge_voffset = face_edge_voffset;
        int edge_hoffset = face_edge_hoffset;

        out[0] = border_start[0];
        out[1] = edge_voffset;
        out[2] = edge_hoffset;
        out[3] = border_start[3];
        edge_voffset++;
        edge_hoffset += size;
        out += 4;

        for (int c = 1; c < faces_per_side - 1; c++) {
          out[0] = border_start[0] + c * border_step[0];
          out[1] = edge_voffset;
          out[2] = edge_hoffset;
          out[3] = edge_voffset - 1;
          edge_voffset++;
          edge_hoffset += size;
          out += 4;
        }

        out[0] = border_start[0] + (faces_per_side - 1) * border_step[0];
        out[1] = border_start[1];
        out[2] = face_edge_hoffset + size * (faces_per_side - 1);
        out[3] = face_edge_voffset + (faces_per_side - 1) - 1;
        out += 4;
      }

      for (int row = 1; row < faces_per_side - 1; row++) {
        int edge_voffset = face_edge_voffset + row * size;
        int edge_hoffset = face_edge_hoffset + row - 1;

        out[0] = edge_hoffset;
        out[1] = edge_voffset;
        out[2] = edge_hoffset + 1;
        out[3] = border_start[3] + row * border_step[3];
        edge_voffset++;
        edge_hoffset += size;
        out += 4;

        for (int col = 1; col < faces_per_side - 1; col++) {
          out[0] = edge_hoffset;
          out[1] = edge_voffset;
          out[2] = edge_hoffset + 1;
          out[3] = edge_voffset - 1;
          edge_voffset++;
          edge_hoffset += size;
          out += 4;
        }

        out[0] = edge_hoffset;
        out[1] = border_start[1] + row * border_step[1];
        out[2] = edge_hoffset + 1;
        out[3] = edge_voffset - 1;
        out += 4;
      }

      {
        int edge_voffset = face_edge_voffset + (faces_per_side - 1) * size;
        int edge_hoffset = face_edge_hoffset + faces_per_side - 2;
        out[0] = edge_hoffset;
        out[1] = edge_voffset;
        out[2] = border_start[2];
        out[3] = border_start[3] + (faces_per_side - 1) * border_step[3];
        edge_voffset++;
        edge_hoffset += size;
        out += 4;

        for (int col = 1; col < faces_per_side - 1; col++) {
          out[0] = edge_hoffset;
          out[1] = edge_voffset;
          out[2] = border_start[2] + border_step[2] * col;
          out[3] = edge_voffset - 1;
          edge_voffset++;
          edge_hoffset += size;
          out += 4;
        }

        out[0] = edge_hoffset;
        out[1] = border_start[1] + border_step[1] * (faces_per_side - 1);
        out[2] = border_start[2] + border_step[2] * (faces_per_side - 1);
        out[3] = edge_voffset - 1;
      }
    }
  });
}

static void calc_leaf_face_offsets_recursive(const int curr_idx,
                                             int &curr_offset,
                                             int &next_child_offset,
                                             const Span<int> face_counts,
                                             const MutableSpan<int> leaf_offsets)
{
  if (face_counts[curr_idx] != -1) {
    leaf_offsets[curr_idx] = curr_offset;
    curr_offset += face_counts[curr_idx];
    return;
  }
  const int children_offset = next_child_offset;
  next_child_offset += 2;
  calc_leaf_face_offsets_recursive(
      children_offset, curr_offset, next_child_offset, face_counts, leaf_offsets);
  calc_leaf_face_offsets_recursive(
      children_offset + 1, curr_offset, next_child_offset, face_counts, leaf_offsets);
}

template<typename OutputPolicy>
static Mesh *eval_topology_lvl1(const BaseTopology &base_topo,
                                const OutputPolicy target_vert_offsets,
                                const OutputPolicy target_edge_offsets,
                                const OutputPolicy target_face_offsets)
{
  const int eval_verts_num = base_topo.verts_num + base_topo.edges_num + base_topo.faces_num;
  const int eval_faces_num = base_topo.corners_num;
  const int eval_edges_num = 2 * base_topo.edges_num + base_topo.corners_num;

  Mesh *eval_mesh = BKE_mesh_new_nomain(
      eval_verts_num, eval_edges_num, eval_faces_num, eval_faces_num * 4);

  const OffsetIndices<int> base_faces = base_topo.faces;
  const Span<int> base_corner_edges = base_topo.corner_edges;
  const MutableSpan<int> eval_face_offsets = eval_mesh->face_offsets_for_write();
  const MutableSpan<int> eval_corner_verts = eval_mesh->corner_verts_for_write();
  const MutableSpan<int> eval_corner_edges = eval_mesh->corner_edges_for_write();

  offset_indices::fill_constant_group_size(4, 0, eval_face_offsets);

  calc_corner_verts(base_topo,
                    1,
                    target_vert_offsets,
                    target_edge_offsets,
                    target_face_offsets,
                    eval_corner_verts);

  const Span<int2> base_edges = base_topo.edges;
  const MutableSpan<int2> eval_edges = eval_mesh->edges_for_write();

  threading::parallel_for(IndexRange(base_topo.edges_num), 8192, [&](const IndexRange range) {
    for (const int edge_idx : range) {
      const int2 &edge = base_edges[edge_idx];
      const int mid = target_edge_offsets[edge_idx];
      eval_edges[2 * edge_idx + 0] = {target_vert_offsets[edge[0]], mid};
      eval_edges[2 * edge_idx + 1] = {mid, target_vert_offsets[edge[1]]};
    }
  });

  const int face_edge_base_start = 2 * base_topo.edges_num;

  threading::parallel_for(IndexRange(base_topo.faces_num), 4096, [&](const IndexRange range) {
    for (const int face_idx : range) {
      const IndexRange face_range = base_faces[face_idx];
      const Span<int> face_edges = base_corner_edges.slice(face_range);
      const int face_center = target_face_offsets[face_idx];
      int2 *out = &eval_edges[face_edge_base_start + face_range.start()];

      if (face_range.size() == 4) {
        const int m0 = target_edge_offsets[face_edges[0]];
        const int m1 = target_edge_offsets[face_edges[1]];
        const int m2 = target_edge_offsets[face_edges[2]];
        const int m3 = target_edge_offsets[face_edges[3]];

        out[0] = {m0, face_center};
        out[1] = {face_center, m2};
        out[2] = {m3, face_center};
        out[3] = {face_center, m1};
      }
      else {
        for (const int edge_idx : face_edges) {
          *out++ = {face_center, target_edge_offsets[edge_idx]};
        }
      }
    }
  });

  const BitSpan face_edge_reversed = base_topo.face_edge_reversed;

  threading::parallel_for(IndexRange(base_topo.faces_num), 2048, [&](const IndexRange range) {
    std::array<int, 4> border_start;
    std::array<int, 4> border_step;
    for (const int face_idx : range) {
      const IndexRange face_range = base_faces[face_idx];
      const Span<int> face_edges = base_corner_edges.slice(face_range);
      const int corners_num = face_range.size();
      const int face_start = face_range.start();

      int *out = &eval_corner_edges[face_start * 4];

      if (corners_num == 4) {
        for (int i = 0; i < 4; i++) {
          const int reverse = (int)face_edge_reversed[face_start + i] ^ (i >> 1);
          border_step[i] = 1 - 2 * reverse;
          border_start[i] = 2 * face_edges[i] + reverse;
        }

        const int face_edge_voffset = face_edge_base_start + face_start;
        const int face_edge_hoffset = face_edge_voffset + 2;

        out[0] = border_start[0];
        out[1] = face_edge_voffset + 0;
        out[2] = face_edge_hoffset + 0;
        out[3] = border_start[3];
        out[4] = border_start[0] + border_step[0];
        out[5] = border_start[1];
        out[6] = face_edge_hoffset + 1;
        out[7] = face_edge_voffset + 0;
        out[8] = face_edge_hoffset + 0;
        out[9] = face_edge_voffset + 1;
        out[10] = border_start[2];
        out[11] = border_start[3] + border_step[3];
        out[12] = face_edge_hoffset + 1;
        out[13] = border_start[1] + border_step[1];
        out[14] = border_start[2] + border_step[2];
        out[15] = face_edge_voffset + 1;
      }
      else {
        for (const int corner : IndexRange(corners_num)) {
          const int prev_corner = (corner - 1 + corners_num) % corners_num;

          const int edge_next = face_edges[corner];
          const int edge_prev = face_edges[prev_corner];

          const bool next_reversed = face_edge_reversed[face_start + corner];
          const bool prev_reversed = face_edge_reversed[face_start + prev_corner];

          out[0] = 2 * edge_next + (next_reversed ? 1 : 0);
          out[1] = face_edge_base_start + face_start + corner;
          out[2] = face_edge_base_start + face_start + prev_corner;
          out[3] = 2 * edge_prev + (prev_reversed ? 0 : 1);

          out += 4;
        }
      }
    }
  });

  return eval_mesh;
}

int calc_total_verts(const int lvl,
                     const int verts_num,
                     const int edges_num,
                     const int faces_num,
                     const int corners_num)
{
  if (lvl == 0) {
    return verts_num;
  }

  const int base1_verts_num = verts_num + edges_num + faces_num;

  if (lvl == 1) {
    return base1_verts_num;
  }

  const int base1_edges_num = 2 * edges_num + corners_num;
  const int base1_faces_num = corners_num;
  const int size = Multiresolution::grid_size(lvl - 1);
  return base1_verts_num + base1_edges_num * size + base1_faces_num * size * size;
}

int calc_total_faces(const int lvl, const int faces_num, const int corners_num)
{
  if (lvl == 0) {
    return faces_num;
  }

  return corners_num * (1 << (2 * (lvl - 1)));
}

template<typename OutputPolicy>
static Mesh *eval_topology(const BaseTopology &base_topo,
                           const int target_lvl,
                           const OutputPolicy target_vert_offsets,
                           const OutputPolicy target_edge_offsets,
                           const OutputPolicy target_face_offsets)
{
  if (target_lvl == 1) {
    return eval_topology_lvl1(
        base_topo, target_vert_offsets, target_edge_offsets, target_face_offsets);
  }

  const int grid_size = Multiresolution::grid_size(target_lvl);
  const int subedges_per_face = 2 * (grid_size + 1) * grid_size;
  const int eval_verts_num = calc_total_verts(target_lvl,
                                              base_topo.verts_num,
                                              base_topo.edges_num,
                                              base_topo.faces_num,
                                              base_topo.corners_num);
  const int eval_edges_num = base_topo.edges_num * (1 << target_lvl) +
                             base_topo.faces_num * subedges_per_face;
  const int eval_faces_num = base_topo.faces_num * (1 << (2 * target_lvl));

  Mesh *eval_mesh = BKE_mesh_new_nomain(
      eval_verts_num, eval_edges_num, eval_faces_num, eval_faces_num * 4);

  const MutableSpan<int> face_offsets = eval_mesh->face_offsets_for_write();
  const MutableSpan<int> corner_verts = eval_mesh->corner_verts_for_write();
  const MutableSpan<int2> edges = eval_mesh->edges_for_write();

  offset_indices::fill_constant_group_size(4, 0, face_offsets);

  bke::multiresolution::calc_corner_verts(base_topo,
                                          target_lvl,
                                          target_vert_offsets,
                                          target_edge_offsets,
                                          target_face_offsets,
                                          corner_verts);

  bke::multiresolution::calc_edges(
      base_topo, target_lvl, target_vert_offsets, target_edge_offsets, target_face_offsets, edges);

  const MutableSpan<int> corner_edges = eval_mesh->corner_edges_for_write();
  bke::multiresolution::calc_corner_edges(base_topo, target_lvl, corner_edges);

  return eval_mesh;
}

MultiresolutionRuntime::MultiresolutionRuntime(const Mesh &base_mesh)
{
  this->layers_active = base_mesh.sculpt_layers_len > 0;
}

MultiresolutionRuntime::~MultiresolutionRuntime()
{
  for (Mesh *mesh : base_meshes) {
    if (mesh != nullptr) {
      BKE_id_free(nullptr, mesh);
      mesh = nullptr;
    }
  }
  base_meshes.clear();
}

bool base_counts_match(const Mesh &base_mesh)
{
  const Multiresolution *multires = base_mesh.multires;
  if (multires == nullptr) {
    return true;
  }
  if (base_mesh.flag & ME_SCULPT_DYNAMIC_TOPOLOGY) {
    return false;
  }
  return base_mesh.verts_num == multires->base_verts_num &&
         base_mesh.edges_num == multires->base_edges_num &&
         base_mesh.faces_num == multires->base_faces_num &&
         base_mesh.corners_num == multires->base_corners_num;
}

bool base_topology_matches(const Mesh &base_mesh)
{
  const Multiresolution *multires = base_mesh.multires;
  if (multires == nullptr) {
    return true;
  }

  if (!base_counts_match(base_mesh)) {
    return false;
  }
  return calc_base_topology_hash(base_mesh) == multires->base_topology_hash;
}

bool is_active(const Mesh &base_mesh)
{
  const Multiresolution *multires = base_mesh.multires;
  if (multires == nullptr || multires->curr_level <= 0) {
    return false;
  }
  return base_topology_matches(base_mesh);
}

bool is_active_fast(const Mesh &base_mesh)
{
  const Multiresolution *multires = base_mesh.multires;
  if (multires == nullptr || multires->curr_level <= 0) {
    return false;
  }
  return base_counts_match(base_mesh);
}

Span<float3> MultiresolutionRuntime::eval_positions() const
{
  return eval_positions_;
}

MutableSpan<float3> MultiresolutionRuntime::eval_positions_for_write()
{
  return eval_positions_;
}

Span<int> MultiresolutionRuntime::corner_verts() const
{
  return corner_verts_;
}

Span<float3> MultiresolutionRuntime::normals() const
{
  return normals_;
}

struct VisibleLayer {
  Span<float3> displacements;
  float opacity;
};

static Vector<VisibleLayer> visible_layers_for_eval(const Mesh &base_mesh)
{
  Vector<VisibleLayer> layers;
  for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
    if (!layer.is_visible || layer.opacity == 0.0f) {
      continue;
    }
    Span<float3> displacements = bke::sculpt_layer::layer_displacements(base_mesh, layer.name);
    if (!displacements.is_empty()) {
      layers.append({displacements, layer.opacity});
    }
  }
  return layers;
}

static void eval_positions_range(const Span<VisibleLayer> layers,
                                 const Span<float3> base_positions,
                                 const MutableSpan<float3> eval_positions,
                                 const IndexRange range)
{
  MutableSpan<float3> out_positions = eval_positions.slice(range);
  out_positions.copy_from(base_positions.slice(range));

  for (const VisibleLayer &layer : layers) {
    const float opacity = layer.opacity;
    Span<float3> displacements = layer.displacements.slice(range);

    if (opacity == 1.0f) {
      for (int j = 0; j < range.size(); j++) {
        out_positions[j] += displacements[j];
      }
    }
    else {
      for (int j = 0; j < range.size(); j++) {
        out_positions[j] += displacements[j] * opacity;
      }
    }
  }
}

void MultiresolutionRuntime::calc_eval_positions(const Mesh &base_mesh,
                                                 const Span<float3> base_positions)
{
  eval_positions_.resize(base_positions.size());

  const Vector<VisibleLayer> layers = visible_layers_for_eval(base_mesh);

  threading::parallel_for(IndexRange(eval_positions_.size()), 2048, [&](const IndexRange range) {
    eval_positions_range(layers, base_positions, eval_positions_.as_mutable_span(), range);
  });
}

void MultiresolutionRuntime::calc_eval_positions(const Mesh &base_mesh,
                                                 const Span<float3> base_positions,
                                                 const IndexRange dirty_range)
{
  const Vector<VisibleLayer> layers = visible_layers_for_eval(base_mesh);
  eval_positions_range(layers, base_positions, eval_positions_.as_mutable_span(), dirty_range);
}

void MultiresolutionRuntime::rebuild(const Mesh &base_mesh, const OffsetsSpan offsets)
{
  const Multiresolution &multires = *base_mesh.multires;
  const BaseTopology base_topo = multires.get_base_topology(base_mesh);
  const int faces_per_corner = multires.faces_per_corner();
  const int verts_num = multires.verts_num();

  if (base_mesh.sculpt_layers_len > 0) {
    calc_eval_positions(base_mesh, multires.base_positions());
  }

  const int curr_base_lvl = multires.base_level(multires.curr_level);
  const int faces_num = base_topo.corners_num * faces_per_corner;

  corner_verts_.reinitialize(faces_num * 4);
  calc_corner_verts(base_topo,
                    multires.curr_level - curr_base_lvl,
                    offsets.verts,
                    offsets.edges,
                    offsets.faces,
                    corner_verts_);

  normals_.reinitialize(verts_num);
}

const Mesh &MultiresolutionRuntime::ensure_base_mesh(const Mesh &real_base_mesh,
                                                     const int lvl,
                                                     const bool all_quads)
{
  BLI_assert(lvl >= 1);

  if (lvl > base_meshes.size()) {
    base_meshes.resize(lvl, nullptr);
  }

  const int idx = lvl - 1;
  if (base_meshes[idx] != nullptr) {
    return *base_meshes[idx];
  }

  if (all_quads) {
    LinearOffset out_vp, out_ep, out_fp;
    make_linear_policies(lvl, real_base_mesh, out_vp, out_ep, out_fp);
    base_meshes[idx] = eval_topology(real_base_mesh, lvl, out_vp, out_ep, out_fp);
  }
  else {
    if (base_meshes[0] == nullptr) {
      LinearOffset out_vp, out_ep, out_fp;
      make_linear_policies(1, real_base_mesh, out_vp, out_ep, out_fp);
      base_meshes[0] = eval_topology_lvl1(real_base_mesh, out_vp, out_ep, out_fp);
    }

    if (lvl == 1) {
      return *base_meshes[0];
    }

    const Mesh &base_1 = *base_meshes[0];
    const int target_delta = lvl - 1;

    LinearOffset out_vp, out_ep, out_fp;
    make_linear_policies(target_delta, base_1, out_vp, out_ep, out_fp);
    base_meshes[idx] = eval_topology(base_1, target_delta, out_vp, out_ep, out_fp);
  }

  return *base_meshes[idx];
}

Span<MultiresCoord> MultiresolutionRuntime::coord_from_index_map_ensure(
    const bke::multiresolution::BaseTopology &base_topo,
    const int rel_lvl,
    const bke::multiresolution::OffsetsSpan offsets)
{
  const int verts_num = calc_total_verts(rel_lvl,
                                         base_topo.verts_num,
                                         base_topo.edges_num,
                                         base_topo.faces_num,
                                         base_topo.corners_num);

  if (coord_from_index_map.size() != verts_num) {
    coord_from_index_map.reinitialize(verts_num);
    const int grid_size = Multiresolution::grid_size(rel_lvl);

    for (const int vert : IndexRange(base_topo.verts_num)) {
      coord_from_index_map[offsets.verts[vert]] = {ElemType::Vert, vert, 0};
    }
    for (const int edge_idx : IndexRange(base_topo.edges_num)) {
      int offset = offsets.edges[edge_idx];
      for (const int i : IndexRange(grid_size)) {
        coord_from_index_map[offset++] = {ElemType::Edge, edge_idx, i};
      }
    }
    for (const int face_idx : IndexRange(base_topo.faces_num)) {
      int offset = offsets.faces[face_idx];
      for (const int i : IndexRange(grid_size * grid_size)) {
        coord_from_index_map[offset++] = {ElemType::Face, face_idx, i};
      }
    }
  }

  return coord_from_index_map;
}

void MultiresolutionRuntime::tag_curr_attr(const StringRef name)
{
  if (name == "position") {
    curr_positions_dirty = true;
  }
  else if (name == ".sculpt_mask") {
    curr_masks_dirty = true;
  }
  else if (name == "Color") {
    curr_colors_dirty = true;
  }
  else if (name.startswith(".sculpt_layer_")) {
    if (curr_all_layers_dirty) {
      return;
    }
    curr_dirty_layers.add(std::string(name));
  }
}

bool MultiresolutionRuntime::is_curr_layer_dirty(const StringRef attr_name) const
{
  if (curr_all_layers_dirty) {
    return true;
  }
  return curr_dirty_layers.contains(std::string(attr_name));
}

void MultiresolutionRuntime::clear_curr_dirty()
{
  curr_positions_dirty = false;
  curr_masks_dirty = false;
  curr_colors_dirty = false;
  curr_all_layers_dirty = false;
  curr_dirty_layers.clear();
}

void MultiresolutionRuntime::clear_sculpt_data()
{
  eval_positions_.clear_and_shrink();
  corner_verts_.clear_and_shrink();
  normals_.clear_and_shrink();
  coord_from_index_map.clear_and_shrink();
}

}  // namespace multiresolution

}  // namespace bke

void Multiresolution::free()
{
  if (this->runtime != nullptr) {
    for (int i = 0; i < this->bvh_partitions_len; i++) {
      bke::multiresolution::BVHPartitionSharingInfo &info =
          this->runtime->bvh_partition_sharing[i];
      implicit_sharing::free_shared_data(&this->bvh_partitions[i].face_indices,
                                         &info.face_indices);
      implicit_sharing::free_shared_data(&this->bvh_partitions[i].face_counts, &info.face_counts);
    }
    MEM_SAFE_DELETE(this->runtime);
  }

  if (this->bvh_partitions != nullptr) {
    MEM_delete(this->bvh_partitions);
    this->bvh_partitions = nullptr;
  }
  this->bvh_partitions_len = 0;

  this->top_attribute_storage.wrap().~AttributeStorage();
  this->curr_attribute_storage.wrap().~AttributeStorage();
}

void Multiresolution::bvh_partition_ensure(const int top_level,
                                           const int face_indices_len,
                                           const int face_counts_len)
{
  if (top_level <= this->bvh_partitions_len) {
    return;
  }

  const int new_len = top_level;
  this->bvh_partitions = static_cast<BVHPartition *>(
      MEM_realloc_zeroed(this->bvh_partitions, sizeof(BVHPartition) * new_len));
  this->bvh_partitions_len = new_len;
  this->runtime->bvh_partition_sharing.resize(new_len);

  BVHPartition &partition = this->bvh_partitions[top_level - 1];
  bke::multiresolution::BVHPartitionSharingInfo &info =
      this->runtime->bvh_partition_sharing[top_level - 1];

  partition.face_indices = MEM_new_array_uninitialized<int>(face_indices_len, "face_indices");
  partition.face_indices_len = face_indices_len;
  info.face_indices = implicit_sharing::info_for_mem_free(partition.face_indices);

  partition.face_counts = MEM_new_array_uninitialized<int>(face_counts_len, "face_counts");
  partition.face_counts_len = face_counts_len;
  info.face_counts = implicit_sharing::info_for_mem_free(partition.face_counts);
}

void Multiresolution::truncate_bvh_partitions(const int new_len)
{
  BLI_assert(new_len >= 0 && new_len <= this->bvh_partitions_len);
  if (new_len >= this->bvh_partitions_len) {
    return;
  }
  if (this->runtime != nullptr) {
    for (int i = new_len; i < this->bvh_partitions_len; i++) {
      bke::multiresolution::BVHPartitionSharingInfo &info =
          this->runtime->bvh_partition_sharing[i];
      implicit_sharing::free_shared_data(&this->bvh_partitions[i].face_indices,
                                         &info.face_indices);
      implicit_sharing::free_shared_data(&this->bvh_partitions[i].face_counts, &info.face_counts);
      this->bvh_partitions[i].face_indices_len = 0;
      this->bvh_partitions[i].face_counts_len = 0;
    }
    this->runtime->bvh_partition_sharing.resize(new_len);
  }
  this->bvh_partitions_len = new_len;
}

Span<int> Multiresolution::bvh_face_indices(const int level) const
{
  BLI_assert(level >= 1 && level <= this->bvh_partitions_len);
  const BVHPartition &partition = this->bvh_partitions[level - 1];
  return {partition.face_indices, partition.face_indices_len};
}

Span<int> Multiresolution::bvh_face_counts(const int level) const
{
  BLI_assert(level >= 1 && level <= this->bvh_partitions_len);
  const BVHPartition &partition = this->bvh_partitions[level - 1];
  return {partition.face_counts, partition.face_counts_len};
}

MutableSpan<int> Multiresolution::bvh_face_indices_for_write(const int level)
{
  BLI_assert(level >= 1 && level <= this->bvh_partitions_len);
  BVHPartition &partition = this->bvh_partitions[level - 1];
  bke::multiresolution::BVHPartitionSharingInfo &info =
      this->runtime->bvh_partition_sharing[level - 1];

  implicit_sharing::make_trivial_data_mutable(
      &partition.face_indices, &info.face_indices, partition.face_indices_len);
  return {partition.face_indices, partition.face_indices_len};
}

MutableSpan<int> Multiresolution::bvh_face_counts_for_write(const int level)
{
  BLI_assert(level >= 1 && level <= this->bvh_partitions_len);
  BVHPartition &partition = this->bvh_partitions[level - 1];
  bke::multiresolution::BVHPartitionSharingInfo &info =
      this->runtime->bvh_partition_sharing[level - 1];

  implicit_sharing::make_trivial_data_mutable(
      &partition.face_counts, &info.face_counts, partition.face_counts_len);
  return {partition.face_counts, partition.face_counts_len};
}

Span<bke::multiresolution::MultiresCoord> Multiresolution::coord_from_index_map_ensure(
    const bke::multiresolution::BaseTopology &base_topo,
    const bke::multiresolution::OffsetsSpan offsets) const
{
  const int curr_base_lvl = base_level(this->curr_level);
  return this->runtime->coord_from_index_map_ensure(
      base_topo, this->curr_level - curr_base_lvl, offsets);
}

bke::multiresolution::MultiresCoord Multiresolution::coord_from_index(const int idx) const
{
  BLI_assert(idx >= 0 && idx < this->verts_num());
  return this->runtime->coord_from_index_map[idx];
}

bke::multiresolution::BaseTopology Multiresolution::get_base_topology(const Mesh &base_mesh,
                                                                      const int lvl) const
{
  const int base_lvl = base_level(lvl);

  if (base_lvl == 0) {
    return base_mesh;
  }

  return this->runtime->ensure_base_mesh(base_mesh, base_lvl, this->base_all_quads);
}

bke::multiresolution::BaseTopology Multiresolution::get_base_topology(const Mesh &base_mesh) const
{
  return get_base_topology(base_mesh, this->curr_level);
}

static void calc_neighbors_vert(const int vert,
                                const bke::multiresolution::BaseTopology &base_topology,
                                const int grid_size,
                                const OffsetsSpan &offsets,
                                const Span<bool> hidden,
                                Vector<MultiresCoord> &r_neighbors)
{
  const Span<int2> base_edges = base_topology.edges;
  const GroupedSpan<int> vert_to_edges = base_topology.vert_to_edge_map;

  for (const int edge_idx : vert_to_edges[vert]) {
    const int2 &edge = base_edges[edge_idx];
    const int k = (edge[0] == vert) ? 0 : grid_size - 1;
    const MultiresCoord neighbor{ElemType::Edge, edge_idx, k};
    if (!hidden.is_empty() && hidden[neighbor.to_index(offsets)]) {
      continue;
    }
    r_neighbors.append(neighbor);
  }
}

static void calc_neighbors_edge(const int edge_idx,
                                const int k,
                                const bke::multiresolution::BaseTopology &base_topology,
                                const int grid_size,
                                const OffsetsSpan &offsets,
                                const Span<bool> hidden,
                                Vector<MultiresCoord> &r_neighbors)
{
  const Span<int2> base_edges = base_topology.edges;
  const int2 &edge = base_edges[edge_idx];

  auto append_if_visible = [&](const MultiresCoord &neighbor) {
    if (!hidden.is_empty() && hidden[neighbor.to_index(offsets)]) {
      return;
    }
    r_neighbors.append(neighbor);
  };

  append_if_visible(k == 0 ? MultiresCoord{ElemType::Vert, edge[0], 0} :
                             MultiresCoord{ElemType::Edge, edge_idx, k - 1});
  append_if_visible(k == grid_size - 1 ? MultiresCoord{ElemType::Vert, edge[1], 0} :
                                         MultiresCoord{ElemType::Edge, edge_idx, k + 1});

  const GroupedSpan<int> edge_to_faces = base_topology.edge_to_face_map;
  const OffsetIndices<int> base_faces = base_topology.faces;
  const Span<int> corner_edges = base_topology.corner_edges;
  const BitSpan face_edge_reversed = base_topology.face_edge_reversed;

  for (const int face_idx : edge_to_faces[edge_idx]) {
    const IndexRange face_range = base_faces[face_idx];

    const Span<int> face_edges = corner_edges.slice(face_range);
    const int local_i = face_edges.first_index(edge_idx);
    const bool reverse = face_edge_reversed[face_range.start() + local_i] ^ bool(local_i >> 1);
    const int j = reverse ? (grid_size - 1 - k) : k;

    int row, col;
    switch (local_i) {
      case 0:
        row = 0;
        col = j;
        break;
      case 1:
        row = j;
        col = grid_size - 1;
        break;
      case 2:
        row = grid_size - 1;
        col = j;
        break;
      default:
        row = j;
        col = 0;
        break;
    }
    append_if_visible({ElemType::Face, face_idx, row * grid_size + col});
  }
}

static void calc_neighbors_face(const int face_idx,
                                const int row,
                                const int col,
                                const bke::multiresolution::BaseTopology &base_topology,
                                const int grid_size,
                                const OffsetsSpan &offsets,
                                const Span<bool> hidden,
                                Vector<MultiresCoord> &r_neighbors)
{
  const OffsetIndices<int> base_faces = base_topology.faces;
  const IndexRange face_range = base_faces[face_idx];
  const Span<int> face_edges = base_topology.corner_edges.slice(face_range);
  const BitSpan face_edge_reversed = base_topology.face_edge_reversed;

  auto append_if_visible = [&](const MultiresCoord &neighbor) {
    if (!hidden.is_empty() && hidden[neighbor.to_index(offsets)]) {
      return;
    }
    r_neighbors.append(neighbor);
  };

  auto cross_edge = [&](const int local_i, const int j) {
    const int edge_idx = face_edges[local_i];
    const bool reverse = face_edge_reversed[face_range.start() + local_i] ^ bool(local_i >> 1);
    const int k = reverse ? (grid_size - 1 - j) : j;
    append_if_visible({ElemType::Edge, edge_idx, k});
  };

  if (grid_size == 1) {
    for (const int local_i : face_range.index_range()) {
      cross_edge(local_i, 0);
    }
    return;
  }

  if (row > 0) {
    append_if_visible({ElemType::Face, face_idx, (row - 1) * grid_size + col});
  }
  else {
    cross_edge(0, col);
  }
  if (row < grid_size - 1) {
    append_if_visible({ElemType::Face, face_idx, (row + 1) * grid_size + col});
  }
  else {
    cross_edge(2, col);
  }
  if (col > 0) {
    append_if_visible({ElemType::Face, face_idx, row * grid_size + (col - 1)});
  }
  else {
    cross_edge(3, row);
  }
  if (col < grid_size - 1) {
    append_if_visible({ElemType::Face, face_idx, row * grid_size + (col + 1)});
  }
  else {
    cross_edge(1, row);
  }
}

void Multiresolution::calc_neighbors(const MultiresCoord &coord,
                                     const bke::multiresolution::BaseTopology &base_topology,
                                     const int grid_size,
                                     const OffsetsSpan &offsets,
                                     const Span<bool> hidden,
                                     Vector<MultiresCoord> &r_neighbors)
{
  r_neighbors.clear();

  switch (coord.type) {
    case ElemType::Vert:
      calc_neighbors_vert(
          coord.base_index, base_topology, grid_size, offsets, hidden, r_neighbors);
      break;
    case ElemType::Edge:
      calc_neighbors_edge(coord.base_index,
                          coord.local_coord,
                          base_topology,
                          grid_size,
                          offsets,
                          hidden,
                          r_neighbors);
      break;
    case ElemType::Face:
      calc_neighbors_face(coord.base_index,
                          coord.local_coord / grid_size,
                          coord.local_coord % grid_size,
                          base_topology,
                          grid_size,
                          offsets,
                          hidden,
                          r_neighbors);
      break;
  }
}

int Multiresolution::base_level(const int lvl) const
{
  if (lvl <= 1) {
    return 0;
  }
  const int min_base_lvl = this->base_all_quads ? 0 : 1;
  return math::max(min_base_lvl, lvl - 5);
}

template<typename T>
static void convert_table_to_linear(const int lvl,
                                    const bke::multiresolution::BaseTopology &base_topo,
                                    const LinearOffset lin_verts,
                                    const LinearOffset lin_edges,
                                    const LinearOffset lin_faces,
                                    const Span<int> table_verts,
                                    const Span<int> table_edges,
                                    const Span<int> table_faces,
                                    const Span<T> in_attr,
                                    const MutableSpan<T> out_attr)
{
  const int grid_size = Multiresolution::grid_size(lvl);
  constexpr int total = 8192;

  threading::parallel_for(IndexRange(base_topo.verts_num), total, [&](const IndexRange range) {
    for (const int v : range) {
      out_attr[lin_verts[v]] = in_attr[table_verts[v]];
    }
  });

  if (grid_size == 0) {
    return;
  }

  const int edge_grain = math::max(1, total / grid_size);
  threading::parallel_for(
      IndexRange(base_topo.edges_num), edge_grain, [&](const IndexRange range) {
        for (const int e : range) {
          out_attr.slice(lin_edges[e], grid_size)
              .copy_from(in_attr.slice(table_edges[e], grid_size));
        }
      });

  const int face_verts_num = grid_size * grid_size;
  const int face_grain = math::max(1, total / face_verts_num);
  threading::parallel_for(
      IndexRange(base_topo.faces_num), face_grain, [&](const IndexRange range) {
        for (const int f : range) {
          out_attr.slice(lin_faces[f], face_verts_num)
              .copy_from(in_attr.slice(table_faces[f], face_verts_num));
        }
      });
}

int Multiresolution::grid_size(const int lvl)
{
  return (1 << lvl) - 1;
}

int Multiresolution::grid_size() const
{
  const int base_lvl = base_level(this->curr_level);
  return grid_size(this->curr_level - base_lvl);
}

int Multiresolution::faces_per_side(const int lvl)
{
  return grid_size(lvl) + 1;
}

int Multiresolution::faces_per_corner(const int lvl)
{
  const int size = faces_per_side(lvl);
  return (size * size) / 4;
}

int Multiresolution::faces_per_side() const
{
  const int base_lvl = base_level(this->curr_level);
  return faces_per_side(this->curr_level - base_lvl);
}

int Multiresolution::faces_per_corner() const
{
  const int base_lvl = base_level(this->curr_level);
  return faces_per_corner(this->curr_level - base_lvl);
}

namespace bke::multiresolution {

void calc_corner_verts(const BaseTopology &base_topo,
                       const int virtual_lvl,
                       const OffsetsSpan &offsets,
                       const MutableSpan<int> r_corner_verts)
{
  calc_corner_verts(
      base_topo, virtual_lvl, offsets.verts, offsets.edges, offsets.faces, r_corner_verts);
}

void calc_top_eval_positions(const Mesh &base_mesh,
                             const Span<float3> top_base_positions,
                             MutableSpan<float3> r_eval_positions)
{
  BLI_assert(top_base_positions.size() == r_eval_positions.size());
  r_eval_positions.copy_from(top_base_positions);
  for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
    bke::sculpt_layer::apply_displacement(
        bke::sculpt_layer::layer_top_displacements(base_mesh, layer.name),
        bke::sculpt_layer::effective_layer_weight(layer),
        r_eval_positions);
  }
}

}  // namespace bke::multiresolution

int Multiresolution::verts_num() const
{
  return bke::multiresolution::calc_total_verts(this->curr_level,
                                                this->base_verts_num,
                                                this->base_edges_num,
                                                this->base_faces_num,
                                                this->base_corners_num);
}

int Multiresolution::top_verts_num() const
{
  return bke::multiresolution::calc_total_verts(this->top_level,
                                                this->base_verts_num,
                                                this->base_edges_num,
                                                this->base_faces_num,
                                                this->base_corners_num);
}

int Multiresolution::faces_num() const
{
  return bke::multiresolution::calc_total_faces(
      this->curr_level, this->base_faces_num, this->base_corners_num);
}

int Multiresolution::top_faces_num() const
{
  return bke::multiresolution::calc_total_faces(
      this->top_level, this->base_faces_num, this->base_corners_num);
}

Span<float3> Multiresolution::positions() const
{
  if (this->runtime->layers_active) {
    return this->runtime->eval_positions();
  }
  return this->base_positions();
}

MutableSpan<float3> Multiresolution::positions_for_write()
{
  if (this->runtime->layers_active) {
    return this->runtime->eval_positions_for_write();
  }
  return this->base_positions_for_write();
}

Span<float3> Multiresolution::base_positions() const
{
  return this->attribute<float3>("position");
}

MutableSpan<float3> Multiresolution::base_positions_for_write()
{
  return this->attribute_for_write<float3>("position");
}

Span<float3> Multiresolution::top_base_positions() const
{
  return this->top_attribute<float3>("position");
}

MutableSpan<float3> Multiresolution::top_base_positions_for_write()
{
  return this->top_attribute_for_write<float3>("position");
}

Span<float> Multiresolution::masks() const
{
  return this->attribute<float>(".sculpt_mask");
}

MutableSpan<float> Multiresolution::masks_for_write()
{
  return this->attribute_for_write<float>(".sculpt_mask");
}

Span<float> Multiresolution::top_masks() const
{
  return this->top_attribute<float>(".sculpt_mask");
}

MutableSpan<float> Multiresolution::top_masks_for_write()
{
  return this->top_attribute_for_write<float>(".sculpt_mask");
}

Span<float4> Multiresolution::colors() const
{
  return this->attribute<ColorGeometry4f>("Color").cast<float4>();
}

MutableSpan<float4> Multiresolution::colors_for_write()
{
  return this->attribute_for_write<ColorGeometry4f>("Color").cast<float4>();
}

Span<float4> Multiresolution::top_colors() const
{
  return this->top_attribute<ColorGeometry4f>("Color").cast<float4>();
}

MutableSpan<float4> Multiresolution::top_colors_for_write()
{
  return this->top_attribute_for_write<ColorGeometry4f>("Color").cast<float4>();
}

Span<bool> Multiresolution::hidden_verts() const
{
  return this->attribute<bool>(".hide_vert");
}

MutableSpan<bool> Multiresolution::hidden_verts_for_write()
{
  return this->attribute_for_write<bool>(".hide_vert");
}

Span<bool> Multiresolution::hidden_faces() const
{
  return this->face_attribute<bool>(".hide_poly");
}

MutableSpan<bool> Multiresolution::hidden_faces_for_write()
{
  return this->face_attribute_for_write<bool>(".hide_poly");
}

Span<int> Multiresolution::face_sets() const
{
  return this->face_attribute<int>(".sculpt_face_set");
}
MutableSpan<int> Multiresolution::face_sets_for_write()
{
  return this->face_attribute_for_write<int>(".sculpt_face_set");
}

Span<int> Multiresolution::top_face_sets() const
{
  return this->top_face_attribute<int>(".sculpt_face_set");
}
MutableSpan<int> Multiresolution::top_face_sets_for_write()
{
  return this->top_face_attribute_for_write<int>(".sculpt_face_set");
}

template<typename T> Span<T> Multiresolution::attribute(const StringRef name) const
{
  if (this->curr_level == this->top_level) {
    return this->top_attribute<T>(name);
  }
  return bke::get_span_attribute<T>(
             this->curr_attribute_storage.wrap(), bke::AttrDomain::Point, name, this->verts_num())
      .value_or(Span<T>());
}

template Span<float> Multiresolution::attribute(const StringRef) const;
template Span<float3> Multiresolution::attribute(const StringRef) const;
template Span<float4> Multiresolution::attribute(const StringRef) const;

template<typename T> MutableSpan<T> Multiresolution::attribute_for_write(const StringRef name)
{
  if (this->curr_level == this->top_level) {
    return this->top_attribute_for_write<T>(name);
  }

  if (this->runtime != nullptr) {
    this->runtime->tag_curr_attr(name);
  }

  return bke::get_mutable_attribute<T>(
      this->curr_attribute_storage.wrap(), bke::AttrDomain::Point, name, this->verts_num());
}

template MutableSpan<float> Multiresolution::attribute_for_write(const StringRef name);
template MutableSpan<float3> Multiresolution::attribute_for_write(const StringRef name);
template MutableSpan<float4> Multiresolution::attribute_for_write(const StringRef name);

template<typename T> Span<T> Multiresolution::top_attribute(const StringRef name) const
{
  return bke::get_span_attribute<T>(this->top_attribute_storage.wrap(),
                                    bke::AttrDomain::Point,
                                    name,
                                    this->top_verts_num())
      .value_or(Span<T>());
}

template Span<float> Multiresolution::top_attribute(const StringRef) const;
template Span<float3> Multiresolution::top_attribute(const StringRef) const;
template Span<float4> Multiresolution::top_attribute(const StringRef) const;

template<typename T> MutableSpan<T> Multiresolution::top_attribute_for_write(const StringRef name)
{
  return bke::get_mutable_attribute<T>(
      this->top_attribute_storage.wrap(), bke::AttrDomain::Point, name, this->top_verts_num());
}

template MutableSpan<float> Multiresolution::top_attribute_for_write(const StringRef name);
template MutableSpan<float3> Multiresolution::top_attribute_for_write(const StringRef name);
template MutableSpan<float4> Multiresolution::top_attribute_for_write(const StringRef name);

template<typename T> Span<T> Multiresolution::face_attribute(const StringRef name) const
{
  if (this->curr_level == this->top_level) {
    return this->top_face_attribute<T>(name);
  }
  return bke::get_span_attribute<T>(
             this->curr_attribute_storage.wrap(), bke::AttrDomain::Face, name, this->faces_num())
      .value_or(Span<T>());
}

template Span<int> Multiresolution::face_attribute(const StringRef) const;
template Span<bool> Multiresolution::face_attribute(const StringRef) const;

template<typename T> MutableSpan<T> Multiresolution::face_attribute_for_write(const StringRef name)
{
  if (this->curr_level == this->top_level) {
    return this->top_face_attribute_for_write<T>(name);
  }

  return bke::get_mutable_attribute<T>(
      this->curr_attribute_storage.wrap(), bke::AttrDomain::Face, name, this->faces_num());
}

template MutableSpan<int> Multiresolution::face_attribute_for_write(const StringRef name);
template MutableSpan<bool> Multiresolution::face_attribute_for_write(const StringRef name);

template<typename T> Span<T> Multiresolution::top_face_attribute(const StringRef name) const
{
  return bke::get_span_attribute<T>(this->top_attribute_storage.wrap(),
                                    bke::AttrDomain::Face,
                                    name,
                                    this->top_faces_num())
      .value_or(Span<T>());
}

template Span<int> Multiresolution::top_face_attribute(const StringRef) const;
template Span<bool> Multiresolution::top_face_attribute(const StringRef) const;

template<typename T>
MutableSpan<T> Multiresolution::top_face_attribute_for_write(const StringRef name)
{
  return bke::get_mutable_attribute<T>(
      this->top_attribute_storage.wrap(), bke::AttrDomain::Face, name, this->top_faces_num());
}

template MutableSpan<int> Multiresolution::top_face_attribute_for_write(const StringRef name);
template MutableSpan<bool> Multiresolution::top_face_attribute_for_write(const StringRef name);

enum class StorageType {
  Curr,
  Top,
};

template<StorageType storage_type>
static int get_domain_size(const void *owner, const bke::AttrDomain domain)
{
  if (!owner) {
    return 0;
  }
  const Multiresolution &multires = *static_cast<const Multiresolution *>(owner);
  switch (domain) {
    case bke::AttrDomain::Point:
      return (storage_type == StorageType::Curr) ? multires.verts_num() : multires.top_verts_num();
    case bke::AttrDomain::Face:
      return (storage_type == StorageType::Curr) ? multires.faces_num() : multires.top_faces_num();
    default:
      return 0;
  }
}

template<StorageType storage_type>
static const bke::AttributeStorage &get_storage(const void *owner)
{
  const Multiresolution &multires = *static_cast<const Multiresolution *>(owner);
  return (storage_type == StorageType::Curr) ? multires.curr_attribute_storage.wrap() :
                                               multires.top_attribute_storage.wrap();
}

template<StorageType storage_type> static bke::AttributeStorage &get_storage(void *owner)
{
  Multiresolution &multires = *static_cast<Multiresolution *>(owner);
  return (storage_type == StorageType::Curr) ? multires.curr_attribute_storage.wrap() :
                                               multires.top_attribute_storage.wrap();
}

template<StorageType storage_type> static bke::AttributeAccessorFunctions get_accessor_functions()
{
  bke::AttributeAccessorFunctions fn{};
  fn.domain_supported = [](const void * /*owner*/, const bke::AttrDomain domain) {
    return ELEM(domain, bke::AttrDomain::Point, bke::AttrDomain::Face);
  };
  fn.domain_size = get_domain_size<storage_type>;
  fn.builtin_domain_and_type = [](const void * /*owner*/, const StringRef /*name*/)
      -> std::optional<bke::AttributeDomainAndType> { return std::nullopt; };
  fn.get_builtin_default = [](const void * /*owner*/, StringRef /*name*/) -> GPointer {
    return {};
  };
  fn.lookup_meta_data = [](const void *owner,
                           StringRef name) -> std::optional<bke::AttributeMetaData> {
    if (!owner) {
      return std::nullopt;
    }
    const bke::Attribute *attr = get_storage<storage_type>(owner).lookup(name);
    if (!attr) {
      return std::nullopt;
    }
    return bke::AttributeMetaData{attr->domain(), attr->data_type()};
  };
  fn.lookup = [](const void *owner, const StringRef name) -> bke::GAttributeReader {
    if (!owner) {
      return {};
    }
    const bke::Attribute *attr = get_storage<storage_type>(owner).lookup(name);
    if (!attr) {
      return {};
    }
    const int domain_size = get_domain_size<storage_type>(owner, attr->domain());
    return attribute_to_reader(*attr, attr->domain(), domain_size);
  };
  fn.adapt_domain = [](const void * /*owner*/,
                       const GVArray &varray,
                       const bke::AttrDomain from_domain,
                       const bke::AttrDomain to_domain) {
    if (from_domain == to_domain) {
      return varray;
    }
    return GVArray{};
  };
  fn.foreach_attribute = [](const void *owner,
                            const FunctionRef<void(const bke::AttributeIter &)> fn,
                            const bke::AttributeAccessor &accessor) {
    if (!owner) {
      return;
    }
    for (const bke::Attribute &attr : get_storage<storage_type>(owner)) {
      const auto get_fn = [&]() {
        const int domain_size = get_domain_size<storage_type>(owner, attr.domain());
        return attribute_to_reader(attr, attr.domain(), domain_size);
      };
      bke::AttributeIter iter(attr.name(), attr.domain(), attr.data_type(), get_fn);
      iter.is_builtin = false;
      iter.storage_type = attr.storage_type();
      iter.accessor = &accessor;
      fn(iter);
      if (iter.is_stopped()) {
        break;
      }
    }
  };
  fn.lookup_validator = [](const void * /*owner*/,
                           const StringRef /*name*/) -> bke::AttributeValidator { return {}; };
  fn.lookup_for_write = [](void *owner, const StringRef name) -> bke::GAttributeWriter {
    if (!owner) {
      return {};
    }
    Multiresolution &multires = *static_cast<Multiresolution *>(owner);
    bke::Attribute *attr = get_storage<storage_type>(owner).lookup(name);
    if (!attr) {
      return {};
    }
    const int domain_size = get_domain_size<storage_type>(owner, attr->domain());
    return attribute_to_writer(&multires, {}, domain_size, *attr);
  };
  fn.remove = [](void *owner, const StringRef name) -> bool {
    if (!owner) {
      return false;
    }
    return get_storage<storage_type>(owner).remove(name);
  };
  fn.add = [](void *owner,
              const StringRef name,
              const bke::AttrDomain domain,
              const bke::AttrType type,
              const bke::AttributeInit &initializer) {
    if (!owner) {
      return false;
    }
    if (!ELEM(domain, bke::AttrDomain::Point, bke::AttrDomain::Face)) {
      return false;
    }
    bke::AttributeStorage &storage = get_storage<storage_type>(owner);
    if (storage.lookup(name)) {
      return false;
    }
    const int domain_size = get_domain_size<storage_type>(owner, domain);
    bke::GList data = attribute_init_to_list(type, domain_size, initializer, false);
    storage.add(name, domain, std::move(data));
    return true;
  };
  fn.assign_data = [](void *owner, StringRef name, const bke::AttributeInit &initializer) {
    if (!owner) {
      return false;
    }
    bke::Attribute *attr = get_storage<storage_type>(owner).lookup(name);
    if (!attr) {
      return false;
    }
    const int domain_size = get_domain_size<storage_type>(owner, attr->domain());
    bke::GList data = attribute_init_to_list(attr->data_type(), domain_size, initializer, false);
    attr->assign_data(std::move(data));
    return true;
  };
  return fn;
}

static const bke::AttributeAccessorFunctions &curr_attribute_accessor_functions()
{
  static const bke::AttributeAccessorFunctions fn = get_accessor_functions<StorageType::Curr>();
  return fn;
}

static const bke::AttributeAccessorFunctions &top_attribute_accessor_functions()
{
  static const bke::AttributeAccessorFunctions fn = get_accessor_functions<StorageType::Top>();
  return fn;
}

bke::AttributeAccessor Multiresolution::curr_attributes() const
{
  return bke::AttributeAccessor(this, curr_attribute_accessor_functions());
}

bke::MutableAttributeAccessor Multiresolution::curr_attributes_for_write()
{
  return bke::MutableAttributeAccessor(this, curr_attribute_accessor_functions());
}

bke::AttributeAccessor Multiresolution::top_attributes() const
{
  return bke::AttributeAccessor(this, top_attribute_accessor_functions());
}

bke::MutableAttributeAccessor Multiresolution::top_attributes_for_write()
{
  return bke::MutableAttributeAccessor(this, top_attribute_accessor_functions());
}

bke::AttributeAccessor Multiresolution::attributes() const
{
  if (this->curr_level == this->top_level) {
    return this->top_attributes();
  }
  return this->curr_attributes();
}

bke::MutableAttributeAccessor Multiresolution::attributes_for_write()
{
  if (this->curr_level == this->top_level) {
    return this->top_attributes_for_write();
  }
  return this->curr_attributes_for_write();
}

void Multiresolution::clear_top_storage()
{
  this->top_attribute_storage.wrap() = bke::AttributeStorage();
}

void Multiresolution::clear_sculpt_data()
{
  if (this->runtime != nullptr) {
    this->runtime->clear_sculpt_data();
  }
}

void Multiresolution::face_sets_free()
{
  this->attributes_for_write().remove(".sculpt_face_set");
}

void Multiresolution::hidden_verts_free()
{
  this->top_attribute_storage.wrap().remove(".hide_vert");
  this->curr_attribute_storage.wrap().remove(".hide_vert");
}

void Multiresolution::hidden_faces_free()
{
  this->attributes_for_write().remove(".hide_poly");
}

Span<float3> Multiresolution::normals() const
{
  return this->runtime->normals_;
}

MutableSpan<float3> Multiresolution::normals_for_write() const
{
  return this->runtime->normals_;
}

Span<int> Multiresolution::corner_verts() const
{
  return this->runtime->corner_verts_;
}

MutableSpan<float> Multiresolution::ensure_masks()
{
  return this->masks_for_write();
}

MutableSpan<float4> Multiresolution::ensure_colors()
{
  if (this->colors().is_empty()) {
    this->colors_for_write().fill(float4(1.0f));
  }
  return this->colors_for_write();
}

MutableSpan<bool> Multiresolution::ensure_hidden_verts()
{
  return this->hidden_verts_for_write();
}

MutableSpan<bool> Multiresolution::ensure_hidden_faces()
{
  return this->hidden_faces_for_write();
}

template<typename T>
void Multiresolution::grid_inner_border(const int grid_offset,
                                        const int local_edge,
                                        const bool reverse,
                                        const Span<T> attr,
                                        const MutableSpan<T> r_inner_border) const
{
  extract_grid_inner_border(grid_size(), grid_offset, local_edge, reverse, attr, r_inner_border);
}

template void Multiresolution::grid_inner_border<float>(
    const int, const int, const bool, const Span<float> attr, const MutableSpan<float>) const;

template void Multiresolution::grid_inner_border<float3>(
    const int, const int, const bool, const Span<float3> attr, const MutableSpan<float3>) const;

template void Multiresolution::grid_inner_border<float4>(
    const int, const int, const bool, const Span<float4> attr, const MutableSpan<float4>) const;

int Multiresolution::grid_inner_border_index(const int grid_offset,
                                             const int local_edge,
                                             const bool reverse,
                                             const int grid_size,
                                             const int k)
{
  const bool need_reverse = (local_edge >= 2) ^ reverse;
  const int i = need_reverse ? (grid_size - 1 - k) : k;
  if (local_edge & 1) {
    const int offset = local_edge == 1 ? grid_size - 1 : 0;
    return grid_offset + offset + i * grid_size;
  }
  const int offset = local_edge == 0 ? 0 : grid_size * (grid_size - 1);
  return grid_offset + offset + i;
}

namespace {

template<typename T, typename InPolicy, typename OutPolicy>
void subdivide_grids(const int lvl,
                     const bke::multiresolution::BaseTopology &base_topo,
                     const InPolicy in_vert_offsets,
                     const InPolicy in_edge_offsets,
                     const InPolicy in_face_offsets,
                     const OutPolicy out_offsets,
                     const Span<T> in_attr,
                     const MutableSpan<T> out_attr)
{
  const Span<int2> edges = base_topo.edges;
  const OffsetIndices<int> faces = base_topo.faces;
  const Span<int> corner_verts = base_topo.corner_verts;
  const Span<int> corner_edges = base_topo.corner_edges;
  const BitSpan face_edge_reversed = base_topo.face_edge_reversed;
  const int in_size = Multiresolution::grid_size(lvl);
  const int out_size = Multiresolution::grid_size(lvl + 1);
  const int in_size_sq = in_size * in_size;
  const int out_size_sq = out_size * out_size;

  if (lvl == 0) {
    threading::parallel_for(IndexRange(base_topo.faces_num), 2048, [&](const IndexRange range) {
      for (const int face_idx : range) {
        const Span<int> face_verts = corner_verts.slice(faces[face_idx]);
        const int out_idx = out_offsets[face_idx];
        T sum = T(0.0f);

        for (const int vert : face_verts) {
          sum += in_attr[in_vert_offsets[vert]];
        }

        out_attr[out_idx] = sum / float(face_verts.size());
      }
    });

    return;
  }

  if (lvl == 1) {
    threading::parallel_for(IndexRange(base_topo.faces_num), 1024, [&](const IndexRange range) {
      for (const int face_idx : range) {
        const T c = in_attr[in_face_offsets[face_idx]];

        const Span<int> face_verts = corner_verts.slice(faces[face_idx]);
        const T v_bl = in_attr[in_vert_offsets[face_verts[0]]];
        const T v_br = in_attr[in_vert_offsets[face_verts[1]]];
        const T v_tr = in_attr[in_vert_offsets[face_verts[2]]];
        const T v_tl = in_attr[in_vert_offsets[face_verts[3]]];

        const Span<int> face_edges = corner_edges.slice(faces[face_idx]);
        const T v_b = in_attr[in_edge_offsets[face_edges[0]]];
        const T v_r = in_attr[in_edge_offsets[face_edges[1]]];
        const T v_t = in_attr[in_edge_offsets[face_edges[2]]];
        const T v_l = in_attr[in_edge_offsets[face_edges[3]]];

        const MutableSpan<T> out_grid = out_attr.slice(out_offsets[face_idx], 9);
        const T f0 = v_bl + v_b + v_l + c;
        const T f1 = v_b + v_br + c + v_r;
        const T f2 = v_l + c + v_tl + v_t;
        const T f3 = c + v_r + v_t + v_tr;

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

  const int grain_size = std::max(1, 4096 / out_size_sq);

  threading::parallel_for(
      IndexRange(base_topo.faces_num), grain_size, [&](const IndexRange range) {
        Array<T> bot_buf(in_size + 2);
        Array<T> top_buf(in_size + 2);
        Array<T> left_buf(in_size + 2);
        Array<T> right_buf(in_size + 2);

        for (const int face_idx : range) {
          Span<T> grid = in_attr.slice(in_face_offsets[face_idx], in_size_sq);
          MutableSpan<T> out_grid = out_attr.slice(out_offsets[face_idx], out_size_sq);
          const IndexRange face_range = faces[face_idx];

          const Span<int> face_verts = corner_verts.slice(face_range);
          const T v_bl = in_attr[in_vert_offsets[face_verts[0]]];
          const T v_br = in_attr[in_vert_offsets[face_verts[1]]];
          const T v_tr = in_attr[in_vert_offsets[face_verts[2]]];
          const T v_tl = in_attr[in_vert_offsets[face_verts[3]]];

          const int corner_offset = face_range.start();
          const Span<int> face_edges = corner_edges.slice(face_range);

          fill_border(v_bl,
                      v_br,
                      face_edge_reversed[corner_offset + 0],
                      in_attr.slice(in_edge_offsets[face_edges[0]], in_size),
                      bot_buf.as_mutable_span());
          fill_border(v_br,
                      v_tr,
                      face_edge_reversed[corner_offset + 1],
                      in_attr.slice(in_edge_offsets[face_edges[1]], in_size),
                      right_buf.as_mutable_span());
          fill_border(v_tl,
                      v_tr,
                      !face_edge_reversed[corner_offset + 2],
                      in_attr.slice(in_edge_offsets[face_edges[2]], in_size),
                      top_buf.as_mutable_span());
          fill_border(v_bl,
                      v_tl,
                      !face_edge_reversed[corner_offset + 3],
                      in_attr.slice(in_edge_offsets[face_edges[3]], in_size),
                      left_buf.as_mutable_span());

          for (int row = 0, row_offset = 0; row < in_size; row++, row_offset += in_size) {
            Span<T> in_bot = row == 0 ? bot_buf.as_span().slice(1, in_size) :
                                        grid.slice(row_offset - in_size, in_size);
            Span<T> in_mid = grid.slice(row_offset, in_size);
            Span<T> in_top = (row == in_size - 1) ? top_buf.as_span().slice(1, in_size) :
                                                    grid.slice(row_offset + in_size, in_size);

            MutableSpan<T> out_even = out_grid.slice(2 * row * out_size, out_size);
            MutableSpan<T> out_odd = out_grid.slice((2 * row + 1) * out_size, out_size);

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
          {
            int j = (in_size - 1) * in_size;
            int idx = (out_size - 1) * out_size;

            T m0 = left_buf[in_size];
            T t0 = left_buf[in_size + 1];
            T m1 = grid[j];
            T t1 = top_buf[1];

            for (int col = 0; col < in_size - 1; col++) {
              const T m2 = grid[j + 1];
              const T t2 = top_buf[col + 2];
              out_grid[idx] = (m0 + m1 + t0 + t1) * 0.25f;
              out_grid[idx + 1] = ((m1 + t1) * 0.375f) + ((m0 + m2 + t0 + t2) * 0.0625f);
              m0 = m1;
              t0 = t1;
              m1 = m2;
              t1 = t2;
              j += 1;
              idx += 2;
            }

            const T m_last = right_buf[in_size];
            const T t_last = right_buf[in_size + 1];
            out_grid[idx] = (m0 + m1 + t0 + t1) * 0.25f;
            out_grid[idx + 1] = ((m1 + t1) * 0.375f) + ((m0 + m_last + t0 + t_last) * 0.0625f);
            out_grid[idx + 2] = (m1 + m_last + t1 + t_last) * 0.25f;
          }
        }
      });
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
  extract_grid_inner_border(size, face_offset, local_edge, false, in_attr, side.slice(1, size));

  const int prev_edge_local = (local_edge + 3) & 3;
  const int prev_edge_idx = corner_edges[corner_start + prev_edge_local];
  const int vert_start = corner_verts[corner_start + local_edge];
  const int2 prev_edge = edges[prev_edge_idx];
  const int prev_pt_idx = (prev_edge[0] == vert_start) ? 0 : size - 1;
  side[0] = in_attr[in_edge_offsets[prev_edge_idx] + prev_pt_idx];

  const int next_edge_local = (local_edge + 1) & 3;
  const int next_edge_idx = corner_edges[corner_start + next_edge_local];
  const int vert_end = corner_verts[corner_start + next_edge_local];
  const int2 next_edge = edges[next_edge_idx];
  const int next_pt_idx = (next_edge[0] == vert_end) ? 0 : size - 1;
  side[size + 1] = in_attr[in_edge_offsets[next_edge_idx] + next_pt_idx];

  if (reverse) {
    std::reverse(side.begin(), side.end());
  }
}

template<typename T, typename InPolicy, typename OutPolicy>
void subdivide_borders(const int lvl,
                       const bke::multiresolution::BaseTopology &base_topo,
                       const InPolicy in_vert_offsets,
                       const InPolicy in_edge_offsets,
                       const InPolicy in_face_offsets,
                       const OutPolicy out_offsets,
                       const Span<T> in_attr,
                       const MutableSpan<T> out_attr)
{
  const int in_size = Multiresolution::grid_size(lvl);
  const int out_size = Multiresolution::grid_size(lvl + 1);
  const Span<int2> edges = base_topo.edges;
  const OffsetIndices<int> faces = base_topo.faces;
  const Span<int> corner_verts = base_topo.corner_verts;
  const Span<int> corner_edges = base_topo.corner_edges;
  const GroupedSpan<int> edge_to_faces = base_topo.edge_to_face_map;
  const BitSpan face_edge_reversed = base_topo.face_edge_reversed;

  if (lvl == 0) {
    threading::parallel_for(IndexRange(base_topo.edges_num), 1024, [&](const IndexRange range) {
      for (const int edge_idx : range) {
        const int2 edge = edges[edge_idx];
        const Span<int> edge_faces = edge_to_faces[edge_idx];
        if (edge_faces.size() == 1) {
          out_attr[out_offsets[edge_idx]] = (in_attr[in_vert_offsets[edge[0]]] +
                                             in_attr[in_vert_offsets[edge[1]]]) *
                                            0.5f;
        }
        else {
          const int face1 = edge_to_faces[edge_idx][0];
          const int face2 = edge_to_faces[edge_idx][1];
          const IndexRange face1_range = faces[face1];
          const IndexRange face2_range = faces[face2];
          T center1 = T(0.0f);
          T center2 = T(0.0f);

          for (const int corner_vert : corner_verts.slice(face1_range)) {
            center1 += in_attr[in_vert_offsets[corner_vert]];
          }
          center1 /= float(face1_range.size());

          for (const int corner_vert : corner_verts.slice(face2_range)) {
            center2 += in_attr[in_vert_offsets[corner_vert]];
          }
          center2 /= float(face2_range.size());

          const T v1 = in_attr[in_vert_offsets[edge[0]]];
          const T v2 = in_attr[in_vert_offsets[edge[1]]];

          out_attr[out_offsets[edge_idx]] = (v1 + v2 + center1 + center2) * 0.25f;
        }
      }
    });
    return;
  }

  threading::parallel_for(IndexRange(base_topo.edges_num), 1024, [&](const IndexRange range) {
    Array<T> top_side(in_size + 2);
    Array<T> bot_side(in_size + 2);

    for (const int edge_idx : range) {
      const int2 edge = edges[edge_idx];
      const Span<T> inner_edge = in_attr.slice(in_edge_offsets[edge_idx], in_size);
      MutableSpan<T> r_edge_out = out_attr.slice(out_offsets[edge_idx], out_size);

      const Span<int> adj_faces = edge_to_faces[edge_idx];

      if (adj_faces.size() == 1) {
        T mid_prev = in_attr[in_vert_offsets[edge[0]]];
        const T v1 = in_attr[in_vert_offsets[edge[1]]];

        for (const int i : IndexRange(in_size)) {
          T mid_curr = inner_edge[i];
          T mid_next = (i == in_size - 1) ? v1 : inner_edge[i + 1];
          r_edge_out[i * 2] = (mid_prev + mid_curr) * 0.5f;
          r_edge_out[i * 2 + 1] = (mid_prev + mid_next) * 0.125f + mid_curr * 0.75f;
          mid_prev = mid_curr;
        }
        r_edge_out[2 * in_size] = (mid_prev + v1) * 0.5f;
        continue;
      }

      for (int i = 0; i < 2; i++) {
        const int face_idx = adj_faces[i];
        const IndexRange face_range = faces[face_idx];
        const int corner_start = face_range.start();

        const Span<int> face_edges = corner_edges.slice(face_range);
        const int local_idx = face_edges.first_index(edge_idx);

        const bool is_reversed = face_edge_reversed[corner_start + local_idx];
        MutableSpan<T> target = (i == 0) ? top_side : bot_side;

        fill_border_side(in_size,
                         corner_edges,
                         corner_verts,
                         edges,
                         corner_start,
                         local_idx,
                         is_reversed,
                         in_face_offsets[face_idx],
                         in_edge_offsets,
                         in_attr,
                         target);
      }

      T mid_prev = in_attr[in_vert_offsets[edge[0]]];
      const T v1 = in_attr[in_vert_offsets[edge[1]]];

      for (int k = 0; k < in_size; k++) {
        T mid_curr = inner_edge[k];
        T mid_next = (k == in_size - 1) ? v1 : inner_edge[k + 1];

        T t0 = top_side[k], t1 = top_side[k + 1], t2 = top_side[k + 2];
        T b0 = bot_side[k], b1 = bot_side[k + 1], b2 = bot_side[k + 2];

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

template<typename T, typename InPolicy, typename OutPolicy>
void subdivide_base(const int lvl,
                    const bke::multiresolution::BaseTopology &base_topo,
                    const InPolicy in_vert_offsets,
                    const InPolicy in_edge_offsets,
                    const InPolicy in_face_offsets,
                    const OutPolicy out_offsets,
                    const Span<T> in_attr,
                    const MutableSpan<T> out_attr)
{
  const int in_size = Multiresolution::grid_size(lvl);
  const Span<int2> edges = base_topo.edges;
  const OffsetIndices<int> faces = base_topo.faces;
  const Span<int> corner_verts = base_topo.corner_verts;
  const Span<int> corner_to_face = base_topo.corner_to_face_map;
  const GroupedSpan<int> vert_to_corners = base_topo.vert_to_corner_map;
  const GroupedSpan<int> vert_to_edges = base_topo.vert_to_edge_map;
  const BitSpan boundary_vert = base_topo.boundary_verts;
  const BitSpan boundary_edge = base_topo.boundary_edges;

  if (lvl == 0) {
    threading::parallel_for(IndexRange(base_topo.verts_num), 1024, [&](const IndexRange range) {
      for (const int i : range) {
        if (boundary_vert[i]) {
          T sum(0.0f);
          for (const int edge_idx : vert_to_edges[i]) {
            if (boundary_edge[edge_idx]) {
              const int2 edge = edges[edge_idx];
              const int other = edge[0] == i ? edge[1] : edge[0];
              sum += in_attr[in_vert_offsets[other]];
            }
          }
          out_attr[out_offsets[i]] = sum * 0.125f + in_attr[in_vert_offsets[i]] * 0.75f;
          continue;
        }

        const Span<int> vert_edges = vert_to_edges[i];
        T sum_edges(0.0f);

        for (const int edge_idx : vert_edges) {
          const int2 edge = edges[edge_idx];
          const int other = (edge[0] == i) ? edge[1] : edge[0];
          sum_edges += in_attr[in_vert_offsets[other]];
        }

        const Span<int> vert_corners = vert_to_corners[i];
        T sum_faces(0.0f);

        for (const int corner : vert_corners) {
          const int face_idx = corner_to_face[corner];
          const IndexRange face_range = faces[face_idx];
          T face_center(0.0f);
          for (const int corner_vert : corner_verts.slice(face_range)) {
            face_center += in_attr[in_vert_offsets[corner_vert]];
          }
          sum_faces += face_center / float(face_range.size());
        }

        const float n = float(vert_edges.size());
        out_attr[out_offsets[i]] = (sum_faces + sum_edges +
                                    in_attr[in_vert_offsets[i]] * (n * (n - 2.0f))) /
                                   (n * n);
      }
    });
    return;
  }

  threading::parallel_for(IndexRange(base_topo.verts_num), 1024, [&](const IndexRange range) {
    for (const int i : range) {
      if (boundary_vert[i]) {
        T sum(0.0f);
        for (const int edge_idx : vert_to_edges[i]) {
          if (boundary_edge[edge_idx]) {
            const int2 edge = edges[edge_idx];
            const int pos = edge[0] == i ? 0 : in_size - 1;
            sum += in_attr[in_edge_offsets[edge_idx] + pos];
          }
        }
        out_attr[out_offsets[i]] = sum * 0.125f + in_attr[in_vert_offsets[i]] * 0.75f;
        continue;
      }

      const Span<int> vert_edges = vert_to_edges[i];
      T sum_edges = T(0.0f);

      for (const int edge_idx : vert_edges) {
        const int2 edge = edges[edge_idx];
        const int pos = edge[0] == i ? 0 : in_size - 1;
        sum_edges += in_attr[in_edge_offsets[edge_idx] + pos];
      }

      const Span<int> v_corners = vert_to_corners[i];
      T sum_faces = T(0.0f);

      for (const int corner : v_corners) {
        const int face_idx = corner_to_face[corner];
        const int local_idx = corner - faces[face_idx].start();

        const int row = local_idx < 2 ? 0 : in_size - 1;
        const int col = local_idx == 0 || local_idx == 3 ? 0 : in_size - 1;
        sum_faces += in_attr[in_face_offsets[face_idx] + row * in_size + col];
      }

      const float n = float(v_corners.size());
      const float d = 4.0f * n * n;

      out_attr[out_offsets[i]] = (sum_faces + 6.0f * sum_edges +
                                  (d - 7.0f * n) * in_attr[in_vert_offsets[i]]) /
                                 d;
    }
  });
}

template<typename T, typename InPolicy, typename OutPolicy>
void subdivide(const int curr_lvl,
               const bke::multiresolution::BaseTopology &base_topo,
               const InPolicy in_vert_offsets,
               const InPolicy in_edge_offsets,
               const InPolicy in_face_offsets,
               const OutPolicy out_vert_offsets,
               const OutPolicy out_edge_offsets,
               const OutPolicy out_face_offsets,
               const Span<T> in_attr,
               const MutableSpan<T> out_attr)
{
  subdivide_base(curr_lvl,
                 base_topo,
                 in_vert_offsets,
                 in_edge_offsets,
                 in_face_offsets,
                 out_vert_offsets,
                 in_attr,
                 out_attr);

  subdivide_borders(curr_lvl,
                    base_topo,
                    in_vert_offsets,
                    in_edge_offsets,
                    in_face_offsets,
                    out_edge_offsets,
                    in_attr,
                    out_attr);

  subdivide_grids(curr_lvl,
                  base_topo,
                  in_vert_offsets,
                  in_edge_offsets,
                  in_face_offsets,
                  out_face_offsets,
                  in_attr,
                  out_attr);
}

template<typename T>
BLI_INLINE void run_row_linear(const int in_size,
                               const T left_bottom,
                               const T left_top,
                               const T right_bottom,
                               const T right_top,
                               const Span<T> bottom_row,
                               const Span<T> top_row,
                               const MutableSpan<T> out_mid_row,
                               const MutableSpan<T> out_top_row)
{
  T bl = left_bottom;
  T tl = left_top;

  for (int col = 0; col < in_size; col++) {
    const T br = bottom_row[col];
    const T tr = top_row[col];

    out_mid_row[2 * col] = (bl + br + tl + tr) * 0.25f;
    out_mid_row[2 * col + 1] = (br + tr) * 0.5f;
    out_top_row[2 * col] = (tl + tr) * 0.5f;
    out_top_row[2 * col + 1] = tr;

    bl = br;
    tl = tr;
  }

  out_mid_row[2 * in_size] = (bl + right_bottom + tl + right_top) * 0.25f;
  out_top_row[2 * in_size] = (tl + right_top) * 0.5f;
}

template<typename T, typename InPolicy, typename OutPolicy>
void subdivide_grids_linear(const int lvl,
                            const bke::multiresolution::BaseTopology &base_topo,
                            const InPolicy in_vert_offsets,
                            const InPolicy in_edge_offsets,
                            const InPolicy in_face_offsets,
                            const OutPolicy out_offsets,
                            const Span<T> in_attr,
                            const MutableSpan<T> out_attr)
{
  const Span<int2> edges = base_topo.edges;
  const OffsetIndices<int> faces = base_topo.faces;
  const Span<int> corner_verts = base_topo.corner_verts;
  const Span<int> corner_edges = base_topo.corner_edges;
  const BitSpan face_edge_reversed = base_topo.face_edge_reversed;
  const int in_size = Multiresolution::grid_size(lvl);
  const int out_size = Multiresolution::grid_size(lvl + 1);
  const int in_size_sq = in_size * in_size;
  const int out_size_sq = out_size * out_size;

  if (lvl == 0) {
    threading::parallel_for(IndexRange(base_topo.faces_num), 1024, [&](const IndexRange range) {
      for (const int face_idx : range) {
        const Span<int> face_verts = corner_verts.slice(faces[face_idx]);
        T sum = T(0.0f);

        for (const int vert : face_verts) {
          sum += in_attr[in_vert_offsets[vert]];
        }

        out_attr[out_offsets[face_idx]] = sum / float(face_verts.size());
      }
    });
    return;
  }

  if (lvl == 1) {
    threading::parallel_for(IndexRange(base_topo.faces_num), 1024, [&](const IndexRange range) {
      for (const int face_idx : range) {
        const T c = in_attr[in_face_offsets[face_idx]];

        const Span<int> face_verts = corner_verts.slice(faces[face_idx]);
        const T v_bl = in_attr[in_vert_offsets[face_verts[0]]];
        const T v_br = in_attr[in_vert_offsets[face_verts[1]]];
        const T v_tr = in_attr[in_vert_offsets[face_verts[2]]];
        const T v_tl = in_attr[in_vert_offsets[face_verts[3]]];

        const Span<int> face_edges = corner_edges.slice(faces[face_idx]);
        const T v_b = in_attr[in_edge_offsets[face_edges[0]]];
        const T v_r = in_attr[in_edge_offsets[face_edges[1]]];
        const T v_t = in_attr[in_edge_offsets[face_edges[2]]];
        const T v_l = in_attr[in_edge_offsets[face_edges[3]]];

        const MutableSpan<T> out_grid = out_attr.slice(out_offsets[face_idx], 9);

        out_grid[0] = (v_bl + v_b + v_l + c) * 0.25f;
        out_grid[1] = (v_b + c) * 0.5f;
        out_grid[2] = (v_b + v_br + c + v_r) * 0.25f;
        out_grid[3] = (v_l + c) * 0.5f;
        out_grid[4] = c;
        out_grid[5] = (c + v_r) * 0.5f;
        out_grid[6] = (v_l + c + v_tl + v_t) * 0.25f;
        out_grid[7] = (c + v_t) * 0.5f;
        out_grid[8] = (c + v_r + v_t + v_tr) * 0.25f;
      }
    });

    return;
  }

  const int grain_size = std::max(1, 1024 / out_size_sq);

  threading::parallel_for(
      IndexRange(base_topo.faces_num), grain_size, [&](const IndexRange range) {
        Array<T> bot_border(in_size + 2);
        Array<T> top_border(in_size + 2);
        Array<T> left_border(in_size + 2);
        Array<T> right_border(in_size + 2);

        for (const int face_idx : range) {
          Span<T> grid = in_attr.slice(in_face_offsets[face_idx], in_size_sq);
          MutableSpan<T> out_grid = out_attr.slice(out_offsets[face_idx], out_size_sq);
          const IndexRange face_range = faces[face_idx];

          const Span<int> face_verts = corner_verts.slice(face_range);
          const T v_bl = in_attr[in_vert_offsets[face_verts[0]]];
          const T v_br = in_attr[in_vert_offsets[face_verts[1]]];
          const T v_tr = in_attr[in_vert_offsets[face_verts[2]]];
          const T v_tl = in_attr[in_vert_offsets[face_verts[3]]];

          const int corner_offset = face_range.start();
          const Span<int> face_edges = corner_edges.slice(face_range);

          fill_border(v_bl,
                      v_br,
                      face_edge_reversed[corner_offset + 0],
                      in_attr.slice(in_edge_offsets[face_edges[0]], in_size),
                      bot_border.as_mutable_span());
          fill_border(v_br,
                      v_tr,
                      face_edge_reversed[corner_offset + 1],
                      in_attr.slice(in_edge_offsets[face_edges[1]], in_size),
                      right_border.as_mutable_span());
          fill_border(v_tl,
                      v_tr,
                      !face_edge_reversed[corner_offset + 2],
                      in_attr.slice(in_edge_offsets[face_edges[2]], in_size),
                      top_border.as_mutable_span());
          fill_border(v_bl,
                      v_tl,
                      !face_edge_reversed[corner_offset + 3],
                      in_attr.slice(in_edge_offsets[face_edges[3]], in_size),
                      left_border.as_mutable_span());

          for (int row = 0, row_offset = 0; row < in_size; row++, row_offset += in_size) {
            Span<T> in1 = (row == 0) ? bot_border.as_span().slice(1, in_size) :
                                       grid.slice(row_offset - in_size, in_size);
            Span<T> in2 = grid.slice(row_offset, in_size);

            MutableSpan<T> out_even = out_grid.slice(2 * row * out_size, out_size);
            MutableSpan<T> out_odd = out_grid.slice((2 * row + 1) * out_size, out_size);

            run_row_linear(in_size,
                           left_border[row],
                           left_border[row + 1],
                           right_border[row],
                           right_border[row + 1],
                           in1,
                           in2,
                           out_even,
                           out_odd);
          }

          Span<T> in_last0 = grid.slice((in_size - 1) * in_size, in_size);
          Span<T> in_last1 = top_border.as_span().slice(1, in_size);
          MutableSpan<T> out_last = out_grid.slice(2 * in_size * out_size, out_size);

          T bl = left_border[in_size];
          T tl = left_border[in_size + 1];

          for (int col = 0; col < in_size; col++) {
            const T br = in_last0[col];
            const T tr = in_last1[col];

            out_last[2 * col] = (bl + br + tl + tr) * 0.25f;
            out_last[2 * col + 1] = (br + tr) * 0.5f;

            bl = br;
            tl = tr;
          }

          out_last[2 * in_size] = (bl + right_border[in_size] + tl + right_border[in_size + 1]) *
                                  0.25f;
        }
      });
}

template<typename T, typename InPolicy, typename OutPolicy>
void subdivide_borders_linear(const int lvl,
                              const bke::multiresolution::BaseTopology &base_topo,
                              const InPolicy in_vert_offsets,
                              const InPolicy in_edge_offsets,
                              const OutPolicy out_offsets,
                              const Span<T> in_attr,
                              const MutableSpan<T> out_attr)
{
  const Span<int2> edges = base_topo.edges;
  const int size = Multiresolution::grid_size(lvl + 1);
  const int grain_size = std::max(1, 4096 / size);

  threading::parallel_for(
      IndexRange(base_topo.edges_num), grain_size, [&](const IndexRange range) {
        for (const int edge_idx : range) {
          const int2 edge = edges[edge_idx];
          int in_offset = in_edge_offsets[edge_idx];
          int out_offset = out_offsets[edge_idx];
          const int last = out_offset + size - 1;
          const T A = in_attr[in_vert_offsets[edge[0]]];
          const T B = in_attr[in_vert_offsets[edge[1]]];
          T curr = A;

          while (out_offset < last - 1) {
            const T next = in_attr[in_offset++];
            out_attr[out_offset++] = (curr + next) * 0.5f;
            out_attr[out_offset++] = next;
            curr = next;
          }

          out_attr[out_offset] = (curr + B) * 0.5;
        }
      });
}

template<typename T, typename InPolicy, typename OutPolicy>
void subdivide_base_linear(const bke::multiresolution::BaseTopology &base_topo,
                           const InPolicy in_vert_offsets,
                           const OutPolicy out_offsets,
                           const Span<T> in_attr,
                           const MutableSpan<T> out_attr)
{
  threading::parallel_for(IndexRange(base_topo.verts_num), 8192, [&](const IndexRange range) {
    for (const int i : range) {
      out_attr[out_offsets[i]] = in_attr[in_vert_offsets[i]];
    }
  });
}

template<typename T, typename InPolicy, typename OutPolicy>
void subdivide_linear(const int curr_lvl,
                      const bke::multiresolution::BaseTopology &base_topo,
                      const InPolicy in_vert_offsets,
                      const InPolicy in_edge_offsets,
                      const InPolicy in_face_offsets,
                      const OutPolicy out_vert_offsets,
                      const OutPolicy out_edge_offsets,
                      const OutPolicy out_face_offsets,
                      const Span<T> in_attr,
                      const MutableSpan<T> out_attr)
{
  subdivide_base_linear(base_topo, in_vert_offsets, out_vert_offsets, in_attr, out_attr);

  subdivide_borders_linear(
      curr_lvl, base_topo, in_vert_offsets, in_edge_offsets, out_edge_offsets, in_attr, out_attr);

  subdivide_grids_linear(curr_lvl,
                         base_topo,
                         in_vert_offsets,
                         in_edge_offsets,
                         in_face_offsets,
                         out_face_offsets,
                         in_attr,
                         out_attr);
}

}  // namespace

template<typename T, typename InputPolicy, typename OutputPolicy>
static void subdivide_chain_same_base(const int from_lvl,
                                      const int to_lvl,
                                      const bke::multiresolution::BaseTopology &base_topo,
                                      const InputPolicy in_vert_offsets,
                                      const InputPolicy in_edge_offsets,
                                      const InputPolicy in_face_offsets,
                                      const OutputPolicy out_vert_offsets,
                                      const OutputPolicy out_edge_offsets,
                                      const OutputPolicy out_face_offsets,
                                      const Span<T> in_attr,
                                      const MutableSpan<T> out_attr,
                                      const bool linear)
{
  BLI_assert(to_lvl > from_lvl);

  auto subdiv = [&](auto &&...args) {
    if (linear)
      return subdivide_linear(std::forward<decltype(args)>(args)...);
    else
      return subdivide(std::forward<decltype(args)>(args)...);
  };

  if (to_lvl == from_lvl + 1) {
    subdiv(from_lvl,
           base_topo,
           in_vert_offsets,
           in_edge_offsets,
           in_face_offsets,
           out_vert_offsets,
           out_edge_offsets,
           out_face_offsets,
           in_attr,
           out_attr);
    return;
  }

  Array<T> buffer_a(bke::multiresolution::calc_total_verts(from_lvl + 1,
                                                           base_topo.verts_num,
                                                           base_topo.edges_num,
                                                           base_topo.faces_num,
                                                           base_topo.corners_num));
  Array<T> buffer_b;

  {
    LinearOffset out_vp, out_ep, out_fp;
    make_linear_policies(from_lvl + 1, base_topo, out_vp, out_ep, out_fp);
    subdiv(from_lvl,
           base_topo,
           in_vert_offsets,
           in_edge_offsets,
           in_face_offsets,
           out_vp,
           out_ep,
           out_fp,
           in_attr,
           buffer_a.as_mutable_span());
  }

  for (int lvl = from_lvl + 1; lvl < to_lvl - 1; lvl++) {
    buffer_b.reinitialize(bke::multiresolution::calc_total_verts(lvl + 1,
                                                                 base_topo.verts_num,
                                                                 base_topo.edges_num,
                                                                 base_topo.faces_num,
                                                                 base_topo.corners_num));

    LinearOffset in_vp, in_ep, in_fp;
    make_linear_policies(lvl, base_topo, in_vp, in_ep, in_fp);

    LinearOffset out_vp, out_ep, out_fp;
    make_linear_policies(lvl + 1, base_topo, out_vp, out_ep, out_fp);

    subdiv(lvl,
           base_topo,
           in_vp,
           in_ep,
           in_fp,
           out_vp,
           out_ep,
           out_fp,
           buffer_a.as_span(),
           buffer_b.as_mutable_span());
    std::swap(buffer_a, buffer_b);
  }

  {
    const int last_lvl = to_lvl - 1;
    LinearOffset in_vp, in_ep, in_fp;
    make_linear_policies(last_lvl, base_topo, in_vp, in_ep, in_fp);

    subdiv(last_lvl,
           base_topo,
           in_vp,
           in_ep,
           in_fp,
           out_vert_offsets,
           out_edge_offsets,
           out_face_offsets,
           buffer_a.as_span(),
           out_attr);
  }
}

template<typename T, typename InputPolicy, typename OutputPolicy>
static void subdivide_to_internal(const bke::multiresolution::BaseTopology &real_base_topo,
                                  const int from_lvl,
                                  const int from_base_lvl,
                                  const bke::multiresolution::BaseTopology &from_base_topo,
                                  const InputPolicy from_vert_offsets,
                                  const InputPolicy from_edge_offsets,
                                  const InputPolicy from_face_offsets,
                                  const Span<T> from_attr,
                                  const int to_lvl,
                                  const int to_base_lvl,
                                  const bke::multiresolution::BaseTopology &to_base_topo,
                                  const OutputPolicy to_vert_offsets,
                                  const OutputPolicy to_edge_offsets,
                                  const OutputPolicy to_face_offsets,
                                  const MutableSpan<T> to_attr,
                                  const bool linear)
{
  if (from_lvl == to_lvl) {
    return;
  }

  BLI_assert(from_lvl >= 0);
  BLI_assert(from_lvl < to_lvl);
  BLI_assert(from_base_lvl <= to_base_lvl);

  const int delta_from = from_lvl - from_base_lvl;
  const int delta_to = to_lvl - to_base_lvl;

  if (from_base_lvl == to_base_lvl) {
    subdivide_chain_same_base<T>(delta_from,
                                 delta_to,
                                 from_base_topo,
                                 from_vert_offsets,
                                 from_edge_offsets,
                                 from_face_offsets,
                                 to_vert_offsets,
                                 to_edge_offsets,
                                 to_face_offsets,
                                 from_attr,
                                 to_attr,
                                 linear);
    return;
  }

  const int pivot_lvl = math::max(from_lvl, to_base_lvl);
  const int delta_pivot_from = pivot_lvl - from_base_lvl;
  const int delta_pivot_to = pivot_lvl - to_base_lvl;

  Array<T> pivot_attr;
  Span<T> pivot_attr_span;
  LinearOffset from_pivot_vp, from_pivot_ep, from_pivot_fp;
  make_linear_policies(
      delta_pivot_from, from_base_topo, from_pivot_vp, from_pivot_ep, from_pivot_fp);

  if (from_lvl < pivot_lvl) {

    pivot_attr.reinitialize(bke::multiresolution::calc_total_verts(delta_pivot_from,
                                                                   from_base_topo.verts_num,
                                                                   from_base_topo.edges_num,
                                                                   from_base_topo.faces_num,
                                                                   from_base_topo.corners_num));

    subdivide_chain_same_base<T>(delta_from,
                                 delta_pivot_from,
                                 from_base_topo,
                                 from_vert_offsets,
                                 from_edge_offsets,
                                 from_face_offsets,
                                 from_pivot_vp,
                                 from_pivot_ep,
                                 from_pivot_fp,
                                 from_attr,
                                 pivot_attr.as_mutable_span(),
                                 linear);

    pivot_attr_span = pivot_attr;
  }
  else {
    pivot_attr_span = from_attr;
  }

  LinearOffset to_pivot_vp, to_pivot_ep, to_pivot_fp;
  make_linear_policies(delta_pivot_to, to_base_topo, to_pivot_vp, to_pivot_ep, to_pivot_fp);

  Array<T> pivot_attr_rebased;
  pivot_attr_rebased.reinitialize(
      bke::multiresolution::calc_total_verts(delta_pivot_to,
                                             to_base_topo.verts_num,
                                             to_base_topo.edges_num,
                                             to_base_topo.faces_num,
                                             to_base_topo.corners_num));

  auto rebase = [&](auto coarse_vp,
                    auto coarse_ep,
                    auto coarse_fp,
                    auto fine_vp,
                    auto fine_ep,
                    auto fine_fp) {
    rebase_attribute_internal<T, true>(real_base_topo,
                                       pivot_lvl,
                                       from_base_lvl,
                                       to_base_lvl,
                                       coarse_vp,
                                       coarse_ep,
                                       coarse_fp,
                                       fine_vp,
                                       fine_ep,
                                       fine_fp,
                                       pivot_attr_span,
                                       pivot_attr_rebased.as_mutable_span());
  };

  if (from_lvl < pivot_lvl) {
    rebase(from_pivot_vp, from_pivot_ep, from_pivot_fp, to_pivot_vp, to_pivot_ep, to_pivot_fp);
  }
  else {
    rebase(from_vert_offsets,
           from_edge_offsets,
           from_face_offsets,
           to_pivot_vp,
           to_pivot_ep,
           to_pivot_fp);
  }

  subdivide_chain_same_base<T>(delta_pivot_to,
                               delta_to,
                               to_base_topo,
                               to_pivot_vp,
                               to_pivot_ep,
                               to_pivot_fp,
                               to_vert_offsets,
                               to_edge_offsets,
                               to_face_offsets,
                               pivot_attr_rebased.as_span(),
                               to_attr,
                               linear);
}

template<typename T>
void Multiresolution::subdivide_to(const Mesh &base_mesh,
                                   const int from_lvl,
                                   const OffsetsSpan from_offsets,
                                   const Span<T> from_attr,
                                   const int to_lvl,
                                   const OffsetsSpan to_offsets,
                                   const MutableSpan<T> to_attr,
                                   const bool linear) const
{
  if (from_lvl == to_lvl) {
    return;
  }

  BLI_assert(from_lvl > 0);
  BLI_assert(from_lvl < to_lvl);

  const bool all_quads = this->base_all_quads;

  if (!all_quads && from_lvl == 1) {
    const Mesh &base_1 = this->runtime->ensure_base_mesh(base_mesh, 1, all_quads);
    const int to_base_lvl = this->base_level(to_lvl);
    const bke::multiresolution::BaseTopology to_base_topo = this->get_base_topology(base_mesh,
                                                                                    to_lvl);

    LinearOffset b0_vp, b0_ep, b0_fp;
    bke::multiresolution::make_linear_policies(1, base_mesh, b0_vp, b0_ep, b0_fp);

    Array<T> base1_attr(base_1.verts_num);
    convert_table_to_linear(1,
                            base_mesh,
                            b0_vp,
                            b0_ep,
                            b0_fp,
                            from_offsets.verts,
                            from_offsets.edges,
                            from_offsets.faces,
                            from_attr,
                            base1_attr.as_mutable_span());

    LinearOffset base1_vp, base1_ep, base1_fp;
    bke::multiresolution::make_linear_policies(0, base_1, base1_vp, base1_ep, base1_fp);

    subdivide_to_internal<T>(base_1,
                             0,
                             0,
                             base_1,
                             base1_vp,
                             base1_ep,
                             base1_fp,
                             base1_attr.as_span(),
                             to_lvl - 1,
                             to_base_lvl - 1,
                             to_base_topo,
                             TableOffset{to_offsets.verts},
                             TableOffset{to_offsets.edges},
                             TableOffset{to_offsets.faces},
                             to_attr,
                             linear);
    return;
  }

  const int shift = all_quads ? 0 : 1;
  const Mesh &eff_base_mesh = all_quads ? base_mesh :
                                          this->runtime->ensure_base_mesh(base_mesh, 1, all_quads);
  const int from_base_lvl = this->base_level(from_lvl);
  const int to_base_lvl = this->base_level(to_lvl);
  const bke::multiresolution::BaseTopology from_base_topo = this->get_base_topology(base_mesh,
                                                                                    from_lvl);
  const bke::multiresolution::BaseTopology to_base_topo = this->get_base_topology(base_mesh,
                                                                                  to_lvl);

  subdivide_to_internal<T>(eff_base_mesh,
                           from_lvl - shift,
                           from_base_lvl - shift,
                           from_base_topo,
                           TableOffset{from_offsets.verts},
                           TableOffset{from_offsets.edges},
                           TableOffset{from_offsets.faces},
                           from_attr,
                           to_lvl - shift,
                           to_base_lvl - shift,
                           to_base_topo,
                           TableOffset{to_offsets.verts},
                           TableOffset{to_offsets.edges},
                           TableOffset{to_offsets.faces},
                           to_attr,
                           linear);
}

template void Multiresolution::subdivide_to<float>(const Mesh &,
                                                   const int,
                                                   const OffsetsSpan,
                                                   const Span<float>,
                                                   const int,
                                                   const OffsetsSpan,
                                                   const MutableSpan<float>,
                                                   const bool) const;

template void Multiresolution::subdivide_to<float3>(const Mesh &,
                                                    const int,
                                                    const OffsetsSpan,
                                                    const Span<float3>,
                                                    const int,
                                                    const OffsetsSpan,
                                                    const MutableSpan<float3>,
                                                    const bool) const;

template void Multiresolution::subdivide_to<float4>(const Mesh &,
                                                    const int,
                                                    const OffsetsSpan,
                                                    const Span<float4>,
                                                    const int,
                                                    const OffsetsSpan,
                                                    const MutableSpan<float4>,
                                                    const bool) const;

template<typename T>
static Array<T> subdivide_base_to_lvl1_linear(const Mesh &base_mesh,
                                              const int base1_verts_num,
                                              const Span<T> in_attr,
                                              const bool linear)
{
  Array<T> base1_attr(base1_verts_num);
  LinearOffset in_vp{0, 1}, in_ep{0, 0}, in_fp{0, 0};
  LinearOffset base1_vp, base1_ep, base1_fp;
  bke::multiresolution::make_linear_policies(1, base_mesh, base1_vp, base1_ep, base1_fp);
  subdivide_chain_same_base<T>(0,
                               1,
                               base_mesh,
                               in_vp,
                               in_ep,
                               in_fp,
                               base1_vp,
                               base1_ep,
                               base1_fp,
                               in_attr,
                               base1_attr.as_mutable_span(),
                               linear);
  return base1_attr;
}

template<typename T>
void Multiresolution::subdivide_base_to(const Mesh &base_mesh,
                                        const int to_lvl,
                                        const OffsetsSpan to_offsets,
                                        const Span<T> in_attr,
                                        const MutableSpan<T> out_attr,
                                        const bool linear) const
{
  BLI_assert(to_lvl > 0);

  const bool all_quads = this->base_all_quads;
  const int to_base_lvl = this->base_level(to_lvl);
  const bke::multiresolution::BaseTopology to_base_topo = this->get_base_topology(base_mesh,
                                                                                  to_lvl);

  if (all_quads || to_lvl == 1) {
    LinearOffset in_vp{0, 1}, in_ep{0, 0}, in_fp{0, 0};

    subdivide_to_internal<T>(base_mesh,
                             0,
                             0,
                             base_mesh,
                             in_vp,
                             in_ep,
                             in_fp,
                             in_attr,
                             to_lvl,
                             to_base_lvl,
                             to_base_topo,
                             TableOffset{to_offsets.verts},
                             TableOffset{to_offsets.edges},
                             TableOffset{to_offsets.faces},
                             out_attr,
                             linear);
    return;
  }

  const Mesh &base_1 = this->runtime->ensure_base_mesh(base_mesh, 1, all_quads);
  const Array<T> base1_attr = subdivide_base_to_lvl1_linear<T>(
      base_mesh, base_1.verts_num, in_attr, linear);

  LinearOffset b1_in_vp, b1_in_ep, b1_in_fp;
  bke::multiresolution::make_linear_policies(0, base_1, b1_in_vp, b1_in_ep, b1_in_fp);

  subdivide_to_internal<T>(base_1,
                           0,
                           0,
                           base_1,
                           b1_in_vp,
                           b1_in_ep,
                           b1_in_fp,
                           base1_attr.as_span(),
                           to_lvl - 1,
                           to_base_lvl - 1,
                           to_base_topo,
                           TableOffset{to_offsets.verts},
                           TableOffset{to_offsets.edges},
                           TableOffset{to_offsets.faces},
                           out_attr,
                           linear);
}

template void Multiresolution::subdivide_base_to<float>(const Mesh &,
                                                        const int,
                                                        const OffsetsSpan,
                                                        const Span<float>,
                                                        const MutableSpan<float>,
                                                        const bool) const;

template void Multiresolution::subdivide_base_to<float3>(const Mesh &,
                                                         const int,
                                                         const OffsetsSpan,
                                                         const Span<float3>,
                                                         const MutableSpan<float3>,
                                                         const bool) const;

template void Multiresolution::subdivide_base_to<float4>(const Mesh &,
                                                         const int,
                                                         const OffsetsSpan,
                                                         const Span<float4>,
                                                         const MutableSpan<float4>,
                                                         const bool) const;

template<typename T, typename InputPolicy, typename OutputPolicy>
static void downsample(const bke::multiresolution::BaseTopology &base_topo,
                       const int from_lvl,
                       const InputPolicy from_vert_offsets,
                       const InputPolicy from_edge_offsets,
                       const InputPolicy from_face_offsets,
                       const Span<T> from_attr,
                       const int to_lvl,
                       const OutputPolicy to_vert_offsets,
                       const OutputPolicy to_edge_offsets,
                       const OutputPolicy to_face_offsets,
                       const MutableSpan<T> to_attr)
{
  BLI_assert(to_lvl < from_lvl);
  BLI_assert(to_lvl >= 0);

  const int to_grid_size = Multiresolution::grid_size(to_lvl);
  const int from_grid_size = Multiresolution::grid_size(from_lvl);

  constexpr int total = 8192;
  const int vert_grain = total;

  threading::parallel_for(
      IndexRange(base_topo.verts_num), vert_grain, [&](const IndexRange range) {
        for (const int vert : range) {
          to_attr[to_vert_offsets[vert]] = from_attr[from_vert_offsets[vert]];
        }
      });

  if (to_grid_size == 0) {
    return;
  }

  const int edge_grain = math::max(1, total / to_grid_size);
  const int edge_step = 1 << (from_lvl - to_lvl);
  threading::parallel_for(
      IndexRange(base_topo.edges_num), edge_grain, [&](const IndexRange range) {
        for (const int edge : range) {
          int to_offset = to_edge_offsets[edge];
          int from_offset = from_edge_offsets[edge] + edge_step - 1;

          for ([[maybe_unused]] const int i : IndexRange(to_grid_size)) {
            to_attr[to_offset++] = from_attr[from_offset];
            from_offset += edge_step;
          }
        }
      });

  const int face_step = 1 << (from_lvl - to_lvl);
  const int row_advance = face_step * (from_grid_size - to_grid_size);
  const int face_grain = math::max(1, total / (to_grid_size * to_grid_size));
  threading::parallel_for(
      IndexRange(base_topo.faces_num), face_grain, [&](const IndexRange range) {
        for (const int face : range) {
          int to_offset = to_face_offsets[face];
          int from_offset = from_face_offsets[face] + (face_step - 1) * from_grid_size +
                            (face_step - 1);
          for (int row = 0; row < to_grid_size; row++) {
            for (int col = 0; col < to_grid_size; col++) {
              to_attr[to_offset++] = from_attr[from_offset];
              from_offset += face_step;
            }
            from_offset += row_advance;
          }
        }
      });
}

template<typename T, typename InputPolicy, typename OutputPolicy>
static void downsample_to_internal(const bke::multiresolution::BaseTopology &real_base_topo,
                                   const int from_lvl,
                                   const int from_base_lvl,
                                   const bke::multiresolution::BaseTopology &from_base_topo,
                                   const InputPolicy from_vert_offsets,
                                   const InputPolicy from_edge_offsets,
                                   const InputPolicy from_face_offsets,
                                   const Span<T> from_attr,
                                   const int to_lvl,
                                   const int to_base_lvl,
                                   const bke::multiresolution::BaseTopology &to_base_topo,
                                   const OutputPolicy to_vert_offsets,
                                   const OutputPolicy to_edge_offsets,
                                   const OutputPolicy to_face_offsets,
                                   const MutableSpan<T> to_attr)
{
  if (from_lvl == to_lvl) {
    return;
  }

  BLI_assert(from_lvl > to_lvl);
  BLI_assert(to_lvl >= 0);
  BLI_assert(from_base_lvl >= to_base_lvl);

  const int delta_from = from_lvl - from_base_lvl;
  const int delta_to = to_lvl - to_base_lvl;

  if (from_base_lvl == to_base_lvl) {
    downsample(from_base_topo,
               delta_from,
               from_vert_offsets,
               from_edge_offsets,
               from_face_offsets,
               from_attr,
               delta_to,
               to_vert_offsets,
               to_edge_offsets,
               to_face_offsets,
               to_attr);
    return;
  }

  const int pivot_lvl = math::max(to_lvl, from_base_lvl);
  const int delta_pivot_from = pivot_lvl - from_base_lvl;
  const int delta_pivot_to = pivot_lvl - to_base_lvl;

  Array<T> pivot_attr;
  Span<T> pivot_attr_span;
  LinearOffset from_pivot_vp, from_pivot_ep, from_pivot_fp;
  make_linear_policies(
      delta_pivot_from, from_base_topo, from_pivot_vp, from_pivot_ep, from_pivot_fp);

  if (from_lvl > pivot_lvl) {
    pivot_attr.reinitialize(bke::multiresolution::calc_total_verts(delta_pivot_from,
                                                                   from_base_topo.verts_num,
                                                                   from_base_topo.edges_num,
                                                                   from_base_topo.faces_num,
                                                                   from_base_topo.corners_num));

    downsample<T>(from_base_topo,
                  delta_from,
                  from_vert_offsets,
                  from_edge_offsets,
                  from_face_offsets,
                  from_attr,
                  delta_pivot_from,
                  from_pivot_vp,
                  from_pivot_ep,
                  from_pivot_fp,
                  pivot_attr.as_mutable_span());

    pivot_attr_span = pivot_attr;
  }
  else {
    pivot_attr_span = from_attr;
  }

  LinearOffset to_pivot_vp, to_pivot_ep, to_pivot_fp;
  make_linear_policies(delta_pivot_to, to_base_topo, to_pivot_vp, to_pivot_ep, to_pivot_fp);

  auto rebase = [&](auto coarse_vp,
                    auto coarse_ep,
                    auto coarse_fp,
                    auto fine_vp,
                    auto fine_ep,
                    auto fine_fp,
                    MutableSpan<T> out_dst) {
    rebase_attribute_internal<T, false>(real_base_topo,
                                        pivot_lvl,
                                        to_base_lvl,
                                        from_base_lvl,
                                        coarse_vp,
                                        coarse_ep,
                                        coarse_fp,
                                        fine_vp,
                                        fine_ep,
                                        fine_fp,
                                        pivot_attr_span,
                                        out_dst);
  };

  if (to_lvl == pivot_lvl) {
    if (from_lvl > pivot_lvl) {
      rebase(to_vert_offsets,
             to_edge_offsets,
             to_face_offsets,
             from_pivot_vp,
             from_pivot_ep,
             from_pivot_fp,
             to_attr);
    }
    else {
      rebase(to_vert_offsets,
             to_edge_offsets,
             to_face_offsets,
             from_vert_offsets,
             from_edge_offsets,
             from_face_offsets,
             to_attr);
    }
  }
  else {
    Array<T> pivot_attr_rebased(bke::multiresolution::calc_total_verts(delta_pivot_to,
                                                                       to_base_topo.verts_num,
                                                                       to_base_topo.edges_num,
                                                                       to_base_topo.faces_num,
                                                                       to_base_topo.corners_num));

    if (from_lvl > pivot_lvl) {
      rebase(to_pivot_vp,
             to_pivot_ep,
             to_pivot_fp,
             from_pivot_vp,
             from_pivot_ep,
             from_pivot_fp,
             pivot_attr_rebased.as_mutable_span());
    }
    else {
      rebase(to_pivot_vp,
             to_pivot_ep,
             to_pivot_fp,
             from_vert_offsets,
             from_edge_offsets,
             from_face_offsets,
             pivot_attr_rebased.as_mutable_span());
    }

    downsample<T>(to_base_topo,
                  delta_pivot_to,
                  to_pivot_vp,
                  to_pivot_ep,
                  to_pivot_fp,
                  pivot_attr_rebased.as_span(),
                  delta_to,
                  to_vert_offsets,
                  to_edge_offsets,
                  to_face_offsets,
                  to_attr);
  }
}

template<typename T>
void Multiresolution::downsample_to(const Mesh &base_mesh,
                                    const int from_lvl,
                                    const OffsetsSpan from_offsets,
                                    const Span<T> from_attr,
                                    const int to_lvl,
                                    const OffsetsSpan to_offsets,
                                    const MutableSpan<T> to_attr) const
{
  if (from_lvl == to_lvl) {
    return;
  }

  BLI_assert(to_lvl > 0);
  BLI_assert(from_lvl > to_lvl);

  const bool all_quads = this->base_all_quads;

  if (!all_quads && to_lvl == 1) {
    const int verts_num = base_mesh.verts_num;
    const int edges_num = base_mesh.edges_num;
    const int faces_num = base_mesh.faces_num;
    threading::parallel_for(IndexRange(verts_num), 8192, [&](const IndexRange range) {
      for (const int v : range) {
        to_attr[to_offsets.verts[v]] = from_attr[from_offsets.verts[v]];
      }
    });
    threading::parallel_for(IndexRange(edges_num), 8192, [&](const IndexRange range) {
      for (const int e : range) {
        to_attr[to_offsets.edges[e]] = from_attr[from_offsets.verts[verts_num + e]];
      }
    });
    threading::parallel_for(IndexRange(faces_num), 8192, [&](const IndexRange range) {
      for (const int f : range) {
        to_attr[to_offsets.faces[f]] = from_attr[from_offsets.verts[verts_num + edges_num + f]];
      }
    });
    return;
  }

  const int shift = all_quads ? 0 : 1;
  const Mesh &eff_base_mesh = all_quads ? base_mesh :
                                          this->runtime->ensure_base_mesh(base_mesh, 1, all_quads);
  const int from_base_lvl = this->base_level(from_lvl);
  const int to_base_lvl = this->base_level(to_lvl);
  const bke::multiresolution::BaseTopology from_base_topo = this->get_base_topology(base_mesh,
                                                                                    from_lvl);
  const bke::multiresolution::BaseTopology to_base_topo = this->get_base_topology(base_mesh,
                                                                                  to_lvl);

  downsample_to_internal<T>(eff_base_mesh,
                            from_lvl - shift,
                            from_base_lvl - shift,
                            from_base_topo,
                            TableOffset{from_offsets.verts},
                            TableOffset{from_offsets.edges},
                            TableOffset{from_offsets.faces},
                            from_attr,
                            to_lvl - shift,
                            to_base_lvl - shift,
                            to_base_topo,
                            TableOffset{to_offsets.verts},
                            TableOffset{to_offsets.edges},
                            TableOffset{to_offsets.faces},
                            to_attr);
}

template void Multiresolution::downsample_to<float>(const Mesh &,
                                                    const int,
                                                    const OffsetsSpan,
                                                    const Span<float>,
                                                    const int,
                                                    const OffsetsSpan,
                                                    const MutableSpan<float>) const;

template void Multiresolution::downsample_to<float3>(const Mesh &,
                                                     const int,
                                                     const OffsetsSpan,
                                                     const Span<float3>,
                                                     const int,
                                                     const OffsetsSpan,
                                                     const MutableSpan<float3>) const;

template void Multiresolution::downsample_to<float4>(const Mesh &,
                                                     const int,
                                                     const OffsetsSpan,
                                                     const Span<float4>,
                                                     const int,
                                                     const OffsetsSpan,
                                                     const MutableSpan<float4>) const;

template void Multiresolution::downsample_to<ColorGeometry4f>(
    const Mesh &,
    const int,
    const OffsetsSpan,
    const Span<ColorGeometry4f>,
    const int,
    const OffsetsSpan,
    const MutableSpan<ColorGeometry4f>) const;

template void Multiresolution::downsample_to<int>(const Mesh &,
                                                  const int,
                                                  const OffsetsSpan,
                                                  const Span<int>,
                                                  const int,
                                                  const OffsetsSpan,
                                                  const MutableSpan<int>) const;

template<typename T>
void Multiresolution::downsample_to_base(const Span<int> from_vert_offsets,
                                         const Span<T> from_attr,
                                         const MutableSpan<T> to_attr)
{
  threading::parallel_for(to_attr.index_range(), 8192, [&](const IndexRange range) {
    for (const int vert : range) {
      to_attr[vert] = from_attr[from_vert_offsets[vert]];
    }
  });
}

template void Multiresolution::downsample_to_base<float>(const Span<int>,
                                                         const Span<float>,
                                                         const MutableSpan<float>);

template void Multiresolution::downsample_to_base<float3>(const Span<int>,
                                                          const Span<float3>,
                                                          const MutableSpan<float3>);

template void Multiresolution::downsample_to_base<float4>(const Span<int>,
                                                          const Span<float4>,
                                                          const MutableSpan<float4>);

template void Multiresolution::downsample_to_base<ColorGeometry4f>(
    const Span<int>, const Span<ColorGeometry4f>, const MutableSpan<ColorGeometry4f>);

Offsets Multiresolution::build_table_offsets(const Multiresolution &multires,
                                             const bke::multiresolution::BaseTopology &base_topo,
                                             const int lvl)
{
  auto atomic_min = [&](std::atomic<int> &a, int value) {
    int curr = a.load(std::memory_order_relaxed);
    while (value < curr && !a.compare_exchange_weak(curr, value, std::memory_order_relaxed)) {
    }
  };

  const Span<int> bvh_face_indices = multires.bvh_face_indices(lvl);
  const Span<int> bvh_face_counts = multires.bvh_face_counts(lvl);
  const int verts_num = base_topo.verts_num;
  const int edges_num = base_topo.edges_num;
  const OffsetIndices<int> faces = base_topo.faces;
  const Span<int> corner_verts = base_topo.corner_verts;
  const Span<int> corner_edges = base_topo.corner_edges;
  const int nodes_num = bvh_face_counts.size();

  Array<int> leaf_offsets(nodes_num);
  {
    int curr_offset = 0;
    int next_child_offset = 1;
    bke::multiresolution::calc_leaf_face_offsets_recursive(
        0, curr_offset, next_child_offset, bvh_face_counts, leaf_offsets);
  }

  auto node_faces = [&](const int i) -> Span<int> {
    if (bvh_face_counts[i] <= 0) {
      return {};
    }
    return bvh_face_indices.slice(leaf_offsets[i], bvh_face_counts[i]);
  };

  Array<std::atomic<int>> vert_owner(verts_num);
  threading::memory_bandwidth_bound_task(vert_owner.as_span().size_in_bytes(), [&]() {
    threading::parallel_for(IndexRange(verts_num), 8192, [&](const IndexRange range) {
      for (int i : range) {
        vert_owner[i].store(INT_MAX, std::memory_order_relaxed);
      }
    });
  });

  Array<std::atomic<int>> edge_owner(edges_num);
  threading::memory_bandwidth_bound_task(edge_owner.as_span().size_in_bytes(), [&]() {
    threading::parallel_for(IndexRange(edges_num), 8192, [&](const IndexRange range) {
      for (int i : range) {
        edge_owner[i].store(INT_MAX, std::memory_order_relaxed);
      }
    });
  });

  Array<Set<int>> unique_verts(nodes_num);
  Array<Set<int>> unique_edges(nodes_num);
  threading::parallel_for(IndexRange(nodes_num), 8, [&](const IndexRange range) {
    for (const int i : range) {
      const Span<int> node_faces_i = node_faces(i);
      if (node_faces_i.is_empty()) {
        continue;
      }
      for (const int face_idx : node_faces_i) {
        const IndexRange face = faces[face_idx];
        unique_verts[i].add_multiple(corner_verts.slice(face));
        unique_edges[i].add_multiple(corner_edges.slice(face));
      }
      for (const int vert : unique_verts[i]) {
        atomic_min(vert_owner[vert], i);
      }
      for (const int edge : unique_edges[i]) {
        atomic_min(edge_owner[edge], i);
      }
    }
  });

  const int base_lvl = multires.base_level(lvl);
  const int virtual_lvl = lvl - base_lvl;
  const int verts_per_edge = Multiresolution::grid_size(virtual_lvl);
  const int verts_per_face = verts_per_edge * verts_per_edge;

  Array<int> node_position_offset(nodes_num);
  int global_offset = 0;
  Array<Vector<int>> owned_verts(nodes_num);
  Array<Vector<int>> owned_edges(nodes_num);

  for (int i = 0; i < nodes_num; i++) {
    const Span<int> node_faces_i = node_faces(i);
    if (node_faces_i.is_empty()) {
      node_position_offset[i] = global_offset;
      continue;
    }
    for (const int vert : unique_verts[i]) {
      if (vert_owner[vert].load(std::memory_order_relaxed) == i) {
        owned_verts[i].append(vert);
      }
    }
    std::sort(owned_verts[i].begin(), owned_verts[i].end());

    for (const int edge : unique_edges[i]) {
      if (edge_owner[edge].load(std::memory_order_relaxed) == i) {
        owned_edges[i].append(edge);
      }
    }
    std::sort(owned_edges[i].begin(), owned_edges[i].end());

    node_position_offset[i] = global_offset;
    global_offset += owned_verts[i].size() + owned_edges[i].size() * verts_per_edge +
                     node_faces_i.size() * verts_per_face;
  }

  Offsets offsets;
  offsets.verts.reinitialize(base_topo.verts_num);
  offsets.edges.reinitialize(base_topo.edges_num);
  offsets.faces.reinitialize(base_topo.faces_num);

  threading::parallel_for(IndexRange(nodes_num), 8, [&](const IndexRange range) {
    for (const int i : range) {
      const Span<int> node_faces_i = node_faces(i);
      if (node_faces_i.is_empty()) {
        continue;
      }
      int offset = node_position_offset[i];
      for (const int vert_idx : owned_verts[i]) {
        offsets.verts[vert_idx] = offset++;
      }
      for (const int edge_idx : owned_edges[i]) {
        offsets.edges[edge_idx] = offset;
        offset += verts_per_edge;
      }
      for (const int face_idx : node_faces_i) {
        offsets.faces[face_idx] = offset;
        offset += verts_per_face;
      }
    }
  });

  return offsets;
}

template<typename T, bool Forward, typename CoarsePolicy, typename FinePolicy>
static void rebase_attribute_internal(const bke::multiresolution::BaseTopology &real_base,
                                      const int lvl,
                                      const int coarse_base_lvl,
                                      const int fine_base_lvl,
                                      const CoarsePolicy coarse_vert_offsets,
                                      const CoarsePolicy coarse_edge_offsets,
                                      const CoarsePolicy coarse_face_offsets,
                                      const FinePolicy fine_vert_offsets,
                                      const FinePolicy fine_edge_offsets,
                                      const FinePolicy fine_face_offsets,
                                      const Span<T> src_attr,
                                      const MutableSpan<T> dst_attr)
{
  BLI_assert(fine_base_lvl >= coarse_base_lvl);
  BLI_assert(lvl >= fine_base_lvl);

  if (coarse_base_lvl == fine_base_lvl) {
    return;
  }

  const int base_verts_num = real_base.verts_num;
  const int base_edges_num = real_base.edges_num;
  const int base_faces_num = real_base.faces_num;

  struct LevelParams {
    int delta_lvl;
    int step_size;
    int grid_size;
    int faces_per_side;
    int base_grid_size;
    int subedges_per_face;
    int face_edges_start;

    LevelParams(const int lvl, const int base_lvl, const int base_edges_num)
        : delta_lvl(lvl - base_lvl),
          step_size(1 << delta_lvl),
          grid_size(step_size - 1),
          faces_per_side(1 << base_lvl),
          base_grid_size(faces_per_side - 1),
          subedges_per_face(2 * faces_per_side * base_grid_size),
          face_edges_start(base_edges_num * faces_per_side)
    {
    }
  };

  const LevelParams coarse_params(lvl, coarse_base_lvl, base_edges_num);
  const LevelParams fine_params(lvl, fine_base_lvl, base_edges_num);
  const int coarse_faces_per_side = coarse_params.faces_per_side;
  const int fine_faces_per_side = fine_params.faces_per_side;
  const int fine_step = fine_params.step_size;
  const int coarse_grid = coarse_params.grid_size;
  const int fine_grid = fine_params.grid_size;
  const int shift = fine_base_lvl - coarse_base_lvl;
  const int subdiv_mask = (1 << shift) - 1;

  auto get_edge_point_index = [=](const auto &vert_offsets,
                                  const auto &edge_offsets,
                                  const LevelParams &p,
                                  const int base_edge_idx,
                                  const int point_coord) {
    const int subedge_idx = point_coord >> p.delta_lvl;
    const int local_offset = point_coord & (p.step_size - 1);

    if (local_offset == 0) {
      const int base_vert_idx = base_verts_num + base_edge_idx * p.base_grid_size +
                                (subedge_idx - 1);
      return vert_offsets[base_vert_idx];
    }

    const int edge_idx = base_edge_idx * p.faces_per_side + subedge_idx;
    return edge_offsets[edge_idx] + (local_offset - 1);
  };

  auto get_face_point_index = [=](const auto &vert_offsets,
                                  const auto &edge_offsets,
                                  const auto &face_offsets,
                                  const LevelParams &p,
                                  const int base_face_idx,
                                  const int point_row,
                                  const int point_col) {
    const int subface_row = point_row >> p.delta_lvl;
    const int local_row = point_row & (p.step_size - 1);
    const int subface_col = point_col >> p.delta_lvl;
    const int local_col = point_col & (p.step_size - 1);

    if (local_row > 0 && local_col > 0) {
      const int subface_idx = base_face_idx * (p.faces_per_side * p.faces_per_side) +
                              subface_row * p.faces_per_side + subface_col;
      return face_offsets[subface_idx] + (local_row - 1) * p.grid_size + (local_col - 1);
    }

    if (local_row == 0 && local_col == 0) {
      const int subvert_idx = base_verts_num + base_edges_num * p.base_grid_size +
                              base_face_idx * (p.base_grid_size * p.base_grid_size) +
                              (subface_row - 1) * p.base_grid_size + (subface_col - 1);
      return vert_offsets[subvert_idx];
    }

    if (local_row > 0 && local_col == 0) {
      const int subedge_idx = p.face_edges_start + base_face_idx * p.subedges_per_face +
                              subface_row * p.base_grid_size + (subface_col - 1);
      return edge_offsets[subedge_idx] + (local_row - 1);
    }

    const int subedge_idx = p.face_edges_start + base_face_idx * p.subedges_per_face +
                            p.faces_per_side * p.base_grid_size + subface_col * p.base_grid_size +
                            (subface_row - 1);
    return edge_offsets[subedge_idx] + (local_col - 1);
  };

  auto copy_elem = [&](const int coarse_idx, const int fine_idx) {
    if constexpr (Forward) {
      dst_attr[fine_idx] = src_attr[coarse_idx];
    }
    else {
      dst_attr[coarse_idx] = src_attr[fine_idx];
    }
  };

  auto copy_range = [&](const int coarse_base, const int fine_base, const int len) {
    if constexpr (Forward) {
      for (int i = 0; i < len; i++) {
        dst_attr[fine_base + i] = src_attr[coarse_base + i];
      }
    }
    else {
      for (int i = 0; i < len; i++) {
        dst_attr[coarse_base + i] = src_attr[fine_base + i];
      }
    }
  };

  threading::parallel_for(IndexRange(base_verts_num), 8192, [&](const IndexRange range) {
    for (const int base_vert_idx : range) {
      copy_elem(coarse_vert_offsets[base_vert_idx], fine_vert_offsets[base_vert_idx]);
    }
  });

  const int edge_points_num = (1 << lvl) - 1;
  if (edge_points_num > 0) {
    threading::parallel_for(IndexRange(base_edges_num), 1024, [&](const IndexRange range) {
      for (const int base_edge_idx : range) {
        if (fine_grid > 0) {
          for (int sf = 0; sf < fine_faces_per_side; sf++) {
            const int sc = sf >> shift;
            const int k = sf & subdiv_mask;
            const int coarse_base =
                coarse_edge_offsets[base_edge_idx * coarse_faces_per_side + sc];
            const int fine_base = fine_edge_offsets[base_edge_idx * fine_faces_per_side + sf];
            copy_range(coarse_base + k * fine_step, fine_base, fine_grid);
          }
        }
        for (int j = 1; j < fine_faces_per_side; j++) {
          const int point_coord = j * fine_step;
          const int coarse_idx = get_edge_point_index(
              coarse_vert_offsets, coarse_edge_offsets, coarse_params, base_edge_idx, point_coord);
          const int fine_idx = get_edge_point_index(
              fine_vert_offsets, fine_edge_offsets, fine_params, base_edge_idx, point_coord);
          copy_elem(coarse_idx, fine_idx);
        }
      }
    });
  }

  const int face_points_dim = (1 << lvl) - 1;
  if (face_points_dim > 0) {
    const int grain_size = std::max(1, 4096 / (face_points_dim * face_points_dim));
    const int coarse_face_stride = coarse_faces_per_side * coarse_faces_per_side;
    const int fine_face_stride = fine_faces_per_side * fine_faces_per_side;
    threading::parallel_for(IndexRange(base_faces_num), grain_size, [&](const IndexRange range) {
      for (const int base_face_idx : range) {
        if (fine_grid > 0) {
          for (int fr = 0; fr < fine_faces_per_side; fr++) {
            const int scr = fr >> shift;
            const int kr = fr & subdiv_mask;
            for (int fc = 0; fc < fine_faces_per_side; fc++) {
              const int scc = fc >> shift;
              const int kc = fc & subdiv_mask;
              const int coarse_base = coarse_face_offsets[base_face_idx * coarse_face_stride +
                                                          scr * coarse_faces_per_side + scc];
              const int fine_base = fine_face_offsets[base_face_idx * fine_face_stride +
                                                      fr * fine_faces_per_side + fc];
              for (int r = 0; r < fine_grid; r++) {
                const int coarse_row = coarse_base + (kr * fine_step + r) * coarse_grid +
                                       kc * fine_step;
                const int fine_row = fine_base + r * fine_grid;
                copy_range(coarse_row, fine_row, fine_grid);
              }
            }
          }
        }
        for (int jr = 1; jr < fine_faces_per_side; jr++) {
          for (int jc = 1; jc < fine_faces_per_side; jc++) {
            const int coarse_idx = get_face_point_index(coarse_vert_offsets,
                                                        coarse_edge_offsets,
                                                        coarse_face_offsets,
                                                        coarse_params,
                                                        base_face_idx,
                                                        jr * fine_step,
                                                        jc * fine_step);
            const int fine_idx = get_face_point_index(fine_vert_offsets,
                                                      fine_edge_offsets,
                                                      fine_face_offsets,
                                                      fine_params,
                                                      base_face_idx,
                                                      jr * fine_step,
                                                      jc * fine_step);
            copy_elem(coarse_idx, fine_idx);
          }
        }
        if (fine_grid > 0) {
          for (int jr = 0; jr < fine_faces_per_side; jr++) {
            for (int jc = 1; jc < fine_faces_per_side; jc++) {
              for (int r = 1; r <= fine_grid; r++) {
                const int row = jr * fine_step + r;
                const int col = jc * fine_step;
                const int coarse_idx = get_face_point_index(coarse_vert_offsets,
                                                            coarse_edge_offsets,
                                                            coarse_face_offsets,
                                                            coarse_params,
                                                            base_face_idx,
                                                            row,
                                                            col);
                const int fine_idx = get_face_point_index(fine_vert_offsets,
                                                          fine_edge_offsets,
                                                          fine_face_offsets,
                                                          fine_params,
                                                          base_face_idx,
                                                          row,
                                                          col);
                copy_elem(coarse_idx, fine_idx);
              }
            }
          }
          for (int jr = 1; jr < fine_faces_per_side; jr++) {
            for (int jc = 0; jc < fine_faces_per_side; jc++) {
              const int row = jr * fine_step;
              const int col = jc * fine_step + 1;
              const int coarse_base = get_face_point_index(coarse_vert_offsets,
                                                           coarse_edge_offsets,
                                                           coarse_face_offsets,
                                                           coarse_params,
                                                           base_face_idx,
                                                           row,
                                                           col);
              const int fine_base = get_face_point_index(fine_vert_offsets,
                                                         fine_edge_offsets,
                                                         fine_face_offsets,
                                                         fine_params,
                                                         base_face_idx,
                                                         row,
                                                         col);
              copy_range(coarse_base, fine_base, fine_grid);
            }
          }
        }
      }
    });
  }
}

template<typename T>
void Multiresolution::rebase_attribute(const Mesh &base_mesh,
                                       const int lvl,
                                       const int coarse_base_lvl,
                                       const int fine_base_lvl,
                                       const OffsetsSpan coarse_offsets,
                                       const OffsetsSpan fine_offsets,
                                       const Span<T> src_attr,
                                       const MutableSpan<T> dst_attr) const
{
  if (coarse_base_lvl == fine_base_lvl) {
    return;
  }

  if (!this->base_all_quads && fine_base_lvl >= 1 && coarse_base_lvl == 0) {
    const Mesh &base_1 = this->runtime->ensure_base_mesh(base_mesh, 1, this->base_all_quads);
    LinearOffset b0_vp, b0_ep, b0_fp;
    bke::multiresolution::make_linear_policies(1, base_mesh, b0_vp, b0_ep, b0_fp);

    Array<T> base1_attr(src_attr.size());
    convert_table_to_linear(1,
                            base_mesh,
                            b0_vp,
                            b0_ep,
                            b0_fp,
                            coarse_offsets.verts,
                            coarse_offsets.edges,
                            coarse_offsets.faces,
                            src_attr,
                            base1_attr.as_mutable_span());

    LinearOffset base1_vp, base1_ep, base1_fp;
    bke::multiresolution::make_linear_policies(0, base_1, base1_vp, base1_ep, base1_fp);

    rebase_attribute_internal<T, true>(base_1,
                                       lvl - 1,
                                       0,
                                       fine_base_lvl - 1,
                                       base1_vp,
                                       base1_ep,
                                       base1_fp,
                                       TableOffset{fine_offsets.verts},
                                       TableOffset{fine_offsets.edges},
                                       TableOffset{fine_offsets.faces},
                                       base1_attr.as_span(),
                                       dst_attr);
    return;
  }

  const int shift = this->base_all_quads ? 0 : 1;
  const Mesh &eff_base_mesh = this->base_all_quads ? base_mesh :
                                                     this->runtime->ensure_base_mesh(
                                                         base_mesh, 1, this->base_all_quads);

  rebase_attribute_internal<T, true>(eff_base_mesh,
                                     lvl - shift,
                                     coarse_base_lvl - shift,
                                     fine_base_lvl - shift,
                                     TableOffset{coarse_offsets.verts},
                                     TableOffset{coarse_offsets.edges},
                                     TableOffset{coarse_offsets.faces},
                                     TableOffset{fine_offsets.verts},
                                     TableOffset{fine_offsets.edges},
                                     TableOffset{fine_offsets.faces},
                                     src_attr,
                                     dst_attr);
}

template void Multiresolution::rebase_attribute<float>(const Mesh &,
                                                       const int,
                                                       const int,
                                                       const int,
                                                       const OffsetsSpan,
                                                       const OffsetsSpan,
                                                       const Span<float>,
                                                       const MutableSpan<float>) const;

template void Multiresolution::rebase_attribute<float3>(const Mesh &,
                                                        const int,
                                                        const int,
                                                        const int,
                                                        const OffsetsSpan,
                                                        const OffsetsSpan,
                                                        const Span<float3>,
                                                        const MutableSpan<float3>) const;

template void Multiresolution::rebase_attribute<float4>(const Mesh &,
                                                        const int,
                                                        const int,
                                                        const int,
                                                        const OffsetsSpan,
                                                        const OffsetsSpan,
                                                        const Span<float4>,
                                                        const MutableSpan<float4>) const;

template<typename T>
static void rebase_face_attr(const int from_base_faces_num,
                             const int from_base_lvl,
                             const int to_base_lvl,
                             const int src_rel_lvl,
                             const Span<T> from_face_attr,
                             const MutableSpan<T> to_face_attr)
{
  const int lvl_diff = to_base_lvl - from_base_lvl;
  BLI_assert(lvl_diff != 0);

  const int from_base_side = 1 << from_base_lvl;
  const int to_base_side = 1 << to_base_lvl;

  const int from_base_faces_per_root = from_base_side * from_base_side;
  const int to_base_faces_per_root = to_base_side * to_base_side;

  const int src_side = 1 << src_rel_lvl;
  const int dst_rel_lvl = src_rel_lvl - lvl_diff;
  BLI_assert(src_rel_lvl >= 0);
  BLI_assert(dst_rel_lvl >= 0);
  const int dst_side = 1 << dst_rel_lvl;

  const int src_face_cells = src_side * src_side;
  const int dst_face_cells = dst_side * dst_side;

  if (lvl_diff > 0) {
    BLI_assert(src_rel_lvl >= lvl_diff);
    const int block_side = 1 << lvl_diff;
    const int grain = math::max(1, 4096 / math::max(1, dst_face_cells));
    threading::parallel_for(IndexRange(from_base_faces_num), grain, [&](const IndexRange range) {
      for (const int src_face : range) {
        const int root_face = src_face / from_base_faces_per_root;
        const int local_src_face = src_face % from_base_faces_per_root;

        const int from_r = local_src_face / from_base_side;
        const int from_c = local_src_face % from_base_side;

        const int src_offset = src_face * src_face_cells;
        const int root_dst_offset = root_face * to_base_faces_per_root;

        for (const int br : IndexRange(block_side)) {
          const int to_r = from_r * block_side + br;
          for (const int bc : IndexRange(block_side)) {
            const int to_c = from_c * block_side + bc;

            const int dst_face_idx = root_dst_offset + to_r * to_base_side + to_c;
            const int dst_offset = dst_face_idx * dst_face_cells;

            for (const int r : IndexRange(dst_side)) {
              const int src_row = src_offset + (br * dst_side + r) * src_side + bc * dst_side;
              const int dst_row = dst_offset + r * dst_side;
              to_face_attr.slice(dst_row, dst_side)
                  .copy_from(from_face_attr.slice(src_row, dst_side));
            }
          }
        }
      }
    });
    return;
  }

  const int block_side = 1 << (-lvl_diff);
  const int grain = math::max(1, 4096 / math::max(1, src_face_cells));
  threading::parallel_for(IndexRange(from_base_faces_num), grain, [&](const IndexRange range) {
    for (const int src_face : range) {
      const int root_face = src_face / from_base_faces_per_root;
      const int local_src_face = src_face % from_base_faces_per_root;

      const int from_r = local_src_face / from_base_side;
      const int from_c = local_src_face % from_base_side;

      const int to_r = from_r / block_side;
      const int to_c = from_c / block_side;
      const int br = from_r % block_side;
      const int bc = from_c % block_side;

      const int dst_face_idx = root_face * to_base_faces_per_root + to_r * to_base_side + to_c;
      const int src_offset = src_face * src_face_cells;
      const int dst_offset = dst_face_idx * dst_face_cells;

      for (const int r : IndexRange(src_side)) {
        const int src_row = src_offset + r * src_side;
        const int dst_row = dst_offset + (br * src_side + r) * dst_side + bc * src_side;
        to_face_attr.slice(dst_row, src_side).copy_from(from_face_attr.slice(src_row, src_side));
      }
    }
  });
}

template<typename T>
static void subdivide_base_to_direct(const Mesh &base_mesh,
                                     const BaseTopology &eff_base,
                                     const int eff_lvl,
                                     const bool use_base1,
                                     const LinearOffset dir_vp,
                                     const LinearOffset dir_ep,
                                     const LinearOffset dir_fp,
                                     const Span<T> in_attr,
                                     const MutableSpan<T> out_attr,
                                     const bool linear)
{
  if (!use_base1) {
    LinearOffset in_vp{0, 1}, in_ep{0, 0}, in_fp{0, 0};
    subdivide_chain_same_base<T>(0,
                                 eff_lvl,
                                 eff_base,
                                 in_vp,
                                 in_ep,
                                 in_fp,
                                 dir_vp,
                                 dir_ep,
                                 dir_fp,
                                 in_attr,
                                 out_attr,
                                 linear);
    return;
  }

  const Array<T> base1_attr = subdivide_base_to_lvl1_linear<T>(
      base_mesh, eff_base.verts_num, in_attr, linear);

  LinearOffset b1_in_vp, b1_in_ep, b1_in_fp;
  bke::multiresolution::make_linear_policies(0, eff_base, b1_in_vp, b1_in_ep, b1_in_fp);
  subdivide_chain_same_base<T>(0,
                               eff_lvl,
                               eff_base,
                               b1_in_vp,
                               b1_in_ep,
                               b1_in_fp,
                               dir_vp,
                               dir_ep,
                               dir_fp,
                               base1_attr.as_span(),
                               out_attr,
                               linear);
}

template<typename T>
static void subdivide_face_attr(const Mesh &base_mesh,
                                const int from_lvl,
                                const Span<T> from_face_attr,
                                const int to_lvl,
                                const MutableSpan<T> to_face_attr);

template<typename T>
static void eval_base_face_attribute(const Mesh &mesh,
                                     const StringRef name,
                                     Mesh &eval_mesh,
                                     const int target_lvl)
{
  const bke::AttributeAccessor base_attributes = mesh.attributes();
  const VArray<T> base_attr = *base_attributes.lookup<T>(name, bke::AttrDomain::Face);
  if (base_attr.is_empty()) {
    return;
  }

  if (base_attr.is_single()) {
    const T single_val = base_attr.get_internal_single();
    const GPointer value(CPPType::get<T>(), &single_val);
    bke::GList data(
        CPPType::get<T>(), bke::GList::SingleData::ForValue(value), eval_mesh.faces_num);
    if (bke::Attribute *attr = eval_mesh.attribute_storage.wrap().lookup(name)) {
      attr->assign_data(std::move(data));
    }
    else {
      eval_mesh.attribute_storage.wrap().add(
          std::string(name), bke::AttrDomain::Face, std::move(data));
    }
    return;
  }

  const MutableSpan<T> target_attr = bke::get_mutable_attribute<T>(
      eval_mesh.attribute_storage.wrap(), bke::AttrDomain::Face, name, eval_mesh.faces_num);
  const VArraySpan<T> base_span(base_attr);
  subdivide_face_attr<T>(mesh, 0, base_span, target_lvl, target_attr);
}

static void eval_face_attributes_outside_sculpt(const Mesh &mesh,
                                                Mesh &eval_mesh,
                                                const int target_lvl)
{
  eval_base_face_attribute<int>(mesh, ".sculpt_face_set", eval_mesh, target_lvl);
  eval_base_face_attribute<bool>(mesh, ".hide_poly", eval_mesh, target_lvl);
  eval_base_face_attribute<int>(mesh, "material_index", eval_mesh, target_lvl);
  eval_base_face_attribute<bool>(mesh, ".select_poly", eval_mesh, target_lvl);
}

static void eval_uvs(const Mesh &mesh, Mesh &eval_mesh, const int target_lvl)
{
  const bke::AttributeAccessor base_attributes = mesh.attributes();
  for (const StringRefNull &uv_name : mesh.uv_map_names()) {
    const VArraySpan<float2> base_uv = *base_attributes.lookup<float2>(uv_name,
                                                                       bke::AttrDomain::Corner);
    if (base_uv.is_empty()) {
      continue;
    }
    MutableSpan<float2> eval_uv = bke::get_mutable_attribute<float2>(
        eval_mesh.attribute_storage.wrap(),
        bke::AttrDomain::Corner,
        uv_name,
        eval_mesh.corners_num);
    bke::multiresolution::uv::calc_smooth_uvs(mesh, base_uv, target_lvl, eval_uv, false);
  }
}

static void copy_sculpt_layers_meta(const Mesh &mesh, Mesh &eval_mesh)
{
  eval_mesh.sculpt_layers = static_cast<SculptLayer *>(MEM_dupalloc(mesh.sculpt_layers));
  eval_mesh.sculpt_layers_len = mesh.sculpt_layers_len;
  eval_mesh.active_layer_index = mesh.active_layer_index;
}

struct EvalMeshContext {
  Mesh *eval_mesh = nullptr;
  BaseTopology eff_base;
  int eff_lvl = 0;
  bool use_base1 = false;
  LinearOffset dir_vp{};
  LinearOffset dir_ep{};
  LinearOffset dir_fp{};
};

static EvalMeshContext create_eval_mesh(const Multiresolution &multires,
                                        const Mesh &mesh,
                                        const int target_lvl)
{
  EvalMeshContext ctx;
  ctx.use_base1 = !multires.base_all_quads && target_lvl > 1;
  const Mesh &eff_mesh = ctx.use_base1 ?
                             multires.runtime->ensure_base_mesh(mesh, 1, multires.base_all_quads) :
                             mesh;
  ctx.eff_base = BaseTopology(eff_mesh);
  ctx.eff_lvl = ctx.use_base1 ? target_lvl - 1 : target_lvl;
  bke::multiresolution::make_linear_policies(
      ctx.eff_lvl, ctx.eff_base, ctx.dir_vp, ctx.dir_ep, ctx.dir_fp);
  ctx.eval_mesh = eval_topology(ctx.eff_base, ctx.eff_lvl, ctx.dir_vp, ctx.dir_ep, ctx.dir_fp);

  BKE_mesh_copy_parameters_for_eval(ctx.eval_mesh, &mesh);
  copy_sculpt_layers_meta(mesh, *ctx.eval_mesh);
  return ctx;
}

template<typename T>
static void table_to_direct_order(const BaseTopology &eff_base,
                                  const int eff_lvl,
                                  const int eff_coarse,
                                  const LinearOffset dir_vp,
                                  const LinearOffset dir_ep,
                                  const LinearOffset dir_fp,
                                  const OffsetsSpan table_offsets,
                                  const Span<T> src,
                                  const MutableSpan<T> dst)
{
  if (eff_coarse == 0) {
    convert_table_to_linear(eff_lvl,
                            eff_base,
                            dir_vp,
                            dir_ep,
                            dir_fp,
                            table_offsets.verts,
                            table_offsets.edges,
                            table_offsets.faces,
                            src,
                            dst);
  }
  else {
    rebase_attribute_internal<T, false>(eff_base,
                                        eff_lvl,
                                        0,
                                        eff_coarse,
                                        dir_vp,
                                        dir_ep,
                                        dir_fp,
                                        TableOffset{table_offsets.verts},
                                        TableOffset{table_offsets.edges},
                                        TableOffset{table_offsets.faces},
                                        src,
                                        dst);
  }
}

Mesh *Multiresolution::eval_to_mesh(const Mesh &mesh, const int target_lvl)
{
  const Multiresolution &multires = *mesh.multires;
  BLI_assert(target_lvl > 0 && target_lvl <= multires.top_level);

  const EvalMeshContext ctx = create_eval_mesh(multires, mesh, target_lvl);
  Mesh &eval_mesh = *ctx.eval_mesh;

  const bke::multiresolution::BaseTopology top_base = multires.get_base_topology(
      mesh, multires.top_level);
  const int top_base_lvl = multires.base_level(multires.top_level);
  const Offsets top_offsets = build_table_offsets(multires, top_base, multires.top_level);

  auto eval_top_to_out = [&]<typename T>(const Span<T> src_attr, const MutableSpan<T> dst_attr) {
    if (target_lvl == multires.top_level) {
      table_to_direct_order<T>(ctx.eff_base,
                               ctx.eff_lvl,
                               ctx.use_base1 ? top_base_lvl - 1 : top_base_lvl,
                               ctx.dir_vp,
                               ctx.dir_ep,
                               ctx.dir_fp,
                               top_offsets,
                               src_attr,
                               dst_attr);
      return;
    }
    if (!multires.base_all_quads && target_lvl == 1) {
      Multiresolution::downsample_to_base<T>(top_offsets.verts, src_attr, dst_attr);
      return;
    }
    const int from_eff = ctx.use_base1 ? multires.top_level - 1 : multires.top_level;
    const int from_coarse = ctx.use_base1 ? top_base_lvl - 1 : top_base_lvl;
    downsample_to_internal<T>(ctx.eff_base,
                              from_eff,
                              from_coarse,
                              top_base,
                              TableOffset{top_offsets.verts},
                              TableOffset{top_offsets.edges},
                              TableOffset{top_offsets.faces},
                              src_attr,
                              ctx.eff_lvl,
                              0,
                              ctx.eff_base,
                              ctx.dir_vp,
                              ctx.dir_ep,
                              ctx.dir_fp,
                              dst_attr);
  };

  auto eval_attr = [&]<typename T, typename GetOutput>(const Span<T> top_attr,
                                                       GetOutput &&get_output,
                                                       const Span<T> base_attr = {},
                                                       const bool subdivide_linear = false,
                                                       const bool clamp_delta = false) {
    if (base_attr.is_empty()) {
      return;
    }
    MutableSpan<T> out = get_output();
    if (top_attr.is_empty()) {
      if (subdivide_linear || target_lvl == multires.top_level) {
        subdivide_base_to_direct<T>(mesh,
                                    ctx.eff_base,
                                    ctx.eff_lvl,
                                    ctx.use_base1,
                                    ctx.dir_vp,
                                    ctx.dir_ep,
                                    ctx.dir_fp,
                                    base_attr,
                                    out,
                                    subdivide_linear);
        return;
      }
      Array<T> top_scratch(multires.top_verts_num());
      multires.subdivide_base_to<T>(
          mesh, multires.top_level, top_offsets, base_attr, top_scratch.as_mutable_span(), false);
      eval_top_to_out(top_scratch.as_span(), out);
      return;
    }

    eval_top_to_out(top_attr, out);

    const Span<int> top_vert_offsets = top_offsets.verts;
    bool dirty = false;
    for (const int i : base_attr.index_range()) {
      if (base_attr[i] != top_attr[top_vert_offsets[i]]) {
        dirty = true;
        break;
      }
    }
    if (!dirty) {
      return;
    }

    Array<T> delta(base_attr.size());
    threading::parallel_for(delta.index_range(), 8192, [&](const IndexRange range) {
      for (const int i : range) {
        delta[i] = base_attr[i] - top_attr[top_vert_offsets[i]];
      }
    });

    Array<T> target_delta(out.size());
    if (subdivide_linear || target_lvl == multires.top_level) {
      subdivide_base_to_direct<T>(mesh,
                                  ctx.eff_base,
                                  ctx.eff_lvl,
                                  ctx.use_base1,
                                  ctx.dir_vp,
                                  ctx.dir_ep,
                                  ctx.dir_fp,
                                  delta.as_span(),
                                  target_delta.as_mutable_span(),
                                  subdivide_linear);
    }
    else {
      Array<T> top_delta(multires.top_verts_num());
      multires.subdivide_base_to<T>(mesh,
                                    multires.top_level,
                                    top_offsets,
                                    delta.as_span(),
                                    top_delta.as_mutable_span(),
                                    false);
      eval_top_to_out(top_delta.as_span(), target_delta.as_mutable_span());
    }

    threading::parallel_for(out.index_range(), 8192, [&](const IndexRange range) {
      if (clamp_delta) {
        for (const int i : range) {
          out[i] = math::clamp(out[i] + target_delta[i], 0.0f, 1.0f);
        }
      }
      else {
        for (const int i : range) {
          out[i] += target_delta[i];
        }
      }
    });
  };

  const bke::AttributeAccessor base_attributes = mesh.attributes();
  const VArraySpan<float> base_masks = *base_attributes.lookup<float>(".sculpt_mask",
                                                                      bke::AttrDomain::Point);

  eval_attr(
      multires.top_base_positions(),
      [&]() { return eval_mesh.vert_positions_for_write(); },
      mesh.vert_positions());

  eval_attr(
      multires.top_masks(),
      [&]() {
        return bke::get_mutable_attribute<float>(eval_mesh.attribute_storage.wrap(),
                                                 bke::AttrDomain::Point,
                                                 ".sculpt_mask",
                                                 eval_mesh.verts_num);
      },
      base_masks,
      true,
      true);

  if (!StringRef(mesh.active_color_attribute).is_empty()) {
    const VArraySpan<ColorGeometry4f> base_colors = *base_attributes.lookup<ColorGeometry4f>(
        mesh.active_color_attribute, bke::AttrDomain::Point);
    eval_attr(
        multires.top_colors(),
        [&]() {
          return bke::get_mutable_attribute<ColorGeometry4f>(eval_mesh.attribute_storage.wrap(),
                                                             bke::AttrDomain::Point,
                                                             mesh.active_color_attribute,
                                                             eval_mesh.verts_num)
              .cast<float4>();
        },
        base_colors.cast<float4>(),
        true,
        true);
  }

  for (const SculptLayer &layer : mesh.sculpt_layers_as_span()) {
    const Span<float3> top_displacements = bke::sculpt_layer::layer_top_displacements(mesh,
                                                                                      layer.name);
    const std::string attr_name = bke::sculpt_layer::attr_name_from_layer_name(layer.name);
    const Span<float3> base_displacements =
        bke::get_span_attribute<float3>(
            mesh.attribute_storage.wrap(), bke::AttrDomain::Point, attr_name, mesh.verts_num)
            .value_or(Span<float3>());

    eval_attr(
        top_displacements,
        [&]() { return bke::sculpt_layer::layer_displacements_for_write(eval_mesh, layer.name); },
        base_displacements);
  }

  eval_face_attributes_outside_sculpt(mesh, eval_mesh, target_lvl);
  eval_uvs(mesh, eval_mesh, target_lvl);
  bke::mesh_smooth_set(eval_mesh, false, false);

  return ctx.eval_mesh;
}

Mesh *Multiresolution::eval_to_mesh_in_sculpt(const Mesh &mesh, const int target_lvl)
{
  const Multiresolution &multires = *mesh.multires;

  BLI_assert(target_lvl > 0 && target_lvl <= multires.top_level);
  BLI_assert(target_lvl == multires.curr_level);

  const int curr_base_lvl = multires.base_level(multires.curr_level);
  const bke::multiresolution::BaseTopology curr_base = multires.get_base_topology(
      mesh, multires.curr_level);
  const Offsets curr_offsets = build_table_offsets(multires, curr_base, multires.curr_level);

  const EvalMeshContext ctx = create_eval_mesh(multires, mesh, target_lvl);
  Mesh &eval_mesh = *ctx.eval_mesh;
  const int eff_coarse = ctx.use_base1 ? curr_base_lvl - 1 : curr_base_lvl;

  auto copy_vertex_attr = [&]<typename T, typename GetOutput>(const Span<T> curr_attr,
                                                              GetOutput &&get_output) {
    if (curr_attr.is_empty()) {
      return;
    }
    MutableSpan<T> out = get_output();
    BLI_assert(curr_attr.size() == out.size());
    table_to_direct_order<T>(ctx.eff_base,
                             ctx.eff_lvl,
                             eff_coarse,
                             ctx.dir_vp,
                             ctx.dir_ep,
                             ctx.dir_fp,
                             curr_offsets,
                             curr_attr,
                             out);
  };

  auto copy_face_attr = [&]<typename T, typename GetOutput>(const Span<T> curr_attr,
                                                            GetOutput &&get_output) {
    if (curr_attr.is_empty()) {
      return;
    }
    MutableSpan<T> out = get_output();
    BLI_assert(curr_attr.size() == out.size());
    if (eff_coarse == 0) {
      out.copy_from(curr_attr);
      return;
    }
    rebase_face_attr<T>(
        curr_base.faces_num, eff_coarse, 0, ctx.eff_lvl - eff_coarse, curr_attr, out);
  };

  copy_vertex_attr(multires.base_positions(),
                   [&]() { return eval_mesh.vert_positions_for_write(); });

  copy_vertex_attr(multires.masks(), [&]() {
    return bke::get_mutable_attribute<float>(eval_mesh.attribute_storage.wrap(),
                                             bke::AttrDomain::Point,
                                             ".sculpt_mask",
                                             eval_mesh.verts_num);
  });

  if (!StringRef(mesh.active_color_attribute).is_empty()) {
    copy_vertex_attr(multires.colors(), [&]() {
      return bke::get_mutable_attribute<ColorGeometry4f>(eval_mesh.attribute_storage.wrap(),
                                                         bke::AttrDomain::Point,
                                                         mesh.active_color_attribute,
                                                         eval_mesh.verts_num)
          .cast<float4>();
    });
  }

  for (const SculptLayer &layer : mesh.sculpt_layers_as_span()) {
    copy_vertex_attr(bke::sculpt_layer::layer_displacements(mesh, layer.name), [&]() {
      return bke::sculpt_layer::layer_displacements_for_write(eval_mesh, layer.name);
    });
  }

  copy_face_attr(multires.face_sets(), [&]() {
    return bke::get_mutable_attribute<int>(eval_mesh.attribute_storage.wrap(),
                                           bke::AttrDomain::Face,
                                           ".sculpt_face_set",
                                           eval_mesh.faces_num);
  });

  copy_face_attr(multires.hidden_faces(), [&]() {
    return bke::get_mutable_attribute<bool>(eval_mesh.attribute_storage.wrap(),
                                            bke::AttrDomain::Face,
                                            ".hide_poly",
                                            eval_mesh.faces_num);
  });

  eval_base_face_attribute<int>(mesh, "material_index", eval_mesh, target_lvl);
  eval_base_face_attribute<bool>(mesh, ".select_poly", eval_mesh, target_lvl);
  eval_uvs(mesh, eval_mesh, target_lvl);

  bke::mesh_smooth_set(eval_mesh, false, false);

  return ctx.eval_mesh;
}

Span<float3> Multiresolution::calc_smooth_positions(const Mesh &base_mesh,
                                                    const int target_lvl,
                                                    Array<float3> &storage)
{
  if (target_lvl == 0) {
    return base_mesh.vert_positions();
  }

  const Multiresolution &multires = *base_mesh.multires;
  const bool use_base1 = !multires.base_all_quads && target_lvl > 1;
  const Mesh &eff_mesh = use_base1 ? multires.runtime->ensure_base_mesh(
                                         base_mesh, 1, multires.base_all_quads) :
                                     base_mesh;
  const BaseTopology eff_base(eff_mesh);
  const int eff_lvl = use_base1 ? target_lvl - 1 : target_lvl;

  LinearOffset dir_vp, dir_ep, dir_fp;
  bke::multiresolution::make_linear_policies(eff_lvl, eff_base, dir_vp, dir_ep, dir_fp);

  storage.reinitialize(bke::multiresolution::calc_total_verts(
      eff_lvl, eff_base.verts_num, eff_base.edges_num, eff_base.faces_num, eff_base.corners_num));

  subdivide_base_to_direct<float3>(base_mesh,
                                   eff_base,
                                   eff_lvl,
                                   use_base1,
                                   dir_vp,
                                   dir_ep,
                                   dir_fp,
                                   base_mesh.vert_positions(),
                                   storage.as_mutable_span(),
                                   false);

  return storage;
}

static int subfaces_per_face(const int lvl)
{
  return 1 << (2 * lvl);
}

BLI_INLINE int mode_of_3(const int a, const int b, const int c)
{
  const int match_a = (a == b) | (a == c);
  const int match_b = b == c;
  return match_a ? a : (match_b ? b : a);
}

BLI_INLINE int mode_of_4(const int a, const int b, const int c, const int d)
{
  const int match_a = (a == b) | (a == c) | (a == d);
  const int match_b = (b == c) | (b == d);
  const int match_c = (c == d);
  return match_a ? a : (match_b ? b : (match_c ? c : a));
}

static int mode_of_span(const Span<int> values)
{
  if (values.size() == 4) {
    return mode_of_4(values[0], values[1], values[2], values[3]);
  }
  if (values.size() == 3) {
    return mode_of_3(values[0], values[1], values[2]);
  }

  struct Entry {
    int id;
    int count;
  };
  Vector<Entry, 8> counts;
  for (const int id : values) {
    bool found = false;
    for (Entry &entry : counts) {
      if (entry.id == id) {
        entry.count++;
        found = true;
        break;
      }
    }
    if (!found) {
      counts.append({id, 1});
    }
  }

  int best_id = values[0];
  int max_count = 0;
  for (const Entry &entry : counts) {
    if (entry.count > max_count) {
      max_count = entry.count;
      best_id = entry.id;
    }
  }
  return best_id;
}

BLI_INLINE int coarse_face_value_of_4(const int a, const int b, const int c, const int d)
{
  return mode_of_4(a, b, c, d);
}

BLI_INLINE bool coarse_face_value_of_4(const bool a, const bool b, const bool c, const bool d)
{
  return (int(a) + int(b) + int(c) + int(d)) >= 2;
}

static bool hidden_of_span(const Span<bool> values)
{
  int hidden_count = 0;
  for (const bool hidden : values) {
    hidden_count += int(hidden);
  }
  return hidden_count * 2 >= int(values.size());
}

static int coarse_face_value_of_span(const Span<int> values)
{
  return mode_of_span(values);
}

static bool coarse_face_value_of_span(const Span<bool> values)
{
  return hidden_of_span(values);
}

template<typename T>
static void downsample_face_attr_step(const int quads_num,
                                      const int from_lvl,
                                      const Span<T> from_face_attr,
                                      const MutableSpan<T> to_face_attr)
{
  BLI_assert(from_lvl >= 1);

  const int fine_faces_per_side = 1 << from_lvl;
  const int coarse_faces_per_side = fine_faces_per_side / 2;
  const int fine_faces_per_quad = fine_faces_per_side * fine_faces_per_side;
  const int coarse_faces_per_quad = coarse_faces_per_side * coarse_faces_per_side;

  if (from_lvl == 1) {
    threading::parallel_for(IndexRange(quads_num), 2048, [&](const IndexRange range) {
      for (const int face_idx : range) {
        const int fine_offset = face_idx * 4;
        to_face_attr[face_idx] = coarse_face_value_of_4(from_face_attr[fine_offset + 0],
                                                        from_face_attr[fine_offset + 1],
                                                        from_face_attr[fine_offset + 2],
                                                        from_face_attr[fine_offset + 3]);
      }
    });
    return;
  }

  const int grain = math::max(1, 4096 / coarse_faces_per_quad);

  threading::parallel_for(IndexRange(quads_num), grain, [&](const IndexRange range) {
    for (const int face_idx : range) {
      const int to_base = face_idx * coarse_faces_per_quad;
      const int from_base = face_idx * fine_faces_per_quad;

      for (const int row : IndexRange(coarse_faces_per_side)) {
        int offset = to_base + row * coarse_faces_per_side;
        const int row0 = from_base + 2 * row * fine_faces_per_side;
        const int row1 = row0 + fine_faces_per_side;

        for (const int col : IndexRange(coarse_faces_per_side)) {
          const int col0 = 2 * col;

          const T a = from_face_attr[row0 + col0];
          const T b = from_face_attr[row0 + col0 + 1];
          const T c = from_face_attr[row1 + col0];
          const T d = from_face_attr[row1 + col0 + 1];
          to_face_attr[offset++] = coarse_face_value_of_4(a, b, c, d);
        }
      }
    }
  });
}

template<typename T>
static void downsample_face_attr_chain(const int num_quads,
                                       const int from_lvl,
                                       const int to_lvl,
                                       const Span<T> from_face_attr,
                                       const MutableSpan<T> to_face_attr)
{
  BLI_assert(from_lvl > to_lvl);

  if (to_lvl == from_lvl - 1) {
    downsample_face_attr_step(num_quads, from_lvl, from_face_attr, to_face_attr);
    return;
  }

  const int max_buffer_size = num_quads * subfaces_per_face(from_lvl - 1);
  Array<T> buffer_a(max_buffer_size);

  downsample_face_attr_step(num_quads, from_lvl, from_face_attr, buffer_a.as_mutable_span());

  Array<T> buffer_b;
  if (from_lvl - 1 > to_lvl + 1) {
    const int second_buffer_size = num_quads * subfaces_per_face(from_lvl - 2);
    buffer_b.reinitialize(second_buffer_size);
  }

  Span<T> src = buffer_a.as_span();
  bool in_buffer_a = true;

  for (int lvl = from_lvl - 1; lvl > to_lvl + 1; lvl--) {
    const int out_size = num_quads * subfaces_per_face(lvl - 1);
    const MutableSpan<T> dst = (in_buffer_a ? buffer_b.as_mutable_span() :
                                              buffer_a.as_mutable_span())
                                   .take_front(out_size);

    downsample_face_attr_step(num_quads, lvl, src, dst);

    src = dst;
    in_buffer_a = !in_buffer_a;
  }

  const int last_lvl = to_lvl + 1;
  downsample_face_attr_step(num_quads, last_lvl, src, to_face_attr);
}

template<typename T>
static void downsample_face_attr_internal(const int from_abs,
                                          const int from_base_rel,
                                          const int from_base_faces_num,
                                          const Span<T> from_face_attr,
                                          const int to_abs,
                                          const int to_base_rel,
                                          const int to_base_faces_num,
                                          const MutableSpan<T> to_face_attr)
{
  BLI_assert(from_abs > to_abs);
  BLI_assert(from_base_rel >= to_base_rel);

  const int delta_from = from_abs - from_base_rel;
  const int delta_to = to_abs - to_base_rel;
  BLI_assert(delta_from >= 0);
  BLI_assert(delta_to >= 0);

  if (from_base_rel == to_base_rel) {
    BLI_assert(from_base_faces_num == to_base_faces_num);
    downsample_face_attr_chain(
        from_base_faces_num, delta_from, delta_to, from_face_attr, to_face_attr);
    return;
  }

  const int pivot_abs = math::max(to_abs, from_base_rel);
  const int delta_pivot_from = pivot_abs - from_base_rel;
  const int delta_pivot_to = pivot_abs - to_base_rel;

  Array<T> pivot_from_buf;
  Span<T> pivot_from_span;
  if (from_abs == pivot_abs) {
    pivot_from_span = from_face_attr;
  }
  else {
    pivot_from_buf.reinitialize(from_base_faces_num * subfaces_per_face(delta_pivot_from));
    downsample_face_attr_chain(from_base_faces_num,
                               delta_from,
                               delta_pivot_from,
                               from_face_attr,
                               pivot_from_buf.as_mutable_span());
    pivot_from_span = pivot_from_buf.as_span();
  }

  if (pivot_abs == to_abs) {
    rebase_face_attr(from_base_faces_num,
                     from_base_rel,
                     to_base_rel,
                     delta_pivot_from,
                     pivot_from_span,
                     to_face_attr);
    return;
  }

  Array<T> pivot_to_buf(to_base_faces_num * subfaces_per_face(delta_pivot_to));
  rebase_face_attr(from_base_faces_num,
                   from_base_rel,
                   to_base_rel,
                   delta_pivot_from,
                   pivot_from_span,
                   pivot_to_buf.as_mutable_span());

  downsample_face_attr_chain(
      to_base_faces_num, delta_pivot_to, delta_to, pivot_to_buf.as_span(), to_face_attr);
}

template<typename T>
static void downsample_face_attr(const Multiresolution &multires,
                                 const Mesh &base_mesh,
                                 const int from_lvl,
                                 const Span<T> from_face_attr,
                                 const int to_lvl,
                                 const MutableSpan<T> to_face_attr)
{
  BLI_assert(from_lvl > to_lvl);
  BLI_assert(to_lvl >= 0);

  const bool all_quads = multires.base_all_quads;

  if (all_quads || to_lvl >= 1) {
    const int shift = all_quads ? 0 : 1;
    const int root_quads_num = all_quads ? base_mesh.faces_num : base_mesh.corners_num;
    const int from_base_rel = math::max(0, multires.base_level(from_lvl) - shift);
    const int to_base_rel = math::max(0, multires.base_level(to_lvl) - shift);

    downsample_face_attr_internal(from_lvl - shift,
                                  from_base_rel,
                                  root_quads_num << (2 * from_base_rel),
                                  from_face_attr,
                                  to_lvl - shift,
                                  to_base_rel,
                                  root_quads_num << (2 * to_base_rel),
                                  to_face_attr);
    return;
  }

  Span<T> lvl1_face_attr;
  Array<T> lvl1_buffer;

  if (from_lvl == 1) {
    lvl1_face_attr = from_face_attr;
  }
  else {
    lvl1_buffer.reinitialize(base_mesh.corners_num);
    const int from_abs = from_lvl - 1;
    const int from_base_rel = multires.base_level(from_lvl) - 1;
    const int from_base_faces_num = base_mesh.corners_num << (2 * from_base_rel);
    downsample_face_attr_internal(from_abs,
                                  from_base_rel,
                                  from_base_faces_num,
                                  from_face_attr,
                                  0,
                                  0,
                                  base_mesh.corners_num,
                                  lvl1_buffer.as_mutable_span());
    lvl1_face_attr = lvl1_buffer;
  }

  const OffsetIndices<int> base_faces = base_mesh.faces();
  threading::parallel_for(IndexRange(base_mesh.faces_num), 2048, [&](const IndexRange range) {
    for (const int f : range) {
      const IndexRange face_corners = base_faces[f];
      to_face_attr[f] = coarse_face_value_of_span(
          lvl1_face_attr.slice(face_corners.start(), face_corners.size()));
    }
  });
}

void Multiresolution::downsample_face_sets(const Mesh &base_mesh,
                                           const int from_lvl,
                                           const Span<int> from_face_sets,
                                           const int to_lvl,
                                           const MutableSpan<int> to_face_sets) const
{
  downsample_face_attr(*this, base_mesh, from_lvl, from_face_sets, to_lvl, to_face_sets);
}

template<typename T>
static void downsample_face_attr_root_quads(const Multiresolution &multires,
                                            const int from_lvl,
                                            const Span<T> from_face_attr,
                                            const int to_lvl,
                                            const MutableSpan<T> to_face_attr,
                                            const Span<int> root_quads)
{
  if (root_quads.is_empty()) {
    return;
  }
  BLI_assert(from_lvl > to_lvl);
  BLI_assert(to_lvl >= 1);

  const bool all_quads = multires.base_all_quads;
  const int shift = all_quads ? 0 : 1;
  const int from_abs = from_lvl - shift;
  const int to_abs = to_lvl - shift;
  const int from_base_rel = math::max(0, multires.base_level(from_lvl) - shift);
  const int to_base_rel = math::max(0, multires.base_level(to_lvl) - shift);

  const int fine_subfaces_per_root = 1 << (2 * from_abs);
  const int coarse_subfaces_per_root = 1 << (2 * to_abs);

  threading::parallel_for(root_quads.index_range(), 64, [&](const IndexRange range) {
    for (const int i : range) {
      const int q = root_quads[i];
      const Span<T> from_slice = from_face_attr.slice(q * fine_subfaces_per_root,
                                                      fine_subfaces_per_root);
      const MutableSpan<T> to_slice = to_face_attr.slice(q * coarse_subfaces_per_root,
                                                         coarse_subfaces_per_root);
      downsample_face_attr_internal<T>(from_abs,
                                       from_base_rel,
                                       1 << (2 * from_base_rel),
                                       from_slice,
                                       to_abs,
                                       to_base_rel,
                                       1 << (2 * to_base_rel),
                                       to_slice);
    }
  });
}

void Multiresolution::downsample_face_sets(const int from_lvl,
                                           const Span<int> from_face_sets,
                                           const int to_lvl,
                                           const MutableSpan<int> to_face_sets,
                                           const Span<int> root_quads) const
{
  downsample_face_attr_root_quads<int>(
      *this, from_lvl, from_face_sets, to_lvl, to_face_sets, root_quads);
}

void Multiresolution::downsample_hidden_faces(const Mesh &base_mesh,
                                              const int from_lvl,
                                              const Span<bool> from_hidden_faces,
                                              const int to_lvl,
                                              const MutableSpan<bool> to_hidden_faces) const
{
  downsample_face_attr(*this, base_mesh, from_lvl, from_hidden_faces, to_lvl, to_hidden_faces);
}

void Multiresolution::downsample_hidden_faces(const int from_lvl,
                                              const Span<bool> from_hidden_faces,
                                              const int to_lvl,
                                              const MutableSpan<bool> to_hidden_faces,
                                              const Span<int> root_quads) const
{
  downsample_face_attr_root_quads<bool>(
      *this, from_lvl, from_hidden_faces, to_lvl, to_hidden_faces, root_quads);
}

template<typename T>
static void subdivide_face_attr_same_base(const int num_quads,
                                          const int from_lvl,
                                          const Span<T> from_face_attr,
                                          const int to_lvl,
                                          const MutableSpan<T> to_face_attr)
{
  BLI_assert(to_lvl > from_lvl);
  BLI_assert(from_lvl >= 0);

  const int coarse_faces_per_side = 1 << from_lvl;
  const int fine_faces_per_side = 1 << to_lvl;
  const int side_ratio = 1 << (to_lvl - from_lvl);

  const int coarse_faces_per_quad = coarse_faces_per_side * coarse_faces_per_side;
  const int fine_faces_per_quad = fine_faces_per_side * fine_faces_per_side;

  if (coarse_faces_per_quad == 1) {
    const int grain = math::max(1, 4096 / fine_faces_per_quad);
    threading::parallel_for(IndexRange(num_quads), grain, [&](const IndexRange range) {
      for (const int face_idx : range) {
        to_face_attr.slice(face_idx * fine_faces_per_quad, fine_faces_per_quad)
            .fill(from_face_attr[face_idx]);
      }
    });
    return;
  }

  if (to_lvl == from_lvl + 1) {
    const int grain = math::max(1, 4096 / fine_faces_per_quad);
    threading::parallel_for(IndexRange(num_quads), grain, [&](const IndexRange range) {
      for (const int face_idx : range) {
        const int coarse_base_offset = face_idx * coarse_faces_per_quad;
        const int fine_base_offset = face_idx * fine_faces_per_quad;

        for (const int r : IndexRange(coarse_faces_per_side)) {
          const int coarse_row_offset = coarse_base_offset + r * coarse_faces_per_side;
          const int row0 = fine_base_offset + (2 * r) * fine_faces_per_side;
          const int row1 = row0 + fine_faces_per_side;

          for (const int c : IndexRange(coarse_faces_per_side)) {
            const T id = from_face_attr[coarse_row_offset + c];
            const int col = 2 * c;

            to_face_attr[row0 + col] = id;
            to_face_attr[row0 + col + 1] = id;
            to_face_attr[row1 + col] = id;
            to_face_attr[row1 + col + 1] = id;
          }
        }
      }
    });
    return;
  }

  const int grain = math::max(1, 4096 / fine_faces_per_quad);
  threading::parallel_for(IndexRange(num_quads), grain, [&](const IndexRange range) {
    for (const int face_idx : range) {
      const int coarse_base_offset = face_idx * coarse_faces_per_quad;
      const int fine_base_offset = face_idx * fine_faces_per_quad;

      for (const int r : IndexRange(coarse_faces_per_side)) {
        const int coarse_row_offset = coarse_base_offset + r * coarse_faces_per_side;

        for (int dr = 0; dr < side_ratio; dr++) {
          const int fine_row_offset = fine_base_offset +
                                      (r * side_ratio + dr) * fine_faces_per_side;

          for (const int c : IndexRange(coarse_faces_per_side)) {
            const T id = from_face_attr[coarse_row_offset + c];
            to_face_attr.slice(fine_row_offset + c * side_ratio, side_ratio).fill(id);
          }
        }
      }
    }
  });
}

template<typename T>
static void subdivide_face_attr_internal(const int from_lvl,
                                         const int from_base_lvl,
                                         const int from_base_faces_num,
                                         const Span<T> from_face_attr,
                                         const int to_lvl,
                                         const int to_base_lvl,
                                         const int to_base_faces_num,
                                         const MutableSpan<T> to_face_attr)
{
  BLI_assert(from_lvl < to_lvl);
  BLI_assert(from_base_lvl <= to_base_lvl);

  const int delta_from = from_lvl - from_base_lvl;
  const int delta_to = to_lvl - to_base_lvl;

  if (from_base_lvl == to_base_lvl) {
    BLI_assert(from_base_faces_num == to_base_faces_num);
    subdivide_face_attr_same_base(
        from_base_faces_num, delta_from, from_face_attr, delta_to, to_face_attr);
    return;
  }

  const int pivot_lvl = math::max(from_lvl, to_base_lvl);
  const int delta_pivot_from = pivot_lvl - from_base_lvl;
  const int delta_pivot_to = pivot_lvl - to_base_lvl;

  Array<T> pivot_face_attr;
  Span<T> pivot_span;

  if (from_lvl < pivot_lvl) {
    pivot_face_attr.reinitialize(from_base_faces_num * subfaces_per_face(delta_pivot_from));
    subdivide_face_attr_same_base(from_base_faces_num,
                                  delta_from,
                                  from_face_attr,
                                  delta_pivot_from,
                                  pivot_face_attr.as_mutable_span());
    pivot_span = pivot_face_attr;
  }
  else {
    pivot_span = from_face_attr;
  }

  Array<T> pivot_rebased(to_base_faces_num * subfaces_per_face(delta_pivot_to));
  rebase_face_attr(from_base_faces_num,
                   from_base_lvl,
                   to_base_lvl,
                   delta_pivot_from,
                   pivot_span,
                   pivot_rebased.as_mutable_span());

  subdivide_face_attr_same_base(
      to_base_faces_num, delta_pivot_to, pivot_rebased.as_span(), delta_to, to_face_attr);
}

template<typename T>
static void subdivide_face_attr(const Mesh &base_mesh,
                                const int from_lvl,
                                const Span<T> from_face_attr,
                                const int to_lvl,
                                const MutableSpan<T> to_face_attr)
{
  BLI_assert(to_lvl > from_lvl);
  BLI_assert(from_lvl >= 0);

  const Multiresolution &multires = *base_mesh.multires;
  const bool all_quads = multires.base_all_quads;

  if (from_lvl == 0) {
    const OffsetIndices<int> base_faces = base_mesh.faces();

    if (all_quads) {
      const int fine_faces_per_quad = subfaces_per_face(to_lvl);
      const int grain = math::max(1, 4096 / fine_faces_per_quad);

      threading::parallel_for(IndexRange(base_mesh.faces_num), grain, [&](const IndexRange range) {
        for (const int face_idx : range) {
          const T id = from_face_attr[face_idx];
          to_face_attr.slice(face_idx * fine_faces_per_quad, fine_faces_per_quad).fill(id);
        }
      });
    }
    else {
      const int faces_per_corner = 1 << (2 * (to_lvl - 1));
      threading::parallel_for(IndexRange(base_mesh.faces_num), 1024, [&](const IndexRange range) {
        for (const int face_idx : range) {
          const IndexRange face_range = base_faces[face_idx];
          const int start = face_range.start() * faces_per_corner;
          const int count = face_range.size() * faces_per_corner;

          to_face_attr.slice(start, count).fill(from_face_attr[face_idx]);
        }
      });
    }
    return;
  }

  const int shift = all_quads ? 0 : 1;
  const int root_quads_num = all_quads ? base_mesh.faces_num : base_mesh.corners_num;
  const int from_base_lvl = math::max(0, multires.base_level(from_lvl) - shift);
  const int to_base_lvl = math::max(0, multires.base_level(to_lvl) - shift);

  subdivide_face_attr_internal(from_lvl - shift,
                               from_base_lvl,
                               root_quads_num << (2 * from_base_lvl),
                               from_face_attr,
                               to_lvl - shift,
                               to_base_lvl,
                               root_quads_num << (2 * to_base_lvl),
                               to_face_attr);
}

void Multiresolution::subdivide_face_sets(const Mesh &base_mesh,
                                          const int from_lvl,
                                          const Span<int> from_face_sets,
                                          const int to_lvl,
                                          const MutableSpan<int> to_face_sets) const
{
  subdivide_face_attr(base_mesh, from_lvl, from_face_sets, to_lvl, to_face_sets);
}

void Multiresolution::subdivide_hidden_faces(const Mesh &base_mesh,
                                             const int from_lvl,
                                             const Span<bool> from_hidden_faces,
                                             const int to_lvl,
                                             const MutableSpan<bool> to_hidden_faces) const
{
  subdivide_face_attr(base_mesh, from_lvl, from_hidden_faces, to_lvl, to_hidden_faces);
}

Span<int> Multiresolution::calc_base_materials(const Mesh &base_mesh,
                                               const int base_lvl,
                                               Array<int> &storage)
{
  BLI_assert(base_lvl >= 0);

  const bke::AttributeAccessor attributes = base_mesh.attributes();
  const VArray<int> varray = *attributes.lookup<int>("material_index", bke::AttrDomain::Face);
  if (varray.is_empty()) {
    return {};
  }

  if (varray.is_single()) {
    return {};
  }

  Array<int> base_mats(base_mesh.faces_num);
  varray.materialize(base_mats.as_mutable_span());
  if (base_mats.is_empty()) {
    return {};
  }
  const int first = base_mats[0];
  if (std::all_of(base_mats.begin(), base_mats.end(), [&](int m) { return m == first; })) {
    return {};
  }

  if (base_lvl == 0) {
    storage = std::move(base_mats);
    return storage;
  }

  const int virtual_faces_num = bke::multiresolution::calc_total_faces(
      base_lvl, base_mesh.faces_num, base_mesh.corners_num);
  storage.reinitialize(virtual_faces_num);
  subdivide_face_attr(base_mesh, 0, base_mats.as_span(), base_lvl, storage.as_mutable_span());
  return storage;
}

void Multiresolution::hide_face_flush()
{
  bke::MutableAttributeAccessor attributes = this->attributes_for_write();

  const VArray<bool> hidden_faces = *attributes.lookup_or_default<bool>(
      ".hide_poly", bke::AttrDomain::Face, false);
  if (hidden_faces.is_single() && !hidden_faces.get_internal_single()) {
    attributes.remove(".hide_vert");
    return;
  }

  const MutableSpan<bool> hidden_verts = this->hidden_verts_for_write();
  const Span<int> corner_verts = this->corner_verts();

  threading::parallel_for(hidden_faces.index_range(), 4096, [&](const IndexRange range) {
    for (const int subface_idx : range) {
      if (hidden_faces[subface_idx]) {
        const int v0 = corner_verts[subface_idx * 4 + 0];
        const int v1 = corner_verts[subface_idx * 4 + 1];
        const int v2 = corner_verts[subface_idx * 4 + 2];
        const int v3 = corner_verts[subface_idx * 4 + 3];
        hidden_verts[v0] = true;
        hidden_verts[v1] = true;
        hidden_verts[v2] = true;
        hidden_verts[v3] = true;
      }
    }
  });

  threading::parallel_for(hidden_faces.index_range(), 4096, [&](const IndexRange range) {
    for (const int subface_idx : range) {
      if (!hidden_faces[subface_idx]) {
        const int v0 = corner_verts[subface_idx * 4 + 0];
        const int v1 = corner_verts[subface_idx * 4 + 1];
        const int v2 = corner_verts[subface_idx * 4 + 2];
        const int v3 = corner_verts[subface_idx * 4 + 3];
        hidden_verts[v0] = false;
        hidden_verts[v1] = false;
        hidden_verts[v2] = false;
        hidden_verts[v3] = false;
      }
    }
  });
}

}  // namespace blender
