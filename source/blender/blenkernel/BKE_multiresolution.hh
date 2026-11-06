/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "DNA_multires_types.h"

#include "BKE_mesh.hh"
#include "BKE_mesh_mapping.hh"

#include "BLI_array.hh"
#include "BLI_index_range.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_set.hh"
#include "BLI_string_ref.hh"

#include <memory>
#include <string>

namespace blender {

Multiresolution *BKE_multiresolution_create(const Mesh &base_mesh);

namespace bke {

namespace pbvh {
class Tree;
}

namespace multiresolution {

bool base_counts_match(const Mesh &base_mesh);
bool base_topology_matches(const Mesh &base_mesh);
bool is_active(const Mesh &base_mesh);
bool is_active_fast(const Mesh &base_mesh);

struct Offsets {
  Array<int, 0> verts;
  Array<int, 0> edges;
  Array<int, 0> faces;
};

struct OffsetsSpan {
  Span<int> verts;
  Span<int> edges;
  Span<int> faces;

  OffsetsSpan(const Offsets &offsets)
  {
    verts = offsets.verts;
    edges = offsets.edges;
    faces = offsets.faces;
  }
};

struct BaseTopology {

  BaseTopology() {}

  BaseTopology(const Mesh &base_mesh)
  {
    this->verts_num = base_mesh.verts_num;
    this->edges_num = base_mesh.edges_num;
    this->faces_num = base_mesh.faces_num;
    this->corners_num = base_mesh.corners_num;
    this->edges = base_mesh.edges();
    this->faces = base_mesh.faces();
    this->corner_verts = base_mesh.corner_verts();
    this->corner_edges = base_mesh.corner_edges();
    this->vert_to_edge_map = base_mesh.vert_to_edge_map();
    this->vert_to_face_map = base_mesh.vert_to_face_map();
    this->edge_to_face_map = base_mesh.edge_to_face_map();
    this->corner_to_face_map = base_mesh.corner_to_face_map();
    this->vert_to_corner_map = base_mesh.vert_to_corner_map();
    this->boundary_verts = base_mesh.boundary_verts();
    this->boundary_edges = base_mesh.boundary_edges();
    this->face_edge_reversed = base_mesh.face_edge_reversed();
  }

  int edges_num = 0;
  int verts_num = 0;
  int faces_num = 0;
  int corners_num = 0;
  Span<int2> edges = {};
  OffsetIndices<int> faces = {};
  Span<int> corner_verts = {};
  Span<int> corner_edges = {};
  GroupedSpan<int> vert_to_edge_map = {};
  GroupedSpan<int> vert_to_face_map = {};
  GroupedSpan<int> edge_to_face_map = {};
  Span<int> corner_to_face_map = {};
  GroupedSpan<int> vert_to_corner_map = {};
  BitSpan face_edge_reversed = {};
  BitSpan boundary_verts = {};
  BitSpan boundary_edges = {};
};

struct BVHPartitionSharingInfo {
  const ImplicitSharingInfo *face_indices = nullptr;
  const ImplicitSharingInfo *face_counts = nullptr;
};

struct MultiresCoarseView {
  MultiresCoarseView();
  ~MultiresCoarseView();

  int fine_level = -1;
  int coarse_level = -1;
  const pbvh::Tree *fine_tree = nullptr;

  std::unique_ptr<pbvh::Tree> tree;

  Array<int> to_fine_vert_map;

  Array<float3> positions;
  Array<int> corner_verts;
  Array<float> masks;
  Array<int> face_sets;
  Array<bool> hidden_faces;
  Array<float4> colors;

  Array<int> fine_to_coarse_offsets;
  Array<int> fine_to_coarse_nodes;

  Array<int> coarse_face_to_node;
  int root_quads_num = 0;
  int fine_base_rel = 0;
  int coarse_base_rel = 0;

  int faces_per_corner = 1;
};

struct MultiresolutionRuntime {
  MultiresolutionRuntime(const Mesh &base_mesh);
  ~MultiresolutionRuntime();
  MultiresolutionRuntime(const MultiresolutionRuntime &) = delete;
  MultiresolutionRuntime &operator=(const MultiresolutionRuntime &) = delete;

  Span<int> corner_verts() const;
  Span<float3> normals() const;
  Span<float3> eval_positions() const;
  MutableSpan<float3> eval_positions_for_write();

  void calc_eval_positions(const Mesh &base_mesh, const Span<float3> base_positions);

  void calc_eval_positions(const Mesh &base_mesh,
                           const Span<float3> base_positions,
                           const IndexRange dirty_range);

  void rebuild(const Mesh &base_mesh, const OffsetsSpan offsets);

  const Mesh &ensure_base_mesh(const Mesh &real_base_mesh, const int lvl, const bool all_quads);

  Span<MultiresCoord> coord_from_index_map_ensure(
      const bke::multiresolution::BaseTopology &base_topo,
      const int rel_lvl,
      const bke::multiresolution::OffsetsSpan offsets);

  bool layers_active = false;

  bool curr_positions_dirty = true;
  bool curr_masks_dirty = true;
  bool curr_colors_dirty = true;
  Set<std::string> curr_dirty_layers;
  bool curr_all_layers_dirty = true;

  void tag_curr_attr(const StringRef name);
  bool is_curr_layer_dirty(const StringRef attr_name) const;
  void clear_curr_dirty();
  void clear_sculpt_data();

  Vector<BVHPartitionSharingInfo> bvh_partition_sharing;
  Vector<float3> eval_positions_;
  Vector<int> corner_verts_;
  Vector<float3> normals_;
  Vector<MultiresCoord> coord_from_index_map;

  Vector<Mesh *> base_meshes;

  std::unique_ptr<MultiresCoarseView> coarse_view;

  MultiresCoarseView *ensure_coarse_view(const Mesh &base_mesh, const pbvh::Tree &fine_tree);
};

int calc_total_verts(const int lvl,
                     const int verts_num,
                     const int edges_num,
                     const int faces_num,
                     const int corners_num);

int calc_total_faces(const int lvl, const int faces_num, const int corners_num);

void calc_corner_verts(const BaseTopology &base_topo,
                       const int virtual_lvl,
                       const OffsetsSpan &offsets,
                       const MutableSpan<int> r_corner_verts);

void calc_top_eval_positions(const Mesh &base_mesh,
                             const Span<float3> top_base_positions,
                             const MutableSpan<float3> r_eval_positions);

}  // namespace multiresolution

namespace multiresolution::uv {

void calc_smooth_uvs(const Mesh &base_mesh,
                     const Span<float2> base_uvs,
                     const int target_lvl,
                     MutableSpan<float2> r_uvs,
                     const bool virtual_base_order = true);

Span<float2> calc_smooth_uvs(const Mesh &base_mesh,
                             const Span<float2> base_uvs,
                             const int target_lvl,
                             Array<float2> &storage,
                             const bool virtual_base_order = true);

}  // namespace multiresolution::uv

}  // namespace bke

}  // namespace blender
