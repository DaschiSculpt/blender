/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup DNA
 */

#pragma once

#include "DNA_attribute_types.h"

#include "BLI_generic_span.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"

namespace blender {

namespace bke {
class AttributeAccessor;
class MutableAttributeAccessor;

namespace multiresolution {
struct MultiresolutionRuntime;
struct BaseTopology;
struct Offsets;
struct OffsetsSpan;

#ifdef __cplusplus
enum class ElemType : uint8_t { Vert, Edge, Face };

struct MultiresCoord {
  ElemType type = ElemType::Vert;
  int base_index = 0;
  int local_coord = 0;

  int to_index(const OffsetsSpan &offsets) const;
};
#endif

}  // namespace multiresolution
}  // namespace bke

struct Mesh;

struct BVHPartition {
  int face_indices_len;
  int face_counts_len;
  int *face_indices;
  int *face_counts;
};

struct Multiresolution {
  int top_level;
  int curr_level;
  int sculpt_level;
  int view_level;
  int render_level;

  int base_verts_num;
  int base_edges_num;
  int base_faces_num;
  int base_corners_num;
  char base_all_quads;
  char _pad[3];
  uint64_t base_topology_hash;

  int bvh_partitions_len;
  char _pad2[4];
  BVHPartition *bvh_partitions;

  struct AttributeStorage top_attribute_storage;
  struct AttributeStorage curr_attribute_storage;

  bke::multiresolution::MultiresolutionRuntime *runtime = nullptr;

#ifdef __cplusplus
  void free();

  Span<int> bvh_face_indices(const int level) const;
  MutableSpan<int> bvh_face_indices_for_write(const int level);
  Span<int> bvh_face_counts(const int level) const;
  MutableSpan<int> bvh_face_counts_for_write(const int level);
  void bvh_partition_ensure(const int level, const int face_indices, const int face_counts);
  void truncate_bvh_partitions(const int new_len);
  Span<bke::multiresolution::MultiresCoord> coord_from_index_map_ensure(
      const bke::multiresolution::BaseTopology &base_topo,
      const bke::multiresolution::OffsetsSpan offsets) const;

  Span<float3> positions() const;
  MutableSpan<float3> positions_for_write();
  Span<float3> base_positions() const;
  MutableSpan<float3> base_positions_for_write();
  Span<float3> top_base_positions() const;
  MutableSpan<float3> top_base_positions_for_write();
  Span<float> masks() const;
  MutableSpan<float> masks_for_write();
  Span<float> top_masks() const;
  MutableSpan<float> top_masks_for_write();
  Span<float4> colors() const;
  MutableSpan<float4> colors_for_write();
  Span<float4> top_colors() const;
  MutableSpan<float4> top_colors_for_write();
  Span<bool> hidden_verts() const;
  MutableSpan<bool> hidden_verts_for_write();
  Span<bool> hidden_faces() const;
  MutableSpan<bool> hidden_faces_for_write();
  Span<int> face_sets() const;
  MutableSpan<int> face_sets_for_write();
  Span<int> top_face_sets() const;
  MutableSpan<int> top_face_sets_for_write();

  bke::AttributeAccessor attributes() const;
  bke::MutableAttributeAccessor attributes_for_write();
  bke::AttributeAccessor curr_attributes() const;
  bke::MutableAttributeAccessor curr_attributes_for_write();
  bke::AttributeAccessor top_attributes() const;
  bke::MutableAttributeAccessor top_attributes_for_write();

  template<typename T> Span<T> attribute(const StringRef name) const;
  template<typename T> Span<T> top_attribute(const StringRef name) const;
  template<typename T> MutableSpan<T> attribute_for_write(const StringRef name);
  template<typename T> MutableSpan<T> top_attribute_for_write(const StringRef name);

  template<typename T> Span<T> face_attribute(const StringRef name) const;
  template<typename T> Span<T> top_face_attribute(const StringRef name) const;
  template<typename T> MutableSpan<T> face_attribute_for_write(const StringRef name);
  template<typename T> MutableSpan<T> top_face_attribute_for_write(const StringRef name);

  void clear_top_storage();
  void clear_sculpt_data();

  Span<float3> normals() const;
  MutableSpan<float3> normals_for_write() const;

  MutableSpan<float> ensure_masks();
  MutableSpan<float4> ensure_colors();
  MutableSpan<bool> ensure_hidden_verts();
  MutableSpan<bool> ensure_hidden_faces();
  void hide_face_flush();

  Span<int> corner_verts() const;

  int verts_num() const;
  int top_verts_num() const;
  int faces_num() const;
  int top_faces_num() const;
  int grid_size() const;
  int faces_per_side() const;
  int faces_per_corner() const;

  void face_sets_free();
  void hidden_verts_free();
  void hidden_faces_free();

  bke::multiresolution::MultiresCoord coord_from_index(const int idx) const;

  bke::multiresolution::BaseTopology get_base_topology(const Mesh &base_mesh, const int lvl) const;
  bke::multiresolution::BaseTopology get_base_topology(const Mesh &base_mesh) const;

  int base_level(const int lvl) const;

  static Span<float3> calc_smooth_positions(const Mesh &base_mesh,
                                            const int target_lvl,
                                            Array<float3> &storage);

  static Span<int> calc_base_materials(const Mesh &base_mesh,
                                       const int base_lvl,
                                       Array<int> &storage);

  static void calc_neighbors(const bke::multiresolution::MultiresCoord &coord,
                             const bke::multiresolution::BaseTopology &base_topology,
                             const int grid_size,
                             const bke::multiresolution::OffsetsSpan &offsets,
                             const Span<bool> hidden,
                             Vector<bke::multiresolution::MultiresCoord> &r_neighbors);

  static Mesh *eval_to_mesh(const Mesh &mesh, const int target_lvl);

  static Mesh *eval_to_mesh_in_sculpt(const Mesh &mesh, const int target_lvl);

  static int grid_size(const int lvl);

  static int faces_per_side(const int lvl);

  static int faces_per_corner(const int lvl);

  template<typename T>
  void grid_inner_border(const int grid_offset,
                         const int local_edge,
                         const bool reverse,
                         const Span<T> attr,
                         const MutableSpan<T> r_inner_strip) const;

  static int grid_inner_border_index(const int grid_offset,
                                     const int local_edge,
                                     const bool reverse,
                                     const int grid_size,
                                     const int k);

  template<typename T>
  void subdivide_to(const Mesh &base_mesh,
                    const int from_lvl,
                    const bke::multiresolution::OffsetsSpan from_offsets,
                    const Span<T> from_attr,
                    const int to_lvl,
                    const bke::multiresolution::OffsetsSpan to_offsets,
                    const MutableSpan<T> to_attr,
                    const bool linear) const;

  template<typename T>
  void subdivide_base_to(const Mesh &base_mesh,
                         const int to_lvl,
                         const bke::multiresolution::OffsetsSpan to_offsets,
                         const Span<T> in_attr,
                         const MutableSpan<T> out_attr,
                         const bool linear) const;

  template<typename T>
  void rebase_attribute(const Mesh &base_mesh,
                        const int lvl,
                        const int coarse_lvl,
                        const int fine_lvl,
                        const bke::multiresolution::OffsetsSpan coarse_offsets,
                        const bke::multiresolution::OffsetsSpan fine_offsets,
                        const Span<T> src_attr,
                        const MutableSpan<T> dst_attr) const;

  template<typename T>
  void downsample_to(const Mesh &base_mesh,
                     const int from_lvl,
                     const bke::multiresolution::OffsetsSpan from_offsets,
                     const Span<T> from_attr,
                     const int to_lvl,
                     const bke::multiresolution::OffsetsSpan to_offsets,
                     const MutableSpan<T> to_attr) const;

  template<typename T>
  static void downsample_to_base(const Span<int> from_vert_offsets,
                                 const Span<T> from_attr,
                                 const MutableSpan<T> to_attr);

  void downsample_face_sets(const Mesh &base_mesh,
                            const int from_lvl,
                            const Span<int> from_face_sets,
                            const int to_lvl,
                            const MutableSpan<int> to_face_sets) const;
  void downsample_face_sets(const int from_lvl,
                            const Span<int> from_face_sets,
                            const int to_lvl,
                            const MutableSpan<int> to_face_sets,
                            const Span<int> root_quads) const;

  void downsample_hidden_faces(const Mesh &base_mesh,
                               const int from_lvl,
                               const Span<bool> from_hidden_faces,
                               const int to_lvl,
                               const MutableSpan<bool> to_hidden_faces) const;
  void downsample_hidden_faces(const int from_lvl,
                               const Span<bool> from_hidden_faces,
                               const int to_lvl,
                               const MutableSpan<bool> to_hidden_faces,
                               const Span<int> root_quads) const;

  void subdivide_face_sets(const Mesh &base_mesh,
                           const int from_lvl,
                           const Span<int> from_face_sets,
                           const int to_lvl,
                           const MutableSpan<int> to_face_sets) const;

  void subdivide_hidden_faces(const Mesh &base_mesh,
                              const int from_lvl,
                              const Span<bool> from_hidden_faces,
                              const int to_lvl,
                              const MutableSpan<bool> to_hidden_faces) const;

  static bke::multiresolution::Offsets build_table_offsets(
      const Multiresolution &multires,
      const bke::multiresolution::BaseTopology &base_topo,
      const int lvl);

#endif
};

}  // namespace blender
