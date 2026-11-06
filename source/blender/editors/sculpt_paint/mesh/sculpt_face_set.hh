/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edsculpt
 */

#pragma once

#include "BLI_array.hh"
#include "BLI_offset_indices.hh"
#include "BLI_set.hh"
#include "BLI_span.hh"

#include "BKE_attribute.hh"
#include "BKE_multiresolution.hh"

namespace blender {

struct BMesh;
struct BMFace;
struct BMVert;
struct Mesh;
struct Object;
struct SubdivCCG;
struct SubdivCCGCoord;

namespace bke::multiresolution {
struct MultiresCoord;
}

namespace bke::pbvh {
struct MultiresNode;
}

namespace ed::sculpt_paint::face_set {

int active_face_set_get(const Object &object);
int vert_face_set_get(bke::multiresolution::MultiresCoord coord,
                      const bke::multiresolution::BaseTopology &base_topo,
                      int grid_size,
                      Span<int> face_sets);
bool vert_has_face_set(bke::multiresolution::MultiresCoord coord,
                       const bke::multiresolution::BaseTopology &base_topo,
                       int grid_size,
                       Span<int> face_sets,
                       int face_set);
bool vert_has_unique_face_set(bke::multiresolution::MultiresCoord coord,
                              const bke::multiresolution::BaseTopology &base_topo,
                              int grid_size,
                              Span<int> face_sets);

/* TODO: vert_face_set_max_get should likely be avoided and existing usages cleaned up, since by
 * definition, a vertex can be associated to more than a single face set. */
int vert_face_set_max_get(GroupedSpan<int> vert_to_face_map, Span<int> face_sets, int vert);
int vert_face_set_get(const SubdivCCG &subdiv_ccg, Span<int> face_sets, int grid);
int vert_face_set_max_get(int face_set_offset, const BMVert &vert);

Set<int> vert_face_sets_get(GroupedSpan<int> vert_to_face_map, Span<int> face_sets, int vert);

bool vert_has_face_set(GroupedSpan<int> vert_to_face_map,
                       Span<int> face_sets,
                       int vert,
                       int face_set);
/* TODO: audit the uses of vert_has_face_set for Multires. */
bool vert_has_face_set(const SubdivCCG &subdiv_ccg, Span<int> face_sets, int grid, int face_set);
bool vert_has_face_set(int face_set_offset, const BMVert &vert, int face_set);

bool vert_has_any_face_set(GroupedSpan<int> vert_to_face_map,
                           Span<int> face_sets,
                           int vert,
                           const Set<int> &allowed_face_sets);

bool vert_has_unique_face_set(GroupedSpan<int> vert_to_face_map, Span<int> face_sets, int vert);
bool vert_has_unique_face_set(OffsetIndices<int> faces,
                              Span<int> corner_verts,
                              GroupedSpan<int> vert_to_face_map,
                              Span<int> face_sets,
                              const SubdivCCG &subdiv_ccg,
                              SubdivCCGCoord coord);
bool vert_has_unique_face_set(int face_set_offset, const BMVert &vert);
bool coord_has_face_set(OffsetIndices<int> faces,
                        Span<int> corner_verts,
                        GroupedSpan<int> vert_to_face_map,
                        Span<int> face_sets,
                        const SubdivCCG &subdiv_ccg,
                        SubdivCCGCoord coord,
                        int face_set);
bool coord_has_any_face_set(OffsetIndices<int> faces,
                            Span<int> corner_verts,
                            GroupedSpan<int> vert_to_face_map,
                            Span<int> face_sets,
                            const SubdivCCG &subdiv_ccg,
                            SubdivCCGCoord coord,
                            const Set<int> &allowed_face_sets);

constexpr float FACE_SET_MIN_FADE = 0.05f;

void fill_factors_from_hide_and_mask(const Multiresolution &multires,
                                     const Span<int> subfaces,
                                     const MutableSpan<float> r_factors);
void fill_factor_from_hide_and_mask(const BMesh &bm,
                                    const Set<BMFace *, 0L> &faces,
                                    const MutableSpan<float> r_factors);
void fill_factor_from_hide_and_mask(const Mesh &mesh,
                                    const Span<int> face_indices,
                                    const MutableSpan<float> r_factors);
void calc_face_centers(const OffsetIndices<int> faces,
                       const Span<int> corner_verts,
                       const Span<float3> vert_positions,
                       const Span<int> face_indices,
                       const MutableSpan<float3> positions);
void gather_subfaces(const int faces_per_corner,
                     const OffsetIndices<int> base_faces,
                     const Span<int> node_faces,
                     const MutableSpan<int> node_subfaces);
void calc_face_centers_and_normals(const Span<int> node_subfaces,
                                   const Span<int> corner_verts,
                                   const Span<float3> positions,
                                   MutableSpan<float3> face_centers,
                                   MutableSpan<float3> face_normals);
void calc_face_centers(const Set<BMFace *, 0L> &faces, const MutableSpan<float3> centers);
void calc_face_indices_grids(const SubdivCCG &subdiv_ccg,
                             const Span<int> grids,
                             const MutableSpan<int> &face_indices);

/**
 * Creates the sculpt face set attribute on the mesh if it doesn't exist.
 *
 * \see face_set::ensure_face_sets_mesh if further writing to the attribute is desired.
 */
bool create_face_sets_mesh(Object &object);

/**
 * Ensures that the sculpt face set attribute exists on the mesh.
 *
 * \see face_set::create_face_sets_mesh to avoid having to remember to call .finish()
 */
bke::SpanAttributeWriter<int> ensure_face_sets_mesh(Mesh &mesh);
MutableSpan<int> ensure_face_sets_multires(Mesh &mesh);
int ensure_face_sets_bmesh(Object &object);
Array<int> duplicate_face_sets(const Object &object);

void filter_verts_with_unique_face_sets_mesh(GroupedSpan<int> vert_to_face_map,
                                             Span<int> face_sets,
                                             bool unique,
                                             Span<int> verts,
                                             MutableSpan<float> factors);
void filter_verts_with_unique_face_sets_multires(
    const bke::multiresolution::BaseTopology &base_topo,
    const Span<int> face_sets,
    const bool unique,
    const int grid_size,
    const bke::pbvh::MultiresNode &node,
    const MutableSpan<float> factors);
void filter_verts_with_unique_face_sets_grids(OffsetIndices<int> faces,
                                              Span<int> corner_verts,
                                              GroupedSpan<int> vert_to_face_map,
                                              Span<int> face_sets,
                                              const SubdivCCG &subdiv_ccg,
                                              bool unique,
                                              Span<int> grids,
                                              MutableSpan<float> factors);
void filter_verts_with_unique_face_sets_bmesh(int face_set_offset,
                                              bool unique,
                                              const Set<BMVert *, 0> &verts,
                                              MutableSpan<float> factors);

}  // namespace ed::sculpt_paint::face_set

}  // namespace blender
