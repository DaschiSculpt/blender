/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edsculpt
 */

#include "editors/sculpt_paint/mesh/brushes/brushes.hh"

#include "DNA_brush_types.h"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "BKE_attribute.hh"
#include "BKE_mesh.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_paint_types.hh"
#include "BKE_subdiv_ccg.hh"

#include "BLI_array.hh"
#include "BLI_enumerable_thread_specific.hh"
#include "BLI_task.hh"

#include "editors/sculpt_paint/mesh/mesh_brush_common.hh"
#include "editors/sculpt_paint/mesh/paint_mask.hh"
#include "editors/sculpt_paint/mesh/sculpt_automask.hh"
#include "editors/sculpt_paint/mesh/sculpt_intern.hh"
#include "editors/sculpt_paint/mesh/sculpt_morph.hh"

#include "bmesh.hh"

namespace blender::ed::sculpt_paint::brushes {

inline namespace layer_cc {

struct LocalData {
  Vector<float3> positions;
  Vector<float3> local_positions;
  Vector<float> factors;
  Vector<float> distances;
  Vector<float> masks;
  Vector<float> displacement_factors;
  Vector<float3> translations;
};

BLI_NOINLINE static void offset_displacement_factors(const MutableSpan<float> displacement_factors,
                                                     const Span<float> factors,
                                                     const float strength)
{
  PRF_scope(ProfileCategory::Editor);
  for (const int i : displacement_factors.index_range()) {
    displacement_factors[i] += factors[i] * strength;
  }
}

BLI_NOINLINE static void clamp_displacement_factors(const MutableSpan<float> displacement_factors,
                                                    const Span<float> masks)
{
  PRF_scope(ProfileCategory::Editor);
  if (masks.is_empty()) {
    for (const int i : displacement_factors.index_range()) {
      displacement_factors[i] = std::clamp(displacement_factors[i], -1.0f, 1.0f);
    }
  }
  else {
    for (const int i : displacement_factors.index_range()) {
      const float clamp_mask = 1.0f - masks[i];
      displacement_factors[i] = std::clamp(displacement_factors[i], -clamp_mask, clamp_mask);
    }
  }
}

BLI_NOINLINE static void calc_displacement_factors_from_positions(
    const Span<float3> orig_positions,
    const Span<float3> orig_normals,
    const Span<float3> positions,
    const float height,
    const MutableSpan<float> r_displacement_factors)
{
  PRF_scope(ProfileCategory::Editor);
  for (const int i : positions.index_range()) {
    r_displacement_factors[i] = math::dot(positions[i] - orig_positions[i], orig_normals[i]) /
                                height;
  }
}

BLI_NOINLINE static void calc_morph_snap_translations(const Span<float3> morph_positions,
                                                      const Span<float3> positions,
                                                      const Span<float> factors,
                                                      const float strength,
                                                      const MutableSpan<float3> r_translations)
{
  PRF_scope(ProfileCategory::Editor);
  for (const int i : positions.index_range()) {
    r_translations[i] = (morph_positions[i] - positions[i]) * factors[i] * strength;
  }
}

BLI_NOINLINE static void calc_morph_normals_mesh(const Span<float3> morph_positions,
                                                 const OffsetIndices<int> faces,
                                                 const Span<int> corner_verts,
                                                 const GroupedSpan<int> vert_to_face_map,
                                                 const Span<int> verts,
                                                 const MutableSpan<float3> r_normals)
{
  PRF_scope(ProfileCategory::Editor);
  for (const int i : verts.index_range()) {
    const int vert = verts[i];
    float3 normal(0.0f);
    for (const int face : vert_to_face_map[vert]) {
      normal += bke::mesh::face_normal_calc(morph_positions, corner_verts.slice(faces[face]));
    }
    float length;
    r_normals[i] = math::normalize_and_get_length(normal, length);
    if (length == 0.0f) {
      r_normals[i] = float3(0, 0, 1);
    }
  }
}

BLI_NOINLINE static void calc_translations(const Span<float3> orig_positions,
                                           const Span<float3> orig_normals,
                                           const Span<float3> positions,
                                           const Span<float> displacement_factors,
                                           const Span<float> factors,
                                           const float height,
                                           const MutableSpan<float3> r_translations)
{
  PRF_scope(ProfileCategory::Editor);
  for (const int i : positions.index_range()) {
    const float3 offset = orig_normals[i] * height * displacement_factors[i];
    const float3 translation = orig_positions[i] + offset - positions[i];
    r_translations[i] = translation * factors[i];
  }
}

static void calc_faces(const Depsgraph &depsgraph,
                       const Sculpt &sd,
                       const Brush &brush,
                       const float4x4 &mat,
                       const Mesh &mesh,
                       const MeshAttributeData &attribute_data,
                       const Span<float3> vert_normals,
                       const Span<float3> morph_positions,
                       Object &object,
                       bke::pbvh::MeshNode &node,
                       const PositionDeformData &position_data)
{
  const SculptSession &ss = *object.runtime->sculpt_session;
  const StrokeCache &cache = *ss.cache;

  const Span<int> verts = node.verts();
  const OrigPositionData orig_data = orig_position_data_get_mesh(object, node);
  Array<float3, bke::pbvh::MESH_LEAF_LIMIT> positions(verts.size());
  gather_data_mesh<float3>(position_data.eval, verts, positions);

  const bool has_morph = !morph_positions.is_empty();
  Array<float3, bke::pbvh::MESH_LEAF_LIMIT> morph_base_positions(has_morph ? verts.size() : 0);
  if (has_morph) {
    gather_data_mesh<float3>(morph_positions, verts, morph_base_positions);
  }
  const Span<float3> dist_positions = has_morph ? morph_base_positions.as_span() :
                                                  orig_data.positions;

  Array<float, bke::pbvh::MESH_LEAF_LIMIT> factors(verts.size());
  fill_factor_from_hide_and_mask(attribute_data.hide_vert, attribute_data.mask, verts, factors);
  filter_region_clip_factors(ss, positions, factors);
  if (brush.flag & BRUSH_FRONTFACE) {
    calc_front_face(cache.view_normal_symm, vert_normals, verts, factors);
  }

  Array<float, bke::pbvh::MESH_LEAF_LIMIT> distances(verts.size());
  if (BKE_brush_has_cube_tip(&brush, PaintMode::Sculpt)) {
    Array<float3, bke::pbvh::MESH_LEAF_LIMIT> local_positions(verts.size());
    calc_local_positions(dist_positions,
                         mat,
                         cache.location_symm,
                         cache.view_normal_symm,
                         eBrushFalloffShape(brush.falloff_shape),
                         local_positions);
    calc_brush_cube_distances<float3>(brush, local_positions, distances);
    filter_distances_with_radius(1.0f, distances, factors);
    apply_hardness_to_distances(1.0f, cache.hardness, distances);
    BKE_brush_calc_curve_factors(eBrushCurvePreset(brush.curve_distance_falloff_preset),
                                 brush.curve_distance_falloff,
                                 distances,
                                 1.0f,
                                 factors);
  }
  else {
    calc_brush_distances(ss, dist_positions, eBrushFalloffShape(brush.falloff_shape), distances);
    filter_distances_with_radius(cache.radius, distances, factors);
    apply_hardness_to_distances(cache, distances);
    calc_brush_strength_factors(cache, brush, distances, factors);
  }

  auto_mask::calc_vert_factors(depsgraph, object, cache.automasking.get(), node, verts, factors);

  calc_brush_texture_factors(PaintMode::Sculpt, ss, brush, positions, factors);

  Array<float3, bke::pbvh::MESH_LEAF_LIMIT> translations(verts.size());

  if (brush.height == 0.0f) {
    if (!has_morph) {
      return;
    }
    calc_morph_snap_translations(
        morph_base_positions, positions, factors, cache.bstrength, translations);

    clip_and_lock_translations(sd, ss, position_data.eval, verts, translations);
    position_data.deform(translations, verts);
    return;
  }

  Array<float3, bke::pbvh::MESH_LEAF_LIMIT> morph_normals(has_morph ? verts.size() : 0);
  if (has_morph) {
    calc_morph_normals_mesh(morph_positions,
                            mesh.faces(),
                            mesh.corner_verts(),
                            mesh.vert_to_face_map(),
                            verts,
                            morph_normals);
  }
  const Span<float3> base_positions = has_morph ? morph_base_positions.as_span() :
                                                  orig_data.positions;
  const Span<float3> base_normals = has_morph ? morph_normals.as_span() : orig_data.normals;

  Array<float, bke::pbvh::MESH_LEAF_LIMIT> masks(attribute_data.mask.is_empty() ? 0 :
                                                                                  verts.size());
  if (!attribute_data.mask.is_empty()) {
    gather_data_mesh<float>(attribute_data.mask, verts, masks);
  }

  Array<float, bke::pbvh::MESH_LEAF_LIMIT> displacement_factors(verts.size());
  calc_displacement_factors_from_positions(
      base_positions, base_normals, positions, brush.height, displacement_factors);
  offset_displacement_factors(displacement_factors, factors, cache.bstrength);
  clamp_displacement_factors(displacement_factors, masks);

  calc_translations(base_positions,
                    base_normals,
                    positions,
                    displacement_factors,
                    factors,
                    brush.height,
                    translations);

  clip_and_lock_translations(sd, ss, position_data.eval, verts, translations);
  position_data.deform(translations, verts);
}

static void calc_multires(const Depsgraph &depsgraph,
                          const Sculpt &sd,
                          const Brush &brush,
                          const float4x4 &mat,
                          Object &object,
                          Multiresolution &multires,
                          const bke::multiresolution::BaseTopology &base_topo,
                          bke::pbvh::MultiresNode &node,
                          const MultiresDeformData &deform_data,
                          const bke::multiresolution::OffsetsSpan &offsets,
                          const Span<float3> morph_positions)
{
  const SculptSession &ss = *object.runtime->sculpt_session;
  const StrokeCache &cache = *ss.cache;

  const int grid_size = multires.grid_size();
  const IndexRange node_range = node.range(grid_size);
  const Span<float3> positions = deform_data.positions.slice(node_range);
  const Span<float3> normals = multires.normals().slice(node_range);

  const OrigPositionData orig_data = orig_position_data_get_multires(object, node);

  const bool has_morph = !morph_positions.is_empty();
  const Span<float3> dist_positions = has_morph ? morph_positions.slice(node_range) :
                                                  orig_data.positions;

  Array<float, bke::pbvh::MULTIRES_LEAF_LIMIT> factors(positions.size());
  fill_factor_from_hide_and_mask(multires.hidden_verts(), multires.masks(), node_range, factors);
  filter_region_clip_factors(ss, positions, factors);
  if (brush.flag & BRUSH_FRONTFACE) {
    calc_front_face(cache.view_normal_symm, normals, factors);
  }

  Array<float, bke::pbvh::MULTIRES_LEAF_LIMIT> distances(positions.size());
  if (BKE_brush_has_cube_tip(&brush, PaintMode::Sculpt)) {
    Array<float3, bke::pbvh::MESH_LEAF_LIMIT> local_positions(positions.size());
    calc_local_positions(dist_positions,
                         mat,
                         cache.location_symm,
                         cache.view_normal_symm,
                         eBrushFalloffShape(brush.falloff_shape),
                         local_positions);
    calc_brush_cube_distances<float3>(brush, local_positions, distances);
    filter_distances_with_radius(1.0f, distances, factors);
    apply_hardness_to_distances(1.0f, cache.hardness, distances);
    BKE_brush_calc_curve_factors(eBrushCurvePreset(brush.curve_distance_falloff_preset),
                                 brush.curve_distance_falloff,
                                 distances,
                                 1.0f,
                                 factors);
  }
  else {
    calc_brush_distances(ss, dist_positions, eBrushFalloffShape(brush.falloff_shape), distances);
    filter_distances_with_radius(cache.radius, distances, factors);
    apply_hardness_to_distances(cache, distances);
    calc_brush_strength_factors(cache, brush, distances, factors);
  }

  auto_mask::calc_multires_factors(
      depsgraph, object, multires, base_topo, cache.automasking.get(), node, factors);

  calc_brush_texture_factors(PaintMode::Sculpt, ss, brush, positions, factors);

  Array<float3, bke::pbvh::MULTIRES_LEAF_LIMIT> translations(positions.size());

  if (brush.height == 0.0f) {
    if (!has_morph) {
      return;
    }
    calc_morph_snap_translations(
        morph_positions.slice(node_range), positions, factors, cache.bstrength, translations);

    clip_and_lock_translations(sd, ss, positions, translations);
    deform_data.deform(node_range, translations);
    return;
  }

  Array<float3, bke::pbvh::MULTIRES_LEAF_LIMIT> morph_normals;
  if (has_morph) {
    morph_normals.reinitialize(node.owned_verts_num(grid_size));
    const int virtual_lvl = multires.curr_level - multires.base_level(multires.curr_level);
    bke::pbvh::calc_normals_multires_node(base_topo,
                                          offsets,
                                          virtual_lvl,
                                          multires.corner_verts(),
                                          morph_positions,
                                          node,
                                          morph_normals);
  }
  const Span<float3> base_positions = has_morph ? morph_positions.slice(node_range) :
                                                  orig_data.positions;
  const Span<float3> base_normals = has_morph ? morph_normals.as_span() : orig_data.normals;

  const Span<float> masks = multires.masks().is_empty() ? Span<float>() :
                                                          multires.masks().slice(node_range);

  Array<float, bke::pbvh::MULTIRES_LEAF_LIMIT> displacement_factors(positions.size());
  calc_displacement_factors_from_positions(
      base_positions, base_normals, positions, brush.height, displacement_factors);
  offset_displacement_factors(displacement_factors, factors, cache.bstrength);
  clamp_displacement_factors(displacement_factors, masks);

  calc_translations(base_positions,
                    base_normals,
                    positions,
                    displacement_factors,
                    factors,
                    brush.height,
                    translations);

  clip_and_lock_translations(sd, ss, positions, translations);
  deform_data.deform(node_range, translations);
}

static void calc_grids(const Depsgraph &depsgraph,
                       const Sculpt &sd,
                       const Brush &brush,
                       const float4x4 &mat,
                       Object &object,
                       bke::pbvh::GridsNode &node,
                       LocalData &tls)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  const StrokeCache &cache = *ss.cache;
  SubdivCCG &subdiv_ccg = *ss.subdiv_ccg;

  if (brush.height == 0.0f) {
    return;
  }

  const Span<int> grids = node.grids();
  const OrigPositionData orig_data = orig_position_data_get_grids(object, node);
  const MutableSpan positions = gather_grids_positions(subdiv_ccg, grids, tls.positions);

  tls.factors.resize(positions.size());
  const MutableSpan<float> factors = tls.factors;
  fill_factor_from_hide_and_mask(subdiv_ccg, grids, factors);
  filter_region_clip_factors(ss, positions, factors);
  if (brush.flag & BRUSH_FRONTFACE) {
    calc_front_face(cache.view_normal_symm, subdiv_ccg, grids, factors);
  }

  tls.distances.resize(positions.size());
  const MutableSpan<float> distances = tls.distances;

  if (BKE_brush_has_cube_tip(&brush, PaintMode::Sculpt)) {
    tls.local_positions.resize(orig_data.positions.size());
    const MutableSpan<float3> local_positions = tls.local_positions;
    calc_local_positions(orig_data.positions,
                         mat,
                         cache.location_symm,
                         cache.view_normal_symm,
                         eBrushFalloffShape(brush.falloff_shape),
                         local_positions);
    calc_brush_cube_distances<float3>(brush, local_positions, distances);
    filter_distances_with_radius(1.0f, distances, factors);
    apply_hardness_to_distances(1.0f, cache.hardness, distances);
    BKE_brush_calc_curve_factors(eBrushCurvePreset(brush.curve_distance_falloff_preset),
                                 brush.curve_distance_falloff,
                                 distances,
                                 1.0f,
                                 factors);
  }
  else {
    calc_brush_distances(
        ss, orig_data.positions, eBrushFalloffShape(brush.falloff_shape), distances);
    filter_distances_with_radius(cache.radius, distances, factors);
    apply_hardness_to_distances(cache.radius, cache.hardness, distances);
    BKE_brush_calc_curve_factors(eBrushCurvePreset(brush.curve_distance_falloff_preset),
                                 brush.curve_distance_falloff,
                                 distances,
                                 cache.radius,
                                 factors);
  }

  auto_mask::calc_grids_factors(depsgraph, object, cache.automasking.get(), node, grids, factors);

  calc_brush_texture_factors(PaintMode::Sculpt, ss, brush, positions, factors);

  if (subdiv_ccg.masks.is_empty()) {
    tls.masks.clear();
  }
  else {
    tls.masks.resize(positions.size());
    gather_data_grids(subdiv_ccg, subdiv_ccg.masks.as_span(), grids, tls.masks.as_mutable_span());
  }
  const MutableSpan<float> masks = tls.masks;

  tls.displacement_factors.resize(positions.size());
  const MutableSpan<float> displacement_factors = tls.displacement_factors;
  calc_displacement_factors_from_positions(
      orig_data.positions, orig_data.normals, positions, brush.height, displacement_factors);
  offset_displacement_factors(displacement_factors, tls.factors, cache.bstrength);
  clamp_displacement_factors(displacement_factors, masks);

  tls.translations.resize(positions.size());
  const MutableSpan<float3> translations = tls.translations;
  calc_translations(orig_data.positions,
                    orig_data.normals,
                    positions,
                    displacement_factors,
                    tls.factors,
                    brush.height,
                    translations);

  clip_and_lock_translations(sd, ss, positions, translations);
  apply_translations(translations, grids, subdiv_ccg);
}

static void calc_bmesh(const Depsgraph &depsgraph,
                       const Sculpt &sd,
                       const Brush &brush,
                       const float4x4 &mat,
                       Object &object,
                       bke::pbvh::BMeshNode &node,
                       LocalData &tls)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  const StrokeCache &cache = *ss.cache;

  if (brush.height == 0.0f) {
    return;
  }

  const Set<BMVert *, 0> &verts = BKE_pbvh_bmesh_node_unique_verts(&node);

  Array<float3> orig_positions(verts.size());
  Array<float3> orig_normals(verts.size());
  orig_position_data_gather_bmesh(*ss.bm_log, verts, orig_positions, orig_normals);

  const MutableSpan positions = gather_bmesh_positions(verts, tls.positions);

  tls.factors.resize(verts.size());
  const MutableSpan<float> factors = tls.factors;
  fill_factor_from_hide_and_mask(*ss.bm, verts, factors);
  filter_region_clip_factors(ss, positions, factors);
  if (brush.flag & BRUSH_FRONTFACE) {
    calc_front_face(cache.view_normal_symm, verts, factors);
  }

  tls.distances.resize(verts.size());
  const MutableSpan<float> distances = tls.distances;

  if (BKE_brush_has_cube_tip(&brush, PaintMode::Sculpt)) {
    tls.local_positions.resize(orig_positions.size());
    const MutableSpan<float3> local_positions = tls.local_positions;
    calc_local_positions(orig_positions,
                         mat,
                         cache.location_symm,
                         cache.view_normal_symm,
                         eBrushFalloffShape(brush.falloff_shape),
                         local_positions);
    calc_brush_cube_distances<float3>(brush, local_positions, distances);
    filter_distances_with_radius(1.0f, distances, factors);
    apply_hardness_to_distances(1.0f, cache.hardness, distances);
    BKE_brush_calc_curve_factors(eBrushCurvePreset(brush.curve_distance_falloff_preset),
                                 brush.curve_distance_falloff,
                                 distances,
                                 1.0f,
                                 factors);
  }
  else {
    calc_brush_distances(ss, orig_positions, eBrushFalloffShape(brush.falloff_shape), distances);
    filter_distances_with_radius(cache.radius, distances, factors);
    apply_hardness_to_distances(cache.radius, cache.hardness, distances);
    BKE_brush_calc_curve_factors(eBrushCurvePreset(brush.curve_distance_falloff_preset),
                                 brush.curve_distance_falloff,
                                 distances,
                                 cache.radius,
                                 factors);
  }

  auto_mask::calc_vert_factors(depsgraph, object, cache.automasking.get(), node, verts, factors);

  calc_brush_texture_factors(PaintMode::Sculpt, ss, brush, positions, factors);

  tls.displacement_factors.resize(verts.size());
  const MutableSpan<float> displacement_factors = tls.displacement_factors;
  calc_displacement_factors_from_positions(
      orig_positions, orig_normals, positions, brush.height, displacement_factors);
  offset_displacement_factors(displacement_factors, tls.factors, cache.bstrength);

  tls.masks.resize(verts.size());
  const MutableSpan<float> masks = tls.masks;
  mask::gather_mask_bmesh(*ss.bm, verts, masks);
  clamp_displacement_factors(displacement_factors, masks);

  tls.translations.resize(verts.size());
  const MutableSpan<float3> translations = tls.translations;
  calc_translations(orig_positions,
                    orig_normals,
                    positions,
                    displacement_factors,
                    tls.factors,
                    brush.height,
                    translations);

  clip_and_lock_translations(sd, ss, positions, translations);
  apply_translations(translations, verts);
}

}  // namespace layer_cc

void do_layer_brush(const Depsgraph &depsgraph,
                    const Sculpt &sd,
                    Object &object,
                    const IndexMask &node_mask)
{
  PRF_scope(ProfileCategory::Editor);
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  const Brush &brush = *BKE_paint_brush_for_read(&sd.paint);

  threading::EnumerableThreadSpecific<LocalData> all_tls;

  const float4x4 mat = cube_tip_init(object, brush);

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh: {
      Mesh &mesh = *id_cast<Mesh *>(object.data);
      const PositionDeformData position_data(depsgraph, object);
      const Span<float3> vert_normals = bke::pbvh::vert_normals_eval(depsgraph, object);

      const MeshAttributeData attribute_data(mesh);

      Span<float3> morph_positions;
      if ((brush.flag & BRUSH_USE_MORPH_AS_BASE) && morph::morph_exists(object)) {
        morph_positions = morph::base_morph_positions(mesh);
      }

      MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
      node_mask.foreach_index(
          [&](const int i) {
            calc_faces(depsgraph,
                       sd,
                       brush,
                       mat,
                       mesh,
                       attribute_data,
                       vert_normals,
                       morph_positions,
                       object,
                       nodes[i],
                       position_data);
            bke::pbvh::update_node_bounds_mesh(position_data.eval, nodes[i]);
          },
          exec_mode::grain_size(1));
      break;
    }
    case bke::pbvh::Type::Multires: {
      Mesh &base_mesh = *id_cast<Mesh *>(object.data);
      Multiresolution multires = *base_mesh.multires;
      const bke::multiresolution::BaseTopology &base_topo = multires.get_base_topology(base_mesh);
      const MultiresDeformData deform_data(base_mesh);

      Span<float3> morph_positions;
      if ((brush.flag & BRUSH_USE_MORPH_AS_BASE) && morph::morph_exists(object)) {
        morph_positions = multires.attribute<float3>(morph::position_attr_name);
      }

      MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
      node_mask.foreach_index(
          [&](const int i) {
            calc_multires(depsgraph,
                          sd,
                          brush,
                          mat,
                          object,
                          multires,
                          base_topo,
                          nodes[i],
                          deform_data,
                          pbvh.offsets,
                          morph_positions);
            bke::pbvh::update_node_bounds_multires(
                multires.grid_size(), multires.positions(), pbvh.offsets, nodes[i]);
          },
          exec_mode::grain_size(1));
      break;
    }
    case bke::pbvh::Type::Grids: {
      SubdivCCG &subdiv_ccg = *object.runtime->sculpt_session->subdiv_ccg;
      MutableSpan<float3> positions = subdiv_ccg.positions;

      MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
      node_mask.foreach_index(
          [&](const int i) {
            LocalData &tls = all_tls.local();
            calc_grids(depsgraph, sd, brush, mat, object, nodes[i], tls);
            bke::pbvh::update_node_bounds_grids(subdiv_ccg.grid_area, positions, nodes[i]);
          },
          exec_mode::grain_size(1));
      break;
    }
    case bke::pbvh::Type::BMesh: {
      MutableSpan<bke::pbvh::BMeshNode> nodes = pbvh.nodes<bke::pbvh::BMeshNode>();
      node_mask.foreach_index(
          [&](const int i) {
            LocalData &tls = all_tls.local();
            calc_bmesh(depsgraph, sd, brush, mat, object, nodes[i], tls);
            bke::pbvh::update_node_bounds_bmesh(nodes[i]);
          },
          exec_mode::grain_size(1));
      break;
    }
  }
  pbvh.tag_positions_changed(node_mask);
  pbvh.flush_bounds_to_parents();
}

}  // namespace blender::ed::sculpt_paint::brushes
