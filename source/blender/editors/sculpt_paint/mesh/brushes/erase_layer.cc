/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "DNA_brush_types.h"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "BKE_attribute.hh"
#include "BKE_mesh.hh"
#include "BKE_multiresolution.hh"
#include "BKE_object.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_paint_types.hh"
#include "BKE_sculpt_layer.hh"

#include "BLI_math_matrix.hh"

#include "editors/sculpt_paint/mesh/brushes/brushes.hh"
#include "editors/sculpt_paint/mesh/mesh_brush_common.hh"
#include "editors/sculpt_paint/mesh/sculpt_automask.hh"
#include "editors/sculpt_paint/mesh/sculpt_intern.hh"

namespace blender::ed::sculpt_paint::brushes {

inline namespace erase_layer_cc {

static void calc_faces(const Depsgraph &depsgraph,
                       const Sculpt &sd,
                       const Brush &brush,
                       const float4x4 &mat,
                       const MeshAttributeData &attribute_data,
                       const Span<float3> vert_normals,
                       const Span<float3> layer_displacements,
                       const float layer_opacity,
                       const bke::pbvh::MeshNode &node,
                       Object &object,
                       const PositionDeformData &position_data)
{
  const SculptSession &ss = *object.runtime->sculpt_session;
  const Span<int> verts = node.verts();
  const float strength = ss.cache->bstrength;
  Array<float, bke::pbvh::MESH_LEAF_LIMIT> factors(verts.size());
  Array<float, bke::pbvh::MESH_LEAF_LIMIT> distances(verts.size());

  if (BKE_brush_has_cube_tip(&brush, PaintMode::Sculpt)) {
    Array<float3, bke::pbvh::MESH_LEAF_LIMIT> local_positions(verts.size());
    calc_cube_tip_factors_common_mesh_indexed(depsgraph,
                                              brush,
                                              object,
                                              mat,
                                              attribute_data,
                                              position_data.eval,
                                              vert_normals,
                                              local_positions,
                                              node,
                                              factors,
                                              distances);
  }
  else {
    calc_factors_common_mesh_indexed(depsgraph,
                                     brush,
                                     object,
                                     attribute_data,
                                     position_data.eval,
                                     vert_normals,
                                     node,
                                     factors,
                                     distances);
  }

  Array<float3, bke::pbvh::MESH_LEAF_LIMIT> translations(verts.size());
  for (const int i : verts.index_range()) {
    translations[i] = -layer_displacements[verts[i]] * layer_opacity * factors[i] * strength;
  }

  clip_and_lock_translations(sd, ss, position_data.eval, verts, translations);
  position_data.deform(translations, verts);
}

static void calc_multires(const Depsgraph &depsgraph,
                          const Sculpt &sd,
                          Multiresolution &multires,
                          const bke::multiresolution::BaseTopology &base_topo,
                          Object &object,
                          const Brush &brush,
                          const float4x4 &mat,
                          const Span<float3> layer_displacements,
                          const float layer_opacity,
                          const bke::pbvh::MultiresNode &node,
                          const MultiresDeformData &deform_data)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  const float strength = ss.cache->bstrength;
  const int grid_size = multires.grid_size();
  const IndexRange node_range = node.range(grid_size);
  const Span<float3> positions = deform_data.positions.slice(node_range);
  const Span<float3> normals = multires.normals().slice(node_range);

  Array<float, bke::pbvh::MULTIRES_LEAF_LIMIT> factors(positions.size());
  Array<float, bke::pbvh::MULTIRES_LEAF_LIMIT> distances(positions.size());

  if (BKE_brush_has_cube_tip(&brush, PaintMode::Sculpt)) {
    Array<float3, bke::pbvh::MULTIRES_LEAF_LIMIT> local_positions(positions.size());
    calc_cube_tip_factors_common_multires(depsgraph,
                                          brush,
                                          object,
                                          mat,
                                          multires,
                                          base_topo,
                                          positions,
                                          normals,
                                          local_positions,
                                          node,
                                          factors,
                                          distances);
  }
  else {
    calc_factors_common_multires(depsgraph,
                                 brush,
                                 object,
                                 multires,
                                 base_topo,
                                 positions,
                                 normals,
                                 node,
                                 factors,
                                 distances);
  }

  Array<float3, bke::pbvh::MULTIRES_LEAF_LIMIT> translations(positions.size());
  for (const int i : positions.index_range()) {
    translations[i] = -layer_displacements[node_range[i]] * layer_opacity * factors[i] * strength;
  }

  clip_and_lock_translations(sd, ss, positions, translations);
  deform_data.deform(node_range, translations);
}

}  // namespace erase_layer_cc

void do_erase_layer_brush(const Depsgraph &depsgraph,
                          const Sculpt &sd,
                          Object &object,
                          const IndexMask &node_mask)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  const Brush &brush = *BKE_paint_brush_for_read(&sd.paint);
  Mesh &mesh = *id_cast<Mesh *>(object.data);

  const SculptLayer *active_layer = bke::sculpt_layer::active_layer(mesh);
  if (active_layer == nullptr || !active_layer->is_visible || active_layer->opacity == 0.0f) {
    return;
  }
  const float layer_opacity = active_layer->opacity;
  const Span<float3> layer_displacements = bke::sculpt_layer::active_layer_displacements(mesh);
  const float4x4 mat = cube_tip_init(object, brush);

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh: {
      const MeshAttributeData attribute_data(mesh);
      const PositionDeformData position_data(depsgraph, object);
      const Span<float3> vert_normals = bke::pbvh::vert_normals_eval(depsgraph, object);

      MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
      node_mask.foreach_index(
          [&](const int i) {
            calc_faces(depsgraph,
                       sd,
                       brush,
                       mat,
                       attribute_data,
                       vert_normals,
                       layer_displacements,
                       layer_opacity,
                       nodes[i],
                       object,
                       position_data);
            bke::pbvh::update_node_bounds_mesh(position_data.eval, nodes[i]);
          },
          exec_mode::grain_size(1));
      break;
    }
    case bke::pbvh::Type::Multires: {
      Multiresolution &multires = *mesh.multires;
      const bke::multiresolution::BaseTopology &base_topo = multires.get_base_topology(mesh);
      const MultiresDeformData deform_data(mesh);

      MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
      node_mask.foreach_index(
          [&](const int i) {
            calc_multires(depsgraph,
                          sd,
                          multires,
                          base_topo,
                          object,
                          brush,
                          mat,
                          layer_displacements,
                          layer_opacity,
                          nodes[i],
                          deform_data);
            bke::pbvh::update_node_bounds_multires(
                multires.grid_size(), multires.positions(), pbvh.offsets, nodes[i]);
          },
          exec_mode::grain_size(1));
      break;
    }
    case bke::pbvh::Type::Grids:
    case bke::pbvh::Type::BMesh: {
      break;
    }
  }
  pbvh.tag_positions_changed(node_mask);
  pbvh.flush_bounds_to_parents();
}

}  // namespace blender::ed::sculpt_paint::brushes
