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

#include "BLI_math_matrix.hh"

#include "editors/sculpt_paint/mesh/brushes/brushes.hh"
#include "editors/sculpt_paint/mesh/mesh_brush_common.hh"
#include "editors/sculpt_paint/mesh/sculpt_automask.hh"
#include "editors/sculpt_paint/mesh/sculpt_intern.hh"
#include "editors/sculpt_paint/mesh/sculpt_morph.hh"

namespace blender::ed::sculpt_paint::brushes {

inline namespace morph_cc {

static void calc_faces(const Depsgraph &depsgraph,
                       const Sculpt &sd,
                       const Brush &brush,
                       const float4x4 &mat,
                       const MeshAttributeData &attribute_data,
                       const Span<float3> vert_normals,
                       const Span<float3> morph_positions,
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
    const int vert = verts[i];
    translations[i] = (morph_positions[vert] - position_data.eval[vert]) * factors[i] * strength;
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
                          const Span<float3> morph_positions,
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
    translations[i] = (morph_positions[node_range[i]] - positions[i]) * factors[i] * strength;
  }

  clip_and_lock_translations(sd, ss, positions, translations);
  deform_data.deform(node_range, translations);
}

}  // namespace morph_cc

void do_morph_brush(const Depsgraph &depsgraph,
                    const Sculpt &sd,
                    Object &object,
                    const IndexMask &node_mask)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  const Brush &brush = *BKE_paint_brush_for_read(&sd.paint);
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  const float4x4 mat = cube_tip_init(object, brush);

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh: {
      const Span<float3> morph_positions = morph::base_morph_positions(mesh);
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
                       morph_positions,
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
      const Span<float3> morph_positions = multires.attribute<float3>(morph::position_attr_name);
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
                          morph_positions,
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
