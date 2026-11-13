/* SPDX-FileCopyrightText: 2020 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edsculpt
 */
#include "sculpt_face_set.hh"

#include <cmath>
#include <cstdlib>
#include <queue>

#include "MEM_guardedalloc.h"

#include "BLI_array.hh"
#include "BLI_array_utils.hh"
#include "BLI_bit_vector.hh"
#include "BLI_enumerable_thread_specific.hh"
#include "BLI_function_ref.hh"
#include "BLI_hash_c.hh"
#include "BLI_math_matrix_c.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_c.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "BLT_translation.hh"

#include "DNA_customdata_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "BKE_attribute.hh"
#include "BKE_ccg.hh"
#include "BKE_context.hh"
#include "BKE_customdata.hh"
#include "BKE_layer.hh"
#include "BKE_mesh.hh"
#include "BKE_mesh_fair.hh"
#include "BKE_mesh_mapping.hh"
#include "BKE_object.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_paint_types.hh"
#include "BKE_subdiv_ccg.hh"

#include "DEG_depsgraph.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "ED_sculpt.hh"

#include "mesh_brush_common.hh"
#include "paint_hide.hh"
#include "sculpt_boundary.hh"
#include "sculpt_gesture.hh"
#include "sculpt_intern.hh"
#include "sculpt_islands.hh"
#include "sculpt_undo.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "bmesh.hh"

namespace blender::ed::sculpt_paint::face_set {

/* -------------------------------------------------------------------- */
/** \name Public API
 * \{ */

int find_next_available_id(const Mesh &mesh)
{
  const bke::AttributeAccessor attributes = mesh.attributes();
  const VArraySpan<int> face_sets = *attributes.lookup<int>(".sculpt_face_set",
                                                            bke::AttrDomain::Face);
  const int max = threading::parallel_reduce(
      face_sets.index_range(),
      4096,
      1,
      [&](const IndexRange range, int max) {
        for (const int id : face_sets.slice(range)) {
          max = std::max(max, id);
        }
        return max;
      },
      [](const int a, const int b) { return std::max(a, b); });
  return max + 1;
}

int find_next_available_id(Object &object)
{

  auto max_face_set = [](const Span<int> face_sets) {
    return threading::parallel_reduce(
        face_sets.index_range(),
        4096,
        1,
        [&](const IndexRange range, int max) {
          for (const int id : face_sets.slice(range)) {
            max = std::max(max, id);
          }
          return max;
        },
        [](const int a, const int b) { return std::max(a, b); });
  };

  SculptSession &ss = *object.runtime->sculpt_session;
  switch (bke::object::pbvh_get(object)->type()) {
    case bke::pbvh::Type::Mesh:
    case bke::pbvh::Type::Grids: {
      Mesh &mesh = *id_cast<Mesh *>(object.data);
      return find_next_available_id(mesh);
    }
    case bke::pbvh::Type::Multires: {
      const Mesh &base_mesh = *id_cast<Mesh *>(object.data);
      const Multiresolution &multires = *base_mesh.multires;
      const Span<int> face_sets = multires.face_sets();
      return (face_sets.is_empty() ? 2 : max_face_set(face_sets)) + 1;
    }
    case bke::pbvh::Type::BMesh: {
      BMesh &bm = *ss.bm;
      const int cd_offset = CustomData_get_offset_named(
          &bm.pdata, CD_PROP_INT32, ".sculpt_face_set");
      if (cd_offset == -1) {
        /* Default face set ID is 1, so the next available id should never be 1 */
        return 2;
      }
      int next_face_set = 1;
      BMIter iter;
      BMFace *f;
      BM_ITER_MESH (f, &iter, &bm, BM_FACES_OF_MESH) {
        const int fset = *static_cast<const int *>(POINTER_OFFSET(f->head.data, cd_offset));
        next_face_set = std::max(next_face_set, fset);
      }

      return next_face_set + 1;
    }
  }
  BLI_assert_unreachable();
  return 0;
}

void initialize_none_to_id(Mesh *mesh, const int new_id)
{
  bke::MutableAttributeAccessor attributes = mesh->attributes_for_write();
  bke::SpanAttributeWriter<int> face_sets = attributes.lookup_for_write_span<int>(
      ".sculpt_face_set");
  if (!face_sets) {
    return;
  }

  for (const int i : face_sets.span.index_range()) {
    if (face_sets.span[i] == face_set_none_id) {
      face_sets.span[i] = new_id;
    }
  }
  face_sets.finish();
}

int active_update_and_get(bContext *C, Object &ob, const float mval[2])
{
  if (!ob.runtime->sculpt_session) {
    return face_set_none_id;
  }

  if (!cursor_geometry_info_update(C, mval, false)) {
    return face_set_none_id;
  }

  return active_face_set_get(ob);
}

bool create_face_sets_mesh(Object &object)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  if (attributes.contains(".sculpt_face_set")) {
    return false;
  }
  attributes.add<int>(".sculpt_face_set", bke::AttrDomain::Face, bke::AttributeInitValue(1));
  mesh.face_sets_color_default = 1;
  return true;
}

bool create_face_sets_multires(Object &object)
{
  Mesh &base_mesh = *id_cast<Mesh *>(object.data);
  Multiresolution &multires = *base_mesh.multires;
  if (!multires.face_sets().is_empty()) {
    return false;
  }

  multires.face_sets_for_write().fill(1);
  base_mesh.face_sets_color_default = 1;
  return true;
}

bke::SpanAttributeWriter<int> ensure_face_sets_mesh(Mesh &mesh)
{
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  if (!attributes.contains(".sculpt_face_set")) {
    attributes.add<int>(".sculpt_face_set", bke::AttrDomain::Face, bke::AttributeInitValue(1));
    mesh.face_sets_color_default = 1;
  }
  return attributes.ensure_for_write_span<int>(".sculpt_face_set", bke::AttrDomain::Face);
}

MutableSpan<int> ensure_face_sets_multires(Mesh &base_mesh)
{
  Multiresolution &multires = *base_mesh.multires;
  if (multires.face_sets().is_empty()) {
    multires.face_sets_for_write().fill(1);
    base_mesh.face_sets_color_default = 1;
  }
  return multires.face_sets_for_write();
}

int ensure_face_sets_bmesh(Object &object)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  SculptSession &ss = *object.runtime->sculpt_session;
  BMesh &bm = *ss.bm;
  if (!CustomData_has_layer_named(&bm.pdata, CD_PROP_INT32, ".sculpt_face_set")) {
    BM_data_layer_add_named(&bm, &bm.pdata, CD_PROP_INT32, ".sculpt_face_set");
    const int offset = CustomData_get_offset_named(&bm.pdata, CD_PROP_INT32, ".sculpt_face_set");
    if (offset == -1) {
      return -1;
    }
    BMIter iter;
    BMFace *face;
    BM_ITER_MESH (face, &iter, &bm, BM_FACES_OF_MESH) {
      BM_ELEM_CD_SET_INT(face, offset, 1);
    }
    mesh.face_sets_color_default = 1;
    return offset;
  }
  return CustomData_get_offset_named(&bm.pdata, CD_PROP_INT32, ".sculpt_face_set");
}

Array<int> duplicate_face_sets(const Object &object)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  const bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);

  if (pbvh.type() == bke::pbvh::Type::Multires) {
    const Multiresolution &multires = *mesh.multires;
    Array<int> face_sets(multires.faces_num());
    if (multires.face_sets().is_empty()) {
      face_sets.fill(0);
    }
    else {
      array_utils::copy(multires.face_sets(), face_sets.as_mutable_span());
    }
    return face_sets;
  }
  else {
    const bke::AttributeAccessor attributes = mesh.attributes();
    const VArray<int> attribute = *attributes.lookup_or_default(
        ".sculpt_face_set", bke::AttrDomain::Face, 0);
    Array<int> face_sets(attribute.size());
    array_utils::copy(attribute, face_sets.as_mutable_span());
    return face_sets;
  }
}

void filter_verts_with_unique_face_sets_mesh(const GroupedSpan<int> vert_to_face_map,
                                             const Span<int> face_sets,
                                             const bool unique,
                                             const Span<int> verts,
                                             const MutableSpan<float> factors)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(verts.size() == factors.size());

  for (const int i : verts.index_range()) {
    if (unique == face_set::vert_has_unique_face_set(vert_to_face_map, face_sets, verts[i])) {
      factors[i] = 0.0f;
    }
  }
}

void filter_verts_with_unique_face_sets_multires(
    const bke::multiresolution::BaseTopology &base_topo,
    const Span<int> face_sets,
    const bool unique,
    const int grid_size,
    const bke::pbvh::MultiresNode &node,
    const MutableSpan<float> factors)
{
  using MultiresCoord = bke::multiresolution::MultiresCoord;

  node.foreach_coord(grid_size, [&](const int node_vert, MultiresCoord coord) {
    if (unique == face_set::vert_has_unique_face_set(coord, base_topo, grid_size, face_sets)) {
      factors[node_vert] = 0.0f;
    }
  });
}

void filter_verts_with_unique_face_sets_grids(const OffsetIndices<int> faces,
                                              const Span<int> corner_verts,
                                              const GroupedSpan<int> vert_to_face_map,
                                              const Span<int> face_sets,
                                              const SubdivCCG &subdiv_ccg,
                                              const bool unique,
                                              const Span<int> grids,
                                              const MutableSpan<float> factors)
{
  PRF_scope(ProfileCategory::Editor);
  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);
  BLI_assert(grids.size() * key.grid_area == factors.size());

  for (const int i : grids.index_range()) {
    const int node_start = i * key.grid_area;
    for (const int y : IndexRange(key.grid_size)) {
      for (const int x : IndexRange(key.grid_size)) {
        const int offset = CCG_grid_xy_to_index(key.grid_size, x, y);
        const int node_vert = node_start + offset;
        if (factors[node_vert] == 0.0f) {
          continue;
        }

        SubdivCCGCoord coord{};
        coord.grid_index = grids[i];
        coord.x = x;
        coord.y = y;
        if (unique == face_set::vert_has_unique_face_set(
                          faces, corner_verts, vert_to_face_map, face_sets, subdiv_ccg, coord))
        {
          factors[node_vert] = 0.0f;
        }
      }
    }
  }
}

void filter_verts_with_unique_face_sets_bmesh(int face_set_offset,
                                              const bool unique,
                                              const Set<BMVert *, 0> &verts,
                                              const MutableSpan<float> factors)
{
  PRF_scope(ProfileCategory::Editor);
  BLI_assert(verts.size() == factors.size());

  int i = 0;
  for (const BMVert *vert : verts) {
    if (unique == face_set::vert_has_unique_face_set(face_set_offset, *vert)) {
      factors[i] = 0.0f;
    }
    i++;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Global Mesh Operators
 * Operators that work on the mesh as a whole.
 * \{ */

void gather_subfaces(const int faces_per_corner,
                     const OffsetIndices<int> base_faces,
                     const Span<int> node_faces,
                     const MutableSpan<int> node_subfaces)
{
  int offset = 0;
  for (const int face_idx : node_faces) {
    const IndexRange face_range = base_faces[face_idx];
    const int first_subface_idx = face_range.start() * faces_per_corner;
    const int subfaces_num = face_range.size() * faces_per_corner;
    array_utils::fill_index_range(node_subfaces.slice(offset, subfaces_num), first_subface_idx);
    offset += subfaces_num;
  }
}

void calc_face_centers_and_normals(const Span<int> node_subfaces,
                                   const Span<int> corner_verts,
                                   const Span<float3> positions,
                                   const MutableSpan<float3> face_centers,
                                   const MutableSpan<float3> face_normals)
{
  for (const int i : node_subfaces.index_range()) {
    const int sf = node_subfaces[i];
    const int v0 = corner_verts[sf * 4 + 0];
    const int v1 = corner_verts[sf * 4 + 1];
    const int v2 = corner_verts[sf * 4 + 2];
    const int v3 = corner_verts[sf * 4 + 3];

    const float3 &p0 = positions[v0];
    const float3 &p1 = positions[v1];
    const float3 &p2 = positions[v2];
    const float3 &p3 = positions[v3];
    face_centers[i] = (p0 + p1 + p2 + p3) * 0.25f;
    face_normals[i] = math::normalize(math::cross(p2 - p0, p3 - p1));
  }
}

static void face_sets_update(
    const Depsgraph &depsgraph,
    Object &object,
    const IndexMask &node_mask,
    const FunctionRef<void(int, Span<int>, MutableSpan<int>)> calc_face_sets)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  Mesh &mesh = *id_cast<Mesh *>(object.data);

  struct TLS {
    Vector<int> face_indices;
    Vector<int> new_face_sets;
  };

  Array<bool> node_changed(pbvh.nodes_num(), false);

  threading::EnumerableThreadSpecific<TLS> all_tls;
  if (pbvh.type() == bke::pbvh::Type::Mesh) {
    bke::SpanAttributeWriter<int> face_sets = ensure_face_sets_mesh(mesh);
    MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
    node_mask.foreach_index(
        [&](const int i) {
          TLS &tls = all_tls.local();
          const Span<int> faces = nodes[i].faces();

          tls.new_face_sets.resize(faces.size());
          MutableSpan<int> new_face_sets = tls.new_face_sets;
          gather_data_mesh(face_sets.span.as_span(), faces, new_face_sets);
          calc_face_sets(i, faces, new_face_sets);
          if (array_utils::indexed_data_equal<int>(face_sets.span, faces, new_face_sets)) {
            return;
          }

          undo::push_node(depsgraph, object, &nodes[i], undo::Type::FaceSet);
          scatter_data_mesh(new_face_sets.as_span(), faces, face_sets.span);
          node_changed[i] = true;
        },
        exec_mode::grain_size(1));
    face_sets.finish();
  }
  else if (pbvh.type() == bke::pbvh::Type::Multires) {
    const Multiresolution &multires = *mesh.multires;
    const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(mesh);
    const MutableSpan<int> face_sets = ensure_face_sets_multires(mesh);
    const OffsetIndices<int> base_faces = base_topo.faces;
    const MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();

    node_mask.foreach_index(
        [&](const int i) {
          TLS &tls = all_tls.local();
          const Span<int> base_node_faces = nodes[i].faces();
          const int subfaces_count = nodes[i].corners_num * multires.faces_per_corner();

          tls.face_indices.resize(subfaces_count);
          gather_subfaces(
              multires.faces_per_corner(), base_faces, base_node_faces, tls.face_indices);
          const Span<int> subfaces = tls.face_indices;

          tls.new_face_sets.resize(subfaces_count);
          const MutableSpan<int> new_face_sets = tls.new_face_sets;
          gather_data_mesh(face_sets.as_span(), subfaces, new_face_sets);

          calc_face_sets(i, subfaces, new_face_sets);

          if (array_utils::indexed_data_equal<int>(face_sets, subfaces, new_face_sets)) {
            return;
          }

          undo::push_node(depsgraph, object, &nodes[i], undo::Type::FaceSet);
          scatter_data_mesh(new_face_sets.as_span(), subfaces, face_sets);
          node_changed[i] = true;
        },
        exec_mode::grain_size(1));
  }
  else if (pbvh.type() == bke::pbvh::Type::Grids) {
    bke::SpanAttributeWriter<int> face_sets = ensure_face_sets_mesh(mesh);
    MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
    node_mask.foreach_index(
        [&](const int i) {
          TLS &tls = all_tls.local();
          const Span<int> faces = bke::pbvh::node_face_indices_calc_grids(
              *ss.subdiv_ccg, nodes[i], tls.face_indices);

          tls.new_face_sets.resize(faces.size());
          MutableSpan<int> new_face_sets = tls.new_face_sets;
          gather_data_mesh(face_sets.span.as_span(), faces, new_face_sets);
          calc_face_sets(i, faces, new_face_sets);
          if (array_utils::indexed_data_equal<int>(face_sets.span, faces, new_face_sets)) {
            return;
          }

          undo::push_node(depsgraph, object, &nodes[i], undo::Type::FaceSet);
          scatter_data_mesh(new_face_sets.as_span(), faces, face_sets.span);
          node_changed[i] = true;
        },
        exec_mode::grain_size(1));
    face_sets.finish();
  }

  IndexMaskMemory memory;
  pbvh.tag_face_sets_changed(IndexMask::from_bools(node_changed, memory));
}

enum class CreateMode {
  Masked = 0,
  Visible = 1,
  All = 2,
  Selection = 3,
};

static void clear_face_sets(const Depsgraph &depsgraph, Object &object, const IndexMask &node_mask)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);

  SculptSession &ss = *object.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);

  Array<bool> node_changed(pbvh.nodes_num(), false);
  const int default_face_set = mesh.face_sets_color_default;

  if (pbvh.type() == bke::pbvh::Type::Mesh) {
    bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
    if (!attributes.contains(".sculpt_face_set")) {
      return;
    }
    const VArraySpan face_sets = *attributes.lookup<int>(".sculpt_face_set",
                                                         bke::AttrDomain::Face);
    MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
    node_mask.foreach_index(
        [&](const int i) {
          const Span<int> faces = nodes[i].faces();
          if (std::any_of(faces.begin(),
                          faces.end(),
                          [&](const int face) { return face_sets[face] != default_face_set; }))
          {
            undo::push_node(depsgraph, object, &nodes[i], undo::Type::FaceSet);
            node_changed[i] = true;
          }
        },
        exec_mode::grain_size(1));
    attributes.remove(".sculpt_face_set");
  }
  else if (pbvh.type() == bke::pbvh::Type::Multires) {
    Multiresolution &multires = *mesh.multires;
    const Span<int> face_sets = multires.face_sets();
    if (face_sets.is_empty()) {
      return;
    }

    const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(mesh);
    const OffsetIndices<int> base_faces = base_topo.faces;
    const int faces_per_corner = multires.faces_per_corner();
    MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
    threading::EnumerableThreadSpecific<Vector<int>> all_subfaces;
    node_mask.foreach_index(
        [&](const int i) {
          Vector<int> &local_subfaces = all_subfaces.local();
          const Span<int> base_node_faces = nodes[i].faces();
          const int subfaces_count = nodes[i].corners_num * faces_per_corner;
          local_subfaces.resize(subfaces_count);
          gather_subfaces(
              faces_per_corner, base_faces, base_node_faces, local_subfaces.as_mutable_span());
          if (std::any_of(
                  local_subfaces.begin(),
                  local_subfaces.end(),
                  [&](const int subface) { return face_sets[subface] != default_face_set; }))
          {
            undo::push_node(depsgraph, object, &nodes[i], undo::Type::FaceSet);
            node_changed[i] = true;
          }
        },
        exec_mode::grain_size(1));

    multires.face_sets_free();
  }
  else if (pbvh.type() == bke::pbvh::Type::Grids) {
    bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
    if (!attributes.contains(".sculpt_face_set")) {
      return;
    }
    const VArraySpan face_sets = *attributes.lookup<int>(".sculpt_face_set",
                                                         bke::AttrDomain::Face);

    threading::EnumerableThreadSpecific<Vector<int>> all_face_indices;
    MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
    node_mask.foreach_index(
        [&](const int i) {
          Vector<int> &face_indices = all_face_indices.local();
          const Span<int> faces = bke::pbvh::node_face_indices_calc_grids(
              *ss.subdiv_ccg, nodes[i], face_indices);
          if (std::any_of(faces.begin(),
                          faces.end(),
                          [&](const int face) { return face_sets[face] != default_face_set; }))
          {
            undo::push_node(depsgraph, object, &nodes[i], undo::Type::FaceSet);
            node_changed[i] = true;
          }
        },
        exec_mode::grain_size(1));
    attributes.remove(".sculpt_face_set");
  }
  IndexMaskMemory memory;
  pbvh.tag_face_sets_changed(IndexMask::from_bools(node_changed, memory));
}

static wmOperatorStatus create_op_exec(bContext *C, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  Object &object = *CTX_data_active_object(C);
  Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);

  const CreateMode mode = CreateMode(RNA_enum_get(op->ptr, "mode"));

  const View3D *v3d = CTX_wm_view3d(C);
  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  ed::sculpt_paint::face_set_overlay_check(*C, *op);

  const bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  if (pbvh.type() == bke::pbvh::Type::BMesh) {
    /* Dyntopo not supported. */
    return OPERATOR_CANCELLED;
  }

  Mesh &mesh = *id_cast<Mesh *>(object.data);

  BKE_sculptsession_update_for_edit(&depsgraph, &object, false);

  undo::push_begin(scene, object, op);

  const int next_face_set = find_next_available_id(object);

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);
  switch (mode) {
    case CreateMode::Masked: {
      if (pbvh.type() == bke::pbvh::Type::Mesh) {
        const OffsetIndices faces = mesh.faces();
        const Span<int> corner_verts = mesh.corner_verts();
        const bke::AttributeAccessor attributes = mesh.attributes();
        const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly",
                                                                    bke::AttrDomain::Face);
        const VArraySpan<float> mask = *attributes.lookup<float>(".sculpt_mask",
                                                                 bke::AttrDomain::Point);
        if (!mask.is_empty()) {
          face_sets_update(
              depsgraph,
              object,
              node_mask,
              [&](const int /*node_idx*/, const Span<int> indices, MutableSpan<int> face_sets) {
                for (const int i : indices.index_range()) {
                  if (!hide_poly.is_empty() && hide_poly[indices[i]]) {
                    continue;
                  }
                  const Span<int> face_verts = corner_verts.slice(faces[indices[i]]);
                  if (!std::any_of(face_verts.begin(),
                                   face_verts.end(),
                                   [&](const int vert) { return mask[vert] > 0.5f; }))
                  {
                    continue;
                  }
                  face_sets[i] = next_face_set;
                }
              });
        }
      }
      else if (pbvh.type() == bke::pbvh::Type::Multires) {
        const Multiresolution &multires = *mesh.multires;
        const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(mesh);
        const Span<int> corner_verts = multires.corner_verts();
        const Span<bool> hidden_faces = multires.hidden_faces();
        const Span<float> masks = multires.masks();

        if (!masks.is_empty()) {
          face_sets_update(depsgraph,
                           object,
                           node_mask,
                           [&](const int /*node_idx*/,
                               const Span<int> subface_indices,
                               MutableSpan<int> face_sets) {
                             for (const int i : subface_indices.index_range()) {
                               const int subface_idx = subface_indices[i];
                               if (!hidden_faces.is_empty() && hidden_faces[subface_idx]) {
                                 continue;
                               }
                               const int v0 = corner_verts[subface_idx * 4 + 0];
                               const int v1 = corner_verts[subface_idx * 4 + 1];
                               const int v2 = corner_verts[subface_idx * 4 + 2];
                               const int v3 = corner_verts[subface_idx * 4 + 3];

                               if (masks[v0] > 0.5f || masks[v1] > 0.5f || masks[v2] > 0.5f ||
                                   masks[v3] > 0.5f) {
                                 face_sets[i] = next_face_set;
                               }
                             }
                           });
        }
      }
      else if (pbvh.type() == bke::pbvh::Type::Grids) {
        const bke::AttributeAccessor attributes = mesh.attributes();
        const OffsetIndices<int> faces = mesh.faces();
        const SculptSession &ss = *object.runtime->sculpt_session;
        const SubdivCCG &subdiv_ccg = *ss.subdiv_ccg;
        const int grid_area = subdiv_ccg.grid_area;
        const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly",
                                                                    bke::AttrDomain::Face);
        const Span<float> masks = subdiv_ccg.masks;
        if (!masks.is_empty()) {
          face_sets_update(
              depsgraph,
              object,
              node_mask,
              [&](const int /*node_idx*/, const Span<int> indices, MutableSpan<int> face_sets) {
                for (const int i : indices.index_range()) {
                  if (!hide_poly.is_empty() && hide_poly[indices[i]]) {
                    continue;
                  }

                  const Span<float> face_masks = masks.slice(
                      bke::ccg::face_range(faces, grid_area, indices[i]));
                  if (!std::any_of(face_masks.begin(),
                                   face_masks.end(),
                                   [&](const float mask) { return mask > 0.5f; }))
                  {
                    continue;
                  }
                  face_sets[i] = next_face_set;
                }
              });
        }
      }
      break;
    }
    case CreateMode::Visible: {
      VArray<bool> hide_poly;

      if (pbvh.type() != bke::pbvh::Type::Multires) {
        const bke::AttributeAccessor attributes = mesh.attributes();
        hide_poly = *attributes.lookup_or_default<bool>(
            ".hide_poly", bke::AttrDomain::Face, false);
      }
      else {
        const bke::AttributeAccessor attributes = mesh.multires->attributes();
        hide_poly = *attributes.lookup_or_default<bool>(
            ".hide_poly", bke::AttrDomain::Face, false);
      }

      switch (array_utils::booleans_mix_calc(hide_poly)) {
        case array_utils::BooleanMix::None:
        case array_utils::BooleanMix::AllTrue:
        case array_utils::BooleanMix::AllFalse:
          /* If all vertices in the sculpt are visible, remove face sets and update the default
           * color. This way the new face set will be white, and it is a quick way of disabling all
           * face sets and the performance hit of rendering the overlay. */
          clear_face_sets(depsgraph, object, node_mask);
          break;
        case array_utils::BooleanMix::Mixed: {
          const VArraySpan<bool> hide_poly_span(hide_poly);
          face_sets_update(
              depsgraph,
              object,
              node_mask,
              [&](const int /*node_idx*/, const Span<int> indices, MutableSpan<int> face_sets) {
                for (const int i : indices.index_range()) {
                  if (!hide_poly_span[indices[i]]) {
                    face_sets[i] = next_face_set;
                  }
                }
              });
          break;
        }
      }
      break;
    }
    case CreateMode::All: {
      face_sets_update(depsgraph,
                       object,
                       node_mask,
                       [&](const int /*node_idx*/,
                           const Span<int> /*indices*/,
                           MutableSpan<int> face_sets) { face_sets.fill(next_face_set); });
      break;
    }
    case CreateMode::Selection: {
      const bke::AttributeAccessor attributes = mesh.attributes();
      const VArray<bool> select_poly = *attributes.lookup_or_default<bool>(
          ".select_poly", bke::AttrDomain::Face, false);

      if (ELEM(array_utils::booleans_mix_calc(select_poly),
               array_utils::BooleanMix::None,
               array_utils::BooleanMix::AllFalse))
      {
        break;
      }

      const VArraySpan<bool> select_poly_span(select_poly);

      if (pbvh.type() != bke::pbvh::Type::Multires) {
        const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly",
                                                                    bke::AttrDomain::Face);

        face_sets_update(
            depsgraph,
            object,
            node_mask,
            [&](const int /*node_idx*/, const Span<int> indices, MutableSpan<int> face_sets) {
              for (const int i : indices.index_range()) {
                if (select_poly_span[indices[i]]) {
                  if (!hide_poly.is_empty() && hide_poly[indices[i]]) {
                    continue;
                  }
                  face_sets[i] = next_face_set;
                }
              }
            });
      }
      else {
        const Multiresolution &multires = *mesh.multires;
        const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(mesh);
        const OffsetIndices<int> base_faces = base_topo.faces;
        const int faces_per_corner = multires.faces_per_corner();
        const Span<bool> hidden_faces = multires.hidden_faces();
        const Span<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();

        const int base_level = multires.base_level(multires.curr_level);
        Array<bool> base_select_storage;
        Span<bool> base_select_poly;
        if (base_level == 0) {
          base_select_poly = select_poly_span;
        }
        else {
          base_select_storage.reinitialize(base_topo.faces_num);
          multires.subdivide_hidden_faces(
              mesh, 0, select_poly_span, base_level, base_select_storage.as_mutable_span());
          base_select_poly = base_select_storage.as_span();
        }

        face_sets_update(
            depsgraph,
            object,
            node_mask,
            [&](const int node_idx, const Span<int> subfaces, MutableSpan<int> face_sets) {
              int offset = 0;
              for (const int face_idx : nodes[node_idx].faces()) {
                const int count = base_faces[face_idx].size() * faces_per_corner;

                if (base_select_poly[face_idx]) {
                  MutableSpan<int> face_slice = face_sets.slice(offset, count);
                  Span<int> subface_slice = subfaces.slice(offset, count);

                  for (const int j : face_slice.index_range()) {
                    const int sf = subface_slice[j];
                    if (!hidden_faces.is_empty() && hidden_faces[sf]) {
                      continue;
                    }
                    face_slice[j] = next_face_set;
                  }
                }
                offset += count;
              }
            });
      }
      break;
    }
  }

  undo::push_end(object);

  tag_update_overlays(C);

  return OPERATOR_FINISHED;
}

void SCULPT_OT_face_sets_create(wmOperatorType *ot)
{
  ot->name = "Create Face Set";
  ot->idname = "SCULPT_OT_face_sets_create";
  ot->description = "Create a new face set";

  ot->exec = create_op_exec;
  ot->poll = sculpt_mode_poll;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  static EnumPropertyItem modes[] = {
      {int(CreateMode::Masked),
       "MASKED",
       0,
       "Face Set from Masked",
       "Create a new face set from the masked faces"},
      {int(CreateMode::Visible),
       "VISIBLE",
       0,
       "Face Set from Visible",
       "Create a new face set from the visible vertices"},
      {int(CreateMode::All),
       "ALL",
       0,
       "Face Set Full Mesh",
       "Create a unique face set with all faces in the sculpt"},
      {int(CreateMode::Selection),
       "SELECTION",
       0,
       "Face Set from Edit Mode Selection",
       "Create a face set corresponding to the Edit Mode face selection"},
      {0, nullptr, 0, nullptr, nullptr},
  };
  RNA_def_enum(ot->srna, "mode", modes, int(CreateMode::Masked), "Mode", "");
}

enum class InitMode {
  LooseParts = 0,
  Materials = 1,
  Normals = 2,
  UVSeams = 3,
  Creases = 4,
  SharpEdges = 5,
  BevelWeight = 6,
  FaceSetBoundaries = 8,
};

using FaceSetsFloodFillFn = FunctionRef<bool(int from_face, int edge, int to_face)>;
using FaceSetsFloodFillFnMultires = FunctionRef<bool(int from_subface, int to_subface)>;

static Set<int> gather_hidden_face_sets(const Span<bool> hide_poly, const Span<int> face_sets)
{
  if (hide_poly.is_empty()) {
    return {};
  }

  Set<int> hidden_face_sets;
  for (const int i : hide_poly.index_range()) {
    if (hide_poly[i]) {
      hidden_face_sets.add(face_sets[i]);
    }
  }

  return hidden_face_sets;
}

static void init_flood_fill_impl(const OffsetIndices<int> faces,
                                 const Span<int> corner_edges,
                                 const GroupedSpan<int> edge_to_face_map,
                                 const VArraySpan<bool> &hide_poly,
                                 const MutableSpan<int> face_sets,
                                 const FaceSetsFloodFillFn &test_fn)
{
  BitVector<> visited_faces(faces.size(), false);
  const Set<int> hidden_face_sets = gather_hidden_face_sets(hide_poly, face_sets);

  int next_face_set = 1;

  for (const int i : faces.index_range()) {
    if (!hide_poly.is_empty() && hide_poly[i]) {
      continue;
    }
    if (visited_faces[i]) {
      continue;
    }
    std::queue<int> queue;

    while (hidden_face_sets.contains(next_face_set)) {
      next_face_set += 1;
    }
    face_sets[i] = next_face_set;
    visited_faces[i].set(true);
    queue.push(i);

    while (!queue.empty()) {
      const int face_idx = queue.front();
      queue.pop();

      for (const int edge_idx : corner_edges.slice(faces[face_idx])) {
        for (const int neighbor_idx : edge_to_face_map[edge_idx]) {
          if (neighbor_idx == face_idx) {
            continue;
          }
          if (visited_faces[neighbor_idx]) {
            continue;
          }
          if (!hide_poly.is_empty() && hide_poly[neighbor_idx]) {
            continue;
          }
          if (!test_fn(face_idx, edge_idx, neighbor_idx)) {
            continue;
          }

          face_sets[neighbor_idx] = next_face_set;
          visited_faces[neighbor_idx].set(true);
          queue.push(neighbor_idx);
        }
      }
    }

    next_face_set += 1;
  }
}

static void init_flood_fill_mesh(Object &ob, const FaceSetsFloodFillFn &test_fn)
{
  SculptSession &ss = *ob.runtime->sculpt_session;
  Mesh *mesh = id_cast<Mesh *>(ob.data);

  bke::SpanAttributeWriter<int> face_sets = ensure_face_sets_mesh(*mesh);

  const Span<int2> edges = mesh->edges();
  const OffsetIndices faces = mesh->faces();
  const Span<int> corner_edges = mesh->corner_edges();

  if (ss.edge_to_face_map.is_empty()) {
    ss.edge_to_face_map = bke::mesh::build_edge_to_face_map(
        faces, corner_edges, edges.size(), ss.edge_to_face_offsets, ss.edge_to_face_indices);
  }

  const bke::AttributeAccessor attributes = mesh->attributes();
  const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly", bke::AttrDomain::Face);
  init_flood_fill_impl(
      faces, corner_edges, ss.edge_to_face_map, hide_poly, face_sets.span, test_fn);

  face_sets.finish();
}

static void init_flood_fill_multires(Mesh &base_mesh, const FaceSetsFloodFillFnMultires &test_fn)
{
  Multiresolution &multires = *base_mesh.multires;
  const int faces_num = multires.faces_num();
  if (faces_num == 0) {
    return;
  }
  const int verts_num = multires.verts_num();

  MutableSpan<int> face_sets = ensure_face_sets_multires(base_mesh);
  const Span<int> corner_verts = multires.corner_verts();
  const VArraySpan<bool> hidden_faces = *multires.attributes().lookup<bool>(".hide_poly",
                                                                            bke::AttrDomain::Face);

  Array<int> vert_to_corner_offsets;
  Array<int> vert_to_corner_indices;
  const GroupedSpan<int> vert_to_corner = bke::mesh::build_vert_to_corner_map(
      corner_verts, verts_num, vert_to_corner_offsets, vert_to_corner_indices);

  BitVector<> visited_faces(faces_num, false);
  const Set<int> hidden_face_sets = gather_hidden_face_sets(hidden_faces, face_sets);

  int next_face_set = 1;

  auto subface_contains_vert = [&](const int sf, const int v) {
    const Span<int> verts = corner_verts.slice(sf * 4, 4);
    return verts[0] == v || verts[1] == v || verts[2] == v || verts[3] == v;
  };

  std::queue<int> queue;
  for (const int i : IndexRange(faces_num)) {
    if (!hidden_faces.is_empty() && hidden_faces[i]) {
      continue;
    }
    if (visited_faces[i]) {
      continue;
    }

    queue = std::queue<int>();
    while (hidden_face_sets.contains(next_face_set)) {
      next_face_set += 1;
    }
    face_sets[i] = next_face_set;
    visited_faces[i].set(true);
    queue.push(i);

    while (!queue.empty()) {
      const int subface_idx = queue.front();
      queue.pop();

      const int v0 = corner_verts[subface_idx * 4 + 0];
      const int v1 = corner_verts[subface_idx * 4 + 1];
      const int v2 = corner_verts[subface_idx * 4 + 2];
      const int v3 = corner_verts[subface_idx * 4 + 3];
      const int face_verts[4] = {v0, v1, v2, v3};

      for (const int e : IndexRange(4)) {
        const int va = face_verts[e];
        const int vb = face_verts[(e + 1) % 4];
        for (const int corner : vert_to_corner[va]) {
          const int neighbor_idx = corner / 4;
          if (neighbor_idx == subface_idx) {
            continue;
          }
          if (visited_faces[neighbor_idx]) {
            continue;
          }
          if (!hidden_faces.is_empty() && hidden_faces[neighbor_idx]) {
            continue;
          }
          if (!subface_contains_vert(neighbor_idx, vb)) {
            continue;
          }
          if (!test_fn(subface_idx, neighbor_idx)) {
            continue;
          }
          face_sets[neighbor_idx] = next_face_set;
          visited_faces[neighbor_idx].set(true);
          queue.push(neighbor_idx);
        }
      }
    }

    next_face_set += 1;
  }
}

static void init_flood_fill_multires_from_base(Object &ob,
                                               Mesh &base_mesh,
                                               const FaceSetsFloodFillFn &test_fn)
{
  SculptSession &ss = *ob.runtime->sculpt_session;
  Multiresolution &multires = *base_mesh.multires;

  const OffsetIndices faces = base_mesh.faces();
  const Span<int> corner_edges = base_mesh.corner_edges();
  const Span<int2> edges = base_mesh.edges();
  if (ss.edge_to_face_map.is_empty()) {
    ss.edge_to_face_map = bke::mesh::build_edge_to_face_map(
        faces, corner_edges, edges.size(), ss.edge_to_face_offsets, ss.edge_to_face_indices);
  }

  const bke::AttributeAccessor attributes = base_mesh.attributes();
  const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly", bke::AttrDomain::Face);

  Array<int> base_face_sets(base_mesh.faces_num);
  if (const auto base_attr = attributes.lookup<int>(".sculpt_face_set", bke::AttrDomain::Face)) {
    array_utils::copy(*base_attr, base_face_sets.as_mutable_span());
  }
  else {
    base_face_sets.fill(1);
  }

  init_flood_fill_impl(faces,
                       corner_edges,
                       ss.edge_to_face_map,
                       hide_poly,
                       base_face_sets.as_mutable_span(),
                       test_fn);

  multires.subdivide_face_sets(
      base_mesh, 0, base_face_sets.as_span(), multires.curr_level, multires.face_sets_for_write());
}

static wmOperatorStatus init_op_exec(bContext *C, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  Object &ob = *CTX_data_active_object(C);
  Depsgraph *depsgraph = CTX_data_depsgraph_pointer(C);

  const InitMode mode = InitMode(RNA_enum_get(op->ptr, "mode"));

  const View3D *v3d = CTX_wm_view3d(C);
  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  ed::sculpt_paint::face_set_overlay_check(*C, *op);

  BKE_sculptsession_update_for_edit(depsgraph, &ob, false);

  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(ob);
  /* Dyntopo not supported. */
  if (pbvh.type() == bke::pbvh::Type::BMesh) {
    return OPERATOR_CANCELLED;
  }

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);
  if (node_mask.is_empty()) {
    return OPERATOR_CANCELLED;
  }

  undo::push_begin(scene, ob, op);
  undo::push_nodes(*depsgraph, ob, node_mask, undo::Type::FaceSet);

  const float threshold = RNA_float_get(op->ptr, "threshold");

  Mesh *mesh = id_cast<Mesh *>(ob.data);
  const bke::AttributeAccessor attributes = mesh->attributes();

  const bool is_multires = pbvh.type() == bke::pbvh::Type::Multires;

  switch (mode) {
    case InitMode::LooseParts: {
      if (is_multires) {
        init_flood_fill_multires(*mesh,
                                 [&](const int /*from_sf*/, const int /*to_sf*/) { return true; });
      }
      else {
        const VArray<bool> hide_poly = *attributes.lookup<bool>(".hide_poly",
                                                                bke::AttrDomain::Face);
        init_flood_fill_mesh(ob, [&](const int from_face, const int /*edge*/, const int to_face) {
          return hide_poly[from_face] == hide_poly[to_face];
        });
      }
      break;
    }
    case InitMode::Materials: {
      const VArraySpan<int> material_indices = *attributes.lookup_or_default<int>(
          "material_index", bke::AttrDomain::Face, 0);
      const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly",
                                                                  bke::AttrDomain::Face);
      if (is_multires) {
        Multiresolution &multires = *mesh->multires;
        Array<int> base_face_sets(mesh->faces_num);
        if (const auto base_attr = attributes.lookup<int>(".sculpt_face_set",
                                                          bke::AttrDomain::Face))
        {
          array_utils::copy(*base_attr, base_face_sets.as_mutable_span());
        }
        else {
          base_face_sets.fill(1);
        }
        for (const int i : IndexRange(mesh->faces_num)) {
          if (!hide_poly.is_empty() && hide_poly[i]) {
            continue;
          }
          base_face_sets[i] = material_indices[i] + 1;
        }
        multires.subdivide_face_sets(*mesh,
                                     0,
                                     base_face_sets.as_span(),
                                     multires.curr_level,
                                     multires.face_sets_for_write());
      }
      else {
        bke::SpanAttributeWriter<int> face_sets = ensure_face_sets_mesh(*mesh);
        for (const int i : IndexRange(mesh->faces_num)) {
          if (!hide_poly.is_empty() && hide_poly[i]) {
            continue;
          }

          /* In some cases material face set index could be same as hidden face set index
           * A more robust implementation is needed to avoid this */
          face_sets.span[i] = material_indices[i] + 1;
        }

        face_sets.finish();
      }
      break;
    }
    case InitMode::Normals: {
      if (is_multires) {
        Multiresolution &multires = *mesh->multires;
        const Span<float3> positions = multires.positions();
        const Span<int> corner_verts = multires.corner_verts();

        auto subface_normal = [&](const int subface) -> float3 {
          const float3 &p0 = positions[corner_verts[subface * 4 + 0]];
          const float3 &p1 = positions[corner_verts[subface * 4 + 1]];
          const float3 &p2 = positions[corner_verts[subface * 4 + 2]];
          const float3 &p3 = positions[corner_verts[subface * 4 + 3]];
          return math::normalize(math::cross(p2 - p0, p3 - p1));
        };

        init_flood_fill_multires(*mesh, [&](const int from_sf, const int to_sf) -> bool {
          const float3 normal_from = subface_normal(from_sf);
          const float3 normal_to = subface_normal(to_sf);
          return std::abs(math::dot(normal_from, normal_to)) > threshold;
        });
      }
      else {
        const Span<float3> face_normals = mesh->face_normals();
        init_flood_fill_mesh(
            ob, [&](const int from_face, const int /*edge*/, const int to_face) -> bool {
              return std::abs(math::dot(face_normals[from_face], face_normals[to_face])) >
                     threshold;
            });
      }
      break;
    }
    case InitMode::UVSeams: {
      const VArraySpan<bool> uv_seams = *mesh->attributes().lookup_or_default<bool>(
          "uv_seam", bke::AttrDomain::Edge, false);
      FaceSetsFloodFillFn test_fn = [&](const int /*from_face*/,
                                        const int edge,
                                        const int /*to_face*/) -> bool { return !uv_seams[edge]; };
      if (is_multires) {
        init_flood_fill_multires_from_base(ob, *mesh, test_fn);
      }
      else {
        init_flood_fill_mesh(ob, test_fn);
      }
      break;
    }
    case InitMode::Creases: {
      const VArraySpan<float> creases = *attributes.lookup_or_default<float>(
          "crease_edge", bke::AttrDomain::Edge, 0.0f);
      FaceSetsFloodFillFn test_fn =
          [&](const int /*from_face*/, const int edge, const int /*to_face*/) -> bool {
        return creases[edge] < threshold;
      };
      if (is_multires) {
        init_flood_fill_multires_from_base(ob, *mesh, test_fn);
      }
      else {
        init_flood_fill_mesh(ob, test_fn);
      }
      break;
    }
    case InitMode::SharpEdges: {
      const VArraySpan<bool> sharp_edges = *mesh->attributes().lookup_or_default<bool>(
          "sharp_edge", bke::AttrDomain::Edge, false);
      FaceSetsFloodFillFn test_fn =
          [&](const int /*from_face*/, const int edge, const int /*to_face*/) -> bool {
        return !sharp_edges[edge];
      };
      if (is_multires) {
        init_flood_fill_multires_from_base(ob, *mesh, test_fn);
      }
      else {
        init_flood_fill_mesh(ob, test_fn);
      }
      break;
    }
    case InitMode::BevelWeight: {
      const VArraySpan<float> bevel_weights = *attributes.lookup_or_default<float>(
          "bevel_weight_edge", bke::AttrDomain::Edge, 0.0f);
      FaceSetsFloodFillFn test_fn =
          [&](const int /*from_face*/, const int edge, const int /*to_face*/) -> bool {
        return bevel_weights[edge] < threshold;
      };
      if (is_multires) {
        init_flood_fill_multires_from_base(ob, *mesh, test_fn);
      }
      else {
        init_flood_fill_mesh(ob, test_fn);
      }
      break;
    }
    case InitMode::FaceSetBoundaries: {
      Array<int> face_sets_copy = duplicate_face_sets(ob);

      if (is_multires) {
        init_flood_fill_multires(*mesh, [&](const int from_sf, const int to_sf) -> bool {
          return face_sets_copy[from_sf] == face_sets_copy[to_sf];
        });
      }
      else {
        init_flood_fill_mesh(
            ob, [&](const int from_face, const int /*edge*/, const int to_face) -> bool {
              return face_sets_copy[from_face] == face_sets_copy[to_face];
            });
      }
      break;
    }
  }

  undo::push_end(ob);

  pbvh.tag_face_sets_changed(node_mask);

  tag_update_overlays(C);

  return OPERATOR_FINISHED;
}

void SCULPT_OT_face_sets_init(wmOperatorType *ot)
{
  ot->name = "Init Face Sets";
  ot->idname = "SCULPT_OT_face_sets_init";
  ot->description = "Initializes all face sets in the mesh";

  ot->exec = init_op_exec;
  ot->poll = sculpt_mode_poll;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  static EnumPropertyItem modes[] = {
      {int(InitMode::LooseParts),
       "LOOSE_PARTS",
       0,
       "Face Sets from Loose Parts",
       "Create a face set per loose part in the mesh"},
      {int(InitMode::Materials),
       "MATERIALS",
       0,
       "Face Sets from Material Slots",
       "Create a face set per material slot"},
      {int(InitMode::Normals),
       "NORMALS",
       0,
       "Face Sets from Mesh Normals",
       "Create face sets for faces that have similar normal"},
      {int(InitMode::UVSeams),
       "UV_SEAMS",
       0,
       "Face Sets from UV Seams",
       "Create face sets using UV seams as boundaries"},
      {int(InitMode::Creases),
       "CREASES",
       0,
       "Face Sets from Edge Creases",
       "Create face sets using edge creases as boundaries"},
      {int(InitMode::BevelWeight),
       "BEVEL_WEIGHT",
       0,
       "Face Sets from Bevel Weight",
       "Create face sets using bevel weights as boundaries"},
      {int(InitMode::SharpEdges),
       "SHARP_EDGES",
       0,
       "Face Sets from Sharp Edges",
       "Create face sets using sharp edges as boundaries"},
      {int(InitMode::FaceSetBoundaries),
       "FACE_SET_BOUNDARIES",
       0,
       "Face Sets from Face Set Boundaries",
       "Create a face set per isolated face set"},
      {0, nullptr, 0, nullptr, nullptr},
  };
  RNA_def_enum(ot->srna, "mode", modes, int(InitMode::LooseParts), "Mode", "");
  RNA_def_float(
      ot->srna,
      "threshold",
      0.5f,
      0.0f,
      1.0f,
      "Threshold",
      "Minimum value to consider a certain attribute a boundary when creating the face sets",
      0.0f,
      1.0f);
}

enum class VisibilityMode {
  Toggle = 0,
  ShowActive = 1,
  HideActive = 2,
};

static void face_hide_update(const Depsgraph &depsgraph,
                             Object &object,
                             const IndexMask &node_mask,
                             const FunctionRef<void(Span<int>, MutableSpan<bool>)> calc_hide)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  Mesh &mesh = *id_cast<Mesh *>(object.data);

  struct TLS {
    Vector<int> face_indices;
    Vector<bool> new_hide;
  };

  Array<bool> node_changed(node_mask.min_array_size(), false);

  threading::EnumerableThreadSpecific<TLS> all_tls;
  if (pbvh.type() == bke::pbvh::Type::Mesh) {
    bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
    bke::SpanAttributeWriter<bool> hide_poly = attributes.ensure_for_write_span<bool>(
        ".hide_poly", bke::AttrDomain::Face);
    MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
    node_mask.foreach_index(
        [&](const int i) {
          TLS &tls = all_tls.local();
          const Span<int> faces = nodes[i].faces();

          tls.new_hide.resize(faces.size());
          MutableSpan<bool> new_hide = tls.new_hide;
          gather_data_mesh(hide_poly.span.as_span(), faces, new_hide);
          calc_hide(faces, new_hide);
          if (array_utils::indexed_data_equal<bool>(hide_poly.span, faces, new_hide)) {
            return;
          }

          undo::push_node(depsgraph, object, &nodes[i], undo::Type::HideFace);
          scatter_data_mesh(new_hide.as_span(), faces, hide_poly.span);
          node_changed[i] = true;
        },
        exec_mode::grain_size(1));
    hide_poly.finish();
  }
  else if (pbvh.type() == bke::pbvh::Type::Multires) {
    Multiresolution &multires = *mesh.multires;
    const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(mesh);
    const int faces_per_corner = multires.faces_per_corner();
    MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
    MutableSpan<bool> hidden_faces = multires.hidden_faces_for_write();
    node_mask.foreach_index(
        [&](const int i) {
          TLS &tls = all_tls.local();

          const Span<int> node_faces = nodes[i].faces();
          tls.face_indices.resize(faces_per_corner * nodes[i].corners_num);
          const MutableSpan<int> node_subfaces = tls.face_indices;
          gather_subfaces(faces_per_corner, base_topo.faces, node_faces, node_subfaces);

          tls.new_hide.resize(node_subfaces.size());
          MutableSpan<bool> new_hide = tls.new_hide;
          gather_data_mesh(hidden_faces.as_span(), node_subfaces, new_hide);
          calc_hide(node_subfaces, new_hide);
          if (array_utils::indexed_data_equal<bool>(hidden_faces, node_subfaces, new_hide)) {
            return;
          }

          undo::push_node(depsgraph, object, &nodes[i], undo::Type::HideFace);
          scatter_data_mesh(new_hide.as_span(), node_subfaces, hidden_faces);
          node_changed[i] = true;
        },
        exec_mode::grain_size(1));
  }
  else if (pbvh.type() == bke::pbvh::Type::Grids) {
    bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
    bke::SpanAttributeWriter<bool> hide_poly = attributes.ensure_for_write_span<bool>(
        ".hide_poly", bke::AttrDomain::Face);
    MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
    node_mask.foreach_index(
        [&](const int i) {
          TLS &tls = all_tls.local();
          const Span<int> faces = bke::pbvh::node_face_indices_calc_grids(
              *ss.subdiv_ccg, nodes[i], tls.face_indices);

          tls.new_hide.resize(faces.size());
          MutableSpan<bool> new_hide = tls.new_hide;
          gather_data_mesh(hide_poly.span.as_span(), faces, new_hide);
          calc_hide(faces, new_hide);
          if (array_utils::indexed_data_equal<bool>(hide_poly.span, faces, new_hide)) {
            return;
          }

          undo::push_node(depsgraph, object, &nodes[i], undo::Type::HideFace);
          scatter_data_mesh(new_hide.as_span(), faces, hide_poly.span);
          node_changed[i] = true;
        },
        exec_mode::grain_size(1));
    hide_poly.finish();
  }

  IndexMaskMemory memory;
  const IndexMask changed_nodes = IndexMask::from_bools(node_changed, memory);
  if (changed_nodes.is_empty()) {
    return;
  }
  hide::sync_all_from_faces(object);
  pbvh.tag_visibility_changed(node_mask);
}

static void show_all(Depsgraph &depsgraph, Object &object, const IndexMask &node_mask)
{
  switch (bke::object::pbvh_get(object)->type()) {
    case bke::pbvh::Type::Mesh:
      hide::mesh_show_all(depsgraph, object, node_mask);
      break;
    case bke::pbvh::Type::Multires:
      hide::multires_show_all(depsgraph, object, node_mask);
      break;
    case bke::pbvh::Type::Grids:
      hide::grids_show_all(depsgraph, object, node_mask);
      break;
    case bke::pbvh::Type::BMesh:
      BLI_assert_unreachable();
      break;
  }
}

static wmOperatorStatus change_visibility_exec(bContext *C, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  Object &object = *CTX_data_active_object(C);
  SculptSession &ss = *object.runtime->sculpt_session;
  Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);

  Mesh *mesh = BKE_object_get_original_mesh(&object);
  BKE_sculptsession_update_for_edit(&depsgraph, &object, false);

  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);

  if (pbvh.type() == bke::pbvh::Type::BMesh) {
    /* Not supported for dyntopo. There is no active face. */
    return OPERATOR_CANCELLED;
  }

  const VisibilityMode mode = VisibilityMode(RNA_enum_get(op->ptr, "mode"));
  const int active_face_set = RNA_int_get(op->ptr, "active_face_set");

  undo::push_begin(scene, object, op);

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);

  VArraySpan<bool> hide_poly;
  VArraySpan<int> face_sets;

  if (pbvh.type() == bke::pbvh::Type::Multires) {
    const bke::AttributeAccessor attributes = mesh->multires->attributes();
    hide_poly = *attributes.lookup<bool>(".hide_poly", bke::AttrDomain::Face);
    face_sets = *attributes.lookup<int>(".sculpt_face_set", bke::AttrDomain::Face);
  }
  else {
    const bke::AttributeAccessor attributes = mesh->attributes();
    hide_poly = *attributes.lookup<bool>(".hide_poly", bke::AttrDomain::Face);
    face_sets = *attributes.lookup<int>(".sculpt_face_set", bke::AttrDomain::Face);
  }

  switch (mode) {
    case VisibilityMode::Toggle: {
      if (hide_poly.contains(true) || face_sets.is_empty()) {
        show_all(depsgraph, object, node_mask);
      }
      else {
        face_hide_update(
            depsgraph, object, node_mask, [&](const Span<int> faces, MutableSpan<bool> hide) {
              for (const int i : hide.index_range()) {
                hide[i] = face_sets[faces[i]] != active_face_set;
              }
            });
      }
      break;
    }
    case VisibilityMode::ShowActive:
      if (face_sets.is_empty()) {
        show_all(depsgraph, object, node_mask);
      }
      else {
        face_hide_update(
            depsgraph, object, node_mask, [&](const Span<int> faces, MutableSpan<bool> hide) {
              for (const int i : hide.index_range()) {
                if (face_sets[faces[i]] == active_face_set) {
                  hide[i] = false;
                }
              }
            });
      }
      break;
    case VisibilityMode::HideActive:
      if (face_sets.is_empty()) {
        face_hide_update(
            depsgraph, object, node_mask, [&](const Span<int> /*faces*/, MutableSpan<bool> hide) {
              hide.fill(true);
            });
      }
      else {
        face_hide_update(
            depsgraph, object, node_mask, [&](const Span<int> faces, MutableSpan<bool> hide) {
              for (const int i : hide.index_range()) {
                if (face_sets[faces[i]] == active_face_set) {
                  hide[i] = true;
                }
              }
            });
      }
      break;
  }

  /* For modes that use the cursor active vertex, update the rotation origin for viewport
   * navigation. */
  if (ELEM(mode, VisibilityMode::Toggle, VisibilityMode::ShowActive)) {
    Paint *paint = BKE_paint_get_active_from_context(C);
    bke::PaintRuntime *paint_runtime = paint->runtime;
    if (std::holds_alternative<std::monostate>(ss.active_vert())) {
      paint_runtime->last_stroke_valid = false;
    }
    else {
      float location[3];
      copy_v3_v3(location, ss.active_vert_position(depsgraph, object));
      mul_m4_v3(object.object_to_world().ptr(), location);
      bke::paint::stroke_set_location(*paint, location);
    }
  }

  undo::push_end(object);

  pbvh.update_visibility(object);

  islands::invalidate(*object.runtime->sculpt_session);
  hide::tag_update_visibility(*C);

  return OPERATOR_FINISHED;
}

static wmOperatorStatus change_visibility_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  Object &ob = *CTX_data_active_object(C);

  const View3D *v3d = CTX_wm_view3d(C);
  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  /* Update the active vertex and face set using the cursor position to avoid relying on the paint
   * cursor updates. */
  const float mval_fl[2] = {float(event->mval[0]), float(event->mval[1])};
  vert_random_access_ensure(ob);
  cursor_geometry_info_update(C, mval_fl, false);

  const int active_face_set = active_face_set_get(ob);
  RNA_int_set(op->ptr, "active_face_set", active_face_set);

  return change_visibility_exec(C, op);
}

void SCULPT_OT_face_set_change_visibility(wmOperatorType *ot)
{
  ot->name = "Face Sets Visibility";
  ot->idname = "SCULPT_OT_face_set_change_visibility";
  ot->description = "Change the visibility of the face sets of the sculpt";

  ot->exec = change_visibility_exec;
  ot->invoke = change_visibility_invoke;
  ot->poll = sculpt_mode_poll;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO | OPTYPE_DEPENDS_ON_CURSOR;

  static EnumPropertyItem modes[] = {
      {int(VisibilityMode::Toggle),
       "TOGGLE",
       0,
       "Toggle Visibility",
       "Hide all face sets except for the active one"},
      {int(VisibilityMode::ShowActive),
       "SHOW_ACTIVE",
       0,
       "Show Active Face Set",
       "Show the active face set"},
      {int(VisibilityMode::HideActive),
       "HIDE_ACTIVE",
       0,
       "Hide Active Face Set",
       "Hide the active face set"},
      {0, nullptr, 0, nullptr, nullptr},
  };
  RNA_def_enum(ot->srna, "mode", modes, int(VisibilityMode::Toggle), "Mode", "");

  PropertyRNA *prop = RNA_def_int(
      ot->srna, "active_face_set", 0, 0, INT_MAX, "Active Face Set", "", 0, INT_MAX);
  RNA_def_property_flag(prop, PROP_SKIP_SAVE);
}

static wmOperatorStatus randomize_colors_exec(bContext *C, wmOperator * /*op*/)
{
  Object &ob = *CTX_data_active_object(C);

  const View3D *v3d = CTX_wm_view3d(C);
  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(ob);

  /* Dyntopo not supported. */
  if (pbvh.type() == bke::pbvh::Type::BMesh) {
    return OPERATOR_CANCELLED;
  }

  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  auto get_random_index = [&](const int faces_num) -> int {
    return math::clamp(int(mesh.faces_num * BLI_hash_int_01(mesh.face_sets_color_seed)),
                       0,
                       math::max(0, faces_num - 1));
  };

  if (pbvh.type() == bke::pbvh::Type::Multires) {
    const Multiresolution &multires = *mesh.multires;
    const Span<int> face_sets = multires.face_sets();
    if (face_sets.is_empty()) {
      return OPERATOR_CANCELLED;
    }
    const int random_index = get_random_index(multires.faces_num());
    mesh.face_sets_color_default = face_sets[random_index];
  }
  else {
    const bke::AttributeAccessor attributes = mesh.attributes();
    if (!attributes.contains(".sculpt_face_set")) {
      return OPERATOR_CANCELLED;
    }
    const VArray<int> face_sets = *attributes.lookup<int>(".sculpt_face_set",
                                                          bke::AttrDomain::Face);
    const int random_index = get_random_index(mesh.faces_num);
    mesh.face_sets_color_default = face_sets[random_index];
  }

  mesh.face_sets_color_seed += 1;

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);
  pbvh.tag_face_sets_changed(node_mask);

  tag_update_overlays(C);

  return OPERATOR_FINISHED;
}

void SCULPT_OT_face_sets_randomize_colors(wmOperatorType *ot)
{
  ot->name = "Randomize Face Sets Colors";
  ot->idname = "SCULPT_OT_face_sets_randomize_colors";
  ot->description = "Generates a new set of random colors to render the face sets in the viewport";

  ot->exec = randomize_colors_exec;
  ot->poll = sculpt_mode_poll;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;
}

enum class EditMode {
  Grow = 0,
  Shrink = 1,
  DeleteGeometry = 2,
  FairPositions = 3,
  FairTangency = 4,
};

static void edit_grow_shrink_mesh(const Depsgraph &depsgraph,
                                  const Scene &scene,
                                  Object &object,
                                  const EditMode mode,
                                  const int active_face_set_id,
                                  const bool modify_hidden,
                                  wmOperator *op)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  const OffsetIndices faces = mesh.faces();
  const Span<int> corner_verts = mesh.corner_verts();
  const GroupedSpan<int> vert_to_face_map = mesh.vert_to_face_map();
  const bke::AttributeAccessor attributes = mesh.attributes();

  BLI_assert(attributes.contains(".sculpt_face_set"));

  const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly", bke::AttrDomain::Face);
  Array<int> prev_face_sets = duplicate_face_sets(object);

  undo::push_begin(scene, object, op);

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);
  face_sets_update(
      depsgraph,
      object,
      node_mask,
      [&](const int /*node_idx*/, const Span<int> indices, MutableSpan<int> face_sets) {
        for (const int i : indices.index_range()) {
          const int face = indices[i];
          if (!modify_hidden && !hide_poly.is_empty() && hide_poly[face]) {
            continue;
          }
          if (mode == EditMode::Grow) {
            for (const int vert : corner_verts.slice(faces[face])) {
              for (const int neighbor_face_index : vert_to_face_map[vert]) {
                if (neighbor_face_index == face) {
                  continue;
                }
                if (prev_face_sets[neighbor_face_index] == active_face_set_id) {
                  face_sets[i] = active_face_set_id;
                }
              }
            }
          }
          else {
            if (prev_face_sets[face] == active_face_set_id) {
              for (const int vert_i : corner_verts.slice(faces[face])) {
                for (const int neighbor_face_index : vert_to_face_map[vert_i]) {
                  if (neighbor_face_index == face) {
                    continue;
                  }
                  if (prev_face_sets[neighbor_face_index] != active_face_set_id) {
                    face_sets[i] = prev_face_sets[neighbor_face_index];
                  }
                }
              }
            }
          }
        }
      });

  undo::push_end(object);
}

static void edit_grow_shrink_multires(const Depsgraph &depsgraph,
                                      const Scene &scene,
                                      Object &object,
                                      const EditMode mode,
                                      const int active_face_set_id,
                                      const bool modify_hidden,
                                      wmOperator *op)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  Multiresolution &multires = *mesh.multires;

  const Span<int> prev_face_sets = multires.face_sets();
  const Span<int> corner_verts = multires.corner_verts();
  const Span<bool> hidden_faces = multires.hidden_faces();
  const int verts_num = multires.verts_num();

  undo::push_begin(scene, object, op);

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);

  if (mode == EditMode::Grow) {
    Array<bool> vert_is_active(verts_num, false);

    for (const int subface_idx : prev_face_sets.index_range()) {
      if (prev_face_sets[subface_idx] != active_face_set_id) {
        continue;
      }
      if (!modify_hidden && !hidden_faces.is_empty() && hidden_faces[subface_idx]) {
        continue;
      }

      const int *cv = &corner_verts[subface_idx * 4];
      vert_is_active[cv[0]] = true;
      vert_is_active[cv[1]] = true;
      vert_is_active[cv[2]] = true;
      vert_is_active[cv[3]] = true;
    }

    face_sets_update(
        depsgraph,
        object,
        node_mask,
        [&](const int /*node_idx*/, const Span<int> subfaces, MutableSpan<int> node_face_sets) {
          for (const int i : subfaces.index_range()) {
            const int sf = subfaces[i];
            if (node_face_sets[i] == active_face_set_id) {
              continue;
            }
            if (!modify_hidden && !hidden_faces.is_empty() && hidden_faces[sf]) {
              continue;
            }

            const int v0 = corner_verts[sf * 4 + 0];
            const int v1 = corner_verts[sf * 4 + 1];
            const int v2 = corner_verts[sf * 4 + 2];
            const int v3 = corner_verts[sf * 4 + 3];
            if (vert_is_active[v0] || vert_is_active[v1] || vert_is_active[v2] ||
                vert_is_active[v3]) {
              node_face_sets[i] = active_face_set_id;
            }
          }
        });
  }
  else {
    Array<int> vert_face_set(verts_num, face_set_none_id);

    for (const int sf : prev_face_sets.index_range()) {
      const int id = prev_face_sets[sf];
      if (id == active_face_set_id) {
        continue;
      }
      if (!modify_hidden && !hidden_faces.is_empty() && hidden_faces[sf]) {
        continue;
      }

      vert_face_set[corner_verts[sf * 4 + 0]] = id;
      vert_face_set[corner_verts[sf * 4 + 1]] = id;
      vert_face_set[corner_verts[sf * 4 + 2]] = id;
      vert_face_set[corner_verts[sf * 4 + 3]] = id;
    }

    face_sets_update(
        depsgraph,
        object,
        node_mask,
        [&](const int /*node_idx*/, const Span<int> subfaces, MutableSpan<int> node_face_sets) {
          for (const int i : subfaces.index_range()) {
            const int sf = subfaces[i];
            if (node_face_sets[i] != active_face_set_id) {
              continue;
            }
            if (!modify_hidden && !hidden_faces.is_empty() && hidden_faces[sf]) {
              continue;
            }

            const Span<int> corners = corner_verts.slice(sf * 4, 4);
            for (const int corner : corners) {
              const int neighbor_id = vert_face_set[corner];
              if (neighbor_id != face_set_none_id) {
                node_face_sets[i] = neighbor_id;
                break;
              }
            }
          }
        });
  }

  undo::push_end(object);
}

static bool check_single_face_set(const Object &object, const bool check_visible_only)
{
  const Mesh &mesh = *id_cast<const Mesh *>(object.data);
  const bke::AttributeAccessor attributes = mesh.attributes();
  const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly", bke::AttrDomain::Face);
  const VArraySpan<int> face_sets = *attributes.lookup<int>(".sculpt_face_set",
                                                            bke::AttrDomain::Face);

  if (face_sets.is_empty()) {
    return true;
  }
  int first_face_set = face_set_none_id;
  if (check_visible_only) {
    for (const int i : face_sets.index_range()) {
      if (!hide_poly.is_empty() && hide_poly[i]) {
        continue;
      }
      first_face_set = face_sets[i];
      break;
    }
  }
  else {
    first_face_set = face_sets[0];
  }

  if (first_face_set == face_set_none_id) {
    return true;
  }

  for (const int i : face_sets.index_range()) {
    if (check_visible_only && !hide_poly.is_empty() && hide_poly[i]) {
      continue;
    }
    if (face_sets[i] != first_face_set) {
      return false;
    }
  }
  return true;
}

static void delete_geometry(Object &ob, const int active_face_set_id, const bool modify_hidden)
{
  Mesh &mesh = *id_cast<Mesh *>(ob.data);
  const bke::AttributeAccessor attributes = mesh.attributes();
  const VArraySpan<bool> hide_poly = *attributes.lookup<bool>(".hide_poly", bke::AttrDomain::Face);
  const VArraySpan<int> face_sets = *attributes.lookup<int>(".sculpt_face_set",
                                                            bke::AttrDomain::Face);

  const BMAllocTemplate allocsize = BMALLOC_TEMPLATE_FROM_ME(&mesh);
  BMeshCreateParams create_params{};
  create_params.use_toolflags = true;
  BMesh *bm = BM_mesh_create(&allocsize, &create_params);

  BMeshFromMeshParams convert_params{};
  convert_params.calc_vert_normal = true;
  convert_params.calc_face_normal = true;
  BM_mesh_bm_from_me(bm, &mesh, &convert_params);

  BM_mesh_elem_table_init(bm, BM_FACE);
  BM_mesh_elem_table_ensure(bm, BM_FACE);
  BM_mesh_elem_hflag_disable_all(bm, BM_VERT | BM_EDGE | BM_FACE, BM_ELEM_TAG, false);
  BMIter iter;
  BMFace *f;
  BM_ITER_MESH (f, &iter, bm, BM_FACES_OF_MESH) {
    const int face_index = BM_elem_index_get(f);
    if (!modify_hidden && !hide_poly.is_empty() && hide_poly[face_index]) {
      continue;
    }
    BM_elem_flag_set(f, BM_ELEM_TAG, face_sets[face_index] == active_face_set_id);
  }
  BM_mesh_delete_hflag_context(bm, BM_ELEM_TAG, DEL_FACES);
  BM_mesh_elem_hflag_disable_all(bm, BM_VERT | BM_EDGE | BM_FACE, BM_ELEM_TAG, false);

  BMeshToMeshParams bmesh_to_mesh_params{};
  bmesh_to_mesh_params.calc_object_remap = false;
  BM_mesh_bm_to_me(nullptr, bm, &mesh, &bmesh_to_mesh_params);

  BM_mesh_free(bm);
}

static void edit_fairing(const Depsgraph &depsgraph,
                         const Sculpt &sd,
                         Object &ob,
                         const int active_face_set_id,
                         const eMeshFairingDepth fair_order,
                         const float strength)
{
  SculptSession &ss = *ob.runtime->sculpt_session;
  Mesh &mesh = *id_cast<Mesh *>(ob.data);
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(ob);
  boundary::ensure_boundary_info(ob);

  const PositionDeformData position_data(depsgraph, ob);
  const Span<float3> positions = position_data.eval;
  const GroupedSpan<int> vert_to_face_map = mesh.vert_to_face_map();
  const BitSpan boundary_verts = ss.boundary_info_cache->verts;
  const bke::AttributeAccessor attributes = mesh.attributes();
  const VArraySpan hide_poly = *attributes.lookup<bool>(".hide_poly", bke::AttrDomain::Face);
  const VArraySpan face_sets = *attributes.lookup<int>(".sculpt_face_set", bke::AttrDomain::Face);

  Array<bool> fair_verts(positions.size(), false);
  for (const int vert : positions.index_range()) {
    if (boundary::vert_is_boundary(vert_to_face_map, hide_poly, boundary_verts, vert)) {
      continue;
    }
    if (!vert_has_face_set(vert_to_face_map, face_sets, vert, active_face_set_id)) {
      continue;
    }
    if (!vert_has_unique_face_set(vert_to_face_map, face_sets, vert)) {
      continue;
    }
    fair_verts[vert] = true;
  }

  Array<float3> new_positions = positions;
  BKE_mesh_prefair_and_fair_verts(&mesh, new_positions, fair_verts.data(), fair_order);

  struct LocalData {
    Vector<float3> translations;
  };

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);
  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();

  threading::EnumerableThreadSpecific<LocalData> all_tls;
  node_mask.foreach_index(
      [&](const int i) {
        LocalData &tls = all_tls.local();
        const Span<int> verts = nodes[i].verts();
        tls.translations.resize(verts.size());
        const MutableSpan<float3> translations = tls.translations;
        for (const int i : verts.index_range()) {
          translations[i] = new_positions[verts[i]] - positions[verts[i]];
        }
        scale_translations(translations, strength);
        clip_and_lock_translations(sd, ss, positions, verts, translations);
        position_data.deform(translations, verts);
      },
      exec_mode::grain_size(1));
}

static bool edit_is_operation_valid(const Object &object,
                                    const EditMode mode,
                                    const bool modify_hidden)
{
  const bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  if (pbvh.type() == bke::pbvh::Type::BMesh) {
    /* Dyntopo is not supported. */
    return false;
  }

  if (mode == EditMode::DeleteGeometry) {
    if (pbvh.type() == bke::pbvh::Type::Multires) {
      return false;
    }
    if (pbvh.type() == bke::pbvh::Type::Grids) {
      /* Modification of base mesh geometry requires special remapping of multi-resolution
       * displacement, which does not happen here.
       * Disable delete operation. It can be supported in the future by doing similar displacement
       * data remapping as what happens in the mesh edit mode. */
      return false;
    }
    if (check_single_face_set(object, !modify_hidden)) {
      /* Cancel the operator if the mesh only contains one face set to avoid deleting the
       * entire object. */
      return false;
    }
  }

  if (ELEM(mode, EditMode::FairPositions, EditMode::FairTangency)) {
    if (pbvh.type() == bke::pbvh::Type::Multires) {
      // todo?
      return false;
    }
    if (pbvh.type() == bke::pbvh::Type::Grids) {
      /* TODO: Multi-resolution topology representation using grids and duplicates can't be used
       * directly by the fair algorithm. Multi-resolution topology needs to be exposed in a
       * different way or converted to a mesh for this operation. */
      return false;
    }
  }

  if (ELEM(mode, EditMode::Grow, EditMode::Shrink)) {
    if (pbvh.type() == bke::pbvh::Type::Multires) {
      const Mesh &mesh = *id_cast<Mesh *>(object.data);
      if (!mesh.multires || mesh.multires->face_sets().is_empty()) {
        return false;
      }
    }
    else {
      const Mesh &mesh = *id_cast<Mesh *>(object.data);
      const bke::AttributeAccessor attributes = mesh.attributes();
      if (!attributes.contains(".sculpt_face_set")) {
        /* If a mesh does not have the face set attribute, growing or shrinking the face set will
         * have no effect, exit early in this case. */
        return false;
      }
    }
  }

  return true;
}

static void edit_modify_geometry(
    bContext *C, Object &ob, const int active_face_set, const bool modify_hidden, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  Mesh *mesh = id_cast<Mesh *>(ob.data);
  undo::geometry_begin(scene, ob, op);
  delete_geometry(ob, active_face_set, modify_hidden);
  undo::geometry_end(ob);
  BKE_sculptsession_free_pbvh(ob);
  BKE_mesh_batch_cache_dirty_tag(mesh, BKE_MESH_BATCH_DIRTY_ALL);
  DEG_id_tag_update(&ob.id, ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_GEOM | ND_DATA, mesh);
}

static void edit_modify_coordinates(
    bContext *C, Object &ob, const int active_face_set, const EditMode mode, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  const Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  const Sculpt &sd = *CTX_data_tool_settings(C)->sculpt;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(ob);
  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);

  const float strength = RNA_float_get(op->ptr, "strength");

  undo::push_begin(scene, ob, op);
  undo::push_nodes(depsgraph, ob, node_mask, undo::Type::Position);

  switch (mode) {
    case EditMode::FairPositions:
      edit_fairing(depsgraph, sd, ob, active_face_set, MESH_FAIRING_DEPTH_POSITION, strength);
      break;
    case EditMode::FairTangency:
      edit_fairing(depsgraph, sd, ob, active_face_set, MESH_FAIRING_DEPTH_TANGENCY, strength);
      break;
    default:
      BLI_assert_unreachable();
  }

  pbvh.tag_positions_changed(node_mask);
  pbvh.update_bounds(depsgraph, ob);
  flush_update_step(C, UpdateType::Position);
  flush_update_done(C, ob, UpdateType::Position);
  undo::push_end(ob);
}

static bool edit_op_init(bContext *C, wmOperator *op)
{
  Object *ob = CTX_data_active_object(C);
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  const EditMode mode = EditMode(RNA_enum_get(op->ptr, "mode"));
  const bool modify_hidden = RNA_boolean_get(op->ptr, "modify_hidden");

  if (!edit_is_operation_valid(*ob, mode, modify_hidden)) {
    return false;
  }

  BKE_sculptsession_update_for_edit(depsgraph, ob, false);

  return true;
}

static wmOperatorStatus edit_op_exec(bContext *C, wmOperator *op)
{
  if (!edit_op_init(C, op)) {
    return OPERATOR_CANCELLED;
  }

  const Scene &scene = *CTX_data_scene(C);
  const Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  Object &ob = *CTX_data_active_object(C);

  const int active_face_set = RNA_int_get(op->ptr, "active_face_set");
  const EditMode mode = EditMode(RNA_enum_get(op->ptr, "mode"));
  const bool modify_hidden = RNA_boolean_get(op->ptr, "modify_hidden");

  if (ELEM(mode, EditMode::Grow, EditMode::Shrink)) {
    ed::sculpt_paint::face_set_overlay_check(*C, *op);
  }

  switch (mode) {
    case EditMode::DeleteGeometry:
      edit_modify_geometry(C, ob, active_face_set, modify_hidden, op);
      break;
    case EditMode::Grow:
    case EditMode::Shrink: {
      if (bke::object::pbvh_get(ob)->type() == bke::pbvh::Type::Multires) {
        edit_grow_shrink_multires(depsgraph, scene, ob, mode, active_face_set, modify_hidden, op);
      }
      else {
        edit_grow_shrink_mesh(depsgraph, scene, ob, mode, active_face_set, modify_hidden, op);
      }
      break;
    }
    case EditMode::FairPositions:
    case EditMode::FairTangency:
      edit_modify_coordinates(C, ob, active_face_set, mode, op);
      break;
  }

  tag_update_overlays(C);

  return OPERATOR_FINISHED;
}

static wmOperatorStatus edit_op_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  Depsgraph *depsgraph = CTX_data_depsgraph_pointer(C);
  Object &ob = *CTX_data_active_object(C);

  const View3D *v3d = CTX_wm_view3d(C);
  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  BKE_sculptsession_update_for_edit(depsgraph, &ob, false);

  /* Update the current active face set and Vertex as the operator can be used directly from the
   * tool without brush cursor. */
  const float mval_fl[2] = {float(event->mval[0]), float(event->mval[1])};
  if (!cursor_geometry_info_update(C, mval_fl, false)) {
    /* The cursor is not over the mesh. Cancel to avoid editing the last updated face set ID. */
    return OPERATOR_CANCELLED;
  }
  RNA_int_set(op->ptr, "active_face_set", active_face_set_get(ob));

  return edit_op_exec(C, op);
}

void SCULPT_OT_face_sets_edit(wmOperatorType *ot)
{
  ot->name = "Edit Face Set";
  ot->idname = "SCULPT_OT_face_set_edit";
  ot->description = "Edits the current active face set";

  ot->invoke = edit_op_invoke;
  ot->exec = edit_op_exec;
  ot->poll = sculpt_mode_poll;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO | OPTYPE_DEPENDS_ON_CURSOR;

  PropertyRNA *prop = RNA_def_int(
      ot->srna, "active_face_set", 1, 0, INT_MAX, "Active Face Set", "", 0, 64);
  RNA_def_property_flag(prop, PROP_HIDDEN);

  static EnumPropertyItem modes[] = {
      {int(EditMode::Grow),
       "GROW",
       0,
       "Grow Face Set",
       "Grows the face set boundary by one face based on mesh topology"},
      {int(EditMode::Shrink),
       "SHRINK",
       0,
       "Shrink Face Set",
       "Shrinks the face set boundary by one face based on mesh topology"},
      {int(EditMode::DeleteGeometry),
       "DELETE_GEOMETRY",
       0,
       "Delete Geometry",
       "Deletes the faces that are assigned to the face set"},
      {int(EditMode::FairPositions),
       "FAIR_POSITIONS",
       0,
       "Fair Positions",
       "Creates the smoothest possible geometry patch from the face set minimizing changes in "
       "vertex positions"},
      {int(EditMode::FairTangency),
       "FAIR_TANGENCY",
       0,
       "Fair Tangency",
       "Creates the smoothest possible geometry patch from the face set minimizing changes in "
       "vertex tangents"},
      {0, nullptr, 0, nullptr, nullptr},
  };
  RNA_def_enum(ot->srna, "mode", modes, int(EditMode::Grow), "Mode", "");
  prop = RNA_def_float(ot->srna, "strength", 1.0f, 0.0f, 1.0f, "Strength", "", 0.0f, 1.0f);
  RNA_def_property_translation_context(prop, BLT_I18NCONTEXT_AMOUNT);

  ot->prop = RNA_def_boolean(ot->srna,
                             "modify_hidden",
                             false,
                             "Modify Hidden",
                             "Apply the edit operation to hidden geometry");
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Gesture Operators
 * Operators that modify face sets based on a selected area.
 * \{ */

void fill_factor_from_hide_and_mask(const BMesh &bm,
                                    const Set<BMFace *, 0L> &faces,
                                    const MutableSpan<float> r_factors)
{
  BLI_assert(faces.size() == r_factors.size());

  /* TODO: Avoid overhead of accessing attributes for every bke::pbvh::Tree node. */
  const int mask_offset = CustomData_get_offset_named(&bm.vdata, CD_PROP_FLOAT, ".sculpt_mask");
  int i = 0;
  for (BMFace *f : faces) {
    if (BM_elem_flag_test(f, BM_ELEM_HIDDEN)) {
      r_factors[i] = 0.0f;
      i++;
      continue;
    }
    if (mask_offset == -1) {
      r_factors[i] = 1.0f;
      i++;
      continue;
    }

    const BMLoop *l_iter = f->l_first = BM_FACE_FIRST_LOOP(f);
    int total_verts = 0;
    float sum = 0.0f;
    do {
      BMVert *vert = l_iter->v;
      sum += BM_ELEM_CD_GET_FLOAT(vert, mask_offset);
      total_verts++;
    } while ((l_iter = l_iter->next) != f->l_first);
    r_factors[i] = 1.0f - sum * math::rcp(float(total_verts));
    i++;
  }
}

void fill_factor_from_hide_and_mask(const Mesh &mesh,
                                    const Span<int> face_indices,
                                    const MutableSpan<float> r_factors)
{
  BLI_assert(face_indices.size() == r_factors.size());

  const OffsetIndices<int> faces = mesh.faces();
  const Span<int> corner_verts = mesh.corner_verts();

  /* TODO: Avoid overhead of accessing attributes for every bke::pbvh::Tree node. */
  const bke::AttributeAccessor attributes = mesh.attributes();
  if (const VArray mask = *attributes.lookup<float>(".sculpt_mask", bke::AttrDomain::Point)) {
    const VArraySpan span(mask);
    for (const int i : face_indices.index_range()) {
      const Span<int> face_verts = corner_verts.slice(faces[face_indices[i]]);
      const float inv_size = math::rcp(float(face_verts.size()));
      float sum = 0.0f;
      for (const int vert : face_verts) {
        sum += span[vert];
      }
      r_factors[i] = 1.0f - sum * inv_size;
    }
  }
  else {
    r_factors.fill(1.0f);
  }

  if (const VArray hide_poly = *attributes.lookup<bool>(".hide_poly", bke::AttrDomain::Face)) {
    const VArraySpan span(hide_poly);
    for (const int i : face_indices.index_range()) {
      if (span[face_indices[i]]) {
        r_factors[i] = 0.0f;
      }
    }
  }
}

void fill_factors_from_hide_and_mask(const Multiresolution &multires,
                                     const Span<int> subfaces,
                                     const MutableSpan<float> r_factors)
{
  BLI_assert(subfaces.size() == r_factors.size());
  r_factors.fill(1.0f);

  const Span<bool> hidden_faces = multires.hidden_faces();
  const Span<float> masks = multires.masks();
  const Span<int> corner_verts = multires.corner_verts();

  if (hidden_faces.is_empty() && masks.is_empty()) {
    return;
  }

  for (const int i : subfaces.index_range()) {
    const int subface_idx = subfaces[i];

    if (!hidden_faces.is_empty() && hidden_faces[subface_idx]) {
      r_factors[i] = 0.0f;
      continue;
    }

    if (!masks.is_empty()) {
      const int v0 = corner_verts[subface_idx * 4 + 0];
      const int v1 = corner_verts[subface_idx * 4 + 1];
      const int v2 = corner_verts[subface_idx * 4 + 2];
      const int v3 = corner_verts[subface_idx * 4 + 3];
      const float avg_mask = (masks[v0] + masks[v1] + masks[v2] + masks[v3]) * 0.25f;
      r_factors[i] = 1.0f - avg_mask;
    }
  }
}

void calc_face_centers(const OffsetIndices<int> faces,
                       const Span<int> corner_verts,
                       const Span<float3> vert_positions,
                       const Span<int> face_indices,
                       const MutableSpan<float3> positions)
{
  BLI_assert(face_indices.size() == positions.size());

  for (const int i : face_indices.index_range()) {
    positions[i] = bke::mesh::face_center_calc(vert_positions,
                                               corner_verts.slice(faces[face_indices[i]]));
  }
}

void calc_face_centers(const Set<BMFace *, 0L> &faces, const MutableSpan<float3> centers)
{
  BLI_assert(faces.size() == centers.size());

  int i = 0;
  for (const BMFace *f : faces) {
    float3 face_center;
    BM_face_calc_center_median(f, face_center);

    centers[i] = face_center;
    i++;
  }
}

void calc_face_indices_grids(const SubdivCCG &subdiv_ccg,
                             const Span<int> grids,
                             const MutableSpan<int> &face_indices)
{
  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);
  BLI_assert(grids.size() * key.grid_area == face_indices.size());

  for (const int i : grids.index_range()) {
    const int start = i * key.grid_area;
    for (const int offset : IndexRange(key.grid_area)) {
      face_indices[start + offset] = BKE_subdiv_ccg_grid_to_face_index(subdiv_ccg, grids[i]);
    }
  }
}

struct FaceSetOperation {
  gesture::Operation op;

  int new_face_set_id;
};

static void gesture_begin(bContext &C, wmOperator &op, gesture::GestureData &gesture_data)
{
  const Scene &scene = *CTX_data_scene(&C);
  Depsgraph *depsgraph = CTX_data_depsgraph_pointer(&C);
  BKE_sculptsession_update_for_edit(depsgraph, gesture_data.vc.obact, false);
  undo::push_begin(scene, *gesture_data.vc.obact, &op);
}

static void gesture_apply_mesh(gesture::GestureData &gesture_data, const IndexMask &node_mask)
{
  FaceSetOperation *face_set_operation = reinterpret_cast<FaceSetOperation *>(
      gesture_data.operation);
  const int new_face_set = face_set_operation->new_face_set_id;
  const Depsgraph &depsgraph = *gesture_data.vc.depsgraph;
  Object &object = *gesture_data.vc.obact;
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  SculptSession &ss = *gesture_data.ss;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);

  const Span<float3> positions_eval = bke::pbvh::vert_positions_eval(depsgraph, object);
  bke::SpanAttributeWriter<int> face_sets = face_set::ensure_face_sets_mesh(mesh);

  struct TLS {
    Vector<int> face_indices;
    Vector<float3> positions;
    Vector<float> factors;
  };

  Array<bool> node_changed(pbvh.nodes_num(), false);

  threading::EnumerableThreadSpecific<TLS> all_tls;
  if (pbvh.type() == bke::pbvh::Type::Mesh) {
    const OffsetIndices<int> faces = mesh.faces();
    const Span<int> corner_verts = mesh.corner_verts();
    MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
    node_mask.foreach_index(
        [&](const int i) {
          TLS &tls = all_tls.local();
          undo::push_node(depsgraph, *gesture_data.vc.obact, &nodes[i], undo::Type::FaceSet);
          bool any_updated = false;

          const Span<int> face_indices = nodes[i].faces();

          tls.positions.resize(face_indices.size());
          const MutableSpan<float3> face_centers = tls.positions;
          calc_face_centers(
              mesh.faces(), corner_verts, positions_eval, face_indices, face_centers);

          tls.factors.resize(face_indices.size());
          const MutableSpan<float> factors = tls.factors;
          fill_factor_from_hide_and_mask(mesh, face_indices, factors);
          filter_region_clip_factors(ss, face_centers, factors);

          for (const int idx : face_indices.index_range()) {
            if (factors[idx] < FACE_SET_MIN_FADE) {
              continue;
            }

            const int face = face_indices[idx];

            const Span<int> face_verts = corner_verts.slice(faces[face]);
            const float3 face_normal = bke::mesh::face_normal_calc(positions_eval, face_verts);
            if (!gesture::is_affected(gesture_data, face_centers[idx], face_normal)) {
              continue;
            }
            face_sets.span[face] = new_face_set;
            any_updated = true;
          }
          if (any_updated) {
            node_changed[i] = true;
          }
        },
        exec_mode::grain_size(1));
  }
  else if (pbvh.type() == bke::pbvh::Type::Grids) {
    const OffsetIndices<int> faces = mesh.faces();
    const Span<int> corner_verts = mesh.corner_verts();
    MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
    node_mask.foreach_index(
        [&](const int i) {
          TLS &tls = all_tls.local();
          undo::push_node(depsgraph, *gesture_data.vc.obact, &nodes[i], undo::Type::FaceSet);

          bool any_updated = false;

          const Span<int> grids = nodes[i].grids();
          const MutableSpan positions = gather_grids_positions(
              *ss.subdiv_ccg, grids, tls.positions);

          tls.factors.resize(positions.size());
          const MutableSpan<float> factors_grid = tls.factors;
          ed::sculpt_paint::fill_factor_from_hide_and_mask(*ss.subdiv_ccg, grids, factors_grid);
          ed::sculpt_paint::filter_region_clip_factors(ss, positions_eval, factors_grid);

          tls.face_indices.resize(positions.size());
          Vector<int> face_indices_grid = tls.face_indices;
          calc_face_indices_grids(*ss.subdiv_ccg, grids, face_indices_grid);

          for (const int idx : face_indices_grid.index_range()) {
            if (factors_grid[idx] < FACE_SET_MIN_FADE) {
              continue;
            }

            const int face = face_indices_grid[idx];

            const Span<int> face_verts = corner_verts.slice(faces[face]);
            const float3 face_center = bke::mesh::face_center_calc(positions_eval, face_verts);
            const float3 face_normal = bke::mesh::face_normal_calc(positions_eval, face_verts);
            if (!gesture::is_affected(gesture_data, face_center, face_normal)) {
              continue;
            }
            face_sets.span[face] = new_face_set;
            any_updated = true;
          }
          if (any_updated) {
            node_changed[i] = true;
          }
        },
        exec_mode::grain_size(1));
  }

  IndexMaskMemory memory;
  pbvh.tag_face_sets_changed(IndexMask::from_bools(node_changed, memory));
  face_sets.finish();
}

static void gesture_apply_multires(gesture::GestureData &gesture_data, const IndexMask &node_mask)
{
  FaceSetOperation *face_set_operation = reinterpret_cast<FaceSetOperation *>(
      gesture_data.operation);
  const int new_face_set = face_set_operation->new_face_set_id;
  const Depsgraph &depsgraph = *gesture_data.vc.depsgraph;
  Object &object = *gesture_data.vc.obact;
  Mesh &base_mesh = *id_cast<Mesh *>(object.data);
  SculptSession &ss = *gesture_data.ss;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);

  MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
  Multiresolution &multires = *base_mesh.multires;
  const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(base_mesh);
  MutableSpan<int> face_sets = ensure_face_sets_multires(base_mesh);
  const Span<int> corner_verts = multires.corner_verts();
  const Span<float3> positions = multires.positions();
  const OffsetIndices<int> base_faces = base_topo.faces;

  struct TLS {
    Vector<int> subfaces;
    Vector<float3> positions;
    Vector<float3> normals;
    Vector<float> factors;
  };
  threading::EnumerableThreadSpecific<TLS> all_tls;
  Array<bool> node_changed(pbvh.nodes_num(), false);

  node_mask.foreach_index(
      [&](const int i) {
        TLS &tls = all_tls.local();
        const Span<int> base_node_faces = nodes[i].faces();
        const int subfaces_count = nodes[i].corners_num * multires.faces_per_corner();

        tls.subfaces.resize(subfaces_count);
        gather_subfaces(multires.faces_per_corner(), base_faces, base_node_faces, tls.subfaces);
        const Span<int> node_subfaces = tls.subfaces;

        tls.positions.resize(subfaces_count);
        tls.normals.resize(subfaces_count);
        calc_face_centers_and_normals(
            node_subfaces, corner_verts, positions, tls.positions, tls.normals);

        tls.factors.resize(subfaces_count);
        fill_factors_from_hide_and_mask(multires, node_subfaces, tls.factors);
        filter_region_clip_factors(ss, tls.positions, tls.factors);

        undo::push_node(depsgraph, *gesture_data.vc.obact, &nodes[i], undo::Type::FaceSet);

        bool any_updated = false;
        for (const int j : node_subfaces.index_range()) {
          if (tls.factors[j] < FACE_SET_MIN_FADE) {
            continue;
          }
          if (!gesture::is_affected(gesture_data, tls.positions[j], tls.normals[j])) {
            continue;
          }

          const int sf = node_subfaces[j];
          if (face_sets[sf] != new_face_set) {
            face_sets[sf] = new_face_set;
            any_updated = true;
          }
        }
        if (any_updated) {
          node_changed[i] = true;
        }
      },
      exec_mode::grain_size(1));

  IndexMaskMemory memory;
  pbvh.tag_face_sets_changed(IndexMask::from_bools(node_changed, memory));
}

static void gesture_apply_bmesh(gesture::GestureData &gesture_data, const IndexMask &node_mask)
{
  SculptSession &ss = *gesture_data.ss;
  FaceSetOperation *face_set_operation = reinterpret_cast<FaceSetOperation *>(
      gesture_data.operation);
  const Depsgraph &depsgraph = *gesture_data.vc.depsgraph;
  const int new_face_set = face_set_operation->new_face_set_id;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(*gesture_data.vc.obact);
  MutableSpan<bke::pbvh::BMeshNode> nodes = pbvh.nodes<bke::pbvh::BMeshNode>();
  const int offset = face_set::ensure_face_sets_bmesh(*gesture_data.vc.obact);

  struct TLS {
    Vector<float3> positions;
    Vector<float> factors;
  };

  Array<bool> node_changed(node_mask.min_array_size(), false);

  threading::EnumerableThreadSpecific<TLS> all_tls;
  node_mask.foreach_index(
      [&](const int i) {
        TLS &tls = all_tls.local();
        undo::push_node(depsgraph, *gesture_data.vc.obact, &nodes[i], undo::Type::FaceSet);

        bool any_updated = false;

        const Set<BMFace *, 0> &faces = BKE_pbvh_bmesh_node_faces(&nodes[i]);

        tls.positions.resize(faces.size());
        const MutableSpan<float3> face_centers = tls.positions;
        calc_face_centers(faces, face_centers);

        tls.factors.resize(faces.size());
        const MutableSpan<float> factors = tls.factors;
        fill_factor_from_hide_and_mask(*ss.bm, faces, factors);
        filter_region_clip_factors(ss, face_centers, factors);

        int idx = 0;
        for (BMFace *face : faces) {
          if (factors[idx] < FACE_SET_MIN_FADE) {
            idx++;
            continue;
          }
          if (!gesture::is_affected(gesture_data, face_centers[idx], face->no)) {
            idx++;
            continue;
          }
          BM_ELEM_CD_SET_INT(face, offset, new_face_set);
          any_updated = true;
          idx++;
        }
        if (any_updated) {
          node_changed[i] = true;
        }
      },
      exec_mode::grain_size(1));

  IndexMaskMemory memory;
  const IndexMask changed_nodes = IndexMask::from_bools(node_changed, memory);
  if (changed_nodes.is_empty()) {
    return;
  }
  pbvh.tag_face_sets_changed(node_mask);
}

static void gesture_apply_for_symmetry_pass(bContext & /*C*/, gesture::GestureData &gesture_data)
{
  switch (bke::object::pbvh_get(*gesture_data.vc.obact)->type()) {
    case bke::pbvh::Type::Mesh:
    case bke::pbvh::Type::Grids:
      gesture_apply_mesh(gesture_data, gesture_data.node_mask);
      break;
    case bke::pbvh::Type::Multires:
      gesture_apply_multires(gesture_data, gesture_data.node_mask);
      break;
    case bke::pbvh::Type::BMesh:
      gesture_apply_bmesh(gesture_data, gesture_data.node_mask);
      break;
  }
}

static void gesture_end(bContext & /*C*/, gesture::GestureData &gesture_data)
{
  undo::push_end(*gesture_data.vc.obact);
}

static void init_operation(gesture::GestureData &gesture_data, wmOperator & /*op*/)
{
  Object &object = *gesture_data.vc.obact;
  gesture_data.operation = reinterpret_cast<gesture::Operation *>(
      MEM_new_zeroed<FaceSetOperation>(__func__));

  FaceSetOperation *face_set_operation = reinterpret_cast<FaceSetOperation *>(
      gesture_data.operation);

  face_set_operation->op.begin = gesture_begin;
  face_set_operation->op.apply_for_symmetry_pass = gesture_apply_for_symmetry_pass;
  face_set_operation->op.end = gesture_end;

  face_set_operation->new_face_set_id = face_set::find_next_available_id(object);
}

static wmOperatorStatus gesture_box_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  const View3D *v3d = CTX_wm_view3d(C);
  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  return WM_gesture_box_invoke(C, op, event);
}

static wmOperatorStatus gesture_box_exec(bContext *C, wmOperator *op)
{
  std::unique_ptr<gesture::GestureData> gesture_data = gesture::init_from_box(C, op);
  if (!gesture_data) {
    return OPERATOR_CANCELLED;
  }
  init_operation(*gesture_data, *op);
  ed::sculpt_paint::face_set_overlay_check(*C, *op);
  gesture::apply(*C, *gesture_data, *op);
  return OPERATOR_FINISHED;
}

static wmOperatorStatus gesture_lasso_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  const View3D *v3d = CTX_wm_view3d(C);
  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  return WM_gesture_lasso_invoke(C, op, event);
}

static wmOperatorStatus gesture_lasso_exec(bContext *C, wmOperator *op)
{
  std::unique_ptr<gesture::GestureData> gesture_data = gesture::init_from_lasso(C, op);
  if (!gesture_data) {
    return OPERATOR_CANCELLED;
  }
  init_operation(*gesture_data, *op);
  ed::sculpt_paint::face_set_overlay_check(*C, *op);
  gesture::apply(*C, *gesture_data, *op);
  return OPERATOR_FINISHED;
}

static wmOperatorStatus gesture_line_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  const View3D *v3d = CTX_wm_view3d(C);
  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  return WM_gesture_straightline_active_side_invoke(C, op, event);
}

static wmOperatorStatus gesture_line_exec(bContext *C, wmOperator *op)
{
  std::unique_ptr<gesture::GestureData> gesture_data = gesture::init_from_line(C, op);
  if (!gesture_data) {
    return OPERATOR_CANCELLED;
  }
  init_operation(*gesture_data, *op);
  ed::sculpt_paint::face_set_overlay_check(*C, *op);
  gesture::apply(*C, *gesture_data, *op);
  return OPERATOR_FINISHED;
}

static wmOperatorStatus gesture_polyline_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  const View3D *v3d = CTX_wm_view3d(C);
  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  return WM_gesture_polyline_invoke(C, op, event);
}

static wmOperatorStatus gesture_polyline_exec(bContext *C, wmOperator *op)
{
  std::unique_ptr<gesture::GestureData> gesture_data = gesture::init_from_polyline(C, op);
  if (!gesture_data) {
    return OPERATOR_CANCELLED;
  }
  init_operation(*gesture_data, *op);
  ed::sculpt_paint::face_set_overlay_check(*C, *op);
  gesture::apply(*C, *gesture_data, *op);
  return OPERATOR_FINISHED;
}

void SCULPT_OT_face_set_polyline_gesture(wmOperatorType *ot)
{
  ot->name = "Face Set Lasso Gesture";
  ot->idname = "SCULPT_OT_face_set_polyline_gesture";
  ot->description = "Add a face set in a shape defined by the cursor";

  ot->invoke = gesture_polyline_invoke;
  ot->modal = WM_gesture_polyline_modal;
  ot->exec = gesture_polyline_exec;

  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_DEPENDS_ON_CURSOR;

  WM_operator_properties_gesture_polyline(ot);
  gesture::operator_properties(ot, gesture::ShapeType::Lasso);
}

void SCULPT_OT_face_set_box_gesture(wmOperatorType *ot)
{
  ot->name = "Face Set Box Gesture";
  ot->idname = "SCULPT_OT_face_set_box_gesture";
  ot->description = "Add a face set in a rectangle defined by the cursor";

  ot->invoke = gesture_box_invoke;
  ot->modal = WM_gesture_box_modal;
  ot->exec = gesture_box_exec;

  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER;

  WM_operator_properties_border(ot);
  gesture::operator_properties(ot, gesture::ShapeType::Box);
}

void SCULPT_OT_face_set_lasso_gesture(wmOperatorType *ot)
{
  ot->name = "Face Set Lasso Gesture";
  ot->idname = "SCULPT_OT_face_set_lasso_gesture";
  ot->description = "Add a face set in a shape defined by the cursor";

  ot->invoke = gesture_lasso_invoke;
  ot->modal = WM_gesture_lasso_modal;
  ot->exec = gesture_lasso_exec;

  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_DEPENDS_ON_CURSOR;

  WM_operator_properties_gesture_lasso(ot);
  gesture::operator_properties(ot, gesture::ShapeType::Lasso);
}

void SCULPT_OT_face_set_line_gesture(wmOperatorType *ot)
{
  ot->name = "Face Set Line Gesture";
  ot->idname = "SCULPT_OT_face_set_line_gesture";
  ot->description = "Add a face set to one side of a line defined by the cursor";

  ot->invoke = gesture_line_invoke;
  ot->modal = WM_gesture_straightline_oneshot_modal;
  ot->exec = gesture_line_exec;

  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER;

  WM_operator_properties_gesture_straightline(ot, WM_CURSOR_EDIT);
  gesture::operator_properties(ot, gesture::ShapeType::Line);
}
/** \} */

}  // namespace blender::ed::sculpt_paint::face_set
