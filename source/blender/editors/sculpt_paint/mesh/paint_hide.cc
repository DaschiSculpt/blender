/* SPDX-FileCopyrightText: 2010 by Nicholas Bishop. All rights reserved.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edsculpt
 * Implements the bke::pbvh::Tree node hiding operator.
 */
#include "paint_hide.hh"

#include "MEM_guardedalloc.h"

#include "BLI_array_utils.hh"
#include "BLI_bit_span_ops.hh"
#include "BLI_enumerable_thread_specific.hh"
#include "BLI_utildefines.hh"
#include "BLI_vector.hh"

#include "DNA_object_types.h"

#include "BKE_attribute.hh"
#include "BKE_ccg.hh"
#include "BKE_context.hh"
#include "BKE_mesh.hh"
#include "BKE_multires.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_subdiv_ccg.hh"

#include "DEG_depsgraph.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "ED_screen.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "bmesh.hh"

#include "../paint_intern.hh"
#include "mesh_brush_common.hh"
#include "sculpt_gesture.hh"
#include "sculpt_intern.hh"
#include "sculpt_islands.hh"
#include "sculpt_undo.hh"

namespace blender::ed::sculpt_paint::hide {

/* -------------------------------------------------------------------- */
/** \name Public API
 * \{ */

static void multires_hide_vert_flush(Multiresolution &multires)
{
  bke::MutableAttributeAccessor attributes = multires.attributes_for_write();

  const VArray<bool> hidden_verts = *attributes.lookup_or_default<bool>(
      ".hide_vert", bke::AttrDomain::Point, false);

  if (hidden_verts.is_single() && !hidden_verts.get_internal_single()) {
    attributes.remove(".hide_poly");
    return;
  }

  const MutableSpan<bool> hidden_faces = multires.hidden_faces_for_write();
  const Span<int> corner_verts = multires.corner_verts();

  threading::parallel_for(hidden_faces.index_range(), 4096, [&](const IndexRange range) {
    for (const int subface_idx : range) {
      const int v0 = corner_verts[subface_idx * 4 + 0];
      const int v1 = corner_verts[subface_idx * 4 + 1];
      const int v2 = corner_verts[subface_idx * 4 + 2];
      const int v3 = corner_verts[subface_idx * 4 + 3];

      hidden_faces[subface_idx] = hidden_verts[v0] || hidden_verts[v1] || hidden_verts[v2] ||
                                  hidden_verts[v3];
    }
  });
}

void sync_all_from_verts(Object &object)
{
  using namespace blender::bke;
  const SculptSession &ss = *object.runtime->sculpt_session;
  const pbvh::Tree &pbvh = *object::pbvh_get(object);
  Mesh &mesh = *id_cast<Mesh *>(object.data);

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh: {
      mesh_hide_vert_flush(mesh);
      break;
    }
    case bke::pbvh::Type::Multires: {
      multires_hide_vert_flush(*mesh.multires);
      break;
    }
    case bke::pbvh::Type::BMesh: {
      BMesh &bm = *ss.bm;
      BMIter iter;
      BMVert *v;
      BMEdge *e;
      BMFace *f;

      BM_ITER_MESH (f, &iter, &bm, BM_FACES_OF_MESH) {
        BM_elem_flag_disable(f, BM_ELEM_HIDDEN);
      }

      BM_ITER_MESH (e, &iter, &bm, BM_EDGES_OF_MESH) {
        BM_elem_flag_disable(e, BM_ELEM_HIDDEN);
      }

      BM_ITER_MESH (v, &iter, &bm, BM_VERTS_OF_MESH) {
        if (!BM_elem_flag_test(v, BM_ELEM_HIDDEN)) {
          continue;
        }
        BMIter iter_l;
        BMLoop *l;

        BM_ITER_ELEM (l, &iter_l, v, BM_LOOPS_OF_VERT) {
          BM_elem_flag_enable(l->e, BM_ELEM_HIDDEN);
          BM_elem_flag_enable(l->f, BM_ELEM_HIDDEN);
        }
      }
      break;
    }
    case bke::pbvh::Type::Grids: {
      Mesh &mesh = *id_cast<Mesh *>(object.data);
      const SubdivCCG &subdiv_ccg = *ss.subdiv_ccg;
      const BitGroupVector<> &grid_hidden = subdiv_ccg.grid_hidden;
      const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);
      const OffsetIndices faces = mesh.faces();

      IndexMaskMemory memory;
      const IndexMask hidden_faces =
          !grid_hidden.is_empty() ?
              IndexMask::from_predicate(
                  faces.index_range(),
                  memory,
                  [&](const int i) {
                    const IndexRange face = faces[i];
                    return std::any_of(face.begin(), face.end(), [&](const int corner) {
                      return grid_hidden[corner][key.grid_area - 1];
                    });
                  },
                  exec_mode::grain_size(1024)) :
              IndexMask();

      MutableAttributeAccessor attributes = mesh.attributes_for_write();
      if (hidden_faces.is_empty()) {
        attributes.remove(".hide_poly");
      }
      else {
        SpanAttributeWriter<bool> hide_poly = attributes.ensure_for_write_span<bool>(
            ".hide_poly", AttrDomain::Face, AttributeInitConstruct());
        hide_poly.span.fill(false);
        index_mask::masked_fill(hide_poly.span, true, hidden_faces);
        hide_poly.finish();
      }

      mesh_hide_face_flush(mesh);
      break;
    }
  }
}

void sync_all_from_faces(Object &object)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  Mesh &mesh = *id_cast<Mesh *>(object.data);

  islands::invalidate(ss);
  const bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh: {
      /* We may have adjusted the ".hide_poly" attribute, now make the hide status attributes for
       * vertices and edges consistent. */
      bke::mesh_hide_face_flush(mesh);
      break;
    }
    case bke::pbvh::Type::Multires: {
      mesh.multires->hide_face_flush();
      break;
    }
    case bke::pbvh::Type::Grids: {
      /* In addition to making the hide status of the base mesh consistent, we also have to
       * propagate the status to the Multires grids. */
      bke::mesh_hide_face_flush(mesh);
      BKE_sculpt_sync_face_visibility_to_grids(mesh, *ss.subdiv_ccg);
      break;
    }
    case bke::pbvh::Type::BMesh: {
      BMesh &bm = *ss.bm;
      BMIter iter;
      BMFace *f;

      /* Hide all verts and edges attached to faces. */
      BM_ITER_MESH (f, &iter, &bm, BM_FACES_OF_MESH) {
        BMLoop *l = f->l_first;
        do {
          BM_elem_flag_enable(l->v, BM_ELEM_HIDDEN);
          BM_elem_flag_enable(l->e, BM_ELEM_HIDDEN);
        } while ((l = l->next) != f->l_first);
      }

      /* Unhide verts and edges attached to visible faces. */
      BM_ITER_MESH (f, &iter, &bm, BM_FACES_OF_MESH) {
        if (BM_elem_flag_test(f, BM_ELEM_HIDDEN)) {
          continue;
        }

        BMLoop *l = f->l_first;
        do {
          BM_elem_flag_disable(l->v, BM_ELEM_HIDDEN);
          BM_elem_flag_disable(l->e, BM_ELEM_HIDDEN);
        } while ((l = l->next) != f->l_first);
      }
      break;
    }
  }
}

void tag_update_visibility(const bContext &C)
{
  ARegion *region = CTX_wm_region(&C);
  ED_region_tag_redraw(region);

  Object *ob = CTX_data_active_object(&C);
  WM_event_add_notifier(&C, NC_OBJECT | ND_DRAW, ob);

  DEG_id_tag_update(&ob->id, ID_RECALC_SHADING);
  const RegionView3D *rv3d = CTX_wm_region_view3d(&C);
  if (!BKE_sculptsession_use_pbvh_draw(ob, rv3d)) {
    DEG_id_tag_update(&ob->id, ID_RECALC_GEOMETRY);
  }
}

void mesh_show_all(const Depsgraph &depsgraph, Object &object, const IndexMask &node_mask)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  const VArraySpan hide_vert = *attributes.lookup<bool>(".hide_vert", bke::AttrDomain::Point);

  if (!hide_vert.is_empty()) {
    IndexMaskMemory memory;
    const IndexMask changed_nodes = IndexMask::from_predicate(
        node_mask,
        memory,
        [&](const int i) {
          const Span<int> verts = nodes[i].verts();
          return std::any_of(
              verts.begin(), verts.end(), [&](const int i) { return hide_vert[i]; });
        },
        exec_mode::grain_size(1));
    undo::push_nodes(depsgraph, object, changed_nodes, undo::Type::HideVert);
    pbvh.tag_visibility_changed(changed_nodes);
  }

  attributes.remove(".hide_vert");
  bke::mesh_hide_vert_flush(mesh);
  pbvh.update_visibility(object);
}

void multires_show_all(Depsgraph &depsgraph, Object &object, const IndexMask &node_mask)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  Multiresolution &multires = *mesh.multires;
  const int grid_size = multires.grid_size();
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
  const Span<bool> hidden_verts = multires.hidden_verts();

  if (!hidden_verts.is_empty()) {
    IndexMaskMemory memory;
    const IndexMask changed_nodes = IndexMask::from_predicate(
        node_mask,
        memory,
        [&](const int i) {
          const Span<bool> node_hidden = hidden_verts.slice(nodes[i].range(grid_size));
          return std::any_of(
              node_hidden.begin(), node_hidden.end(), [&](const bool value) { return value; });
        },
        exec_mode::grain_size(1));

    undo::push_nodes(depsgraph, object, changed_nodes, undo::Type::HideVert);
    pbvh.tag_visibility_changed(changed_nodes);
  }

  multires.hidden_verts_free();
  multires_hide_vert_flush(multires);
  pbvh.update_visibility(object);
}

void grids_show_all(Depsgraph &depsgraph, Object &object, const IndexMask &node_mask)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
  SubdivCCG &subdiv_ccg = *object.runtime->sculpt_session->subdiv_ccg;
  const BitGroupVector<> &grid_hidden = subdiv_ccg.grid_hidden;

  if (!grid_hidden.is_empty()) {
    IndexMaskMemory memory;
    const IndexMask changed_nodes = IndexMask::from_predicate(
        node_mask,
        memory,
        [&](const int i) {
          const Span<int> grids = nodes[i].grids();
          return std::any_of(grids.begin(), grids.end(), [&](const int i) {
            return bits::any_bit_set(grid_hidden[i]);
          });
        },
        exec_mode::grain_size(1));
    if (changed_nodes.is_empty()) {
      return;
    }
    undo::push_nodes(depsgraph, object, changed_nodes, undo::Type::HideVert);
    pbvh.tag_visibility_changed(changed_nodes);
  }

  BKE_subdiv_ccg_grid_hidden_free(subdiv_ccg);
  sync_all_from_verts(object);
  pbvh.update_visibility(object);
  multires_mark_as_modified(&depsgraph, &object, MULTIRES_HIDDEN_MODIFIED);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Internal Visibility Utilities
 * Functions that assist with applying changes to the different bke::pbvh::Tree types.
 * \{ */

enum class VisAction {
  Hide = 0,
  Show = 1,
};

static bool action_to_hide(const VisAction action)
{
  return action == VisAction::Hide;
}

/* Calculates whether a face should be hidden based on all of its corner vertices. */
static void calc_face_hide(const Span<int> node_faces,
                           const OffsetIndices<int> faces,
                           const Span<int> corner_verts,
                           const Span<bool> hide_vert,
                           MutableSpan<bool> hide_face)
{
  for (const int i : node_faces.index_range()) {
    Span<int> face_verts = corner_verts.slice(faces[node_faces[i]]);
    hide_face[i] = std::any_of(
        face_verts.begin(), face_verts.end(), [&](const int v) { return hide_vert[v]; });
  }
}

/* Updates a node's face's visibility based on the updated vertex visibility. */
static void flush_face_changes_node(Mesh &mesh,
                                    bke::pbvh::Tree &pbvh,
                                    const IndexMask &node_mask,
                                    const Span<bool> hide_vert)
{
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();

  const OffsetIndices<int> faces = mesh.faces();
  const Span<int> corner_verts = mesh.corner_verts();

  bke::SpanAttributeWriter<bool> hide_poly = attributes.ensure_for_write_span<bool>(
      ".hide_poly", bke::AttrDomain::Face);

  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();

  Array<bool> node_changed(node_mask.min_array_size(), false);

  struct TLS {
    Vector<bool> new_hide;
  };
  threading::EnumerableThreadSpecific<TLS> all_tls;
  node_mask.foreach_index(
      [&](const int i) {
        TLS &tls = all_tls.local();
        const Span<int> node_faces = nodes[i].faces();

        tls.new_hide.resize(node_faces.size());
        gather_data_mesh(hide_poly.span.as_span(), node_faces, tls.new_hide.as_mutable_span());

        calc_face_hide(node_faces, faces, corner_verts, hide_vert, tls.new_hide.as_mutable_span());

        if (array_utils::indexed_data_equal<bool>(hide_poly.span, node_faces, tls.new_hide)) {
          return;
        }

        scatter_data_mesh(tls.new_hide.as_span(), node_faces, hide_poly.span);
        node_changed[i] = true;
        bke::pbvh::node_update_visibility_mesh(hide_vert, nodes[i]);
      },
      exec_mode::grain_size(1));
  hide_poly.finish();

  IndexMaskMemory memory;
  const IndexMask changed_nodes = IndexMask::from_bools(node_changed, memory);
  if (changed_nodes.is_empty()) {
    return;
  }
  pbvh.tag_visibility_changed(node_mask);
}

/* Updates a node's face's visibility based on the updated vertex visibility. */
static void flush_face_changes(Mesh &mesh, const Span<bool> hide_vert)
{
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();

  bke::SpanAttributeWriter<bool> hide_poly = attributes.ensure_for_write_span<bool>(
      ".hide_poly", bke::AttrDomain::Face);

  bke::mesh_face_hide_from_vert(mesh.faces(), mesh.corner_verts(), hide_vert, hide_poly.span);
  hide_poly.finish();
}

/* Updates all of a mesh's edge visibility based on vertex visibility. */
static void flush_edge_changes(Mesh &mesh, const Span<bool> hide_vert)
{
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();

  bke::SpanAttributeWriter<bool> hide_edge = attributes.ensure_for_write_only_span<bool>(
      ".hide_edge", bke::AttrDomain::Edge);
  bke::mesh_edge_hide_from_vert(mesh.edges(), hide_vert, hide_edge.span);
  hide_edge.finish();
}

static void vert_hide_update(const Depsgraph &depsgraph,
                             Object &object,
                             const IndexMask &node_mask,
                             const FunctionRef<void(Span<int>, MutableSpan<bool>)> calc_hide)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();

  Mesh &mesh = *id_cast<Mesh *>(object.data);
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  bke::SpanAttributeWriter<bool> hide_vert = attributes.ensure_for_write_span<bool>(
      ".hide_vert", bke::AttrDomain::Point);

  bool any_changed = false;
  threading::EnumerableThreadSpecific<Vector<bool>> all_new_hide;
  node_mask.foreach_index(
      [&](const int i) {
        Vector<bool> &new_hide = all_new_hide.local();
        const Span<int> verts = nodes[i].verts();

        new_hide.resize(verts.size());
        gather_data_mesh(hide_vert.span.as_span(), verts, new_hide.as_mutable_span());
        calc_hide(verts, new_hide);
        if (array_utils::indexed_data_equal<bool>(hide_vert.span, verts, new_hide)) {
          return;
        }

        any_changed = true;
        undo::push_node(depsgraph, object, &nodes[i], undo::Type::HideVert);
        scatter_data_mesh(new_hide.as_span(), verts, hide_vert.span);
      },
      exec_mode::grain_size(1));

  hide_vert.finish();
  if (any_changed) {
    /* We handle flushing ourselves at the node level instead of delegating to
     * #bke::mesh_hide_vert_flush because we need to tag node visibility changes as well in cases
     * where the vertices hidden are on a node boundary. */
    flush_face_changes_node(mesh, pbvh, node_mask, hide_vert.span);
    flush_edge_changes(mesh, hide_vert.span);
  }
}

static void multires_hide_update(
    Depsgraph &depsgraph,
    Object &object,
    const IndexMask &node_mask,
    const FunctionRef<void(const IndexRange, MutableSpan<bool>)> calc_hide)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  Multiresolution &multires = *mesh.multires;
  const int grid_size = multires.grid_size();
  const MutableSpan<bool> hidden = multires.hidden_verts_for_write();
  Array<bool> node_changed(node_mask.min_array_size(), false);

  node_mask.foreach_index(
      [&](const int i) {
        const IndexRange node_range = nodes[i].range(grid_size);
        MutableSpan<bool> node_hidden = hidden.slice(node_range);
        Array<bool> hidden_new(node_hidden.size());
        MutableSpan<bool>(hidden_new).copy_from(node_hidden);

        calc_hide(node_range, hidden_new);

        if (std::equal(
                node_hidden.begin(), node_hidden.end(), hidden_new.begin(), hidden_new.end())) {
          return;
        }

        node_changed[i] = true;
        undo::push_node(depsgraph, object, &nodes[i], undo::Type::HideVert);
        node_hidden.copy_from(hidden_new);
        bke::pbvh::node_update_visibility_multires(hidden, grid_size, nodes[i]);
      },
      exec_mode::grain_size(1));

  IndexMaskMemory memory;
  const IndexMask changed_nodes = IndexMask::from_bools(node_changed, memory);
  if (changed_nodes.is_empty()) {
    return;
  }
  pbvh.tag_visibility_changed(changed_nodes);
  sync_all_from_verts(object);
}

static void grid_hide_update(Depsgraph &depsgraph,
                             Object &object,
                             const IndexMask &node_mask,
                             const FunctionRef<void(const int, MutableBoundedBitSpan)> calc_hide)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();

  SubdivCCG &subdiv_ccg = *object.runtime->sculpt_session->subdiv_ccg;
  BitGroupVector<> &grid_hidden = BKE_subdiv_ccg_grid_hidden_ensure(subdiv_ccg);

  Array<bool> node_changed(node_mask.min_array_size(), false);

  bool any_changed = false;
  node_mask.foreach_index(
      [&](const int i) {
        const Span<int> grids = nodes[i].grids();
        BitGroupVector<> new_hide(grids.size(), grid_hidden.group_size());
        for (const int i : grids.index_range()) {
          new_hide[i].copy_from(grid_hidden[grids[i]].as_span());
        }

        for (const int i : grids.index_range()) {
          calc_hide(grids[i], new_hide[i]);
        }

        if (std::all_of(grids.index_range().begin(),
                        grids.index_range().end(),
                        [&](const int i) {
                          return bits::spans_equal(grid_hidden[grids[i]], new_hide[i]);
                        }))
        {
          return;
        }

        any_changed = true;
        undo::push_node(depsgraph, object, &nodes[i], undo::Type::HideVert);

        for (const int i : grids.index_range()) {
          grid_hidden[grids[i]].copy_from(new_hide[i].as_span());
        }

        node_changed[i] = true;
        bke::pbvh::node_update_visibility_grids(grid_hidden, nodes[i]);
      },
      exec_mode::grain_size(1));

  IndexMaskMemory memory;
  const IndexMask changed_nodes = IndexMask::from_bools(node_changed, memory);
  if (changed_nodes.is_empty()) {
    return;
  }
  pbvh.tag_visibility_changed(node_mask);
  multires_mark_as_modified(&depsgraph, &object, MULTIRES_HIDDEN_MODIFIED);
  sync_all_from_verts(object);
}

static void partialvis_update_bmesh_verts(const Set<BMVert *, 0> &verts,
                                          const VisAction action,
                                          const FunctionRef<bool(BMVert *v)> should_update,
                                          bool *any_changed,
                                          bool *any_visible)
{
  for (BMVert *v : verts) {
    if (should_update(v)) {
      if (action == VisAction::Hide) {
        BM_elem_flag_enable(v, BM_ELEM_HIDDEN);
      }
      else {
        BM_elem_flag_disable(v, BM_ELEM_HIDDEN);
      }
      (*any_changed) = true;
    }

    if (!BM_elem_flag_test(v, BM_ELEM_HIDDEN)) {
      (*any_visible) = true;
    }
  }
}

static void partialvis_update_bmesh_faces(const Set<BMFace *, 0> &faces)
{
  for (BMFace *f : faces) {
    if (BKE_paint_is_bmesh_face_hidden(f)) {
      BM_elem_flag_enable(f, BM_ELEM_HIDDEN);
    }
    else {
      BM_elem_flag_disable(f, BM_ELEM_HIDDEN);
    }
  }
}

static void partialvis_update_bmesh_nodes(const Depsgraph &depsgraph,
                                          Object &ob,
                                          const IndexMask &node_mask,
                                          const VisAction action,
                                          const FunctionRef<bool(BMVert *v)> vert_test_fn)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(ob);
  MutableSpan<bke::pbvh::BMeshNode> nodes = pbvh.nodes<bke::pbvh::BMeshNode>();

  node_mask.foreach_index([&](const int i) {
    bool any_changed = false;
    bool any_visible = false;

    undo::push_node(depsgraph, ob, &nodes[i], undo::Type::HideVert);

    partialvis_update_bmesh_verts(BKE_pbvh_bmesh_node_unique_verts(&nodes[i]),
                                  action,
                                  vert_test_fn,
                                  &any_changed,
                                  &any_visible);

    partialvis_update_bmesh_verts(BKE_pbvh_bmesh_node_other_verts(&nodes[i]),
                                  action,
                                  vert_test_fn,
                                  &any_changed,
                                  &any_visible);

    /* Finally loop over node faces and tag the ones that are fully hidden. */
    partialvis_update_bmesh_faces(BKE_pbvh_bmesh_node_faces(&nodes[i]));

    if (any_changed) {
      BKE_pbvh_node_fully_hidden_set(nodes[i], !any_visible);
    }
  });

  pbvh.tag_visibility_changed(node_mask);
  pbvh.update_visibility(ob);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Global Visibility Operators
 * Operators that act upon the entirety of a given object's mesh.
 * \{ */

static void partialvis_all_update_mesh(const Depsgraph &depsgraph,
                                       Object &object,
                                       const VisAction action,
                                       const IndexMask &node_mask)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  if (action == VisAction::Show && !attributes.contains(".hide_vert")) {
    /* If everything is already visible, don't do anything. */
    return;
  }

  switch (action) {
    case VisAction::Hide:
      vert_hide_update(
          depsgraph, object, node_mask, [&](const Span<int> /*verts*/, MutableSpan<bool> hide) {
            hide.fill(true);
          });
      break;
    case VisAction::Show:
      mesh_show_all(depsgraph, object, node_mask);
      break;
  }
}

static void partialvis_all_update_multires(Depsgraph &depsgraph,
                                           Object &object,
                                           const VisAction action,
                                           const IndexMask &node_mask)
{
  switch (action) {
    case VisAction::Hide:
      multires_hide_update(
          depsgraph, object, node_mask, [&](const IndexRange /*range*/, MutableSpan<bool> hidden) {
            hidden.fill(true);
          });
      break;
    case VisAction::Show:
      multires_show_all(depsgraph, object, node_mask);
      break;
  }
}

static void partialvis_all_update_grids(Depsgraph &depsgraph,
                                        Object &object,
                                        const VisAction action,
                                        const IndexMask &node_mask)
{
  switch (action) {
    case VisAction::Hide:
      grid_hide_update(depsgraph,
                       object,
                       node_mask,
                       [&](const int /*verts*/, MutableBoundedBitSpan hide) { hide.fill(true); });
      break;
    case VisAction::Show:
      grids_show_all(depsgraph, object, node_mask);
      break;
  }
}

static void partialvis_all_update_bmesh(const Depsgraph &depsgraph,
                                        Object &ob,
                                        const VisAction action,
                                        const IndexMask &node_mask)
{
  partialvis_update_bmesh_nodes(
      depsgraph, ob, node_mask, action, [](const BMVert * /*vert*/) { return true; });
}

static wmOperatorStatus hide_show_all_exec(bContext *C, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  Object &ob = *CTX_data_active_object(C);
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);

  const VisAction action = VisAction(RNA_enum_get(op->ptr, "action"));

  bke::pbvh::Tree &pbvh = bke::object::pbvh_ensure(*depsgraph, ob);

  /* Start undo. */
  switch (action) {
    case VisAction::Hide:
      undo::push_begin_ex(scene, ob, "Hide area");
      break;
    case VisAction::Show:
      undo::push_begin_ex(scene, ob, "Show area");
      break;
  }

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh:
      partialvis_all_update_mesh(*depsgraph, ob, action, node_mask);
      break;
    case bke::pbvh::Type::Multires:
      partialvis_all_update_multires(*depsgraph, ob, action, node_mask);
      break;
    case bke::pbvh::Type::Grids:
      partialvis_all_update_grids(*depsgraph, ob, action, node_mask);
      break;
    case bke::pbvh::Type::BMesh:
      partialvis_all_update_bmesh(*depsgraph, ob, action, node_mask);
      break;
  }

  /* End undo. */
  undo::push_end(ob);

  islands::invalidate(*ob.runtime->sculpt_session);
  tag_update_visibility(*C);

  return OPERATOR_FINISHED;
}

static void partialvis_masked_update_mesh(const Depsgraph &depsgraph,
                                          Object &object,
                                          const VisAction action,
                                          const IndexMask &node_mask)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  if (action == VisAction::Show && !attributes.contains(".hide_vert")) {
    /* If everything is already visible, don't do anything. */
    return;
  }

  const bool value = action_to_hide(action);
  const VArraySpan<float> mask = *attributes.lookup<float>(".sculpt_mask", bke::AttrDomain::Point);
  if (action == VisAction::Show && mask.is_empty()) {
    mesh_show_all(depsgraph, object, node_mask);
  }
  else if (!mask.is_empty()) {
    vert_hide_update(
        depsgraph, object, node_mask, [&](const Span<int> verts, MutableSpan<bool> hide) {
          for (const int i : verts.index_range()) {
            if (mask[verts[i]] > 0.5f) {
              hide[i] = value;
            }
          }
        });
  }
}

static void partialvis_masked_update_multires(Depsgraph &depsgraph,
                                              Object &object,
                                              const VisAction action,
                                              const IndexMask &node_mask)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  Multiresolution &multires = *mesh.multires;

  if (action == VisAction::Show && multires.hidden_verts().is_empty()) {
    return;
  }

  const bool value = action_to_hide(action);
  const Span<float> masks = multires.masks();

  if (action == VisAction::Show && masks.is_empty()) {
    multires_show_all(depsgraph, object, node_mask);
  }
  else if (!masks.is_empty()) {
    multires_hide_update(
        depsgraph, object, node_mask, [&](const IndexRange range, MutableSpan<bool> hidden) {
          for (const int i : range.index_range()) {
            if (masks[range[i]] > 0.5f) {
              hidden[i] = value;
            }
          }
        });
  }
}

static void partialvis_masked_update_grids(Depsgraph &depsgraph,
                                           Object &object,
                                           const VisAction action,
                                           const IndexMask &node_mask)
{
  SubdivCCG &subdiv_ccg = *object.runtime->sculpt_session->subdiv_ccg;

  const bool value = action_to_hide(action);
  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);
  const Span<float> masks = subdiv_ccg.masks;
  if (masks.is_empty()) {
    grid_hide_update(depsgraph,
                     object,
                     node_mask,
                     [&](const int /*verts*/, MutableBoundedBitSpan hide) { hide.fill(value); });
  }
  else {
    grid_hide_update(
        depsgraph, object, node_mask, [&](const int grid, MutableBoundedBitSpan hide) {
          const Span<float> grid_masks = masks.slice(bke::ccg::grid_range(key, grid));
          for (const int i : grid_masks.index_range()) {
            if (grid_masks[i] > 0.5f) {
              hide[i].set(value);
            }
          }
        });
  }
}

static void partialvis_masked_update_bmesh(const Depsgraph &depsgraph,
                                           Object &ob,
                                           const VisAction action,
                                           const IndexMask &node_mask)
{
  BMesh *bm = ob.runtime->sculpt_session->bm;
  const int mask_offset = CustomData_get_offset_named(&bm->vdata, CD_PROP_FLOAT, ".sculpt_mask");
  const auto mask_test_fn = [&](const BMVert *v) {
    const float vmask = BM_ELEM_CD_GET_FLOAT(v, mask_offset);
    return vmask > 0.5f;
  };

  partialvis_update_bmesh_nodes(depsgraph, ob, node_mask, action, mask_test_fn);
}

static wmOperatorStatus hide_show_masked_exec(bContext *C, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  Object &ob = *CTX_data_active_object(C);
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);

  const VisAction action = VisAction(RNA_enum_get(op->ptr, "action"));

  bke::pbvh::Tree &pbvh = bke::object::pbvh_ensure(*depsgraph, ob);

  /* Start undo. */
  switch (action) {
    case VisAction::Hide:
      undo::push_begin_ex(scene, ob, "Hide area");
      break;
    case VisAction::Show:
      undo::push_begin_ex(scene, ob, "Show area");
      break;
  }

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh:
      partialvis_masked_update_mesh(*depsgraph, ob, action, node_mask);
      break;
    case bke::pbvh::Type::Multires:
      partialvis_masked_update_multires(*depsgraph, ob, action, node_mask);
      break;
    case bke::pbvh::Type::Grids:
      partialvis_masked_update_grids(*depsgraph, ob, action, node_mask);
      break;
    case bke::pbvh::Type::BMesh:
      partialvis_masked_update_bmesh(*depsgraph, ob, action, node_mask);
      break;
  }

  /* End undo. */
  undo::push_end(ob);

  islands::invalidate(*ob.runtime->sculpt_session);
  tag_update_visibility(*C);

  return OPERATOR_FINISHED;
}

static void hide_show_operator_properties(wmOperatorType *ot)
{
  static const EnumPropertyItem action_items[] = {
      {int(VisAction::Hide), "HIDE", 0, "Hide", "Hide vertices"},
      {int(VisAction::Show), "SHOW", 0, "Show", "Show vertices"},
      {0, nullptr, 0, nullptr, nullptr},
  };

  RNA_def_enum(ot->srna,
               "action",
               action_items,
               int(VisAction::Hide),
               "Visibility Action",
               "Whether to hide or show vertices");
}

void PAINT_OT_hide_show_masked(wmOperatorType *ot)
{
  ot->name = "Hide/Show Masked";
  ot->idname = "PAINT_OT_hide_show_masked";
  ot->description = "Hide/show all masked vertices above a threshold";

  ot->exec = hide_show_masked_exec;
  /* Sculpt-only for now. */
  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER;

  hide_show_operator_properties(ot);
}

void PAINT_OT_hide_show_all(wmOperatorType *ot)
{
  ot->name = "Hide/Show All";
  ot->idname = "PAINT_OT_hide_show_all";
  ot->description = "Hide/show all vertices";

  ot->exec = hide_show_all_exec;
  /* Sculpt-only for now. */
  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER;

  hide_show_operator_properties(ot);
}

static void invert_visibility_mesh(const Depsgraph &depsgraph,
                                   Object &object,
                                   const IndexMask &node_mask)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();

  Mesh &mesh = *id_cast<Mesh *>(object.data);
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  bke::SpanAttributeWriter<bool> hide_poly = attributes.ensure_for_write_span<bool>(
      ".hide_poly", bke::AttrDomain::Face);

  undo::push_nodes(depsgraph, object, node_mask, undo::Type::HideFace);

  node_mask.foreach_index(
      [&](const int i) {
        for (const int face : nodes[i].faces()) {
          hide_poly.span[face] = !hide_poly.span[face];
        }
      },
      exec_mode::grain_size(1));

  hide_poly.finish();
  bke::mesh_hide_face_flush(mesh);
  pbvh.tag_visibility_changed(node_mask);
  pbvh.update_visibility(object);
}

static void invert_visibility_multires(Depsgraph &depsgraph,
                                       Object &object,
                                       const IndexMask &node_mask)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  const MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();

  const Mesh &base_mesh = *id_cast<Mesh *>(object.data);
  Multiresolution &multires = *base_mesh.multires;
  const int faces_per_corner = multires.faces_per_corner();
  const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(base_mesh);
  const OffsetIndices<int> base_faces = base_topo.faces;
  const MutableSpan<bool> hidden_faces = multires.hidden_faces_for_write();

  undo::push_nodes(depsgraph, object, node_mask, undo::Type::HideFace);

  node_mask.foreach_index(
      [&](const int i) {
        for (const int face_idx : nodes[i].faces()) {
          const IndexRange face_range = base_faces[face_idx];
          const int first_subface_idx = face_range.start() * faces_per_corner;
          const int subfaces_num = face_range.size() * faces_per_corner;

          for (const int j : IndexRange(subfaces_num)) {
            const int idx = first_subface_idx + j;
            hidden_faces[idx] = !hidden_faces[idx];
          }
        }
      },
      exec_mode::grain_size(1));

  multires.hide_face_flush();
  pbvh.tag_visibility_changed(node_mask);
  pbvh.update_visibility(object);
}

static void invert_visibility_grids(Depsgraph &depsgraph,
                                    Object &object,
                                    const IndexMask &node_mask)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();
  SubdivCCG &subdiv_ccg = *object.runtime->sculpt_session->subdiv_ccg;

  undo::push_nodes(depsgraph, object, node_mask, undo::Type::HideVert);

  BitGroupVector<> &grid_hidden = BKE_subdiv_ccg_grid_hidden_ensure(subdiv_ccg);
  node_mask.foreach_index(
      [&](const int i) {
        for (const int i : nodes[i].grids()) {
          bits::invert(grid_hidden[i]);
        }
        bke::pbvh::node_update_visibility_grids(grid_hidden, nodes[i]);
      },
      exec_mode::grain_size(1));

  pbvh.tag_visibility_changed(node_mask);
  multires_mark_as_modified(&depsgraph, &object, MULTIRES_HIDDEN_MODIFIED);
  sync_all_from_verts(object);
}

static void invert_visibility_bmesh(const Depsgraph &depsgraph,
                                    Object &object,
                                    const IndexMask &node_mask)
{
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::BMeshNode> nodes = pbvh.nodes<bke::pbvh::BMeshNode>();
  undo::push_nodes(depsgraph, object, node_mask, undo::Type::HideVert);

  node_mask.foreach_index(
      [&](const int i) {
        bool fully_hidden = true;
        for (BMVert *vert : BKE_pbvh_bmesh_node_unique_verts(&nodes[i])) {
          BM_elem_flag_toggle(vert, BM_ELEM_HIDDEN);
          fully_hidden &= BM_elem_flag_test_bool(vert, BM_ELEM_HIDDEN);
        }
        BKE_pbvh_node_fully_hidden_set(nodes[i], fully_hidden);
      },
      exec_mode::grain_size(1));
  node_mask.foreach_index(
      [&](const int i) { partialvis_update_bmesh_faces(BKE_pbvh_bmesh_node_faces(&nodes[i])); },
      exec_mode::grain_size(1));
  pbvh.tag_visibility_changed(node_mask);
}

static wmOperatorStatus visibility_invert_exec(bContext *C, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  Object &object = *CTX_data_active_object(C);
  Depsgraph &depsgraph = *CTX_data_ensure_evaluated_depsgraph(C);

  bke::pbvh::Tree &pbvh = bke::object::pbvh_ensure(depsgraph, object);

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);
  undo::push_begin(scene, object, op);
  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh:
      invert_visibility_mesh(depsgraph, object, node_mask);
      break;
    case bke::pbvh::Type::Multires:
      invert_visibility_multires(depsgraph, object, node_mask);
      break;
    case bke::pbvh::Type::Grids:
      invert_visibility_grids(depsgraph, object, node_mask);
      break;
    case bke::pbvh::Type::BMesh:
      invert_visibility_bmesh(depsgraph, object, node_mask);
      break;
  }

  undo::push_end(object);

  islands::invalidate(*object.runtime->sculpt_session);
  tag_update_visibility(*C);

  return OPERATOR_FINISHED;
}

void PAINT_OT_visibility_invert(wmOperatorType *ot)
{
  ot->name = "Invert Visibility";
  ot->idname = "PAINT_OT_visibility_invert";
  ot->description = "Invert the visibility of all vertices";

  ot->exec = visibility_invert_exec;
  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER;
}

/* Number of vertices per iteration step size when growing or shrinking visibility. */
static constexpr float VERTEX_ITERATION_THRESHOLD = 50000.0f;

/* Extracting the loop and comparing against / writing with a constant `false` or `true` instead of
 * using #action_to_hide results in a nearly 600ms speedup on a mesh with 1.5m verts. */
template<bool value>
static void affect_visibility_mesh(const IndexRange face,
                                   const Span<int> corner_verts,
                                   const Span<bool> read_buffer,
                                   MutableSpan<bool> write_buffer)
{
  for (const int corner : face) {
    int vert = corner_verts[corner];
    if (read_buffer[vert] != value) {
      continue;
    }

    const int prev = bke::mesh::face_corner_prev(face, corner);
    const int prev_vert = corner_verts[prev];
    write_buffer[prev_vert] = value;

    const int next = bke::mesh::face_corner_next(face, corner);
    const int next_vert = corner_verts[next];
    write_buffer[next_vert] = value;
  }
}

struct DualBuffer {
  Array<bool> front;
  Array<bool> back;

  MutableSpan<bool> write_buffer(int count)
  {
    return count % 2 == 0 ? back.as_mutable_span() : front.as_mutable_span();
  }

  Span<bool> read_buffer(int count)
  {
    return count % 2 == 0 ? front.as_span() : back.as_span();
  }
};

static void propagate_vertex_visibility(Mesh &mesh,
                                        DualBuffer &buffers,
                                        const VArraySpan<bool> &hide_poly,
                                        const VisAction action,
                                        const int iterations)
{
  const OffsetIndices faces = mesh.faces();
  const Span<int> corner_verts = mesh.corner_verts();

  for (const int i : IndexRange(iterations)) {
    Span<bool> read_buffer = buffers.read_buffer(i);
    MutableSpan<bool> write_buffer = buffers.write_buffer(i);
    threading::parallel_for(faces.index_range(), 1024, [&](const IndexRange range) {
      for (const int face_index : range) {
        if (!hide_poly[face_index]) {
          continue;
        }
        const IndexRange face = faces[face_index];
        if (action == VisAction::Hide) {
          affect_visibility_mesh<true>(face, corner_verts, read_buffer, write_buffer);
        }
        else {
          affect_visibility_mesh<false>(face, corner_verts, read_buffer, write_buffer);
        }
      }
    });

    flush_face_changes(mesh, write_buffer);
  }
}

static void update_undo_state(const Depsgraph &depsgraph,
                              Object &object,
                              const IndexMask &node_mask,
                              const Span<bool> old_hide_vert,
                              const Span<bool> new_hide_vert)
{
  const bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  const Span<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();

  node_mask.foreach_index(
      [&](const int i) {
        for (const int vert : nodes[i].verts()) {
          if (old_hide_vert[vert] != new_hide_vert[vert]) {
            undo::push_node(depsgraph, object, &nodes[i], undo::Type::HideVert);
            break;
          }
        }
      },
      exec_mode::grain_size(1));
}

static void update_node_visibility_from_face_changes(bke::pbvh::Tree &pbvh,
                                                     const IndexMask &node_mask,
                                                     const Span<bool> orig_hide_poly,
                                                     const Span<bool> new_hide_poly,
                                                     const Span<bool> hide_vert)
{
  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
  Array<bool> node_changed(node_mask.min_array_size(), false);

  node_mask.foreach_index(
      [&](const int i) {
        bool any_changed = false;
        const Span<int> indices = nodes[i].faces();
        for (const int face_index : indices) {
          if (orig_hide_poly[face_index] != new_hide_poly[face_index]) {
            any_changed = true;
            break;
          }
        }

        if (any_changed) {
          node_changed[i] = true;
          bke::pbvh::node_update_visibility_mesh(hide_vert, nodes[i]);
        }
      },
      exec_mode::grain_size(1));

  IndexMaskMemory memory;
  const IndexMask changed_nodes = IndexMask::from_bools(node_changed, memory);
  if (pbvh.draw_data) {
    /* Only tag draw data. Nodes have already been updated above. */
    pbvh.draw_data->tag_visibility_changed(changed_nodes);
  }
}

static void grow_shrink_visibility_mesh(const Depsgraph &depsgraph,
                                        Object &object,
                                        const IndexMask &node_mask,
                                        const VisAction action,
                                        const int iterations)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  if (!attributes.contains(".hide_vert")) {
    /* If the entire mesh is visible, we can neither grow nor shrink the boundary. */
    return;
  }

  bke::SpanAttributeWriter<bool> hide_vert = attributes.ensure_for_write_span<bool>(
      ".hide_vert", bke::AttrDomain::Point);
  const VArraySpan hide_poly = *attributes.lookup_or_default<bool>(
      ".hide_poly", bke::AttrDomain::Face, false);

  DualBuffer buffers;
  buffers.back.reinitialize(hide_vert.span.size());
  buffers.front.reinitialize(hide_vert.span.size());
  array_utils::copy(hide_vert.span.as_span(), buffers.back.as_mutable_span());
  array_utils::copy(hide_vert.span.as_span(), buffers.front.as_mutable_span());

  Array<bool> orig_hide_poly(hide_poly);
  propagate_vertex_visibility(mesh, buffers, hide_poly, action, iterations);

  const Span<bool> last_buffer = buffers.write_buffer(iterations - 1);

  update_undo_state(depsgraph, object, node_mask, hide_vert.span, last_buffer);

  /* We can wait until after all iterations are done to flush edge changes as they are
   * not used for coarse filtering while iterating. */
  flush_edge_changes(mesh, last_buffer);

  update_node_visibility_from_face_changes(
      *bke::object::pbvh_get(object), node_mask, orig_hide_poly, hide_poly, last_buffer);
  array_utils::copy(last_buffer, hide_vert.span);
  hide_vert.finish();
}

template<bool value>
static void affect_visibility_quad(const int v0,
                                   const int v1,
                                   const int v2,
                                   const int v3,
                                   const Span<bool> read_buffer,
                                   MutableSpan<bool> write_buffer)
{
  if (read_buffer[v0] == value) {
    write_buffer[v3] = value;
    write_buffer[v1] = value;
  }
  if (read_buffer[v1] == value) {
    write_buffer[v0] = value;
    write_buffer[v2] = value;
  }
  if (read_buffer[v2] == value) {
    write_buffer[v1] = value;
    write_buffer[v3] = value;
  }
  if (read_buffer[v3] == value) {
    write_buffer[v2] = value;
    write_buffer[v0] = value;
  }
}

static void flush_face_changes_multires(const Span<int> corner_verts,
                                        const Span<bool> hide_vert,
                                        MutableSpan<bool> hide_face)
{
  threading::parallel_for(hide_face.index_range(), 4096, [&](const IndexRange range) {
    for (const int subface_idx : range) {
      const int v0 = corner_verts[subface_idx * 4 + 0];
      const int v1 = corner_verts[subface_idx * 4 + 1];
      const int v2 = corner_verts[subface_idx * 4 + 2];
      const int v3 = corner_verts[subface_idx * 4 + 3];

      hide_face[subface_idx] = hide_vert[v0] || hide_vert[v1] || hide_vert[v2] || hide_vert[v3];
    }
  });
}

static void propagate_vertex_visibility_multires(Multiresolution &multires,
                                                 DualBuffer &buffers,
                                                 const VisAction action,
                                                 const int iterations)
{
  const Span<int> corner_verts = multires.corner_verts();
  const MutableSpan<bool> hidden_faces = multires.hidden_faces_for_write();

  for (const int i : IndexRange(iterations)) {
    Span<bool> read_buffer = buffers.read_buffer(i);
    MutableSpan<bool> write_buffer = buffers.write_buffer(i);
    write_buffer.copy_from(read_buffer);

    threading::parallel_for(hidden_faces.index_range(), 4096, [&](const IndexRange range) {
      for (const int subface_idx : range) {
        if (!hidden_faces[subface_idx]) {
          continue;
        }

        const int v0 = corner_verts[subface_idx * 4 + 0];
        const int v1 = corner_verts[subface_idx * 4 + 1];
        const int v2 = corner_verts[subface_idx * 4 + 2];
        const int v3 = corner_verts[subface_idx * 4 + 3];

        if (action == VisAction::Hide) {
          affect_visibility_quad<true>(v0, v1, v2, v3, read_buffer, write_buffer);
        }
        else {
          affect_visibility_quad<false>(v0, v1, v2, v3, read_buffer, write_buffer);
        }
      }
    });

    flush_face_changes_multires(corner_verts, write_buffer, hidden_faces);
  }
}

static void grow_shrink_visibility_multires(const Depsgraph &depsgraph,
                                            Object &object,
                                            const IndexMask &node_mask,
                                            const VisAction action,
                                            const int iterations)
{
  const Mesh &base_mesh = *id_cast<const Mesh *>(object.data);
  Multiresolution &multires = *base_mesh.multires;

  if (multires.hidden_verts().is_empty()) {
    return;
  }

  const MutableSpan<bool> hidden_verts = multires.hidden_verts_for_write();
  const MutableSpan<bool> hidden_faces = multires.ensure_hidden_faces();
  const Span<int> corner_verts = multires.corner_verts();

  DualBuffer buffers;
  buffers.back.reinitialize(hidden_verts.size());
  buffers.front.reinitialize(hidden_verts.size());
  buffers.back.as_mutable_span().copy_from(hidden_verts);
  buffers.front.as_mutable_span().copy_from(hidden_verts);

  Array<bool> orig_hidden_verts(hidden_verts.size());
  orig_hidden_verts.as_mutable_span().copy_from(hidden_verts);

  flush_face_changes_multires(corner_verts, hidden_verts, hidden_faces);

  Array<bool> orig_hidden_faces(hidden_faces.size());
  orig_hidden_faces.as_mutable_span().copy_from(hidden_faces);

  propagate_vertex_visibility_multires(multires, buffers, action, iterations);

  const Span<bool> last_buffer = buffers.write_buffer(iterations - 1);

  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
  const int grid_size = multires.grid_size();
  const int faces_per_corner = multires.faces_per_corner();
  const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(base_mesh);
  const OffsetIndices<int> base_faces = base_topo.faces;

  Array<bool> undo_nodes(node_mask.min_array_size(), false);
  Array<bool> vis_nodes(node_mask.min_array_size(), false);

  node_mask.foreach_index(
      [&](const int i) {
        const IndexRange node_range = nodes[i].range(grid_size);
        const Span<bool> old_verts = orig_hidden_verts.as_span().slice(node_range);
        const Span<bool> new_verts = last_buffer.slice(node_range);
        const bool vert_changed = !std::equal(
            old_verts.begin(), old_verts.end(), new_verts.begin());

        if (vert_changed) {
          undo_nodes[i] = true;
        }

        bool face_changed = false;
        for (const int face_idx : nodes[i].faces()) {
          const IndexRange face_range = base_faces[face_idx];
          const int first_subface_idx = face_range.start() * faces_per_corner;
          const int subfaces_num = face_range.size() * faces_per_corner;
          const Span<bool> orig_sub = orig_hidden_faces.as_span().slice(first_subface_idx,
                                                                        subfaces_num);
          const Span<bool> new_sub = hidden_faces.slice(first_subface_idx, subfaces_num);

          if (!std::equal(orig_sub.begin(), orig_sub.end(), new_sub.begin())) {
            face_changed = true;
            break;
          }
        }

        if (vert_changed || face_changed) {
          vis_nodes[i] = true;
        }
      },
      exec_mode::grain_size(1));

  IndexMaskMemory memory;
  const IndexMask undo_mask = IndexMask::from_bools(undo_nodes, memory);
  if (undo_mask.is_empty()) {
    return;
  }

  undo::push_nodes(depsgraph, object, undo_mask, undo::Type::HideVert);
  hidden_verts.copy_from(last_buffer);

  const IndexMask vis_mask = IndexMask::from_bools(vis_nodes, memory);
  vis_mask.foreach_index(
      [&](const int i) {
        bke::pbvh::node_update_visibility_multires(hidden_verts, grid_size, nodes[i]);
      },
      exec_mode::grain_size(1));

  pbvh.tag_visibility_changed(vis_mask);
  pbvh.update_visibility(object);
}

struct DualBitBuffer {
  BitGroupVector<> front;
  BitGroupVector<> back;

  BitGroupVector<> &write_buffer(int count)
  {
    return count % 2 == 0 ? back : front;
  }

  BitGroupVector<> &read_buffer(int count)
  {
    return count % 2 == 0 ? front : back;
  }
};

static void grow_shrink_visibility_grid(Depsgraph &depsgraph,
                                        Object &object,
                                        const IndexMask &node_mask,
                                        const VisAction action,
                                        const int iterations)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::GridsNode> nodes = pbvh.nodes<bke::pbvh::GridsNode>();

  SubdivCCG &subdiv_ccg = *ss.subdiv_ccg;

  BitGroupVector<> &grid_hidden = BKE_subdiv_ccg_grid_hidden_ensure(subdiv_ccg);

  const bool desired_state = action_to_hide(action);
  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);

  DualBitBuffer buffers;
  buffers.front = grid_hidden;
  buffers.back = grid_hidden;

  Array<bool> node_changed(nodes.size(), false);

  for (const int i : IndexRange(iterations)) {
    BitGroupVector<> &read_buffer = buffers.read_buffer(i);
    BitGroupVector<> &write_buffer = buffers.write_buffer(i);

    node_mask.foreach_index(
        [&](const int i) {
          for (const int grid : nodes[i].grids()) {
            for (const int y : IndexRange(key.grid_size)) {
              for (const int x : IndexRange(key.grid_size)) {
                const int grid_elem_idx = CCG_grid_xy_to_index(key.grid_size, x, y);
                if (read_buffer[grid][grid_elem_idx] != desired_state) {
                  continue;
                }

                SubdivCCGCoord coord{};
                coord.grid_index = grid;
                coord.x = x;
                coord.y = y;

                SubdivCCGNeighbors neighbors;
                BKE_subdiv_ccg_neighbor_coords_get(subdiv_ccg, coord, true, neighbors);

                for (const SubdivCCGCoord neighbor : neighbors.coords) {
                  const int neighbor_grid_elem_idx = CCG_grid_xy_to_index(
                      key.grid_size, neighbor.x, neighbor.y);

                  write_buffer[neighbor.grid_index][neighbor_grid_elem_idx].set(desired_state);
                }
              }
            }
          }

          node_changed[i] = true;
        },
        exec_mode::grain_size(1));
  }

  IndexMaskMemory memory;
  const IndexMask changed_nodes = IndexMask::from_bools(node_changed, memory);

  undo::push_nodes(depsgraph, object, changed_nodes, undo::Type::HideVert);

  BitGroupVector<> &last_buffer = buffers.write_buffer(iterations - 1);
  grid_hidden = std::move(last_buffer);

  pbvh.tag_visibility_changed(node_mask);
  pbvh.update_visibility(object);

  multires_mark_as_modified(&depsgraph, &object, MULTIRES_HIDDEN_MODIFIED);
  sync_all_from_verts(object);
}

static Array<bool> duplicate_visibility_bmesh(const Object &object)
{
  const SculptSession &ss = *object.runtime->sculpt_session;
  BMesh &bm = *ss.bm;
  Array<bool> result(bm.totvert);
  vert_random_access_ensure(const_cast<Object &>(object));
  for (const int i : result.index_range()) {
    result[i] = BM_elem_flag_test_bool(BM_vert_at_index(&bm, i), BM_ELEM_HIDDEN);
  }
  return result;
}

static void grow_shrink_visibility_bmesh(const Depsgraph &depsgraph,
                                         Object &object,
                                         const IndexMask &node_mask,
                                         const VisAction action,
                                         const int iterations)
{
  for (const int i : IndexRange(iterations)) {
    UNUSED_VARS(i);
    const Array<bool> prev_visibility = duplicate_visibility_bmesh(object);
    partialvis_update_bmesh_nodes(depsgraph, object, node_mask, action, [&](BMVert *vert) {
      BMeshNeighborVerts neighbors;
      for (BMVert *neighbor : vert_neighbors_get_bmesh(*vert, neighbors)) {
        if (prev_visibility[BM_elem_index_get(neighbor)] == action_to_hide(action)) {
          return true;
        }
      }
      return false;
    });
  }
}

static wmOperatorStatus visibility_filter_exec(bContext *C, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  Object &object = *CTX_data_active_object(C);
  Depsgraph &depsgraph = *CTX_data_ensure_evaluated_depsgraph(C);

  bke::pbvh::Tree &pbvh = bke::object::pbvh_ensure(depsgraph, object);

  const VisAction mode = VisAction(RNA_enum_get(op->ptr, "action"));

  IndexMaskMemory memory;
  const IndexMask node_mask = bke::pbvh::all_leaf_nodes(pbvh, memory);

  int num_verts = vertex_count_get(object);

  int iterations = RNA_int_get(op->ptr, "iterations");

  if (RNA_boolean_get(op->ptr, "auto_iteration_count")) {
    /* Automatically adjust the number of iterations based on the number
     * of vertices in the mesh. */
    iterations = int(num_verts / VERTEX_ITERATION_THRESHOLD) + 1;
  }

  undo::push_begin(scene, object, op);
  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh:
      grow_shrink_visibility_mesh(depsgraph, object, node_mask, mode, iterations);
      break;
    case bke::pbvh::Type::Multires:
      grow_shrink_visibility_multires(depsgraph, object, node_mask, mode, iterations);
      break;
    case bke::pbvh::Type::Grids:
      grow_shrink_visibility_grid(depsgraph, object, node_mask, mode, iterations);
      break;
    case bke::pbvh::Type::BMesh:
      grow_shrink_visibility_bmesh(depsgraph, object, node_mask, mode, iterations);
      break;
  }
  undo::push_end(object);

  islands::invalidate(*object.runtime->sculpt_session);
  tag_update_visibility(*C);

  return OPERATOR_FINISHED;
}

void PAINT_OT_visibility_filter(wmOperatorType *ot)
{
  static EnumPropertyItem actions[] = {
      {int(VisAction::Show),
       "GROW",
       0,
       "Grow Visibility",
       "Grow the visibility by one face based on mesh topology"},
      {int(VisAction::Hide),
       "SHRINK",
       0,
       "Shrink Visibility",
       "Shrink the visibility by one face based on mesh topology"},
      {0, nullptr, 0, nullptr, nullptr},
  };

  ot->name = "Visibility Filter";
  ot->idname = "PAINT_OT_visibility_filter";
  ot->description = "Edit the visibility of the current mesh";

  ot->exec = visibility_filter_exec;
  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  RNA_def_enum(ot->srna, "action", actions, int(VisAction::Show), "Action", "");

  RNA_def_int(ot->srna,
              "iterations",
              1,
              1,
              100,
              "Iterations",
              "Number of times that the filter is going to be applied",
              1,
              100);
  RNA_def_boolean(
      ot->srna,
      "auto_iteration_count",
      true,
      "Auto Iteration Count",
      "Use an automatic number of iterations based on the number of vertices of the sculpt");
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Gesture-based Visibility Operators
 * Operators that act upon a user-selected area.
 * \{ */

struct HideShowOperation {
  gesture::Operation op;

  VisAction action;
};

static void partialvis_gesture_update_mesh(gesture::GestureData &gesture_data)
{
  HideShowOperation *operation = reinterpret_cast<HideShowOperation *>(gesture_data.operation);
  Object *object = gesture_data.vc.obact;
  const Depsgraph &depsgraph = *gesture_data.vc.depsgraph;
  const VisAction action = operation->action;
  const IndexMask &node_mask = gesture_data.node_mask;

  Mesh *mesh = id_cast<Mesh *>(object->data);
  bke::MutableAttributeAccessor attributes = mesh->attributes_for_write();
  if (action == VisAction::Show && !attributes.contains(".hide_vert")) {
    /* If everything is already visible, don't do anything. */
    return;
  }

  const View3D *v3d = gesture_data.vc.v3d;
  RegionView3D *rv3d = gesture_data.vc.rv3d;
  const bool clipping_enabled = RV3D_CLIPPING_ENABLED(v3d, rv3d);
  if (clipping_enabled) {
    ED_view3d_clipping_local(rv3d, object->object_to_world().ptr());
  }
  const bool value = action_to_hide(action);
  const Span<float3> positions = bke::pbvh::vert_positions_eval(depsgraph, *object);
  const Span<float3> normals = bke::pbvh::vert_normals_eval(depsgraph, *object);
  vert_hide_update(
      depsgraph, *object, node_mask, [&](const Span<int> verts, MutableSpan<bool> hide) {
        for (const int i : verts.index_range()) {
          if (clipping_enabled && ED_view3d_clipping_test(rv3d, positions[verts[i]], true)) {
            continue;
          }
          if (gesture::is_affected(gesture_data, positions[verts[i]], normals[verts[i]])) {
            hide[i] = value;
          }
        }
      });
}

static void partialvis_gesture_update_multires(Depsgraph &depsgraph,
                                               gesture::GestureData &gesture_data)
{
  HideShowOperation *operation = reinterpret_cast<HideShowOperation *>(gesture_data.operation);
  Object *object = gesture_data.vc.obact;
  const VisAction action = operation->action;
  const IndexMask &node_mask = gesture_data.node_mask;

  const Mesh &base_mesh = *id_cast<Mesh *>(object->data);
  Multiresolution &multires = *base_mesh.multires;

  const bool value = action_to_hide(action);
  multires_hide_update(
      depsgraph, *object, node_mask, [&](const IndexRange range, MutableSpan<bool> hidden) {
        const Span<float3> positions = multires.positions().slice(range);
        const Span<float3> normals = multires.normals().slice(range);
        for (const int i : positions.index_range()) {
          if (gesture::is_affected(gesture_data, positions[i], normals[i])) {
            hidden[i] = value;
          }
        }
      });
}

static void partialvis_gesture_update_grids(Depsgraph &depsgraph,
                                            gesture::GestureData &gesture_data)
{
  HideShowOperation *operation = reinterpret_cast<HideShowOperation *>(gesture_data.operation);
  Object *object = gesture_data.vc.obact;
  const VisAction action = operation->action;
  const IndexMask &node_mask = gesture_data.node_mask;

  SubdivCCG &subdiv_ccg = *object->runtime->sculpt_session->subdiv_ccg;

  const View3D *v3d = gesture_data.vc.v3d;
  RegionView3D *rv3d = gesture_data.vc.rv3d;
  const bool clipping_enabled = RV3D_CLIPPING_ENABLED(v3d, rv3d);
  if (clipping_enabled) {
    ED_view3d_clipping_local(rv3d, object->object_to_world().ptr());
  }

  const bool value = action_to_hide(action);
  const CCGKey key = BKE_subdiv_ccg_key_top_level(subdiv_ccg);
  const Span<float3> positions = subdiv_ccg.positions;
  const Span<float3> normals = subdiv_ccg.normals;
  grid_hide_update(depsgraph, *object, node_mask, [&](const int grid, MutableBoundedBitSpan hide) {
    const Span<float3> grid_positions = positions.slice(bke::ccg::grid_range(key, grid));
    const Span<float3> grid_normals = normals.slice(bke::ccg::grid_range(key, grid));
    for (const int i : grid_positions.index_range()) {
      if (clipping_enabled && ED_view3d_clipping_test(rv3d, grid_positions[i], true)) {
        continue;
      }
      if (gesture::is_affected(gesture_data, grid_positions[i], grid_normals[i])) {
        hide[i].set(value);
      }
    }
  });
}

static void partialvis_gesture_update_bmesh(gesture::GestureData &gesture_data)
{
  const View3D *v3d = gesture_data.vc.v3d;
  RegionView3D *rv3d = gesture_data.vc.rv3d;
  Object *object = gesture_data.vc.obact;

  const bool clipping_enabled = RV3D_CLIPPING_ENABLED(v3d, rv3d);
  if (clipping_enabled) {
    ED_view3d_clipping_local(rv3d, object->object_to_world().ptr());
  }

  const auto selection_test_fn = [&](const BMVert *v) {
    if (clipping_enabled && ED_view3d_clipping_test(rv3d, v->co, true)) {
      return false;
    }
    return gesture::is_affected(gesture_data, v->co, v->no);
  };

  HideShowOperation *operation = reinterpret_cast<HideShowOperation *>(gesture_data.operation);

  partialvis_update_bmesh_nodes(*gesture_data.vc.depsgraph,
                                *gesture_data.vc.obact,
                                gesture_data.node_mask,
                                operation->action,
                                selection_test_fn);
}

static void hide_show_begin(bContext &C, wmOperator &op, gesture::GestureData & /*gesture_data*/)
{
  const Scene &scene = *CTX_data_scene(&C);
  Object *ob = CTX_data_active_object(&C);
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(&C);

  undo::push_begin(scene, *ob, &op);
  bke::object::pbvh_ensure(*depsgraph, *ob);
}

static void hide_show_apply_for_symmetry_pass(bContext &C, gesture::GestureData &gesture_data)
{
  Depsgraph *depsgraph = CTX_data_depsgraph_pointer(&C);

  switch (bke::object::pbvh_get(*gesture_data.vc.obact)->type()) {
    case bke::pbvh::Type::Mesh:
      partialvis_gesture_update_mesh(gesture_data);
      break;
    case bke::pbvh::Type::Multires:
      partialvis_gesture_update_multires(*depsgraph, gesture_data);
      break;
    case bke::pbvh::Type::Grids:
      partialvis_gesture_update_grids(*depsgraph, gesture_data);
      break;
    case bke::pbvh::Type::BMesh:
      partialvis_gesture_update_bmesh(gesture_data);
      break;
  }
}
static void hide_show_end(bContext &C, gesture::GestureData &gesture_data)
{
  islands::invalidate(*gesture_data.vc.obact->runtime->sculpt_session);
  tag_update_visibility(C);
  undo::push_end(*gesture_data.vc.obact);
}

static void hide_show_init_properties(bContext & /*C*/,
                                      gesture::GestureData &gesture_data,
                                      wmOperator &op)
{
  gesture_data.operation = reinterpret_cast<gesture::Operation *>(
      MEM_new_zeroed<HideShowOperation>(__func__));

  HideShowOperation *operation = reinterpret_cast<HideShowOperation *>(gesture_data.operation);

  operation->op.begin = hide_show_begin;
  operation->op.apply_for_symmetry_pass = hide_show_apply_for_symmetry_pass;
  operation->op.end = hide_show_end;

  operation->action = VisAction(RNA_enum_get(op.ptr, "action"));
  gesture_data.selection_type = gesture::SelectionType(RNA_enum_get(op.ptr, "area"));
}

static wmOperatorStatus hide_show_gesture_box_exec(bContext *C, wmOperator *op)
{
  std::unique_ptr<gesture::GestureData> gesture_data = gesture::init_from_box(C, op);
  if (!gesture_data) {
    return OPERATOR_CANCELLED;
  }
  hide_show_init_properties(*C, *gesture_data, *op);
  gesture::apply(*C, *gesture_data, *op);
  return OPERATOR_FINISHED;
}

static wmOperatorStatus hide_show_gesture_lasso_exec(bContext *C, wmOperator *op)
{
  std::unique_ptr<gesture::GestureData> gesture_data = gesture::init_from_lasso(C, op);
  if (!gesture_data) {
    return OPERATOR_CANCELLED;
  }
  hide_show_init_properties(*C, *gesture_data, *op);
  gesture::apply(*C, *gesture_data, *op);
  return OPERATOR_FINISHED;
}

static wmOperatorStatus hide_show_gesture_line_exec(bContext *C, wmOperator *op)
{
  std::unique_ptr<gesture::GestureData> gesture_data = gesture::init_from_line(C, op);
  if (!gesture_data) {
    return OPERATOR_CANCELLED;
  }
  hide_show_init_properties(*C, *gesture_data, *op);
  gesture::apply(*C, *gesture_data, *op);
  return OPERATOR_FINISHED;
}

static wmOperatorStatus hide_show_gesture_polyline_exec(bContext *C, wmOperator *op)
{
  std::unique_ptr<gesture::GestureData> gesture_data = gesture::init_from_polyline(C, op);
  if (!gesture_data) {
    return OPERATOR_CANCELLED;
  }
  hide_show_init_properties(*C, *gesture_data, *op);
  gesture::apply(*C, *gesture_data, *op);
  return OPERATOR_FINISHED;
}

static void hide_show_operator_gesture_properties(wmOperatorType *ot)
{
  static const EnumPropertyItem area_items[] = {
      {int(gesture::SelectionType::Outside),
       "OUTSIDE",
       0,
       "Outside",
       "Hide or show vertices outside the selection"},
      {int(gesture::SelectionType::Inside),
       "Inside",
       0,
       "Inside",
       "Hide or show vertices inside the selection"},
      {0, nullptr, 0, nullptr, nullptr},
  };

  RNA_def_enum(ot->srna,
               "area",
               area_items,
               int(gesture::SelectionType::Inside),
               "Visibility Area",
               "Which vertices to hide or show");
}

void PAINT_OT_hide_show(wmOperatorType *ot)
{
  ot->name = "Hide/Show";
  ot->idname = "PAINT_OT_hide_show";
  ot->description = "Hide/show some vertices";

  ot->invoke = WM_gesture_box_invoke;
  ot->modal = WM_gesture_box_modal;
  ot->exec = hide_show_gesture_box_exec;
  /* Sculpt-only for now. */
  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER;

  WM_operator_properties_border(ot);
  hide_show_operator_properties(ot);
  hide_show_operator_gesture_properties(ot);
  gesture::operator_properties(ot, gesture::ShapeType::Box);
}

void PAINT_OT_hide_show_lasso_gesture(wmOperatorType *ot)
{
  ot->name = "Hide/Show Lasso";
  ot->idname = "PAINT_OT_hide_show_lasso_gesture";
  ot->description = "Hide/show some vertices";

  ot->invoke = WM_gesture_lasso_invoke;
  ot->modal = WM_gesture_lasso_modal;
  ot->exec = hide_show_gesture_lasso_exec;
  /* Sculpt-only for now. */
  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER | OPTYPE_DEPENDS_ON_CURSOR;

  WM_operator_properties_gesture_lasso(ot);
  hide_show_operator_properties(ot);
  hide_show_operator_gesture_properties(ot);
  gesture::operator_properties(ot, gesture::ShapeType::Lasso);
}

void PAINT_OT_hide_show_line_gesture(wmOperatorType *ot)
{
  ot->name = "Hide/Show Line";
  ot->idname = "PAINT_OT_hide_show_line_gesture";
  ot->description = "Hide/show some vertices";

  ot->invoke = WM_gesture_straightline_active_side_invoke;
  ot->modal = WM_gesture_straightline_oneshot_modal;
  ot->exec = hide_show_gesture_line_exec;
  /* Sculpt-only for now. */
  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER;

  WM_operator_properties_gesture_straightline(ot, WM_CURSOR_EDIT);
  hide_show_operator_properties(ot);
  hide_show_operator_gesture_properties(ot);
  gesture::operator_properties(ot, gesture::ShapeType::Line);
}

void PAINT_OT_hide_show_polyline_gesture(wmOperatorType *ot)
{
  ot->name = "Hide/Show Polyline";
  ot->idname = "PAINT_OT_hide_show_polyline_gesture";
  ot->description = "Hide/show some vertices";

  ot->invoke = WM_gesture_polyline_invoke;
  ot->modal = WM_gesture_polyline_modal;
  ot->exec = hide_show_gesture_polyline_exec;
  /* Sculpt-only for now. */
  ot->poll = sculpt_mode_poll_view3d;

  ot->flag = OPTYPE_REGISTER | OPTYPE_DEPENDS_ON_CURSOR;

  WM_operator_properties_gesture_polyline(ot);
  hide_show_operator_properties(ot);
  hide_show_operator_gesture_properties(ot);
  gesture::operator_properties(ot, gesture::ShapeType::Lasso);
}

/** \} */

}  // namespace blender::ed::sculpt_paint::hide
