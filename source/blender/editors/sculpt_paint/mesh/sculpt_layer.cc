/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BLI_array.hh"
#include "BLI_task.hh"

#include "BKE_context.hh"
#include "BKE_mesh.hh"
#include "BKE_multiresolution.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_sculpt_layer.hh"

#include "DNA_object_types.h"

#include "DEG_depsgraph.hh"

#include "WM_api.hh"
#include "WM_message.hh"
#include "WM_toolsystem.hh"
#include "WM_types.hh"

#include "sculpt_intern.hh"
#include "sculpt_storage_undo.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

namespace blender::ed::sculpt_paint::layer {

static bool poll(bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  return ob != nullptr && ob->type == OB_MESH && BKE_mesh_from_object(ob) != nullptr;
}

static bool active_poll(bContext *C)
{
  if (!poll(C)) {
    return false;
  }
  const Mesh *mesh = BKE_mesh_from_object(CTX_data_active_object(C));
  return mesh->active_layer_index >= 0 && mesh->active_layer_index < mesh->sculpt_layers_len;
}

static bool is_sculpt(const Object &ob)
{
  return (ob.mode & OB_MODE_SCULPT) && ob.runtime->sculpt_session;
}

static void calc_layers_only_eval_positions(Object &object)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  SculptSession &ss = *object.runtime->sculpt_session;
  BLI_assert(ss.layers_only_eval);
  if (ss.layers_cos.size() != mesh.verts_num) {
    ss.layers_cos.reinitialize(mesh.verts_num);
    ss.vert_normals_deform = {};
    ss.face_normals_deform = {};
  }

  bke::sculpt_layer::calc_eval_positions(mesh, ss.layers_cos.as_mutable_span());
}

bool sync_after_layer_change(Object &object,
                             const Depsgraph &depsgraph,
                             const bool positions_changed)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  if (mesh.multires != nullptr && mesh.multires->curr_level > 0) {
    if (mesh.multires->runtime != nullptr) {
      mesh.multires->runtime->layers_active = mesh.sculpt_layers_len > 0;
      if (mesh.multires->runtime->layers_active) {
        if (positions_changed || mesh.multires->runtime->eval_positions().is_empty()) {
          mesh.multires->runtime->calc_eval_positions(mesh, mesh.multires->base_positions());
        }
      }
      else {
        mesh.multires->runtime->eval_positions_.clear_and_shrink();
      }
    }
    return false;
  }

  if (!positions_changed) {
    return false;
  }

  SculptSession *ss = object.runtime->sculpt_session;
  if (ss != nullptr && BKE_eval_layers_only(depsgraph, &object)) {
    ss->layers_only_eval = true;
    calc_layers_only_eval_positions(object);
    return false;
  }
  if (ss != nullptr && ss->layers_only_eval) {
    ss->layers_cos = {};
    ss->layers_only_eval = false;
    if (!ss->deform_modifiers_active && ss->shapekey_active == nullptr) {
      ss->vert_normals_deform = {};
      ss->face_normals_deform = {};
      return false;
    }
  }
  return ss == nullptr || ss->shapekey_active != nullptr || ss->deform_modifiers_active;
}

static void finish(bContext *C,
                   Object &ob,
                   const Depsgraph &depsgraph,
                   const bool in_sculpt,
                   const bool need_geometry,
                   const bool positions_changed = true)
{
  if (bke::pbvh::Tree *pbvh = bke::object::pbvh_get(ob)) {
    pbvh->clear_layer_non_zero_nodes_cache();
    if (positions_changed && !need_geometry) {
      IndexMaskMemory memory;
      pbvh->tag_positions_changed(bke::pbvh::all_leaf_nodes(*pbvh, memory));
      pbvh->update_bounds(depsgraph, ob);
      bke::pbvh::store_bounds_orig(*pbvh);
    }
  }
  if (positions_changed) {
    if (!in_sculpt || need_geometry) {
      DEG_id_tag_update(&ob.id, ID_RECALC_GEOMETRY);
    }
    if (in_sculpt) {
      DEG_id_tag_update(&ob.id, ID_RECALC_SHADING);
    }
  }
  WM_event_add_notifier(C, NC_OBJECT | ND_DRAW, &ob);
  WM_event_add_notifier(C, NC_GEOM | ND_DATA, ob.data);
}

static wmOperatorStatus layer_add_exec(bContext *C, wmOperator *op)
{
  const Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  Object &ob = *CTX_data_active_object(C);
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  const bool in_sculpt = is_sculpt(ob);

  if (in_sculpt) {
    storage_undo::push_begin(ob, op, storage_undo::StorageUndoOpType::AddLayer);
  }

  bke::sculpt_layer::add_layer(mesh, "Layer");

  bool need_geometry = false;
  if (in_sculpt) {
    need_geometry = sync_after_layer_change(ob, depsgraph, false);
    storage_undo::push_end(ob);
  }
  else {
    ED_undo_push(C, op->type->name);
  }

  finish(C, ob, depsgraph, in_sculpt, need_geometry, false);
  return OPERATOR_FINISHED;
}

void SCULPT_OT_layer_add(wmOperatorType *ot)
{
  ot->name = "Add Sculpt Layer";
  ot->idname = "SCULPT_OT_layer_add";
  ot->description = "Add a new sculpt displacement layer";
  ot->exec = layer_add_exec;
  ot->poll = poll;
  ot->flag = OPTYPE_REGISTER;
}

static wmOperatorStatus layer_remove_exec(bContext *C, wmOperator *op)
{
  const Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  Object &ob = *CTX_data_active_object(C);
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  if (mesh.sculpt_layers_len == 0) {
    return OPERATOR_CANCELLED;
  }

  const int index = mesh.active_layer_index;
  if (index < 0 || index >= mesh.sculpt_layers_len) {
    return OPERATOR_CANCELLED;
  }

  const bool in_sculpt = is_sculpt(ob);

  if (in_sculpt) {
    storage_undo::push_begin(ob, op, storage_undo::StorageUndoOpType::DeleteLayer);

    const StringRef layer_name = mesh.sculpt_layers[index].name;
    const std::string attr_name = bke::sculpt_layer::active_layer_attr_name(mesh);

    Span<float3> curr_disp = bke::sculpt_layer::active_layer_displacements(mesh);
    if (!curr_disp.is_empty()) {
      storage_undo::push_attribute_deleted(
          storage_undo::StorageTarget::Curr, attr_name, bke::AttrType::Float3, curr_disp);
    }

    if (mesh.multires != nullptr && mesh.multires->curr_level != mesh.multires->top_level) {
      Span<float3> top_disp = bke::sculpt_layer::layer_top_displacements(mesh, layer_name);
      if (!top_disp.is_empty()) {
        storage_undo::push_attribute_deleted(
            storage_undo::StorageTarget::Top, attr_name, bke::AttrType::Float3, top_disp);
      }
    }
  }

  bke::sculpt_layer::remove_active_layer(mesh);
  bool need_geometry = true;
  if (in_sculpt) {
    need_geometry = sync_after_layer_change(ob, depsgraph, true);
  }

  if (in_sculpt) {
    storage_undo::push_end(ob);
  }
  else {
    ED_undo_push(C, op->type->name);
  }

  finish(C, ob, depsgraph, in_sculpt, need_geometry);
  return OPERATOR_FINISHED;
}

void SCULPT_OT_layer_remove(wmOperatorType *ot)
{
  ot->name = "Remove Layer";
  ot->idname = "SCULPT_OT_layer_remove";
  ot->description = "Remove the active layer";
  ot->exec = layer_remove_exec;
  ot->poll = active_poll;
  ot->flag = OPTYPE_REGISTER;
}

static const EnumPropertyItem layer_move_direction_items[] = {
    {1, "UP", 0, "Up", "Move the layer up"},
    {-1, "DOWN", 0, "Down", "Move the layer down"},
    {0, nullptr, 0, nullptr, nullptr},
};

static wmOperatorStatus layer_move_exec(bContext *C, wmOperator *op)
{
  Object *ob = CTX_data_active_object(C);
  Mesh *mesh = BKE_mesh_from_object(ob);
  if (!mesh || mesh->sculpt_layers_len < 2) {
    return OPERATOR_CANCELLED;
  }

  const int index = mesh->active_layer_index;
  const int direction = RNA_enum_get(op->ptr, "direction");
  const int new_index = index + direction;

  if (new_index < 0 || new_index >= mesh->sculpt_layers_len) {
    return OPERATOR_CANCELLED;
  }

  const bool in_sculpt = is_sculpt(*ob);
  if (in_sculpt) {
    storage_undo::push_begin(*ob, op, storage_undo::StorageUndoOpType::MoveLayer);
  }

  std::swap(mesh->sculpt_layers[index], mesh->sculpt_layers[new_index]);
  mesh->active_layer_index = new_index;

  if (in_sculpt) {
    storage_undo::push_end(*ob);
  }
  else {
    ED_undo_push(C, op->type->name);
  }

  DEG_id_tag_update(&mesh->id, ID_RECALC_SHADING);
  WM_event_add_notifier(C, NC_GEOM | ND_DATA, mesh);
  return OPERATOR_FINISHED;
}

void SCULPT_OT_layer_move(wmOperatorType *ot)
{
  ot->name = "Move Layer";
  ot->idname = "SCULPT_OT_layer_move";
  ot->description = "Move the active layer up or down";
  ot->exec = layer_move_exec;
  ot->poll = active_poll;
  ot->flag = OPTYPE_REGISTER;

  RNA_def_enum(ot->srna, "direction", layer_move_direction_items, 1, "Direction", "");
}

static wmOperatorStatus layer_duplicate_exec(bContext *C, wmOperator *op)
{
  const Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  Object &ob = *CTX_data_active_object(C);
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  if (mesh.sculpt_layers_len == 0) {
    return OPERATOR_CANCELLED;
  }
  const int index = mesh.active_layer_index;
  if (index < 0 || index >= mesh.sculpt_layers_len) {
    return OPERATOR_CANCELLED;
  }

  const bool in_sculpt = is_sculpt(ob);
  if (in_sculpt) {
    storage_undo::push_begin(ob, op, storage_undo::StorageUndoOpType::DuplicateLayer);
  }

  bke::sculpt_layer::duplicate_active_layer(mesh);

  bool need_geometry = true;
  if (in_sculpt) {
    need_geometry = sync_after_layer_change(ob, depsgraph, true);
    storage_undo::push_end(ob);
  }
  else {
    ED_undo_push(C, op->type->name);
  }

  finish(C, ob, depsgraph, in_sculpt, need_geometry);
  return OPERATOR_FINISHED;
}

void SCULPT_OT_layer_duplicate(wmOperatorType *ot)
{
  ot->name = "Duplicate Layer";
  ot->idname = "SCULPT_OT_layer_duplicate";
  ot->description = "Duplicate the active layer above it";
  ot->exec = layer_duplicate_exec;
  ot->poll = active_poll;
  ot->flag = OPTYPE_REGISTER;
}

static wmOperatorStatus layer_merge_to_base_exec(bContext *C, wmOperator *op)
{
  const Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  Object &ob = *CTX_data_active_object(C);
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  if (mesh.sculpt_layers_len == 0) {
    return OPERATOR_CANCELLED;
  }
  const int index = mesh.active_layer_index;
  if (index < 0 || index >= mesh.sculpt_layers_len) {
    return OPERATOR_CANCELLED;
  }

  const bool in_sculpt = is_sculpt(ob);

  if (in_sculpt) {
    storage_undo::push_begin(ob, op, storage_undo::StorageUndoOpType::MergeLayerToBase);

    const StringRef src_name = mesh.sculpt_layers[index].name;
    const std::string src_attr = bke::sculpt_layer::attr_name_from_layer_name(src_name);

    const Span<float3> curr_disp = bke::sculpt_layer::active_layer_displacements(mesh);
    if (!curr_disp.is_empty()) {
      storage_undo::push_attribute_deleted(
          storage_undo::StorageTarget::Curr, src_attr, bke::AttrType::Float3, curr_disp);
    }

    if (mesh.multires != nullptr && mesh.multires->curr_level < mesh.multires->top_level) {
      const Span<float3> top_disp = bke::sculpt_layer::layer_top_displacements(mesh, src_name);
      if (!top_disp.is_empty()) {
        storage_undo::push_attribute_deleted(
            storage_undo::StorageTarget::Top, src_attr, bke::AttrType::Float3, top_disp);
      }
    }
  }

  bke::sculpt_layer::merge_active_layer_to_base(mesh);

  bool need_geometry = true;
  if (in_sculpt) {
    need_geometry = sync_after_layer_change(ob, depsgraph, true);
    storage_undo::push_end(ob);
  }
  else {
    ED_undo_push(C, op->type->name);
  }

  finish(C, ob, depsgraph, in_sculpt, need_geometry);
  return OPERATOR_FINISHED;
}

void SCULPT_OT_layer_merge_to_base(wmOperatorType *ot)
{
  ot->name = "Merge to base";
  ot->idname = "SCULPT_OT_layer_merge_to_base";
  ot->description = "Bake the active layer into the base positions";
  ot->exec = layer_merge_to_base_exec;
  ot->poll = active_poll;
  ot->flag = OPTYPE_REGISTER;
}

static wmOperatorStatus layer_merge_down_exec(bContext *C, wmOperator *op)
{
  Object &ob = *CTX_data_active_object(C);
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  if (mesh.sculpt_layers_len == 0) {
    return OPERATOR_CANCELLED;
  }
  const int index = mesh.active_layer_index;
  if (index < 0 || index >= mesh.sculpt_layers_len) {
    return OPERATOR_CANCELLED;
  }

  if (index == 0) {
    return layer_merge_to_base_exec(C, op);
  }

  const Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  const bool in_sculpt = is_sculpt(ob);
  const bool merge_top = mesh.multires != nullptr &&
                         mesh.multires->curr_level != mesh.multires->top_level;

  const std::string dst_name = mesh.sculpt_layers[index - 1].name;
  const std::string dst_attr_name = bke::sculpt_layer::attr_name_from_layer_name(dst_name);
  Array<float3> dst_curr_pre;
  Array<float3> dst_top_pre;

  if (in_sculpt) {
    storage_undo::push_begin(ob, op, storage_undo::StorageUndoOpType::MergeLayer);

    const StringRef src_name = mesh.sculpt_layers[index].name;
    const std::string src_attr_name = bke::sculpt_layer::attr_name_from_layer_name(src_name);

    const Span<float3> src_disp = bke::sculpt_layer::active_layer_displacements(mesh);
    if (!src_disp.is_empty()) {
      storage_undo::push_attribute_deleted(
          storage_undo::StorageTarget::Curr, src_attr_name, bke::AttrType::Float3, src_disp);
    }

    const Span<float3> dst_disp = bke::sculpt_layer::layer_displacements_for_write(mesh, dst_name);
    dst_curr_pre.reinitialize(dst_disp.size());
    dst_curr_pre.as_mutable_span().copy_from(dst_disp);

    if (merge_top) {
      const Span<float3> top_src_disp = bke::sculpt_layer::layer_top_displacements(mesh, src_name);
      if (!top_src_disp.is_empty()) {
        storage_undo::push_attribute_deleted(
            storage_undo::StorageTarget::Top, src_attr_name, bke::AttrType::Float3, top_src_disp);
      }

      const Span<float3> top_dst_disp = bke::sculpt_layer::layer_top_displacements_for_write(
          mesh, dst_name);
      dst_top_pre.reinitialize(top_dst_disp.size());
      dst_top_pre.as_mutable_span().copy_from(top_dst_disp);
    }
  }

  bke::sculpt_layer::merge_active_layer_down(mesh);

  bool need_geometry = true;
  if (in_sculpt) {
    const Span<float3> dst_disp = bke::sculpt_layer::layer_displacements(mesh, dst_name);
    storage_undo::push_attribute_modified(storage_undo::StorageTarget::Curr,
                                          dst_attr_name,
                                          bke::AttrType::Float3,
                                          dst_curr_pre.as_span(),
                                          dst_disp);
    if (merge_top) {
      const Span<float3> top_dst_disp = bke::sculpt_layer::layer_top_displacements(mesh, dst_name);
      storage_undo::push_attribute_modified(storage_undo::StorageTarget::Top,
                                            dst_attr_name,
                                            bke::AttrType::Float3,
                                            dst_top_pre.as_span(),
                                            top_dst_disp);
    }
    need_geometry = sync_after_layer_change(ob, depsgraph, true);
    storage_undo::push_end(ob);
  }
  else {
    ED_undo_push(C, op->type->name);
  }

  finish(C, ob, depsgraph, in_sculpt, need_geometry);
  return OPERATOR_FINISHED;
}

void SCULPT_OT_layer_merge_down(wmOperatorType *ot)
{
  ot->name = "Merge Down";
  ot->idname = "SCULPT_OT_layer_merge_down";
  ot->description =
      "Bake the active layer into the layer below it (or into base for the first layer)";
  ot->exec = layer_merge_down_exec;
  ot->poll = active_poll;
  ot->flag = OPTYPE_REGISTER;
}

}  // namespace blender::ed::sculpt_paint::layer
