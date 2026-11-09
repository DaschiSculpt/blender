/* SPDX-FileCopyrightText: 2026 Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BLI_array.hh"
#include "BLI_array_utils.hh"
#include "BLI_enumerable_thread_specific.hh"
#include "BLI_task.hh"

#include "BKE_attribute_storage.hh"
#include "BKE_context.hh"
#include "BKE_mesh.hh"
#include "BKE_multiresolution.hh"
#include "BKE_object.hh"
#include "BKE_object_types.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_report.hh"
#include "BKE_sculpt_layer.hh"

#include "DEG_depsgraph.hh"

#include "DNA_object_types.h"

#include "WM_api.hh"
#include "WM_types.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "ED_sculpt.hh"

#include "mesh_brush_common.hh"
#include "sculpt_intern.hh"
#include "sculpt_morph.hh"
#include "sculpt_storage_undo.hh"

namespace blender::ed::sculpt_paint::morph {

Span<float3> base_morph_positions(const Mesh &mesh)
{
  const bke::AttributeReader<float3> reader = mesh.attributes().lookup<float3>(
      position_attr_name, bke::AttrDomain::Point);
  if (!reader || !reader.varray.is_span()) {
    return {};
  }
  return reader.varray.get_internal_span();
}

bool morph_exists(const Object &object)
{
  const bke::pbvh::Tree *pbvh = bke::object::pbvh_get(object);
  if (pbvh == nullptr) {
    return false;
  }
  const Mesh &mesh = *id_cast<const Mesh *>(object.data);

  if (pbvh->type() == bke::pbvh::Type::Mesh) {
    const Span<float3> morph_positions = base_morph_positions(mesh);
    return !morph_positions.is_empty() && morph_positions.size() == mesh.verts_num;
  }

  if (pbvh->type() == bke::pbvh::Type::Multires) {
    const Multiresolution &multires = *mesh.multires;
    const Span<float3> morph_positions = multires.attribute<float3>(morph::position_attr_name);
    return !morph_positions.is_empty() && morph_positions.size() == multires.verts_num();
  }

  return false;
}

static bool is_sculpt(const Object &ob)
{
  return (ob.mode & OB_MODE_SCULPT) && ob.runtime->sculpt_session;
}

static void morph_finish(bContext *C, Object &ob, const Depsgraph &depsgraph)
{
  if (bke::pbvh::Tree *pbvh = bke::object::pbvh_get(ob)) {
    IndexMaskMemory memory;
    pbvh->tag_positions_changed(bke::pbvh::all_leaf_nodes(*pbvh, memory));
    pbvh->update_bounds(depsgraph, ob);
    bke::pbvh::store_bounds_orig(*pbvh);
  }
  DEG_id_tag_update(&ob.id, ID_RECALC_SHADING);
  WM_event_add_notifier(C, NC_OBJECT | ND_DRAW, &ob);
  WM_event_add_notifier(C, NC_GEOM | ND_DATA, ob.data);
}

static void capture_store_added()
{
  push_attribute_added(
      storage_undo::StorageTarget::Curr, position_attr_name, bke::AttrType::Float3);
}

static Span<float3> curr_storage_span(const Multiresolution &multires)
{
  const bke::AttributeReader<float3> reader = multires.curr_attributes().lookup<float3>(
      position_attr_name, bke::AttrDomain::Point);
  if (!reader || !reader.varray.is_span()) {
    return {};
  }
  return reader.varray.get_internal_span();
}

static void capture_delete_attr(const Span<float3> attr,
                                const storage_undo::StorageTarget target,
                                const storage_undo::StorageTarget origin)
{
  if (attr.is_empty()) {
    return;
  }
  storage_undo::push_attribute_deleted(
      target, position_attr_name, bke::AttrType::Float3, attr, origin);
}

static void capture_multires_morph(const Mesh &mesh)
{
  if (mesh.multires == nullptr) {
    return;
  }
  const Multiresolution &multires = *mesh.multires;

  capture_delete_attr(curr_storage_span(multires),
                      storage_undo::StorageTarget::Curr,
                      storage_undo::StorageTarget::Curr);

  capture_delete_attr(multires.top_attribute<float3>(position_attr_name),
                      storage_undo::StorageTarget::Top,
                      storage_undo::StorageTarget::Top);
}

static void remove_multires_morph(const Mesh &mesh)
{
  if (mesh.multires == nullptr) {
    return;
  }

  Multiresolution &multires = *mesh.multires;
  multires.curr_attribute_storage.wrap().remove(position_attr_name);
  multires.top_attribute_storage.wrap().remove(position_attr_name);
}

static bool is_type_supported(const bke::pbvh::Tree *pbvh)
{
  return pbvh != nullptr &&
         (pbvh->type() == bke::pbvh::Type::Mesh || pbvh->type() == bke::pbvh::Type::Multires);
}

static bool any_morph_exists(const Mesh &mesh)
{
  if (mesh.attributes().contains(position_attr_name)) {
    return true;
  }
  if (mesh.multires == nullptr) {
    return false;
  }

  const Multiresolution &multires = *mesh.multires;
  return !curr_storage_span(multires).is_empty() ||
         !multires.top_attribute<float3>(position_attr_name).is_empty();
}

static bool store_can_run(Object &ob, const Depsgraph &depsgraph)
{
  Mesh &mesh = *id_cast<Mesh *>(ob.data);
  const bke::pbvh::Tree *pbvh = bke::object::pbvh_get(ob);
  if (!is_type_supported(pbvh)) {
    return false;
  }

  if (any_morph_exists(mesh)) {
    return false;
  }

  if (pbvh->type() == bke::pbvh::Type::Mesh) {
    const PositionDeformData position_data(depsgraph, ob);
    return position_data.eval.size() == mesh.verts_num;
  }
  if (!bke::multiresolution::base_topology_matches(mesh)) {
    return false;
  }
  if (mesh.multires == nullptr || mesh.multires->positions().is_empty()) {
    return false;
  }

  return true;
}

static void calc_subdivided_base_positions(const Mesh &mesh,
                                           const Multiresolution &multires,
                                           const Span<float3> positions,
                                           const MutableSpan<float3> r_smoothed)
{
  const int curr_lvl = multires.curr_level;
  if (curr_lvl == 0) {
    r_smoothed.copy_from(positions);
    return;
  }
  const bke::multiresolution::BaseTopology curr_base = multires.get_base_topology(mesh, curr_lvl);
  const bke::multiresolution::Offsets curr_offsets = Multiresolution::build_table_offsets(
      multires, curr_base, curr_lvl);

  Array<float3> base_positions(mesh.verts_num);
  Multiresolution::downsample_to_base<float3>(
      curr_offsets.verts, positions, base_positions.as_mutable_span());

  const bool linear = mesh.multires_use_linear != 0;
  multires.subdivide_base_to<float3>(
      mesh, curr_lvl, curr_offsets, base_positions.as_span(), r_smoothed, linear);
}

void store_apply(Object &ob, const Depsgraph &depsgraph, const MorphStoreMode store_mode)
{
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  const bke::pbvh::Tree *pbvh = bke::object::pbvh_get(ob);
  if (pbvh == nullptr) {
    return;
  }

  if (any_morph_exists(mesh)) {
    return;
  }

  switch (pbvh->type()) {
    case bke::pbvh::Type::Mesh: {
      const PositionDeformData position_data(depsgraph, ob);
      const Span<float3> eval_positions = position_data.eval;

      bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
      bke::SpanAttributeWriter<float3> morph_positions = attributes.ensure_for_write_span<float3>(
          position_attr_name, bke::AttrDomain::Point);

      morph_positions.span.copy_from(eval_positions);
      morph_positions.finish();

      capture_store_added();
      return;
    }
    case bke::pbvh::Type::Multires: {
      if (!bke::multiresolution::base_topology_matches(mesh)) {
        return;
      }
      Multiresolution &multires = *mesh.multires;
      const Span<float3> positions = multires.positions();

      const MutableSpan<float3> morph_positions = multires.attribute_for_write<float3>(
          position_attr_name);

      switch (store_mode) {
        case MorphStoreMode::CurrentPositions: {
          morph_positions.copy_from(positions);
          break;
        }
        case MorphStoreMode::SubdividedBase: {
          calc_subdivided_base_positions(mesh, multires, positions, morph_positions);
          break;
        }
      }

      capture_store_added();
      return;
    }
    case bke::pbvh::Type::Grids:
    case bke::pbvh::Type::BMesh: {
      BLI_assert_unreachable();
      return;
    }
  }
}

static bool morph_store_poll(bContext *C)
{
  if (!sculpt_mode_poll(C)) {
    return false;
  }
  const Object &ob = *CTX_data_active_object(C);
  const Mesh &mesh = *id_cast<const Mesh *>(ob.data);
  return is_type_supported(bke::object::pbvh_get(ob)) && !any_morph_exists(mesh);
}

static wmOperatorStatus morph_store_exec(bContext *C, wmOperator *op)
{
  Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  Object &ob = *CTX_data_active_object(C);

  if (!is_sculpt(ob) || !store_can_run(ob, depsgraph)) {
    return OPERATOR_CANCELLED;
  }

  const MorphStoreMode source = MorphStoreMode(RNA_enum_get(op->ptr, "store_mode"));
  const bool use_undo = RNA_boolean_get(op->ptr, "use_undo");

  const storage_undo::StorageUndoOpType op_type =
      (source == MorphStoreMode::SubdividedBase) ?
          storage_undo::StorageUndoOpType::MorphStoreFromSubdividedBase :
          storage_undo::StorageUndoOpType::MorphStore;
  if (use_undo) {
    storage_undo::push_begin(ob, op, op_type);
  }
  store_apply(ob, depsgraph, source);
  if (use_undo) {
    storage_undo::push_end(ob);
  }
  else {
    WM_file_tag_modified();
  }
  morph_finish(C, ob, depsgraph);

  return OPERATOR_FINISHED;
}

static const EnumPropertyItem morph_store_mode_items[] = {
    {int(MorphStoreMode::CurrentPositions),
     "CURRENT_POSITIONS",
     0,
     "Current Positions",
     "Store the current positions"},
    {int(MorphStoreMode::SubdividedBase),
     "SUBDIVIDED_BASE",
     0,
     "Subdivided Base",
     "Store the base mesh subdivided to the current level"},
    {0, nullptr, 0, nullptr, nullptr},
};

void SCULPT_OT_morph_store(wmOperatorType *ot)
{
  ot->name = "Store Morph";
  ot->idname = "SCULPT_OT_morph_store";
  ot->description = "Store the morph";
  ot->exec = morph_store_exec;
  ot->poll = morph_store_poll;
  ot->flag = OPTYPE_REGISTER;

  RNA_def_enum(ot->srna,
               "store_mode",
               morph_store_mode_items,
               int(MorphStoreMode::CurrentPositions),
               "Store Mode",
               "Which positions to store");
  RNA_def_boolean(
      ot->srna, "use_undo", true, "Use Undo", "Push an undo step for the morph operators");
}

static void push_swap_span(const storage_undo::StorageTarget target,
                           const StringRef attr_name,
                           const Span<float3> before,
                           const Span<float3> after)
{
  if (before.is_empty() || after.is_empty()) {
    BLI_assert_unreachable();
    return;
  }
  if (before.size() != after.size()) {
    BLI_assert_unreachable();
    return;
  }
  storage_undo::push_attribute_modified(target, attr_name, bke::AttrType::Float3, before, after);
}

void swap_apply(Object &ob, const Depsgraph &depsgraph, const bool use_mask, const bool use_hide)
{
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  const bke::pbvh::Tree *pbvh = bke::object::pbvh_get(ob);
  if (pbvh == nullptr) {
    return;
  }

  const bool use_factors = use_mask || use_hide;

  struct LocalData {
    Vector<int> verts;
    Vector<float3> translations;
    Vector<float> factors;
  };

  switch (pbvh->type()) {
    case bke::pbvh::Type::Mesh: {
      const PositionDeformData position_data(depsgraph, ob);
      const Span<float3> eval_positions = position_data.eval;

      bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
      bke::SpanAttributeWriter<float3> morph_positions = attributes.ensure_for_write_span<float3>(
          position_attr_name, bke::AttrDomain::Point);

      const MeshAttributeData attribute_data(mesh);
      const Span<bool> hide_span = use_hide ? Span<bool>(attribute_data.hide_vert) : Span<bool>();
      const Span<float> mask_span = use_mask ? Span<float>(attribute_data.mask) : Span<float>();

      threading::EnumerableThreadSpecific<LocalData> all_tls;

      threading::parallel_for(IndexRange(mesh.verts_num), 4096, [&](const IndexRange range) {
        LocalData &tls = all_tls.local();

        const Span<float3> local_eval_positions = eval_positions.slice(range);
        const MutableSpan<float3> local_morph_positions = morph_positions.span.slice(range);

        const int size = range.size();
        tls.verts.resize(size);
        const MutableSpan<int> verts = tls.verts;
        array_utils::fill_index_range(verts, int(range.first()));

        tls.translations.resize(size);
        for (const int i : tls.translations.index_range()) {
          tls.translations[i] = local_morph_positions[i] - local_eval_positions[i];
        }
        if (use_factors) {
          tls.factors.resize(size);
          fill_factor_from_hide_and_mask(
              hide_span, mask_span, verts, tls.factors.as_mutable_span());
          scale_translations(tls.translations.as_mutable_span(), tls.factors.as_span());
          for (const int i : tls.translations.index_range()) {
            local_morph_positions[i] -= tls.translations[i];
          }
        }
        else {
          local_morph_positions.copy_from(local_eval_positions);
        }
        position_data.deform(tls.translations, verts);
      });

      morph_positions.finish();

      return;
    }
    case bke::pbvh::Type::Multires: {
      if (mesh.multires == nullptr) {
        return;
      }
      if (!bke::multiresolution::base_topology_matches(mesh)) {
        return;
      }
      Multiresolution &multires = *mesh.multires;
      const MultiresDeformData deform_data(mesh);
      const MutableSpan<float3> eval_positions = multires.positions_for_write();
      const MutableSpan<float3> morph_positions = multires.attribute_for_write<float3>(
          position_attr_name);

      const Span<bool> hidden_span = use_hide ? multires.hidden_verts() : Span<bool>();
      const Span<float> mask_span = use_mask ? multires.masks() : Span<float>();

      threading::EnumerableThreadSpecific<LocalData> all_tls;

      threading::parallel_for(eval_positions.index_range(), 4096, [&](const IndexRange range) {
        LocalData &tls = all_tls.local();

        const MutableSpan<float3> local_eval_positions = eval_positions.slice(range);
        const MutableSpan<float3> local_morph_positions = morph_positions.slice(range);

        const int size = range.size();
        tls.translations.resize(size);
        for (const int i : tls.translations.index_range()) {
          tls.translations[i] = local_morph_positions[i] - local_eval_positions[i];
        }

        if (use_factors) {
          tls.factors.resize(size);
          fill_factor_from_hide_and_mask(
              hidden_span, mask_span, range, tls.factors.as_mutable_span());
          scale_translations(tls.translations.as_mutable_span(), tls.factors.as_span());
          for (const int i : tls.translations.index_range()) {
            local_morph_positions[i] -= tls.translations[i];
          }
        }
        else {
          local_morph_positions.copy_from(local_eval_positions);
        }
        deform_data.deform(range, tls.translations.as_span());
      });

      return;
    }
    case bke::pbvh::Type::Grids:
    case bke::pbvh::Type::BMesh: {
      BLI_assert_unreachable();
      return;
    }
  }
}

static void swap_apply_with_undo_capture(Object &ob,
                                         const Depsgraph &depsgraph,
                                         const bool use_mask,
                                         const bool use_hide)
{
  Mesh &mesh = *id_cast<Mesh *>(ob.data);
  const bke::pbvh::Tree *pbvh = bke::object::pbvh_get(ob);
  BLI_assert(pbvh != nullptr);
  const bool is_multires = pbvh->type() == bke::pbvh::Type::Multires &&
                           bke::multiresolution::base_topology_matches(mesh);

  const storage_undo::StorageTarget target = (is_multires && mesh.multires != nullptr &&
                                              mesh.multires->curr_level ==
                                                  mesh.multires->top_level) ?
                                                 storage_undo::StorageTarget::Top :
                                                 storage_undo::StorageTarget::Curr;

  const SculptLayer *active = bke::sculpt_layer::active_layer(mesh);
  const bool use_layer = active != nullptr;

  const auto current_morph_positions = [&]() -> Span<float3> {
    if (is_multires) {
      return mesh.multires->attribute<float3>(position_attr_name);
    }
    return base_morph_positions(mesh);
  };

  const auto current_base_positions = [&]() -> Span<float3> {
    if (is_multires) {
      BLI_assert(mesh.multires != nullptr);
      return mesh.multires->base_positions();
    }
    return mesh.vert_positions();
  };

  const Array<float3> morph_pos_before(current_morph_positions());

  Array<float3> base_before;
  std::string layer_name;
  std::string layer_attr;
  Array<float3> layer_before;
  if (use_layer) {
    layer_name = active->name;
    layer_attr = bke::sculpt_layer::attr_name_from_layer_name(layer_name);
    layer_before = Array<float3>(
        bke::sculpt_layer::layer_displacements_for_write(mesh, layer_name).as_span());
  }
  else {
    base_before = Array<float3>(current_base_positions());
  }

  Array<float3> shape_active_before;
  if (!use_layer && !is_multires) {
    if (std::optional<ShapeKeyData> shape_data = ShapeKeyData::from_object(ob)) {
      shape_active_before = Array<float3>(shape_data->active_key_data.as_span());
    }
  }

  swap_apply(ob, depsgraph, use_mask, use_hide);

  push_swap_span(target, position_attr_name, morph_pos_before, current_morph_positions());
  if (use_layer) {
    push_swap_span(target,
                   layer_attr,
                   layer_before,
                   bke::sculpt_layer::layer_displacements(mesh, layer_name));
  }
  else {
    push_swap_span(target, "position", base_before, current_base_positions());
  }
  if (!shape_active_before.is_empty()) {
    std::optional<ShapeKeyData> shape_after = ShapeKeyData::from_object(ob);
    if (shape_after.has_value()) {
      push_swap_span(target,
                     storage_undo::swap_active_shapekey_name,
                     shape_active_before,
                     shape_after->active_key_data);
    }
  }
}

static wmOperatorStatus morph_swap_exec(bContext *C, wmOperator *op)
{
  Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  Object &ob = *CTX_data_active_object(C);

  if (!is_sculpt(ob) || !morph_exists(ob)) {
    return OPERATOR_CANCELLED;
  }

  if (!sculpt_layer_check(ob, op->reports)) {
    return OPERATOR_CANCELLED;
  }

  const bool use_undo = RNA_boolean_get(op->ptr, "use_undo");
  const bool use_mask = RNA_boolean_get(op->ptr, "use_mask");
  const bool use_hide = RNA_boolean_get(op->ptr, "use_hide");

  if (use_undo) {
    storage_undo::push_begin(ob, op, storage_undo::StorageUndoOpType::MorphSwap);
  }
  swap_apply_with_undo_capture(ob, depsgraph, use_mask, use_hide);
  if (use_undo) {
    storage_undo::push_end(ob);
  }
  else {
    WM_file_tag_modified();
  }
  morph_finish(C, ob, depsgraph);

  return OPERATOR_FINISHED;
}

static bool morph_swap_poll(bContext *C)
{
  return sculpt_mode_poll(C) && morph_exists(*CTX_data_active_object(C));
}

void SCULPT_OT_morph_swap(wmOperatorType *ot)
{
  ot->name = "Swap Morph";
  ot->idname = "SCULPT_OT_morph_swap";
  ot->description = "Swap the current positions with the stored morph";
  ot->exec = morph_swap_exec;
  ot->poll = morph_swap_poll;
  ot->flag = OPTYPE_REGISTER;

  RNA_def_boolean(ot->srna,
                  "use_undo",
                  true,
                  "Use Undo",
                  "Store an undo step for the morph swap; disable to keep the redo stack");
  RNA_def_boolean(ot->srna,
                  "use_mask",
                  false,
                  "Respect Mask",
                  "Only swap vertices that are not masked (partially masked vertices are "
                  "interpolated by their mask factor)");
  RNA_def_boolean(
      ot->srna, "use_hide", false, "Respect Visibility", "Skip hidden vertices when swapping");
}

void delete_apply(Object &ob)
{
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  capture_delete_attr(base_morph_positions(mesh),
                      storage_undo::StorageTarget::Curr,
                      storage_undo::StorageTarget::Base);

  capture_multires_morph(mesh);

  mesh.attributes_for_write().remove(position_attr_name);
  remove_multires_morph(mesh);
}

static bool delete_can_run(const Object &ob)
{
  const Mesh &mesh = *id_cast<const Mesh *>(ob.data);
  return is_type_supported(bke::object::pbvh_get(ob)) && any_morph_exists(mesh);
}

static wmOperatorStatus morph_delete_exec(bContext *C, wmOperator *op)
{
  Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  Object &ob = *CTX_data_active_object(C);

  if (!is_sculpt(ob) || !delete_can_run(ob)) {
    return OPERATOR_CANCELLED;
  }

  const bool use_undo = RNA_boolean_get(op->ptr, "use_undo");

  if (use_undo) {
    storage_undo::push_begin(ob, op, storage_undo::StorageUndoOpType::MorphDelete);
  }
  delete_apply(ob);
  if (use_undo) {
    storage_undo::push_end(ob);
  }
  else {
    WM_file_tag_modified();
  }
  morph_finish(C, ob, depsgraph);

  return OPERATOR_FINISHED;
}

static bool morph_delete_poll(bContext *C)
{
  return sculpt_mode_poll(C) && delete_can_run(*CTX_data_active_object(C));
}

void SCULPT_OT_morph_delete(wmOperatorType *ot)
{
  ot->name = "Delete Morph";
  ot->idname = "SCULPT_OT_morph_delete";
  ot->description = "Delete the stored morph";
  ot->exec = morph_delete_exec;
  ot->poll = morph_delete_poll;
  ot->flag = OPTYPE_REGISTER;

  RNA_def_boolean(ot->srna,
                  "use_undo",
                  true,
                  "Use Undo",
                  "Store an undo step for the morph deletion; disable to keep the redo stack");
}

}  // namespace blender::ed::sculpt_paint::morph
