/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <cstdlib>

#include "DNA_mesh_types.h"
#include "DNA_scene_types.h"

#include "BLI_math_vector.hh"

#include "BKE_attribute_math.hh"
#include "BKE_attribute_storage.hh"
#include "BKE_context.hh"
#include "BKE_global.hh"
#include "BKE_layer.hh"
#include "BKE_lib_id.hh"
#include "BKE_main.hh"
#include "BKE_mesh.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_scene.hh"
#include "BKE_sculpt_layer.hh"

#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_query.hh"

#include "ED_object.hh"
#include "ED_sculpt.hh"

#include "sculpt_intern.hh"
#include "sculpt_morph.hh"
#include "sculpt_multires.hh"
#include "sculpt_storage_undo.hh"

namespace blender {

namespace ed::sculpt_paint::storage_undo {

static constexpr int BLOCK_SIZE = 128;

static undo::SculptUndoStep *get_init_step()
{
  UndoStack *ustack = ED_undo_stack_get();
  if (!ustack || !ustack->step_init) {
    return nullptr;
  }
  auto *us = reinterpret_cast<undo::SculptUndoStep *>(ustack->step_init);
  if (us->kind != undo::SculptUndoStepKind::Storage) {
    return nullptr;
  }
  return us;
}

template<typename T>
static void scatter_blocks(const MutableSpan<T> attr,
                           const Span<int> blocks,
                           const Span<std::byte> values)
{
  const Span<T> stored = values.cast<T>();
  threading::parallel_for(IndexRange(blocks.size()), 64, [&](const IndexRange range) {
    for (const int i : range) {
      const int start = blocks[i] * BLOCK_SIZE;
      const int block_size = math::min(BLOCK_SIZE, (int)attr.size() - start);
      attr.slice(start, block_size).copy_from(stored.slice(i * BLOCK_SIZE, block_size));
    }
  });
}

static bool is_layer_op(const StorageUndoOpType type)
{
  return ELEM(type,
              StorageUndoOpType::AddLayer,
              StorageUndoOpType::DeleteLayer,
              StorageUndoOpType::MoveLayer,
              StorageUndoOpType::SetActiveLayer,
              StorageUndoOpType::RenameLayer,
              StorageUndoOpType::DuplicateLayer,
              StorageUndoOpType::MergeLayer,
              StorageUndoOpType::MergeLayerToBase);
}

static bool is_morph_op(const StorageUndoOpType type)
{
  return ELEM(type,
              StorageUndoOpType::MorphStore,
              StorageUndoOpType::MorphStoreFromSubdividedBase,
              StorageUndoOpType::MorphSwap,
              StorageUndoOpType::MorphDelete);
}

static void snapshot_layers(const Mesh &mesh, LayerSnapshot &snapshot)
{
  snapshot.layers.reinitialize(mesh.sculpt_layers_len);
  if (mesh.sculpt_layers_len > 0) {
    snapshot.layers.as_mutable_span().copy_from(mesh.sculpt_layers_as_span());
  }
  snapshot.active_layer_index = mesh.active_layer_index;
  snapshot.layers_active = (mesh.multires != nullptr && mesh.multires->runtime != nullptr) ?
                               mesh.multires->runtime->layers_active :
                               (mesh.sculpt_layers_len > 0);
}

static void restore_layers(Mesh &mesh, const LayerSnapshot &snapshot)
{
  const int new_len = int(snapshot.layers.size());
  if (new_len == 0) {
    MEM_SAFE_DELETE(mesh.sculpt_layers);
  }
  else {
    mesh.sculpt_layers = static_cast<SculptLayer *>(
        MEM_realloc_uninitialized(mesh.sculpt_layers, sizeof(SculptLayer) * new_len));
    MutableSpan<SculptLayer>(mesh.sculpt_layers, new_len).copy_from(snapshot.layers.as_span());
  }
  mesh.sculpt_layers_len = new_len;
  mesh.active_layer_index = snapshot.active_layer_index;
  if (mesh.multires != nullptr && mesh.multires->runtime != nullptr) {
    mesh.multires->runtime->layers_active = snapshot.layers_active;
  }
}

static void remove_layer_attributes(Mesh &mesh, const StringRef attr_name)
{
  mesh.attributes_for_write().remove(attr_name);
  if (mesh.multires != nullptr) {
    mesh.multires->curr_attribute_storage.wrap().remove(attr_name);
    mesh.multires->top_attribute_storage.wrap().remove(attr_name);
    if (mesh.multires->runtime != nullptr) {
      mesh.multires->runtime->curr_dirty_layers.remove(attr_name);
    }
  }
}

static void rename_layer_attributes(Mesh &mesh,
                                    const StringRef old_attr_name,
                                    const StringRef new_attr_name)
{
  if (old_attr_name == new_attr_name) {
    return;
  }
  bke::AttributeStorage &base_storage = mesh.attribute_storage.wrap();
  if (base_storage.lookup(old_attr_name) != nullptr) {
    base_storage.rename(old_attr_name, new_attr_name);
  }
  if (mesh.multires != nullptr) {
    bke::AttributeStorage &curr_storage = mesh.multires->curr_attribute_storage.wrap();
    if (curr_storage.lookup(old_attr_name) != nullptr) {
      curr_storage.rename(old_attr_name, new_attr_name);
    }
    bke::AttributeStorage &top_storage = mesh.multires->top_attribute_storage.wrap();
    if (top_storage.lookup(old_attr_name) != nullptr) {
      top_storage.rename(old_attr_name, new_attr_name);
    }
    if (mesh.multires->runtime != nullptr) {
      if (mesh.multires->runtime->curr_dirty_layers.remove(old_attr_name)) {
        mesh.multires->runtime->curr_dirty_layers.add(new_attr_name);
      }
    }
  }
}

static void rename_changed_layer_attributes(Mesh &mesh,
                                            const LayerSnapshot &from,
                                            const LayerSnapshot &to)
{
  const int num = math::min(int(from.layers.size()), int(to.layers.size()));
  for (const int i : IndexRange(num)) {
    const StringRef from_name(from.layers[i].name);
    const StringRef to_name(to.layers[i].name);
    if (from_name != to_name) {
      rename_layer_attributes(mesh,
                              bke::sculpt_layer::attr_name_from_layer_name(from_name),
                              bke::sculpt_layer::attr_name_from_layer_name(to_name));
    }
  }
}

static void remove_morph_attr_curr(Mesh &mesh, const int curr_level_before, const StringRef name)
{
  if (mesh.multires == nullptr || curr_level_before == 0) {
    mesh.attributes_for_write().remove(name);
  }
  else {
    mesh.multires->curr_attribute_storage.wrap().remove(name);
  }
}

static void restore_morph_deleted(Mesh &mesh, const Vector<AttributeDiff> &diffs)
{
  const auto restore_to_storage =
      [](AttributeStorage &storage, const StringRef name, const Span<float3> values) {
        storage.wrap().remove(name);
        storage.wrap().add(
            std::string(name), bke::AttrDomain::Point, bke::GList(GArray(GSpan(values))));
      };
  for (const AttributeDiff &diff : diffs) {
    if (diff.kind != AttributeDiff::Kind::Deleted) {
      continue;
    }
    BLI_assert(diff.type == bke::AttrType::Float3);
    const Span<float3> values = diff.values.as_span().cast<float3>();
    switch (diff.origin) {
      case StorageTarget::Base: {
        bke::SpanAttributeWriter<float3> writer =
            mesh.attributes_for_write().lookup_or_add_for_write_span<float3>(
                diff.name, bke::AttrDomain::Point);
        writer.span.copy_from(values);
        writer.finish();
        break;
      }
      case StorageTarget::Curr: {
        BLI_assert(mesh.multires != nullptr);
        restore_to_storage(mesh.multires->curr_attribute_storage, diff.name, values);
        break;
      }
      case StorageTarget::Top: {
        BLI_assert(mesh.multires != nullptr);
        restore_to_storage(mesh.multires->top_attribute_storage, diff.name, values);
        break;
      }
    }
  }
}

template<typename Fn>
static void modify_float3_attribute(
    Mesh &mesh, const int level, const StorageTarget target, const StringRef name, Fn &&fn)
{
  if (target == StorageTarget::Top) {
    BLI_assert(mesh.multires != nullptr);
    fn(mesh.multires->top_attribute_for_write<float3>(name));
  }
  else if (mesh.multires != nullptr && level > 0) {
    fn(mesh.multires->attribute_for_write<float3>(name));
  }
  else if (name == "position") {
    fn(mesh.vert_positions_for_write());
  }
  else {
    bke::SpanAttributeWriter<float3> writer =
        mesh.attributes_for_write().lookup_or_add_for_write_span<float3>(name,
                                                                         bke::AttrDomain::Point);
    fn(writer.span);
    writer.finish();
  }
}

static void restore_deleted_layer_attributes(Mesh &mesh,
                                             const int curr_level_before,
                                             const Vector<AttributeDiff> &curr_diffs,
                                             const Vector<AttributeDiff> &top_diffs)
{
  for (const AttributeDiff &diff : curr_diffs) {
    if (diff.kind != AttributeDiff::Kind::Deleted) {
      continue;
    }
    BLI_assert(diff.type == bke::AttrType::Float3);
    const Span<float3> values = diff.values.as_span().cast<float3>();
    modify_float3_attribute(mesh,
                            curr_level_before,
                            StorageTarget::Curr,
                            diff.name,
                            [&](const MutableSpan<float3> dst) { dst.copy_from(values); });
  }
  if (mesh.multires != nullptr) {
    for (const AttributeDiff &diff : top_diffs) {
      if (diff.kind != AttributeDiff::Kind::Deleted) {
        continue;
      }
      BLI_assert(diff.type == bke::AttrType::Float3);
      mesh.multires->top_attribute_for_write<float3>(diff.name).copy_from(
          diff.values.as_span().cast<float3>());
    }
  }
}

static void restore_merged_attributes(Mesh &mesh,
                                      const int level,
                                      const StorageTarget target,
                                      const Vector<AttributeDiff> &diffs)
{
  for (const AttributeDiff &diff : diffs) {
    if (diff.kind != AttributeDiff::Kind::Modified) {
      continue;
    }
    BLI_assert(diff.type == bke::AttrType::Float3);
    const Span<std::byte> stored = diff.values.as_span();
    if (stored.is_empty()) {
      continue;
    }
    modify_float3_attribute(mesh, level, target, diff.name, [&](const MutableSpan<float3> dst) {
      scatter_blocks<float3>(dst, diff.blocks, stored);
    });
  }
}

static const AttributeDiff *find_layer_diff(const Vector<AttributeDiff> &diffs,
                                            const AttributeDiff::Kind kind)
{
  for (const AttributeDiff &diff : diffs) {
    if (diff.kind == kind) {
      return &diff;
    }
  }
  return nullptr;
}

static float merge_src_weight(const StorageUndoStep &us)
{
  const int index = us.state_before.active_layer_index;
  BLI_assert(index >= 0 && index < int(us.layers_before.layers.size()));
  return bke::sculpt_layer::effective_layer_weight(us.layers_before.layers[index]);
}

static void undo_merge_to_base(Mesh &mesh, const StorageUndoStep &us)
{
  const float weight = -merge_src_weight(us);
  if (weight == 0.0f) {
    return;
  }
  const int level = us.state_before.curr_level;
  if (const AttributeDiff *curr = find_layer_diff(us.curr_diffs, AttributeDiff::Kind::Deleted)) {
    BLI_assert(curr->type == bke::AttrType::Float3);
    modify_float3_attribute(
        mesh, level, StorageTarget::Curr, "position", [&](const MutableSpan<float3> base) {
          bke::sculpt_layer::apply_displacement(
              curr->values.as_span().cast<float3>(), weight, base);
        });
  }
  if (mesh.multires != nullptr) {
    if (const AttributeDiff *top = find_layer_diff(us.top_diffs, AttributeDiff::Kind::Deleted)) {
      BLI_assert(top->type == bke::AttrType::Float3);
      bke::sculpt_layer::apply_displacement(top->values.as_span().cast<float3>(),
                                            weight,
                                            mesh.multires->top_base_positions_for_write());
    }
  }
}

static void update_after_restore(Object &object,
                                 Depsgraph &depsgraph,
                                 const bool positions_changed)
{
  const bool need_geometry = layer::sync_after_layer_change(object, depsgraph, positions_changed);

  bke::pbvh::Tree *pbvh = bke::object::pbvh_get(object);
  if (pbvh != nullptr) {
    pbvh->clear_layer_non_zero_nodes_cache();
    if (positions_changed) {
      IndexMaskMemory memory;
      const IndexMask all_nodes = bke::pbvh::all_leaf_nodes(*pbvh, memory);
      pbvh->tag_positions_changed(all_nodes);
      pbvh->update_bounds(depsgraph, object);
      bke::pbvh::store_bounds_orig(*pbvh);
    }
  }

  if (positions_changed && need_geometry) {
    DEG_id_tag_update(&object.id, ID_RECALC_GEOMETRY);
  }
  DEG_id_tag_update(&object.id, ID_RECALC_SHADING);
}

static ObjectState capture_object_state(const Mesh &mesh)
{
  ObjectState state;
  state.curr_level = (mesh.multires != nullptr) ? mesh.multires->curr_level : 0;
  state.sculpt_level = (mesh.multires != nullptr) ? mesh.multires->sculpt_level : 0;
  state.top_level = (mesh.multires != nullptr) ? mesh.multires->top_level : 0;
  state.active_layer_index = mesh.active_layer_index;
  return state;
}

void push_begin_with_name(Object &ob, const char *name, StorageUndoOpType type)
{
  UndoStack *ustack = ED_undo_stack_get();

  ED_undosys_stack_memfile_id_changed_tag(ustack, &ob.id);
  ED_undosys_stack_memfile_id_changed_tag(ustack, ob.data);

  StorageUndoStep *us = reinterpret_cast<StorageUndoStep *>(
      BKE_undosys_step_push_init_with_type(ustack, nullptr, name, BKE_UNDOSYS_TYPE_SCULPT));

  us->kind = undo::SculptUndoStepKind::Storage;
  us->op_type = type;
  us->object_name = ob.id.name;

  Mesh *mesh = id_cast<Mesh *>(ob.data);

  us->state_before = capture_object_state(*mesh);
  if (is_layer_op(type)) {
    snapshot_layers(*mesh, us->layers_before);
  }
}

void push_begin(Object &ob, const wmOperator *op, StorageUndoOpType type)
{
  push_begin_with_name(ob, op->type->name, type);
}

void push_set_active_layer_begin(Object &ob)
{
  push_begin_with_name(ob, "Set Active Sculpt Layer", StorageUndoOpType::SetActiveLayer);
}

void push_rename_layer_begin(Object &ob)
{
  push_begin_with_name(ob, "Rename Sculpt Layer", StorageUndoOpType::RenameLayer);
}

void push_end(Object &ob)
{
  UndoStack *ustack = ED_undo_stack_get();
  StorageUndoStep *us = get_init_step();
  if (!us) {
    return;
  }
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  us->state_after = capture_object_state(mesh);
  if (is_layer_op(us->op_type)) {
    snapshot_layers(mesh, us->layers_after);
  }

  BKE_undosys_step_push(ustack, nullptr, nullptr, UndoEncodeHints::None);

  us->step.is_applied = true;

  wmWindowManager *wm = G_MAIN->wm.first();
  if (wm->op_undo_depth == 0) {
    BKE_undosys_stack_limit_steps_and_memory_defaults(ustack);
  }

  WM_file_tag_modified();
}

bool is_active()
{
  return get_init_step() != nullptr;
}

static Vector<AttributeDiff> &get_target_diffs(StorageUndoStep &step, const StorageTarget target)
{
  switch (target) {
    case StorageTarget::Top:
      return step.top_diffs;
    case StorageTarget::Curr:
      return step.curr_diffs;
    default: {
      BLI_assert_unreachable();
      return step.top_diffs;
    }
  }
}

void push_attribute_added(const StorageTarget target,
                          const StringRef attr_name,
                          const bke::AttrType type)
{
  StorageUndoStep *us = get_init_step();
  if (!us) {
    return;
  }
  Vector<AttributeDiff> &target_diffs = get_target_diffs(*us, target);

  AttributeDiff diff;
  diff.name = std::string(attr_name);
  diff.type = type;
  diff.kind = AttributeDiff::Kind::Added;

  target_diffs.append(std::move(diff));
}

template<typename T>
void push_attribute_deleted(const StorageTarget target,
                            const StringRef attr_name,
                            const bke::AttrType attr_type,
                            const Span<T> attr,
                            const StorageTarget origin)
{
  StorageUndoStep *us = get_init_step();
  if (!us) {
    return;
  }

  Vector<AttributeDiff> &target_diffs = get_target_diffs(*us, target);

  AttributeDiff diff;
  diff.name = std::string(attr_name);
  diff.kind = AttributeDiff::Kind::Deleted;
  diff.type = attr_type;
  diff.values = Vector<std::byte>(attr.cast<std::byte>());
  diff.origin = origin;

  target_diffs.append(std::move(diff));
}

template void push_attribute_deleted<float>(const StorageTarget target,
                                            const StringRef attr_name,
                                            const bke::AttrType attr_type,
                                            const Span<float> attr,
                                            const StorageTarget origin);

template void push_attribute_deleted<float3>(const StorageTarget target,
                                             const StringRef attr_name,
                                             const bke::AttrType attr_type,
                                             const Span<float3> attr,
                                             const StorageTarget origin);

template void push_attribute_deleted<float4>(const StorageTarget target,
                                             const StringRef attr_name,
                                             const bke::AttrType attr_type,
                                             const Span<float4> attr,
                                             const StorageTarget origin);

template<typename T, typename IsBlockDirtyFn>
static bool push_modified_common(Vector<AttributeDiff> *target_diffs,
                                 const StringRef attr_name,
                                 const bke::AttrType type,
                                 const Span<T> attr,
                                 IsBlockDirtyFn &&is_block_dirty)
{
  const int N = int(attr.size());
  const int num_blocks = N / BLOCK_SIZE + (N % BLOCK_SIZE > 0);
  Array<bool> dirty(num_blocks);

  constexpr int GRAIN_SIZE = 64;

  threading::parallel_for(IndexRange(num_blocks), GRAIN_SIZE, [&](const IndexRange range) {
    for (const int block : range) {
      const int start = block * BLOCK_SIZE;
      const int end = math::min(start + BLOCK_SIZE, N);
      dirty[block] = is_block_dirty(start, end);
    }
  });

  Array<int> offsets(num_blocks + 1, 0);

  for (const int block : IndexRange(num_blocks)) {
    offsets[block + 1] = offsets[block] + dirty[block];
  }

  const int total_dirty = offsets[num_blocks];
  if (total_dirty == 0) {
    return false;
  }

  if (target_diffs == nullptr) {
    return true;
  }

  AttributeDiff diff;
  diff.name = std::string(attr_name);
  diff.type = type;
  diff.kind = AttributeDiff::Kind::Modified;
  diff.blocks.reinitialize(total_dirty);
  diff.values.reinitialize(total_dirty * BLOCK_SIZE * sizeof(T));

  MutableSpan<T> values = diff.values.as_mutable_span().cast<T>();

  threading::parallel_for(IndexRange(num_blocks), GRAIN_SIZE, [&](const IndexRange range) {
    for (const int block : range) {
      if (dirty[block]) {
        const int offset = offsets[block];
        diff.blocks[offset] = block;
        const int start = block * BLOCK_SIZE;
        const int block_size = math::min(BLOCK_SIZE, N - start);
        values.slice(offset * BLOCK_SIZE, block_size).copy_from(attr.slice(start, block_size));
      }
    }
  });

  target_diffs->append(std::move(diff));
  return true;
}

template<typename T> static bool is_nonzero_diff(const T &val)
{
  constexpr float EPS = 1e-10f;
  if constexpr (std::is_floating_point_v<T>) {
    return math::abs(val) > EPS;
  }
  else {
    return !math::is_zero(val, EPS);
  }
}

template<typename T>
bool push_attribute_modified(const StorageTarget target,
                             const StringRef attr_name,
                             const bke::AttrType type,
                             const Span<T> old_attr,
                             const Span<T> new_attr)
{
  BLI_assert(old_attr.size() == new_attr.size());
  StorageUndoStep *us = get_init_step();
  Vector<AttributeDiff> *target_diffs = us ? &get_target_diffs(*us, target) : nullptr;
  return push_modified_common(
      target_diffs, attr_name, type, old_attr, [&](const int start, const int end) {
        for (int i = start; i < end; i++) {
          if (is_nonzero_diff(new_attr[i] - old_attr[i])) {
            return true;
          }
        }
        return false;
      });
}

template<typename T>
bool push_attribute_modified_delta(const StorageTarget target,
                                   const StringRef attr_name,
                                   const bke::AttrType type,
                                   const Span<T> attr,
                                   const Span<T> delta)
{
  BLI_assert(attr.size() == delta.size());
  StorageUndoStep *us = get_init_step();
  Vector<AttributeDiff> *target_diffs = us ? &get_target_diffs(*us, target) : nullptr;
  return push_modified_common(
      target_diffs, attr_name, type, attr, [&](const int start, const int end) {
        for (int i = start; i < end; i++) {
          if (is_nonzero_diff(delta[i])) {
            return true;
          }
        }
        return false;
      });
}

template bool push_attribute_modified<float>(const StorageTarget target,
                                             const StringRef attr_name,
                                             const bke::AttrType type,
                                             const Span<float> old_attr,
                                             const Span<float> new_attr);
template bool push_attribute_modified<float3>(const StorageTarget target,
                                              const StringRef attr_name,
                                              const bke::AttrType type,
                                              const Span<float3> old_attr,
                                              const Span<float3> new_attr);
template bool push_attribute_modified<float4>(const StorageTarget target,
                                              const StringRef attr_name,
                                              const bke::AttrType type,
                                              const Span<float4> old_attr,
                                              const Span<float4> new_attr);

template bool push_attribute_modified_delta<float>(const StorageTarget target,
                                                   const StringRef attr_name,
                                                   const bke::AttrType type,
                                                   const Span<float> attr,
                                                   const Span<float> delta);
template bool push_attribute_modified_delta<float3>(const StorageTarget target,
                                                    const StringRef attr_name,
                                                    const bke::AttrType type,
                                                    const Span<float3> attr,
                                                    const Span<float3> delta);
template bool push_attribute_modified_delta<float4>(const StorageTarget target,
                                                    const StringRef attr_name,
                                                    const bke::AttrType type,
                                                    const Span<float4> attr,
                                                    const Span<float4> delta);

template<typename T>
void push_face_modified(const StringRef attr_name,
                        const bke::AttrType type,
                        const Span<T> src,
                        const Span<T> roundtrip)
{
  StorageUndoStep *us = get_init_step();
  if (!us) {
    return;
  }
  BLI_assert(src.size() == roundtrip.size());
  push_modified_common(&us->face_diffs, attr_name, type, src, [&](const int start, const int end) {
    for (int i = start; i < end; i++) {
      if (src[i] != roundtrip[i]) {
        return true;
      }
    }
    return false;
  });
}

template void push_face_modified<int>(const StringRef attr_name,
                                      const bke::AttrType type,
                                      const Span<int> src,
                                      const Span<int> roundtrip);
template void push_face_modified<bool>(const StringRef attr_name,
                                       const bke::AttrType type,
                                       const Span<bool> src,
                                       const Span<bool> roundtrip);

template<typename T>
void push_face_deleted(const StringRef attr_name,
                       const bke::AttrType attr_type,
                       const Span<T> attr)
{
  StorageUndoStep *us = get_init_step();
  if (!us) {
    return;
  }

  AttributeDiff diff;
  diff.name = std::string(attr_name);
  diff.kind = AttributeDiff::Kind::Deleted;
  diff.type = attr_type;
  diff.values = Vector<std::byte>(attr.cast<std::byte>());

  us->face_diffs.append(std::move(diff));
}

template void push_face_deleted<int>(const StringRef attr_name,
                                     const bke::AttrType attr_type,
                                     const Span<int> attr);
template void push_face_deleted<bool>(const StringRef attr_name,
                                      const bke::AttrType attr_type,
                                      const Span<bool> attr);

void push_face_added(const StringRef attr_name, const bke::AttrType type)
{
  StorageUndoStep *us = get_init_step();
  if (!us) {
    return;
  }

  AttributeDiff diff;
  diff.name = std::string(attr_name);
  diff.type = type;
  diff.kind = AttributeDiff::Kind::Added;

  us->face_diffs.append(std::move(diff));
}

template<typename T>
inline constexpr bool is_type_supported = std::is_same_v<T, float> || std::is_same_v<T, float3> ||
                                          std::is_same_v<T, ColorGeometry4f>;

static void swap_flip_span(const MutableSpan<float3> attr, AttributeDiff &diff)
{
  MutableSpan<float3> stored = diff.values.as_mutable_span().cast<float3>();
  threading::parallel_for(IndexRange(diff.blocks.size()), 64, [&](const IndexRange range) {
    for (const int i : range) {
      const int start = diff.blocks[i] * BLOCK_SIZE;
      const int block_size = math::min(BLOCK_SIZE, int(attr.size()) - start);
      const int offset = i * BLOCK_SIZE;
      for (const int j : IndexRange(block_size)) {
        std::swap(attr[start + j], stored[offset + j]);
      }
    }
  });
}

static void restore_swap_active_shapekey(Object &object, Mesh &mesh, AttributeDiff &diff)
{
  std::optional<ShapeKeyData> shape_data = ShapeKeyData::from_object(object);
  if (!shape_data.has_value() || shape_data->active_key_data.is_empty() || diff.blocks.is_empty())
  {
    return;
  }
  MutableSpan<float3> active = shape_data->active_key_data;
  const MutableSpan<float3> orig = shape_data->basis_key_active ? mesh.vert_positions_for_write() :
                                                                  MutableSpan<float3>();
  if (int64_t(diff.blocks.last()) * BLOCK_SIZE >= int64_t(active.size())) {
    BLI_assert_unreachable();
    return;
  }
  MutableSpan<float3> stored = diff.values.as_mutable_span().cast<float3>();
  threading::parallel_for(IndexRange(diff.blocks.size()), 64, [&](const IndexRange range) {
    for (const int i : range) {
      const int start = diff.blocks[i] * BLOCK_SIZE;
      const int block_size = math::min(BLOCK_SIZE, int(active.size()) - start);
      const int offset = i * BLOCK_SIZE;
      if (!orig.is_empty()) {
        orig.slice(start, block_size).copy_from(stored.slice(offset, block_size));
      }
      for (const int j : IndexRange(block_size)) {
        const float3 translation = stored[offset + j] - active[start + j];
        for (MutableSpan<float3> dep : shape_data->dependent_keys) {
          dep[start + j] += translation;
        }
        std::swap(active[start + j], stored[offset + j]);
      }
    }
  });
}

static void restore_swap_diffs(Object &object,
                               Mesh &mesh,
                               const int level,
                               const StorageTarget target,
                               Vector<AttributeDiff> &diffs)
{
  for (AttributeDiff &diff : diffs) {
    BLI_assert(diff.type == bke::AttrType::Float3);
    BLI_assert(diff.kind == AttributeDiff::Kind::Modified);
    const StringRef name(diff.name);
    if (name == swap_active_shapekey_name) {
      restore_swap_active_shapekey(object, mesh, diff);
      continue;
    }
    modify_float3_attribute(mesh, level, target, name, [&](const MutableSpan<float3> dst) {
      swap_flip_span(dst, diff);
    });
  }
}

void undo_step_impl(Object &object, StorageUndoStep &us, Depsgraph *depsgraph)
{
  if (is_layer_op(us.op_type)) {
    Mesh &mesh = *id_cast<Mesh *>(object.data);
    restore_layers(mesh, us.layers_before);

    switch (us.op_type) {
      case StorageUndoOpType::AddLayer: {
        if (!us.layers_after.layers.is_empty() && us.state_after.active_layer_index >= 0 &&
            us.state_after.active_layer_index < int(us.layers_after.layers.size()))
        {
          const SculptLayer &added = us.layers_after.layers[us.state_after.active_layer_index];
          remove_layer_attributes(mesh, bke::sculpt_layer::attr_name_from_layer_name(added.name));
        }
        update_after_restore(object, *depsgraph, false);
        break;
      }
      case StorageUndoOpType::DuplicateLayer: {
        if (!us.layers_after.layers.is_empty() && us.state_after.active_layer_index >= 0 &&
            us.state_after.active_layer_index < int(us.layers_after.layers.size()))
        {
          const SculptLayer &added = us.layers_after.layers[us.state_after.active_layer_index];
          remove_layer_attributes(mesh, bke::sculpt_layer::attr_name_from_layer_name(added.name));
        }
        update_after_restore(object, *depsgraph, true);
        break;
      }
      case StorageUndoOpType::DeleteLayer: {
        restore_deleted_layer_attributes(
            mesh, us.state_before.curr_level, us.curr_diffs, us.top_diffs);
        update_after_restore(object, *depsgraph, true);
        break;
      }
      case StorageUndoOpType::MoveLayer:
      case StorageUndoOpType::SetActiveLayer: {
        update_after_restore(object, *depsgraph, false);
        break;
      }
      case StorageUndoOpType::RenameLayer: {
        rename_changed_layer_attributes(mesh, us.layers_after, us.layers_before);
        update_after_restore(object, *depsgraph, false);
        break;
      }
      case StorageUndoOpType::MergeLayer: {
        restore_deleted_layer_attributes(
            mesh, us.state_before.curr_level, us.curr_diffs, us.top_diffs);
        restore_merged_attributes(
            mesh, us.state_before.curr_level, StorageTarget::Curr, us.curr_diffs);
        restore_merged_attributes(
            mesh, us.state_before.curr_level, StorageTarget::Top, us.top_diffs);
        update_after_restore(object, *depsgraph, true);
        break;
      }
      case StorageUndoOpType::MergeLayerToBase: {
        restore_deleted_layer_attributes(
            mesh, us.state_before.curr_level, us.curr_diffs, us.top_diffs);
        undo_merge_to_base(mesh, us);
        update_after_restore(object, *depsgraph, true);
        break;
      }
      default:
        BLI_assert_unreachable();
    }
    return;
  }

  if (is_morph_op(us.op_type)) {
    Mesh &mesh = *id_cast<Mesh *>(object.data);
    switch (us.op_type) {
      case StorageUndoOpType::MorphStore:
      case StorageUndoOpType::MorphStoreFromSubdividedBase: {
        for (const AttributeDiff &diff : us.curr_diffs) {
          if (diff.kind == AttributeDiff::Kind::Added) {
            remove_morph_attr_curr(mesh, us.state_before.curr_level, diff.name);
          }
        }
        break;
      }
      case StorageUndoOpType::MorphSwap: {
        restore_swap_diffs(
            object, mesh, us.state_before.curr_level, StorageTarget::Curr, us.curr_diffs);
        restore_swap_diffs(
            object, mesh, us.state_before.curr_level, StorageTarget::Top, us.top_diffs);
        update_after_restore(object, *depsgraph, true);
        break;
      }
      case StorageUndoOpType::MorphDelete: {
        restore_morph_deleted(mesh, us.curr_diffs);
        restore_morph_deleted(mesh, us.top_diffs);
        break;
      }
      default:
        BLI_assert_unreachable();
    }
    return;
  }

  Mesh &base_mesh = *id_cast<Mesh *>(object.data);
  Multiresolution &multires = *base_mesh.multires;

  switch (us.op_type) {
    case StorageUndoOpType::Subdivide: {
      const Array<int> subd_face_sets(multires.top_face_sets());
      const Array<bool> subd_hidden(multires.top_face_attribute<bool>(".hide_poly"));

      multires.top_level = us.state_before.top_level;
      multires.curr_level = us.state_before.top_level;
      multires.sculpt_level = us.state_before.sculpt_level;
      multires.clear_top_storage();
      multires.hidden_verts_free();

      if (multires.top_level > 0) {
        for (const AttributeDiff &diff : us.top_diffs) {
          BLI_assert(diff.kind == AttributeDiff::Kind::Deleted);

          bke::attribute_math::to_static_type(diff.type, [&]<typename T>() {
            if constexpr (is_type_supported<T>) {
              MutableSpan<T> attr = multires.attribute_for_write<T>(diff.name);
              attr.copy_from(diff.values.as_span().cast<T>());
            }
          });
        }

        bke::AttributeStorage &curr_storage = multires.curr_attribute_storage.wrap();
        bke::AttributeStorage &top_storage = multires.top_attribute_storage.wrap();
        {
          const StringRef name = morph::position_attr_name;
          const bke::Attribute *morph_attr = curr_storage.lookup(name);
          if (morph_attr) {
            if (morph_attr->data().size() == multires.top_verts_num() &&
                std::holds_alternative<bke::GList::ArrayData>(morph_attr->data().data()))
            {
              bke::GList stashed = morph_attr->data();
              curr_storage.remove(name);
              top_storage.remove(name);
              top_storage.add(std::string(name), bke::AttrDomain::Point, std::move(stashed));
            }
          }
        }

        if (!subd_face_sets.is_empty()) {
          multires.downsample_face_sets(base_mesh,
                                        us.state_after.top_level,
                                        subd_face_sets.as_span(),
                                        multires.top_level,
                                        multires.top_face_sets_for_write());
        }
        if (!subd_hidden.is_empty()) {
          multires.downsample_hidden_faces(
              base_mesh,
              us.state_after.top_level,
              subd_hidden.as_span(),
              multires.top_level,
              multires.top_face_attribute_for_write<bool>(".hide_poly"));
        }
      }
      DEG_id_tag_update(&object.id, ID_RECALC_GEOMETRY);
      break;
    }
    case StorageUndoOpType::SwitchLevel: {
      const int face_src_lvl = us.state_before.curr_level;
      const int face_dst_lvl = us.state_after.curr_level;

      Array<int> dst_face_sets;
      Array<bool> dst_hidden;
      if (face_dst_lvl == 0) {
        const bke::AttributeAccessor acc = base_mesh.attributes();
        dst_face_sets = Array<int>(
            VArraySpan<int>(*acc.lookup<int>(".sculpt_face_set", bke::AttrDomain::Face)));
        dst_hidden = Array<bool>(
            VArraySpan<bool>(*acc.lookup<bool>(".hide_poly", bke::AttrDomain::Face)));
      }
      else {
        dst_face_sets = Array<int>(multires.face_sets());
        dst_hidden = Array<bool>(multires.hidden_faces());
      }

      multires.curr_level = us.state_before.curr_level;
      multires.sculpt_level = us.state_before.sculpt_level;
      multires.hidden_verts_free();

      for (const AttributeDiff &diff : us.top_diffs) {
        BLI_assert(diff.kind == AttributeDiff::Kind::Modified);

        bke::attribute_math::to_static_type(diff.type, [&]<typename T>() {
          if constexpr (is_type_supported<T>) {
            MutableSpan<T> attr = multires.top_attribute_for_write<T>(diff.name);
            scatter_blocks(attr, diff.blocks, diff.values);
          }
        });
      }

      multires::clear_curr_vertex_attributes(base_mesh, multires);

      if (multires.curr_level < multires.top_level) {
        const bke::multiresolution::BaseTopology top_base = multires.get_base_topology(
            base_mesh, multires.top_level);
        const bke::multiresolution::Offsets top_offsets = Multiresolution::build_table_offsets(
            multires, top_base, multires.top_level);

        if (multires.curr_level == 0) {
          multires::downsample_to_base(base_mesh, top_offsets.verts);
        }
        else {
          multires::downsample_to_level(base_mesh, multires, top_offsets, multires.curr_level);
        }
        if (multires.runtime != nullptr) {
          multires.runtime->clear_curr_dirty();
        }

        bke::MutableAttributeAccessor attributes = base_mesh.attributes_for_write();

        for (const AttributeDiff &diff : us.curr_diffs) {
          BLI_assert(diff.kind == AttributeDiff::Kind::Modified);

          bke::attribute_math::to_static_type(diff.type, [&]<typename T>() {
            if constexpr (is_type_supported<T>) {
              if (multires.curr_level > 0) {
                MutableSpan<T> attr = multires.attribute_for_write<T>(diff.name);
                scatter_blocks(attr, diff.blocks, diff.values);
              }
              else {
                const StringRef base_name = (diff.name == "Color") ?
                                                StringRef(base_mesh.active_color_attribute) :
                                                StringRef(diff.name);
                bke::SpanAttributeWriter<T> attr_writer =
                    attributes.lookup_or_add_for_write_span<T>(base_name, bke::AttrDomain::Point);
                scatter_blocks(attr_writer.span, diff.blocks, diff.values);
                attr_writer.finish();
              }
            }
          });
        }
      }
      else if (multires.runtime != nullptr) {
        multires.runtime->clear_curr_dirty();
      }

      {
        const auto find_face_diff = [&](const StringRef name,
                                        const AttributeDiff::Kind kind) -> const AttributeDiff * {
          for (const AttributeDiff &diff : us.face_diffs) {
            if (StringRef(diff.name) == name && diff.kind == kind) {
              return &diff;
            }
          }
          return nullptr;
        };

        auto restore_face_attr = [&]<typename T>(const StringRef attr_name,
                                                 const Span<T> dst_snap) {
          const auto downsample = [&](const int from_lvl,
                                      const Span<T> from,
                                      const int to_lvl,
                                      const MutableSpan<T> to) {
            if constexpr (std::is_same_v<T, int>) {
              multires.downsample_face_sets(base_mesh, from_lvl, from, to_lvl, to);
            }
            else {
              multires.downsample_hidden_faces(base_mesh, from_lvl, from, to_lvl, to);
            }
          };
          const auto subdivide = [&](const int from_lvl,
                                     const Span<T> from,
                                     const int to_lvl,
                                     const MutableSpan<T> to) {
            if constexpr (std::is_same_v<T, int>) {
              multires.subdivide_face_sets(base_mesh, from_lvl, from, to_lvl, to);
            }
            else {
              multires.subdivide_hidden_faces(base_mesh, from_lvl, from, to_lvl, to);
            }
          };
          const auto remove_at = [&](const int lvl) {
            if (lvl == 0) {
              base_mesh.attributes_for_write().remove(attr_name);
            }
            else if (lvl == multires.top_level) {
              multires.top_attribute_storage.wrap().remove(attr_name);
            }
            else {
              multires.curr_attribute_storage.wrap().remove(attr_name);
            }
          };

          if (face_src_lvl > face_dst_lvl) {
            const AttributeDiff *modified = find_face_diff(attr_name,
                                                           AttributeDiff::Kind::Modified);
            if (!dst_snap.is_empty() || modified != nullptr) {
              remove_at(face_dst_lvl);
              remove_at(face_src_lvl);
              MutableSpan<T> dst = multires.face_attribute_for_write<T>(attr_name);
              subdivide(face_dst_lvl, dst_snap, face_src_lvl, dst);
              if (modified != nullptr) {
                scatter_blocks(dst, modified->blocks, modified->values);
              }
            }
          }
          else if (face_src_lvl < face_dst_lvl) {
            if (!dst_snap.is_empty()) {
              remove_at(face_dst_lvl);
              if (face_src_lvl > 0) {
                remove_at(face_src_lvl);
                MutableSpan<T> dst = multires.face_attribute_for_write<T>(attr_name);
                downsample(face_dst_lvl, dst_snap, face_src_lvl, dst);
              }
            }
          }

          if (const AttributeDiff *deleted = find_face_diff(attr_name,
                                                            AttributeDiff::Kind::Deleted);
              deleted != nullptr)
          {
            bke::SpanAttributeWriter<T> writer =
                base_mesh.attributes_for_write().ensure_for_write_span<T>(attr_name,
                                                                          bke::AttrDomain::Face);
            writer.span.copy_from(deleted->values.as_span().cast<T>());
            writer.finish();
          }
          else if (find_face_diff(attr_name, AttributeDiff::Kind::Added) != nullptr) {
            base_mesh.attributes_for_write().remove(attr_name);
          }
        };

        restore_face_attr(".sculpt_face_set", dst_face_sets.as_span());
        restore_face_attr(".hide_poly", dst_hidden.as_span());
      }

      DEG_id_tag_update(&object.id, ID_RECALC_GEOMETRY);
      break;
    }
    default:
      BLI_assert_unreachable();
  }
}

void redo_step_impl(Object &object, StorageUndoStep &us_ref, Depsgraph *depsgraph)
{
  StorageUndoStep *us = &us_ref;
  Mesh &mesh = *id_cast<Mesh *>(object.data);

  if (is_layer_op(us->op_type)) {
    switch (us->op_type) {
      case StorageUndoOpType::AddLayer: {
        restore_layers(mesh, us->layers_after);
        update_after_restore(object, *depsgraph, false);
        break;
      }
      case StorageUndoOpType::DeleteLayer: {
        bke::sculpt_layer::remove_active_layer(mesh);
        restore_layers(mesh, us->layers_after);
        update_after_restore(object, *depsgraph, true);
        break;
      }
      case StorageUndoOpType::MoveLayer:
      case StorageUndoOpType::SetActiveLayer: {
        restore_layers(mesh, us->layers_after);
        update_after_restore(object, *depsgraph, false);
        break;
      }
      case StorageUndoOpType::RenameLayer: {
        restore_layers(mesh, us->layers_after);
        rename_changed_layer_attributes(mesh, us->layers_before, us->layers_after);
        update_after_restore(object, *depsgraph, false);
        break;
      }
      case StorageUndoOpType::DuplicateLayer: {
        bke::sculpt_layer::duplicate_active_layer(mesh);
        restore_layers(mesh, us->layers_after);
        update_after_restore(object, *depsgraph, true);
        break;
      }
      case StorageUndoOpType::MergeLayer: {
        bke::sculpt_layer::merge_active_layer_down(mesh);
        restore_layers(mesh, us->layers_after);
        update_after_restore(object, *depsgraph, true);
        break;
      }
      case StorageUndoOpType::MergeLayerToBase: {
        bke::sculpt_layer::merge_active_layer_to_base(mesh);
        restore_layers(mesh, us->layers_after);
        update_after_restore(object, *depsgraph, true);
        break;
      }
      default:
        BLI_assert_unreachable();
    }
    return;
  }

  if (is_morph_op(us->op_type)) {
    switch (us->op_type) {
      case StorageUndoOpType::MorphStore: {
        morph::store_apply(object, *depsgraph, morph::MorphStoreMode::CurrentPositions);
        break;
      }
      case StorageUndoOpType::MorphStoreFromSubdividedBase: {
        morph::store_apply(object, *depsgraph, morph::MorphStoreMode::SubdividedBase);
        break;
      }
      case StorageUndoOpType::MorphSwap: {
        restore_swap_diffs(
            object, mesh, us->state_after.curr_level, StorageTarget::Curr, us->curr_diffs);
        restore_swap_diffs(
            object, mesh, us->state_after.curr_level, StorageTarget::Top, us->top_diffs);
        update_after_restore(object, *depsgraph, true);
        break;
      }
      case StorageUndoOpType::MorphDelete: {
        morph::delete_apply(object);
        break;
      }
      default:
        BLI_assert_unreachable();
    }
    return;
  }

  switch (us->op_type) {
    case StorageUndoOpType::Subdivide: {
      if (mesh.multires) {
        BLI_assert(mesh.multires->top_level == us->state_before.top_level);
      }
      else {
        BLI_assert(us->state_before.top_level == 0);
      }
      multires::subdivide(object);
      DEG_id_tag_update(&object.id, ID_RECALC_GEOMETRY);
      break;
    }
    case StorageUndoOpType::SwitchLevel: {
      BLI_assert(mesh.multires != nullptr);
      BLI_assert(mesh.multires->curr_level == us->state_before.curr_level);
      BLI_assert(mesh.multires->sculpt_level == us->state_before.sculpt_level);
      multires::switch_level(object, us->state_after.curr_level);
      DEG_id_tag_update(&object.id, ID_RECALC_GEOMETRY);
      break;
    }
    default:
      BLI_assert_unreachable();
  }
}

bool step_needs_layer_notifier(const undo::SculptUndoStep &us)
{
  return is_layer_op(us.op_type) || is_morph_op(us.op_type);
}

void free_storage_payload(undo::SculptUndoStep &us)
{
  us.top_diffs.~Vector();
  us.curr_diffs.~Vector();
  us.face_diffs.~Vector();
  us.layers_before.~LayerSnapshot();
  us.layers_after.~LayerSnapshot();
  us.object_name.~basic_string();
}

}  // namespace ed::sculpt_paint::storage_undo

}  // namespace blender
