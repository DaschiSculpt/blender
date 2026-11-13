/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "WM_api.hh"

#include "ED_undo.hh"

#include "sculpt_undo.hh"

#include "ED_sculpt.hh"

namespace blender {

struct Mesh;

namespace ed::sculpt_paint::storage_undo {

using StorageUndoStep = undo::SculptUndoStep;

void push_begin(Object &ob, const wmOperator *op, StorageUndoOpType type);
void push_begin_with_name(Object &ob, const char *name, StorageUndoOpType type);

bool is_active();

void push_attribute_added(const StorageTarget target,
                          const StringRef attr_name,
                          const bke::AttrType type);

template<typename T>
void push_attribute_deleted(const StorageTarget target,
                            const StringRef attr_name,
                            const bke::AttrType attr_type,
                            const Span<T> attr,
                            const StorageTarget origin = StorageTarget::Curr);

template<typename T>
bool push_attribute_modified(const StorageTarget target,
                             const StringRef attr_name,
                             const bke::AttrType type,
                             const Span<T> old_attr,
                             const Span<T> new_attr);

template<typename T>
bool push_attribute_modified_delta(const StorageTarget target,
                                   const StringRef attr_name,
                                   const bke::AttrType type,
                                   const Span<T> attr,
                                   const Span<T> delta);

template<typename T>
void push_face_modified(const StringRef attr_name,
                        const bke::AttrType type,
                        const Span<T> src,
                        const Span<T> roundtrip);

template<typename T>
void push_face_deleted(const StringRef attr_name,
                       const bke::AttrType attr_type,
                       const Span<T> attr);

void push_face_added(const StringRef attr_name, const bke::AttrType type);

void undo_step_impl(Object &object, undo::SculptUndoStep &us, Depsgraph *depsgraph);
void redo_step_impl(Object &object, undo::SculptUndoStep &us, Depsgraph *depsgraph);

bool step_needs_layer_notifier(const undo::SculptUndoStep &us);

inline const StringRef swap_active_shapekey_name = ".swap_active_shapekey";

}  // namespace ed::sculpt_paint::storage_undo

}  // namespace blender
