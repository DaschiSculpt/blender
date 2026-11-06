/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BLI_span.hh"

#include "DNA_object_types.h"

struct wmOperatorType;

namespace blender {

struct Mesh;
struct Multiresolution;

namespace bke::multiresolution {
struct OffsetsSpan;
}

namespace ed::sculpt_paint::multires {

void clear_curr_vertex_attributes(Mesh &base_mesh, Multiresolution &multires);

void downsample_to_base(Mesh &base_mesh, Span<int> vert_offsets);

void downsample_to_level(Mesh &base_mesh,
                         Multiresolution &multires,
                         bke::multiresolution::OffsetsSpan top_offsets,
                         int target_lvl);

void switch_level(Object &object, const int target_lvl);

void subdivide(Object &object);

void flush(Object &object);

void ensure_curr_storage(Object &object);

void SCULPT_OT_multires_set_sculpt_level(wmOperatorType *ot);
void SCULPT_OT_multires_set_view_level(wmOperatorType *ot);
void SCULPT_OT_multires_set_render_level(wmOperatorType *ot);
void SCULPT_OT_multires_switch_down(wmOperatorType *ot);
void SCULPT_OT_multires_switch_up(wmOperatorType *ot);
void SCULPT_OT_multires_subdiv(wmOperatorType *ot);
void SCULPT_OT_multires_delete_lower(wmOperatorType *ot);
void SCULPT_OT_multires_delete_higher(wmOperatorType *ot);
void SCULPT_OT_multires_bake(wmOperatorType *ot);

}  // namespace ed::sculpt_paint::multires

}  // namespace blender
