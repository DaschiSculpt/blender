/* SPDX-FileCopyrightText: 2026 Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_attribute.hh"

#include "BLI_index_mask.hh"
#include "BLI_span.hh"

namespace blender {

struct Depsgraph;
struct Object;

namespace bke::pbvh {
class Tree;
}

namespace bke {

namespace sculpt_layer {

std::string attr_name_from_layer_name(const StringRef layer_name);

std::string active_layer_attr_name(const Mesh &mesh);

bool mesh_has_visible_layers(const Mesh &base_mesh);

void apply_layer_displacements(const Mesh &mesh,
                               const Span<int> verts,
                               const MutableSpan<float3> positions);

void apply_layer_displacements(const Mesh &mesh,
                               const IndexRange range,
                               const MutableSpan<float3> positions);

void evaluate_deform(const Mesh &orig_mesh, Mesh &mesh_deform);

void evaluate_final(const Mesh &orig_mesh, Mesh &eval_mesh);

void calc_eval_positions(const Mesh &mesh, MutableSpan<float3> positions);

bool is_layers_only_eval(const Object &object);

bool can_use_fast_path_update(const Object &object);

bool update_eval_positions(Object &object, Depsgraph &depsgraph, const SculptLayer &changed_layer);

SculptLayer *active_layer(const Mesh &mesh);

Span<float3> layer_displacements(const Mesh &mesh, StringRef layer_name);

MutableSpan<float3> layer_displacements_for_write(Mesh &mesh, StringRef layer_name);

Span<float3> layer_top_displacements(const Mesh &mesh, StringRef layer_name);

MutableSpan<float3> layer_top_displacements_for_write(const Mesh &mesh, StringRef layer_name);

Span<float3> active_layer_displacements(const Mesh &mesh);

MutableSpan<float3> active_layer_displacements_for_write(Mesh &mesh);

void add_layer(Mesh &mesh, const StringRef layer_name);

void remove_active_layer(Mesh &mesh);

float effective_layer_weight(const SculptLayer &layer);

void combine_displacements(const float src_weight,
                           const float dst_weight,
                           const Span<float3> src,
                           const MutableSpan<float3> dst);

void apply_displacement(const Span<float3> disp,
                        const float weight,
                        const MutableSpan<float3> base);

void duplicate_active_layer(Mesh &mesh);

void merge_active_layer_to_base(Mesh &mesh);

void merge_active_layer_down(Mesh &mesh);

void rename_layer(Mesh &mesh, SculptLayer &layer, const StringRef new_name);

void clear_layer_non_zero_nodes_cache(Object &object);

}  // namespace sculpt_layer

}  // namespace bke

}  // namespace blender
