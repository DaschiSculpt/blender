/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_string_ref.hh"

struct Mesh;
struct Object;
struct Depsgraph;

namespace blender::ed::sculpt_paint::morph {

inline constexpr StringRef position_attr_name = ".sculpt_morph_position";

enum class MorphStoreMode : int {
  CurrentPositions = 0,
  SubdividedBase = 1,
};

Span<float3> base_morph_positions(const Mesh &mesh);

bool morph_exists(const Object &object);

void store_apply(Object &ob, const Depsgraph &depsgraph, MorphStoreMode source);
void swap_apply(Object &ob,
                const Depsgraph &depsgraph,
                bool use_mask = false,
                bool use_hide = false);
void delete_apply(Object &ob);

}  // namespace blender::ed::sculpt_paint::morph
