/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BLI_color_types.hh"

#include "DNA_object_types.h"

#include "BKE_attribute.hh"
#include "BKE_attribute_storage.hh"
#include "BKE_context.hh"
#include "BKE_mesh.hh"
#include "BKE_multiresolution.hh"
#include "BKE_object.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_report.hh"
#include "BKE_sculpt_layer.hh"

#include "DEG_depsgraph.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "ED_sculpt.hh"

#include "sculpt_intern.hh"
#include "sculpt_morph.hh"
#include "sculpt_storage_undo.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include <algorithm>
#include <optional>
#include <type_traits>

namespace blender::ed::sculpt_paint::multires {

static bool is_sculpt(const Object &ob)
{
  return (ob.mode & OB_MODE_SCULPT) && ob.runtime->sculpt_session;
}

static std::optional<bke::GList> grab_array_data(const bke::AttributeStorage &storage,
                                                 const StringRef name)
{
  const bke::Attribute *attr = storage.lookup(name);
  if (!attr) {
    return std::nullopt;
  }
  if (!std::holds_alternative<bke::GList::ArrayData>(attr->data().data())) {
    return std::nullopt;
  }
  return attr->data();
}

template<typename T> static Span<T> get_array_span(const bke::GList &list)
{
  const auto &arr = std::get<bke::GList::ArrayData>(list.data());
  return Span<T>(static_cast<const T *>(arr.data), list.size());
}

static bke::multiresolution::Offsets steal_or_build_offsets(SculptSession *ss,
                                                            const Mesh &base_mesh,
                                                            const Multiresolution &multires,
                                                            const int lvl)
{
  if (ss && ss->pbvh) {
    bke::multiresolution::Offsets offsets;
    offsets.verts = std::move(ss->pbvh->offsets.verts);
    offsets.edges = std::move(ss->pbvh->offsets.edges);
    offsets.faces = std::move(ss->pbvh->offsets.faces);
    return offsets;
  }
  const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(base_mesh, lvl);
  return Multiresolution::build_table_offsets(multires, base_topo, lvl);
}

static bke::multiresolution::Offsets store_bvh_partition_for_level(
    Multiresolution &multires,
    const int lvl,
    const Mesh &base_mesh,
    const bke::multiresolution::BaseTopology &base_topo)
{
  const int base_lvl = multires.base_level(lvl);

  Array<float3> positions_storage;
  const Span<float3> positions = Multiresolution::calc_smooth_positions(
      base_mesh, base_lvl, positions_storage);
  Array<int> material_storage;
  const Span<int> material_indices = Multiresolution::calc_base_materials(
      base_mesh, base_lvl, material_storage);

  bke::pbvh::Tree pbvh = bke::pbvh::Tree::from_multires(
      lvl - base_lvl, base_topo, positions, material_indices);

  const Span<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();

  multires.bvh_partition_ensure(lvl, pbvh.face_indices.size(), nodes.size());
  multires.bvh_face_indices_for_write(lvl).copy_from(pbvh.face_indices);
  bke::pbvh::Tree::store_nodes_multires(nodes, multires.bvh_face_counts_for_write(lvl));

  bke::multiresolution::Offsets offsets;
  offsets.verts = std::move(pbvh.offsets.verts);
  offsets.edges = std::move(pbvh.offsets.edges);
  offsets.faces = std::move(pbvh.offsets.faces);
  return offsets;
}

template<typename T>
static void apply_delta(const MutableSpan<T> attr, const Span<T> delta, const bool clamp)
{
  threading::parallel_for(IndexRange(attr.size()), 8192, [&](const IndexRange range) {
    if (clamp) {
      for (const int i : range) {
        attr[i] = math::clamp(attr[i] + delta[i], 0.0f, 1.0f);
      }
    }
    else {
      for (const int i : range) {
        attr[i] += delta[i];
      }
    }
  });
}

template<typename T>
static void propagate_to_top(const Mesh &base_mesh,
                             const Multiresolution &multires,
                             const int curr_lvl,
                             const int top_lvl,
                             const bke::multiresolution::OffsetsSpan &curr_offsets,
                             const bke::multiresolution::OffsetsSpan &top_offsets,
                             const StringRef name,
                             const bke::AttrType type,
                             const Span<T> curr_attr,
                             const MutableSpan<T> top_attr,
                             const bool clamp)
{
  Array<T> curr_delta(curr_attr.size());
  if (curr_lvl == 0) {
    Multiresolution::downsample_to_base<T>(
        top_offsets.verts, top_attr, curr_delta.as_mutable_span());
  }
  else {
    multires.downsample_to<T>(base_mesh,
                              top_lvl,
                              top_offsets,
                              top_attr,
                              curr_lvl,
                              curr_offsets,
                              curr_delta.as_mutable_span());
  }

  threading::parallel_for(IndexRange(curr_attr.size()), 8192, [&](const IndexRange range) {
    for (const int i : range) {
      curr_delta[i] = curr_attr[i] - curr_delta[i];
    }
  });

  const bool curr_dirty = storage_undo::push_attribute_modified_delta(
      storage_undo::StorageTarget::Curr, name, type, curr_attr, curr_delta.as_span());
  if (!curr_dirty) {
    return;
  }

  Array<T> top_delta(top_attr.size());
  if (curr_lvl == 0) {
    multires.subdivide_base_to(
        base_mesh, top_lvl, top_offsets, curr_delta.as_span(), top_delta.as_mutable_span(), false);
  }
  else {
    multires.subdivide_to(base_mesh,
                          curr_lvl,
                          curr_offsets,
                          curr_delta.as_span(),
                          top_lvl,
                          top_offsets,
                          top_delta.as_mutable_span(),
                          false);
  }

  storage_undo::push_attribute_modified_delta(
      storage_undo::StorageTarget::Top, name, type, top_attr.as_span(), top_delta.as_span());

  apply_delta(top_attr, top_delta.as_span(), clamp);
}

void propagate_from_base(const Mesh &base_mesh,
                         const bke::multiresolution::OffsetsSpan &top_offsets)
{
  Multiresolution &multires = *base_mesh.multires;
  bke::AttributeAccessor attributes = base_mesh.attributes();
  const bke::multiresolution::OffsetsSpan empty_offsets{bke::multiresolution::Offsets()};

  propagate_to_top<float3>(base_mesh,
                           multires,
                           0,
                           multires.top_level,
                           empty_offsets,
                           top_offsets,
                           "position",
                           bke::AttrType::Float3,
                           base_mesh.vert_positions(),
                           multires.top_base_positions_for_write(),
                           false);

  const VArraySpan<float> masks = *attributes.lookup<float>(".sculpt_mask",
                                                            bke::AttrDomain::Point);
  if (!masks.is_empty()) {
    propagate_to_top<float>(base_mesh,
                            multires,
                            0,
                            multires.top_level,
                            empty_offsets,
                            top_offsets,
                            ".sculpt_mask",
                            bke::AttrType::Float,
                            masks,
                            multires.top_masks_for_write(),
                            true);
  }

  const VArraySpan<ColorGeometry4f> base_colors = *attributes.lookup<ColorGeometry4f>(
      base_mesh.active_color_attribute, bke::AttrDomain::Point);
  if (!base_colors.is_empty()) {
    propagate_to_top<float4>(base_mesh,
                             multires,
                             0,
                             multires.top_level,
                             empty_offsets,
                             top_offsets,
                             "Color",
                             bke::AttrType::ColorFloat,
                             base_colors.cast<float4>(),
                             multires.top_colors_for_write(),
                             true);
  }

  for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
    const std::string attr_name = bke::sculpt_layer::attr_name_from_layer_name(layer.name);
    const VArraySpan<float3> displacements = *attributes.lookup<float3>(attr_name,
                                                                        bke::AttrDomain::Point);
    if (!displacements.is_empty()) {
      MutableSpan<float3> top_displacements = bke::sculpt_layer::layer_top_displacements_for_write(
          base_mesh, layer.name);
      propagate_to_top<float3>(base_mesh,
                               multires,
                               0,
                               multires.top_level,
                               empty_offsets,
                               top_offsets,
                               attr_name,
                               bke::AttrType::Float3,
                               displacements,
                               top_displacements,
                               false);
    }
  }
}

static void propagate_curr_to_top(const Mesh &base_mesh,
                                  Multiresolution &multires,
                                  const int curr_lvl,
                                  const bke::multiresolution::OffsetsSpan &curr_offsets,
                                  const bke::multiresolution::OffsetsSpan &top_offsets)
{
  const bool pos_dirty = multires.runtime == nullptr || multires.runtime->curr_positions_dirty;
  const bool mask_dirty = multires.runtime == nullptr || multires.runtime->curr_masks_dirty;
  const bool color_dirty = multires.runtime == nullptr || multires.runtime->curr_colors_dirty;

  if (pos_dirty) {
    propagate_to_top<float3>(base_mesh,
                             multires,
                             curr_lvl,
                             multires.top_level,
                             curr_offsets,
                             top_offsets,
                             "position",
                             bke::AttrType::Float3,
                             multires.base_positions(),
                             multires.top_base_positions_for_write(),
                             false);
  }

  if (!multires.masks().is_empty() && mask_dirty) {
    propagate_to_top<float>(base_mesh,
                            multires,
                            curr_lvl,
                            multires.top_level,
                            curr_offsets,
                            top_offsets,
                            ".sculpt_mask",
                            bke::AttrType::Float,
                            multires.masks(),
                            multires.top_masks_for_write(),
                            true);
  }

  if (!multires.colors().is_empty() && color_dirty) {
    propagate_to_top<float4>(base_mesh,
                             multires,
                             curr_lvl,
                             multires.top_level,
                             curr_offsets,
                             top_offsets,
                             "Color",
                             bke::AttrType::ColorFloat,
                             multires.colors(),
                             multires.top_colors_for_write(),
                             true);
  }

  for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
    const Span<float3> displacements = bke::sculpt_layer::layer_displacements(base_mesh,
                                                                              layer.name);
    if (displacements.is_empty()) {
      continue;
    }
    const std::string attr_name = bke::sculpt_layer::attr_name_from_layer_name(layer.name);
    if (multires.runtime != nullptr && !multires.runtime->is_curr_layer_dirty(attr_name)) {
      continue;
    }
    MutableSpan<float3> top_displacements = bke::sculpt_layer::layer_top_displacements_for_write(
        base_mesh, layer.name);
    propagate_to_top<float3>(base_mesh,
                             multires,
                             curr_lvl,
                             multires.top_level,
                             curr_offsets,
                             top_offsets,
                             attr_name,
                             bke::AttrType::Float3,
                             displacements,
                             top_displacements,
                             false);
  }
}

void clear_curr_vertex_attributes(Mesh &base_mesh, Multiresolution &multires)
{
  bke::AttributeStorage &curr_storage = multires.curr_attribute_storage.wrap();
  curr_storage.remove("position");
  curr_storage.remove(".sculpt_mask");
  curr_storage.remove("Color");
  curr_storage.remove(".hide_vert");

  for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
    curr_storage.remove(bke::sculpt_layer::attr_name_from_layer_name(layer.name));
  }
}

void downsample_to_base(Mesh &base_mesh, const Span<int> vert_offsets)
{
  bke::MutableAttributeAccessor attributes = base_mesh.attributes_for_write();

  Multiresolution &multires = *base_mesh.multires;
  Multiresolution::downsample_to_base<float3>(
      vert_offsets, multires.top_base_positions(), base_mesh.vert_positions_for_write());

  const Span<float> top_masks = multires.top_masks();
  if (!top_masks.is_empty()) {
    bke::SpanAttributeWriter<float> base_masks = attributes.ensure_for_write_span<float>(
        ".sculpt_mask", bke::AttrDomain::Point);
    Multiresolution::downsample_to_base<float>(vert_offsets, top_masks, base_masks.span);
    base_masks.finish();
  }

  const Span<float4> top_colors = multires.top_colors();
  if (!top_colors.is_empty()) {
    bke::SpanAttributeWriter<ColorGeometry4f> base_colors =
        attributes.lookup_or_add_for_write_span<ColorGeometry4f>(base_mesh.active_color_attribute,
                                                                 bke::AttrDomain::Point);
    Multiresolution::downsample_to_base<float4>(
        vert_offsets, top_colors, base_colors.span.cast<float4>());
    base_colors.finish();
  }

  for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
    const Span<float3> top_displacements = bke::sculpt_layer::layer_top_displacements(base_mesh,
                                                                                      layer.name);
    if (!top_displacements.is_empty()) {
      const std::string attr_name = bke::sculpt_layer::attr_name_from_layer_name(layer.name);
      bke::SpanAttributeWriter<float3> displacements =
          attributes.lookup_or_add_for_write_span<float3>(attr_name, bke::AttrDomain::Point);
      Multiresolution::downsample_to_base<float3>(
          vert_offsets, top_displacements, displacements.span);
      displacements.finish();
    }
  }

  base_mesh.tag_positions_changed();
}

void downsample_to_level(Mesh &base_mesh,
                         Multiresolution &multires,
                         const bke::multiresolution::OffsetsSpan top_offsets,
                         const int target_lvl)
{
  BLI_assert(target_lvl > 0);
  BLI_assert(!multires.top_base_positions().is_empty());

  const bke::multiresolution::BaseTopology target_base = multires.get_base_topology(base_mesh,
                                                                                    target_lvl);
  bke::multiresolution::Offsets target_offsets = Multiresolution::build_table_offsets(
      multires, target_base, target_lvl);

  multires.downsample_to<float3>(base_mesh,
                                 multires.top_level,
                                 top_offsets,
                                 multires.top_base_positions(),
                                 target_lvl,
                                 target_offsets,
                                 multires.base_positions_for_write());

  if (!multires.top_masks().is_empty()) {
    multires.downsample_to<float>(base_mesh,
                                  multires.top_level,
                                  top_offsets,
                                  multires.top_masks(),
                                  target_lvl,
                                  target_offsets,
                                  multires.masks_for_write());
  }

  if (!multires.top_colors().is_empty()) {
    multires.downsample_to<float4>(base_mesh,
                                   multires.top_level,
                                   top_offsets,
                                   multires.top_colors(),
                                   target_lvl,
                                   target_offsets,
                                   multires.colors_for_write());
  }

  for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
    const Span<float3> top_displacements = bke::sculpt_layer::layer_top_displacements(base_mesh,
                                                                                      layer.name);
    if (!top_displacements.is_empty()) {
      MutableSpan<float3> displacements = bke::sculpt_layer::layer_displacements_for_write(
          base_mesh, layer.name);
      multires.downsample_to<float3>(base_mesh,
                                     multires.top_level,
                                     top_offsets,
                                     top_displacements,
                                     target_lvl,
                                     target_offsets,
                                     displacements);
    }
  }
}

template<typename T>
static void downsample_face_attr(const Multiresolution &multires,
                                 const Mesh &base_mesh,
                                 const int from_lvl,
                                 const Span<T> src,
                                 const int to_lvl,
                                 const MutableSpan<T> dst)
{
  if constexpr (std::is_same_v<T, int>) {
    multires.downsample_face_sets(base_mesh, from_lvl, src, to_lvl, dst);
  }
  else {
    multires.downsample_hidden_faces(base_mesh, from_lvl, src, to_lvl, dst);
  }
}

template<typename T>
static void subdivide_face_attr(const Multiresolution &multires,
                                const Mesh &base_mesh,
                                const int from_lvl,
                                const Span<T> src,
                                const int to_lvl,
                                const MutableSpan<T> dst)
{
  if constexpr (std::is_same_v<T, int>) {
    multires.subdivide_face_sets(base_mesh, from_lvl, src, to_lvl, dst);
  }
  else {
    multires.subdivide_hidden_faces(base_mesh, from_lvl, src, to_lvl, dst);
  }
}

template<typename T>
static void backup_base_face_attribute(Mesh &base_mesh,
                                       const StringRef attr_name,
                                       const bke::AttrType attr_type)
{
  if (!storage_undo::is_active()) {
    return;
  }
  const bke::AttributeAccessor attributes = base_mesh.attributes();
  const VArraySpan<T> old = *attributes.lookup<T>(attr_name, bke::AttrDomain::Face);
  if (old.is_empty()) {
    storage_undo::push_face_added(attr_name, attr_type);
  }
  else {
    storage_undo::push_face_deleted(attr_name, attr_type, old);
  }
}

template<typename T>
static void switch_face_attribute(Mesh &base_mesh,
                                  Multiresolution &multires,
                                  const int src_lvl,
                                  const int target_lvl,
                                  const StringRef attr_name,
                                  const bke::AttrType attr_type)
{
  VArraySpan<T> base_src_holder;
  std::optional<bke::GList> storage_src_holder;
  Span<T> src;

  if (src_lvl == 0) {
    base_src_holder = *base_mesh.attributes().lookup<T>(attr_name, bke::AttrDomain::Face);
    src = base_src_holder;
  }
  else {
    bke::AttributeStorage &src_storage = (src_lvl == multires.top_level) ?
                                             multires.top_attribute_storage.wrap() :
                                             multires.curr_attribute_storage.wrap();
    storage_src_holder = grab_array_data(src_storage, attr_name);
    if (storage_src_holder) {
      src = get_array_span<T>(*storage_src_holder);
      src_storage.remove(attr_name);
    }
  }

  if (src.is_empty()) {
    if (target_lvl == 0) {
      backup_base_face_attribute<T>(base_mesh, attr_name, attr_type);
      base_mesh.attributes_for_write().remove(attr_name);
    }
    else if (target_lvl == multires.top_level) {
      multires.top_attribute_storage.wrap().remove(attr_name);
    }
    else {
      multires.curr_attribute_storage.wrap().remove(attr_name);
    }
    return;
  }

  auto downsample_with_undo = [&](const MutableSpan<T> dst) {
    downsample_face_attr<T>(multires, base_mesh, src_lvl, src, target_lvl, dst);
    if (storage_undo::is_active()) {
      Array<T> roundtrip(src.size());
      subdivide_face_attr<T>(
          multires, base_mesh, target_lvl, dst, src_lvl, roundtrip.as_mutable_span());
      storage_undo::push_face_modified(attr_name, attr_type, src, roundtrip.as_span());
    }
  };

  if (target_lvl == 0) {
    backup_base_face_attribute<T>(base_mesh, attr_name, attr_type);
    bke::MutableAttributeAccessor attributes = base_mesh.attributes_for_write();
    bke::SpanAttributeWriter<T> dst = attributes.ensure_for_write_span<T>(attr_name,
                                                                          bke::AttrDomain::Face);
    downsample_with_undo(dst.span);
    dst.finish();
    return;
  }

  if (target_lvl == multires.top_level) {
    multires.top_attribute_storage.wrap().remove(attr_name);
    MutableSpan<T> dst = multires.top_face_attribute_for_write<T>(attr_name);
    subdivide_face_attr<T>(multires, base_mesh, src_lvl, src, target_lvl, dst);
    return;
  }

  multires.curr_attribute_storage.wrap().remove(attr_name);
  MutableSpan<T> dst = multires.face_attribute_for_write<T>(attr_name);
  if (target_lvl < src_lvl) {
    downsample_with_undo(dst);
  }
  else {
    subdivide_face_attr<T>(multires, base_mesh, src_lvl, src, target_lvl, dst);
  }
}

void switch_level(Object &ob, const int target_lvl)
{
  SculptSession *ss = ob.runtime->sculpt_session;
  Mesh &base_mesh = *id_cast<Mesh *>(ob.data);
  if (!bke::multiresolution::base_topology_matches(base_mesh)) {
    return;
  }
  Multiresolution &multires = *base_mesh.multires;
  multires.hidden_verts_free();

  BLI_assert(target_lvl != multires.curr_level);

  const int src_lvl = multires.curr_level;
  const bool from_top = (src_lvl == multires.top_level);
  const bool to_top = (target_lvl == multires.top_level);
  const bool from_base = (src_lvl == 0);

  bke::multiresolution::Offsets top_offsets;

  if (from_top) {
    top_offsets = steal_or_build_offsets(ss, base_mesh, multires, multires.top_level);
  }
  else {
    const bke::multiresolution::BaseTopology top_base = multires.get_base_topology(
        base_mesh, multires.top_level);
    top_offsets = Multiresolution::build_table_offsets(multires, top_base, multires.top_level);

    if (from_base) {
      propagate_from_base(base_mesh, top_offsets);
    }
    else {
      const bke::multiresolution::Offsets curr_offsets = steal_or_build_offsets(
          ss, base_mesh, multires, src_lvl);
      propagate_curr_to_top(base_mesh, multires, src_lvl, curr_offsets, top_offsets);
    }
  }

  BKE_sculptsession_free_pbvh(ob);

  if (to_top) {
    multires.curr_level = multires.top_level;
    multires.sculpt_level = multires.top_level;
  }
  else {
    multires.curr_level = target_lvl;
    multires.sculpt_level = target_lvl;
  }
  clear_curr_vertex_attributes(base_mesh, multires);

  if (!to_top) {
    if (target_lvl == 0) {
      downsample_to_base(base_mesh, top_offsets.verts);
    }
    else {
      downsample_to_level(base_mesh, multires, top_offsets, target_lvl);
    }
  }

  if (multires.runtime != nullptr) {
    multires.runtime->clear_curr_dirty();
  }

  switch_face_attribute<int>(
      base_mesh, multires, src_lvl, target_lvl, ".sculpt_face_set", bke::AttrType::Int32);
  switch_face_attribute<bool>(
      base_mesh, multires, src_lvl, target_lvl, ".hide_poly", bke::AttrType::Bool);
}

void flush(Object &object)
{
  Mesh &base_mesh = *id_cast<Mesh *>(object.data);
  if (!base_mesh.multires) {
    return;
  }
  if (!bke::multiresolution::base_topology_matches(base_mesh)) {
    return;
  }

  Multiresolution &multires = *base_mesh.multires;
  multires.hidden_verts_free();

  if (multires.curr_level == 0) {
    multires.clear_sculpt_data();
    return;
  }

  const int src_lvl = multires.curr_level;

  const bke::multiresolution::BaseTopology top_base = multires.get_base_topology(
      base_mesh, multires.top_level);
  bke::multiresolution::Offsets top_offsets = Multiresolution::build_table_offsets(
      multires, top_base, multires.top_level);

  if (src_lvl < multires.top_level && !multires.base_positions().is_empty()) {
    const bke::multiresolution::BaseTopology curr_base = multires.get_base_topology(base_mesh,
                                                                                    src_lvl);
    bke::multiresolution::Offsets curr_offsets = Multiresolution::build_table_offsets(
        multires, curr_base, src_lvl);
    propagate_curr_to_top(base_mesh, multires, src_lvl, curr_offsets, top_offsets);
  }

  downsample_to_base(base_mesh, top_offsets.verts);
  clear_curr_vertex_attributes(base_mesh, multires);
  if (multires.runtime != nullptr) {
    multires.runtime->clear_curr_dirty();
  }

  if (!multires.face_sets().is_empty()) {
    switch_face_attribute<int>(
        base_mesh, multires, src_lvl, 0, ".sculpt_face_set", bke::AttrType::Int32);
  }
  if (!multires.hidden_faces().is_empty()) {
    switch_face_attribute<bool>(
        base_mesh, multires, src_lvl, 0, ".hide_poly", bke::AttrType::Bool);
  }

  multires.clear_sculpt_data();
}

void ensure_curr_storage(Object &object)
{
  Mesh &base_mesh = *id_cast<Mesh *>(object.data);
  if (!base_mesh.multires) {
    return;
  }
  if (!bke::multiresolution::base_topology_matches(base_mesh)) {
    return;
  }
  Multiresolution &multires = *base_mesh.multires;

  const int curr_lvl = multires.curr_level;
  if (curr_lvl == 0) {
    return;
  }
  if (multires.top_base_positions().is_empty()) {
    return;
  }
  if (curr_lvl < multires.top_level && !multires.base_positions().is_empty()) {
    return;
  }

  const bke::multiresolution::BaseTopology top_base = multires.get_base_topology(
      base_mesh, multires.top_level);
  bke::multiresolution::Offsets top_offsets = Multiresolution::build_table_offsets(
      multires, top_base, multires.top_level);

  propagate_from_base(base_mesh, top_offsets);

  const bke::AttributeAccessor attributes = base_mesh.attributes();
  const VArraySpan<int> base_face_sets = *attributes.lookup<int>(".sculpt_face_set",
                                                                 bke::AttrDomain::Face);
  if (multires.face_sets().is_empty() && !base_face_sets.is_empty()) {
    multires.subdivide_face_sets(
        base_mesh, 0, base_face_sets, curr_lvl, multires.face_sets_for_write());
  }

  const VArraySpan<bool> base_hidden_faces = *attributes.lookup<bool>(".hide_poly",
                                                                      bke::AttrDomain::Face);
  if (!base_hidden_faces.is_empty()) {
    multires.subdivide_hidden_faces(
        base_mesh, 0, base_hidden_faces, curr_lvl, multires.hidden_faces_for_write());
  }

  if (curr_lvl < multires.top_level) {
    downsample_to_level(base_mesh, multires, top_offsets, curr_lvl);
  }

  if (multires.runtime != nullptr) {
    multires.runtime->clear_curr_dirty();
  }
}

void subdivide(Object &object)
{
  Mesh &base_mesh = *id_cast<Mesh *>(object.data);
  const bke::AttributeAccessor attributes = base_mesh.attributes();

  const bool in_sculpt = is_sculpt(object);

  if (!base_mesh.multires) {
    base_mesh.multires = BKE_multiresolution_create(base_mesh);
  }
  else if (!bke::multiresolution::base_topology_matches(base_mesh)) {
    return;
  }

  Multiresolution &multires = *base_mesh.multires;
  const int from_lvl = multires.top_level;
  const int target_lvl = multires.top_level + 1;
  const bool linear = base_mesh.multires_use_linear != 0;

  const bke::multiresolution::BaseTopology target_base = multires.get_base_topology(base_mesh,
                                                                                    target_lvl);
  const bke::multiresolution::Offsets target_offsets =
      (target_lvl > multires.bvh_partitions_len) ?
          store_bvh_partition_for_level(multires, target_lvl, base_mesh, target_base) :
          Multiresolution::build_table_offsets(multires, target_base, target_lvl);

  if (target_lvl == 1) {
    multires.top_level = 1;
    multires.subdivide_base_to(base_mesh,
                               target_lvl,
                               target_offsets,
                               base_mesh.vert_positions(),
                               multires.top_base_positions_for_write(),
                               linear);

    const VArraySpan base_mask = *attributes.lookup<float>(".sculpt_mask", bke::AttrDomain::Point);
    if (!base_mask.is_empty()) {
      multires.subdivide_base_to(base_mesh,
                                 target_lvl,
                                 target_offsets,
                                 base_mask,
                                 multires.top_masks_for_write(),
                                 linear);
    }

    const VArraySpan<ColorGeometry4f> base_color = *attributes.lookup<ColorGeometry4f>(
        base_mesh.active_color_attribute, bke::AttrDomain::Point);
    if (!base_color.is_empty()) {
      multires.subdivide_base_to(base_mesh,
                                 target_lvl,
                                 target_offsets,
                                 base_color.cast<float4>(),
                                 multires.top_colors_for_write(),
                                 linear);
    }

    for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
      const Span<float3> displacements = bke::sculpt_layer::layer_displacements(base_mesh,
                                                                                layer.name);
      if (!displacements.is_empty()) {
        MutableSpan<float3> top_displacements =
            bke::sculpt_layer::layer_top_displacements_for_write(base_mesh, layer.name);
        multires.subdivide_base_to(
            base_mesh, target_lvl, target_offsets, displacements, top_displacements, linear);
      }
    }

    const VArraySpan<int> face_set = *attributes.lookup<int>(".sculpt_face_set",
                                                             bke::AttrDomain::Face);
    if (!face_set.is_empty()) {
      multires.subdivide_face_sets(base_mesh, 0, face_set, 1, multires.top_face_sets_for_write());
    }

    const VArraySpan<bool> base_hidden = *attributes.lookup<bool>(".hide_poly",
                                                                  bke::AttrDomain::Face);
    if (!base_hidden.is_empty()) {
      multires.subdivide_hidden_faces(
          base_mesh, 0, base_hidden, 1, multires.top_face_attribute_for_write<bool>(".hide_poly"));
    }

    multires.sculpt_level = 1;
    multires.curr_level = in_sculpt ? 1 : 0;
  }
  else {
    multires.hidden_verts_free();

    const bke::multiresolution::Offsets curr_offsets = steal_or_build_offsets(
        in_sculpt ? object.runtime->sculpt_session : nullptr, base_mesh, multires, from_lvl);

    bke::AttributeStorage &top_storage = multires.top_attribute_storage.wrap();

    if (std::optional<bke::GList> morph_data = grab_array_data(top_storage,
                                                               morph::position_attr_name))
    {
      top_storage.remove(morph::position_attr_name);
      bke::AttributeStorage &curr_storage = multires.curr_attribute_storage.wrap();
      curr_storage.remove(morph::position_attr_name);
      curr_storage.add(
          std::string(morph::position_attr_name), bke::AttrDomain::Point, std::move(*morph_data));
    }

    multires.top_level = target_lvl;
    multires.sculpt_level = target_lvl;
    multires.curr_level = in_sculpt ? target_lvl : 0;

    auto subdivide_top_vert_attr = [&]<typename T, typename GetDstFn>(
                                       const StringRef name,
                                       const bke::AttrType attr_type,
                                       GetDstFn &&get_dst) {
      std::optional<bke::GList> data = grab_array_data(top_storage, name);
      if (!data) {
        return;
      }
      const Span<T> src = get_array_span<T>(*data);
      top_storage.remove(name);
      storage_undo::push_attribute_deleted(storage_undo::StorageTarget::Top, name, attr_type, src);
      if (!src.is_empty()) {
        multires.subdivide_to(
            base_mesh, from_lvl, curr_offsets, src, target_lvl, target_offsets, get_dst(), linear);
      }
    };

    auto subdivide_top_face_attr = [&]<typename T>(const StringRef name) {
      std::optional<bke::GList> data = grab_array_data(top_storage, name);
      if (!data) {
        return;
      }
      const Span<T> src = get_array_span<T>(*data);
      top_storage.remove(name);
      if (!src.is_empty()) {
        subdivide_face_attr<T>(multires,
                               base_mesh,
                               from_lvl,
                               src,
                               target_lvl,
                               multires.top_face_attribute_for_write<T>(name));
      }
    };

    subdivide_top_vert_attr.template operator()<float3>("position", bke::AttrType::Float3, [&]() {
      return multires.top_base_positions_for_write();
    });
    subdivide_top_vert_attr.template operator()<float>(
        ".sculpt_mask", bke::AttrType::Float, [&]() { return multires.top_masks_for_write(); });
    subdivide_top_vert_attr.template operator()<float4>(
        "Color", bke::AttrType::ColorFloat, [&]() { return multires.top_colors_for_write(); });

    for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
      const std::string attr_name = bke::sculpt_layer::attr_name_from_layer_name(layer.name);
      subdivide_top_vert_attr.template operator()<float3>(attr_name, bke::AttrType::Float3, [&]() {
        return bke::sculpt_layer::layer_top_displacements_for_write(base_mesh, layer.name);
      });
    }

    subdivide_top_face_attr.template operator()<int>(".sculpt_face_set");
    subdivide_top_face_attr.template operator()<bool>(".hide_poly");
  }

  if (multires.runtime != nullptr) {
    multires.runtime->clear_curr_dirty();
  }

  BKE_sculptsession_free_pbvh(object);
}

static void delete_lower(Object &object, const int target_lvl)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  if (!bke::multiresolution::base_topology_matches(mesh)) {
    return;
  }
  Multiresolution &old_multires = *mesh.multires;
  BLI_assert(target_lvl > 0 && target_lvl <= old_multires.top_level);

  Mesh &new_base = ((object.mode & OB_MODE_ALL_SCULPT) != 0) ?
                       *Multiresolution::eval_to_mesh_in_sculpt(mesh, target_lvl) :
                       *Multiresolution::eval_to_mesh(mesh, target_lvl);
  Multiresolution *new_multires = nullptr;

  if (target_lvl < old_multires.top_level) {
    new_base.multires = BKE_multiresolution_create(new_base);
    new_multires = new_base.multires;

    const int old_top_lvl = old_multires.top_level;
    const int new_top_lvl = old_top_lvl - target_lvl;

    new_multires->top_level = new_top_lvl;
    new_multires->sculpt_level = is_sculpt(object) ?
                                     0 :
                                     std::max(0, old_multires.sculpt_level - target_lvl);
    new_multires->curr_level = 0;
    new_multires->view_level = std::max(0, old_multires.view_level - target_lvl);
    new_multires->render_level = std::max(0, old_multires.render_level - target_lvl);

    bke::multiresolution::Offsets new_top_offsets;
    for (int lvl = 1; lvl <= new_top_lvl; lvl++) {
      const bke::multiresolution::BaseTopology lvl_base = new_multires->get_base_topology(new_base,
                                                                                          lvl);
      bke::multiresolution::Offsets offsets = store_bvh_partition_for_level(
          *new_multires, lvl, new_base, lvl_base);
      if (lvl == new_top_lvl) {
        new_top_offsets = std::move(offsets);
      }
    }

    const bke::multiresolution::BaseTopology old_top_base = old_multires.get_base_topology(
        mesh, old_top_lvl);
    const bke::multiresolution::Offsets old_top_offsets = Multiresolution::build_table_offsets(
        old_multires, old_top_base, old_top_lvl);

    const int old_top_base_lvl = old_multires.base_level(old_top_lvl);
    const int new_top_base_lvl = new_multires->base_level(new_top_lvl);
    const int new_top_base_abs_lvl = target_lvl + new_top_base_lvl;
    auto rebase = [&]<typename T>(const Span<T> old_top, const MutableSpan<T> new_top) {
      if (old_top_base_lvl == new_top_base_abs_lvl && new_top_base_lvl == 0) {
        const int virtual_lvl = old_top_lvl - old_top_base_lvl;
        const int grid = Multiresolution::grid_size(virtual_lvl);
        threading::parallel_for(
            IndexRange(old_top_base.verts_num), 8192, [&](const IndexRange range) {
              for (const int v : range) {
                new_top[new_top_offsets.verts[v]] = old_top[old_top_offsets.verts[v]];
              }
            });
        if (grid > 0) {
          const int edge_grain = std::max(1, 8192 / grid);
          threading::parallel_for(
              IndexRange(old_top_base.edges_num), edge_grain, [&](const IndexRange range) {
                for (const int e : range) {
                  new_top.slice(new_top_offsets.edges[e], grid)
                      .copy_from(old_top.slice(old_top_offsets.edges[e], grid));
                }
              });
          const int face_verts_num = grid * grid;
          const int face_grain = std::max(1, 8192 / face_verts_num);
          threading::parallel_for(
              IndexRange(old_top_base.faces_num), face_grain, [&](const IndexRange range) {
                for (const int f : range) {
                  new_top.slice(new_top_offsets.faces[f], face_verts_num)
                      .copy_from(old_top.slice(old_top_offsets.faces[f], face_verts_num));
                }
              });
        }
        return;
      }
      old_multires.rebase_attribute(mesh,
                                    old_top_lvl,
                                    old_top_base_lvl,
                                    new_top_base_abs_lvl,
                                    old_top_offsets,
                                    new_top_offsets,
                                    old_top,
                                    new_top);
    };

    rebase(old_multires.top_base_positions(), new_multires->top_base_positions_for_write());

    const Span<float> old_top_masks = old_multires.top_masks();
    if (!old_top_masks.is_empty()) {
      rebase(old_top_masks, new_multires->top_masks_for_write());
    }

    const Span<float4> old_top_colors = old_multires.top_colors();
    if (!old_top_colors.is_empty()) {
      rebase(old_top_colors, new_multires->top_colors_for_write());
    }

    for (const SculptLayer &layer : mesh.sculpt_layers_as_span()) {
      const Span<float3> old_displacements = bke::sculpt_layer::layer_top_displacements(
          mesh, layer.name);
      if (!old_displacements.is_empty()) {
        const MutableSpan<float3> new_displacements =
            bke::sculpt_layer::layer_top_displacements_for_write(new_base, layer.name);
        rebase(old_displacements, new_displacements);
      }
    }
  }

  const bool had_replacement = (new_multires != nullptr);
  BKE_mesh_nomain_to_mesh(&new_base, &mesh, &object);
  if (!had_replacement && mesh.multires) {
    mesh.multires->free();
    MEM_SAFE_DELETE(mesh.multires);
  }
  BKE_sculptsession_free_pbvh(object);
}

static void delete_higher(Object &object, const int target_lvl)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);

  if (target_lvl == 0) {
    mesh.multires->free();
    MEM_SAFE_DELETE(mesh.multires);
    return;
  }

  Multiresolution &multires = *mesh.multires;

  const bool in_sculpt = is_sculpt(object);
  if (!in_sculpt && target_lvl < multires.top_level) {
    const int old_top = multires.top_level;
    const bke::multiresolution::BaseTopology top_base = multires.get_base_topology(mesh, old_top);
    const bke::multiresolution::Offsets top_offsets = Multiresolution::build_table_offsets(
        multires, top_base, old_top);
    multires.curr_level = target_lvl;

    downsample_to_level(mesh, multires, top_offsets, target_lvl);
  }

  bke::AttributeStorage &top_storage = multires.top_attribute_storage.wrap();
  bke::AttributeStorage &curr_storage = multires.curr_attribute_storage.wrap();
  top_storage = std::move(curr_storage);
  curr_storage = bke::AttributeStorage();

  multires.truncate_bvh_partitions(target_lvl);
  multires.top_level = target_lvl;
  multires.sculpt_level = std::min(multires.sculpt_level, target_lvl);
  multires.view_level = std::min(multires.view_level, target_lvl);
  multires.render_level = std::min(multires.render_level, target_lvl);
  if (!in_sculpt) {
    multires.curr_level = 0;
  }
}

static bool multires_counts_poll(bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  if (ob == nullptr || ob->type != OB_MESH || ob->data == nullptr) {
    return false;
  }
  if (BKE_object_sculpt_use_dyntopo(ob)) {
    return false;
  }
  Mesh *mesh = id_cast<Mesh *>(ob->data);
  if (mesh == nullptr || mesh->multires == nullptr || (mesh->flag & ME_SCULPT_DYNAMIC_TOPOLOGY)) {
    return false;
  }
  return bke::multiresolution::base_counts_match(*mesh);
}

static bool multires_subdiv_poll(bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  if (ob == nullptr || ob->type != OB_MESH || ob->data == nullptr) {
    return false;
  }
  if (BKE_object_sculpt_use_dyntopo(ob)) {
    return false;
  }
  Mesh *mesh = id_cast<Mesh *>(ob->data);
  if (mesh == nullptr || (mesh->flag & ME_SCULPT_DYNAMIC_TOPOLOGY)) {
    return false;
  }
  if (mesh->multires != nullptr) {
    return bke::multiresolution::base_counts_match(*mesh);
  }
  return true;
}

static bool is_sculpting_mesh(const Object *ob, const Mesh &mesh)
{
  return ob != nullptr && ob->data == &mesh.id && is_sculpt(*ob);
}

static void add_level_change_notifiers(bContext *C, Object *ob, Mesh &mesh)
{
  if (ob != nullptr && ob->data == &mesh.id) {
    WM_event_add_notifier(C, NC_OBJECT | ND_DRAW, ob);
  }
  WM_main_add_notifier(NC_GEOM | ND_DATA, &mesh.id);
}

bool set_sculpt_level(bContext *C, Object *ob, Mesh &mesh, int target_lvl, const char *undo_name)
{
  if (ob != nullptr && BKE_object_sculpt_use_dyntopo(ob)) {
    return false;
  }
  if (mesh.multires == nullptr) {
    return false;
  }
  Multiresolution &multires = *mesh.multires;
  const bool in_sculpt = is_sculpting_mesh(ob, mesh);

  if (in_sculpt) {
    multires.sculpt_level = multires.curr_level;
  }

  if (!bke::multiresolution::base_counts_match(mesh)) {
    return false;
  }

  target_lvl = math::clamp(target_lvl, 0, multires.top_level);

  if (in_sculpt) {
    if (target_lvl == multires.curr_level) {
      return false;
    }
    storage_undo::push_begin_with_name(
        *ob, undo_name, storage_undo::StorageUndoOpType::SwitchLevel);
    switch_level(*ob, target_lvl);
    storage_undo::push_end(*ob);
    DEG_id_tag_update(&ob->id, ID_RECALC_GEOMETRY);
  }
  else {
    if (target_lvl == multires.sculpt_level) {
      return false;
    }
    multires.sculpt_level = target_lvl;
  }

  add_level_change_notifiers(C, ob, mesh);
  return true;
}

void view_level_changed(bContext *C, Object *ob, Mesh &mesh)
{
  if (!is_sculpting_mesh(ob, mesh)) {
    DEG_id_tag_update(&mesh.id, ID_RECALC_GEOMETRY);
  }
  add_level_change_notifiers(C, ob, mesh);
}

static wmOperatorStatus multires_switch_step(bContext *C, wmOperator *op, const int delta)
{
  Object *ob = CTX_data_active_object(C);
  if (ob == nullptr || ob->type != OB_MESH) {
    return OPERATOR_CANCELLED;
  }
  Mesh &mesh = *id_cast<Mesh *>(ob->data);
  if (mesh.multires == nullptr) {
    return OPERATOR_CANCELLED;
  }

  const Multiresolution &multires = *mesh.multires;
  const int src_lvl = is_sculpt(*ob) ? multires.curr_level : multires.sculpt_level;
  const int target_lvl = src_lvl + delta;
  if (target_lvl < 0 || target_lvl > multires.top_level) {
    return OPERATOR_CANCELLED;
  }

  if (!set_sculpt_level(C, ob, mesh, target_lvl, op->type->name)) {
    return OPERATOR_CANCELLED;
  }
  return OPERATOR_FINISHED;
}

static wmOperatorStatus multires_switch_down_exec(bContext *C, wmOperator *op)
{
  return multires_switch_step(C, op, -1);
}

static wmOperatorStatus multires_switch_up_exec(bContext *C, wmOperator *op)
{
  return multires_switch_step(C, op, 1);
}

static bool base_mesh_is_manifold(const Mesh &mesh)
{
  if (!mesh.loose_verts().is_empty()) {
    return false;
  }
  if (!mesh.loose_edges().is_empty()) {
    return false;
  }
  if (!mesh.verts_no_face().is_empty()) {
    return false;
  }

  const GroupedSpan<int> edge_to_face = mesh.edge_to_face_map();
  for (const int edge_idx : IndexRange(mesh.edges_num)) {
    const int face_count = edge_to_face[edge_idx].size();
    if (face_count != 1 && face_count != 2) {
      return false;
    }
  }

  return true;
}

static wmOperatorStatus multires_subdiv_exec(bContext *C, wmOperator *op)
{
  Object &ob = *CTX_data_active_object(C);
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  if (BKE_object_sculpt_use_dyntopo(&ob) || (mesh.flag & ME_SCULPT_DYNAMIC_TOPOLOGY)) {
    BKE_report(op->reports, RPT_ERROR, "Cannot subdivide while Dynamic Topology is active.");
    return OPERATOR_CANCELLED;
  }

  const bool in_sculpt = is_sculpt(ob);
  if (in_sculpt && mesh.multires && mesh.multires->curr_level != mesh.multires->top_level) {
    return OPERATOR_CANCELLED;
  }
  if (mesh.multires && !bke::multiresolution::base_counts_match(mesh)) {
    return OPERATOR_CANCELLED;
  }

  if (!mesh.multires && !base_mesh_is_manifold(mesh)) {
    BKE_report(op->reports, RPT_ERROR, "Multiresolution only supports manifold meshes.");
    return OPERATOR_CANCELLED;
  }

  if (in_sculpt) {
    storage_undo::push_begin(ob, op, storage_undo::StorageUndoOpType::Subdivide);
  }

  BKE_paint_ensure_from_paintmode(CTX_data_scene(C), PaintMode::Sculpt);
  subdivide(ob);

  if (in_sculpt) {
    storage_undo::push_end(ob);
  }
  else {
    mesh.multires->sculpt_level = mesh.multires->top_level;
    mesh.multires->view_level = mesh.multires->top_level;
    mesh.multires->render_level = mesh.multires->top_level;
    ED_undo_push(C, op->type->name);
  }

  DEG_id_tag_update(&ob.id, ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_OBJECT | ND_DRAW, &ob);

  return OPERATOR_FINISHED;
}

static wmOperatorStatus multires_delete_lower_exec(bContext *C, wmOperator *op)
{
  Object &ob = *CTX_data_active_object(C);
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  if (BKE_object_sculpt_use_dyntopo(&ob) || (mesh.flag & ME_SCULPT_DYNAMIC_TOPOLOGY)) {
    return OPERATOR_CANCELLED;
  }
  if (!mesh.multires) {
    return OPERATOR_CANCELLED;
  }
  const bool in_sculpt = is_sculpt(ob);
  const int src_lvl = in_sculpt ? mesh.multires->sculpt_level : mesh.multires->view_level;
  if (src_lvl == 0) {
    return OPERATOR_CANCELLED;
  }
  if (!bke::multiresolution::base_counts_match(mesh)) {
    return OPERATOR_CANCELLED;
  }

  if (in_sculpt) {
    undo::geometry_begin(*CTX_data_scene(C), ob, op);
  }
  delete_lower(ob, src_lvl);
  if (in_sculpt) {
    undo::geometry_end(ob);
  }
  else {
    ED_undo_push(C, op->type->name);
  }

  DEG_id_tag_update(&ob.id, ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_OBJECT | ND_DRAW, &ob);

  return OPERATOR_FINISHED;
}

static wmOperatorStatus multires_delete_higher_exec(bContext *C, wmOperator *op)
{
  Object &ob = *CTX_data_active_object(C);
  Mesh &mesh = *id_cast<Mesh *>(ob.data);

  if (BKE_object_sculpt_use_dyntopo(&ob) || (mesh.flag & ME_SCULPT_DYNAMIC_TOPOLOGY)) {
    return OPERATOR_CANCELLED;
  }
  if (!mesh.multires) {
    return OPERATOR_CANCELLED;
  }
  const bool in_sculpt = is_sculpt(ob);
  const int src_lvl = in_sculpt ? mesh.multires->sculpt_level : mesh.multires->view_level;
  if (src_lvl == mesh.multires->top_level) {
    return OPERATOR_CANCELLED;
  }
  if (!bke::multiresolution::base_counts_match(mesh)) {
    return OPERATOR_CANCELLED;
  }

  if (in_sculpt) {
    undo::geometry_begin(*CTX_data_scene(C), ob, op);
  }
  delete_higher(ob, src_lvl);
  if (in_sculpt) {
    undo::geometry_end(ob);
  }
  else {
    ED_undo_push(C, op->type->name);
  }

  DEG_id_tag_update(&ob.id, ID_RECALC_GEOMETRY);
  WM_event_add_notifier(C, NC_OBJECT | ND_DRAW, &ob);

  return OPERATOR_FINISHED;
}

static wmOperatorStatus multires_set_sculpt_level_exec(bContext *C, wmOperator *op)
{
  Object *ob = CTX_data_active_object(C);
  if (ob->type != OB_MESH) {
    return OPERATOR_CANCELLED;
  }
  Mesh &mesh = *id_cast<Mesh *>(ob->data);
  const int target_lvl = RNA_int_get(op->ptr, "level");

  if (!set_sculpt_level(C, ob, mesh, target_lvl, op->type->name)) {
    return OPERATOR_CANCELLED;
  }

  return OPERATOR_FINISHED;
}

void SCULPT_OT_multires_set_sculpt_level(wmOperatorType *ot)
{
  ot->name = "Multires Set Sculpt Level";
  ot->idname = "SCULPT_OT_multires_set_sculpt_level";
  ot->description = "Directly switch to a multires sculpt level";
  ot->exec = multires_set_sculpt_level_exec;
  ot->poll = multires_counts_poll;
  ot->flag = OPTYPE_REGISTER;

  RNA_def_int(ot->srna,
              "level",
              0,
              0,
              32,
              "Sculpt Level",
              "Target multires sculpt level (clamped to the top level)",
              0,
              10);
}

static wmOperatorStatus multires_set_view_level_exec(bContext *C, wmOperator *op)
{
  Object *ob = CTX_data_active_object(C);
  if (ob->type != OB_MESH) {
    return OPERATOR_CANCELLED;
  }
  Mesh &mesh = *id_cast<Mesh *>(ob->data);
  Multiresolution &multires = *mesh.multires;
  const int target_lvl = math::clamp(RNA_int_get(op->ptr, "level"), 0, multires.top_level);

  if (target_lvl == multires.view_level) {
    return OPERATOR_CANCELLED;
  }

  multires.view_level = target_lvl;
  view_level_changed(C, ob, mesh);

  return OPERATOR_FINISHED;
}

void SCULPT_OT_multires_set_view_level(wmOperatorType *ot)
{
  ot->name = "Multires Set View Level";
  ot->idname = "SCULPT_OT_multires_set_view_level";
  ot->description = "Set the multires view level";
  ot->exec = multires_set_view_level_exec;
  ot->poll = multires_counts_poll;
  ot->flag = OPTYPE_REGISTER;

  RNA_def_int(ot->srna, "level", 0, 0, 32, "View Level", "Target multires view level", 0, 10);
}

static wmOperatorStatus multires_set_render_level_exec(bContext *C, wmOperator *op)
{
  Object *ob = CTX_data_active_object(C);
  if (ob->type != OB_MESH) {
    return OPERATOR_CANCELLED;
  }
  Mesh &mesh = *id_cast<Mesh *>(ob->data);
  Multiresolution &multires = *mesh.multires;
  const int target_lvl = math::clamp(RNA_int_get(op->ptr, "level"), 0, multires.top_level);

  if (target_lvl == multires.render_level) {
    return OPERATOR_CANCELLED;
  }

  multires.render_level = target_lvl;
  add_level_change_notifiers(C, ob, mesh);

  return OPERATOR_FINISHED;
}

void SCULPT_OT_multires_set_render_level(wmOperatorType *ot)
{
  ot->name = "Multires Set Render Level";
  ot->idname = "SCULPT_OT_multires_set_render_level";
  ot->description = "Set the multires render level";
  ot->exec = multires_set_render_level_exec;
  ot->poll = multires_counts_poll;
  ot->flag = OPTYPE_REGISTER;

  RNA_def_int(ot->srna, "level", 0, 0, 32, "Render Level", "Target multires render level", 0, 10);
}

void SCULPT_OT_multires_switch_down(wmOperatorType *ot)
{
  ot->name = "Multires Switch Down";
  ot->idname = "SCULPT_OT_multires_switch_down";
  ot->description = "Multires Switch Down";
  ot->exec = multires_switch_down_exec;
  ot->poll = multires_counts_poll;
  ot->flag = OPTYPE_REGISTER;
}

void SCULPT_OT_multires_switch_up(wmOperatorType *ot)
{
  ot->name = "Multires Switch Up";
  ot->idname = "SCULPT_OT_multires_switch_up";
  ot->description = "Multires Switch Up";
  ot->exec = multires_switch_up_exec;
  ot->poll = multires_counts_poll;
  ot->flag = OPTYPE_REGISTER;
}

void SCULPT_OT_multires_subdiv(wmOperatorType *ot)
{
  ot->name = "Multires Subdiv";
  ot->idname = "SCULPT_OT_multires_subdiv";
  ot->description = "Multires Subdiv";
  ot->exec = multires_subdiv_exec;
  ot->poll = multires_subdiv_poll;
  ot->flag = OPTYPE_REGISTER;
}

void SCULPT_OT_multires_delete_lower(wmOperatorType *ot)
{
  ot->name = "Multires Delete Lower";
  ot->idname = "SCULPT_OT_multires_delete_lower";
  ot->description = "Multires Delete Lower";
  ot->exec = multires_delete_lower_exec;
  ot->poll = multires_counts_poll;
  ot->flag = OPTYPE_REGISTER;
}

void SCULPT_OT_multires_delete_higher(wmOperatorType *ot)
{
  ot->name = "Multires Delete Higher";
  ot->idname = "SCULPT_OT_multires_delete_higher";
  ot->description = "Multires Delete Higher";
  ot->exec = multires_delete_higher_exec;
  ot->poll = multires_counts_poll;
  ot->flag = OPTYPE_REGISTER;
}

}  // namespace blender::ed::sculpt_paint::multires
