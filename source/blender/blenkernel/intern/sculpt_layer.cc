/* SPDX-FileCopyrightText: 2026 Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 */

#pragma once

#include "DNA_attribute_types.h"
#include "DNA_object_types.h"

#include "BLI_assert.hh"
#include "BLI_index_mask.hh"
#include "BLI_string_utf8.hh"
#include "BLI_string_utils.hh"
#include "BLI_task.hh"

#include "BKE_multiresolution.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"
#include "BKE_sculpt_layer.hh"

#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_query.hh"

#include "attribute_storage_access.hh"

namespace blender {

namespace bke {

namespace sculpt_layer {

static std::string get_unique_layer_name(const Mesh &mesh,
                                         const StringRef layer_name,
                                         const StringRefNull skip = {})
{
  return BLI_uniquename_cb(
      [&](const StringRef name) {
        for (const SculptLayer &layer : mesh.sculpt_layers_as_span()) {
          if (StringRef(layer.name) == skip) {
            continue;
          }
          if (StringRef(layer.name) == name) {
            return true;
          }
        }
        return false;
      },
      '.',
      layer_name);
}

std::string attr_name_from_layer_name(const StringRef layer_name)
{
  return ".sculpt_layer_" + layer_name;
}

std::string active_layer_attr_name(const Mesh &mesh)
{
  const int index = mesh.active_layer_index;
  BLI_assert(index >= 0 && index < mesh.sculpt_layers_len);
  return attr_name_from_layer_name(mesh.sculpt_layers[index].name);
}

bool mesh_has_visible_layers(const Mesh &base_mesh)
{
  for (const SculptLayer &layer : base_mesh.sculpt_layers_as_span()) {
    if (layer.is_visible && layer.opacity != 0.0f) {
      return true;
    }
  }
  return false;
}

static void add_visible_layers_to_positions(const Mesh &mesh,
                                            const Span<int> verts,
                                            const MutableSpan<float3> positions)
{
  for (const SculptLayer &layer : mesh.sculpt_layers_as_span()) {
    if (!layer.is_visible || layer.opacity == 0.0f) {
      continue;
    }
    const Span<float3> displacement = layer_displacements(mesh, layer.name);
    if (displacement.is_empty()) {
      continue;
    }
    if (layer.opacity == 1.0f) {
      for (const int vert : verts) {
        positions[vert] += displacement[vert];
      }
    }
    else {
      for (const int vert : verts) {
        positions[vert] += displacement[vert] * layer.opacity;
      }
    }
  }
}

void apply_layer_displacements(const Mesh &mesh,
                               const Span<int> verts,
                               const MutableSpan<float3> positions)
{
  add_visible_layers_to_positions(mesh, verts, positions);
}

void apply_layer_displacements(const Mesh &mesh,
                               const IndexRange range,
                               const MutableSpan<float3> positions)
{
  for (const SculptLayer &layer : mesh.sculpt_layers_as_span()) {
    if (!layer.is_visible || layer.opacity == 0.0f) {
      continue;
    }
    const Span<float3> displacement = layer_displacements(mesh, layer.name);
    if (displacement.is_empty()) {
      continue;
    }
    if (layer.opacity == 1.0f) {
      for (const int vert : range) {
        positions[vert] += displacement[vert];
      }
    }
    else {
      for (const int vert : range) {
        positions[vert] += displacement[vert] * layer.opacity;
      }
    }
  }
}

void evaluate_deform(const Mesh &orig_mesh, Mesh &mesh_deform)
{
  if (!mesh_has_visible_layers(orig_mesh)) {
    return;
  }
  BLI_assert(mesh_deform.verts_num == orig_mesh.verts_num);

  MutableSpan<float3> positions = mesh_deform.vert_positions_for_write();

  threading::parallel_for(IndexRange(mesh_deform.verts_num), 4096, [&](IndexRange range) {
    apply_layer_displacements(orig_mesh, range, positions);
  });

  mesh_deform.tag_positions_changed();
}

void evaluate_final(const Mesh &orig_mesh, Mesh &eval_mesh)
{
  if (!mesh_has_visible_layers(orig_mesh)) {
    return;
  }

  MutableSpan<float3> positions = eval_mesh.vert_positions_for_write();
  const bool topology_matches = (orig_mesh.verts_num == eval_mesh.verts_num);

  threading::parallel_for(IndexRange(eval_mesh.verts_num), 4096, [&](IndexRange range) {
    for (const SculptLayer &layer : orig_mesh.sculpt_layers_as_span()) {
      if (!layer.is_visible || layer.opacity == 0.0f) {
        continue;
      }

      Span<float3> disp = layer_displacements(eval_mesh, layer.name);
      if ((disp.is_empty() || disp.size() != eval_mesh.verts_num) && topology_matches) {
        disp = layer_displacements(orig_mesh, layer.name);
      }
      if (disp.is_empty()) {
        continue;
      }
      if (layer.opacity == 1.0f) {
        for (const int vert : range) {
          positions[vert] += disp[vert];
        }
      }
      else {
        for (const int vert : range) {
          positions[vert] += disp[vert] * layer.opacity;
        }
      }
    }
  });

  eval_mesh.tag_positions_changed();
}

void calc_eval_positions(const Mesh &mesh, MutableSpan<float3> positions)
{
  BLI_assert(positions.size() == mesh.verts_num);
  positions.copy_from(mesh.vert_positions());
  apply_layer_displacements(mesh, positions.index_range(), positions);
}

void calc_eval_positions_range(const Mesh &mesh,
                               const Span<int> verts,
                               const MutableSpan<float3> positions)
{
  const Span<float3> orig_pos = mesh.vert_positions();
  for (const int v : verts) {
    positions[v] = orig_pos[v];
  }
  add_visible_layers_to_positions(mesh, verts, positions);
}

SculptLayer *active_layer(const Mesh &mesh)
{
  const int index = mesh.active_layer_index;
  if (index == -1) {
    return nullptr;
  }

  BLI_assert(index >= 0 && index < mesh.sculpt_layers_len);
  return &mesh.sculpt_layers[index];
}

Span<float3> layer_displacements(const Mesh &mesh, StringRef layer_name)
{
  const std::string attr_name = attr_name_from_layer_name(layer_name);
  if (mesh.multires == nullptr || mesh.multires->curr_level == 0) {
    return bke::get_span_attribute<float3>(
               mesh.attribute_storage.wrap(), bke::AttrDomain::Point, attr_name, mesh.verts_num)
        .value_or(Span<float3>());
  }
  else {
    return mesh.multires->attribute<float3>(attr_name);
  }
}

MutableSpan<float3> layer_displacements_for_write(Mesh &mesh, StringRef layer_name)
{
  const std::string attr_name = attr_name_from_layer_name(layer_name);
  if (mesh.multires == nullptr || mesh.multires->curr_level == 0) {
    return bke::get_mutable_attribute<float3>(
        mesh.attribute_storage.wrap(), bke::AttrDomain::Point, attr_name, mesh.verts_num);
  }
  else {
    return mesh.multires->attribute_for_write<float3>(attr_name);
  }
}

Span<float3> layer_top_displacements(const Mesh &mesh, StringRef layer_name)
{
  const std::string attr_name = attr_name_from_layer_name(layer_name);
  return mesh.multires->top_attribute<float3>(attr_name);
}

MutableSpan<float3> layer_top_displacements_for_write(const Mesh &mesh, StringRef layer_name)
{
  const std::string attr_name = attr_name_from_layer_name(layer_name);
  return mesh.multires->top_attribute_for_write<float3>(attr_name);
}

Span<float3> active_layer_displacements(const Mesh &mesh)
{
  const int index = mesh.active_layer_index;
  BLI_assert(index >= 0 && index < mesh.sculpt_layers_len);

  return layer_displacements(mesh, mesh.sculpt_layers[index].name);
}

MutableSpan<float3> active_layer_displacements_for_write(Mesh &mesh)
{
  const int index = mesh.active_layer_index;
  BLI_assert(index >= 0 && index < mesh.sculpt_layers_len);

  const std::string attr_name = active_layer_attr_name(mesh);
  if (mesh.multires == nullptr || mesh.multires->curr_level == 0) {
    return bke::get_mutable_attribute<float3>(
        mesh.attribute_storage.wrap(), bke::AttrDomain::Point, attr_name, mesh.verts_num);
  }
  else {
    return mesh.multires->attribute_for_write<float3>(attr_name);
  }
}

float effective_layer_weight(const SculptLayer &layer)
{
  return layer.is_visible ? layer.opacity : 0.0f;
}

void combine_displacements(const float src_weight,
                           const float dst_weight,
                           const Span<float3> src,
                           const MutableSpan<float3> dst)
{
  if (dst.is_empty()) {
    return;
  }
  BLI_assert(src.is_empty() || src.size() == dst.size());

  if (src.is_empty() || src_weight == 0.0f) {
    if (dst_weight == 1.0f) {
      return;
    }
    threading::parallel_for(dst.index_range(), 4096, [&](const IndexRange range) {
      for (const int i : range) {
        dst[i] *= dst_weight;
      }
    });
    return;
  }
  threading::parallel_for(dst.index_range(), 4096, [&](const IndexRange range) {
    for (const int i : range) {
      dst[i] = dst[i] * dst_weight + src[i] * src_weight;
    }
  });
}

static void copy_layer_attributes(Mesh &mesh, const StringRef src_name, const StringRef dst_name)
{
  const Span<float3> src = layer_displacements(mesh, src_name);
  if (!src.is_empty()) {
    layer_displacements_for_write(mesh, dst_name).copy_from(src);
  }
  if (mesh.multires != nullptr && mesh.multires->curr_level != mesh.multires->top_level) {
    const Span<float3> top_src = layer_top_displacements(mesh, src_name);
    if (!top_src.is_empty()) {
      layer_top_displacements_for_write(mesh, dst_name).copy_from(top_src);
    }
  }
}

void duplicate_active_layer(Mesh &mesh)
{
  const int src_index = mesh.active_layer_index;
  BLI_assert(src_index >= 0 && src_index < mesh.sculpt_layers_len);

  const std::string unique_name = get_unique_layer_name(
      mesh, std::string(mesh.sculpt_layers[src_index].name));

  const int new_len = mesh.sculpt_layers_len + 1;
  mesh.sculpt_layers = static_cast<SculptLayer *>(
      MEM_realloc_uninitialized(mesh.sculpt_layers, sizeof(SculptLayer) * new_len));

  for (int i = mesh.sculpt_layers_len - 1; i > src_index; --i) {
    mesh.sculpt_layers[i + 1] = mesh.sculpt_layers[i];
  }

  SculptLayer &copy = mesh.sculpt_layers[src_index + 1];
  copy = mesh.sculpt_layers[src_index];
  STRNCPY_UTF8(copy.name, unique_name.c_str());
  mesh.sculpt_layers_len = new_len;
  mesh.active_layer_index = src_index + 1;

  copy_layer_attributes(mesh, mesh.sculpt_layers[src_index].name, copy.name);
}

void apply_displacement(const Span<float3> disp,
                        const float weight,
                        const MutableSpan<float3> base)
{
  if (weight == 0.0f || disp.is_empty()) {
    return;
  }
  BLI_assert(disp.size() == base.size());

  threading::parallel_for(base.index_range(), 4096, [&](const IndexRange range) {
    for (const int i : range) {
      base[i] += weight * disp[i];
    }
  });
}

void merge_active_layer_to_base(Mesh &mesh)
{
  const int index = mesh.active_layer_index;
  BLI_assert(index >= 0 && index < mesh.sculpt_layers_len);

  const SculptLayer &src = mesh.sculpt_layers[index];
  const float weight = effective_layer_weight(src);

  if (weight != 0.0f) {
    const int curr_lvl = mesh.multires == nullptr ? 0 : mesh.multires->curr_level;

    const Span<float3> disp = layer_displacements(mesh, src.name);
    if (!disp.is_empty()) {
      const MutableSpan<float3> base = curr_lvl == 0 ? mesh.vert_positions_for_write() :
                                                       mesh.multires->base_positions_for_write();
      apply_displacement(disp, weight, base);
    }

    if (mesh.multires && mesh.multires->curr_level < mesh.multires->top_level) {
      const Span<float3> top_disp = layer_top_displacements(mesh, src.name);
      if (!top_disp.is_empty()) {
        apply_displacement(top_disp, weight, mesh.multires->top_base_positions_for_write());
      }
    }

    mesh.tag_positions_changed();
  }
  remove_active_layer(mesh);
  mesh.active_layer_index = -1;
}

void merge_active_layer_down(Mesh &mesh)
{
  const int index = mesh.active_layer_index;
  BLI_assert(index >= 0 && index < mesh.sculpt_layers_len);
  if (index == 0) {
    merge_active_layer_to_base(mesh);
    return;
  }
  const float src_weight = effective_layer_weight(mesh.sculpt_layers[index]);
  const float dst_weight = effective_layer_weight(mesh.sculpt_layers[index - 1]);
  const StringRef src_name = mesh.sculpt_layers[index].name;
  const StringRef dst_name = mesh.sculpt_layers[index - 1].name;
  const Span<float3> src_disp = layer_displacements(mesh, src_name);
  const Span<float3> dst_disp = layer_displacements(mesh, dst_name);

  if (!src_disp.is_empty() || !dst_disp.is_empty()) {
    combine_displacements(
        src_weight, dst_weight, src_disp, layer_displacements_for_write(mesh, dst_name));
  }
  if (mesh.multires != nullptr && mesh.multires->curr_level != mesh.multires->top_level) {
    const Span<float3> top_src = layer_top_displacements(mesh, src_name);
    const Span<float3> top_dst = layer_top_displacements(mesh, dst_name);
    if (!top_src.is_empty() || !top_dst.is_empty()) {
      combine_displacements(
          src_weight, dst_weight, top_src, layer_top_displacements_for_write(mesh, dst_name));
    }
  }

  mesh.sculpt_layers[index - 1].opacity = 1.0f;
  mesh.sculpt_layers[index - 1].is_visible = true;
  mesh.sculpt_layers[index - 1].slider_min = 0.0f;
  mesh.sculpt_layers[index - 1].slider_max = 1.0f;

  mesh.active_layer_index = index;
  remove_active_layer(mesh);
  mesh.active_layer_index = index - 1;
}

void add_layer(Mesh &mesh, const StringRef layer_name)
{
  const int new_len = mesh.sculpt_layers_len + 1;
  mesh.sculpt_layers = static_cast<SculptLayer *>(
      MEM_realloc_uninitialized(mesh.sculpt_layers, sizeof(SculptLayer) * new_len));

  std::string unique_name = get_unique_layer_name(mesh, layer_name);

  SculptLayer &layer = mesh.sculpt_layers[mesh.sculpt_layers_len];
  layer = {};
  STRNCPY_UTF8(layer.name, unique_name.c_str());
  layer.opacity = 1.0f;
  layer.is_visible = true;
  layer.slider_min = 0.0f;
  layer.slider_max = 1.0f;

  mesh.sculpt_layers_len = new_len;
  mesh.active_layer_index = new_len - 1;
}

void remove_active_layer(Mesh &mesh)
{
  const int index = mesh.active_layer_index;
  const std::string name = active_layer_attr_name(mesh);

  mesh.attributes_for_write().remove(name);
  if (mesh.multires) {
    mesh.multires->curr_attribute_storage.wrap().remove(name);
    mesh.multires->top_attribute_storage.wrap().remove(name);
    if (mesh.multires->runtime) {
      mesh.multires->runtime->curr_dirty_layers.remove(name);
    }
  }

  const int new_len = mesh.sculpt_layers_len - 1;
  for (int i = index; i < new_len; i++) {
    mesh.sculpt_layers[i] = mesh.sculpt_layers[i + 1];
  }
  mesh.sculpt_layers_len = new_len;

  if (new_len == 0) {
    MEM_SAFE_DELETE(mesh.sculpt_layers);
    mesh.active_layer_index = -1;
  }
  else {
    mesh.sculpt_layers = static_cast<SculptLayer *>(
        MEM_realloc_uninitialized(mesh.sculpt_layers, sizeof(SculptLayer) * new_len));
    mesh.active_layer_index = std::min(index, new_len - 1);
  }
}

void rename_layer(Mesh &mesh, SculptLayer &layer, const StringRef new_name)
{
  const std::string old_attr_name = attr_name_from_layer_name(layer.name);
  const std::string new_name_unique = get_unique_layer_name(mesh, new_name, layer.name);
  const std::string new_attr_name = attr_name_from_layer_name(new_name_unique);

  bke::AttributeStorage &base_storage = mesh.attribute_storage.wrap();
  if (base_storage.lookup(old_attr_name) != nullptr) {
    base_storage.rename(old_attr_name, new_attr_name);
  }
  if (mesh.multires) {
    bke::AttributeStorage &curr_storage = mesh.multires->curr_attribute_storage.wrap();
    if (curr_storage.lookup(old_attr_name) != nullptr) {
      curr_storage.rename(old_attr_name, new_attr_name);
    }
    bke::AttributeStorage &top_storage = mesh.multires->top_attribute_storage.wrap();
    if (top_storage.lookup(old_attr_name) != nullptr) {
      top_storage.rename(old_attr_name, new_attr_name);
    }
    if (mesh.multires->runtime) {
      if (mesh.multires->runtime->curr_dirty_layers.remove(old_attr_name)) {
        mesh.multires->runtime->curr_dirty_layers.add(new_attr_name);
      }
    }
  }

  STRNCPY_UTF8(layer.name, new_name_unique.c_str());
}

bool is_layers_only_eval(const Object &object)
{
  if ((object.mode & OB_MODE_SCULPT) == 0) {
    return false;
  }
  const SculptSession *ss = object.runtime->sculpt_session;
  return ss != nullptr && ss->bm == nullptr && ss->layers_only_eval && !ss->layers_cos.is_empty();
}

bool can_use_fast_path_update(const Object &object)
{
  const Mesh &mesh = *id_cast<const Mesh *>(object.data);
  const SculptSession *ss = object.runtime->sculpt_session;
  if (ss == nullptr) {
    return false;
  }
  if (ss->bm != nullptr) {
    return false;
  }
  const bke::pbvh::Tree *pbvh = bke::object::pbvh_get(object);
  if (pbvh == nullptr || !ELEM(pbvh->type(), bke::pbvh::Type::Mesh, bke::pbvh::Type::Multires)) {
    return false;
  }

  if (mesh.multires == nullptr || mesh.multires->curr_level == 0) {
    if (ss->shapekey_active != nullptr || ss->deform_modifiers_active) {
      return false;
    }
  }
  return true;
}

static Vector<int> calc_non_zero_leaf_nodes_for_displacements(const bke::pbvh::Tree &pbvh,
                                                              const Span<float3> displacements,
                                                              const int grid_size)
{
  if (displacements.is_empty()) {
    return {};
  }
  Vector<int> non_zero_nodes;
  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh: {
      const Span<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
      Array<bool> dirty(nodes.size(), false);
      threading::parallel_for(nodes.index_range(), 8, [&](const IndexRange range) {
        for (const int i : range) {
          for (const int v : nodes[i].verts()) {
            if (displacements[v] != float3(0.0f)) {
              dirty[i] = true;
              break;
            }
          }
        }
      });
      for (const int i : nodes.index_range()) {
        if (dirty[i]) {
          non_zero_nodes.append(i);
        }
      }
      break;
    }
    case bke::pbvh::Type::Multires: {
      const Span<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
      Array<bool> dirty(nodes.size(), false);
      threading::parallel_for(nodes.index_range(), 8, [&](const IndexRange range) {
        for (const int i : range) {
          const IndexRange node_range = nodes[i].range(grid_size);
          for (const float3 &d : displacements.slice(node_range)) {
            if (d != float3(0.0f)) {
              dirty[i] = true;
              break;
            }
          }
        }
      });
      for (const int i : nodes.index_range()) {
        if (dirty[i]) {
          non_zero_nodes.append(i);
        }
      }
      break;
    }
    case bke::pbvh::Type::Grids:
    case bke::pbvh::Type::BMesh: {
      BLI_assert_unreachable();
      break;
    }
  }
  return non_zero_nodes;
}

static Span<int> get_or_calc_layer_non_zero_nodes(bke::pbvh::Tree &pbvh,
                                                  const StringRef layer_name,
                                                  const Span<float3> displacements,
                                                  const int grid_size)
{
  return pbvh.layer_non_zero_nodes_ensure(layer_name, [&]() {
    return calc_non_zero_leaf_nodes_for_displacements(pbvh, displacements, grid_size);
  });
}

static void calc_positions_masked(const Mesh &mesh,
                                  const MutableSpan<float3> positions,
                                  const Span<bke::pbvh::MeshNode> nodes,
                                  const IndexMask &node_mask)
{
  node_mask.foreach_index(
      [&](const int i) { calc_eval_positions_range(mesh, nodes[i].verts(), positions); },
      exec_mode::grain_size(1));
}

bool update_eval_positions(Object &object, Depsgraph &depsgraph, const SculptLayer &changed_layer)
{
  Mesh &mesh = *id_cast<Mesh *>(object.data);
  if (!can_use_fast_path_update(object)) {
    return false;
  }

  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  const Span<float3> changed_disp = layer_displacements(mesh, changed_layer.name);
  const int grid_size = (pbvh.type() == bke::pbvh::Type::Multires) ? mesh.multires->grid_size() :
                                                                     0;

  const Span<int> non_zero_nodes = get_or_calc_layer_non_zero_nodes(
      pbvh, changed_layer.name, changed_disp, grid_size);

  IndexMaskMemory memory;
  const IndexMask node_mask = IndexMask::from_indices<int>(non_zero_nodes, memory);

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh: {
      const Span<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();

      if (is_layers_only_eval(object)) {
        SculptSession &ss = *object.runtime->sculpt_session;
        BLI_assert(ss.layers_cos.size() == mesh.verts_num);
        calc_positions_masked(mesh, ss.layers_cos.as_mutable_span(), nodes, node_mask);
        break;
      }

      MutableSpan<float3> eval_pos = bke::pbvh::vert_positions_eval_for_write(depsgraph, object);
      calc_positions_masked(mesh, eval_pos, nodes, node_mask);
      break;
    }
    case bke::pbvh::Type::Multires: {
      const Span<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
      const Span<float3> base_positions = mesh.multires->base_positions();
      node_mask.foreach_index(
          [&](const int i) {
            mesh.multires->runtime->calc_eval_positions(
                mesh, base_positions, nodes[i].range(grid_size));
          },
          exec_mode::grain_size(1));
      break;
    }
    case bke::pbvh::Type::Grids:
    case bke::pbvh::Type::BMesh: {
      BLI_assert_unreachable();
      return false;
    }
  }

  pbvh.tag_positions_changed(node_mask, true);
  pbvh.update_bounds(depsgraph, object);
  bke::pbvh::store_bounds_orig(pbvh);
  return true;
}

void clear_layer_non_zero_nodes_cache(Object &object)
{
  if (bke::pbvh::Tree *pbvh = bke::object::pbvh_get(object)) {
    pbvh->clear_layer_non_zero_nodes_cache();
  }
}

}  // namespace sculpt_layer

}  // namespace bke

}  // namespace blender
