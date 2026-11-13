/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bke
 *
 */

#include "DNA_mesh_types.h"

#include "BKE_multiresolution.hh"
#include "BKE_paint_bvh.hh"

#include "BLI_array_utils.hh"
#include "BLI_assert.hh"
#include "BLI_bit_vector.hh"
#include "BLI_index_mask.hh"
#include "BLI_offset_indices.hh"
#include "BLI_task.hh"

#include <algorithm>
#include <climits>
#include <iterator>
#include <utility>

namespace blender::bke::multiresolution {

MultiresCoarseView::MultiresCoarseView() = default;
MultiresCoarseView::~MultiresCoarseView() = default;

static int target_coarse_level(const Multiresolution &multires)
{
  const int fine_lvl = multires.curr_level;
  if (fine_lvl <= 1) {
    return -1;
  }
  constexpr int max_coarse_verts = 100'000;
  if (multires.verts_num() <= max_coarse_verts) {
    return -1;
  }
  for (int lvl = fine_lvl - 1; lvl >= 1; lvl--) {
    const int verts_num = calc_total_verts(lvl,
                                           multires.base_verts_num,
                                           multires.base_edges_num,
                                           multires.base_faces_num,
                                           multires.base_corners_num);
    if (verts_num <= max_coarse_verts) {
      return lvl;
    }
  }
  return 1;
}

static int total_verts_at_level(const Multiresolution &multires,
                                const Mesh &base_mesh,
                                const int lvl)
{
  const BaseTopology topo = multires.get_base_topology(base_mesh, lvl);
  const int virtual_lvl = lvl - multires.base_level(lvl);
  return calc_total_verts(
      virtual_lvl, topo.verts_num, topo.edges_num, topo.faces_num, topo.corners_num);
}

static void build_fine_to_coarse_nodes(MultiresCoarseView &view,
                                       const Multiresolution &multires,
                                       const pbvh::Tree &fine_tree)
{
  const Span<pbvh::MultiresNode> fine_nodes = fine_tree.nodes<pbvh::MultiresNode>();
  const Span<pbvh::MultiresNode> coarse_nodes = view.tree->nodes<pbvh::MultiresNode>();
  const int fine_grid_size = Multiresolution::grid_size(view.fine_level -
                                                        multires.base_level(view.fine_level));
  const int coarse_grid_size = Multiresolution::grid_size(view.coarse_level -
                                                          multires.base_level(view.coarse_level));

  Vector<std::pair<int, int>> fine_range_starts;
  for (const int i : fine_nodes.index_range()) {
    const pbvh::MultiresNode &node = fine_nodes[i];
    if ((node.flag_ & pbvh::Node::Leaf) != 0 && !node.range(fine_grid_size).is_empty()) {
      fine_range_starts.append({node.offset(), i});
    }
  }
  std::sort(fine_range_starts.begin(), fine_range_starts.end());

  const auto fine_owner = [&](const int fine_vert) {
    const auto it = std::upper_bound(fine_range_starts.begin(),
                                     fine_range_starts.end(),
                                     std::pair<int, int>(fine_vert, INT_MAX));
    BLI_assert(it != fine_range_starts.begin());
    return std::prev(it)->second;
  };

  Vector<int64_t> pairs;
  for (const int k : coarse_nodes.index_range()) {
    const pbvh::MultiresNode &node = coarse_nodes[k];
    if ((node.flag_ & pbvh::Node::Leaf) == 0) {
      continue;
    }
    int64_t last_pair = -1;
    for (const int c : node.range(coarse_grid_size)) {
      const int64_t pair = (int64_t(fine_owner(view.to_fine_vert_map[c])) << 32) | int64_t(k);
      if (pair != last_pair) {
        pairs.append(pair);
        last_pair = pair;
      }
    }
  }
  std::sort(pairs.begin(), pairs.end());
  pairs.resize(std::unique(pairs.begin(), pairs.end()) - pairs.begin());

  view.fine_to_coarse_offsets.reinitialize(fine_nodes.size() + 1);
  view.fine_to_coarse_offsets.as_mutable_span().fill(0);
  view.fine_to_coarse_nodes.reinitialize(pairs.size());
  for (const int64_t i : pairs.index_range()) {
    view.fine_to_coarse_offsets[int(pairs[i] >> 32) + 1]++;
    view.fine_to_coarse_nodes[i] = int(pairs[i] & 0xffffffff);
  }
  for (const int i : fine_nodes.index_range()) {
    view.fine_to_coarse_offsets[i + 1] += view.fine_to_coarse_offsets[i];
  }
}

static IndexMask stale_coarse_nodes(const MultiresCoarseView &view,
                                    const BitVector<> &fine_dirty,
                                    IndexMaskMemory &memory)
{
  const OffsetIndices<int> fine_to_coarse(view.fine_to_coarse_offsets);
  BitVector<> coarse_dirty(view.tree->nodes_num(), false);
  const int fine_nodes_num = std::min(int(fine_dirty.size()), int(fine_to_coarse.size()));
  for (const int fine_node : IndexRange(fine_nodes_num)) {
    if (!fine_dirty[fine_node]) {
      continue;
    }
    for (const int coarse_node :
         view.fine_to_coarse_nodes.as_span().slice(fine_to_coarse[fine_node]))
    {
      coarse_dirty[coarse_node].set();
    }
  }
  return IndexMask::from_bits(coarse_dirty, memory);
}

template<typename T>
static IndexMask sync_vertex_attr(const MultiresCoarseView &view,
                                  const Span<T> fine,
                                  Array<T> &coarse,
                                  const int coarse_grid_size,
                                  const BitVector<> &fine_dirty,
                                  IndexMaskMemory &memory)
{
  const Span<pbvh::MultiresNode> nodes = view.tree->nodes<pbvh::MultiresNode>();
  const IndexMask all_nodes(nodes.index_range());
  if (fine.is_empty()) {
    coarse.reinitialize(0);
    return all_nodes;
  }

  const Span<int> map = view.to_fine_vert_map;
  if (coarse.size() != map.size() || fine_dirty.is_empty()) {
    coarse.reinitialize(map.size());
    MutableSpan<T> coarse_span = coarse;
    threading::parallel_for(map.index_range(), 4096, [&](const IndexRange range) {
      for (const int i : range) {
        coarse_span[i] = fine[map[i]];
      }
    });
    return all_nodes;
  }

  const IndexMask stale_nodes = stale_coarse_nodes(view, fine_dirty, memory);
  MutableSpan<T> coarse_span = coarse;
  stale_nodes.foreach_index(
      [&](const int node_i) {
        for (const int c : nodes[node_i].range(coarse_grid_size)) {
          coarse_span[c] = fine[map[c]];
        }
      },
      exec_mode::grain_size(1));
  return stale_nodes;
}

static int view_grid_size(const MultiresCoarseView &view, const Multiresolution &multires)
{
  return Multiresolution::grid_size(view.coarse_level - multires.base_level(view.coarse_level));
}

static void sync_positions(MultiresCoarseView &view,
                           const Multiresolution &multires,
                           const BitVector<> &dirty_positions)
{
  const Span<float3> fine_positions = multires.positions();
  if (fine_positions.is_empty()) {
    return;
  }
  const int grid_size = view_grid_size(view, multires);
  IndexMaskMemory memory;
  const IndexMask changed_nodes = sync_vertex_attr<float3>(
      view, fine_positions, view.positions, grid_size, dirty_positions, memory);
  if (changed_nodes.is_empty()) {
    return;
  }
  view.tree->tag_positions_changed(changed_nodes);
  view.tree->update_bounds_multires(view.positions, grid_size);
}

static void sync_masks(MultiresCoarseView &view,
                       const Multiresolution &multires,
                       const BitVector<> &dirty_masks)
{
  IndexMaskMemory memory;
  const IndexMask changed_nodes = sync_vertex_attr<float>(
      view, multires.masks(), view.masks, view_grid_size(view, multires), dirty_masks, memory);
  if (!changed_nodes.is_empty()) {
    view.tree->tag_masks_changed(changed_nodes);
  }
}

static Vector<int> dirty_root_quads_from_fine_nodes(const MultiresCoarseView &view,
                                                    const BitVector<> &dirty_fine_nodes)
{
  if (dirty_fine_nodes.is_empty()) {
    return {};
  }
  const Span<pbvh::MultiresNode> fine_nodes = view.fine_tree->nodes<pbvh::MultiresNode>();
  const int fine_nodes_num = std::min(int(dirty_fine_nodes.size()), int(fine_nodes.size()));
  BitVector<> dirty_root_quads(view.root_quads_num, false);
  for (const int fine_node_i : IndexRange(fine_nodes_num)) {
    if (!dirty_fine_nodes[fine_node_i]) {
      continue;
    }
    for (const int f : fine_nodes[fine_node_i].faces()) {
      const int q = f >> (2 * view.fine_base_rel);
      if (q < view.root_quads_num) {
        dirty_root_quads[q].set();
      }
    }
  }

  Vector<int> dirty_quad_indices;
  for (const int q : IndexRange(view.root_quads_num)) {
    if (dirty_root_quads[q]) {
      dirty_quad_indices.append(q);
    }
  }
  return dirty_quad_indices;
}

static IndexMask coarse_nodes_from_root_quads(const MultiresCoarseView &view,
                                              const Mesh &base_mesh,
                                              const Multiresolution &multires,
                                              const Span<int> dirty_quad_indices,
                                              IndexMaskMemory &memory)
{
  BitVector<> coarse_dirty(view.tree->nodes_num(), false);
  if (!multires.base_all_quads && multires.base_level(view.coarse_level) == 0) {
    const Span<int> corner_to_face = base_mesh.corner_to_face_map();
    for (const int q : dirty_quad_indices) {
      if (q < corner_to_face.size()) {
        coarse_dirty[view.coarse_face_to_node[corner_to_face[q]]].set();
      }
    }
    return IndexMask::from_bits(coarse_dirty, memory);
  }
  const int coarse_faces_per_root = 1 << (2 * view.coarse_base_rel);
  for (const int q : dirty_quad_indices) {
    const int start_f = q << (2 * view.coarse_base_rel);
    for (const int f : IndexRange(start_f, coarse_faces_per_root)) {
      coarse_dirty[view.coarse_face_to_node[f]].set();
    }
  }
  return IndexMask::from_bits(coarse_dirty, memory);
}

static void sync_face_sets(MultiresCoarseView &view,
                           const Mesh &base_mesh,
                           const Multiresolution &multires,
                           const BitVector<> &dirty_face_sets)
{
  const Span<int> fine_face_sets = multires.face_sets();
  const Span<pbvh::MultiresNode> all_coarse_nodes = view.tree->nodes<pbvh::MultiresNode>();
  if (fine_face_sets.is_empty()) {
    view.face_sets.reinitialize(0);
    view.tree->tag_face_sets_changed(all_coarse_nodes.index_range());
    return;
  }

  const int total_coarse_faces = view.corner_verts.size() / 4;
  if (view.face_sets.size() != total_coarse_faces || dirty_face_sets.is_empty()) {
    view.face_sets.reinitialize(total_coarse_faces);
    multires.downsample_face_sets(
        base_mesh, view.fine_level, fine_face_sets, view.coarse_level, view.face_sets);
    view.tree->tag_face_sets_changed(all_coarse_nodes.index_range());
    return;
  }

  const Vector<int> dirty_quad_indices = dirty_root_quads_from_fine_nodes(view, dirty_face_sets);
  if (dirty_quad_indices.is_empty()) {
    return;
  }

  if (dirty_quad_indices.size() > view.root_quads_num / 3) {
    multires.downsample_face_sets(
        base_mesh, view.fine_level, fine_face_sets, view.coarse_level, view.face_sets);
    view.tree->tag_face_sets_changed(all_coarse_nodes.index_range());
    return;
  }

  multires.downsample_face_sets(
      view.fine_level, fine_face_sets, view.coarse_level, view.face_sets, dirty_quad_indices);

  IndexMaskMemory memory;
  const IndexMask coarse_mask = coarse_nodes_from_root_quads(
      view, base_mesh, multires, dirty_quad_indices, memory);
  if (!coarse_mask.is_empty()) {
    view.tree->tag_face_sets_changed(coarse_mask);
  }
}

static void sync_hidden_faces(MultiresCoarseView &view,
                              const Mesh &base_mesh,
                              const Multiresolution &multires,
                              const BitVector<> &dirty_visibility)
{
  const Span<bool> fine_hidden = multires.hidden_faces();
  const Span<pbvh::MultiresNode> all_coarse_nodes = view.tree->nodes<pbvh::MultiresNode>();
  if (fine_hidden.is_empty()) {
    view.hidden_faces.reinitialize(0);
    view.tree->tag_visibility_changed(all_coarse_nodes.index_range());
    return;
  }

  const int total_coarse_faces = view.corner_verts.size() / 4;
  if (view.hidden_faces.size() != total_coarse_faces || dirty_visibility.is_empty()) {
    view.hidden_faces.reinitialize(total_coarse_faces);
    multires.downsample_hidden_faces(
        base_mesh, view.fine_level, fine_hidden, view.coarse_level, view.hidden_faces);
    view.tree->tag_visibility_changed(all_coarse_nodes.index_range());
    return;
  }

  const Vector<int> dirty_quad_indices = dirty_root_quads_from_fine_nodes(view, dirty_visibility);
  if (dirty_quad_indices.is_empty()) {
    return;
  }

  if (dirty_quad_indices.size() > view.root_quads_num / 3) {
    multires.downsample_hidden_faces(
        base_mesh, view.fine_level, fine_hidden, view.coarse_level, view.hidden_faces);
    view.tree->tag_visibility_changed(all_coarse_nodes.index_range());
    return;
  }

  multires.downsample_hidden_faces(
      view.fine_level, fine_hidden, view.coarse_level, view.hidden_faces, dirty_quad_indices);

  IndexMaskMemory memory;
  const IndexMask coarse_mask = coarse_nodes_from_root_quads(
      view, base_mesh, multires, dirty_quad_indices, memory);
  if (!coarse_mask.is_empty()) {
    view.tree->tag_visibility_changed(coarse_mask);
  }
}

static void sync_colors(MultiresCoarseView &view,
                        const Mesh &base_mesh,
                        const Multiresolution &multires,
                        const BitVector<> &dirty_colors)
{
  IndexMaskMemory memory;
  const IndexMask changed_nodes = sync_vertex_attr<float4>(
      view, multires.colors(), view.colors, view_grid_size(view, multires), dirty_colors, memory);
  if (changed_nodes.is_empty()) {
    return;
  }
  if (base_mesh.active_color_attribute) {
    view.tree->tag_attribute_changed(changed_nodes, base_mesh.active_color_attribute);
  }
  view.tree->tag_attribute_changed(changed_nodes, "Color");
}

static std::unique_ptr<MultiresCoarseView> build_coarse_view(const Mesh &base_mesh,
                                                             const pbvh::Tree &fine_tree,
                                                             const int coarse_lvl)
{
  const Multiresolution &multires = *base_mesh.multires;
  const int fine_lvl = multires.curr_level;

  auto view = std::make_unique<MultiresCoarseView>();
  view->fine_level = fine_lvl;
  view->coarse_level = coarse_lvl;
  view->fine_tree = &fine_tree;

  BLI_assert(coarse_lvl <= multires.bvh_partitions_len);
  view->tree = std::make_unique<pbvh::Tree>(
      pbvh::Tree::from_stored_nodes_multires(coarse_lvl, base_mesh));
  const pbvh::Tree &coarse_tree = *view->tree;

  const BaseTopology base_topo = multires.get_base_topology(base_mesh, coarse_lvl);
  const int virtual_lvl = coarse_lvl - multires.base_level(coarse_lvl);
  view->faces_per_corner = Multiresolution::faces_per_corner(virtual_lvl);

  view->corner_verts.reinitialize(base_topo.corners_num * view->faces_per_corner * 4);
  calc_corner_verts(base_topo, virtual_lvl, coarse_tree.offsets, view->corner_verts);

  const int coarse_verts_num = total_verts_at_level(multires, base_mesh, coarse_lvl);

  Array<int> fine_indices(total_verts_at_level(multires, base_mesh, fine_lvl));
  array_utils::fill_index_range(fine_indices.as_mutable_span());
  view->to_fine_vert_map.reinitialize(coarse_verts_num);
  multires.downsample_to<int>(base_mesh,
                              fine_lvl,
                              fine_tree.offsets,
                              fine_indices,
                              coarse_lvl,
                              coarse_tree.offsets,
                              view->to_fine_vert_map);

  fine_tree.enable_dirty_tracking();
  build_fine_to_coarse_nodes(*view, multires, fine_tree);

  const bool all_quads = multires.base_all_quads;
  const int shift = all_quads ? 0 : 1;
  view->root_quads_num = all_quads ? base_mesh.faces_num : base_mesh.corners_num;
  view->fine_base_rel = std::max(0, multires.base_level(fine_lvl) - shift);
  view->coarse_base_rel = std::max(0, multires.base_level(coarse_lvl) - shift);

  view->coarse_face_to_node.reinitialize(base_topo.faces_num);
  const Span<pbvh::MultiresNode> coarse_nodes = coarse_tree.nodes<pbvh::MultiresNode>();
  for (const int k : coarse_nodes.index_range()) {
    const pbvh::MultiresNode &node = coarse_nodes[k];
    if ((node.flag_ & pbvh::Node::Leaf) != 0) {
      for (const int f : node.faces()) {
        view->coarse_face_to_node[f] = k;
      }
    }
  }

  fine_tree.take_dirty_nodes();

  sync_positions(*view, multires, {});
  if (!multires.masks().is_empty()) {
    sync_masks(*view, multires, {});
  }
  if (!multires.face_sets().is_empty()) {
    sync_face_sets(*view, base_mesh, multires, {});
  }
  if (!multires.hidden_faces().is_empty()) {
    sync_hidden_faces(*view, base_mesh, multires, {});
  }
  if (!multires.colors().is_empty()) {
    sync_colors(*view, base_mesh, multires, {});
  }

  return view;
}

MultiresCoarseView *MultiresolutionRuntime::ensure_coarse_view(const Mesh &base_mesh,
                                                               const pbvh::Tree &fine_tree)
{
  if (base_mesh.multires == nullptr || fine_tree.type() != pbvh::Type::Multires) {
    return nullptr;
  }
  const Multiresolution &multires = *base_mesh.multires;
  const int coarse_lvl = target_coarse_level(multires);
  if (coarse_lvl < 1) {
    this->coarse_view.reset();
    return nullptr;
  }

  if (!this->coarse_view || this->coarse_view->coarse_level != coarse_lvl ||
      this->coarse_view->fine_level != multires.curr_level ||
      this->coarse_view->fine_tree != &fine_tree)
  {
    this->coarse_view = build_coarse_view(base_mesh, fine_tree, coarse_lvl);
  }

  MultiresCoarseView &view = *this->coarse_view;

  const pbvh::DirtyNodes dirty = fine_tree.take_dirty_nodes();
  if (!dirty.positions.is_empty()) {
    sync_positions(view, multires, dirty.positions);
  }
  if (!dirty.masks.is_empty()) {
    sync_masks(view, multires, dirty.masks);
  }
  if (!dirty.face_sets.is_empty()) {
    sync_face_sets(view, base_mesh, multires, dirty.face_sets);
  }
  if (!dirty.visibility.is_empty()) {
    sync_hidden_faces(view, base_mesh, multires, dirty.visibility);
  }
  if (!dirty.colors.is_empty()) {
    sync_colors(view, base_mesh, multires, dirty.colors);
  }
  return &view;
}

}  // namespace blender::bke::multiresolution
