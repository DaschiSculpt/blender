/* SPDX-FileCopyrightText: 2020 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edsculpt
 */

#include "MEM_guardedalloc.h"

#include "BLI_enumerable_thread_specific.hh"
#include "BLI_math_color_blend.hh"
#include "BLI_math_color_c.hh"
#include "BLI_math_vector.hh"

#include "BLT_translation.hh"

#include "BKE_attribute.hh"
#include "BKE_brush.hh"
#include "BKE_context.hh"
#include "BKE_layer.hh"
#include "BKE_mesh.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"

#include "IMB_colormanagement.hh"

#include "WM_api.hh"
#include "WM_toolsystem.hh"
#include "WM_types.hh"

#include "ED_paint.hh"
#include "ED_screen.hh"

#include "mesh_brush_common.hh"
#include "sculpt_automask.hh"
#include "sculpt_color.hh"
#include "sculpt_filter.hh"
#include "sculpt_intern.hh"
#include "sculpt_smooth.hh"
#include "sculpt_undo.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include <cmath>

namespace blender::ed::sculpt_paint::color {

enum class FilterType {
  Fill = 0,
  Hue,
  Saturation,
  Value,
  Brightness,
  Contrast,
  Red,
  Green,
  Blue,
  Smooth,
};

static const float fill_filter_default_color[4] = {1.0f, 1.0f, 1.0f, 1.0f};

static float3 fill_color_resolve_from_paint(const bContext *C, const bool use_secondary_color)
{
  const ToolSettings *ts = CTX_data_tool_settings(C);
  const Sculpt *sd = ts->sculpt;
  const Paint *paint = &sd->paint;
  const Brush *brush = BKE_paint_brush_for_read(paint);

  /* Use brush colors if a brush tool is active, otherwise use unified paint colors. */
  if (WM_toolsystem_active_tool_is_brush(C) && brush) {
    const float3 color = use_secondary_color ? BKE_brush_secondary_color_get(paint, brush) :
                                               BKE_brush_color_get(paint, brush);
    return color;
  }

  /* Use unified paint colors when filter tool is active. */
  const float3 color = use_secondary_color ? paint->unified_paint_settings.secondary_color :
                                             paint->unified_paint_settings.color;
  return color;
}

static float3 fill_color_resolve(const bContext *C, wmOperator *op, const bool use_secondary_color)
{
  if (RNA_struct_property_is_set(op->ptr, "fill_color")) {
    float3 fill_color;
    RNA_float_get_array(op->ptr, "fill_color", fill_color);
    return fill_color;
  }

  return fill_color_resolve_from_paint(C, use_secondary_color);
}

/* Stores the used color as the fill color to ensure the redo panel works as expected. */
static void fill_color_store_current(const bContext *C, wmOperator *op)
{
  const bool use_secondary_color = RNA_boolean_get(op->ptr, "use_secondary_color");
  const float3 fill_color = fill_color_resolve_from_paint(C, use_secondary_color);
  RNA_float_set_array(op->ptr, "fill_color", fill_color);
}

static EnumPropertyItem prop_color_filter_types[] = {
    {int(FilterType::Fill), "FILL", 0, "Fill", "Fill with a specific color"},
    {int(FilterType::Hue), "HUE", 0, "Hue", "Change hue"},
    {int(FilterType::Saturation), "SATURATION", 0, "Saturation", "Change saturation"},
    {int(FilterType::Value), "VALUE", 0, "Value", "Change value"},
    {int(FilterType::Brightness), "BRIGHTNESS", 0, "Brightness", "Change brightness"},
    {int(FilterType::Contrast), "CONTRAST", 0, "Contrast", "Change contrast"},
    {int(FilterType::Smooth), "SMOOTH", 0, "Smooth", "Smooth colors"},
    {int(FilterType::Red), "RED", 0, "Red", "Change red channel"},
    {int(FilterType::Green), "GREEN", 0, "Green", "Change green channel"},
    {int(FilterType::Blue), "BLUE", 0, "Blue", "Change blue channel"},
    {0, nullptr, 0, nullptr, nullptr},
};

struct LocalData {
  Vector<int> neighbor_offsets;
  Vector<int> neighbor_data;
};

BLI_NOINLINE static void clamp_factors(const MutableSpan<float> factors,
                                       const float min,
                                       const float max)
{
  for (float &factor : factors) {
    factor = std::clamp(factor, min, max);
  }
}

static void fill_filter(const float3 filter_fill_color,
                        const Span<float4> orig_colors,
                        const MutableSpan<float> factors,
                        const MutableSpan<float4> new_colors)
{
  clamp_factors(factors, 0.0f, 1.0f);
  const float4 fill_color_rgba = float4(filter_fill_color, 1.0f);

  for (const int i : new_colors.index_range()) {
    blend_color_mix_float(new_colors[i], orig_colors[i], factors[i] * fill_color_rgba);
  }
}

static void hue_filter(const Span<float4> orig_colors,
                       const MutableSpan<float> factors,
                       const MutableSpan<float4> new_colors)
{
  for (const int i : new_colors.index_range()) {
    float3 hsv_color;
    rgb_to_hsv_v(orig_colors[i], hsv_color);
    const float hue = hsv_color[0];
    hsv_color[0] = fmod((hsv_color[0] + fabs(factors[i])) - hue, 1);
    hsv_to_rgb_v(hsv_color, new_colors[i]);
  }
}

static void saturation_filter(const Span<float4> orig_colors,
                              const MutableSpan<float> factors,
                              const MutableSpan<float4> new_colors)
{
  for (const int i : new_colors.index_range()) {
    float3 hsv_color;
    rgb_to_hsv_v(orig_colors[i], hsv_color);

    if (hsv_color[1] > 0.001f) {
      hsv_color[1] = std::clamp(hsv_color[1] + factors[i] * hsv_color[1], 0.0f, 1.0f);
      hsv_to_rgb_v(hsv_color, new_colors[i]);
    }
    else {
      new_colors[i] = orig_colors[i];
    }
  }
}

static void value_filter(const Span<float4> orig_colors,
                         const MutableSpan<float> factors,
                         const MutableSpan<float4> new_colors)
{
  for (const int i : new_colors.index_range()) {
    float3 hsv_color;
    rgb_to_hsv_v(orig_colors[i], hsv_color);
    hsv_color[2] = std::clamp(hsv_color[2] + factors[i], 0.0f, 1.0f);
    hsv_to_rgb_v(hsv_color, new_colors[i]);
  }
}

static void component_filter(const int component,
                             const Span<float4> orig_colors,
                             const MutableSpan<float> factors,
                             const MutableSpan<float4> new_colors)
{
  for (const int i : new_colors.index_range()) {
    new_colors[i] = orig_colors[i];
    new_colors[i][component] = std::clamp(orig_colors[i][component] + factors[i], 0.0f, 1.0f);
  }
}

static void brightness_filter(const Span<float4> orig_colors,
                              const MutableSpan<float> factors,
                              const MutableSpan<float4> new_colors)
{
  clamp_factors(factors, -1.0f, 1.0f);
  for (const int i : new_colors.index_range()) {
    for (int component = 0; component < 3; component++) {
      new_colors[i][component] = std::clamp(orig_colors[i][component] + factors[i], 0.0f, 1.0f);
    }
  }
}

static void contrast_filter(const Span<float4> orig_colors,
                            const MutableSpan<float> factors,
                            const MutableSpan<float4> new_colors)
{
  clamp_factors(factors, -1.0f, 1.0f);
  for (const int i : new_colors.index_range()) {
    const float contrast = factors[i];
    float delta = contrast / 2.0f;
    float gain = 1.0f - delta * 2.0f;

    if (contrast > 0) {
      gain = 1.0f / ((gain != 0.0f) ? gain : FLT_EPSILON);
    }

    const float offset = gain * (-delta);

    for (int component = 0; component < 3; component++) {
      new_colors[i][component] = std::clamp(gain * orig_colors[i][component] + offset, 0.0f, 1.0f);
    }
  }
}

static void color_filter_task_mesh(const Depsgraph &depsgraph,
                                   Object &ob,
                                   const OffsetIndices<int> faces,
                                   const Span<int> corner_verts,
                                   const GroupedSpan<int> vert_to_face_map,
                                   const MeshAttributeData &attribute_data,
                                   const FilterType mode,
                                   const float filter_strength,
                                   const float3 &filter_fill_color,
                                   const bke::pbvh::MeshNode &node,
                                   LocalData &tls,
                                   bke::GSpanAttributeWriter &color_attribute)
{
  SculptSession &ss = *ob.runtime->sculpt_session;
  const Span<float4> orig_colors = orig_color_data_get_mesh(ob, node);
  const Span<int> verts = node.verts();

  Array<float, bke::pbvh::MESH_LEAF_LIMIT> factors(verts.size());
  fill_factor_from_hide_and_mask(attribute_data.hide_vert, attribute_data.mask, verts, factors);
  auto_mask::calc_vert_factors(
      depsgraph, ob, ss.filter_cache->automasking.get(), node, verts, factors);
  scale_factors(factors, filter_strength);

  Array<float4, bke::pbvh::MESH_LEAF_LIMIT> new_colors(verts.size());

  /* Copy alpha. */
  for (const int i : verts.index_range()) {
    new_colors[i][3] = orig_colors[i][3];
  }

  switch (mode) {
    case FilterType::Fill: {
      fill_filter(filter_fill_color, orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Hue: {
      hue_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Saturation: {
      saturation_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Value: {
      value_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Red: {
      component_filter(0, orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Green: {
      component_filter(1, orig_colors, factors, new_colors);

      break;
    }
    case FilterType::Blue: {
      component_filter(2, orig_colors, factors, new_colors);

      break;
    }
    case FilterType::Brightness: {
      brightness_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Contrast: {
      contrast_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Smooth: {
      clamp_factors(factors, -1.0f, 1.0f);

      Array<float4, bke::pbvh::MESH_LEAF_LIMIT> colors(verts.size());
      for (const int i : verts.index_range()) {
        colors[i] = color_vert_get(faces,
                                   corner_verts,
                                   vert_to_face_map,
                                   color_attribute.span,
                                   color_attribute.domain,
                                   verts[i]);
      }

      const GroupedSpan<int> neighbors = calc_vert_neighbors(faces,
                                                             corner_verts,
                                                             vert_to_face_map,
                                                             {},
                                                             verts,
                                                             tls.neighbor_offsets,
                                                             tls.neighbor_data);

      Array<float4, bke::pbvh::MESH_LEAF_LIMIT> average_colors(verts.size());
      smooth::neighbor_color_average(faces,
                                     corner_verts,
                                     vert_to_face_map,
                                     color_attribute.span,
                                     color_attribute.domain,
                                     neighbors,
                                     average_colors);

      for (const int i : verts.index_range()) {
        const int vert = verts[i];

        if (factors[i] < 0.0f) {
          interp_v4_v4v4(average_colors[i], average_colors[i], colors[i], 0.5f);
        }

        bool copy_alpha = colors[i][3] == average_colors[i][3];

        if (factors[i] < 0.0f) {
          float4 delta_color;

          /* Unsharp mask. */
          delta_color = ss.filter_cache->pre_smoothed_color[vert] - average_colors[i];
          new_colors[i] += delta_color * factors[i];
        }
        else {
          blend_color_interpolate_float(new_colors[i], colors[i], average_colors[i], factors[i]);
        }

        new_colors[i] = math::clamp(new_colors[i], 0.0f, 1.0f);

        /* Prevent accumulated numeric error from corrupting alpha. */
        if (copy_alpha) {
          new_colors[i][3] = average_colors[i][3];
        }
      }
      break;
    }
  }

  for (const int i : verts.index_range()) {
    color_vert_set(faces,
                   corner_verts,
                   vert_to_face_map,
                   color_attribute.domain,
                   verts[i],
                   new_colors[i],
                   color_attribute.span);
  }
}

static void color_filter_task_multires(const Depsgraph &depsgraph,
                                       Object &ob,
                                       Multiresolution &multires,
                                       const bke::multiresolution::BaseTopology &base_topo,
                                       const FilterType mode,
                                       const float filter_strength,
                                       const float3 filter_fill_color,
                                       const bke::pbvh::MultiresNode &node)
{
  SculptSession &ss = *ob.runtime->sculpt_session;
  const bke::pbvh::Tree &pbvh = *ss.pbvh;
  const int grid_size = multires.grid_size();
  const Span<bool> hidden_verts = multires.hidden_verts();
  const Span<float4> global_colors = multires.colors();
  const IndexRange node_range = node.range(grid_size);
  const MutableSpan<float4> node_colors = multires.colors_for_write().slice(node_range);
  const Span<float4> orig_colors = orig_color_data_get_multires(ob, node);

  Array<float, bke::pbvh::MULTIRES_LEAF_LIMIT> factors(node_colors.size());
  fill_factor_from_hide_and_mask(hidden_verts, multires.masks(), node_range, factors);
  auto_mask::calc_multires_factors(
      depsgraph, ob, multires, base_topo, ss.filter_cache->automasking.get(), node, factors);
  scale_factors(factors, filter_strength);

  const MutableSpan<float4> new_colors = node_colors;

  for (const int i : new_colors.index_range()) {
    new_colors[i][3] = orig_colors[i][3];
  }

  switch (mode) {
    case FilterType::Fill: {
      fill_filter(filter_fill_color, orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Hue: {
      hue_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Saturation: {
      saturation_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Value: {
      value_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Red: {
      component_filter(0, orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Green: {
      component_filter(1, orig_colors, factors, new_colors);

      break;
    }
    case FilterType::Blue: {
      component_filter(2, orig_colors, factors, new_colors);

      break;
    }
    case FilterType::Brightness: {
      brightness_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Contrast: {
      contrast_filter(orig_colors, factors, new_colors);
      break;
    }
    case FilterType::Smooth: {
      const int offset = node.offset();
      clamp_factors(factors, -1.0f, 1.0f);

      Array<float4, bke::pbvh::MULTIRES_LEAF_LIMIT> average_colors(node_colors.size());

      const int verts_count = node.verts().size();
      const int edges_count = node.edges().size() * grid_size;
      const int faces_count = node.faces().size() * grid_size * grid_size;
      const IndexRange local_verts_range(0, verts_count);
      const IndexRange local_edges_range(verts_count, edges_count);
      const IndexRange local_faces_range(verts_count + edges_count, faces_count);

      smooth::neighbor_data_average_node_multires<float4>(
          multires, base_topo, node, pbvh.offsets, global_colors, average_colors, hidden_verts);

      for (const int i : node_colors.index_range()) {
        if (factors[i] < 0.0f) {
          interp_v4_v4v4(average_colors[i], average_colors[i], node_colors[i], 0.5f);
        }

        bool copy_alpha = node_colors[i][3] == average_colors[i][3];

        if (factors[i] < 0.0f) {
          float4 delta_color;

          /* Unsharp mask. */
          delta_color = ss.filter_cache->pre_smoothed_color[offset + i] - average_colors[i];
          new_colors[i] += delta_color * factors[i];
        }
        else {
          blend_color_interpolate_float(
              new_colors[i], node_colors[i], average_colors[i], factors[i]);
        }

        new_colors[i] = math::clamp(new_colors[i], 0.0f, 1.0f);

        /* Prevent accumulated numeric error from corrupting alpha. */
        if (copy_alpha) {
          new_colors[i][3] = average_colors[i][3];
        }
      }
      break;
    }
  }
}

static void sculpt_color_presmooth_init_mesh(const Mesh &mesh, Object &object)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
  const IndexMask &node_mask = ss.filter_cache->node_mask;
  const OffsetIndices<int> faces = mesh.faces();
  const Span<int> corner_verts = mesh.corner_verts();
  const GroupedSpan<int> vert_to_face_map = mesh.vert_to_face_map();
  const bke::GAttributeReader color_attribute = active_color_attribute(mesh);
  const GVArraySpan colors = *color_attribute;

  if (ss.filter_cache->pre_smoothed_color.is_empty()) {
    ss.filter_cache->pre_smoothed_color = Array<float4>(mesh.verts_num);
  }
  const MutableSpan<float4> pre_smoothed_color = ss.filter_cache->pre_smoothed_color;

  node_mask.foreach_index(
      [&](const int i) {
        for (const int vert : nodes[i].verts()) {
          pre_smoothed_color[vert] = color_vert_get(
              faces, corner_verts, vert_to_face_map, colors, color_attribute.domain, vert);
        }
      },
      exec_mode::grain_size(1));

  struct LocalData {
    Vector<int> neighbor_offsets;
    Vector<int> neighbor_data;
    Vector<float4> averaged_colors;
  };
  threading::EnumerableThreadSpecific<LocalData> all_tls;
  for ([[maybe_unused]] const int iteration : IndexRange(2)) {
    node_mask.foreach_index(
        [&](const int i) {
          LocalData &tls = all_tls.local();
          const Span<int> verts = nodes[i].verts();

          const GroupedSpan<int> neighbors = calc_vert_neighbors(faces,
                                                                 corner_verts,
                                                                 vert_to_face_map,
                                                                 {},
                                                                 verts,
                                                                 tls.neighbor_offsets,
                                                                 tls.neighbor_data);

          Array<float4, bke::pbvh::MESH_LEAF_LIMIT> averaged_colors(verts.size());
          smooth::neighbor_data_average_mesh(
              pre_smoothed_color.as_span(), neighbors, averaged_colors.as_mutable_span());

          for (const int i : verts.index_range()) {
            pre_smoothed_color[verts[i]] = math::interpolate(
                pre_smoothed_color[verts[i]], averaged_colors[i], 0.5f);
          }
        },
        exec_mode::grain_size(1));
  }
}

static void sculpt_color_presmooth_init_multires(Object &object)
{
  SculptSession &ss = *object.runtime->sculpt_session;
  const Mesh &base_mesh = *id_cast<Mesh *>(object.data);
  const Multiresolution &multires = *base_mesh.multires;
  const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(base_mesh);
  const int grid_size = multires.grid_size();
  const Span<bool> hidden_verts = multires.hidden_verts();
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(object);
  MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
  const IndexMask &node_mask = ss.filter_cache->node_mask;

  if (ss.filter_cache->pre_smoothed_color.is_empty()) {
    ss.filter_cache->pre_smoothed_color = Array<float4>(multires.verts_num());
  }
  const MutableSpan<float4> pre_smoothed_color = ss.filter_cache->pre_smoothed_color;
  pre_smoothed_color.copy_from(multires.colors());

  for ([[maybe_unused]] const int iteration : IndexRange(2)) {
    node_mask.foreach_index(
        [&](const int i) {
          const int offset = nodes[i].offset();
          const Span<float4> colors = multires.colors().slice(
              nodes[i].range(multires.grid_size()));

          const bke::pbvh::MultiresNode &node = nodes[i];
          const int verts_count = node.verts().size();
          const int edges_count = node.edges().size() * grid_size;
          const int faces_count = node.faces().size() * grid_size * grid_size;
          const IndexRange local_verts_range(0, verts_count);
          const IndexRange local_edges_range(verts_count, edges_count);
          const IndexRange local_faces_range(verts_count + edges_count, faces_count);

          Array<float4, bke::pbvh::MULTIRES_LEAF_LIMIT> averaged_colors(colors.size());

          smooth::neighbor_data_average_node_multires<float4>(multires,
                                                              base_topo,
                                                              node,
                                                              pbvh.offsets,
                                                              pre_smoothed_color,
                                                              averaged_colors,
                                                              hidden_verts);

          for (const int i : colors.index_range()) {
            pre_smoothed_color[offset + i] = math::interpolate(
                pre_smoothed_color[offset + i], averaged_colors[i], 0.5f);
          }
        },
        exec_mode::grain_size(1));
  }
}

static void sculpt_color_filter_apply(bContext *C, wmOperator *op, Object &ob)
{
  const Depsgraph &depsgraph = *CTX_data_depsgraph_pointer(C);
  const Sculpt &sd = *CTX_data_tool_settings(C)->sculpt;
  SculptSession &ss = *ob.runtime->sculpt_session;
  bke::pbvh::Tree &pbvh = *bke::object::pbvh_get(ob);

  const FilterType mode = FilterType(RNA_enum_get(op->ptr, "type"));
  float filter_strength = RNA_float_get(op->ptr, "strength");
  const bool use_secondary_color = RNA_boolean_get(op->ptr, "use_secondary_color");
  const float3 fill_color = fill_color_resolve(C, op, use_secondary_color);

  Mesh &mesh = *id_cast<Mesh *>(ob.data);
  const bool presmooth = filter_strength < 0.0 && ss.filter_cache->pre_smoothed_color.is_empty() &&
                         mode == FilterType::Smooth;

  const IndexMask &node_mask = ss.filter_cache->node_mask;
  if (auto_mask::is_enabled(sd.paint, ob, nullptr) && ss.filter_cache->automasking &&
      ss.filter_cache->automasking->settings.flags & BRUSH_AUTOMASKING_CAVITY_ALL)
  {
    ss.filter_cache->automasking->calc_cavity_factor(depsgraph, ob, node_mask);
  }

  switch (pbvh.type()) {
    case bke::pbvh::Type::Mesh: {
      if (presmooth) {
        sculpt_color_presmooth_init_mesh(mesh, ob);
      }
      const OffsetIndices<int> faces = mesh.faces();
      const Span<int> corner_verts = mesh.corner_verts();
      const GroupedSpan<int> vert_to_face_map = mesh.vert_to_face_map();
      bke::GSpanAttributeWriter color_attribute = active_color_attribute_for_write(mesh);
      const MeshAttributeData attribute_data(mesh);
      MutableSpan<bke::pbvh::MeshNode> nodes = pbvh.nodes<bke::pbvh::MeshNode>();
      threading::EnumerableThreadSpecific<LocalData> all_tls;
      node_mask.foreach_index(
          [&](const int i) {
            LocalData &tls = all_tls.local();
            color_filter_task_mesh(depsgraph,
                                   ob,
                                   faces,
                                   corner_verts,
                                   vert_to_face_map,
                                   attribute_data,
                                   mode,
                                   filter_strength,
                                   fill_color,
                                   nodes[i],
                                   tls,
                                   color_attribute);
          },
          exec_mode::grain_size(1));
      color_attribute.finish();
      break;
    }
    case bke::pbvh::Type::Multires: {
      Multiresolution &multires = *mesh.multires;
      const bke::multiresolution::BaseTopology base_topo = multires.get_base_topology(mesh);
      multires.ensure_colors();
      if (presmooth) {
        sculpt_color_presmooth_init_multires(ob);
      }
      MutableSpan<bke::pbvh::MultiresNode> nodes = pbvh.nodes<bke::pbvh::MultiresNode>();
      node_mask.foreach_index(
          [&](const int i) {
            color_filter_task_multires(
                depsgraph, ob, multires, base_topo, mode, filter_strength, fill_color, nodes[i]);
          },
          exec_mode::grain_size(1));
      break;
    }
    case bke::pbvh::Type::Grids:
    case bke::pbvh::Type::BMesh:
      BLI_assert_unreachable();
      break;
  }

  pbvh.tag_attribute_changed(node_mask, mesh.active_color_attribute);
  flush_update_step(C, UpdateType::Color);
}

static void sculpt_color_filter_end(bContext *C, wmOperator *op, Object &ob)
{
  SculptSession &ss = *ob.runtime->sculpt_session;

  if (FilterType(RNA_enum_get(op->ptr, "type")) == FilterType::Fill &&
      !RNA_struct_property_is_set(op->ptr, "fill_color"))
  {
    fill_color_store_current(C, op);
  }

  undo::push_end(ob);
  MEM_delete(ss.filter_cache);
  ss.filter_cache = nullptr;
  flush_update_done(C, ob, UpdateType::Color);
}

static wmOperatorStatus sculpt_color_filter_modal(bContext *C,
                                                  wmOperator *op,
                                                  const wmEvent *event)
{
  Object &ob = *CTX_data_active_object(C);
  SculptSession &ss = *ob.runtime->sculpt_session;

  if (event->type == LEFTMOUSE && event->val == KM_RELEASE) {
    /* For Fill: if this was a click (not a drag), apply once at the tool's configured strength */
    if (FilterType(RNA_enum_get(op->ptr, "type")) == FilterType::Fill &&
        !ss.filter_cache->has_dragged)
    {
      RNA_float_set(op->ptr, "strength", ss.filter_cache->start_filter_strength);
      sculpt_color_filter_apply(C, op, ob);
    }

    sculpt_color_filter_end(C, op, ob);
    return OPERATOR_FINISHED;
  }

  if (event->type != MOUSEMOVE) {
    return OPERATOR_RUNNING_MODAL;
  }

  /* Use a pixel threshold to distinguish a click from a drag */
  int2 start_mouse;
  RNA_int_get_array(op->ptr, "start_mouse", &start_mouse[0]);
  const int2 mouse_2d(event->mval[0], event->mval[1]);

  const int drag_threshold = WM_event_drag_threshold(event);
  ss.filter_cache->has_dragged |= math::distance_manhattan(start_mouse, mouse_2d) > drag_threshold;

  if (!ss.filter_cache->has_dragged) {
    return OPERATOR_RUNNING_MODAL;
  }

  const float len = (start_mouse[0] - event->mval[0]) * 0.001f;
  float filter_strength = ss.filter_cache->start_filter_strength * -len;

  RNA_float_set(op->ptr, "strength", filter_strength);

  sculpt_color_filter_apply(C, op, ob);

  return OPERATOR_RUNNING_MODAL;
}

static int sculpt_color_filter_init(bContext *C, wmOperator *op)
{
  const Scene &scene = *CTX_data_scene(C);
  Object &ob = *CTX_data_active_object(C);
  Sculpt &sd = *CTX_data_tool_settings(C)->sculpt;
  View3D *v3d = CTX_wm_view3d(C);

  const Base *base = CTX_data_active_base(C);
  if (!BKE_base_is_visible(v3d, base)) {
    return OPERATOR_CANCELLED;
  }

  int mval[2];
  RNA_int_get_array(op->ptr, "start_mouse", mval);
  float mval_fl[2] = {float(mval[0]), float(mval[1])};

  const bool use_automasking = auto_mask::is_enabled(sd.paint, ob, nullptr);
  if (use_automasking) {
    if (v3d) {
      /* Update the active face set manually as the paint cursor is not enabled when using the Mesh
       * Filter Tool. */
      cursor_geometry_info_update(C, mval_fl, false);
    }
  }

  /* Disable for multires and dyntopo for now */
  if (!color_supported_check(scene, ob, op->reports)) {
    return OPERATOR_CANCELLED;
  }

  /* Ensure that we have a PBVH to be able to push changes on only visible nodes. */
  bke::object::pbvh_ensure(*CTX_data_ensure_evaluated_depsgraph(C), ob);

  undo::push_begin(scene, ob, op);
  BKE_sculpt_color_layer_create_if_needed(&ob);

  /* CTX_data_ensure_evaluated_depsgraph should be used at the end to include the potential
   * creation of color layer data. */
  Depsgraph *depsgraph = CTX_data_ensure_evaluated_depsgraph(C);
  BKE_sculptsession_update_for_edit(depsgraph, &ob, true);

  filter::cache_init(C,
                     ob,
                     sd,
                     undo::Type::Color,
                     mval_fl,
                     RNA_float_get(op->ptr, "area_normal_radius"),
                     RNA_float_get(op->ptr, "strength"));
  const SculptSession &ss = *ob.runtime->sculpt_session;
  filter::Cache *filter_cache = ss.filter_cache;
  filter_cache->active_face_set = face_set_none_id;
  if (auto_mask::is_enabled(sd.paint, ob, nullptr)) {
    auto_mask::filter_cache_ensure(*depsgraph, sd.paint, ob);
  }

  return OPERATOR_PASS_THROUGH;
}

static wmOperatorStatus sculpt_color_filter_exec(bContext *C, wmOperator *op)
{
  Object &ob = *CTX_data_active_object(C);

  if (sculpt_color_filter_init(C, op) == OPERATOR_CANCELLED) {
    return OPERATOR_CANCELLED;
  }

  sculpt_color_filter_apply(C, op, ob);
  sculpt_color_filter_end(C, op, ob);

  return OPERATOR_FINISHED;
}

static wmOperatorStatus sculpt_color_filter_invoke(bContext *C,
                                                   wmOperator *op,
                                                   const wmEvent *event)
{
  Object &ob = *CTX_data_active_object(C);
  View3D *v3d = CTX_wm_view3d(C);
  if (v3d && v3d->shading.type == OB_SOLID) {
    v3d->shading.color_type = V3D_SHADING_VERTEX_COLOR;
  }

  RNA_int_set_array(op->ptr, "start_mouse", event->mval);

  /* Immediate execution path (used by key-bindings like `Ctrl+X`). */
  if (RNA_boolean_get(op->ptr, "use_immediate")) {
    RNA_float_set(op->ptr, "strength", 1.0f);
  }

  if (sculpt_color_filter_init(C, op) == OPERATOR_CANCELLED) {
    return OPERATOR_CANCELLED;
  }

  if (RNA_boolean_get(op->ptr, "use_immediate")) {
    sculpt_color_filter_apply(C, op, ob);
    sculpt_color_filter_end(C, op, ob);
    return OPERATOR_FINISHED;
  }

  ED_paint_brush_type_update_sticky_shading_color(C, &ob);

  WM_event_add_modal_handler(C, op);
  return OPERATOR_RUNNING_MODAL;
}

static std::string sculpt_color_filter_get_name(wmOperatorType * /*ot*/, PointerRNA *ptr)
{
  PropertyRNA *prop = RNA_struct_find_property(ptr, "type");
  const int value = RNA_property_enum_get(ptr, prop);
  const char *ui_name = nullptr;

  RNA_property_enum_name_gettexted(nullptr, ptr, prop, value, &ui_name);
  return ui_name;
}

static void sculpt_color_filter_ui(bContext * /*C*/, wmOperator *op)
{
  ui::Layout &layout = *op->layout;

  layout.prop(op->ptr, "strength", UI_ITEM_NONE, std::nullopt, ICON_NONE);

  if (FilterType(RNA_enum_get(op->ptr, "type")) == FilterType::Fill) {
    layout.prop(op->ptr, "fill_color", UI_ITEM_NONE, std::nullopt, ICON_NONE);
  }
}

void SCULPT_OT_color_filter(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Filter Color";
  ot->idname = "SCULPT_OT_color_filter";
  ot->description = "Applies a filter to modify the active color attribute";

  /* API callbacks. */
  ot->invoke = sculpt_color_filter_invoke;
  ot->exec = sculpt_color_filter_exec;
  ot->modal = sculpt_color_filter_modal;
  ot->poll = sculpt_mode_poll;
  ot->ui = sculpt_color_filter_ui;
  ot->get_name = sculpt_color_filter_get_name;

  ot->flag = OPTYPE_REGISTER | OPTYPE_UNDO;

  /* rna */
  filter::register_operator_props(ot);

  RNA_def_enum(
      ot->srna, "type", prop_color_filter_types, int(FilterType::Fill), "Filter Type", "");

  PropertyRNA *prop = RNA_def_float_color(ot->srna,
                                          "fill_color",
                                          3,
                                          fill_filter_default_color,
                                          0.0f,
                                          FLT_MAX,
                                          "Fill Color",
                                          "",
                                          0.0f,
                                          1.0f);
  RNA_def_property_translation_context(prop, BLT_I18NCONTEXT_ID_MESH);
  RNA_def_property_subtype(prop, PROP_COLOR);

  prop = RNA_def_boolean(ot->srna,
                         "use_immediate",
                         false,
                         "Immediate Apply",
                         "Apply once without entering modal interaction");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);

  prop = RNA_def_boolean(ot->srna, "use_secondary_color", false, "Use Secondary Color", "");
  RNA_def_property_flag(prop, PROP_HIDDEN | PROP_SKIP_SAVE);
}

}  // namespace blender::ed::sculpt_paint::color
