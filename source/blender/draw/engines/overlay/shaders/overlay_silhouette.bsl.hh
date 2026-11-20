/* SPDX-FileCopyrightText: 2026 Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "gpu_shader_compat.hh"
#include "gpu_shader_fullscreen.bsl.hh"

namespace overlay::silhouette {

struct VertOut {
  [[smooth]] float2 uv;
};

[[vertex]] void vert_main([[vertex_id]] const int &vert_id,
                          [[out]] VertOut &v_out,
                          [[position]] float4 &position)
{
  fullscreen_vertex(vert_id, position, v_out.uv);
}

struct FragOut {
  [[frag_color(0)]] float4 color;
};

struct DownsampleResources {
  [[sampler(0)]] const sampler2DDepth depth_tx;
  [[sampler(1)]] const sampler2DDepth depth_in_front_tx;

  [[push_constant]] const float2 visible_uv_min;
  [[push_constant]] const float2 visible_uv_size;
  [[push_constant]] const float2 cell_uv_size;
  [[push_constant]] const bool has_in_front_depth;
};

[[fragment]] void downsample_frag([[resource_table]] DownsampleResources &srt,
                                  [[in]] const VertOut &v_in,
                                  [[out]] FragOut &frag_out)
{
  float2 base_uv = srt.visible_uv_min + v_in.uv * srt.visible_uv_size;
  float2 uv_max = srt.visible_uv_min + srt.visible_uv_size;

  float coverage = 0.0f;
  for (int iy = 0; iy < 4; iy++) {
    for (int ix = 0; ix < 4; ix++) {
      float2 offset = ((float2(float(ix), float(iy)) + 0.5f) * 0.25f - 0.5f) * srt.cell_uv_size;
      float2 sample_uv = clamp(base_uv + offset, srt.visible_uv_min, uv_max);
      bool hit = (textureLod(srt.depth_tx, sample_uv, 0.0f).r < 1.0f);
      if (!hit && srt.has_in_front_depth) {
        hit = (textureLod(srt.depth_in_front_tx, sample_uv, 0.0f).r < 1.0f);
      }
      coverage += hit ? 1.0f : 0.0f;
    }
  }
  coverage *= (1.0f / 16.0f);

  frag_out.color = float4(coverage, 0.0f, 0.0f, 1.0f);
}

struct CompositeResources {
  [[sampler(0)]] const sampler2D silhouette_tx;

  [[push_constant]] const float4 silhouette_color;
  [[push_constant]] const float4 background_color;
  [[push_constant]] const float2 box_ndc_min;
  [[push_constant]] const float2 box_ndc_size;
  [[push_constant]] const float2 box_size_px;
  [[push_constant]] const float border_px;
};

[[vertex]] void composite_vert([[resource_table]] const CompositeResources &srt,
                               [[vertex_id]] const int &vert_id,
                               [[out]] VertOut &v_out,
                               [[position]] float4 &position)
{
  float2 uv = float2(float(vert_id & 1), float((vert_id >> 1) & 1));
  float2 ndc = srt.box_ndc_min + uv * srt.box_ndc_size;
  position = float4(ndc, 0.0f, 1.0f);
  v_out.uv = uv;
}

[[fragment]] void composite_frag([[resource_table]] const CompositeResources &srt,
                                 [[in]] const VertOut &v_in,
                                 [[out]] FragOut &frag_out)
{
  float2 px = v_in.uv * srt.box_size_px;
  float edge_dist = min(min(px.x, srt.box_size_px.x - px.x), min(px.y, srt.box_size_px.y - px.y));

  float coverage = textureLod(srt.silhouette_tx, v_in.uv, 0.0f).r;
  float4 sil_premul = float4(srt.silhouette_color.rgb * srt.silhouette_color.a,
                             srt.silhouette_color.a) *
                      coverage;
  float4 bg_premul = float4(srt.background_color.rgb * srt.background_color.a,
                            srt.background_color.a);
  float4 out_col = sil_premul + bg_premul * (1.0f - sil_premul.a);

  if (edge_dist < srt.border_px && srt.background_color.a > 0.0f) {
    float4 border_premul = float4(0.03f, 0.03f, 0.03f, 1.0f) * srt.background_color.a;
    out_col = border_premul + out_col * (1.0f - border_premul.a);
  }

  frag_out.color = out_col;
}

PipelineGraphic downsample(vert_main, downsample_frag);
PipelineGraphic composite(composite_vert, composite_frag);

}  // namespace overlay::silhouette
