/* SPDX-FileCopyrightText: 2026 Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup overlay
 */

#pragma once

#include "BLI_math_base.hh"
#include "BLI_rect.hh"

#include "BKE_screen.hh"

#include "DNA_userdef_types.h"

#include "ED_screen.hh"

#include "UI_interface_c.hh"

#include "overlay_base.hh"

namespace blender::draw::overlay {

class Silhouette : Overlay {
 private:
  TextureFromPool silhouette_tx_ = {"overlay_silhouette_tx"};
  Framebuffer downsample_fb_ = {"overlay_silhouette_downsample_fb"};

  PassSimple downsample_ps_ = {"SilhouetteDownsample"};
  PassSimple composite_ps_ = {"SilhouetteComposite"};

  int2 sil_size_ = int2(0);

 public:
  void begin_sync(Resources &res, const State &state) final
  {
    auto disable = [&]() {
      enabled_ = false;
      downsample_ps_.init();
      composite_ps_.init();
    };

    const bool has_visible_color = (state.overlay.silhouette_color[3] > 0.0f) ||
                                   (state.overlay.silhouette_background_color[3] > 0.0f);

    enabled_ = state.is_space_v3d() && !res.is_selection() && !state.is_depth_only_drawing &&
               !state.is_image_render && !state.is_viewport_image_render &&
               !state.is_viewport_xr && !state.hide_overlays && state.show_silhouette() &&
               has_visible_color && (state.region != nullptr) && (state.region->winx > 0) &&
               (state.region->winy > 0);

    if (enabled_ && state.region->alignment == RGN_ALIGN_QSPLIT) {
      enabled_ = (state.region->runtime->quadview_index == bke::ARegionQuadviewIndex::TopRight);
    }

    if (!enabled_) {
      disable();
      return;
    }

    const int2 win_size = int2(state.region->winx, state.region->winy);
    const rcti rect_visible = *ED_region_visible_rect(const_cast<ARegion *>(state.region));
    const int vis_w = BLI_rcti_size_x(&rect_visible);
    const int vis_h = BLI_rcti_size_y(&rect_visible);

    const float min_region_px = 96.0f * UI_SCALE_FAC;
    if (vis_w < int(min_region_px) || vis_h < int(min_region_px)) {
      disable();
      return;
    }

    const float sil_h_f = math::min(125.0f * UI_SCALE_FAC, float(vis_h) * 0.25f);
    const int sil_h = int(math::round(sil_h_f));
    const int sil_w = int(math::round(sil_h_f * (float(vis_w) / float(vis_h))));
    if (sil_w < 16 || sil_h < 16) {
      disable();
      return;
    }

    const int margin = int(math::round(10.0f * UI_SCALE_FAC));
    const int box_xmax = rect_visible.xmax - margin;
    const int box_ymin = rect_visible.ymin + margin;
    const int box_xmin = box_xmax - sil_w;
    const int box_ymax = box_ymin + sil_h;
    if (box_xmin < rect_visible.xmin || box_ymax > rect_visible.ymax) {
      disable();
      return;
    }

    sil_size_ = int2(sil_w, sil_h);

    const float2 win_f = float2(win_size);
    const float2 visible_uv_min = float2(rect_visible.xmin, rect_visible.ymin) / win_f;
    const float2 visible_uv_size = float2(vis_w, vis_h) / win_f;
    const float2 cell_uv_size = visible_uv_size / float2(sil_size_);
    const float2 box_ndc_min = (float2(box_xmin, box_ymin) / win_f) * 2.0f - 1.0f;
    const float2 box_ndc_size = (float2(sil_size_) / win_f) * 2.0f;
    const bool has_in_front_depth = !state.xray_enabled;

    {
      PassSimple &pass = downsample_ps_;
      pass.init();
      pass.state_set(DRW_STATE_WRITE_COLOR);
      pass.shader_set(res.shaders->silhouette_downsample.get());
      pass.bind_texture("depth_tx", &res.depth_tx);
      pass.bind_texture("depth_in_front_tx",
                        has_in_front_depth ? &res.depth_in_front_tx : &res.dummy_depth_tx);
      pass.push_constant("visible_uv_min", visible_uv_min);
      pass.push_constant("visible_uv_size", visible_uv_size);
      pass.push_constant("cell_uv_size", cell_uv_size);
      pass.push_constant("has_in_front_depth", has_in_front_depth);
      pass.draw_procedural(GPU_PRIM_TRIS, 1, 3);
    }

    const float border_px = math::max(1.0f, math::round(U.pixelsize));
    const float4 silhouette_color(state.overlay.silhouette_color);
    const float4 background_color(state.overlay.silhouette_background_color);

    {
      PassSimple &pass = composite_ps_;
      pass.init();
      pass.state_set(DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ALPHA_PREMUL);
      pass.shader_set(res.shaders->silhouette_composite.get());
      pass.bind_texture("silhouette_tx", &silhouette_tx_);
      pass.push_constant("silhouette_color", silhouette_color);
      pass.push_constant("background_color", background_color);
      pass.push_constant("box_ndc_min", box_ndc_min);
      pass.push_constant("box_ndc_size", box_ndc_size);
      pass.push_constant("box_size_px", float2(sil_size_));
      pass.push_constant("border_px", border_px);
      pass.draw_procedural(GPU_PRIM_TRI_STRIP, 1, 4);
    }
  }

  void capture_depth(Manager &manager)
  {
    if (!enabled_) {
      return;
    }

    eGPUTextureUsage usage = GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_ATTACHMENT;
    silhouette_tx_.acquire_2d(sil_size_, gpu::TextureFormat::UNORM_8, usage);
    downsample_fb_.ensure(GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(silhouette_tx_));
    GPU_framebuffer_bind(downsample_fb_);
    manager.submit(downsample_ps_);
  }

  void draw_output(Framebuffer &framebuffer, Manager &manager, View & /*view*/) final
  {
    if (!enabled_) {
      return;
    }

    GPU_framebuffer_bind(framebuffer);
    manager.submit(composite_ps_);
    silhouette_tx_.release();
  }
};

}  // namespace blender::draw::overlay
