/* SPDX-FileCopyrightText: 2026 by Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BKE_context.hh"
#include "BKE_object_types.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "ED_sculpt.hh"

#include "BLT_translation.hh"

#include "UI_interface_layout.hh"
#include "UI_tree_view.hh"

#include "RNA_access.hh"
#include "RNA_prototypes.hh"

#include "WM_api.hh"

#include "ED_undo.hh"

namespace blender::ed::sculpt {

namespace {

void set_active_sculpt_layer(bContext &C, Mesh &mesh, const int index)
{
  Object *ob = CTX_data_active_object(&C);
  const bool in_sculpt = (ob->mode & OB_MODE_SCULPT) != 0 &&
                         ob->runtime->sculpt_session != nullptr;
  if (in_sculpt) {
    sculpt_paint::storage_undo::push_set_active_layer_begin(*ob);
  }
  PointerRNA mesh_ptr = RNA_pointer_create_discrete(&mesh.id, RNA_Mesh, &mesh);
  PropertyRNA *prop = RNA_struct_find_property(&mesh_ptr, "active_sculpt_layer_index");
  RNA_property_int_set(&mesh_ptr, prop, index);
  RNA_property_update(&C, &mesh_ptr, prop);
  if (in_sculpt) {
    sculpt_paint::storage_undo::push_end(*ob);
  }
  else {
    ED_undo_push(&C, "Set Active Sculpt Layer");
  }
}

}  // namespace

class SculptLayerTreeView : public ui::AbstractTreeView {
 public:
  Mesh &mesh_;

  explicit SculptLayerTreeView(Mesh &mesh) : mesh_(mesh)
  {
    is_flat_ = true;
  }

  void build_tree() override;
};

class SculptBaseLayerItem : public ui::AbstractTreeViewItem {
 private:
  Mesh &mesh_;

 public:
  explicit SculptBaseLayerItem(Mesh &mesh) : mesh_(mesh)
  {
    label_ = "Base";
  }

  void build_row(ui::Layout &row) override
  {
    row.use_property_decorate_set(false);
    uiItemL_ex(&row, label_, ICON_OUTLINER_OB_MESH, false, false);
  }

  std::optional<bool> should_be_active() const override
  {
    return mesh_.active_layer_index == -1;
  }

  void on_activate(bContext &C) override
  {
    set_active_sculpt_layer(C, mesh_, -1);
  }

  bool supports_renaming() const override
  {
    return false;
  }

  std::unique_ptr<ui::AbstractViewItemDragController> create_drag_controller() const override
  {
    return nullptr;
  }
};

class SculptLayerItem : public ui::AbstractTreeViewItem {
 private:
  Mesh &mesh_;
  SculptLayer &layer_;
  int index_;

 public:
  SculptLayerItem(Mesh &mesh, SculptLayer &layer, const int index)
      : mesh_(mesh), layer_(layer), index_(index)
  {
    label_ = layer.name;
  }

  void build_row(ui::Layout &row) override
  {
    PointerRNA layer_ptr = RNA_pointer_create_discrete(&mesh_.id, RNA_SculptLayer, &layer_);

    const bool visible = layer_.is_visible != 0;
    row.active_set(visible);
    row.use_property_decorate_set(false);

    uiItemL_ex(&row, label_, ICON_SHAPEKEY_DATA, false, false);

    ui::Layout &opacity_sub = row.row(true);
    opacity_sub.ui_units_x_set(5.0f);
    opacity_sub.prop(&layer_ptr, "opacity", ui::ITEM_R_SLIDER, "", ICON_NONE);

    ui::Layout &toggles_row = row.row(true);
    toggles_row.prop(&layer_ptr, "is_visible", ui::ITEM_R_ICON_ONLY, std::nullopt, ICON_NONE);
  }

  std::optional<bool> should_be_active() const override
  {
    return mesh_.active_layer_index == index_;
  }

  void on_activate(bContext &C) override
  {
    set_active_sculpt_layer(C, mesh_, index_);
  }

  bool supports_renaming() const override
  {
    return true;
  }

  bool rename(const bContext &C, StringRefNull new_name) override
  {
    Object *ob = CTX_data_active_object(&C);
    const bool in_sculpt = (ob != nullptr) && (ob->mode & OB_MODE_SCULPT) != 0 &&
                           ob->runtime->sculpt_session != nullptr;
    if (in_sculpt) {
      sculpt_paint::storage_undo::push_rename_layer_begin(*ob);
    }
    PointerRNA layer_ptr = RNA_pointer_create_discrete(&mesh_.id, RNA_SculptLayer, &layer_);
    RNA_string_set(&layer_ptr, "name", new_name.c_str());
    if (PropertyRNA *name_prop = RNA_struct_find_property(&layer_ptr, "name")) {
      RNA_property_update(const_cast<bContext *>(&C), &layer_ptr, name_prop);
    }
    if (in_sculpt) {
      sculpt_paint::storage_undo::push_end(*ob);
    }
    else {
      ED_undo_push(const_cast<bContext *>(&C), "Rename Sculpt Layer");
    }
    return true;
  }

  StringRef get_rename_string() const override
  {
    return label_;
  }
};

void SculptLayerTreeView::build_tree()
{
  for (int i = mesh_.sculpt_layers_len - 1; i >= 0; i--) {
    this->add_tree_item<SculptLayerItem>(mesh_, mesh_.sculpt_layers[i], i);
  }

  this->add_tree_item<SculptBaseLayerItem>(mesh_);
}

void template_sculpt_layer_tree(ui::Layout *layout, bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  if (!ob || ob->type != OB_MESH) {
    return;
  }

  Mesh *mesh = id_cast<Mesh *>(ob->data);

  ui::Block *block = layout->block();
  ui::AbstractTreeView *tree_view = block_add_view(
      *block, "Sculpt Layer Tree View", std::make_unique<SculptLayerTreeView>(*mesh));

  tree_view->set_default_rows(6);

  ui::TreeViewBuilder::build_tree_view(*C, *tree_view, *layout);
}

}  // namespace blender::ed::sculpt
