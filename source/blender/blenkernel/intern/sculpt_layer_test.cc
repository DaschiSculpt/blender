/* SPDX-FileCopyrightText: 2026 Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "testing/testing.h"

#include "BKE_gtest_base.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.h"
#include "BKE_sculpt_layer.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BLI_array.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

namespace blender::bke::sculpt_layer::tests {

class SculptLayerTest : public BlenderGTestBase {
 public:
  Mesh *mesh = nullptr;

  void TearDown() override
  {
    if (mesh != nullptr) {
      BKE_id_free(nullptr, mesh);
      mesh = nullptr;
    }
  }

  static Mesh *create_test_mesh()
  {
    Mesh *mesh = BKE_mesh_new_nomain(4, 4, 1, 4);

    MutableSpan<float3> positions = mesh->vert_positions_for_write();
    positions[0] = float3(0.0f, 0.0f, 0.0f);
    positions[1] = float3(1.0f, 0.0f, 0.0f);
    positions[2] = float3(1.0f, 1.0f, 0.0f);
    positions[3] = float3(0.0f, 1.0f, 0.0f);

    MutableSpan<int2> edges = mesh->edges_for_write();
    edges[0] = int2(0, 1);
    edges[1] = int2(1, 2);
    edges[2] = int2(2, 3);
    edges[3] = int2(3, 0);

    MutableSpan<int> face_offsets = mesh->face_offsets_for_write();
    face_offsets[0] = 0;
    face_offsets[1] = 4;

    MutableSpan<int> corner_verts = mesh->corner_verts_for_write();
    corner_verts[0] = 0;
    corner_verts[1] = 1;
    corner_verts[2] = 2;
    corner_verts[3] = 3;

    MutableSpan<int> corner_edges = mesh->corner_edges_for_write();
    corner_edges[0] = 0;
    corner_edges[1] = 1;
    corner_edges[2] = 2;
    corner_edges[3] = 3;

    mesh->tag_topology_changed();
    return mesh;
  }
};

TEST_F(SculptLayerTest, AddAndRemoveLayers)
{
  mesh = create_test_mesh();
  ASSERT_NE(mesh, nullptr);

  EXPECT_EQ(mesh->sculpt_layers_len, 0);
  EXPECT_EQ(mesh->active_layer_index, -1);
  EXPECT_EQ(active_layer(*mesh), nullptr);

  /* Add first layer. */
  add_layer(*mesh, "Layer 1");
  EXPECT_EQ(mesh->sculpt_layers_len, 1);
  EXPECT_EQ(mesh->active_layer_index, 0);

  SculptLayer *layer0 = active_layer(*mesh);
  ASSERT_NE(layer0, nullptr);
  EXPECT_STREQ(layer0->name, "Layer 1");
  EXPECT_FLOAT_EQ(layer0->opacity, 1.0f);
  EXPECT_TRUE(layer0->is_visible);

  add_layer(*mesh, "Layer 1");
  EXPECT_EQ(mesh->sculpt_layers_len, 2);
  EXPECT_EQ(mesh->active_layer_index, 1);

  SculptLayer *layer1 = active_layer(*mesh);
  ASSERT_NE(layer1, nullptr);
  EXPECT_STRNE(layer1->name, "Layer 1");
  EXPECT_STREQ(layer1->name, "Layer 1.001");

  /* Remove the active (second) layer. */
  remove_active_layer(*mesh);
  EXPECT_EQ(mesh->sculpt_layers_len, 1);
  EXPECT_EQ(mesh->active_layer_index, 0);
  EXPECT_STREQ(active_layer(*mesh)->name, "Layer 1");

  remove_active_layer(*mesh);
  EXPECT_EQ(mesh->sculpt_layers_len, 0);
  EXPECT_EQ(mesh->active_layer_index, -1);
  EXPECT_EQ(mesh->sculpt_layers, nullptr);
}

TEST_F(SculptLayerTest, DuplicateLayer)
{
  mesh = create_test_mesh();

  add_layer(*mesh, "Detail");
  MutableSpan<float3> disp = layer_displacements_for_write(*mesh, "Detail");
  disp[0] = float3(1.0f, 2.0f, 3.0f);

  duplicate_active_layer(*mesh);

  EXPECT_EQ(mesh->sculpt_layers_len, 2);
  EXPECT_EQ(mesh->active_layer_index, 1);

  SculptLayer *duplicated = active_layer(*mesh);
  ASSERT_NE(duplicated, nullptr);
  EXPECT_STREQ(duplicated->name, "Detail.001");

  Span<float3> dup_disp = layer_displacements(*mesh, duplicated->name);
  ASSERT_EQ(dup_disp.size(), mesh->verts_num);
  EXPECT_EQ(dup_disp[0], float3(1.0f, 2.0f, 3.0f));

  MutableSpan<float3> dup_disp_mut = layer_displacements_for_write(*mesh, duplicated->name);
  dup_disp_mut[0] = float3(10.0f, 20.0f, 30.0f);

  Span<float3> orig_disp = layer_displacements(*mesh, "Detail");
  EXPECT_EQ(orig_disp[0], float3(1.0f, 2.0f, 3.0f));
}

TEST_F(SculptLayerTest, RenameLayer)
{
  mesh = create_test_mesh();

  add_layer(*mesh, "OldName");
  MutableSpan<float3> disp = layer_displacements_for_write(*mesh, "OldName");
  disp[0] = float3(0.5f, 0.5f, 0.5f);

  SculptLayer *layer = active_layer(*mesh);
  rename_layer(*mesh, *layer, "NewName");

  EXPECT_STREQ(layer->name, "NewName");

  Span<float3> renamed_disp = layer_displacements(*mesh, "NewName");
  ASSERT_FALSE(renamed_disp.is_empty());
  EXPECT_EQ(renamed_disp[0], float3(0.5f, 0.5f, 0.5f));

  /* The old attribute should no longer exist. */
  Span<float3> old_disp = layer_displacements(*mesh, "OldName");
  EXPECT_TRUE(old_disp.is_empty());
}

TEST_F(SculptLayerTest, DisplacementEvaluation)
{
  mesh = create_test_mesh();

  add_layer(*mesh, "Layer A");
  layer_displacements_for_write(*mesh, "Layer A")[0] = float3(0.0f, 0.0f, 2.0f);

  add_layer(*mesh, "Layer B");
  layer_displacements_for_write(*mesh, "Layer B")[0] = float3(0.0f, 4.0f, 0.0f);
  mesh->sculpt_layers[1].opacity = 0.5f;

  Array<float3> eval_positions(mesh->verts_num);
  calc_eval_positions(*mesh, eval_positions.as_mutable_span());

  EXPECT_FLOAT_EQ(eval_positions[0].x, 0.0f);
  EXPECT_FLOAT_EQ(eval_positions[0].y, 2.0f);
  EXPECT_FLOAT_EQ(eval_positions[0].z, 2.0f);

  mesh->sculpt_layers[1].is_visible = false;
  calc_eval_positions(*mesh, eval_positions.as_mutable_span());

  EXPECT_FLOAT_EQ(eval_positions[0].x, 0.0f);
  EXPECT_FLOAT_EQ(eval_positions[0].y, 0.0f);
  EXPECT_FLOAT_EQ(eval_positions[0].z, 2.0f);
}

TEST_F(SculptLayerTest, MergeActiveLayerToBase)
{
  mesh = create_test_mesh();

  mesh->vert_positions_for_write()[0] = float3(1.0f, 1.0f, 1.0f);

  add_layer(*mesh, "OffsetLayer");
  layer_displacements_for_write(*mesh, "OffsetLayer")[0] = float3(0.0f, 0.0f, 2.0f);
  mesh->sculpt_layers[0].opacity = 0.5f;

  merge_active_layer_to_base(*mesh);

  EXPECT_EQ(mesh->sculpt_layers_len, 0);
  EXPECT_EQ(mesh->active_layer_index, -1);

  const Span<float3> base_positions = mesh->vert_positions();
  EXPECT_FLOAT_EQ(base_positions[0].x, 1.0f);
  EXPECT_FLOAT_EQ(base_positions[0].y, 1.0f);
  EXPECT_FLOAT_EQ(base_positions[0].z, 2.0f);
}

TEST_F(SculptLayerTest, MergeActiveLayerDown)
{
  mesh = create_test_mesh();

  add_layer(*mesh, "BottomLayer");
  layer_displacements_for_write(*mesh, "BottomLayer")[0] = float3(1.0f, 0.0f, 0.0f);

  add_layer(*mesh, "TopLayer");
  layer_displacements_for_write(*mesh, "TopLayer")[0] = float3(0.0f, 2.0f, 0.0f);
  mesh->sculpt_layers[1].opacity = 0.5f;

  ASSERT_EQ(mesh->active_layer_index, 1);
  merge_active_layer_down(*mesh);

  EXPECT_EQ(mesh->sculpt_layers_len, 1);
  EXPECT_EQ(mesh->active_layer_index, 0);

  Span<float3> merged_disp = layer_displacements(*mesh, "BottomLayer");
  ASSERT_FALSE(merged_disp.is_empty());
  EXPECT_FLOAT_EQ(merged_disp[0].x, 1.0f);
  EXPECT_FLOAT_EQ(merged_disp[0].y, 1.0f);
  EXPECT_FLOAT_EQ(merged_disp[0].z, 0.0f);
}

TEST_F(SculptLayerTest, CombineDisplacementsKernel)
{
  Array<float3> src = {float3(2.0f, 0.0f, 0.0f), float3(0.0f, 4.0f, 0.0f)};
  Array<float3> dst = {float3(1.0f, 1.0f, 1.0f), float3(2.0f, 2.0f, 2.0f)};

  combine_displacements(0.25f, 0.5f, src.as_span(), dst.as_mutable_span());

  EXPECT_FLOAT_EQ(dst[0].x, 1.0f);
  EXPECT_FLOAT_EQ(dst[0].y, 0.5f);
  EXPECT_FLOAT_EQ(dst[0].z, 0.5f);

  EXPECT_FLOAT_EQ(dst[1].x, 1.0f);
  EXPECT_FLOAT_EQ(dst[1].y, 2.0f);
  EXPECT_FLOAT_EQ(dst[1].z, 1.0f);
}

TEST_F(SculptLayerTest, MultiVertexDisplacementsAndEvaluation)
{
  mesh = create_test_mesh();

  add_layer(*mesh, "Layer 1");
  MutableSpan<float3> disp1 = layer_displacements_for_write(*mesh, "Layer 1");
  disp1[0] = float3(1.0f, 2.0f, 3.0f);
  disp1[1] = float3(4.0f, 5.0f, 6.0f);
  disp1[2] = float3(7.0f, 8.0f, 9.0f);
  disp1[3] = float3(10.0f, 11.0f, 12.0f);

  add_layer(*mesh, "Layer 2");
  MutableSpan<float3> disp2 = layer_displacements_for_write(*mesh, "Layer 2");
  disp2[0] = float3(2.0f, 0.0f, -2.0f);
  disp2[1] = float3(0.0f, 4.0f, -4.0f);
  disp2[2] = float3(-2.0f, 0.0f, 2.0f);
  disp2[3] = float3(0.0f, -4.0f, 4.0f);
  mesh->sculpt_layers[1].opacity = 0.5f;

  Array<float3> eval_positions(mesh->verts_num);
  calc_eval_positions(*mesh, eval_positions.as_mutable_span());

  const Span<float3> base_positions = mesh->vert_positions();
  for (int i = 0; i < mesh->verts_num; i++) {
    const float3 expected = base_positions[i] + disp1[i] * 1.0f + disp2[i] * 0.5f;
    EXPECT_FLOAT_EQ(eval_positions[i].x, expected.x);
    EXPECT_FLOAT_EQ(eval_positions[i].y, expected.y);
    EXPECT_FLOAT_EQ(eval_positions[i].z, expected.z);
  }

  mesh->sculpt_layers[0].is_visible = false;
  calc_eval_positions(*mesh, eval_positions.as_mutable_span());

  for (int i = 0; i < mesh->verts_num; i++) {
    const float3 expected = base_positions[i] + disp2[i] * 0.5f;
    EXPECT_FLOAT_EQ(eval_positions[i].x, expected.x);
    EXPECT_FLOAT_EQ(eval_positions[i].y, expected.y);
    EXPECT_FLOAT_EQ(eval_positions[i].z, expected.z);
  }
}

TEST_F(SculptLayerTest, RemoveLayerOrderAndShifting)
{
  mesh = create_test_mesh();

  add_layer(*mesh, "Bottom");
  layer_displacements_for_write(*mesh, "Bottom")[0] = float3(1.0f, 0.0f, 0.0f);

  add_layer(*mesh, "Middle");
  layer_displacements_for_write(*mesh, "Middle")[0] = float3(0.0f, 2.0f, 0.0f);

  add_layer(*mesh, "Top");
  layer_displacements_for_write(*mesh, "Top")[0] = float3(0.0f, 0.0f, 3.0f);

  EXPECT_EQ(mesh->sculpt_layers_len, 3);
  EXPECT_EQ(mesh->active_layer_index, 2);

  mesh->active_layer_index = 0;
  remove_active_layer(*mesh);

  EXPECT_EQ(mesh->sculpt_layers_len, 2);
  EXPECT_EQ(mesh->active_layer_index, 0);
  EXPECT_STREQ(mesh->sculpt_layers[0].name, "Middle");
  EXPECT_STREQ(mesh->sculpt_layers[1].name, "Top");

  Span<float3> middle_disp = layer_displacements(*mesh, "Middle");
  ASSERT_EQ(middle_disp.size(), mesh->verts_num);
  EXPECT_EQ(middle_disp[0], float3(0.0f, 2.0f, 0.0f));

  Span<float3> top_disp = layer_displacements(*mesh, "Top");
  ASSERT_EQ(top_disp.size(), mesh->verts_num);
  EXPECT_EQ(top_disp[0], float3(0.0f, 0.0f, 3.0f));

  EXPECT_TRUE(layer_displacements(*mesh, "Bottom").is_empty());
}

TEST_F(SculptLayerTest, MergeActiveLayerDownBottomLayer)
{
  mesh = create_test_mesh();

  mesh->vert_positions_for_write()[0] = float3(1.0f, 1.0f, 1.0f);

  add_layer(*mesh, "BaseOnlyLayer");
  layer_displacements_for_write(*mesh, "BaseOnlyLayer")[0] = float3(0.0f, 2.0f, 4.0f);
  mesh->sculpt_layers[0].opacity = 0.5f;

  ASSERT_EQ(mesh->active_layer_index, 0);

  merge_active_layer_down(*mesh);

  EXPECT_EQ(mesh->sculpt_layers_len, 0);
  EXPECT_EQ(mesh->active_layer_index, -1);

  const Span<float3> base_pos = mesh->vert_positions();
  EXPECT_FLOAT_EQ(base_pos[0].x, 1.0f);
  EXPECT_FLOAT_EQ(base_pos[0].y, 2.0f);
  EXPECT_FLOAT_EQ(base_pos[0].z, 3.0f);
}

TEST_F(SculptLayerTest, RenameLayerToExistingNameCollides)
{
  mesh = create_test_mesh();

  add_layer(*mesh, "Detail");
  add_layer(*mesh, "Detail2");

  SculptLayer &layer2 = mesh->sculpt_layers[1];
  rename_layer(*mesh, layer2, "Detail");

  EXPECT_STREQ(layer2.name, "Detail.001");
}

}  // namespace blender::bke::sculpt_layer::tests
