/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "testing/testing.h"

#include "BKE_gtest_base.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.hh"
#include "BKE_multiresolution.hh"
#include "BKE_object.hh"
#include "BKE_sculpt_layer.hh"

#include "DNA_mesh_types.h"
#include "DNA_multires_types.h"
#include "DNA_object_types.h"

#include "GEO_mesh_primitive_cuboid.hh"
#include "GEO_mesh_primitive_grid.hh"

#include "BLI_array.hh"

#include "sculpt_multires.hh"

namespace blender::ed::sculpt_paint::multires::tests {

class SculptMultiresTest : public bke::BlenderGTestBase {
 public:
  Object *ob = nullptr;
  Mesh *mesh = nullptr;

  void SetUp() override
  {
    ob = BKE_object_add_only_object(nullptr, OB_MESH, "MultiresTestObject");
  }

  void TearDown() override
  {
    if (ob != nullptr) {
      ob->data = nullptr;
      BKE_id_free(nullptr, ob);
      ob = nullptr;
    }
    if (mesh != nullptr) {
      if (mesh->multires != nullptr) {
        mesh->multires->free();
        MEM_delete(mesh->multires);
        mesh->multires = nullptr;
      }
      BKE_id_free(nullptr, mesh);
      mesh = nullptr;
    }
  }

  void setup_mesh(Mesh *new_mesh)
  {
    mesh = new_mesh;
    ob->data = id_cast<ID *>(mesh);
  }
};

TEST_F(SculptMultiresTest, SubdivideCubeLevels)
{
  /* Standard 8-vertex unit cube: 8 verts, 12 edges, 6 faces. */
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));
  ASSERT_NE(mesh, nullptr);
  EXPECT_EQ(mesh->multires, nullptr);

  /* Level 0 -> 1 */
  subdivide(*ob);

  ASSERT_NE(mesh->multires, nullptr);
  Multiresolution &multires = *mesh->multires;
  EXPECT_EQ(multires.top_level, 1);
  EXPECT_EQ(multires.sculpt_level, 1);

  /* At Level 1 (grid_size = 1):
   * Vertices = 8 (base) + 12 (edges * 1) + 6 (faces * 1) = 26 vertices. */
  EXPECT_EQ(multires.top_base_positions().size(), 26);

  /* Level 1 -> 2 */
  subdivide(*ob);
  EXPECT_EQ(multires.top_level, 2);
  EXPECT_EQ(multires.sculpt_level, 2);

  /* At Level 2 (grid_size = 3):
   * Vertices = 8 (base) + 12 * 3 (edges) + 6 * 9 (faces) = 8 + 36 + 54 = 98 vertices. */
  EXPECT_EQ(multires.top_base_positions().size(), 98);

  /* Level 2 -> 3 */
  subdivide(*ob);
  EXPECT_EQ(multires.top_level, 3);
  EXPECT_EQ(multires.sculpt_level, 3);

  /* At Level 3 (grid_size = 7):
   * Vertices = 8 (base) + 12 * 7 (edges) + 6 * 49 (faces) = 8 + 84 + 294 = 386 vertices. */
  EXPECT_EQ(multires.top_base_positions().size(), 386);
}

TEST_F(SculptMultiresTest, SwitchLevelRoundtrip)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));

  /* Subdivide to level 3. */
  subdivide(*ob);
  subdivide(*ob);
  subdivide(*ob);

  Multiresolution &multires = *mesh->multires;
  EXPECT_EQ(multires.top_level, 3);
  const int level3_verts = multires.top_base_positions().size();
  EXPECT_EQ(level3_verts, 386);

  /* Switch down to level 1. */
  switch_level(*ob, 1);
  EXPECT_EQ(multires.curr_level, 1);
  EXPECT_EQ(multires.sculpt_level, 1);

  /* Top level storage must be preserved while at level 1. */
  EXPECT_EQ(multires.top_base_positions().size(), 386);

  /* Switch down to base mesh (level 0). */
  switch_level(*ob, 0);
  EXPECT_EQ(multires.curr_level, 0);
  EXPECT_EQ(multires.sculpt_level, 0);
  EXPECT_EQ(mesh->verts_num, 8);

  /* Switch directly back up to top level (level 3). */
  switch_level(*ob, 3);
  EXPECT_EQ(multires.curr_level, 3);
  EXPECT_EQ(multires.sculpt_level, 3);
  EXPECT_EQ(multires.top_base_positions().size(), 386);
}

TEST_F(SculptMultiresTest, SubdivideWithLayers)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));

  /* Add a sculpt layer prior to subdivision. */
  bke::sculpt_layer::add_layer(*mesh, "FineDetails");
  EXPECT_EQ(mesh->sculpt_layers_len, 1);

  /* Subdivide to level 1. */
  subdivide(*ob);

  Multiresolution &multires = *mesh->multires;
  EXPECT_EQ(multires.top_level, 1);

  /* Layer displacements should automatically be propagated to top storage. */
  Span<float3> layer_top_disp = bke::sculpt_layer::layer_top_displacements(*mesh, "FineDetails");
  EXPECT_EQ(layer_top_disp.size(), 26);

  /* Subdivide to level 2. */
  subdivide(*ob);
  EXPECT_EQ(multires.top_level, 2);

  layer_top_disp = bke::sculpt_layer::layer_top_displacements(*mesh, "FineDetails");
  EXPECT_EQ(layer_top_disp.size(), 98);
}

TEST_F(SculptMultiresTest, LinearSubdividePlanar)
{
  /* 2x2 vertices = 1 quad face flat on XY plane. */
  setup_mesh(geometry::create_grid_mesh(2, 2, 2.0f, 2.0f, {}));

  mesh->multires_use_linear = 1;
  subdivide(*ob);

  Multiresolution &multires = *mesh->multires;
  EXPECT_EQ(multires.top_level, 1);

  const Span<float3> top_positions = multires.top_base_positions();
  EXPECT_EQ(top_positions.size(), 9);

  /* With linear subdivision on XY plane, all coordinates must stay planar (z == 0). */
  for (const float3 &pos : top_positions) {
    EXPECT_NEAR(pos.z, 0.0f, 1e-6f);
  }
}

TEST_F(SculptMultiresTest, FlushUpdatesBase)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));

  subdivide(*ob);
  subdivide(*ob);

  Multiresolution &multires = *mesh->multires;
  EXPECT_EQ(multires.top_level, 2);

  /* Call flush to ensure base positions downsample and state is synchronized. */
  flush(*ob);

  EXPECT_EQ(mesh->verts_num, 8);
  for (const float3 &pos : mesh->vert_positions()) {
    /* Since the cube is centered at origin with size 2, coordinates are roughly within [-1, 1]. */
    EXPECT_GE(pos.x, -1.01f);
    EXPECT_LE(pos.x, 1.01f);
  }
}

TEST_F(SculptMultiresTest, FlushUpdatesBaseWithDisplacement)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));

  subdivide(*ob);
  subdivide(*ob);

  Multiresolution &multires = *mesh->multires;
  EXPECT_EQ(multires.top_level, 2);

  const bke::multiresolution::BaseTopology top_base = multires.get_base_topology(
      *mesh, multires.top_level);
  const bke::multiresolution::Offsets top_offsets = Multiresolution::build_table_offsets(
      multires, top_base, multires.top_level);

  /* Displace base vertex 0 at the top level. */
  const int top_v0_idx = top_offsets.verts[0];
  const float3 displacement(5.0f, -3.0f, 2.0f);
  multires.top_base_positions_for_write()[top_v0_idx] += displacement;
  const float3 expected_new_pos0 = multires.top_base_positions()[top_v0_idx];

  /* Flush top positions down to the base mesh. */
  flush(*ob);

  EXPECT_EQ(mesh->verts_num, 8);
  EXPECT_FLOAT_EQ(mesh->vert_positions()[0].x, expected_new_pos0.x);
  EXPECT_FLOAT_EQ(mesh->vert_positions()[0].y, expected_new_pos0.y);
  EXPECT_FLOAT_EQ(mesh->vert_positions()[0].z, expected_new_pos0.z);

  /* All other base vertices should match their corresponding top positions. */
  for (int i = 1; i < 8; i++) {
    const int top_idx = top_offsets.verts[i];
    EXPECT_FLOAT_EQ(mesh->vert_positions()[i].x, multires.top_base_positions()[top_idx].x);
    EXPECT_FLOAT_EQ(mesh->vert_positions()[i].y, multires.top_base_positions()[top_idx].y);
    EXPECT_FLOAT_EQ(mesh->vert_positions()[i].z, multires.top_base_positions()[top_idx].z);
  }
}

TEST_F(SculptMultiresTest, SwitchLevelPreservesEdits)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));

  subdivide(*ob);
  subdivide(*ob);

  Multiresolution &multires = *mesh->multires;
  EXPECT_EQ(multires.top_level, 2);

  /* Apply distinctive sculpted offsets across all 98 vertices. */
  MutableSpan<float3> top_pos = multires.top_base_positions_for_write();
  for (int i = 0; i < top_pos.size(); i++) {
    top_pos[i] += float3(float(i) * 0.1f, float(i) * -0.2f, float(i) * 0.05f);
  }

  Array<float3> snapshot(top_pos.size());
  snapshot.as_mutable_span().copy_from(top_pos);

  /* Switch down to level 1, then base mesh (level 0). */
  switch_level(*ob, 1);
  EXPECT_EQ(multires.curr_level, 1);

  switch_level(*ob, 0);
  EXPECT_EQ(multires.curr_level, 0);

  /* Switch back to top level (level 2). */
  switch_level(*ob, 2);
  EXPECT_EQ(multires.curr_level, 2);

  /* The sculpted edits on the top level must be fully preserved. */
  EXPECT_EQ_SPAN(multires.top_base_positions(), snapshot.as_span());
}

TEST_F(SculptMultiresTest, SubdivideMultipleLayersWithDisplacements)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));

  /* Add two sculpt layers and set distinct displacements across the base vertices. */
  bke::sculpt_layer::add_layer(*mesh, "LayerA");
  bke::sculpt_layer::add_layer(*mesh, "LayerB");

  MutableSpan<float3> dispA = bke::sculpt_layer::layer_displacements_for_write(*mesh, "LayerA");
  MutableSpan<float3> dispB = bke::sculpt_layer::layer_displacements_for_write(*mesh, "LayerB");

  for (int i = 0; i < 8; i++) {
    dispA[i] = float3(float(i + 1), 0.0f, 0.0f);
    dispB[i] = float3(0.0f, float(i + 1) * 2.0f, 0.0f);
  }

  /* Subdivide to level 1. */
  subdivide(*ob);

  Multiresolution &multires = *mesh->multires;
  EXPECT_EQ(multires.top_level, 1);

  const bke::multiresolution::BaseTopology top_base_1 = multires.get_base_topology(
      *mesh, multires.top_level);
  const bke::multiresolution::Offsets offsets_1 = Multiresolution::build_table_offsets(
      multires, top_base_1, multires.top_level);

  Span<float3> top_dispA = bke::sculpt_layer::layer_top_displacements(*mesh, "LayerA");
  Span<float3> top_dispB = bke::sculpt_layer::layer_top_displacements(*mesh, "LayerB");

  ASSERT_EQ(top_dispA.size(), 26);
  ASSERT_EQ(top_dispB.size(), 26);

  /* Base vertices in both top layers must match the original base displacements. */
  for (int i = 0; i < 8; i++) {
    const int vert_idx = offsets_1.verts[i];
    EXPECT_FLOAT_EQ(top_dispA[vert_idx].x, dispA[i].x);
    EXPECT_FLOAT_EQ(top_dispB[vert_idx].y, dispB[i].y);
  }

  /* Subdivide to level 2. */
  subdivide(*ob);
  EXPECT_EQ(multires.top_level, 2);

  const bke::multiresolution::BaseTopology top_base_2 = multires.get_base_topology(
      *mesh, multires.top_level);
  const bke::multiresolution::Offsets offsets_2 = Multiresolution::build_table_offsets(
      multires, top_base_2, multires.top_level);

  top_dispA = bke::sculpt_layer::layer_top_displacements(*mesh, "LayerA");
  top_dispB = bke::sculpt_layer::layer_top_displacements(*mesh, "LayerB");

  ASSERT_EQ(top_dispA.size(), 98);
  ASSERT_EQ(top_dispB.size(), 98);

  for (int i = 0; i < 8; i++) {
    const int vert_idx = offsets_2.verts[i];
    EXPECT_FLOAT_EQ(top_dispA[vert_idx].x, dispA[i].x);
    EXPECT_FLOAT_EQ(top_dispB[vert_idx].y, dispB[i].y);
  }
}

}  // namespace blender::ed::sculpt_paint::multires::tests
