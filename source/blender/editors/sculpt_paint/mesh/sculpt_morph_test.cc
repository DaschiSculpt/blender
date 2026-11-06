/* SPDX-FileCopyrightText: 2026 Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "testing/testing.h"

#include "BKE_gtest_base.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.hh"
#include "BKE_multiresolution.hh"
#include "BKE_object.hh"
#include "BKE_object_types.hh"
#include "BKE_paint.hh"
#include "BKE_paint_bvh.hh"

#include "DNA_mesh_types.h"
#include "DNA_multires_types.h"
#include "DNA_object_types.h"

#include "GEO_mesh_primitive_cuboid.hh"

#include "sculpt_intern.hh"
#include "sculpt_morph.hh"
#include "sculpt_multires.hh"

namespace blender::ed::sculpt_paint::morph::tests {

class SculptMorphTest : public bke::BlenderGTestBase {
 public:
  Object *ob = nullptr;
  Mesh *mesh = nullptr;

  void SetUp() override
  {
    ob = BKE_object_add_only_object(nullptr, OB_MESH, "MorphTestObject");
  }

  void TearDown() override
  {
    if (ob != nullptr) {
      if (ob->runtime->sculpt_session != nullptr) {
        BKE_sculptsession_free(ob);
      }
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

  void subdivide_in_sculpt_mode()
  {
    ob->mode = OB_MODE_SCULPT;
    if (ob->runtime->sculpt_session == nullptr) {
      ob->runtime->sculpt_session = MEM_new<SculptSession>(__func__);
    }
    multires::subdivide(*ob);
  }

  void setup_multires_pbvh()
  {
    if (ob->runtime->sculpt_session == nullptr) {
      ob->runtime->sculpt_session = MEM_new<SculptSession>(__func__);
    }
    SculptSession &ss = *ob->runtime->sculpt_session;
    const bke::multiresolution::BaseTopology base_topo(*mesh);
    Array<float3> positions_storage;
    const Span<float3> positions = Multiresolution::calc_smooth_positions(
        *mesh, 0, positions_storage);
    Array<int> material_storage;
    const Span<int> material_indices = Multiresolution::calc_base_materials(
        *mesh, 0, material_storage);
    ss.pbvh = std::make_unique<bke::pbvh::Tree>(
        bke::pbvh::Tree::from_multires(1, base_topo, positions, material_indices));
  }
};

TEST_F(SculptMorphTest, StoreAndCheckExists)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));

  subdivide_in_sculpt_mode();
  setup_multires_pbvh();

  EXPECT_FALSE(morph_exists(*ob));

  store_apply(*ob, *(Depsgraph *)nullptr, MorphStoreMode::CurrentPositions);

  EXPECT_TRUE(morph_exists(*ob));

  Multiresolution &multires = *mesh->multires;
  Span<float3> morph_pos = multires.attribute<float3>(position_attr_name);
  EXPECT_EQ(morph_pos.size(), 26);
  EXPECT_EQ_SPAN(morph_pos, multires.positions());
}

TEST_F(SculptMorphTest, SwapWithoutMask)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));
  subdivide_in_sculpt_mode();
  setup_multires_pbvh();

  Multiresolution &multires = *mesh->multires;

  store_apply(*ob, *(Depsgraph *)nullptr, MorphStoreMode::CurrentPositions);

  const float3 orig_vert0 = multires.positions()[0];

  multires.positions_for_write()[0] += float3(1.0f, 2.0f, 3.0f);
  const float3 sculpt_vert0 = multires.positions()[0];
  EXPECT_NE(sculpt_vert0, orig_vert0);

  swap_apply(*ob, *(Depsgraph *)nullptr, false, false);

  EXPECT_FLOAT_EQ(multires.positions()[0].x, orig_vert0.x);
  EXPECT_FLOAT_EQ(multires.positions()[0].y, orig_vert0.y);
  EXPECT_FLOAT_EQ(multires.positions()[0].z, orig_vert0.z);

  Span<float3> morph_pos = multires.attribute<float3>(position_attr_name);
  EXPECT_FLOAT_EQ(morph_pos[0].x, sculpt_vert0.x);
  EXPECT_FLOAT_EQ(morph_pos[0].y, sculpt_vert0.y);
  EXPECT_FLOAT_EQ(morph_pos[0].z, sculpt_vert0.z);

  swap_apply(*ob, *(Depsgraph *)nullptr, false, false);

  EXPECT_FLOAT_EQ(multires.positions()[0].x, sculpt_vert0.x);
  EXPECT_FLOAT_EQ(multires.positions()[0].y, sculpt_vert0.y);
  EXPECT_FLOAT_EQ(multires.positions()[0].z, sculpt_vert0.z);
}

TEST_F(SculptMorphTest, SwapWithMask)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));
  subdivide_in_sculpt_mode();
  setup_multires_pbvh();

  Multiresolution &multires = *mesh->multires;

  store_apply(*ob, *(Depsgraph *)nullptr, MorphStoreMode::CurrentPositions);

  const float3 orig_v0 = multires.positions()[0];
  const float3 orig_v1 = multires.positions()[1];

  multires.positions_for_write()[0] += float3(0.0f, 0.0f, 5.0f);
  multires.positions_for_write()[1] += float3(0.0f, 0.0f, 5.0f);

  const float3 displaced_v0 = multires.positions()[0];

  MutableSpan<float> masks = multires.ensure_masks();
  masks.fill(0.0f);
  masks[0] = 1.0f;

  swap_apply(*ob, *(Depsgraph *)nullptr, true /* use_mask */, false /* use_hide */);

  EXPECT_FLOAT_EQ(multires.positions()[0].z, displaced_v0.z);

  EXPECT_FLOAT_EQ(multires.positions()[1].z, orig_v1.z);
}

TEST_F(SculptMorphTest, DeleteMorph)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));
  subdivide_in_sculpt_mode();
  setup_multires_pbvh();

  store_apply(*ob, *(Depsgraph *)nullptr, MorphStoreMode::CurrentPositions);
  EXPECT_TRUE(morph_exists(*ob));

  delete_apply(*ob);

  EXPECT_FALSE(morph_exists(*ob));

  Multiresolution &multires = *mesh->multires;
  EXPECT_TRUE(multires.attribute<float3>(position_attr_name).is_empty());
}

TEST_F(SculptMorphTest, SwapWithPartialMask)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));
  subdivide_in_sculpt_mode();
  setup_multires_pbvh();

  Multiresolution &multires = *mesh->multires;

  store_apply(*ob, *(Depsgraph *)nullptr, MorphStoreMode::CurrentPositions);

  const float3 orig_v0 = multires.positions()[0];

  multires.positions_for_write()[0] += float3(0.0f, 0.0f, 4.0f);
  const float3 displaced_v0 = multires.positions()[0];

  MutableSpan<float> masks = multires.ensure_masks();
  masks.fill(0.0f);
  masks[0] = 0.5f;

  swap_apply(*ob, *(Depsgraph *)nullptr, true /* use_mask */, false /* use_hide */);

  const float expected_z = orig_v0.z + 2.0f;
  EXPECT_FLOAT_EQ(multires.positions()[0].z, expected_z);

  Span<float3> morph_pos = multires.attribute<float3>(position_attr_name);
  EXPECT_FLOAT_EQ(morph_pos[0].z, expected_z);
}

TEST_F(SculptMorphTest, SwapWithHide)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));
  subdivide_in_sculpt_mode();
  setup_multires_pbvh();

  Multiresolution &multires = *mesh->multires;

  store_apply(*ob, *(Depsgraph *)nullptr, MorphStoreMode::CurrentPositions);

  const float3 orig_v0 = multires.positions()[0];
  const float3 orig_v1 = multires.positions()[1];

  multires.positions_for_write()[0] += float3(0.0f, 0.0f, 3.0f);
  multires.positions_for_write()[1] += float3(0.0f, 0.0f, 3.0f);

  const float3 displaced_v0 = multires.positions()[0];

  MutableSpan<bool> hidden = multires.ensure_hidden_verts();
  hidden.fill(false);
  hidden[0] = true;

  swap_apply(*ob, *(Depsgraph *)nullptr, false /* use_mask */, true /* use_hide */);

  EXPECT_FLOAT_EQ(multires.positions()[0].z, displaced_v0.z);

  EXPECT_FLOAT_EQ(multires.positions()[1].z, orig_v1.z);
}

TEST_F(SculptMorphTest, StoreSubdividedBase)
{
  setup_mesh(geometry::create_cuboid_mesh(float3(2.0f, 2.0f, 2.0f), 2, 2, 2));
  subdivide_in_sculpt_mode();
  setup_multires_pbvh();

  Multiresolution &multires = *mesh->multires;

  const bke::multiresolution::BaseTopology base_topo(*mesh);
  const bke::multiresolution::Offsets offsets = Multiresolution::build_table_offsets(
      multires, base_topo, 1);
  const int face_vert_idx = offsets.faces[0];

  const float3 orig_face_v = multires.positions()[face_vert_idx];

  multires.positions_for_write()[face_vert_idx] += float3(10.0f, 10.0f, 10.0f);
  EXPECT_NE(multires.positions()[face_vert_idx], orig_face_v);

  store_apply(*ob, *(Depsgraph *)nullptr, MorphStoreMode::SubdividedBase);

  EXPECT_TRUE(morph_exists(*ob));

  Span<float3> morph_pos = multires.attribute<float3>(position_attr_name);

  EXPECT_NE(morph_pos[face_vert_idx], multires.positions()[face_vert_idx]);

  Array<float3> expected_positions(multires.verts_num());
  Array<float3> base_positions(mesh->verts_num);
  Multiresolution::downsample_to_base<float3>(
      offsets.verts, multires.positions(), base_positions.as_mutable_span());
  const bool linear = mesh->multires_use_linear != 0;
  multires.subdivide_base_to<float3>(
      *mesh, 1, offsets, base_positions.as_span(), expected_positions.as_mutable_span(), linear);

  EXPECT_EQ_SPAN(morph_pos, expected_positions.as_span());
}

}  // namespace blender::ed::sculpt_paint::morph::tests
