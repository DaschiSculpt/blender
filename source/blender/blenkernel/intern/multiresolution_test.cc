/* SPDX-FileCopyrightText: 2026 Daschi
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "testing/testing.h"

#include "BKE_gtest_base.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.h"
#include "BKE_multiresolution.hh"
#include "BKE_paint_bvh.hh"

#include "DNA_mesh_types.h"
#include "DNA_multires_types.h"

#include "BLI_array.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

namespace blender::bke::multiresolution::tests {

class MultiresolutionTest : public BlenderGTestBase {
 public:
  Mesh *mesh = nullptr;

  void TearDown() override
  {
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

  static Mesh *create_quad_plane_mesh()
  {
    Mesh *mesh = BKE_mesh_new_nomain(4, 4, 1, 4);

    MutableSpan<float3> positions = mesh->vert_positions_for_write();
    positions[0] = float3(-1.0f, -1.0f, 0.0f);
    positions[1] = float3(1.0f, -1.0f, 0.0f);
    positions[2] = float3(1.0f, 1.0f, 0.0f);
    positions[3] = float3(-1.0f, 1.0f, 0.0f);

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

  static Mesh *create_two_quad_plane_mesh()
  {
    Mesh *mesh = BKE_mesh_new_nomain(6, 7, 2, 8);

    MutableSpan<float3> positions = mesh->vert_positions_for_write();
    positions[0] = float3(0.0f, 0.0f, 0.0f);
    positions[1] = float3(1.0f, 0.0f, 0.0f);
    positions[2] = float3(2.0f, 0.0f, 0.0f);
    positions[3] = float3(0.0f, 1.0f, 0.0f);
    positions[4] = float3(1.0f, 1.0f, 0.0f);
    positions[5] = float3(2.0f, 1.0f, 0.0f);

    MutableSpan<int2> edges = mesh->edges_for_write();
    edges[0] = int2(0, 1);
    edges[1] = int2(1, 2);
    edges[2] = int2(3, 4);
    edges[3] = int2(4, 5);
    edges[4] = int2(0, 3);
    edges[5] = int2(1, 4);
    edges[6] = int2(2, 5);

    MutableSpan<int> face_offsets = mesh->face_offsets_for_write();
    face_offsets[0] = 0;
    face_offsets[1] = 4;
    face_offsets[2] = 8;

    MutableSpan<int> corner_verts = mesh->corner_verts_for_write();
    corner_verts[0] = 0;
    corner_verts[1] = 1;
    corner_verts[2] = 4;
    corner_verts[3] = 3;

    corner_verts[4] = 1;
    corner_verts[5] = 2;
    corner_verts[6] = 5;
    corner_verts[7] = 4;

    MutableSpan<int> corner_edges = mesh->corner_edges_for_write();
    corner_edges[0] = 0;
    corner_edges[1] = 5;
    corner_edges[2] = 2;
    corner_edges[3] = 4;

    corner_edges[4] = 1;
    corner_edges[5] = 6;
    corner_edges[6] = 3;
    corner_edges[7] = 5;

    mesh->tag_topology_changed();
    return mesh;
  }

  static Mesh *create_non_planar_quad_mesh()
  {
    Mesh *mesh = BKE_mesh_new_nomain(4, 4, 1, 4);

    MutableSpan<float3> positions = mesh->vert_positions_for_write();
    positions[0] = float3(-1.0f, -1.0f, 0.0f);
    positions[1] = float3(1.0f, -1.0f, 2.0f);
    positions[2] = float3(1.0f, 1.0f, 6.0f);
    positions[3] = float3(-1.0f, 1.0f, 4.0f);

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

  static Mesh *create_triangle_mesh()
  {
    Mesh *mesh = BKE_mesh_new_nomain(3, 3, 1, 3);

    MutableSpan<float3> positions = mesh->vert_positions_for_write();
    positions[0] = float3(0.0f, 0.0f, 0.0f);
    positions[1] = float3(1.0f, 0.0f, 0.0f);
    positions[2] = float3(0.0f, 1.0f, 0.0f);

    MutableSpan<int2> edges = mesh->edges_for_write();
    edges[0] = int2(0, 1);
    edges[1] = int2(1, 2);
    edges[2] = int2(2, 0);

    MutableSpan<int> face_offsets = mesh->face_offsets_for_write();
    face_offsets[0] = 0;
    face_offsets[1] = 3;

    MutableSpan<int> corner_verts = mesh->corner_verts_for_write();
    corner_verts[0] = 0;
    corner_verts[1] = 1;
    corner_verts[2] = 2;

    MutableSpan<int> corner_edges = mesh->corner_edges_for_write();
    corner_edges[0] = 0;
    corner_edges[1] = 1;
    corner_edges[2] = 2;

    mesh->tag_topology_changed();
    return mesh;
  }

  static Offsets ensure_test_bvh_partition(Multiresolution &multires,
                                           const int lvl,
                                           const Mesh &base_mesh,
                                           const BaseTopology &base_topo)
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

    Offsets offsets;
    offsets.verts = std::move(pbvh.offsets.verts);
    offsets.edges = std::move(pbvh.offsets.edges);
    offsets.faces = std::move(pbvh.offsets.faces);
    return offsets;
  }
};

TEST_F(MultiresolutionTest, BaseLevelAllQuads)
{
  mesh = create_quad_plane_mesh();
  ASSERT_NE(mesh, nullptr);

  Multiresolution *multires = BKE_multiresolution_create(*mesh);
  mesh->multires = multires;
  ASSERT_NE(multires, nullptr);

  EXPECT_TRUE(multires->base_all_quads);

  EXPECT_EQ(multires->base_level(0), 0);
  EXPECT_EQ(multires->base_level(1), 0);
  EXPECT_EQ(multires->base_level(2), 0);
  EXPECT_EQ(multires->base_level(3), 0);
  EXPECT_EQ(multires->base_level(4), 0);
  EXPECT_EQ(multires->base_level(5), 0);
  EXPECT_EQ(multires->base_level(6), 1);
  EXPECT_EQ(multires->base_level(7), 2);
}

TEST_F(MultiresolutionTest, BaseLevelNonQuads)
{
  mesh = create_triangle_mesh();
  ASSERT_NE(mesh, nullptr);

  Multiresolution *multires = BKE_multiresolution_create(*mesh);
  mesh->multires = multires;
  ASSERT_NE(multires, nullptr);

  EXPECT_FALSE(multires->base_all_quads);

  EXPECT_EQ(multires->base_level(0), 0);
  EXPECT_EQ(multires->base_level(1), 0);
  EXPECT_EQ(multires->base_level(2), 1);
  EXPECT_EQ(multires->base_level(3), 1);
  EXPECT_EQ(multires->base_level(4), 1);
  EXPECT_EQ(multires->base_level(5), 1);
  EXPECT_EQ(multires->base_level(6), 1);
  EXPECT_EQ(multires->base_level(7), 2);
}

TEST_F(MultiresolutionTest, GridSizeAndFaceCounts)
{
  EXPECT_EQ(Multiresolution::grid_size(1), 1);
  EXPECT_EQ(Multiresolution::grid_size(2), 3);
  EXPECT_EQ(Multiresolution::grid_size(3), 7);
  EXPECT_EQ(Multiresolution::grid_size(4), 15);

  EXPECT_EQ(Multiresolution::faces_per_side(1), 2);
  EXPECT_EQ(Multiresolution::faces_per_side(2), 4);
  EXPECT_EQ(Multiresolution::faces_per_side(3), 8);

  EXPECT_EQ(Multiresolution::faces_per_corner(1), 1);
  EXPECT_EQ(Multiresolution::faces_per_corner(2), 4);
  EXPECT_EQ(Multiresolution::faces_per_corner(3), 16);
}

TEST_F(MultiresolutionTest, BVHPartitionAndTableOffsets)
{
  mesh = create_quad_plane_mesh();
  mesh->multires = BKE_multiresolution_create(*mesh);
  Multiresolution &multires = *mesh->multires;

  const BaseTopology base_topo(*mesh);

  const Offsets direct_offsets = ensure_test_bvh_partition(multires, 1, *mesh, base_topo);

  EXPECT_EQ(direct_offsets.verts.size(), mesh->verts_num);
  EXPECT_EQ(direct_offsets.edges.size(), mesh->edges_num);
  EXPECT_EQ(direct_offsets.faces.size(), mesh->faces_num);

  const Offsets table_offsets = Multiresolution::build_table_offsets(multires, base_topo, 1);

  EXPECT_EQ_SPAN(direct_offsets.verts.as_span(), table_offsets.verts.as_span());
  EXPECT_EQ_SPAN(direct_offsets.edges.as_span(), table_offsets.edges.as_span());
  EXPECT_EQ_SPAN(direct_offsets.faces.as_span(), table_offsets.faces.as_span());
}

TEST_F(MultiresolutionTest, SubdivideBaseToLinearPlanar)
{
  mesh = create_quad_plane_mesh();
  mesh->multires = BKE_multiresolution_create(*mesh);
  Multiresolution &multires = *mesh->multires;

  const BaseTopology base_topo(*mesh);
  const Offsets offsets = ensure_test_bvh_partition(multires, 1, *mesh, base_topo);

  const int total_subdiv_verts = 9;
  Array<float3> subdivided_pos(total_subdiv_verts);

  multires.subdivide_base_to(*mesh,
                             1,
                             offsets,
                             mesh->vert_positions(),
                             subdivided_pos.as_mutable_span(),
                             true /* linear */);

  for (const float3 &pos : subdivided_pos) {
    EXPECT_NEAR(pos.z, 0.0f, 1e-6f);
  }

  const int face_center_idx = offsets.faces[0];
  EXPECT_NEAR(subdivided_pos[face_center_idx].x, 0.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[face_center_idx].y, 0.0f, 1e-6f);

  for (int i = 0; i < 4; i++) {
    const int vert_idx = offsets.verts[i];
    EXPECT_NEAR(subdivided_pos[vert_idx].x, mesh->vert_positions()[i].x, 1e-6f);
    EXPECT_NEAR(subdivided_pos[vert_idx].y, mesh->vert_positions()[i].y, 1e-6f);
  }
}

TEST_F(MultiresolutionTest, SubdivideBaseToCatmullClarkPlanar)
{
  mesh = create_quad_plane_mesh();
  mesh->multires = BKE_multiresolution_create(*mesh);
  Multiresolution &multires = *mesh->multires;

  const BaseTopology base_topo(*mesh);
  const Offsets offsets = ensure_test_bvh_partition(multires, 1, *mesh, base_topo);

  const int total_subdiv_verts = 9;
  Array<float3> subdivided_pos(total_subdiv_verts);

  multires.subdivide_base_to(*mesh,
                             1,
                             offsets,
                             mesh->vert_positions(),
                             subdivided_pos.as_mutable_span(),
                             false /* smooth */);

  for (const float3 &pos : subdivided_pos) {
    EXPECT_NEAR(pos.z, 0.0f, 1e-6f);
  }

  const int face_center_idx = offsets.faces[0];
  EXPECT_NEAR(subdivided_pos[face_center_idx].x, 0.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[face_center_idx].y, 0.0f, 1e-6f);
}

TEST_F(MultiresolutionTest, DownsampleToBaseRoundtrip)
{
  mesh = create_quad_plane_mesh();
  mesh->multires = BKE_multiresolution_create(*mesh);
  Multiresolution &multires = *mesh->multires;

  const BaseTopology base_topo(*mesh);
  const Offsets offsets = ensure_test_bvh_partition(multires, 1, *mesh, base_topo);

  Array<float3> subdivided_pos(9);
  multires.subdivide_base_to(*mesh,
                             1,
                             offsets,
                             mesh->vert_positions(),
                             subdivided_pos.as_mutable_span(),
                             true /* linear */);

  Array<float3> restored_base_pos(mesh->verts_num);
  Multiresolution::downsample_to_base(
      offsets.verts, subdivided_pos.as_span(), restored_base_pos.as_mutable_span());

  EXPECT_EQ_SPAN(mesh->vert_positions(), restored_base_pos.as_span());
}

TEST_F(MultiresolutionTest, SubdivideFaceSets)
{
  mesh = create_quad_plane_mesh();
  mesh->multires = BKE_multiresolution_create(*mesh);
  Multiresolution &multires = *mesh->multires;

  Array<int> from_face_sets = {42};

  Array<int> to_face_sets(4, 0);

  multires.subdivide_face_sets(
      *mesh, 0, from_face_sets.as_span(), 1, to_face_sets.as_mutable_span());

  for (const int val : to_face_sets) {
    EXPECT_EQ(val, 42);
  }
}

TEST_F(MultiresolutionTest, SubdivideHiddenFaces)
{
  mesh = create_quad_plane_mesh();
  mesh->multires = BKE_multiresolution_create(*mesh);
  Multiresolution &multires = *mesh->multires;

  Array<bool> from_hidden_faces = {true};

  Array<bool> to_hidden_faces(4, false);

  multires.subdivide_hidden_faces(
      *mesh, 0, from_hidden_faces.as_span(), 1, to_hidden_faces.as_mutable_span());

  for (const bool val : to_hidden_faces) {
    EXPECT_TRUE(val);
  }
}

TEST_F(MultiresolutionTest, SubdivideBaseToLinearNonPlanar3D)
{
  mesh = create_non_planar_quad_mesh();
  mesh->multires = BKE_multiresolution_create(*mesh);
  Multiresolution &multires = *mesh->multires;

  const BaseTopology base_topo(*mesh);
  const Offsets offsets = ensure_test_bvh_partition(multires, 1, *mesh, base_topo);

  const int total_subdiv_verts = 9;
  Array<float3> subdivided_pos(total_subdiv_verts);

  multires.subdivide_base_to(*mesh,
                             1,
                             offsets,
                             mesh->vert_positions(),
                             subdivided_pos.as_mutable_span(),
                             true /* linear */);

  for (int i = 0; i < 4; i++) {
    const int vert_idx = offsets.verts[i];
    EXPECT_FLOAT_EQ(subdivided_pos[vert_idx].x, mesh->vert_positions()[i].x);
    EXPECT_FLOAT_EQ(subdivided_pos[vert_idx].y, mesh->vert_positions()[i].y);
    EXPECT_FLOAT_EQ(subdivided_pos[vert_idx].z, mesh->vert_positions()[i].z);
  }

  const int e0_idx = offsets.edges[0];
  EXPECT_NEAR(subdivided_pos[e0_idx].x, 0.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[e0_idx].y, -1.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[e0_idx].z, 1.0f, 1e-6f);

  const int e1_idx = offsets.edges[1];
  EXPECT_NEAR(subdivided_pos[e1_idx].x, 1.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[e1_idx].y, 0.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[e1_idx].z, 4.0f, 1e-6f);

  const int e2_idx = offsets.edges[2];
  EXPECT_NEAR(subdivided_pos[e2_idx].x, 0.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[e2_idx].y, 1.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[e2_idx].z, 5.0f, 1e-6f);

  const int e3_idx = offsets.edges[3];
  EXPECT_NEAR(subdivided_pos[e3_idx].x, -1.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[e3_idx].y, 0.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[e3_idx].z, 2.0f, 1e-6f);

  const int f0_idx = offsets.faces[0];
  EXPECT_NEAR(subdivided_pos[f0_idx].x, 0.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[f0_idx].y, 0.0f, 1e-6f);
  EXPECT_NEAR(subdivided_pos[f0_idx].z, 3.0f, 1e-6f);
}

TEST_F(MultiresolutionTest, SubdivideMultiFaceSetsAndHiddenRoundtrip)
{
  mesh = create_two_quad_plane_mesh();
  mesh->multires = BKE_multiresolution_create(*mesh);
  Multiresolution &multires = *mesh->multires;

  Array<int> from_face_sets = {101, 202};
  Array<bool> from_hidden_faces = {false, true};

  Array<int> face_sets_lvl1(8, 0);
  Array<bool> hidden_faces_lvl1(8, false);

  multires.subdivide_face_sets(
      *mesh, 0, from_face_sets.as_span(), 1, face_sets_lvl1.as_mutable_span());
  multires.subdivide_hidden_faces(
      *mesh, 0, from_hidden_faces.as_span(), 1, hidden_faces_lvl1.as_mutable_span());

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(face_sets_lvl1[i], 101);
    EXPECT_FALSE(hidden_faces_lvl1[i]);
  }
  for (int i = 4; i < 8; i++) {
    EXPECT_EQ(face_sets_lvl1[i], 202);
    EXPECT_TRUE(hidden_faces_lvl1[i]);
  }

  Array<int> face_sets_lvl2(32, 0);
  Array<bool> hidden_faces_lvl2(32, false);

  multires.subdivide_face_sets(
      *mesh, 1, face_sets_lvl1.as_span(), 2, face_sets_lvl2.as_mutable_span());
  multires.subdivide_hidden_faces(
      *mesh, 1, hidden_faces_lvl1.as_span(), 2, hidden_faces_lvl2.as_mutable_span());

  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(face_sets_lvl2[i], 101);
    EXPECT_FALSE(hidden_faces_lvl2[i]);
  }
  for (int i = 16; i < 32; i++) {
    EXPECT_EQ(face_sets_lvl2[i], 202);
    EXPECT_TRUE(hidden_faces_lvl2[i]);
  }

  Array<int> downsampled_face_sets_lvl1(8, 0);
  Array<bool> downsampled_hidden_faces_lvl1(8, false);

  multires.downsample_face_sets(
      *mesh, 2, face_sets_lvl2.as_span(), 1, downsampled_face_sets_lvl1.as_mutable_span());
  multires.downsample_hidden_faces(
      *mesh, 2, hidden_faces_lvl2.as_span(), 1, downsampled_hidden_faces_lvl1.as_mutable_span());

  EXPECT_EQ_SPAN(downsampled_face_sets_lvl1.as_span(), face_sets_lvl1.as_span());
  EXPECT_EQ_SPAN(downsampled_hidden_faces_lvl1.as_span(), hidden_faces_lvl1.as_span());

  Array<int> granular_face_sets_lvl1(8, 0);
  Array<bool> granular_hidden_faces_lvl1(8, false);
  const int dirty_quad = 1;
  multires.downsample_face_sets(2,
                                face_sets_lvl2.as_span(),
                                1,
                                granular_face_sets_lvl1.as_mutable_span(),
                                Span<int>(&dirty_quad, 1));
  multires.downsample_hidden_faces(2,
                                   hidden_faces_lvl2.as_span(),
                                   1,
                                   granular_hidden_faces_lvl1.as_mutable_span(),
                                   Span<int>(&dirty_quad, 1));

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(granular_face_sets_lvl1[i], 0);
    EXPECT_FALSE(granular_hidden_faces_lvl1[i]);
  }
  for (int i = 4; i < 8; i++) {
    EXPECT_EQ(granular_face_sets_lvl1[i], face_sets_lvl1[i]);
    EXPECT_EQ(granular_hidden_faces_lvl1[i], hidden_faces_lvl1[i]);
  }

  Array<int> downsampled_face_sets_lvl0(2, 0);
  Array<bool> downsampled_hidden_faces_lvl0(2, false);

  multires.downsample_face_sets(
      *mesh, 1, face_sets_lvl1.as_span(), 0, downsampled_face_sets_lvl0.as_mutable_span());
  multires.downsample_hidden_faces(
      *mesh, 1, hidden_faces_lvl1.as_span(), 0, downsampled_hidden_faces_lvl0.as_mutable_span());

  EXPECT_EQ(downsampled_face_sets_lvl0[0], 101);
  EXPECT_EQ(downsampled_face_sets_lvl0[1], 202);
  EXPECT_FALSE(downsampled_hidden_faces_lvl0[0]);
  EXPECT_TRUE(downsampled_hidden_faces_lvl0[1]);
}

TEST_F(MultiresolutionTest, SubdivideNonQuadLevel1AndLevel2)
{
  mesh = create_triangle_mesh();
  mesh->multires = BKE_multiresolution_create(*mesh);
  Multiresolution &multires = *mesh->multires;

  EXPECT_FALSE(multires.base_all_quads);

  const BaseTopology base_topo_1 = multires.get_base_topology(*mesh, 1);
  const Offsets offsets_1 = ensure_test_bvh_partition(multires, 1, *mesh, base_topo_1);

  Array<float3> subdivided_pos_1(7);
  multires.subdivide_base_to(*mesh,
                             1,
                             offsets_1,
                             mesh->vert_positions(),
                             subdivided_pos_1.as_mutable_span(),
                             true /* linear */);

  for (const float3 &pos : subdivided_pos_1) {
    EXPECT_NEAR(pos.z, 0.0f, 1e-6f);
  }

  const BaseTopology base_topo_2 = multires.get_base_topology(*mesh, 2);
  const Offsets offsets_2 = ensure_test_bvh_partition(multires, 2, *mesh, base_topo_2);

  Array<float3> subdivided_pos_2(19);
  multires.subdivide_base_to(*mesh,
                             2,
                             offsets_2,
                             mesh->vert_positions(),
                             subdivided_pos_2.as_mutable_span(),
                             true /* linear */);

  for (const float3 &pos : subdivided_pos_2) {
    EXPECT_NEAR(pos.z, 0.0f, 1e-6f);
  }

  Array<float3> downsampled_pos_1(7);
  multires.downsample_to(*mesh,
                         2,
                         offsets_2,
                         subdivided_pos_2.as_span(),
                         1,
                         offsets_1,
                         downsampled_pos_1.as_mutable_span());

  EXPECT_EQ_SPAN(downsampled_pos_1.as_span(), subdivided_pos_1.as_span());
}

}  // namespace blender::bke::multiresolution::tests
