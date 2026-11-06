/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup edsculpt
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "BLI_array.hh"
#include "BLI_bit_group_vector.hh"
#include "BLI_bit_vector.hh"
#include "BLI_implicit_sharing.hh"
#include "BLI_index_mask_fwd.hh"
#include "BLI_map.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_mutex.hh"
#include "BLI_span.hh"
#include "BLI_task_c.hh"
#include "BLI_utildefines.hh"
#include "BLI_vector.hh"

#include "DNA_customdata_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_attribute.hh"
#include "BKE_attribute_storage.hh"
#include "BKE_undo_system.hh"

namespace blender {

struct BMLogEntry;
struct Depsgraph;
struct Main;
struct Mesh;
struct Object;
struct Scene;
struct bContext;
struct wmOperator;
namespace bke::pbvh {
class Node;
}

namespace ed::sculpt_paint::undo {

enum class Type : int8_t {
  None,
  Position,
  HideVert,
  HideFace,
  Mask,
  DyntopoBegin,
  DyntopoEnd,
  Geometry,
  FaceSet,
  Color,
};

enum class UndoMeshType : uint8_t {
  Mesh,
  Multires,
  Grids,
};

struct Node {
  Array<float3, 0> position;
  Array<float3, 0> orig_position;
  Array<float3, 0> orig_eval_position;
  Array<float3, 0> normal;
  Array<float4, 0> col;
  Array<float, 0> mask;

  Array<float4, 0> loop_col;

  /* Mesh. */

  Array<int, 0> vert_indices;
  int unique_verts_num;

  /**
   * \todo Storing corners rather than faces is unnecessary.
   */
  Vector<int, 0> corner_indices;

  BitVector<0> vert_hidden;
  BitVector<0> face_hidden;

  /* Multires. */
  int index;

  /* Grids. */

  /** Indices of grids in the pbvh::Tree node. */
  Array<int, 0> grids;
  BitGroupVector<0> grid_hidden;

  /* Sculpt Face Sets */
  Array<int, 0> face_sets;

  Vector<int> face_indices;
};

struct SculptAttrRef {
  std::optional<bke::AttrDomain> domain;
  bke::AttrType type;
  std::string name;
  bool was_set;
};

/**
 * Full standalone snapshot of a mesh's base geometry + multires, used by operations that
 * replace topology wholesale (trim, symmetrize, apply-base) or replace/free `Mesh::multires`
 * (e.g. Delete Lower) rather than just diffing attribute values at a fixed topology. Unlike
 * `storage_undo::AttributeDiff`, this is not a delta: it owns an entire independent copy of
 * the mesh (including its own independent `Multiresolution`). It must be paired with a
 * `step_free` that releases `mesh` (see `free_snapshot`).
 */
struct FullMeshSnapshot {
  /* Is used for sanity check, helping with ensuring that two and only two
   * geometry pushes happened in the undo stack. */
  bool is_initialized = false;

  /* Owning. A standalone (no-main) Mesh duplicate, including its own independent
   * `Multiresolution`. Freed in `step_free`. */
  Mesh *mesh = nullptr;
};

/* Duplicate `source` with an independent deep copy of `multires` (see impl notes in
 * sculpt_undo.cc: `BKE_mesh_copy_for_eval` alone would only alias it). */
Mesh *full_mesh_duplicate_with_multires(const Mesh &source);

/* Restore `object`'s mesh (and its `multires`, which may be null) to an independent copy of
 * `snapshot.mesh`. `snapshot` itself is left untouched so it can be replayed across further
 * undo/redo cycles. */
void restore_full_mesh_snapshot(Object &object, const FullMeshSnapshot &snapshot);

/* Release the mesh (and its owned `multires`) held by `snapshot`. */
void free_snapshot(FullMeshSnapshot &snapshot);

/* Approximate retained memory of `snapshot` for undo-stack limiting. */
size_t snapshot_memory_size(const FullMeshSnapshot &snapshot);

struct PositionUndoStorage;

struct StepData {
 private:
  bool applied_ = true;

 public:
  /**
   * The type of data stored in this undo step. For historical reasons this is often set when the
   * first undo node is pushed.
   */
  Type type = Type::None;

  /** Name of the object associated with this undo data (`object.id.name`). */
  std::string object_name;

  /** Name of the object's active shape key when the undo step was created. */
  std::string active_shape_key_name;

  /* TODO: Combine the three structs into a variant, since they specify data that is only valid
   * within a single mode. */
  struct {
    /* The number of vertices in the entire mesh. */
    int verts_num;
    /* The number of face corners in the entire mesh. */
    int corners_num;
  } mesh;

  struct {
    int level;
  } multires;

  struct {
    /** The number of grids in the entire mesh. A copy of #SubdivCCG::grids_num. */
    int grids_num;
    /** A copy of #SubdivCCG::grid_size. */
    int grid_size;
  } grids;

  struct {
    /**
     * The current log entry for the given BMLog step. Represents the most recent step at the time
     * that this entry is added.
     *
     * There are two usages of this pointer:
     * - If undoing or redoing a enter / exit from Dyntopo, this entry is used to rebuild the
     *   BMLog from all of the relevant entries
     * - When an undo step is no longer valid, this is used to free the data that it holds and
     *   remove it from the underlying list.
     */
    BMLogEntry *bm_entry;

    /* Geometry at the bmesh enter moment. */
    FullMeshSnapshot geometry_enter;
  } bmesh;

  float3 pivot_pos;
  float4 pivot_rot;

  /* Geometry modification operations. */
  /* Original geometry is stored before the modification and is restored from when undoing. */
  FullMeshSnapshot geometry_original;
  /* Modified geometry is stored after the modification and is restored from when redoing. */
  FullMeshSnapshot geometry_modified;

  Mutex nodes_mutex;

  /**
   * #undo::Node is stored per #pbvh::Node to reduce data storage needed for changes only impacting
   * small portions of the mesh. During undo step creation and brush evaluation we often need to
   * look up the undo state for a specific node. That lookup must be protected by a lock since
   * nodes are pushed from multiple threads. This map speeds up undo node access to reduce the
   * amount of time we wait for the lock.
   *
   * This is only accessible when building the undo step, in between #push_begin and #push_end.
   */
  Map<const bke::pbvh::Node *, std::unique_ptr<Node>> undo_nodes_by_pbvh_node;

  /** Storage of per-node undo data after creation of the undo step is finished. */
  Vector<std::unique_ptr<Node>> nodes;
  std::unique_ptr<PositionUndoStorage> position_step_storage;
  size_t undo_size;

  /** Whether processing code needs to handle the current data as an undo step. */
  bool needs_undo() const
  {
    return applied_;
  }

  void tag_needs_undo()
  {
    applied_ = true;
  }

  void tag_needs_redo()
  {
    applied_ = false;
  }
};

struct PositionUndoStorage : NonMovable {
  Vector<std::unique_ptr<Node>> nodes_to_compress;
  UndoMeshType type;

  Array<Array<std::byte>> compressed_indices;

  /* As undo and redo happen, the data in these arrays is swapped (an undo step becomes a redo
   * step, and vice versa). */
  Array<Array<std::byte>> compressed_positions;

  Array<int> unique_verts_nums;

  /* Multires. */
  Array<int> node_indices;

  TaskPool *compression_task_pool;
  std::atomic<bool> compression_ready = false;
  std::atomic<bool> compression_started = false;
  StepData *owner_step_data = nullptr;

  explicit PositionUndoStorage(StepData &step_data, const UndoMeshType type);
  ~PositionUndoStorage();

  void ensure_compression_complete();

  static void compress_fn(TaskPool *pool, void *task_data);
};

}  // namespace ed::sculpt_paint::undo

namespace ed::sculpt_paint::storage_undo {

enum class StorageTarget : int8_t {
  Top,
  Curr,
  /* Base-mesh attribute storage. Only used as `AttributeDiff::origin` by the morph-delete
   * path (never as a push target): a base-mesh snapshot and a stale curr-storage snapshot
   * can share a name. */
  Base,
};

struct AttributeDiff {
  std::string name;
  bke::AttrType type;

  enum class Kind { Added, Deleted, Modified };
  Kind kind;

  Vector<int> blocks;
  Vector<std::byte> values;

  /* Origin storage the values were captured from. Only the morph-delete path sets this: a
   * base-mesh snapshot and a stale curr-storage snapshot can share a name (level switches
   * never clear morph attributes), so vector membership alone cannot route the restore.
   * Defaults to Curr; all other producers ignore it. */
  StorageTarget origin = StorageTarget::Curr;
};

struct ObjectState {
  int curr_level = 0;
  int sculpt_level = 0;
  int top_level = 0;
  int active_layer_index = -1;
};

struct LayerSnapshot {
  Vector<SculptLayer> layers;
  int active_layer_index = -1;
  bool layers_active = false;
};

enum class StorageUndoOpType {
  Subdivide,
  SwitchLevel,
  AddLayer,
  DeleteLayer,
  MoveLayer,
  SetActiveLayer,
  RenameLayer,
  DuplicateLayer,
  MergeLayer,
  MergeLayerToBase,
  MorphStore,
  MorphStoreFromSubdividedBase,
  MorphSwap,
  MorphDelete,
};

}  // namespace ed::sculpt_paint::storage_undo

namespace ed::sculpt_paint::undo {

/**
 * Discriminates which payload of the unified sculpt undo step is valid.
 *
 * Phase 1 of the undo unification (see plan.md): both brush-stroke steps and
 * multires/layer/morph storage steps share a single `SculptUndoStep` allocated
 * under `BKE_UNDOSYS_TYPE_SCULPT`. The payloads stay specialized (PBVH nodes
 * for strokes, block diffs for storage) and both are constructed for every
 * step; only the payload selected by `kind` is ever read or written.
 */
enum class SculptUndoStepKind : uint8_t {
  Stroke,
  Storage,
};

struct SculptUndoStep {
  UndoStep step;
  SculptUndoStepKind kind = SculptUndoStepKind::Stroke;

  /* Stroke payload (valid when `kind == Stroke`). */
  StepData data;

  /* Active color attribute at the start of this undo step. */
  SculptAttrRef active_color_start;

  /* Active color attribute at the end of this undo step. */
  SculptAttrRef active_color_end;

  /* Storage payload (valid when `kind == Storage`). */
  std::string object_name;
  storage_undo::StorageUndoOpType op_type = storage_undo::StorageUndoOpType::Subdivide;
  storage_undo::ObjectState state_before;
  storage_undo::ObjectState state_after;
  storage_undo::LayerSnapshot layers_before;
  storage_undo::LayerSnapshot layers_after;
  Vector<storage_undo::AttributeDiff> top_diffs;
  Vector<storage_undo::AttributeDiff> curr_diffs;
  Vector<storage_undo::AttributeDiff> face_diffs;
};

}  // namespace ed::sculpt_paint::undo

namespace ed::sculpt_paint::storage_undo {

/* Destroy the storage payload members of a unified step (defined in
 * sculpt_storage_undo.cc, where the member type names are in scope).
 * Called from the unified `step_free`. */
void free_storage_payload(undo::SculptUndoStep &us);

}  // namespace ed::sculpt_paint::storage_undo

namespace ed::sculpt_paint::undo {

/**
 * Store undo data of the given type for a pbvh::Tree node. This function can be called by multiple
 * threads concurrently, as long as they don't pass the same pbvh::Tree node.
 *
 * This is only possible when building an undo step, in between #push_begin and #push_end.
 */
void push_node(const Depsgraph &depsgraph,
               const Object &object,
               const bke::pbvh::Node *node,
               undo::Type type);
void push_nodes(const Depsgraph &depsgraph,
                Object &object,
                const IndexMask &node_mask,
                undo::Type type);

/**
 * Pushes an undo step using the operator name. This is necessary for
 * redo panels to work; operators that do not support that may use
 * #push_begin_ex instead if so desired.
 */
void push_begin(const Scene &scene, Object &ob, const wmOperator *op);

/**
 * Pushes an undo step when entering Sculpt mode.
 *
 * Similar to geometry_push, this undo type does not need the PBVH to be constructed.
 */
void push_enter_sculpt_mode(const Scene &scene, Object &ob, const wmOperator *op);

/**
 * NOTE: #push_begin is preferred since `name`
 * must match operator name for redo panels to work.
 */
void push_begin_ex(const Scene &scene, Object &ob, const char *name);
void push_end(Object &ob);
void push_end_ex(Object &ob, bool use_nested_undo);

/**
 * Phase 2 of the undo unification (see plan.md): single shared helper that ensures the active
 * object is in sculpt mode before decoding any sculpt undo step (stroke or storage). Replaces
 * the duplicated sculpt-mode entry blocks previously in `sculpt_undo.cc` and
 * `sculpt_storage_undo.cc`.
 *
 * \param ob_name: object name (`object.id.name`) of the step being decoded, used to reject
 * steps that do not belong to the active object. May be empty to skip the check.
 * \return the active mesh object in sculpt mode, or nullptr when there is none (or on name
 * mismatch).
 */
Object *ensure_sculpt_mode_for_undo(bContext *C, Main *bmain, const std::string &ob_name);

void restore_from_bmesh_enter_geometry(const StepData &step_data, Object &object);
bool has_bmesh_log_entry();

void restore_position_from_undo_step(const Depsgraph &depsgraph, Object &object);

namespace compression {

/**
 * Compress a span with ZSTD, using a prefiltering step that can improve compression speed and
 * ratios for certain data.
 */
template<typename T>
void filter_compress(const Span<T> src,
                     Vector<std::byte> &filter_buffer,
                     Vector<std::byte> &compress_buffer);

/** Decompress data compressed with #filter_compress. */
template<typename T>
void filter_decompress(const Span<std::byte> src, Vector<std::byte> &buffer, Vector<T> &dst);

}  // namespace compression

}  // namespace ed::sculpt_paint::undo

}  // namespace blender
