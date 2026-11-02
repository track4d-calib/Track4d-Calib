from __future__ import annotations

import tempfile
import time
from copy import deepcopy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def target_star_view_pairs(
    *,
    support_views: Sequence[str],
    target_views: Sequence[str],
) -> list[tuple[str, str]]:
    supports = [str(view) for view in support_views]
    targets = [str(view) for view in target_views]
    if len(supports) != len(set(supports)) or len(targets) != len(set(targets)):
        raise ValueError("support_views and target_views must be unique")
    if set(supports) & set(targets):
        raise ValueError("support_views and target_views must be disjoint")
    return [(target, support) for target in targets for support in supports]


def target_star_view_pairs_by_target(
    *,
    support_views: Sequence[str],
    target_views: Sequence[str],
    support_views_by_target: Mapping[str, Sequence[str]],
) -> list[tuple[str, str]]:
    """Return a target-star graph with a deterministic support subset per target."""

    supports = [str(view) for view in support_views]
    targets = [str(view) for view in target_views]
    if len(supports) != len(set(supports)) or len(targets) != len(set(targets)):
        raise ValueError("support_views and target_views must be unique")
    if set(supports) & set(targets):
        raise ValueError("support_views and target_views must be disjoint")
    normalized = {
        str(target): [str(view) for view in views]
        for target, views in support_views_by_target.items()
    }
    if set(normalized) != set(targets):
        raise ValueError("support_views_by_target must exactly cover target_views")
    support_set = set(supports)
    pairs: list[tuple[str, str]] = []
    for target in targets:
        selected = normalized[target]
        if len(selected) != len(set(selected)):
            raise ValueError(f"support subset for target {target} contains duplicates")
        unknown = sorted(set(selected) - support_set)
        if unknown:
            raise ValueError(
                f"support subset for target {target} contains unknown views: {unknown}"
            )
        pairs.extend((target, support) for support in selected)
    return pairs


def target_closure_view_pairs(
    *,
    support_views: Sequence[str],
    target_views: Sequence[str],
) -> list[tuple[str, str]]:
    """Return target-support stars plus every unordered target-target edge."""

    supports = [str(view) for view in support_views]
    targets = [str(view) for view in target_views]
    star_pairs = target_star_view_pairs(
        support_views=supports,
        target_views=targets,
    )
    target_pairs = [
        (targets[first], targets[second])
        for first in range(len(targets))
        for second in range(first + 1, len(targets))
    ]
    return star_pairs + target_pairs


def _inside_mask(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values[..., 0]
    h, w = values.shape[:2]
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, max(w - 1, 0))
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, max(h - 1, 0))
    return values[y, x] > 0


def filter_pair_matches(
    *,
    target_keypoints: np.ndarray,
    support_keypoints: np.ndarray,
    matches0: np.ndarray,
    matching_scores0: np.ndarray | None,
    target_mask: np.ndarray,
    support_mask: np.ndarray,
    f_ransac_px: float,
    rng_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Keep direct target-support matches consistent with one epipolar model."""

    target_points = np.asarray(target_keypoints, dtype=np.float32).reshape(-1, 2)
    support_points = np.asarray(support_keypoints, dtype=np.float32).reshape(-1, 2)
    target_to_support = np.asarray(matches0, dtype=np.int64).reshape(-1)
    count = min(len(target_points), len(target_to_support))
    target_indices = np.flatnonzero(target_to_support[:count] >= 0)
    support_indices = target_to_support[target_indices]
    valid = (support_indices >= 0) & (support_indices < len(support_points))
    target_indices = target_indices[valid]
    support_indices = support_indices[valid]
    if len(target_indices) > 0:
        mask_valid = _inside_mask(target_mask, target_points[target_indices]) & _inside_mask(
            support_mask,
            support_points[support_indices],
        )
        target_indices = target_indices[mask_valid]
        support_indices = support_indices[mask_valid]

    if matching_scores0 is None:
        scores = np.ones((len(target_indices),), dtype=np.float64)
    else:
        all_scores = np.asarray(matching_scores0, dtype=np.float64).reshape(-1)
        scores = np.asarray(
            [all_scores[index] if index < len(all_scores) else 1.0 for index in target_indices],
            dtype=np.float64,
        )
    finite = np.isfinite(scores)
    target_indices = target_indices[finite]
    support_indices = support_indices[finite]
    scores = scores[finite]

    if len(target_indices) < 8:
        return {
            "target_indices": np.empty((0,), dtype=np.int64),
            "support_indices": np.empty((0,), dtype=np.int64),
            "scores": np.empty((0,), dtype=np.float64),
        }

    cv2.setRNGSeed(int(rng_seed))
    try:
        _fundamental, inlier_mask = cv2.findFundamentalMat(
            target_points[target_indices],
            support_points[support_indices],
            method=int(getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)),
            ransacReprojThreshold=float(f_ransac_px),
            confidence=0.999,
            maxIters=2000,
        )
    except cv2.error:
        # USAC can raise when a numerically degenerate sample has no model.
        # Such a pair contributes no geometry-verified correspondences.
        inlier_mask = None
    if inlier_mask is None:
        keep = np.zeros((len(target_indices),), dtype=bool)
    else:
        keep = np.asarray(inlier_mask).reshape(-1) > 0
        if len(keep) != len(target_indices):
            keep = np.zeros((len(target_indices),), dtype=bool)
    return {
        "target_indices": target_indices[keep].astype(np.int64),
        "support_indices": support_indices[keep].astype(np.int64),
        "scores": scores[keep].astype(np.float64),
    }


def _validated_keypoints(
    views: Sequence[str],
    keypoints_by_view: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for raw_view in views:
        view = str(raw_view)
        if view not in keypoints_by_view:
            raise KeyError(f"missing keypoints for view {view}")
        points = np.asarray(keypoints_by_view[view], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"keypoints for view {view} must have shape (N,2)")
        output[view] = points
    return output


def _pair_rows(
    value: Mapping[str, Sequence[float] | Sequence[int]],
    *,
    target_count: int,
    support_count: int,
) -> list[tuple[int, int, float]]:
    target_indices = np.asarray(value["target_indices"], dtype=np.int64).reshape(-1)
    support_indices = np.asarray(value["support_indices"], dtype=np.int64).reshape(-1)
    scores = np.asarray(value.get("scores", np.ones(len(target_indices))), dtype=np.float64).reshape(-1)
    if not (len(target_indices) == len(support_indices) == len(scores)):
        raise ValueError("target_indices, support_indices, and scores must have equal length")
    rows = []
    for target_index, support_index, score in zip(target_indices, support_indices, scores):
        if not (0 <= int(target_index) < int(target_count)):
            continue
        if not (0 <= int(support_index) < int(support_count)):
            continue
        if not np.isfinite(float(score)):
            continue
        rows.append((int(target_index), int(support_index), float(score)))
    return rows


def build_target_star_seed_output(
    *,
    views: Sequence[str],
    ref_view: str,
    target_views: Sequence[str],
    keypoints_by_view: Mapping[str, np.ndarray],
    matches_by_target_support: Mapping[
        str,
        Mapping[str, Mapping[str, Sequence[float] | Sequence[int]]],
    ],
    num_points: int,
    min_support_views: int = 2,
) -> dict[str, Any]:
    """Build stable IDs from direct target-to-support matches only."""

    view_order = [str(view) for view in views]
    if len(view_order) != len(set(view_order)):
        raise ValueError("views must be unique")
    ref = str(ref_view)
    if ref not in view_order:
        raise ValueError(f"ref_view {ref} is absent from views")
    targets = [str(view) for view in target_views]
    if len(targets) != len(set(targets)) or not targets:
        raise ValueError("target_views must be non-empty and unique")
    if set(targets) - set(view_order):
        raise ValueError("target_views must be contained in views")
    targets = sorted(targets)
    supports = [view for view in view_order if view not in set(targets)]
    minimum_support = max(2, int(min_support_views))
    points = _validated_keypoints(view_order, keypoints_by_view)

    candidates_by_target: dict[str, list[dict[str, Any]]] = {}
    edge_count_by_target: dict[str, int] = {}
    for target in targets:
        observations_by_target_index: dict[int, dict[str, tuple[int, float]]] = {}
        edge_count = 0
        pair_mapping = matches_by_target_support.get(target, {})
        for support in supports:
            if support not in pair_mapping:
                continue
            rows = _pair_rows(
                pair_mapping[support],
                target_count=len(points[target]),
                support_count=len(points[support]),
            )
            edge_count += len(rows)
            for target_index, support_index, score in rows:
                support_matches = observations_by_target_index.setdefault(target_index, {})
                previous = support_matches.get(support)
                if previous is None or score > previous[1]:
                    support_matches[support] = (support_index, score)
        edge_count_by_target[target] = int(edge_count)

        candidates = []
        for target_index, support_matches in observations_by_target_index.items():
            if len(support_matches) < minimum_support:
                continue
            observations = {target: int(target_index)}
            observations.update(
                {support: int(index_score[0]) for support, index_score in support_matches.items()}
            )
            scores = [float(index_score[1]) for index_score in support_matches.values()]
            candidates.append(
                {
                    "target": target,
                    "target_index": int(target_index),
                    "observations": observations,
                    "support_count": int(len(support_matches)),
                    "score": float(np.mean(scores)),
                }
            )
        candidates.sort(
            key=lambda row: (
                -int(row["support_count"]),
                -float(row["score"]),
                int(row["target_index"]),
            )
        )
        candidates_by_target[target] = candidates

    budget = max(0, int(num_points))
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < budget:
        added = False
        for target in targets:
            candidates = candidates_by_target[target]
            if depth >= len(candidates):
                continue
            selected.append(candidates[depth])
            added = True
            if len(selected) >= budget:
                break
        if not added:
            break
        depth += 1

    selected_count = len(selected)
    xy_by_view = {
        view: np.zeros((selected_count, 2), dtype=np.float32) for view in view_order
    }
    valid_by_view = {
        view: np.zeros((selected_count,), dtype=bool) for view in view_order
    }
    for point_index, landmark in enumerate(selected):
        for view, keypoint_index in landmark["observations"].items():
            xy_by_view[view][point_index] = points[view][int(keypoint_index)]
            valid_by_view[view][point_index] = True

    selected_count_by_target = {
        target: int(sum(landmark["target"] == target for landmark in selected))
        for target in targets
    }
    diag = {
        "builder": "spsg_target_star",
        "support_views": supports,
        "target_views": targets,
        "min_support_views": int(minimum_support),
        "edge_count_by_target": edge_count_by_target,
        "candidate_count_by_target": {
            target: int(len(candidates_by_target[target])) for target in targets
        },
        "selected_count_by_target": selected_count_by_target,
        "selected_target_sequence": [str(landmark["target"]) for landmark in selected],
        "selected_total": int(selected_count),
        "requested_total": int(budget),
    }
    return {
        "ref_xy": xy_by_view[ref],
        "ref_valid": valid_by_view[ref],
        "xy_by_view": {view: xy_by_view[view] for view in view_order if view != ref},
        "valid_by_view": {view: valid_by_view[view] for view in view_order if view != ref},
        "diag": diag,
    }


class _TargetUnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(int(size), dtype=np.int64)
        self.rank = np.zeros((int(size),), dtype=np.uint8)

    def find(self, value: int) -> int:
        index = int(value)
        while int(self.parent[index]) != index:
            self.parent[index] = self.parent[int(self.parent[index])]
            index = int(self.parent[index])
        return index

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if int(self.rank[root_first]) < int(self.rank[root_second]):
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if int(self.rank[root_first]) == int(self.rank[root_second]):
            self.rank[root_first] = np.uint8(int(self.rank[root_first]) + 1)


def build_target_closure_seed_output(
    *,
    views: Sequence[str],
    ref_view: str,
    target_views: Sequence[str],
    keypoints_by_view: Mapping[str, np.ndarray],
    matches_by_target_support: Mapping[
        str,
        Mapping[str, Mapping[str, Sequence[float] | Sequence[int]]],
    ],
    matches_by_target_target: Mapping[
        str,
        Mapping[str, Mapping[str, Sequence[float] | Sequence[int]]],
    ],
    num_points: int,
    min_support_views: int = 2,
    allow_target_only_multi_target: bool = False,
    require_common_static_observation: bool = False,
) -> dict[str, Any]:
    """Build target-star landmarks with shared IDs across target views."""

    view_order = [str(view) for view in views]
    if len(view_order) != len(set(view_order)):
        raise ValueError("views must be unique")
    ref = str(ref_view)
    if ref not in view_order:
        raise ValueError(f"ref_view {ref} is absent from views")
    targets = [str(view) for view in target_views]
    if len(targets) != len(set(targets)) or not targets:
        raise ValueError("target_views must be non-empty and unique")
    if set(targets) - set(view_order):
        raise ValueError("target_views must be contained in views")
    targets = sorted(targets)
    target_set = set(targets)
    supports = [view for view in view_order if view not in target_set]
    minimum_support = max(2, int(min_support_views))
    points = _validated_keypoints(view_order, keypoints_by_view)

    node_offset: dict[str, int] = {}
    node_target: list[str] = []
    node_keypoint: list[int] = []
    for target in targets:
        node_offset[target] = len(node_target)
        node_target.extend([target] * len(points[target]))
        node_keypoint.extend(range(len(points[target])))
    union_find = _TargetUnionFind(len(node_target))
    active = np.zeros((len(node_target),), dtype=bool)
    support_by_node: dict[int, dict[str, tuple[int, float]]] = {}
    direct_support_count_by_node: dict[int, int] = {}
    edge_count_by_target = {target: 0 for target in targets}

    for target in targets:
        pair_mapping = matches_by_target_support.get(target, {})
        for support in supports:
            if support not in pair_mapping:
                continue
            rows = _pair_rows(
                pair_mapping[support],
                target_count=len(points[target]),
                support_count=len(points[support]),
            )
            edge_count_by_target[target] += len(rows)
            for target_index, support_index, score in rows:
                node = int(node_offset[target] + target_index)
                active[node] = True
                support_matches = support_by_node.setdefault(node, {})
                previous = support_matches.get(support)
                if previous is None or score > previous[1]:
                    support_matches[support] = (support_index, score)

    for node, support_matches in support_by_node.items():
        direct_support_count_by_node[node] = len(support_matches)

    target_target_edge_count = 0
    static_identity_checked_edge_total = 0
    static_identity_verified_edge_total = 0
    static_identity_rejected_edge_total = 0
    target_target_edges: list[tuple[float, int, int]] = []
    direct_target_edges: set[tuple[int, int]] = set()
    for first_index, first in enumerate(targets):
        for second in targets[first_index + 1 :]:
            first_mapping = matches_by_target_target.get(first, {})
            second_mapping = matches_by_target_target.get(second, {})
            reverse = False
            pair_value = first_mapping.get(second)
            if pair_value is None:
                pair_value = second_mapping.get(first)
                reverse = pair_value is not None
            if pair_value is None:
                continue
            if reverse:
                rows = _pair_rows(
                    pair_value,
                    target_count=len(points[second]),
                    support_count=len(points[first]),
                )
                rows = [(second_keypoint, first_keypoint, score) for first_keypoint, second_keypoint, score in rows]
            else:
                rows = _pair_rows(
                    pair_value,
                    target_count=len(points[first]),
                    support_count=len(points[second]),
                )
            target_target_edge_count += len(rows)
            for first_keypoint, second_keypoint, score in rows:
                first_node = int(node_offset[first] + first_keypoint)
                second_node = int(node_offset[second] + second_keypoint)
                if require_common_static_observation:
                    static_identity_checked_edge_total += 1
                    first_support = support_by_node.get(first_node, {})
                    second_support = support_by_node.get(second_node, {})
                    has_common_static_observation = any(
                        support in second_support
                        and int(first_value[0]) == int(second_support[support][0])
                        for support, first_value in first_support.items()
                    )
                    if not has_common_static_observation:
                        static_identity_rejected_edge_total += 1
                        continue
                    static_identity_verified_edge_total += 1
                active[first_node] = True
                active[second_node] = True
                target_target_edges.append((float(score), first_node, second_node))
                direct_target_edges.add(
                    (min(first_node, second_node), max(first_node, second_node))
                )

    members_by_root = {
        int(node): {int(node)} for node in np.flatnonzero(active).tolist()
    }
    complete_link_rejected_edge_total = 0
    for _score, first_node, second_node in sorted(
        target_target_edges,
        key=lambda row: (-row[0], row[1], row[2]),
    ):
        first_root = union_find.find(first_node)
        second_root = union_find.find(second_node)
        if first_root == second_root:
            continue
        first_members = members_by_root[first_root]
        second_members = members_by_root[second_root]
        first_targets = {node_target[node] for node in first_members}
        second_targets = {node_target[node] for node in second_members}
        target_disjoint = not bool(first_targets & second_targets)
        complete_link = all(
            (min(first_member, second_member), max(first_member, second_member))
            in direct_target_edges
            for first_member in first_members
            for second_member in second_members
        )
        if not target_disjoint or not complete_link:
            complete_link_rejected_edge_total += 1
            continue
        del members_by_root[first_root]
        del members_by_root[second_root]
        union_find.union(first_root, second_root)
        merged_root = union_find.find(first_root)
        members_by_root[merged_root] = first_members | second_members

    components: dict[int, list[int]] = {}
    for node in np.flatnonzero(active).tolist():
        components.setdefault(union_find.find(node), []).append(int(node))
    target_target_scores_by_component: dict[int, list[float]] = {}
    for score, first_node, second_node in target_target_edges:
        component = union_find.find(first_node)
        if component == union_find.find(second_node):
            target_target_scores_by_component.setdefault(component, []).extend(
                [float(score), float(score)]
            )

    candidates: list[dict[str, Any]] = []
    target_conflict_component_total = 0
    support_conflict_observation_total = 0
    support_conflict_component_total = 0
    insufficient_direct_support_component_total = 0
    target_only_relative_component_total = 0
    for component, nodes in components.items():
        node_by_target: dict[str, int] = {}
        target_conflict = False
        for node in nodes:
            target = node_target[node]
            if target in node_by_target:
                target_conflict = True
                break
            node_by_target[target] = node
        if target_conflict:
            target_conflict_component_total += 1
            continue
        has_complete_direct_support = not any(
            direct_support_count_by_node.get(node, 0) < minimum_support
            for node in nodes
        )
        target_only_relative = bool(
            allow_target_only_multi_target
            and not has_complete_direct_support
            and len(node_by_target) >= 2
        )
        if not has_complete_direct_support and not target_only_relative:
            insufficient_direct_support_component_total += 1
            continue
        if target_only_relative:
            target_only_relative_component_total += 1

        support_observations: dict[str, int] = {}
        support_scores: list[float] = []
        if not target_only_relative:
            support_candidates: dict[str, dict[int, float]] = {}
            for node in nodes:
                for support, (support_index, score) in support_by_node.get(
                    node, {}
                ).items():
                    by_index = support_candidates.setdefault(support, {})
                    by_index[support_index] = max(
                        float(score), by_index.get(support_index, -np.inf)
                    )

            support_conflict = False
            for support, by_index in support_candidates.items():
                if len(by_index) != 1:
                    support_conflict_observation_total += 1
                    support_conflict = True
                    break
                support_index, score = next(iter(by_index.items()))
                support_observations[support] = int(support_index)
                support_scores.append(float(score))
            if support_conflict:
                support_conflict_component_total += 1
                continue
            if len(support_observations) < minimum_support:
                continue

        observations = {
            target: int(node_keypoint[node]) for target, node in node_by_target.items()
        }
        observations.update(support_observations)
        target_target_scores = target_target_scores_by_component.get(component, [])
        scores = support_scores + target_target_scores
        candidates.append(
            {
                "targets": tuple(target for target in targets if target in node_by_target),
                "nodes": node_by_target,
                "observations": observations,
                "support_count": int(len(support_observations)),
                "score": float(np.mean(scores)) if scores else 0.0,
                "rescued_targets": (),
                "target_only_relative": target_only_relative,
            }
        )

    candidates.sort(
        key=lambda row: (
            -int(row["support_count"]),
            -len(row["targets"]),
            -float(row["score"]),
            tuple((target, int(node_keypoint[row["nodes"][target]])) for target in row["targets"]),
        )
    )
    candidate_indices_by_target = {
        target: [index for index, row in enumerate(candidates) if target in row["targets"]]
        for target in targets
    }
    offsets = {target: 0 for target in targets}
    selected_indices: list[int] = []
    selected_index_set: set[int] = set()
    selected_target_sequence: list[str] = []
    budget = max(0, int(num_points))
    while len(selected_indices) < budget:
        added = False
        for target in targets:
            indices = candidate_indices_by_target[target]
            while offsets[target] < len(indices) and indices[offsets[target]] in selected_index_set:
                offsets[target] += 1
            if offsets[target] >= len(indices):
                continue
            candidate_index = int(indices[offsets[target]])
            offsets[target] += 1
            selected_indices.append(candidate_index)
            selected_index_set.add(candidate_index)
            selected_target_sequence.append(target)
            added = True
            if len(selected_indices) >= budget:
                break
        if not added:
            break

    selected = [candidates[index] for index in selected_indices]
    selected_count = len(selected)
    xy_by_view = {
        view: np.zeros((selected_count, 2), dtype=np.float32) for view in view_order
    }
    valid_by_view = {
        view: np.zeros((selected_count,), dtype=bool) for view in view_order
    }
    for point_index, landmark in enumerate(selected):
        for view, keypoint_index in landmark["observations"].items():
            xy_by_view[view][point_index] = points[view][int(keypoint_index)]
            valid_by_view[view][point_index] = True

    selected_count_by_target = {
        target: int(sum(target in landmark["targets"] for landmark in selected))
        for target in targets
    }
    diag = {
        "builder": (
            "spsg_target_relative_closure"
            if allow_target_only_multi_target
            else (
                "spsg_target_verified_closure"
                if require_common_static_observation
                else "spsg_target_closure"
            )
        ),
        "relative_track_policy": (
            "multi_target_components_need_no_static_support; "
            "absolute_gauge_is_deferred_to_primary_anchors"
            if allow_target_only_multi_target
            else "all_target_nodes_require_direct_static_support"
        ),
        "support_views": supports,
        "target_views": targets,
        "min_support_views": int(minimum_support),
        "edge_count_by_target": edge_count_by_target,
        "target_target_edge_count": int(target_target_edge_count),
        "require_common_static_observation": bool(
            require_common_static_observation
        ),
        "static_identity_checked_edge_total": int(
            static_identity_checked_edge_total
        ),
        "static_identity_verified_edge_total": int(
            static_identity_verified_edge_total
        ),
        "static_identity_rejected_edge_total": int(
            static_identity_rejected_edge_total
        ),
        "complete_link_rejected_edge_total": int(complete_link_rejected_edge_total),
        "component_total": int(len(components)),
        "target_conflict_component_total": int(target_conflict_component_total),
        "support_conflict_observation_total": int(support_conflict_observation_total),
        "support_conflict_component_total": int(support_conflict_component_total),
        "insufficient_direct_support_component_total": int(
            insufficient_direct_support_component_total
        ),
        "target_only_relative_component_total": int(
            target_only_relative_component_total
        ),
        "candidate_total": int(len(candidates)),
        "candidate_multi_target_total": int(
            sum(len(landmark["targets"]) > 1 for landmark in candidates)
        ),
        "selected_multi_target_total": int(
            sum(len(landmark["targets"]) > 1 for landmark in selected)
        ),
        "candidate_target_only_multi_target_total": int(
            sum(bool(landmark["target_only_relative"]) for landmark in candidates)
        ),
        "selected_target_only_multi_target_total": int(
            sum(bool(landmark["target_only_relative"]) for landmark in selected)
        ),
        "selected_count_by_target": selected_count_by_target,
        "selected_target_sequence": selected_target_sequence,
        "rescued_target_observation_total": int(
            sum(len(landmark["rescued_targets"]) for landmark in selected)
        ),
        "selected_total": int(selected_count),
        "requested_total": int(budget),
    }
    return {
        "ref_xy": xy_by_view[ref],
        "ref_valid": valid_by_view[ref],
        "xy_by_view": {view: xy_by_view[view] for view in view_order if view != ref},
        "valid_by_view": {view: valid_by_view[view] for view in view_order if view != ref},
        "diag": diag,
    }


def build_target_relay_seed_output(
    *,
    views: Sequence[str],
    ref_view: str,
    target_views: Sequence[str],
    keypoints_by_view: Mapping[str, np.ndarray],
    matches_by_target_support: Mapping[
        str,
        Mapping[str, Mapping[str, Sequence[float] | Sequence[int]]],
    ],
    matches_by_target_target: Mapping[
        str,
        Mapping[str, Mapping[str, Sequence[float] | Sequence[int]]],
    ],
    num_points: int,
    min_support_views: int = 2,
) -> dict[str, Any]:
    """Build direct target-centered relay tracks without transitive merging.

    Every landmark is anchored at one target keypoint and contains only direct
    pair matches from that keypoint.  Static and target neighbors contribute to
    the same support count, allowing registered same-time target cameras to
    triangulate a point when direct static-map coverage is weak.
    """

    view_order = [str(view) for view in views]
    if len(view_order) != len(set(view_order)):
        raise ValueError("views must be unique")
    ref = str(ref_view)
    if ref not in view_order:
        raise ValueError(f"ref_view {ref} is absent from views")
    targets = [str(view) for view in target_views]
    if len(targets) != len(set(targets)) or not targets:
        raise ValueError("target_views must be non-empty and unique")
    if set(targets) - set(view_order):
        raise ValueError("target_views must be contained in views")
    target_set = set(targets)
    supports = [view for view in view_order if view not in target_set]
    minimum_support = max(2, int(min_support_views))
    points = _validated_keypoints(view_order, keypoints_by_view)

    candidates_by_signature: dict[
        tuple[tuple[str, int], ...], dict[str, Any]
    ] = {}
    direct_static_edge_count = 0
    direct_target_edge_count = 0
    raw_candidate_count_by_focus: dict[str, int] = {}
    for focus in targets:
        observations_by_focus_index: dict[
            int, dict[str, tuple[int, float, str]]
        ] = {}
        for support in supports:
            pair_value = matches_by_target_support.get(focus, {}).get(support)
            if pair_value is None:
                continue
            rows = _pair_rows(
                pair_value,
                target_count=len(points[focus]),
                support_count=len(points[support]),
            )
            direct_static_edge_count += len(rows)
            for focus_index, support_index, score in rows:
                by_view = observations_by_focus_index.setdefault(focus_index, {})
                previous = by_view.get(support)
                if previous is None or float(score) > previous[1]:
                    by_view[support] = (support_index, float(score), "static")

        for neighbor in targets:
            if neighbor == focus:
                continue
            pair_value = matches_by_target_target.get(focus, {}).get(neighbor)
            reverse = False
            if pair_value is None:
                pair_value = matches_by_target_target.get(neighbor, {}).get(focus)
                reverse = pair_value is not None
            if pair_value is None:
                continue
            if reverse:
                reverse_rows = _pair_rows(
                    pair_value,
                    target_count=len(points[neighbor]),
                    support_count=len(points[focus]),
                )
                rows = [
                    (focus_index, neighbor_index, score)
                    for neighbor_index, focus_index, score in reverse_rows
                ]
            else:
                rows = _pair_rows(
                    pair_value,
                    target_count=len(points[focus]),
                    support_count=len(points[neighbor]),
                )
            direct_target_edge_count += len(rows)
            for focus_index, neighbor_index, score in rows:
                by_view = observations_by_focus_index.setdefault(focus_index, {})
                previous = by_view.get(neighbor)
                if previous is None or float(score) > previous[1]:
                    by_view[neighbor] = (neighbor_index, float(score), "target")

        raw_candidate_count = 0
        for focus_index, support_matches in observations_by_focus_index.items():
            if len(support_matches) < minimum_support:
                continue
            raw_candidate_count += 1
            observations = {focus: int(focus_index)}
            observations.update(
                {
                    support: int(index_score_kind[0])
                    for support, index_score_kind in support_matches.items()
                }
            )
            static_count = int(
                sum(value[2] == "static" for value in support_matches.values())
            )
            target_count = int(
                sum(value[2] == "target" for value in support_matches.values())
            )
            scores = [float(value[1]) for value in support_matches.values()]
            signature = tuple(sorted(observations.items()))
            candidate = {
                "focus": focus,
                "focus_index": int(focus_index),
                "observations": observations,
                "support_count": int(len(support_matches)),
                "static_support_count": static_count,
                "target_support_count": target_count,
                "score": float(np.mean(scores)),
                "relay_only": bool(static_count < minimum_support),
            }
            previous = candidates_by_signature.get(signature)
            previous_rank = (
                int(previous["support_count"]),
                float(previous["score"]),
                -targets.index(str(previous["focus"])),
            ) if previous is not None else None
            candidate_rank = (
                int(candidate["support_count"]),
                float(candidate["score"]),
                -targets.index(focus),
            )
            if previous_rank is None or candidate_rank > previous_rank:
                candidates_by_signature[signature] = candidate
        raw_candidate_count_by_focus[focus] = int(raw_candidate_count)

    candidates = list(candidates_by_signature.values())
    candidates.sort(
        key=lambda row: (
            -int(row["support_count"]),
            -int(row["target_support_count"]),
            -float(row["score"]),
            targets.index(str(row["focus"])),
            int(row["focus_index"]),
        )
    )
    candidate_indices_by_focus = {
        target: [
            index
            for index, row in enumerate(candidates)
            if str(row["focus"]) == target
        ]
        for target in targets
    }
    offsets = {target: 0 for target in targets}
    selected_indices: list[int] = []
    budget = max(0, int(num_points))
    while len(selected_indices) < budget:
        added = False
        for target in targets:
            indices = candidate_indices_by_focus[target]
            if offsets[target] >= len(indices):
                continue
            selected_indices.append(int(indices[offsets[target]]))
            offsets[target] += 1
            added = True
            if len(selected_indices) >= budget:
                break
        if not added:
            break

    selected = [candidates[index] for index in selected_indices]
    selected_count = len(selected)
    xy_by_view = {
        view: np.zeros((selected_count, 2), dtype=np.float32)
        for view in view_order
    }
    valid_by_view = {
        view: np.zeros((selected_count,), dtype=bool) for view in view_order
    }
    for point_index, landmark in enumerate(selected):
        for view, keypoint_index in landmark["observations"].items():
            xy_by_view[view][point_index] = points[view][int(keypoint_index)]
            valid_by_view[view][point_index] = True

    selected_count_by_focus = {
        target: int(sum(landmark["focus"] == target for landmark in selected))
        for target in targets
    }
    diag = {
        "builder": "spsg_target_relay",
        "support_views": supports,
        "target_views": targets,
        "min_support_views": int(minimum_support),
        "direct_support_policy": "focus_centered_direct_edges_only",
        "transitive_merge_count": 0,
        "direct_static_edge_count": int(direct_static_edge_count),
        "direct_target_edge_count": int(direct_target_edge_count // 2),
        "raw_candidate_count_by_focus": raw_candidate_count_by_focus,
        "candidate_count_by_focus": {
            target: int(len(candidate_indices_by_focus[target]))
            for target in targets
        },
        "selected_count_by_focus": selected_count_by_focus,
        "selected_focus_sequence": [str(landmark["focus"]) for landmark in selected],
        "candidate_relay_only_total": int(
            sum(bool(landmark["relay_only"]) for landmark in candidates)
        ),
        "selected_relay_only_total": int(
            sum(bool(landmark["relay_only"]) for landmark in selected)
        ),
        "candidate_total": int(len(candidates)),
        "selected_total": int(selected_count),
        "requested_total": int(budget),
    }
    return {
        "ref_xy": xy_by_view[ref],
        "ref_valid": valid_by_view[ref],
        "xy_by_view": {
            view: xy_by_view[view] for view in view_order if view != ref
        },
        "valid_by_view": {
            view: valid_by_view[view] for view in view_order if view != ref
        },
        "diag": diag,
    }


def build_target_star_seed_from_images(
    *,
    rgb_by_view: Mapping[str, np.ndarray],
    mask_by_view: Mapping[str, np.ndarray],
    views: Sequence[str],
    ref_view: str,
    target_views: Sequence[str],
    num_points: int,
    min_support_views: int = 2,
    f_ransac_px: float = 2.0,
    rng_seed: int = 0,
    feature_conf_name: str = "superpoint_aachen",
    max_keypoints: int | None = 2048,
    support_views_by_target: Mapping[str, Sequence[str]] | None = None,
    _target_closure: bool = False,
    _target_verified_closure: bool = False,
    _target_relative_closure: bool = False,
    _target_relay: bool = False,
) -> dict[str, Any]:
    """Extract SPSG target-star matches and return multi-view seed points."""

    import h5py
    from hloc import extract_features, match_features
    from hloc.utils.parsers import names_to_pair

    view_order = [str(view) for view in views]
    targets = [str(view) for view in target_views]
    target_set = set(targets)
    supports = [view for view in view_order if view not in target_set]
    if sum(
        (
            _target_closure,
            _target_verified_closure,
            _target_relative_closure,
            _target_relay,
        )
    ) > 1:
        raise ValueError("target closure modes and target relay are mutually exclusive")
    if support_views_by_target is not None and (
        _target_closure
        or _target_verified_closure
        or _target_relative_closure
        or _target_relay
    ):
        raise ValueError("per-target support subsets are only valid for target-star matching")
    if support_views_by_target is not None:
        pairs = target_star_view_pairs_by_target(
            support_views=supports,
            target_views=targets,
            support_views_by_target=support_views_by_target,
        )
    elif (
        _target_closure
        or _target_verified_closure
        or _target_relative_closure
        or _target_relay
    ):
        pairs = target_closure_view_pairs(
            support_views=supports,
            target_views=targets,
        )
    else:
        pairs = target_star_view_pairs(
            support_views=supports,
            target_views=targets,
        )
    if not pairs:
        raise ValueError("target-star matching requires targets and support views")
    for view in view_order:
        if view not in rgb_by_view or view not in mask_by_view:
            raise KeyError(f"missing image or mask for view {view}")

    image_name_by_view = {view: f"view{view}.png" for view in view_order}
    runtime_started = time.perf_counter()
    feature_runtime_sec = 0.0
    matching_runtime_sec = 0.0
    with tempfile.TemporaryDirectory(prefix="track4d_calib_spsg_target_star_") as tmp:
        root = Path(tmp)
        image_dir = root / "images"
        hloc_dir = root / "hloc"
        image_dir.mkdir(parents=True)
        hloc_dir.mkdir(parents=True)
        for view in view_order:
            rgb = np.asarray(rgb_by_view[view], dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"RGB image for view {view} must have shape (H,W,3)")
            destination = image_dir / image_name_by_view[view]
            if not cv2.imwrite(str(destination), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"failed to write temporary SPSG image {destination}")

        pairs_path = hloc_dir / "pairs.txt"
        pairs_path.write_text(
            "".join(
                f"{image_name_by_view[target]} {image_name_by_view[support]}\n"
                for target, support in pairs
            ),
            encoding="utf-8",
        )
        feature_name = str(feature_conf_name)
        if feature_name not in {
            "superpoint_aachen",
            "superpoint_max",
            "superpoint_native",
        }:
            raise ValueError(
                "target-star SPSG feature_conf_name must be "
                "superpoint_aachen, superpoint_max, or superpoint_native"
            )
        if feature_name == "superpoint_native":


            feature_conf = {
                "output": "feats-superpoint-n4096-rnative4096",
                "model": {
                    "name": "superpoint",
                    "nms_radius": 3,
                    "max_keypoints": 4096,
                },
                "preprocessing": {
                    "grayscale": True,
                    "resize_max": 4096,
                    "resize_force": False,
                },
            }
        else:
            feature_conf = deepcopy(extract_features.confs[feature_name])
        if max_keypoints is not None:
            keypoint_budget = int(max_keypoints)
            if keypoint_budget <= 0:
                raise ValueError("max_keypoints must be positive when specified")
            feature_conf["model"]["max_keypoints"] = keypoint_budget
            feature_conf["output"] = str(feature_conf["output"]).replace(
                "n4096", f"n{keypoint_budget}"
            )
        matcher_conf = match_features.confs["superglue"]
        feature_started = time.perf_counter()
        features_path = extract_features.main(
            feature_conf,
            image_dir=image_dir,
            export_dir=hloc_dir,
            image_list=[image_name_by_view[view] for view in view_order],
            overwrite=True,
        )
        feature_runtime_sec = float(time.perf_counter() - feature_started)
        matches_out = hloc_dir / (
            f"{features_path.stem}_{matcher_conf['output']}_{pairs_path.stem}.h5"
        )
        matching_started = time.perf_counter()
        matches_path = match_features.main(
            matcher_conf,
            pairs=pairs_path,
            features=features_path,
            matches=matches_out,
            overwrite=True,
        )
        matching_runtime_sec = float(time.perf_counter() - matching_started)

        keypoints_by_view: dict[str, np.ndarray] = {}
        with h5py.File(features_path, "r") as features_file:
            for view in view_order:
                name = image_name_by_view[view]
                if name not in features_file:
                    raise RuntimeError(f"missing SPSG features for {name}")
                keypoints_by_view[view] = np.asarray(
                    features_file[name]["keypoints"],
                    dtype=np.float32,
                )

        matches_by_target_support: dict[str, dict[str, dict[str, np.ndarray]]] = {
            target: {} for target in targets
        }
        matches_by_target_target: dict[str, dict[str, dict[str, np.ndarray]]] = {
            target: {} for target in targets
        }
        pair_used = 0
        inlier_total = 0
        pair_inlier_counts: dict[str, int] = {}
        with h5py.File(matches_path, "r") as matches_file:
            for pair_index, (target, support) in enumerate(pairs):
                target_name = image_name_by_view[target]
                support_name = image_name_by_view[support]
                pair_key = names_to_pair(target_name, support_name)
                group = None
                if pair_key in matches_file:
                    group = matches_file[pair_key]
                elif target_name in matches_file and support_name in matches_file[target_name]:
                    group = matches_file[target_name][support_name]
                if group is None or "matches0" not in group:
                    pair_inlier_counts[f"{target}:{support}"] = 0
                    continue
                filtered = filter_pair_matches(
                    target_keypoints=keypoints_by_view[target],
                    support_keypoints=keypoints_by_view[support],
                    matches0=np.asarray(group["matches0"]),
                    matching_scores0=(
                        np.asarray(group["matching_scores0"])
                        if "matching_scores0" in group
                        else None
                    ),
                    target_mask=np.asarray(mask_by_view[target]),
                    support_mask=np.asarray(mask_by_view[support]),
                    f_ransac_px=float(f_ransac_px),
                    rng_seed=int(rng_seed + pair_index),
                )
                count = int(len(filtered["target_indices"]))
                pair_inlier_counts[f"{target}:{support}"] = count
                if count <= 0:
                    continue
                pair_used += 1
                inlier_total += count
                if support in target_set:
                    matches_by_target_target[target][support] = filtered
                else:
                    matches_by_target_support[target][support] = filtered

    if _target_closure or _target_verified_closure or _target_relative_closure:
        output = build_target_closure_seed_output(
            views=view_order,
            ref_view=str(ref_view),
            target_views=targets,
            keypoints_by_view=keypoints_by_view,
            matches_by_target_support=matches_by_target_support,
            matches_by_target_target=matches_by_target_target,
            num_points=int(num_points),
            min_support_views=int(min_support_views),
            allow_target_only_multi_target=bool(_target_relative_closure),
            require_common_static_observation=bool(_target_verified_closure),
        )
    elif _target_relay:
        output = build_target_relay_seed_output(
            views=view_order,
            ref_view=str(ref_view),
            target_views=targets,
            keypoints_by_view=keypoints_by_view,
            matches_by_target_support=matches_by_target_support,
            matches_by_target_target=matches_by_target_target,
            num_points=int(num_points),
            min_support_views=int(min_support_views),
        )
    else:
        output = build_target_star_seed_output(
            views=view_order,
            ref_view=str(ref_view),
            target_views=targets,
            keypoints_by_view=keypoints_by_view,
            matches_by_target_support=matches_by_target_support,
            num_points=int(num_points),
            min_support_views=int(min_support_views),
        )
    output["diag"].update(
        {
            "pair_total": int(len(pairs)),
            "pair_used": int(pair_used),
            "match_inlier_total": int(inlier_total),
            "pair_inlier_counts": pair_inlier_counts,
            "f_ransac_px": float(f_ransac_px),
            "feature_conf_name": feature_name,
            "feature_output": str(feature_conf["output"]),
            "feature_resize_max": int(
                feature_conf["preprocessing"]["resize_max"]
            ),
            "feature_resize_force": bool(
                feature_conf["preprocessing"].get("resize_force", False)
            ),
            "max_keypoints": int(feature_conf["model"]["max_keypoints"]),
            "matcher_conf_name": "superglue",
            "support_views_by_target": (
                {
                    str(target): [str(view) for view in support_views_by_target[target]]
                    for target in targets
                }
                if support_views_by_target is not None
                else None
            ),
            "runtime_sec": {
                "feature_extraction": feature_runtime_sec,
                "superglue_matching": matching_runtime_sec,
                "geometry_postprocess": float(
                    time.perf_counter()
                    - runtime_started
                    - feature_runtime_sec
                    - matching_runtime_sec
                ),
                "total": float(time.perf_counter() - runtime_started),
            },
        }
    )
    return output


def build_target_closure_seed_from_images(
    *,
    rgb_by_view: Mapping[str, np.ndarray],
    mask_by_view: Mapping[str, np.ndarray],
    views: Sequence[str],
    ref_view: str,
    target_views: Sequence[str],
    num_points: int,
    min_support_views: int = 2,
    f_ransac_px: float = 2.0,
    rng_seed: int = 0,
    feature_conf_name: str = "superpoint_aachen",
) -> dict[str, Any]:
    """Extract target stars plus target-target edges for shared landmark IDs."""

    return build_target_star_seed_from_images(
        rgb_by_view=rgb_by_view,
        mask_by_view=mask_by_view,
        views=views,
        ref_view=ref_view,
        target_views=target_views,
        num_points=num_points,
        min_support_views=min_support_views,
        f_ransac_px=f_ransac_px,
        rng_seed=rng_seed,
        feature_conf_name=feature_conf_name,
        _target_closure=True,
    )


def build_target_verified_closure_seed_from_images(
    *,
    rgb_by_view: Mapping[str, np.ndarray],
    mask_by_view: Mapping[str, np.ndarray],
    views: Sequence[str],
    ref_view: str,
    target_views: Sequence[str],
    num_points: int,
    min_support_views: int = 2,
    f_ransac_px: float = 2.0,
    rng_seed: int = 0,
    feature_conf_name: str = "superpoint_aachen",
) -> dict[str, Any]:
    """Share target IDs only when a direct static observation confirms identity."""

    return build_target_star_seed_from_images(
        rgb_by_view=rgb_by_view,
        mask_by_view=mask_by_view,
        views=views,
        ref_view=ref_view,
        target_views=target_views,
        num_points=num_points,
        min_support_views=min_support_views,
        f_ransac_px=f_ransac_px,
        rng_seed=rng_seed,
        feature_conf_name=feature_conf_name,
        _target_verified_closure=True,
    )


def build_target_relative_closure_seed_from_images(
    *,
    rgb_by_view: Mapping[str, np.ndarray],
    mask_by_view: Mapping[str, np.ndarray],
    views: Sequence[str],
    ref_view: str,
    target_views: Sequence[str],
    num_points: int,
    min_support_views: int = 2,
    f_ransac_px: float = 2.0,
    rng_seed: int = 0,
    feature_conf_name: str = "superpoint_aachen",
) -> dict[str, Any]:
    """Extract target-relative tracks while deferring absolute gauge to anchors."""

    return build_target_star_seed_from_images(
        rgb_by_view=rgb_by_view,
        mask_by_view=mask_by_view,
        views=views,
        ref_view=ref_view,
        target_views=target_views,
        num_points=num_points,
        min_support_views=min_support_views,
        f_ransac_px=f_ransac_px,
        rng_seed=rng_seed,
        feature_conf_name=feature_conf_name,
        _target_relative_closure=True,
    )


def build_target_relay_seed_from_images(
    *,
    rgb_by_view: Mapping[str, np.ndarray],
    mask_by_view: Mapping[str, np.ndarray],
    views: Sequence[str],
    ref_view: str,
    target_views: Sequence[str],
    num_points: int,
    min_support_views: int = 2,
    f_ransac_px: float = 2.0,
    rng_seed: int = 0,
    feature_conf_name: str = "superpoint_aachen",
) -> dict[str, Any]:
    """Extract direct target-centered stars including target neighbors."""

    return build_target_star_seed_from_images(
        rgb_by_view=rgb_by_view,
        mask_by_view=mask_by_view,
        views=views,
        ref_view=ref_view,
        target_views=target_views,
        num_points=num_points,
        min_support_views=min_support_views,
        f_ransac_px=f_ransac_px,
        rng_seed=rng_seed,
        feature_conf_name=feature_conf_name,
        _target_relay=True,
    )
