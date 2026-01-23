#!/usr/bin/env python3
"""
Path tracking infrastructure for incremental computation with OOM recovery.

Tracks three states per path:
1. Completed - path computed successfully, never retry
2. Failed at tier M - path OOM'd at memory tier M, retry at M+1
3. Untried - path never attempted at this tier

Handles branch failures: if intermediate multiplication fails, all downstream
paths are marked as failed.
"""

import json
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict


def generate_path_id(node_types: list[str], predicates: list[str], directions: list[str]) -> str:
    """
    Generate unique identifier for an N-hop path.

    Args:
        node_types: List of node types in path (length N+1)
        predicates: List of predicates in path (length N)
        directions: List of directions in path (length N)

    Returns:
        Path ID string with format: Type1|pred1|dir1|Type2__Type2|pred2|dir2|Type3

    Example:
        node_types = ["SmallMolecule", "Disease", "Gene"]
        predicates = ["treats", "affects"]
        directions = ["F", "F"]
        -> "SmallMolecule|treats|F|Disease__Disease|affects|F|Gene"
    """
    if len(node_types) != len(predicates) + 1:
        raise ValueError(f"Expected {len(predicates)+1} node types, got {len(node_types)}")

    if len(predicates) != len(directions):
        raise ValueError(f"Predicates and directions must have same length")

    segments = []
    for i in range(len(predicates)):
        segment = f"{node_types[i]}|{predicates[i]}|{directions[i]}|{node_types[i+1]}"
        segments.append(segment)

    return "__".join(segments)


def parse_path_id(path_id: str) -> Tuple[list[str], list[str], list[str]]:
    """
    Parse path ID back into components.

    Returns:
        Tuple of (node_types, predicates, directions)
    """
    segments = path_id.split("__")

    node_types = []
    predicates = []
    directions = []

    for segment in segments:
        parts = segment.split("|")
        if len(parts) != 4:
            raise ValueError(f"Invalid segment format: {segment}")

        src_type, pred, direction, tgt_type = parts

        if not node_types:
            node_types.append(src_type)
        node_types.append(tgt_type)
        predicates.append(pred)
        directions.append(direction)

    return node_types, predicates, directions


def get_tracking_dir(results_dir: str, matrix1_index: int) -> Path:
    """Get directory for tracking files for a specific Matrix1 job."""
    tracking_dir = Path(results_dir) / "tracking" / f"matrix1_{matrix1_index:03d}"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    return tracking_dir


def get_completed_paths_file(results_dir: str, matrix1_index: int) -> Path:
    """Get path to completed paths file."""
    return get_tracking_dir(results_dir, matrix1_index) / "completed_paths.txt"


def get_failed_paths_file(results_dir: str, matrix1_index: int) -> Path:
    """Get path to failed paths file."""
    return get_tracking_dir(results_dir, matrix1_index) / "failed_paths.jsonl"


def get_path_in_progress_file(results_dir: str, matrix1_index: int) -> Path:
    """Get path to current path in progress file."""
    return get_tracking_dir(results_dir, matrix1_index) / "path_in_progress.txt"


def load_completed_paths(results_dir: str, matrix1_index: int) -> Set[str]:
    """
    Load set of completed path IDs.

    Returns:
        Set of path ID strings
    """
    completed_file = get_completed_paths_file(results_dir, matrix1_index)

    if not completed_file.exists():
        return set()

    completed = set()
    with open(completed_file, 'r') as f:
        for line in f:
            path_id = line.strip()
            if path_id:
                completed.add(path_id)

    return completed


def load_failed_paths(results_dir: str, matrix1_index: int, memory_gb: int = None) -> Set[str]:
    """
    Load set of failed path IDs, optionally filtered by memory tier.

    Args:
        results_dir: Results directory
        matrix1_index: Matrix1 index
        memory_gb: If specified, only return paths that failed at this tier

    Returns:
        Set of path ID strings
    """
    failed_file = get_failed_paths_file(results_dir, matrix1_index)

    if not failed_file.exists():
        return set()

    failed = set()
    with open(failed_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            path_id = record['path_id']
            tier = record['memory_gb']

            if memory_gb is None or tier == memory_gb:
                failed.add(path_id)

    return failed


def load_failed_paths_by_tier(results_dir: str, matrix1_index: int) -> Dict[int, Set[str]]:
    """
    Load failed paths grouped by memory tier.

    Returns:
        Dict mapping memory_gb -> set of path IDs
    """
    failed_file = get_failed_paths_file(results_dir, matrix1_index)

    if not failed_file.exists():
        return defaultdict(set)

    by_tier = defaultdict(set)
    with open(failed_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            path_id = record['path_id']
            tier = record['memory_gb']

            by_tier[tier].add(path_id)

    return by_tier


def record_completed_path(path_id: str, results_dir: str, matrix1_index: int):
    """
    Record that a path was completed successfully.

    Appends to completed_paths.txt (one path ID per line).
    """
    completed_file = get_completed_paths_file(results_dir, matrix1_index)

    with open(completed_file, 'a') as f:
        f.write(path_id + '\n')
        f.flush()  # Ensure written to disk


def record_failed_path(path_id: str, results_dir: str, matrix1_index: int,
                      memory_gb: int, depth: int = None, reason: str = "oom"):
    """
    Record that a path failed at a specific memory tier.

    Appends to failed_paths.jsonl with metadata.

    Args:
        path_id: Path identifier
        results_dir: Results directory
        matrix1_index: Matrix1 index
        memory_gb: Memory tier where failure occurred
        depth: Hop depth where failure occurred (None if at final N-hop)
        reason: Failure reason (default: "oom")
    """
    failed_file = get_failed_paths_file(results_dir, matrix1_index)

    record = {
        'path_id': path_id,
        'memory_gb': memory_gb,
        'depth': depth,
        'reason': reason,
        'timestamp': None  # Will be added by caller if needed
    }

    with open(failed_file, 'a') as f:
        f.write(json.dumps(record) + '\n')
        f.flush()


def record_path_in_progress(path_id: str, results_dir: str, matrix1_index: int,
                           depth: int, memory_gb: int):
    """
    Record which path is currently being computed.

    Overwrites path_in_progress.txt with single line.
    If job OOMs, orchestrator can read this to know what failed.

    Args:
        path_id: Path identifier (may be partial if depth < n_hops)
        results_dir: Results directory
        matrix1_index: Matrix1 index
        depth: Current hop depth
        memory_gb: Current memory tier
    """
    progress_file = get_path_in_progress_file(results_dir, matrix1_index)

    record = {
        'path_id': path_id,
        'depth': depth,
        'memory_gb': memory_gb
    }

    with open(progress_file, 'w') as f:
        f.write(json.dumps(record) + '\n')
        f.flush()


def clear_path_in_progress(results_dir: str, matrix1_index: int):
    """Clear the path in progress file (called on successful path completion)."""
    progress_file = get_path_in_progress_file(results_dir, matrix1_index)
    if progress_file.exists():
        progress_file.unlink()


def read_path_in_progress(results_dir: str, matrix1_index: int) -> dict:
    """
    Read which path was in progress when job died.

    Returns:
        Dict with 'path_id', 'depth', 'memory_gb' or None if no path in progress
    """
    progress_file = get_path_in_progress_file(results_dir, matrix1_index)

    if not progress_file.exists():
        return None

    with open(progress_file, 'r') as f:
        line = f.readline().strip()
        if not line:
            return None
        return json.loads(line)


def enumerate_downstream_paths(
    partial_path_id: str,
    all_matrices: list,
    n_hops: int,
    current_depth: int
) -> List[str]:
    """
    Enumerate all possible N-hop completions of a partial path.

    This is called when an intermediate multiplication fails. We need to mark
    all downstream paths as failed since we can't compute them without the
    intermediate result.

    Args:
        partial_path_id: Path ID up to failure point (may be shorter than N hops)
        all_matrices: List of (src_type, pred, tgt_type, matrix, direction) tuples
        n_hops: Target number of hops
        current_depth: Depth where failure occurred

    Returns:
        List of complete N-hop path IDs that would have extended this partial path

    Example:
        partial_path_id = "SmallMolecule|treats|F|Disease"
        current_depth = 1
        n_hops = 3
        -> Returns all 3-hop paths starting with "SmallMolecule|treats|F|Disease__..."
    """
    if current_depth >= n_hops:
        # Already at full depth, no enumeration needed
        return [partial_path_id]

    # Parse partial path to get current endpoint type
    node_types, predicates, directions = parse_path_id(partial_path_id)
    current_target_type = node_types[-1]

    # Build index of matrices by source type for efficient lookup
    by_source_type = defaultdict(list)
    for src_type, pred, tgt_type, matrix, direction in all_matrices:
        by_source_type[src_type].append((src_type, pred, tgt_type, direction))

    # Recursively enumerate all extensions
    def enumerate_recursive(path_types, path_preds, path_dirs, depth):
        """Recursively build all possible path extensions."""
        if depth == n_hops:
            # Reached target depth, generate path ID
            return [generate_path_id(path_types, path_preds, path_dirs)]

        current_end_type = path_types[-1]

        if current_end_type not in by_source_type:
            return []

        all_paths = []
        for src_type, pred, tgt_type, direction in by_source_type[current_end_type]:
            # Extend path
            extended_paths = enumerate_recursive(
                path_types + [tgt_type],
                path_preds + [pred],
                path_dirs + [direction],
                depth + 1
            )
            all_paths.extend(extended_paths)

        return all_paths

    # Start enumeration from current partial path
    all_completions = enumerate_recursive(node_types, predicates, directions, current_depth)

    return all_completions


def count_completed_paths(results_dir: str, matrix1_index: int) -> int:
    """Count number of completed paths."""
    completed = load_completed_paths(results_dir, matrix1_index)
    return len(completed)


def count_failed_paths(results_dir: str, matrix1_index: int, memory_gb: int = None) -> int:
    """Count number of failed paths, optionally at specific tier."""
    failed = load_failed_paths(results_dir, matrix1_index, memory_gb)
    return len(failed)


def get_path_statistics(results_dir: str, matrix1_index: int) -> dict:
    """
    Get comprehensive statistics about path computation progress.

    Returns:
        Dict with keys:
        - completed: Number of completed paths
        - failed_by_tier: Dict of memory_gb -> count
        - total_failed: Total failed across all tiers
    """
    completed = load_completed_paths(results_dir, matrix1_index)
    failed_by_tier = load_failed_paths_by_tier(results_dir, matrix1_index)

    total_failed = sum(len(paths) for paths in failed_by_tier.values())

    return {
        'completed': len(completed),
        'failed_by_tier': {tier: len(paths) for tier, paths in failed_by_tier.items()},
        'total_failed': total_failed
    }


if __name__ == '__main__':
    # Test path ID generation
    print("Testing path ID generation:")

    node_types = ["SmallMolecule", "Disease", "Gene"]
    predicates = ["treats", "affects"]
    directions = ["F", "F"]

    path_id = generate_path_id(node_types, predicates, directions)
    print(f"Generated: {path_id}")

    parsed = parse_path_id(path_id)
    print(f"Parsed: {parsed}")

    assert parsed == (node_types, predicates, directions), "Round-trip failed"
    print("✓ Round-trip test passed")

    # Test 3-hop path
    node_types_3 = ["SmallMolecule", "Disease", "Gene", "Protein"]
    predicates_3 = ["treats", "affects", "regulates"]
    directions_3 = ["F", "F", "R"]

    path_id_3 = generate_path_id(node_types_3, predicates_3, directions_3)
    print(f"\n3-hop path: {path_id_3}")

    parsed_3 = parse_path_id(path_id_3)
    assert parsed_3 == (node_types_3, predicates_3, directions_3), "3-hop round-trip failed"
    print("✓ 3-hop round-trip test passed")
