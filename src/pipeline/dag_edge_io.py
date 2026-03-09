from __future__ import annotations

import gzip
import os
import struct
import shutil
from typing import Dict, Iterable, Iterator, Tuple


EDGE_BIN_MAGIC = b"DAGEDG1\n"
EDGE_PAIR_STRUCT = struct.Struct("<II")


def edges_bin_path_for_dir(dag_dir: str) -> str:
    return os.path.join(dag_dir, "edges.bin")


def node_ids_path_for_dir(dag_dir: str) -> str:
    return os.path.join(dag_dir, "node_ids.tsv")


def resolve_node_ids_path(dag_dir: str) -> str:
    plain = os.path.join(dag_dir, "node_ids.tsv")
    gz = plain + ".gz"
    if os.path.exists(plain):
        return plain
    if os.path.exists(gz):
        return gz
    return plain


def default_node_ids_path_for_edges(edges_path: str) -> str:
    return resolve_node_ids_path(os.path.dirname(edges_path))


def _open_text_read(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _open_text_write(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def write_node_ids(node_ids_path: str, node_to_id: Dict[str, int]) -> None:
    items = sorted(node_to_id.items(), key=lambda kv: kv[1])
    with _open_text_write(node_ids_path) as f:
        f.write("node_id\tmetapath\n")
        for metapath, node_id in items:
            f.write(f"{node_id}\t{metapath}\n")


def load_node_ids(node_ids_path: str) -> tuple[Dict[str, int], list[str]]:
    node_to_id: Dict[str, int] = {}
    id_to_node: list[str] = []
    with _open_text_read(node_ids_path) as f:
        header = f.readline().strip().split("\t")
        if header[:2] != ["node_id", "metapath"]:
            raise ValueError(f"Invalid node_ids header in {node_ids_path}: {header}")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            node_id_str, metapath = line.split("\t", 1)
            node_id = int(node_id_str)
            if node_id != len(id_to_node):
                raise ValueError(
                    f"Non-contiguous node ids in {node_ids_path}: expected {len(id_to_node)}, got {node_id}"
                )
            id_to_node.append(metapath)
            node_to_id[metapath] = node_id
    return node_to_id, id_to_node


class BinaryEdgeWriter:
    def __init__(self, path: str, append: bool = False):
        self.path = path
        mode = "ab" if append else "wb"
        self._f = open(path, mode)
        if (not append) or os.path.getsize(path) == 0:
            self._f.write(EDGE_BIN_MAGIC)
        self.count = 0

    def write_ids(self, child_id: int, parent_id: int) -> None:
        self._f.write(EDGE_PAIR_STRUCT.pack(child_id, parent_id))
        self.count += 1

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> "BinaryEdgeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def iter_edge_id_pairs_from_bin(edges_bin_path: str) -> Iterator[Tuple[int, int]]:
    with open(edges_bin_path, "rb") as f:
        magic = f.read(len(EDGE_BIN_MAGIC))
        if magic != EDGE_BIN_MAGIC:
            raise ValueError(f"Invalid binary edge magic in {edges_bin_path!r}")
        while True:
            chunk = f.read(EDGE_PAIR_STRUCT.size)
            if not chunk:
                break
            if len(chunk) != EDGE_PAIR_STRUCT.size:
                raise ValueError(f"Truncated binary edge file: {edges_bin_path!r}")
            yield EDGE_PAIR_STRUCT.unpack(chunk)


def iter_edge_pairs_from_bin(edges_bin_path: str, node_ids_path: str) -> Iterator[Tuple[str, str]]:
    _node_to_id, id_to_node = load_node_ids(node_ids_path)
    for child_id, parent_id in iter_edge_id_pairs_from_bin(edges_bin_path):
        try:
            yield id_to_node[child_id], id_to_node[parent_id]
        except IndexError as e:
            raise ValueError(
                f"Edge id out of range in {edges_bin_path!r}: child={child_id} parent={parent_id}"
            ) from e


def iter_edge_pairs_auto(edges_path: str) -> Iterator[Tuple[str, str]]:
    if edges_path.endswith(".bin"):
        node_ids_path = default_node_ids_path_for_edges(edges_path)
        if not os.path.exists(node_ids_path):
            # Persistent shard edges may share a node_ids.tsv one directory up.
            parent_node_ids = resolve_node_ids_path(os.path.dirname(os.path.dirname(edges_path)))
            if os.path.exists(parent_node_ids):
                node_ids_path = parent_node_ids
        yield from iter_edge_pairs_from_bin(edges_path, node_ids_path)
        return
    with open(edges_path, "r") as f:
        header = f.readline().strip().split("\t")
        if header[:2] != ["child", "parent"]:
            raise ValueError(f"Invalid edges file header in {edges_path}: {header}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            child, parent = line.split("\t")[:2]
            yield child, parent


def copy_or_write_node_ids(node_ids_path: str, node_to_id: Dict[str, int], src_node_ids_path: str | None = None) -> None:
    if src_node_ids_path and os.path.exists(src_node_ids_path):
        shutil.copyfile(src_node_ids_path, node_ids_path)
        return
    write_node_ids(node_ids_path, node_to_id)
