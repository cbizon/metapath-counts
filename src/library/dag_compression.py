from __future__ import annotations

from functools import lru_cache
from typing import Iterable, FrozenSet

from .aggregation import parse_compound_predicate
from .hierarchy import get_predicate_parents, get_type_parents


def normalize_type(name: str) -> str:
    if not name:
        return name
    name = name.replace("biolink:", "")
    # Biolink types are CamelCase. Preserve already-normalized labels and only
    # normalize when separators are present.
    if "_" not in name and " " not in name:
        return name
    name = name.replace("_", " ")
    return "".join(word.capitalize() for word in name.split())


def normalize_predicate(name: str) -> str:
    if not name:
        return name
    return name.replace("biolink:", "").replace(" ", "_")


def normalize_excluded_types(values: Iterable[str]) -> FrozenSet[str]:
    return frozenset(normalize_type(v.strip()) for v in values if v and v.strip())


def normalize_excluded_predicates(values: Iterable[str]) -> FrozenSet[str]:
    return frozenset(normalize_predicate(v.strip()) for v in values if v and v.strip())


def predicate_is_excluded(predicate: str, excluded_predicates: FrozenSet[str]) -> bool:
    if not excluded_predicates:
        return False
    pred_norm = normalize_predicate(predicate)
    if pred_norm in excluded_predicates:
        return True
    base_pred = parse_compound_predicate(pred_norm)[0]
    return base_pred in excluded_predicates


def metapath_has_excluded(
    metapath: str,
    excluded_types: FrozenSet[str],
    excluded_predicates: FrozenSet[str],
) -> bool:
    if not excluded_types and not excluded_predicates:
        return False
    parts = metapath.split("|")
    for i, part in enumerate(parts):
        if i % 3 == 0:
            if normalize_type(part) in excluded_types:
                return True
        elif i % 3 == 1:
            if predicate_is_excluded(part, excluded_predicates):
                return True
    return False


@lru_cache(maxsize=None)
def get_allowed_type_parents(type_name: str, excluded_types: FrozenSet[str]) -> tuple[str, ...]:
    seen = set()
    out = set()
    stack = list(get_type_parents(type_name))
    while stack:
        parent = stack.pop()
        if parent in seen:
            continue
        seen.add(parent)
        if normalize_type(parent) in excluded_types:
            stack.extend(get_type_parents(parent))
            continue
        out.add(parent)
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def get_allowed_predicate_parents(
    predicate: str,
    excluded_predicates: FrozenSet[str],
) -> tuple[str, ...]:
    pred_norm = normalize_predicate(predicate)
    seen = set()
    out = set()
    stack = list(get_predicate_parents(pred_norm))
    while stack:
        parent = stack.pop()
        parent_norm = normalize_predicate(parent)
        if parent_norm in seen:
            continue
        seen.add(parent_norm)
        if predicate_is_excluded(parent_norm, excluded_predicates):
            stack.extend(get_predicate_parents(parent_norm))
            continue
        out.add(parent_norm)
    return tuple(sorted(out))
