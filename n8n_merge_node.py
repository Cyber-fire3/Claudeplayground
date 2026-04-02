"""
n8n Merge Node — Python implementation

Replicates all five merge modes from n8n's Merge node:
  - append      : concatenate all inputs in order
  - by_index    : zip items by position, merging fields side-by-side
  - by_fields   : join on matching field values (inner / left / right / outer)
  - keep_matches    : keep only items whose key value appears in both inputs (inner)
  - keep_non_matches: keep items whose key value does NOT appear in the other input
  - multiplex   : cartesian product of two inputs

Each item is a plain dict (mirrors n8n's JSON item structure).
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Literal


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Item = dict  # {"field": value, ...}
JoinType = Literal["inner", "left", "right", "outer"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _index_by_field(items: list[Item], field: str) -> dict[str, list[Item]]:
    """Group items by the value of a given field."""
    index: dict[str, list[Item]] = {}
    for item in items:
        key = str(item.get(field, ""))
        index.setdefault(key, []).append(item)
    return index


def _merge_items(a: Item, b: Item) -> Item:
    """Shallow-merge two items; b's values take precedence on key conflicts."""
    merged = deepcopy(a)
    merged.update(deepcopy(b))
    return merged


# ---------------------------------------------------------------------------
# Merge modes
# ---------------------------------------------------------------------------

def append(*inputs: list[Item]) -> list[Item]:
    """
    Append mode — concatenate all input lists in order.

    Equivalent to n8n Merge → Mode: Append.
    """
    result: list[Item] = []
    for stream in inputs:
        result.extend(deepcopy(stream))
    return result


def by_index(*inputs: list[Item], include_unmatched: bool = False) -> list[Item]:
    """
    By-index mode — pair items at the same position across all inputs.

    Parameters
    ----------
    inputs:
        Two or more item lists to zip together.
    include_unmatched:
        When True, shorter lists are padded with empty dicts so no item is
        dropped (mirrors n8n's "Add remaining" option).

    Equivalent to n8n Merge → Mode: Merge By Index.
    """
    if not inputs:
        return []

    if include_unmatched:
        max_len = max(len(s) for s in inputs)
        padded = [list(s) + [{}] * (max_len - len(s)) for s in inputs]
    else:
        padded = list(inputs)

    result: list[Item] = []
    for row in zip(*padded):
        merged: Item = {}
        for item in row:
            merged = _merge_items(merged, item)
        result.append(merged)
    return result


def by_fields(
    input1: list[Item],
    input2: list[Item],
    *,
    fields1: list[str],
    fields2: list[str] | None = None,
    join: JoinType = "inner",
) -> list[Item]:
    """
    By-fields mode — SQL-style join on one or more field pairs.

    Parameters
    ----------
    input1, input2:
        The two item lists to join.
    fields1:
        Field names from input1 to join on.
    fields2:
        Corresponding field names from input2 (defaults to fields1).
    join:
        "inner"  — only items that match in both inputs  (default)
        "left"   — all of input1; unmatched input2 fields are absent
        "right"  — all of input2; unmatched input1 fields are absent
        "outer"  — all items from both, merging where possible

    Equivalent to n8n Merge → Mode: Merge By Fields.
    """
    if fields2 is None:
        fields2 = fields1

    if len(fields1) != len(fields2):
        raise ValueError("fields1 and fields2 must have the same length")

    def make_key(item: Item, fields: list[str]) -> tuple:
        return tuple(str(item.get(f, "")) for f in fields)

    index2: dict[tuple, list[Item]] = {}
    for item in input2:
        k = make_key(item, fields2)
        index2.setdefault(k, []).append(item)

    result: list[Item] = []
    matched_keys: set[tuple] = set()

    for item1 in input1:
        k = make_key(item1, fields1)
        matches = index2.get(k, [])
        if matches:
            matched_keys.add(k)
            for item2 in matches:
                result.append(_merge_items(item1, item2))
        elif join in ("left", "outer"):
            result.append(deepcopy(item1))

    if join in ("right", "outer"):
        for item2 in input2:
            k = make_key(item2, fields2)
            if k not in matched_keys:
                result.append(deepcopy(item2))

    return result


def keep_matches(
    input1: list[Item],
    input2: list[Item],
    *,
    field1: str,
    field2: str | None = None,
) -> list[Item]:
    """
    Keep-matches mode — return input1 items whose key value also exists in input2.

    Equivalent to n8n Merge → Mode: Keep Key Matches.
    """
    if field2 is None:
        field2 = field1

    keys_in_2 = {str(item.get(field2, "")) for item in input2}
    return [deepcopy(i) for i in input1 if str(i.get(field1, "")) in keys_in_2]


def keep_non_matches(
    input1: list[Item],
    input2: list[Item],
    *,
    field1: str,
    field2: str | None = None,
) -> list[Item]:
    """
    Keep-non-matches mode — return input1 items whose key value is absent from input2.

    Equivalent to n8n Merge → Mode: Keep Non Matches.
    """
    if field2 is None:
        field2 = field1

    keys_in_2 = {str(item.get(field2, "")) for item in input2}
    return [deepcopy(i) for i in input1 if str(i.get(field1, "")) not in keys_in_2]


def multiplex(input1: list[Item], input2: list[Item]) -> list[Item]:
    """
    Multiplex mode — every item in input1 paired with every item in input2.

    Equivalent to n8n Merge → Mode: Multiplex.
    """
    return [_merge_items(a, b) for a, b in product(input1, input2)]


# ---------------------------------------------------------------------------
# Convenience wrapper — mirrors a node-like interface
# ---------------------------------------------------------------------------

class MergeNode:
    """
    High-level wrapper that mirrors how you'd configure an n8n Merge node.

    Usage
    -----
    node = MergeNode(mode="by_fields", fields1=["id"], join="left")
    output = node.run(input1, input2)
    """

    MODES = {
        "append", "by_index", "by_fields",
        "keep_matches", "keep_non_matches", "multiplex",
    }

    def __init__(
        self,
        mode: str = "append",
        *,
        # by_index options
        include_unmatched: bool = False,
        # by_fields / keep_matches / keep_non_matches options
        fields1: list[str] | None = None,
        fields2: list[str] | None = None,
        field1: str | None = None,
        field2: str | None = None,
        join: JoinType = "inner",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {self.MODES}")
        self.mode = mode
        self.include_unmatched = include_unmatched
        self.fields1 = fields1
        self.fields2 = fields2
        self.field1 = field1
        self.field2 = field2
        self.join = join

    def run(self, *inputs: list[Item]) -> list[Item]:
        if self.mode == "append":
            return append(*inputs)

        if self.mode == "by_index":
            return by_index(*inputs, include_unmatched=self.include_unmatched)

        # The remaining modes operate on exactly two inputs
        if len(inputs) != 2:
            raise ValueError(f"Mode '{self.mode}' requires exactly 2 inputs, got {len(inputs)}")
        i1, i2 = inputs

        if self.mode == "by_fields":
            if not self.fields1:
                raise ValueError("'by_fields' mode requires fields1")
            return by_fields(i1, i2, fields1=self.fields1, fields2=self.fields2, join=self.join)

        if self.mode == "keep_matches":
            if not self.field1:
                raise ValueError("'keep_matches' mode requires field1")
            return keep_matches(i1, i2, field1=self.field1, field2=self.field2)

        if self.mode == "keep_non_matches":
            if not self.field1:
                raise ValueError("'keep_non_matches' mode requires field1")
            return keep_non_matches(i1, i2, field1=self.field1, field2=self.field2)

        if self.mode == "multiplex":
            return multiplex(i1, i2)

        raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    users = [
        {"id": "1", "name": "Alice"},
        {"id": "2", "name": "Bob"},
        {"id": "3", "name": "Carol"},
    ]
    orders = [
        {"userId": "1", "item": "Book"},
        {"userId": "2", "item": "Pen"},
        {"userId": "4", "item": "Desk"},
    ]
    tags = [
        {"label": "vip"},
        {"label": "new"},
    ]

    def show(title: str, data: list[Item]) -> None:
        print(f"\n{'─' * 50}")
        print(f"  {title}")
        print('─' * 50)
        print(json.dumps(data, indent=2))

    show("append (users + orders)", append(users, orders))

    show("by_index (users + orders, inner)", by_index(users, orders))

    show(
        "by_fields inner join  id ↔ userId",
        by_fields(users, orders, fields1=["id"], fields2=["userId"], join="inner"),
    )

    show(
        "by_fields left join  id ↔ userId",
        by_fields(users, orders, fields1=["id"], fields2=["userId"], join="left"),
    )

    show(
        "keep_matches  (users whose id is in orders.userId)",
        keep_matches(users, orders, field1="id", field2="userId"),
    )

    show(
        "keep_non_matches  (users with NO order)",
        keep_non_matches(users, orders, field1="id", field2="userId"),
    )

    show("multiplex  users × tags", multiplex(users, tags))

    # Node wrapper example
    node = MergeNode(mode="by_fields", fields1=["id"], fields2=["userId"], join="outer")
    show("MergeNode outer join", node.run(users, orders))
