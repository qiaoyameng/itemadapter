from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from unittest import mock

import pytest

from itemadapter import DELETE, ItemAdapter, evolve
from itemadapter._evolve import (
    EvolveKeyError,
    EvolvePathError,
    EvolveTypeError,
)

try:
    import attr
except ImportError:
    attr = None


@dataclass(frozen=True)
class FrozenInner:
    x: int = 0
    y: int = 0


@dataclass
class MutableInner:
    x: int = 0
    y: int = 0


@dataclass(frozen=True)
class FrozenRoot:
    name: str = "root"
    inner: FrozenInner = field(default_factory=FrozenInner)
    items: tuple = ()
    meta: dict = field(default_factory=dict)


@dataclass
class MutableRoot:
    inner: MutableInner = field(default_factory=MutableInner)


def build_root() -> FrozenRoot:
    return FrozenRoot(
        name="n",
        inner=FrozenInner(x=1, y=2),
        items=[FrozenInner(x=9), FrozenInner(x=8)],
        meta={"k": {"v": 1}},
    )


class BasicEvolveTest(unittest.TestCase):
    def test_dict_root(self):
        item = {"a": 1, "b": {"c": 2}}
        result = evolve(item, {("b", "c"): 3, ("a",): 4})
        assert result == {"a": 4, "b": {"c": 3}}
        # original untouched
        assert item == {"a": 1, "b": {"c": 2}}

    def test_frozen_dataclass_root(self):
        item = build_root()
        result = evolve(
            item,
            {
                ("inner", "x"): 100,
                ("items", 0, "y"): 55,
                ("meta", "k", "v"): 42,
                ("name",): None,
            },
        )
        assert result.name is None
        assert result.inner == FrozenInner(x=100, y=2)
        assert result.items[0] == FrozenInner(x=9, y=55)
        assert result.items[1] == FrozenInner(x=8)
        assert result.meta == {"k": {"v": 42}}

    def test_string_paths(self):
        item = build_root()
        result = evolve(item, {"inner.x": 100, "items[0].y": 55, 'meta["k"]["v"]': 42})
        assert result.inner.x == 100
        assert result.items[0].y == 55
        assert result.meta["k"]["v"] == 42

    def test_mixed_types_in_one_path(self):
        @attr.s(frozen=True)
        class Tag:
            label = attr.ib()

        @dataclass(frozen=True)
        class Document:
            meta: dict

        document = Document(meta={"tags": [Tag(label="x")]})
        result = evolve(document, {"meta.tags[0].label": "y"})
        assert result.meta["tags"][0].label == "y"
        assert document.meta["tags"][0].label == "x"

    def test_adapter_method(self):
        item = {"a": 1}
        assert ItemAdapter(item).evolve({("a",): 2}) == {"a": 2}


class IdentityTest(unittest.TestCase):
    def test_mutable_dataclass_is_not_mutated(self):
        item = MutableRoot(inner=MutableInner(x=1, y=2))
        result = evolve(item, {("inner", "x"): 3})
        assert result is not item
        assert result.inner is not item.inner
        assert item.inner.x == 1
        assert result.inner.y == 2

    def test_untouched_branches_keep_identity(self):
        item = build_root()
        result = evolve(item, {("inner", "x"): 3})
        assert result.inner is not item.inner
        assert result.items is item.items
        assert result.items[0] is item.items[0]
        assert result.items[1] is item.items[1]
        assert result.meta is item.meta

    def test_no_effective_change_keeps_root_identity(self):
        item = build_root()
        assert evolve(item, {("inner", "x"): 1}) is item
        assert evolve(item, {("items", 0, "x"): 9}) is item

    def test_public_ancestor_rebuilt_once(self):
        shared = FrozenInner(x=1)

        @dataclass(frozen=True)
        class Holder:
            a: FrozenInner
            b: FrozenInner

        holder = Holder(a=shared, b=shared)
        result = evolve(holder, {("a", "x"): 5, ("b", "x"): 5})
        assert result.a is result.b

    def test_empty_updates(self):
        item = build_root()
        assert evolve(item, {}) is item


class MissingNoneDeleteTest(unittest.TestCase):
    def test_explicit_none_is_a_value(self):
        item = {"a": 1}
        assert evolve(item, {("a",): None})["a"] is None
        dc = evolve(FrozenInner(), {("x",): None})
        assert dc.x is None

    def test_setting_new_mapping_key_is_allowed(self):
        item = {"a": 1}
        result = evolve(item, {("b",): None})
        assert result == {"a": 1, "b": None}
        assert "b" not in item

    def test_traversing_missing_key_fails(self):
        item = {"a": 1}
        with pytest.raises(EvolveKeyError):
            evolve(item, {("b", "c"): 1})

    def test_deleting_missing_key_fails(self):
        item = {"a": 1}
        with pytest.raises(EvolveKeyError):
            evolve(item, {("b",): DELETE})

    def test_unknown_dataclass_field_fails(self):
        with pytest.raises(EvolveKeyError):
            evolve(FrozenInner(), {("unknown",): 1})

    def test_delete_dict_key(self):
        item = {"a": 1, "b": 2}
        assert evolve(item, {("a",): DELETE}) == {"b": 2}

    def test_delete_list_index(self):
        item = {"a": [0, 1, 2]}
        assert evolve(item, {("a", 1): DELETE}) == {"a": [0, 2]}

    def test_delete_dataclass_field_uses_default(self):
        @dataclass(frozen=True)
        class Item:
            a: int = 10

        assert evolve(Item(a=5), {("a",): DELETE}).a == 10

    def test_delete_field_without_default_fails(self):
        @dataclass(frozen=True)
        class Item:
            a: int

        with pytest.raises(TypeError):
            evolve(Item(a=1), {("a",): DELETE})

    def test_delete_invokes_default_factory(self):
        calls = []

        @dataclass(frozen=True)
        class Item:
            values: list = field(default_factory=lambda: calls.append("called") or [])

        evolve(Item(values=[1]), {("values",): DELETE})
        assert calls == ["called"]

    def test_list_index_out_of_range(self):
        item = {"a": [1]}
        with pytest.raises(EvolveKeyError):
            evolve(item, {("a", 5): 1})
        with pytest.raises(EvolveKeyError):
            evolve(item, {("a", 1): DELETE})

    def test_list_set_can_append(self):
        item = {"a": [1]}
        assert evolve(item, {("a", 1): 2}) == {"a": [1, 2]}


@unittest.skipIf(attr is None, "attrs module is not available")
class AttrsTest(unittest.TestCase):
    def test_frozen_attrs(self):
        @attr.s(frozen=True)
        class Item:
            name = attr.ib(default="x")
            inner = attr.ib(factory=lambda: FrozenInner(x=3))

        result = evolve(Item(), {("inner", "x"): 77, ("name",): None})
        assert result.inner.x == 77
        assert result.name is None

    def test_attrs_delete(self):
        @attr.s(frozen=True)
        class Item:
            value = attr.ib()
            name = attr.ib(default="n")

        assert evolve(Item(value=1, name="x"), {("name",): DELETE}).name == "n"

    def test_private_attribute_uses_init_alias(self):
        @attr.s(frozen=True)
        class Item:
            _name = attr.ib()

        result = evolve(Item(name="a"), {("name",): "b"})
        assert result._name == "b"

    def test_attrs_validator_runs(self):
        @attr.s(frozen=True)
        class Item:
            n = attr.ib(validator=attr.validators.instance_of(int))

        with pytest.raises((TypeError, ValueError)):
            evolve(Item(n=1), {("n",): "bad"})

    def test_attrs_converters_run(self):
        @attr.s(frozen=True)
        class Item:
            n = attr.ib(converter=int)

        assert evolve(Item(n="1"), {("n",): "5"}).n == 5


class InitFalseTest(unittest.TestCase):
    def test_dataclass_init_failed_field_cannot_be_set(self):
        @dataclass(frozen=True)
        class Item:
            x: int
            computed: int = field(init=False, default=0)

        @dataclass(frozen=True)
        class Holder:
            item: Item

        holder = Holder(item=Item(x=1))
        with pytest.raises(EvolveTypeError):
            evolve(holder, {("item", "computed"): 5})
        with pytest.raises(EvolveTypeError):
            evolve(holder, {("item", "computed"): DELETE})
        with pytest.raises(EvolveTypeError):
            evolve(holder, {("item", "computed", "x"): 1})

    @unittest.skipIf(attr is None, "attrs module is not available")
    def test_attrs_init_false_field_cannot_be_set(self):
        @attr.s(frozen=True)
        class Item:
            x = attr.ib()
            computed = attr.ib(init=False, default=0)

        with pytest.raises(EvolveTypeError):
            evolve(Item(x=1), {("computed",): 5})


class ConflictTest(unittest.TestCase):
    def test_parent_then_child(self):
        with pytest.raises(EvolvePathError):
            evolve({"a": {"b": 1}}, {("a",): 2, ("a", "b"): 3})

    def test_child_then_parent(self):
        with pytest.raises(EvolvePathError):
            evolve({"a": {"b": 1}}, {("a", "b"): 3, ("a",): 2})

    def test_root_vs_nested(self):
        with pytest.raises(EvolvePathError):
            evolve({"a": 1}, {(): "x", ("a",): 2})
        with pytest.raises(EvolvePathError):
            evolve({"a": 1}, {("a",): 2, (): "x"})

    def test_root_replacement(self):
        assert evolve({"a": 1}, {(): "new"}) == "new"

    def test_conflict_does_not_depend_on_dict_order(self):
        from itemadapter._evolve import _build_trie

        class _Ordered:
            def __init__(self, pairs):
                self._pairs = pairs

            def items(self):
                return iter(self._pairs)

        for pairs in [
            [(("a", "b"), 1), (("a",), 2)],
            [(("a",), 2), (("a", "b"), 1)],
        ]:
            with pytest.raises(EvolvePathError):
                _build_trie(_Ordered(pairs))


class PathValidationTest(unittest.TestCase):
    def test_invalid_paths(self):
        for path in [
            "a..b",
            "a.",
            ".a",
            "a[",
            "a[]",
            "a[0]b",
            "a[b]",
            'a["x"]extra',
            42,
            ("a", 1.0),
            ("a", True),
        ]:
            with pytest.raises(EvolvePathError):
                evolve({"a": 1}, {path: 1})

    def test_updates_must_be_mapping(self):
        with pytest.raises(EvolveTypeError):
            evolve({}, [("a", 1)])


class AtomicityTest(unittest.TestCase):
    def test_constructor_failure_leaves_original_untouched(self):
        @dataclass(frozen=True)
        class Validated:
            n: int

            def __post_init__(self):
                if self.n < 0:
                    raise ValueError("negative")

        nested_list = [Validated(n=1)]

        @dataclass(frozen=True)
        class Root:
            values: list

        root = Root(values=nested_list)
        with pytest.raises(ValueError):
            evolve(root, {("values", 0, "n"): -1})
        assert root.values is nested_list
        assert nested_list[0].n == 1

    def test_location_does_not_invoke_constructors(self):
        constructed = []

        @dataclass(frozen=True)
        class Noisy:
            n: int = 0

            def __post_init__(self):
                constructed.append(self.n)

        @dataclass(frozen=True)
        class Root:
            child: Noisy

        baseline = len(constructed)
        root = Root(child=Noisy(n=1))
        after_root = len(constructed)
        with pytest.raises(EvolveKeyError):
            evolve(root, {("child", "missing"): 1})
        # only the original construction in the test setup happened
        assert len(constructed) == after_root
        assert len(constructed) >= baseline

    def test_failed_validation_in_sibling_leaves_all_untouched(self):
        @attr.s(frozen=True)
        class Item:
            n = attr.ib(validator=attr.validators.instance_of(int))
            child = attr.ib(factory=lambda: FrozenInner(x=1))

        item = Item(n=1)
        original_child = item.child
        with pytest.raises((TypeError, ValueError)):
            evolve(item, {("n",): "bad", ("child", "x"): 5})
        assert item.n == 1
        assert item.child is original_child

    def test_failing_branch_leaves_sibling_mutable_untouched(self):
        @dataclass(frozen=True)
        class Guard:
            v: int

            def __post_init__(self):
                if self.v < 0:
                    raise ValueError("negative")

        @dataclass(frozen=True)
        class Box:
            good: list
            bad: Guard

        box = Box(good=[1, 2], bad=Guard(v=1))
        with pytest.raises(ValueError):
            evolve(box, {("good", 0): 99, ("bad", "v"): -1})
        assert box.good == [1, 2]

    def test_multiple_fields_rebuilt_together(self):
        @dataclass(frozen=True)
        class Item:
            a: int
            b: int

        result = evolve(Item(a=1, b=2), {("a",): 10})
        assert result.a == 10
        assert result.b == 2

    def test_constructor_validation_is_not_bypassed(self):
        @dataclass(frozen=True)
        class Range:
            lo: int
            hi: int

            def __post_init__(self):
                if self.lo > self.hi:
                    raise ValueError("lo > hi")

        @dataclass(frozen=True)
        class Wrap:
            r: Range

        wrapped = Wrap(r=Range(lo=1, hi=2))
        with pytest.raises(ValueError):
            evolve(wrapped, {("r", "lo"): 9})
        assert wrapped.r.lo == 1


class UnsupportedTypesTest(unittest.TestCase):
    def test_unsupported_root(self):
        with pytest.raises(EvolveTypeError):
            evolve((1, 2), {(0,): 1})

    def test_unsupported_intermediate(self):
        with pytest.raises(EvolveTypeError):
            evolve({"t": (1, 2)}, {("t", 0): 1})

    def test_plain_object_intermediate(self):
        class Plain:
            pass

        with pytest.raises(EvolveTypeError):
            evolve({"o": Plain()}, {("o", "x"): 1})


class OptionalAttrsImportTest(unittest.TestCase):
    @mock.patch("itemadapter._evolve.attr", None)
    def test_evolve_works_without_attrs_for_dict_and_dataclass(self):
        @dataclass(frozen=True)
        class Item:
            a: int = 1

        assert evolve(Item(a=1), {("a",): 2}).a == 2
        assert evolve({"a": 1}, {("a",): 2}) == {"a": 2}

    def test_package_imports_without_attrs(self):
        with mock.patch("builtins.__import__", _block_import("attr")):
            import importlib
            import sys

            backup = {
                key: sys.modules.pop(key)
                for key in list(sys.modules)
                if key.startswith("itemadapter")
            }
            try:
                itemadapter = importlib.import_module("itemadapter")
                assert hasattr(itemadapter, "evolve")
            finally:
                sys.modules.update(backup)


def _block_import(name):
    import importlib

    def fake_import(import_name, *args, **kwargs):
        if import_name.split(".", 1)[0] == name:
            raise ImportError(import_name)
        return importlib.__import__(import_name, *args, **kwargs)

    return fake_import
