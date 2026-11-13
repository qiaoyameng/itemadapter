from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pytest

from itemadapter import (
    ExportError,
    ExportPath,
    FieldSegment,
    IndexSegment,
    ItemAdapter,
    RuleExportStrategy,
    export,
)
from itemadapter.adapter import AdapterInterface
from itemadapter.export import FieldContext


class BoomError(RuntimeError):
    pass


@dataclass
class GuardedItem:
    name: str
    # A normal dataclass field; the test adapter below raises on __getitem__
    # for it, so any read of this value is observable as a failure.
    guarded: str = field(default="do-not-read")


@dataclass
class Address:
    city: str
    internal_code: str = field(metadata={"export": "internal"})


@dataclass
class User:
    name: str
    password: str = field(metadata={"export": "secret"})
    ssn: str = field(metadata={"export": "secret"})
    address: Address | None = None
    tags: list[str] = field(default_factory=list)


class CustomItem:
    """A non-standard item handled through a custom adapter."""

    def __init__(self, label: str, hidden: str) -> None:
        self._values = {"label": label, "hidden": hidden}


class CustomAdapter(AdapterInterface):
    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return issubclass(item_class, CustomItem)

    def get_field_meta(self, field_name: str) -> MappingProxyType[str, Any]:
        metas = {
            "label": MappingProxyType({"export": "public"}),
            "hidden": MappingProxyType({"export": "internal"}),
        }
        return metas[field_name]

    def field_names(self):
        from collections.abc import KeysView

        return KeysView(self.item._values)

    def __getitem__(self, field_name: str) -> Any:
        return self.item._values[field_name]

    def __setitem__(self, field_name: str, value: Any) -> None:
        self.item._values[field_name] = value

    def __delitem__(self, field_name: str) -> None:
        del self.item._values[field_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.item._values)

    def __len__(self) -> int:
        return len(self.item._values)


class ExplodingAdapter(AdapterInterface):
    """Adapter whose guarded getter raises the moment it is evaluated."""

    getter_calls = 0

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return issubclass(item_class, GuardedItem)

    def get_field_meta(self, field_name: str) -> MappingProxyType[str, Any]:
        return MappingProxyType({})

    def field_names(self):
        from collections.abc import KeysView

        return KeysView({"name", "guarded"})

    def __getitem__(self, field_name: str) -> Any:
        if field_name == "guarded":
            ExplodingAdapter.getter_calls += 1
            raise BoomError("the guarded getter must never be evaluated")
        return self.item.name

    def __setitem__(self, field_name: str, value: Any) -> None:
        setattr(self.item, field_name, value)

    def __delitem__(self, field_name: str) -> None:
        delattr(self.item, field_name)

    def __iter__(self) -> Iterator[str]:
        return iter(["name", "guarded"])

    def __len__(self) -> int:
        return 2


@pytest.fixture
def custom_adapter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        ItemAdapter,
        "ADAPTER_CLASSES",
        (CustomAdapter, *ItemAdapter.ADAPTER_CLASSES),
    )
    _clear_caches()
    yield
    _clear_caches()


def _clear_caches() -> None:
    from itemadapter.adapter import _find_adapter_class, _has_item_overrides

    _find_adapter_class.cache_clear()
    _has_item_overrides.cache_clear()


@pytest.fixture
def exploding_adapter(monkeypatch: pytest.MonkeyPatch):
    ExplodingAdapter.getter_calls = 0
    monkeypatch.setattr(
        ItemAdapter,
        "ADAPTER_CLASSES",
        (ExplodingAdapter, *ItemAdapter.ADAPTER_CLASSES),
    )
    _clear_caches()
    yield
    _clear_caches()


def build_user() -> User:
    return User(
        name="alice",
        password="hunter2",
        ssn="123-45-6789",
        address=Address(city="Montevideo", internal_code="IC-9"),
        tags=["admin", "user"],
    )


class TestExportPath:
    def test_path_rendering_and_tuple(self):
        path = ExportPath(
            (FieldSegment("user"), IndexSegment(0), FieldSegment("name"))
        )
        assert path.as_tuple() == ("user", 0, "name")
        assert path.render() == "user[0].name"
        assert str(path) == "user[0].name"

    def test_root_path(self):
        assert not ExportPath()
        assert str(ExportPath()) == "<root>"

    def test_child_extends_path(self):
        child = ExportPath().child(FieldSegment("a")).child(IndexSegment(2))
        assert child.as_tuple() == ("a", 2)


class TestRuleExportStrategy:
    def test_metadata_exclude_and_mask(self):
        strategy = RuleExportStrategy(
            exclude_metadata={"internal"},
            mask_metadata={"secret"},
        )
        result = export(build_user(), strategy)
        assert result == {
            "name": "alice",
            "password": "<redacted>",
            "ssn": "<redacted>",
            "address": {"city": "Montevideo"},
            "tags": ["admin", "user"],
        }

    def test_path_rules_inside_nested_dict_and_sequences(self):
        strategy = RuleExportStrategy(
            mask_paths=[("password",), ("tags", 1)],
            exclude_paths=[("ssn",), ("address", "internal_code")],
        )
        result = export(build_user(), strategy)
        assert result == {
            "name": "alice",
            "password": "<redacted>",
            "address": {"city": "Montevideo"},
            "tags": ["admin", "<redacted>"],
        }

    def test_exclude_beats_mask_and_include(self):
        strategy = RuleExportStrategy(
            include_paths=[("password",)],
            mask_paths=[("password",)],
            exclude_paths=[("password",)],
        )
        result = export(build_user(), strategy)
        assert "password" not in result

    def test_mask_beats_include(self):
        strategy = RuleExportStrategy(
            include_paths=[("password",)],
            mask_paths=[("password",)],
        )
        assert export(build_user(), strategy)["password"] == "<redacted>"

    def test_custom_redact_value(self):
        strategy = RuleExportStrategy(
            mask_metadata={"secret"}, redact_value=None
        )
        result = export(build_user(), strategy)
        assert result["password"] is None
        assert result["ssn"] is None


class TestGetterEvaluation:
    def test_excluded_field_getter_never_runs(self, exploding_adapter):
        # The strategy excludes a field before its value is read; the
        # exploding property getter must therefore never be touched.
        strategy = RuleExportStrategy(exclude_paths=[("guarded",)])
        item = GuardedItem(name="ok")
        result = ItemAdapter(item).export(strategy)
        assert result == {"name": "ok"}
        assert ExplodingAdapter.getter_calls == 0

    def test_masked_field_getter_never_runs(self, exploding_adapter):
        strategy = RuleExportStrategy(mask_paths=[("guarded",)])
        result = ItemAdapter(GuardedItem(name="ok")).export(strategy)
        assert result == {"name": "ok", "guarded": "<redacted>"}
        assert ExplodingAdapter.getter_calls == 0

    def test_kept_exploding_field_still_raises(self, exploding_adapter):
        with pytest.raises(BoomError):
            ItemAdapter(GuardedItem(name="ok")).export(RuleExportStrategy())
        assert ExplodingAdapter.getter_calls == 1


class TestRedactCallback:
    def test_callback_receives_only_approved_fields(self):
        seen_contexts: list[FieldContext] = []

        class CountingStrategy(RuleExportStrategy):
            def redact(self, context: FieldContext) -> Any:
                seen_contexts.append(context)
                return f"MASKED:{context.field_name}"

        strategy = CountingStrategy(
            mask_metadata={"secret"},
            exclude_metadata={"internal"},
        )
        result = export(build_user(), strategy)
        assert result["password"] == "MASKED:password"
        assert result["ssn"] == "MASKED:ssn"
        assert {context.field_name for context in seen_contexts} == {
            "password",
            "ssn",
        }
        # The callback inspects metadata and path, never the raw value.
        password_context = next(
            context for context in seen_contexts if context.field_name == "password"
        )
        assert password_context.path.as_tuple() == ("password",)
        assert password_context.metadata["export"] == "secret"
        assert password_context.item_class is User


class TestContainersAndContext:
    def test_nested_items_dicts_and_sequences_share_path(self):
        nested = {
            "users": [
                User(
                    name="bob",
                    password="p",
                    ssn="s",
                    address=Address(city="Salto", internal_code="x"),
                )
            ],
            "records": ({"k": "v"}, {"k": "w"}),
            "flags": {1, 2},
        }
        strategy = RuleExportStrategy(
            exclude_paths=[
                ("users", 0, "ssn"),
                ("records", 1, "k"),
            ],
            mask_paths=[("users", 0, "password",)],
        )
        result = export(nested, strategy)
        assert result["users"][0] == {
            "name": "bob",
            "password": "<redacted>",
            "address": {"city": "Salto", "internal_code": "x"},
            "tags": [],
        }
        assert result["records"] == ({"k": "v"}, {})
        assert result["flags"] == {1, 2}

    def test_set_and_frozenset_preserve_type(self):
        strategy = RuleExportStrategy()
        result = export({"s": {1, 2}, "f": frozenset({3})}, strategy)
        assert isinstance(result["s"], set)
        assert isinstance(result["f"], frozenset)

    def test_sequence_element_exclude_and_mask(self):
        strategy = RuleExportStrategy(
            exclude_paths=[("xs", 1)],
            mask_paths=[("xs", 2)],
        )
        result = export({"xs": [10, 20, 30]}, strategy)
        assert result == {"xs": [10, "<redacted>"]}

    def test_element_redact_context_carries_path_only(self):
        contexts: list[FieldContext] = []

        class ElementStrategy(RuleExportStrategy):
            def redact(self, context: FieldContext) -> Any:
                contexts.append(context)
                return "M"

        result = export(
            {"xs": [1, 2]}, ElementStrategy(mask_paths=[("xs", 0)])
        )
        assert result == {"xs": ["M", 2]}
        assert contexts[0].path.as_tuple() == ("xs", 0)
        assert dict(contexts[0].metadata) == {}

    def test_custom_adapter_item(self, custom_adapter):
        item = CustomItem(label="thing", hidden="secret-value")
        strategy = RuleExportStrategy(exclude_metadata={"internal"})
        result = ItemAdapter(item).export(strategy)
        assert result == {"label": "thing"}


class TestErrors:
    def test_unsupported_type_reports_path_without_value(self):
        class Weird:
            def __repr__(self):
                return "WEIRD-SENSITIVE-REPR"

        strategy = RuleExportStrategy()
        with pytest.raises(ExportError) as exc_info:
            export({"outer": [Weird()]}, strategy)
        message = str(exc_info.value)
        assert "Weird" in message
        assert "outer[0]" in message
        assert "WEIRD-SENSITIVE-REPR" not in message

    def test_unsupported_dict_key_reports_path(self):
        with pytest.raises(ExportError) as exc_info:
            export({("nested",): "x"}, RuleExportStrategy())
        assert "tuple" in str(exc_info.value)

    def test_cycle_in_item_reports_path(self):
        @dataclass
        class Node:
            name: str
            peer: Any = None

        node_a = Node("a")
        node_b = Node("b", node_a)
        node_a.peer = node_b
        with pytest.raises(ExportError) as exc_info:
            export(node_a, RuleExportStrategy())
        assert "Circular reference" in str(exc_info.value)
        assert exc_info.value.path.as_tuple() == ("peer", "peer")

    def test_cycle_in_container_reports_path(self):
        container: dict[str, Any] = {"a": []}
        container["a"].append(container)
        with pytest.raises(ExportError) as exc_info:
            export(container, RuleExportStrategy())
        assert exc_info.value.path.as_tuple() == ("a", 0)


class TestOriginalObjectAndIsolation:
    def test_original_object_is_unchanged(self):
        user = build_user()
        before = ItemAdapter(user).asdict()
        strategy = RuleExportStrategy(
            exclude_metadata={"internal"},
            mask_metadata={"secret"},
        )
        export(user, strategy)
        assert ItemAdapter(user).asdict() == before
        assert user.password == "hunter2"
        assert user.address.internal_code == "IC-9"

    def test_subsequent_plain_asdict_ignores_strategy(self):
        user = build_user()
        adapter = ItemAdapter(user)
        strategy = RuleExportStrategy(
            exclude_paths=[("ssn",)], mask_metadata={"secret"}
        )
        filtered = adapter.export(strategy)
        assert "ssn" not in filtered
        # A plain asdict() on the same adapter must not carry over any rule.
        assert adapter.asdict() == {
            "name": "alice",
            "password": "hunter2",
            "ssn": "123-45-6789",
            "address": {
                "city": "Montevideo",
                "internal_code": "IC-9",
            },
            "tags": ["admin", "user"],
        }

    def test_export_does_not_mutate_shared_nested_containers(self):
        shared_list = [1, 2]
        payload = {"items": shared_list}
        export(payload, RuleExportStrategy(mask_paths=[("items", 0)]))
        assert shared_list == [1, 2]
