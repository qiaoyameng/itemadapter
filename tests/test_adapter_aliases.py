from __future__ import annotations

from collections.abc import KeysView
from dataclasses import dataclass, field
from typing import Any

import pytest

from itemadapter import AliasError, ItemAdapter


@dataclass
class DataclassItem:
    product_id: str
    sku: str


@dataclass
class DataclassPlainItem:
    product_id: str
    quantity: int


@dataclass
class DataclassMetadataItem:
    product_id: str = field(metadata={"alias": "sku"})
    quantity: int = 1


@dataclass
class DataclassCollisionItem:
    product_id: str
    sku: str


@dataclass
class ChildItem:
    product_id: str = field(metadata={"alias": "sku"})


@dataclass
class ParentItem:
    product_id: str = field(metadata={"alias": "sku"})
    child: ChildItem = field(default_factory=lambda: ChildItem("child"))


@pytest.fixture
def attrs_classes():
    attrs = pytest.importorskip("attrs")

    @attrs.define
    class AttrsItem:
        product_id: str
        sku: str

    @attrs.define
    class AttrsMetadataItem:
        product_id: str = attrs.field(metadata={"alias": "sku"})
        quantity: int = 1

    return attrs, AttrsItem, AttrsMetadataItem


def make_items() -> list[tuple[str, Any, dict[str, str]]]:
    items: list[tuple[str, Any, dict[str, str]]] = [
        ("dict", {"product_id": "p", "sku": "s"}, {"product_id": "sku", "sku": "product_id"}),
        (
            "dataclass",
            DataclassItem("p", "s"),
            {"product_id": "sku", "sku": "product_id"},
        ),
    ]
    attrs = pytest.importorskip("attrs")

    @attrs.define
    class AttrsItem:
        product_id: str
        sku: str

    items.append(
        ("attrs", AttrsItem("p", "s"), {"product_id": "sku", "sku": "product_id"})
    )
    return items


@pytest.mark.parametrize(("kind", "item", "aliases"), make_items())
def test_swap_aliases_share_one_namespace(kind: str, item: Any, aliases: dict[str, str]) -> None:
    adapter = ItemAdapter(item, aliases=aliases)

    assert isinstance(adapter.field_names(), KeysView)
    assert list(adapter.field_names()) == ["sku", "product_id"]
    assert set(adapter) == {"sku", "product_id"}
    assert dict(adapter) == {"sku": "p", "product_id": "s"}
    assert adapter["sku"] == "p"
    assert adapter["product_id"] == "s"

    adapter["sku"] = "new-product"
    adapter["product_id"] = "new-sku"
    assert adapter.asdict() == {"sku": "new-product", "product_id": "new-sku"}

    del adapter["sku"]
    assert "sku" not in adapter
    assert adapter["product_id"] == "new-sku"


@pytest.mark.parametrize(
    ("item", "aliases"),
    [
        (DataclassPlainItem("p", 1), {"product_id": "sku"}),
        ({"product_id": "p", "quantity": 1}, {"product_id": "sku"}),
    ],
)
def test_original_names_are_not_accessible(item: Any, aliases: dict[str, str]) -> None:
    adapter = ItemAdapter(item, aliases=aliases)
    assert adapter["sku"] == "p"
    with pytest.raises(KeyError):
        adapter["product_id"]
    with pytest.raises(KeyError):
        adapter["product_id"] = "changed"
    with pytest.raises(KeyError):
        del adapter["product_id"]


def test_original_names_are_not_accessible_for_attrs(attrs_classes) -> None:
    _, AttrsItem, _ = attrs_classes
    adapter = ItemAdapter(AttrsItem("p", "s"), aliases={"product_id": "item_id"})
    assert adapter["item_id"] == "p"
    with pytest.raises(KeyError):
        adapter["product_id"]


def test_metadata_alias_is_applied_to_all_operations() -> None:
    item = DataclassMetadataItem("p")
    adapter = ItemAdapter(item)

    assert list(adapter.field_names()) == ["sku", "quantity"]
    assert adapter.asdict() == {"sku": "p", "quantity": 1}
    assert adapter["sku"] == "p"
    adapter["sku"] = "q"
    assert item.product_id == "q"
    assert adapter.get_field_meta("sku")["alias"] == "sku"
    assert ItemAdapter.get_field_meta_from_class(DataclassMetadataItem, "sku")["alias"] == "sku"
    assert ItemAdapter.get_field_names_from_class(DataclassMetadataItem) == ["sku", "quantity"]

    schema = ItemAdapter.get_json_schema(DataclassMetadataItem)
    assert list(schema["properties"]) == ["sku", "quantity"]
    assert schema["required"] == ["sku"]


def test_attrs_metadata_alias(attrs_classes) -> None:
    _, _, AttrsMetadataItem = attrs_classes
    item = AttrsMetadataItem("p")
    adapter = ItemAdapter(item)

    assert adapter["sku"] == "p"
    adapter["sku"] = "q"
    assert item.product_id == "q"
    assert adapter.asdict() == {"sku": "q", "quantity": 1}
    assert list(ItemAdapter.get_json_schema(AttrsMetadataItem)["properties"]) == [
        "sku",
        "quantity",
    ]


def test_swap_aliases_in_json_schema() -> None:
    aliases = {"product_id": "sku", "sku": "product_id"}
    expected_properties = {
        "sku": {"type": "string"},
        "product_id": {"type": "string"},
    }

    assert (
        ItemAdapter.get_json_schema(DataclassItem, aliases=aliases)["properties"]
        == expected_properties
    )
    attrs = pytest.importorskip("attrs")

    @attrs.define
    class AttrsItem:
        product_id: str
        sku: str

    assert (
        ItemAdapter.get_json_schema(AttrsItem, aliases=aliases)["properties"]
        == expected_properties
    )


def test_collision_and_duplicate_aliases_are_rejected_before_operations() -> None:
    for aliases in (
        {"product_id": "sku"},
        {"product_id": "same", "sku": "same"},
        {"missing": "sku"},
    ):
        with pytest.raises(AliasError):
            ItemAdapter(DataclassCollisionItem("p", "s"), aliases=aliases)
        with pytest.raises(AliasError):
            ItemAdapter.get_json_schema(DataclassCollisionItem, aliases=aliases)


@dataclass
class SpyItem:
    product_id: str = "p"
    sku: str = "s"
    getter_calls: int = 0

    def __getattribute__(self, name: str) -> Any:
        if name in {"product_id", "sku"}:
            object.__setattr__(
                self, "getter_calls", object.__getattribute__(self, "getter_calls") + 1
            )
        return object.__getattribute__(self, name)


@pytest.mark.parametrize(
    "aliases",
    [
        {"product_id": "sku"},
        {"product_id": "same", "sku": "same"},
        {"missing": "sku"},
    ],
)
def test_failed_alias_definition_calls_no_getter_and_does_not_mutate(
    aliases: dict[str, str],
) -> None:
    item = SpyItem()
    before = dict(item.__dict__)
    with pytest.raises(AliasError):
        ItemAdapter(item, aliases=aliases)
    assert item.getter_calls == 0
    assert item.__dict__ == before


def test_failed_operation_calls_no_getter_and_does_not_mutate() -> None:
    item = SpyItem()
    adapter = ItemAdapter(item, aliases={"product_id": "item_id"})
    before = dict(item.__dict__)

    with pytest.raises(KeyError):
        adapter["product_id"]
    with pytest.raises(KeyError):
        adapter["product_id"] = "changed"
    with pytest.raises(KeyError):
        del adapter["product_id"]

    assert item.getter_calls == 0
    assert item.__dict__ == before


def test_nested_items_use_their_own_alias_mapping() -> None:
    item = ParentItem("parent", ChildItem("child"))
    adapter = ItemAdapter(item)

    assert adapter.asdict() == {"sku": "parent", "child": {"sku": "child"}}
    schema = ItemAdapter.get_json_schema(ParentItem)
    assert list(schema["properties"]) == ["sku", "child"]
    assert list(schema["properties"]["child"]["properties"]) == ["sku"]

    dict_item = {"product_id": "parent", "child": ChildItem("child")}
    assert ItemAdapter(dict_item, aliases={"product_id": "sku"}).asdict() == {
        "sku": "parent",
        "child": {"sku": "child"},
    }


def test_alias_view_exports_schema_and_can_be_created_from_class() -> None:
    item = DataclassItem("p", "s")
    view = ItemAdapter(item).alias_view({"product_id": "sku", "sku": "product_id"})
    assert list(view.get_json_schema()["properties"]) == ["sku", "product_id"]
    assert list(ItemAdapter.alias_view(item, {"product_id": "item_id"}).field_names()) == [
        "item_id",
        "sku",
    ]


def test_default_item_adapter_keeps_original_names() -> None:
    item = DataclassPlainItem("p", 1)
    adapter = ItemAdapter(item)
    assert list(adapter.field_names()) == ["product_id", "quantity"]
    assert adapter.asdict() == {"product_id": "p", "quantity": 1}
    assert list(ItemAdapter.get_json_schema(DataclassPlainItem)["properties"]) == [
        "product_id",
        "quantity",
    ]

    dict_item = {"product_id": "p", "quantity": 1}
    assert ItemAdapter(dict_item).asdict() == dict_item
