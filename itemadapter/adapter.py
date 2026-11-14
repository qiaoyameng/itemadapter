from __future__ import annotations

import dataclasses
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, KeysView, Mapping, MutableMapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, ClassVar, cast

from itemadapter._aliases import (
    AliasError,
    AliasNamespace,
    alias_from_metadata,
    apply_aliases_to_schema,
    resolve_aliases,
)
from itemadapter._imports import _is_typeddict, _scrapy_item_classes, attrs
from itemadapter._json_schema import (
    _json_schema_from_attrs,
    _json_schema_from_dataclass,
    _json_schema_from_item_class,
    _json_schema_from_pydantic,
    _json_schema_from_typed_dict,
    _JsonSchemaState,
    _setdefault_attribute_docstrings_on_json_schema,
    _setdefault_attribute_types_on_json_schema,
)
from itemadapter._utils import (
    _get_pydantic_model_metadata,
    _get_typed_dict_field_metadata,
    _is_attrs_class,
    _is_pydantic_model,
)

__all__ = [
    "AdapterInterface",
    "AliasError",
    "AttrsAdapter",
    "DataclassAdapter",
    "DictAdapter",
    "ItemAdapter",
    "PydanticAdapter",
    "ScrapyItemAdapter",
]


class AdapterInterface(MutableMapping[str, Any], metaclass=ABCMeta):
    """Abstract Base Class for adapters.

    An adapter that handles a specific type of item should inherit from this
    class and implement the abstract methods defined here, plus the
    abtract methods inherited from the MutableMapping base class.
    """

    def __init__(self, item: Any) -> None:
        self.item = item

    @classmethod
    @abstractmethod
    def is_item_class(cls, item_class: type) -> bool:
        """Return True if the adapter can handle the given item class, False otherwise."""
        raise NotImplementedError

    @classmethod
    def is_item(cls, item: object) -> bool:
        """Return True if the adapter can handle the given item, False otherwise."""
        return cls.is_item_class(item.__class__)

    @classmethod
    def get_field_meta_from_class(
        cls, item_class: type, field_name: str
    ) -> MappingProxyType[str, Any]:
        return MappingProxyType({})

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> list[str] | None:
        """Return a list of fields defined for ``item_class``.
        If a class doesn't support fields, None is returned."""
        return None

    @classmethod
    def get_json_schema(
        cls, item_class: type, *, _state: _JsonSchemaState | None = None
    ) -> dict[str, Any]:
        return _json_schema_from_item_class(cls, item_class, _state)

    def get_field_meta(self, field_name: str) -> MappingProxyType[str, Any]:
        """Return metadata for the given field name, if available."""
        return self.get_field_meta_from_class(self.item.__class__, field_name)

    def field_names(self) -> KeysView[str]:
        """Return a dynamic view of the item's field names."""
        return self.keys()


class _MixinAttrsDataclassAdapter:
    _fields_dict: dict[str, Any]
    item: Any

    def get_field_meta(self, field_name: str) -> MappingProxyType[str, Any]:
        return cast("MappingProxyType[str, Any]", self._fields_dict[field_name].metadata)

    def field_names(self) -> KeysView[str]:
        return KeysView(self._fields_dict)

    def __getitem__(self, field_name: str) -> Any:
        if field_name in self._fields_dict:
            try:
                return getattr(self.item, field_name)
            except AttributeError as ex:
                raise KeyError(field_name) from ex
        raise KeyError(field_name)

    def __setitem__(self, field_name: str, value: Any) -> None:
        if field_name in self._fields_dict:
            setattr(self.item, field_name, value)
        else:
            raise KeyError(f"{self.item.__class__.__name__} does not support field: {field_name}")

    def __delitem__(self, field_name: str) -> None:
        if field_name in self._fields_dict:
            try:
                if hasattr(self.item, field_name):
                    delattr(self.item, field_name)
                else:
                    raise AttributeError
            except AttributeError as ex:
                raise KeyError(field_name) from ex
        else:
            raise KeyError(f"{self.item.__class__.__name__} does not support field: {field_name}")

    def __iter__(self) -> Iterator[str]:
        return iter(attr for attr in self._fields_dict if hasattr(self.item, attr))

    def __len__(self) -> int:
        return len(list(iter(self)))


class AttrsAdapter(_MixinAttrsDataclassAdapter, AdapterInterface):
    def __init__(self, item: Any) -> None:
        super().__init__(item)
        if attrs is None:
            raise RuntimeError("attrs module is not available")
        # store a reference to the item's fields to avoid O(n) lookups and O(n^2) traversals
        self._fields_dict = attrs.fields_dict(self.item.__class__)

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return _is_attrs_class(item_class)

    @classmethod
    def get_field_meta_from_class(
        cls, item_class: type, field_name: str
    ) -> MappingProxyType[str, Any]:
        if attrs is None:
            raise RuntimeError("attrs module is not available")
        try:
            return cast(
                "MappingProxyType[str, Any]", attrs.fields_dict(item_class)[field_name].metadata
            )
        except KeyError as ex:
            raise KeyError(f"{item_class.__name__} does not support field: {field_name}") from ex

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> list[str] | None:
        if attrs is None:
            raise RuntimeError("attrs module is not available")
        return [a.name for a in attrs.fields(item_class)]

    @classmethod
    def get_json_schema(
        cls, item_class: type, *, _state: _JsonSchemaState | None = None
    ) -> dict[str, Any]:
        _state = _state or _JsonSchemaState(adapter=cls, containers={item_class})
        return _json_schema_from_attrs(item_class, _state)


class DataclassAdapter(_MixinAttrsDataclassAdapter, AdapterInterface):
    def __init__(self, item: Any) -> None:
        super().__init__(item)
        # store a reference to the item's fields to avoid O(n) lookups and O(n^2) traversals
        self._fields_dict = {field.name: field for field in dataclasses.fields(self.item)}

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return dataclasses.is_dataclass(item_class)

    @classmethod
    def get_field_meta_from_class(
        cls, item_class: type, field_name: str
    ) -> MappingProxyType[str, Any]:
        for field in dataclasses.fields(item_class):
            if field.name == field_name:
                return field.metadata
        raise KeyError(f"{item_class.__name__} does not support field: {field_name}")

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> list[str] | None:
        return [a.name for a in dataclasses.fields(item_class)]

    @classmethod
    def get_json_schema(
        cls, item_class: type, *, _state: _JsonSchemaState | None = None
    ) -> dict[str, Any]:
        _state = _state or _JsonSchemaState(adapter=cls, containers={item_class})
        return _json_schema_from_dataclass(item_class, _state)


class PydanticAdapter(AdapterInterface):
    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return _is_pydantic_model(item_class)

    @classmethod
    def get_field_meta_from_class(
        cls, item_class: type, field_name: str
    ) -> MappingProxyType[str, Any]:
        try:
            return _get_pydantic_model_metadata(item_class, field_name)
        except KeyError as ex:
            raise KeyError(f"{item_class.__name__} does not support field: {field_name}") from ex

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> list[str] | None:
        return list(item_class.model_fields.keys())  # type: ignore[attr-defined]

    @classmethod
    def get_json_schema(
        cls, item_class: type, *, _state: _JsonSchemaState | None = None
    ) -> dict[str, Any]:
        return _json_schema_from_pydantic(cls, item_class, _state)

    def field_names(self) -> KeysView[str]:
        return KeysView(self.item.__class__.model_fields)

    def __getitem__(self, field_name: str) -> Any:
        if field_name not in self.item.__class__.model_fields:
            raise KeyError(field_name)
        try:
            return getattr(self.item, field_name)
        except AttributeError as ex:
            raise KeyError(field_name) from ex

    def __setitem__(self, field_name: str, value: Any) -> None:
        if field_name not in self.item.__class__.model_fields:
            raise KeyError(f"{self.item.__class__.__name__} does not support field: {field_name}")
        setattr(self.item, field_name, value)

    def __delitem__(self, field_name: str) -> None:
        if field_name not in self.item.__class__.model_fields:
            raise KeyError(f"{self.item.__class__.__name__} does not support field: {field_name}")
        try:
            if hasattr(self.item, field_name):
                delattr(self.item, field_name)
                return
            raise AttributeError
        except AttributeError as ex:
            raise KeyError(field_name) from ex

    def __iter__(self) -> Iterator[str]:
        return iter(attr for attr in self.item.__class__.model_fields if hasattr(self.item, attr))

    def __len__(self) -> int:
        return len(list(iter(self)))


class _MixinDictScrapyItemAdapter:
    item: Any

    def __getitem__(self, field_name: str) -> Any:
        return self.item[field_name]

    def __setitem__(self, field_name: str, value: Any) -> None:
        self.item[field_name] = value

    def __delitem__(self, field_name: str) -> None:
        del self.item[field_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.item)

    def __len__(self) -> int:
        return len(self.item)


class DictAdapter(_MixinDictScrapyItemAdapter, AdapterInterface):
    """Adapter for :class:`dict` items.

    :class:`~typing.TypedDict` subclasses are also supported as item classes.
    Their instances are plain dictionaries at run time, so they are handled
    like any other dictionary.
    """

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return issubclass(item_class, dict)

    @classmethod
    def get_field_meta_from_class(
        cls, item_class: type, field_name: str
    ) -> MappingProxyType[str, Any]:
        if _is_typeddict(item_class):
            return _get_typed_dict_field_metadata(item_class, field_name)
        return MappingProxyType({})

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> list[str] | None:
        if _is_typeddict(item_class):
            return list(item_class.__annotations__)
        return None

    @classmethod
    def get_json_schema(
        cls, item_class: type, *, _state: _JsonSchemaState | None = None
    ) -> dict[str, Any]:
        if _is_typeddict(item_class):
            _state = _state or _JsonSchemaState(adapter=cls, containers={item_class})
            return _json_schema_from_typed_dict(item_class, _state)
        return {"type": "object"}

    def field_names(self) -> KeysView[str]:
        return KeysView(self.item)


class ScrapyItemAdapter(_MixinDictScrapyItemAdapter, AdapterInterface):
    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return issubclass(item_class, _scrapy_item_classes)

    @classmethod
    def get_field_meta_from_class(
        cls, item_class: type, field_name: str
    ) -> MappingProxyType[str, Any]:
        return MappingProxyType(item_class.fields[field_name])  # type: ignore[attr-defined]

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> list[str] | None:
        return list(item_class.fields.keys())  # type: ignore[attr-defined]

    @classmethod
    def get_json_schema(
        cls, item_class: type, *, _state: _JsonSchemaState | None = None
    ) -> dict[str, Any]:
        _state = _state or _JsonSchemaState(adapter=cls, containers={item_class})
        schema = _json_schema_from_item_class(
            cls, item_class, _state, _apply_aliases=False
        )
        _setdefault_attribute_types_on_json_schema(schema, item_class, _state)
        _setdefault_attribute_docstrings_on_json_schema(schema, item_class)
        schema.pop("required", None)
        field_names = list(cls.get_field_names_from_class(item_class) or ())
        fields_meta = {
            field_name: cls.get_field_meta_from_class(item_class, field_name)
            for field_name in field_names
        }
        metadata_aliases = {
            field_name: alias
            for field_name in field_names
            if (alias := alias_from_metadata(fields_meta[field_name])) is not None
        }
        apply_aliases_to_schema(
            schema,
            resolve_aliases(field_names, metadata_aliases=metadata_aliases)
            if metadata_aliases
            else None,
        )
        return schema

    def field_names(self) -> KeysView[str]:
        return KeysView(self.item.fields)


@lru_cache
def _has_item_overrides(adapter_classes: tuple[type[AdapterInterface], ...]) -> bool:
    """Return True if any of the given adapter classes overrides is_item(), i.e.
    decides on an item basis, and hence cannot be matched to an item by item
    class alone."""
    base_is_item = AdapterInterface.is_item.__func__  # type: ignore[attr-defined]
    overriding = [
        adapter_class
        for adapter_class in adapter_classes
        if getattr(adapter_class.is_item, "__func__", adapter_class.is_item) is not base_is_item
    ]
    if overriding:
        names = ", ".join(adapter_class.__qualname__ for adapter_class in overriding)
        warnings.warn(
            f"The following adapter classes override the is_item() class method: {names}. "
            f"Support for that will be removed, and items will then be matched to an "
            f"adapter class by item class alone. Override is_item_class() instead.",
            DeprecationWarning,
            stacklevel=4,
        )
    return bool(overriding)


@lru_cache(maxsize=1024)
def _find_adapter_class(
    adapter_classes: tuple[type[AdapterInterface], ...], item_class: type
) -> tuple[type[AdapterInterface] | None, bool]:
    """Return the first of the given adapter classes that handles the given item
    class, or None, along with the return value of _has_item_overrides()."""
    for adapter_class in adapter_classes:
        if adapter_class.is_item_class(item_class):
            return adapter_class, _has_item_overrides(adapter_classes)
    return None, _has_item_overrides(adapter_classes)


class _ItemAdapterGetJsonSchema:
    """Expose get_json_schema() on both ItemAdapter classes and alias views."""

    def __get__(self, instance: Any, owner: type[ItemAdapter]) -> Any:
        def get_json_schema(
            item_class: type | None = None,
            *,
            aliases: Mapping[str, str] | None = None,
            alias_map: Mapping[str, str] | None = None,
            allow_original_names: bool = False,
            _state: _JsonSchemaState | None = None,
        ) -> dict[str, Any]:
            if aliases is not None and alias_map is not None:
                raise TypeError("Pass either aliases or alias_map, not both")
            if item_class is None:
                if instance is None:
                    raise TypeError("get_json_schema() missing required item_class")
                item_class = instance.item.__class__
                if (
                    aliases is None
                    and alias_map is None
                    and instance._alias_namespace is not None
                ):
                    aliases = dict(instance._alias_namespace.canonical_to_public)
                if not allow_original_names:
                    allow_original_names = instance._allow_original_names
            return owner._get_json_schema(
                item_class,
                aliases=aliases,
                alias_map=alias_map,
                allow_original_names=allow_original_names,
                _state=_state,
            )

        return get_json_schema


class ItemAdapter(MutableMapping[str, Any]):
    """Wrapper class to interact with data container objects. It provides a common interface
    to extract and set data without having to take the object's type into account.
    """

    ADAPTER_CLASSES: ClassVar[Iterable[type[AdapterInterface]]] = (
        ScrapyItemAdapter,
        DictAdapter,
        DataclassAdapter,
        AttrsAdapter,
        PydanticAdapter,
    )

    def __init__(
        self,
        item: Any,
        aliases: Mapping[str, str] | None = None,
        *,
        alias_map: Mapping[str, str] | None = None,
        allow_original_names: bool = False,
    ) -> None:
        if aliases is not None and alias_map is not None:
            raise TypeError("Pass either aliases or alias_map, not both")
        caller_aliases = alias_map if alias_map is not None else aliases
        adapter_classes = tuple(self.ADAPTER_CLASSES)
        adapter_class, item_overrides = _find_adapter_class(adapter_classes, item.__class__)
        if item_overrides:
            adapter_class = next(
                (
                    candidate_class
                    for candidate_class in adapter_classes
                    if candidate_class.is_item(item)
                ),
                None,
            )
        if adapter_class is None:
            raise TypeError(f"No adapter found for objects of type: {type(item)} ({item})")
        self.adapter = adapter_class(item)
        self._allow_original_names = allow_original_names
        self._alias_namespace = self._build_instance_alias_namespace(
            caller_aliases, allow_original_names
        )

    def _build_instance_alias_namespace(
        self,
        caller_aliases: Mapping[str, str] | None,
        allow_original_names: bool = False,
    ) -> AliasNamespace | None:
        field_names = list(self.adapter.field_names())
        metadata_aliases: dict[str, str] = {}
        for field_name in field_names:
            try:
                metadata = self.adapter.get_field_meta_from_class(
                    self.item.__class__, field_name
                )
            except (KeyError, NameError, TypeError):
                metadata = MappingProxyType({})
            alias = alias_from_metadata(metadata)
            if alias is not None:
                metadata_aliases[field_name] = alias
        if not metadata_aliases and not caller_aliases:
            return None
        return resolve_aliases(
            field_names,
            metadata_aliases=metadata_aliases,
            caller_aliases=caller_aliases,
            allow_original_names=allow_original_names,
        )

    @classmethod
    def _build_class_alias_namespace(
        cls,
        item_class: type,
        caller_aliases: Mapping[str, str] | None,
        allow_original_names: bool = False,
    ) -> AliasNamespace | None:
        adapter_class = cls._get_adapter_class(item_class)
        field_names = adapter_class.get_field_names_from_class(item_class)
        if field_names is None:
            if caller_aliases is None or (
                issubclass(item_class, dict) and not _is_typeddict(item_class)
            ):
                return None
            return resolve_aliases(
                (),
                caller_aliases=caller_aliases,
                allow_original_names=allow_original_names,
            )
        metadata_aliases: dict[str, str] = {}
        for field_name in field_names:
            try:
                metadata = adapter_class.get_field_meta_from_class(item_class, field_name)
            except (KeyError, NameError, TypeError):
                metadata = MappingProxyType({})
            alias = alias_from_metadata(metadata)
            if alias is not None:
                metadata_aliases[field_name] = alias
        if not metadata_aliases and not caller_aliases:
            return None
        return resolve_aliases(
            field_names,
            metadata_aliases=metadata_aliases,
            caller_aliases=caller_aliases,
            allow_original_names=allow_original_names,
        )

    def _translate_name(self, public_name: str) -> str:
        if self._alias_namespace is None:
            return public_name
        field_name = self._alias_namespace.resolve(public_name)
        if field_name is None:
            raise KeyError(public_name)
        return field_name

    @classmethod
    def is_item(cls, item: Any) -> bool:
        adapter_classes = tuple(cls.ADAPTER_CLASSES)
        adapter_class, item_overrides = _find_adapter_class(adapter_classes, item.__class__)
        if item_overrides:
            return any(adapter_class.is_item(item) for adapter_class in adapter_classes)
        return adapter_class is not None

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return _find_adapter_class(tuple(cls.ADAPTER_CLASSES), item_class)[0] is not None

    @classmethod
    def _get_adapter_class(cls, item_class: type) -> type[AdapterInterface]:
        adapter_class = _find_adapter_class(tuple(cls.ADAPTER_CLASSES), item_class)[0]
        if adapter_class is None:
            raise TypeError(f"{item_class} is not a valid item class")
        return adapter_class

    @classmethod
    def get_field_meta_from_class(
        cls,
        item_class: type,
        field_name: str,
        *,
        aliases: Mapping[str, str] | None = None,
        alias_map: Mapping[str, str] | None = None,
    ) -> MappingProxyType[str, Any]:
        if aliases is not None and alias_map is not None:
            raise TypeError("Pass either aliases or alias_map, not both")
        caller_aliases = alias_map if alias_map is not None else aliases
        adapter_class = cls._get_adapter_class(item_class)
        namespace = cls._build_class_alias_namespace(item_class, caller_aliases)
        if namespace is not None:
            resolved_name = namespace.resolve(field_name)
            if resolved_name is None:
                raise KeyError(field_name)
            field_name = resolved_name
        return adapter_class.get_field_meta_from_class(item_class, field_name)

    @classmethod
    def get_field_names_from_class(
        cls,
        item_class: type,
        *,
        aliases: Mapping[str, str] | None = None,
        alias_map: Mapping[str, str] | None = None,
        allow_original_names: bool = False,
    ) -> list[str] | None:
        if aliases is not None and alias_map is not None:
            raise TypeError("Pass either aliases or alias_map, not both")
        caller_aliases = alias_map if alias_map is not None else aliases
        adapter_class = cls._get_adapter_class(item_class)
        field_names = adapter_class.get_field_names_from_class(item_class)
        if field_names is None:
            if caller_aliases is not None:
                cls._build_class_alias_namespace(item_class, caller_aliases, allow_original_names)
            return None
        namespace = cls._build_class_alias_namespace(
            item_class, caller_aliases, allow_original_names
        )
        if namespace is None:
            return field_names
        return [namespace.public_name(field_name) for field_name in field_names]

    get_json_schema = _ItemAdapterGetJsonSchema()

    @classmethod
    def _get_json_schema(
        cls,
        item_class: type,
        *,
        aliases: Mapping[str, str] | None = None,
        alias_map: Mapping[str, str] | None = None,
        allow_original_names: bool = False,
        _state: _JsonSchemaState | None = None,
    ) -> dict[str, Any]:
        if aliases is not None and alias_map is not None:
            raise TypeError("Pass either aliases or alias_map, not both")
        caller_aliases = alias_map if alias_map is not None else aliases
        namespace = cls._build_class_alias_namespace(
            item_class, caller_aliases, allow_original_names
        )
        _state = _state or _JsonSchemaState(adapter=cls, containers={item_class})
        adapter_class = cls._get_adapter_class(item_class)
        schema = adapter_class.get_json_schema(item_class, _state=_state)
        if namespace is not None and caller_aliases is not None:
            metadata_namespace = cls._build_class_alias_namespace(item_class, None)
            cls._remap_schema_names(schema, metadata_namespace, namespace)
        return schema

    @staticmethod
    def _remap_schema_names(
        schema: dict[str, Any],
        old_namespace: AliasNamespace | None,
        new_namespace: AliasNamespace,
    ) -> None:
        rename_map = {}
        for field_name in new_namespace.field_names:
            old_name = (
                old_namespace.public_name(field_name)
                if old_namespace is not None
                else field_name
            )
            new_name = new_namespace.public_name(field_name)
            if old_name != new_name:
                rename_map[old_name] = new_name
        if not rename_map:
            return
        properties = schema.get("properties")
        if properties:
            schema["properties"] = {
                rename_map.get(name, name): prop for name, prop in properties.items()
            }
        if "required" in schema:
            schema["required"] = [rename_map.get(name, name) for name in schema["required"]]

    @property
    def item(self) -> Any:
        return self.adapter.item

    def __repr__(self) -> str:
        values = ", ".join([f"{key}={value!r}" for key, value in self.items()])
        return f"<{self.__class__.__name__} for {self.item.__class__.__name__}({values})>"

    def __getitem__(self, field_name: str) -> Any:
        return self.adapter.__getitem__(self._translate_name(field_name))

    def __setitem__(self, field_name: str, value: Any) -> None:
        self.adapter.__setitem__(self._translate_name(field_name), value)

    def __delitem__(self, field_name: str) -> None:
        self.adapter.__delitem__(self._translate_name(field_name))

    def __iter__(self) -> Iterator[str]:
        if self._alias_namespace is None:
            return self.adapter.__iter__()
        return self._iter_public_names()

    def _iter_public_names(self) -> Iterator[str]:
        assert self._alias_namespace is not None
        namespace = self._alias_namespace
        for field_name in self.adapter.__iter__():
            if field_name in namespace.field_names:
                public_name = namespace.public_name(field_name)
                yield public_name
                if (
                    namespace.allow_original_names
                    and field_name in namespace.renamed
                    and field_name != public_name
                    and namespace.public_to_canonical.get(field_name) == field_name
                ):
                    yield field_name
            else:
                public_name = namespace.public_name(field_name)
                if public_name in namespace.public_to_canonical:
                    raise AliasError(
                        f"Runtime field {field_name!r} collides with the alias namespace"
                    )
                yield public_name

    def __len__(self) -> int:
        if self._alias_namespace is not None and self._alias_namespace.allow_original_names:
            return sum(1 for _ in self)
        return self.adapter.__len__()

    def get_field_meta(self, field_name: str) -> MappingProxyType[str, Any]:
        """Return metadata for the given field name."""
        return self.adapter.get_field_meta(self._translate_name(field_name))

    def field_names(self) -> KeysView[str]:
        """Return read-only key view with the names of all the defined fields for the item."""
        if self._alias_namespace is not None:
            return KeysView(
                MappingProxyType(
                    {
                        public_name: field_name
                        for public_name, field_name in (
                            self._alias_namespace.public_to_canonical.items()
                        )
                    }
                )
            )
        return self.adapter.field_names()

    @property
    def aliases(self) -> MappingProxyType[str, str]:
        """Return the configured ``field name -> alias`` mapping."""
        if self._alias_namespace is None:
            return MappingProxyType({})
        return MappingProxyType(dict(self._alias_namespace.canonical_to_public))

    @property
    def alias_map(self) -> MappingProxyType[str, str]:
        """Alias of :attr:`aliases`."""
        return self.aliases

    def with_aliases(
        self,
        aliases: Mapping[str, str],
        *,
        allow_original_names: bool | None = None,
    ) -> ItemAdapter:
        """Return another adapter view for the same item with caller aliases."""
        if allow_original_names is None:
            allow_original_names = self._allow_original_names
        return self.__class__(
            self.item,
            aliases=aliases,
            allow_original_names=allow_original_names,
        )

    def alias_view(
        self,
        aliases: Mapping[str, str] | None = None,
        *,
        allow_original_names: bool | None = None,
    ) -> ItemAdapter:
        """Return an alias view of this adapter.

        This method can also be called on the class with the item as the first
        argument, e.g. ``ItemAdapter.alias_view(item, aliases)``.
        """
        if isinstance(self, ItemAdapter):
            if allow_original_names is None:
                allow_original_names = self._allow_original_names
            return self.__class__(
                self.item,
                aliases=aliases,
                allow_original_names=allow_original_names,
            )
        return ItemAdapter(
            self,
            aliases=aliases,
            allow_original_names=bool(allow_original_names),
        )

    def asdict(self) -> dict[str, Any]:
        """Return a dict object with the contents of the adapter. This works slightly different
        than calling `dict(adapter)`: it's applied recursively to nested items (if there are any).
        """
        return {key: self._asdict(value) for key, value in self.items()}

    @classmethod
    def _asdict(cls, obj: object) -> object:
        if isinstance(obj, dict):
            return {key: cls._asdict(value) for key, value in obj.items()}
        if isinstance(obj, (list, set, tuple)):
            return obj.__class__(cls._asdict(x) for x in obj)
        if isinstance(obj, cls):
            return obj.asdict()
        if cls.is_item(obj):
            return cls(obj).asdict()
        return obj
