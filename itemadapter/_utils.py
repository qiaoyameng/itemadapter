from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, Required, get_args, get_origin

if TYPE_CHECKING:
    from itemadapter.adapter import AdapterInterface

from itemadapter._imports import (
    PydanticUndefined,
    _get_type_hints,
    attrs,
    pydantic,
    typing_extensions,
)


def _is_attrs_class(obj: type) -> bool:
    if attrs is None:
        return False
    return bool(attrs.has(obj))


def _is_pydantic_model(obj: type) -> bool:
    if pydantic is None:
        return False
    return issubclass(obj, pydantic.BaseModel)


def _get_pydantic_model_metadata(item_model: Any, field_name: str) -> MappingProxyType[str, Any]:
    metadata = {}
    field = item_model.model_fields[field_name]

    for attribute in [
        "alias_priority",
        "alias",
        "allow_inf_nan",
        "annotation",
        "coerce_numbers_to_str",
        "decimal_places",
        "default_factory",
        "deprecated",
        "description",
        "discriminator",
        "examples",
        "exclude",
        "fail_fast",
        "field_title_generator",
        "frozen",
        "ge",
        "gt",
        "init_var",
        "init",
        "json_schema_extra",
        "kw_only",
        "le",
        "lt",
        "max_digits",
        "max_length",
        "min_length",
        "multiple_of",
        "pattern",
        "repr",
        "serialization_alias",
        "strict",
        "title",
        "union_mode",
        "validate_default",
        "validation_alias",
    ]:
        if hasattr(field, attribute) and (value := getattr(field, attribute)) is not None:
            metadata[attribute] = value

    for attribute, default_value in [
        ("default", PydanticUndefined),
        ("metadata", []),
    ]:
        if hasattr(field, attribute) and (value := getattr(field, attribute)) != default_value:
            metadata[attribute] = value

    return MappingProxyType(metadata)


_EMPTY_METADATA: MappingProxyType[str, Any] = MappingProxyType({})
_READ_ONLY = {getattr(typing, "ReadOnly", None), getattr(typing_extensions, "ReadOnly", None)}
_READ_ONLY.discard(None)


def _split_typed_dict_hint(type_hint: Any) -> tuple[MappingProxyType[str, Any], bool | None]:
    """Return the metadata and the requiredness of a ``TypedDict`` field, given
    its type hint including extras.

    The metadata is the first mapping in the metadata of an
    :data:`~typing.Annotated` type hint. The requiredness is ``None`` unless
    the type hint is wrapped in :data:`~typing.Required` or
    :data:`~typing.NotRequired`.
    """
    metadata = None
    required = None
    while True:
        origin = get_origin(type_hint)
        if origin is Annotated:
            if metadata is None:
                metadata = next(
                    (entry for entry in type_hint.__metadata__ if isinstance(entry, Mapping)),
                    None,
                )
            type_hint = type_hint.__origin__
        # Required and NotRequired are missing from __required_keys__ and
        # __optional_keys__ when annotations are postponed, hence the need to
        # read requiredness from type hints as well.
        elif origin is Required or origin is NotRequired:
            required = origin is Required
            type_hint = get_args(type_hint)[0]
        elif origin in _READ_ONLY:
            type_hint = get_args(type_hint)[0]
        else:
            break
    return MappingProxyType(metadata) if metadata is not None else _EMPTY_METADATA, required


def _get_typed_dict_field_metadata(
    item_class: type, field_name: str
) -> MappingProxyType[str, Any]:
    type_hints = _get_type_hints(item_class, include_extras=True)
    if field_name not in type_hints:
        raise KeyError(f"{item_class.__name__} does not support field: {field_name}")
    return _split_typed_dict_hint(type_hints[field_name])[0]


ALIAS_METADATA_KEY = "alias"


def _metadata_aliases_from_class(
    adapter_class: type[AdapterInterface], item_class: type, field_names: Iterable[str]
) -> dict[str, str]:
    """Return the field aliases declared in the field metadata of the given
    item class, as a {field_name: alias} dict."""
    aliases: dict[str, str] = {}
    for field_name in field_names:
        metadata = adapter_class.get_field_meta_from_class(item_class, field_name)
        alias = metadata.get(ALIAS_METADATA_KEY)
        if alias is not None:
            aliases[field_name] = alias
    return aliases


def _resolve_aliases(
    field_names: tuple[str, ...],
    aliases: Mapping[str, str],
    *,
    item_class_name: str,
) -> MappingProxyType[str, str]:
    """Validate the given {field_name: alias} mapping against the defined
    field names and return it as a read-only mapping.

    Aliases must not target unknown fields, two fields must not share an
    alias, and an alias must not collide with the name of a field that is not
    itself renamed.
    """
    for field_name in aliases:
        if field_name not in field_names:
            raise ValueError(
                f"Cannot define an alias for unknown field {field_name!r} of "
                f"{item_class_name}; valid fields are: {', '.join(field_names)}"
            )
    used_aliases: set[str] = set()
    for field_name, alias in aliases.items():
        if alias in used_aliases:
            raise ValueError(
                f"Duplicate alias {alias!r} for field {field_name!r} of "
                f"{item_class_name}: another field already uses the same alias"
            )
        used_aliases.add(alias)
    renamed_fields = set(aliases)
    for field_name, alias in aliases.items():
        if alias in field_names and alias not in renamed_fields:
            raise ValueError(
                f"Alias {alias!r} for field {field_name!r} of {item_class_name} "
                f"collides with the name of an unrenamed field"
            )
    return MappingProxyType(dict(aliases))
