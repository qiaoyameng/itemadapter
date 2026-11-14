from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


class AliasError(ValueError, KeyError):
    """Invalid alias definition.

    Inherits from both :class:`ValueError` and :class:`KeyError` so callers
    can catch it with either exception type, matching the conventions used by
    mapping and configuration APIs respectively.
    """


ALIAS_METADATA_KEYS = ("alias", "aliases", "itemadapter_alias")


def alias_from_metadata(metadata: Mapping[str, Any] | None) -> str | None:
    """Return the alias declared in field metadata, if any."""
    if not metadata:
        return None
    for key in ALIAS_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _normalize_caller_aliases(
    field_names: Iterable[str],
    aliases: Mapping[str, str] | None,
    public_to_field: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Normalize a caller-provided alias mapping to ``field name -> alias``.

    ``field name -> alias`` is the required direction.  Keys must be the
    underlying field names, and values are the public alias names.
    """
    if aliases is None:
        return {}
    if not isinstance(aliases, Mapping):
        raise AliasError("aliases must be a mapping of field names to aliases")

    known_fields = set(field_names)
    public_to_field = public_to_field or {}
    items = list(aliases.items())
    if not items:
        return {}
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in items):
        raise AliasError("alias names and target field names must be strings")

    accepted_keys = known_fields | set(public_to_field)
    unknown = next((key for key, _ in items if key not in accepted_keys), None)
    if unknown is not None:
        raise AliasError(f"Unknown field in alias mapping: {unknown!r}")
    normalized = {
        public_to_field.get(key, key): value for key, value in items
    }

    for field, alias in normalized.items():
        if field not in known_fields:
            raise AliasError(f"Unknown field in alias mapping: {field!r}")
        if not alias:
            raise AliasError(f"Alias for field {field!r} cannot be empty")
    return normalized


@dataclass(frozen=True)
class AliasNamespace:
    """Bidirectional, validated mapping between field names and public names."""

    field_names: tuple[str, ...]
    canonical_to_public: Mapping[str, str]
    public_to_canonical: Mapping[str, str]
    renamed: frozenset[str]
    allow_original_names: bool = False

    def resolve(self, public_name: str) -> str | None:
        """Translate a public name to the underlying field name."""
        field_name = self.public_to_canonical.get(public_name)
        if field_name is not None:
            return field_name
        if self.allow_original_names and public_name in self.renamed:
            return public_name
        return None

    def public_name(self, field_name: str) -> str:
        """Return the public name of an underlying field."""
        return self.canonical_to_public.get(field_name, field_name)

    @property
    def public_names(self) -> tuple[str, ...]:
        return tuple(self.public_to_canonical)


def resolve_aliases(
    field_names: Iterable[str],
    *,
    metadata_aliases: Mapping[str, str] | None = None,
    caller_aliases: Mapping[str, str] | None = None,
    allow_original_names: bool = False,
) -> AliasNamespace:
    """Build and validate an alias namespace.

    Field metadata aliases are applied first and caller aliases override them.
    A mapping is invalid if it refers to an unknown field, assigns the same
    public name to two fields, or gives a renamed field an alias that is already
    the public name of another field.
    """
    fields = tuple(field_names)
    if len(set(fields)) != len(fields):
        raise AliasError("Duplicate field names in alias namespace")

    canonical_to_public = dict(metadata_aliases or {})
    metadata_public_to_field = {
        public_name: field_name
        for field_name in fields
        for public_name in (canonical_to_public.get(field_name) or field_name,)
    }
    canonical_to_public.update(
        _normalize_caller_aliases(fields, caller_aliases, metadata_public_to_field)
    )

    known_fields = set(fields)
    for field_name, alias in tuple(canonical_to_public.items()):
        if field_name not in known_fields:
            raise AliasError(f"Unknown field in alias mapping: {field_name!r}")
        if not isinstance(alias, str) or not alias:
            raise AliasError(f"Alias for field {field_name!r} must be a non-empty string")
        if alias == field_name:
            del canonical_to_public[field_name]

    public_to_canonical: dict[str, str] = {}
    for field_name in fields:
        public_name = canonical_to_public.get(field_name, field_name)
        existing = public_to_canonical.get(public_name)
        if existing is not None and existing != field_name:
            raise AliasError(
                f"Duplicate public field name {public_name!r}: "
                f"used by {existing!r} and {field_name!r}"
            )
        public_to_canonical[public_name] = field_name

    if allow_original_names:
        for field_name in canonical_to_public:
            if field_name in public_to_canonical:
                other = public_to_canonical[field_name]
                if other != field_name:
                    raise AliasError(
                        f"Cannot expose original field name {field_name!r}: "
                        f"it is already the alias of {other!r}"
                    )
            else:
                public_to_canonical[field_name] = field_name

    return AliasNamespace(
        field_names=fields,
        canonical_to_public=MappingProxyType(canonical_to_public),
        public_to_canonical=MappingProxyType(public_to_canonical),
        renamed=frozenset(canonical_to_public),
        allow_original_names=allow_original_names,
    )


def apply_aliases_to_schema(
    schema: dict[str, Any], namespace: AliasNamespace | None
) -> None:
    """Rename JSON Schema properties and required entries in place."""
    if namespace is None:
        return
    properties = schema.get("properties")
    if not properties:
        return

    field_names = set(namespace.field_names)
    renamed_properties: dict[str, Any] = {}
    for field_name, prop in properties.items():
        public_name = (
            namespace.public_name(field_name) if field_name in field_names else field_name
        )
        renamed_properties[public_name] = prop
    schema["properties"] = renamed_properties

    if "required" in schema:
        schema["required"] = [
            namespace.public_name(field_name) if field_name in field_names else field_name
            for field_name in schema["required"]
        ]
