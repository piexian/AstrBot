from __future__ import annotations

import hashlib
from dataclasses import dataclass

from astrbot import __version__

DEFAULT_USER_AGENT = f"astrbot/{__version__}"
CONVERSATION_ID_HEADER = "x-astrbot-conversation-id"

_MAX_TEMPLATE_DEPTH = 8
_CONTROL_CHARACTERS = ("\r", "\n", "\0")
_TEMPLATE_VARIABLES = frozenset(
    {"conversation_id", "umo", "request_id", "model", "provider_id", "version"}
)
_REQUEST_VARIABLES = frozenset({"conversation_id", "umo", "request_id"})
_REQUEST_TIME_VARIABLES = _REQUEST_VARIABLES | {"model"}

# Parsed template nodes are tuples of ("text", str), ("var", name) or
# ("sha256", nodes).
TemplateNode = tuple


class HeaderTemplateError(ValueError):
    """Invalid provider custom header template, header name or header value."""


@dataclass(frozen=True)
class RequestHeaderContext:
    """Values available to provider custom header templates."""

    conversation_id: str = ""
    umo: str = ""
    request_id: str = ""
    model: str = ""
    provider_id: str = ""
    version: str = ""


def _parse_expression(
    template: str, index: int, depth: int
) -> tuple[TemplateNode, int]:
    """Parse one expression starting right after its opening brace.

    Args:
        template: Header value template.
        index: Index of the first expression character.
        depth: Current nesting depth of the expression.

    Returns:
        The parsed node and the index right after the closing brace.

    Raises:
        HeaderTemplateError: If the expression is malformed or unknown.
    """
    start = index
    while index < len(template) and (
        template[index].isalnum() or template[index] == "_"
    ):
        index += 1
    name = template[start:index]
    if not name:
        raise HeaderTemplateError("Header template contains an empty expression")

    if index < len(template) and template[index] == ":":
        if name != "sha256":
            raise HeaderTemplateError(
                "Unknown header template function; only 'sha256' is supported"
            )
        nodes, index = _parse_nodes(template, index + 1, depth, inside_expression=True)
        if not nodes:
            raise HeaderTemplateError(
                "Header template function 'sha256' requires a nested template"
            )
        return ("sha256", tuple(nodes)), index

    if name not in _TEMPLATE_VARIABLES:
        raise HeaderTemplateError(
            "Unknown header template variable; supported variables: "
            + ", ".join(sorted(_TEMPLATE_VARIABLES))
        )
    if index >= len(template):
        raise HeaderTemplateError("Header template contains an unclosed '{'")
    if template[index] != "}":
        raise HeaderTemplateError(
            f"Unexpected character after header template variable '{name}'"
        )
    return ("var", name), index + 1


def _parse_nodes(
    template: str, index: int, depth: int, *, inside_expression: bool
) -> tuple[list[TemplateNode], int]:
    """Parse template nodes from ``index``.

    Args:
        template: Header value template.
        index: Index to start parsing at.
        depth: Current expression nesting depth.
        inside_expression: Whether parsing stops at the closing brace of the
            enclosing expression instead of requiring the end of the template.

    Returns:
        The parsed nodes and the index to continue from.

    Raises:
        HeaderTemplateError: If the template is malformed or nests too deeply.
    """
    nodes: list[TemplateNode] = []
    literal: list[str] = []

    def flush_literal() -> None:
        if literal:
            nodes.append(("text", "".join(literal)))
            literal.clear()

    while index < len(template):
        character = template[index]
        following = template[index + 1] if index + 1 < len(template) else ""
        if not inside_expression and character == "{" and following == "{":
            literal.append("{")
            index += 2
            continue
        if not inside_expression and character == "}" and following == "}":
            literal.append("}")
            index += 2
            continue
        if character == "{":
            if depth >= _MAX_TEMPLATE_DEPTH:
                raise HeaderTemplateError(
                    f"Header template nests deeper than {_MAX_TEMPLATE_DEPTH} levels"
                )
            flush_literal()
            node, index = _parse_expression(template, index + 1, depth + 1)
            nodes.append(node)
            continue
        if character == "}":
            if not inside_expression:
                raise HeaderTemplateError("Header template contains an unmatched '}'")
            flush_literal()
            return nodes, index + 1
        literal.append(character)
        index += 1

    if inside_expression:
        raise HeaderTemplateError("Header template contains an unclosed '{'")
    flush_literal()
    return nodes, index


def _parse_template(template: str) -> list[TemplateNode]:
    """Parse a header value template into nodes.

    Args:
        template: Header value template.

    Returns:
        The parsed nodes.

    Raises:
        HeaderTemplateError: If the template is malformed.
    """
    nodes, _ = _parse_nodes(str(template), 0, 0, inside_expression=False)
    return nodes


def _collect_variables(nodes: list[TemplateNode]) -> set[str]:
    """Collect the variable names referenced by parsed template nodes.

    Args:
        nodes: Parsed template nodes.

    Returns:
        Referenced variable names.
    """
    variables: set[str] = set()
    for kind, value in nodes:
        if kind == "var":
            variables.add(value)
        elif kind == "sha256":
            variables |= _collect_variables(list(value))
    return variables


def _is_request_time_dynamic(nodes: list[TemplateNode]) -> bool:
    """Check whether parsed template nodes need per-request values.

    Args:
        nodes: Parsed template nodes.

    Returns:
        True when the template references a request-time variable. Templates
        that only reference provider_id or version render once at startup.
    """
    return bool(_collect_variables(nodes) & _REQUEST_TIME_VARIABLES)


def _variable_value(name: str, ctx: RequestHeaderContext) -> str | None:
    """Read one template variable value.

    Args:
        name: Variable name.
        ctx: Request values available to the template.

    Returns:
        The value, or None when it is missing for this request. An empty
        conversation ID falls back to the request ID.
    """
    if name == "conversation_id":
        value = str(ctx.conversation_id or ctx.request_id or "")
    else:
        value = str(getattr(ctx, name, "") or "")
    return value or None


def _render_nodes(nodes: list[TemplateNode], ctx: RequestHeaderContext) -> str | None:
    """Render parsed template nodes.

    Args:
        nodes: Parsed template nodes.
        ctx: Request values available to the template.

    Returns:
        The rendered value, or None when a referenced variable is missing.
    """
    parts: list[str] = []
    for kind, value in nodes:
        if kind == "text":
            parts.append(value)
            continue
        if kind == "sha256":
            inner = _render_nodes(list(value), ctx)
            if inner is None:
                return None
            parts.append(hashlib.sha256(inner.encode("utf-8")).hexdigest())
            continue
        text = _variable_value(value, ctx)
        if text is None:
            return None
        parts.append(text)
    return "".join(parts)


def _ensure_safe_header(name: str, value: str) -> None:
    """Reject header names and values that would inject extra headers.

    Args:
        name: Header name.
        value: Header value or value template.

    Raises:
        HeaderTemplateError: If a control character is present.
    """
    for text in (name, value):
        if any(character in text for character in _CONTROL_CHARACTERS):
            raise HeaderTemplateError(
                f"Header '{name}' contains a control character in its name or value"
            )


def _parse_checked_header(name: str, template: str) -> list[TemplateNode] | None:
    """Validate one header name and parse its value template.

    Args:
        name: Header name.
        template: Header value template.

    Returns:
        Parsed template nodes, or None for the reserved conversation header.

    Raises:
        HeaderTemplateError: If the name, value or template is invalid.
    """
    try:
        _ensure_safe_header(name, template)
        if name.casefold() == CONVERSATION_ID_HEADER:
            return None
        return _parse_template(template)
    except HeaderTemplateError as exc:
        raise HeaderTemplateError(f"header '{name}': {exc}") from exc


def validate_header_templates(custom_headers: object, *, provider_id: str = "") -> None:
    """Validate header templates without rendering their values.

    Used for configuration whose template variables cannot all be resolved yet,
    such as a provider source whose model lives on its child providers.

    Args:
        custom_headers: Header name to template mapping.
        provider_id: Provider or source ID used in error messages.

    Raises:
        HeaderTemplateError: If a name, value or template is invalid.
    """
    if not isinstance(custom_headers, dict):
        return
    label = provider_id or "unknown"
    for raw_name, raw_template in custom_headers.items():
        try:
            _parse_checked_header(str(raw_name), str(raw_template))
        except HeaderTemplateError as exc:
            raise HeaderTemplateError(f"Provider '{label}': {exc}") from exc


def render_header_template(
    template: str,
    ctx: RequestHeaderContext,
) -> str | None:
    """Render one custom header value template.

    Args:
        template: Header value template.
        ctx: Request values available to the template.

    Returns:
        The rendered value, or None when a referenced variable is missing.

    Raises:
        HeaderTemplateError: If the template is invalid.
    """
    return _render_nodes(_parse_template(template), ctx)


def render_request_headers(
    custom_headers: dict,
    ctx: RequestHeaderContext,
) -> dict[str, str]:
    """Render request-time custom headers for one chat request.

    Headers whose template references a missing value are omitted. The
    conversation ID header is reserved and never rendered from configuration.

    Args:
        custom_headers: Header name to template mapping.
        ctx: Request values available to the templates.

    Returns:
        Rendered header mapping.

    Raises:
        HeaderTemplateError: If a template or header name is invalid.
    """
    headers: dict[str, str] = {}
    if not isinstance(custom_headers, dict):
        return headers
    for raw_name, raw_template in custom_headers.items():
        name = str(raw_name)
        _ensure_safe_header(name, str(raw_template))
        if name.casefold() == CONVERSATION_ID_HEADER:
            continue
        rendered = render_header_template(str(raw_template), ctx)
        if rendered is None:
            continue
        # A rendered value with control characters (e.g. a UMO containing CR/LF)
        # must not fail the whole request; omit the header instead.
        if any(character in rendered for character in _CONTROL_CHARACTERS):
            continue
        headers[name] = rendered
    return headers


def split_provider_headers(
    custom_headers: object,
    ctx: RequestHeaderContext,
    *,
    request_scoped: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    """Split configured custom headers into static headers and request templates.

    Args:
        custom_headers: Header name to template mapping from provider config.
        ctx: Values available when the provider instance is created.
        request_scoped: Whether the provider can inject headers per request.

    Returns:
        A ``(static_headers, request_templates)`` tuple. Static headers hold
        rendered values; request templates keep their source template.

    Raises:
        HeaderTemplateError: If a template is invalid or cannot be rendered
            without request-time values.
    """
    static_headers: dict[str, str] = {}
    request_templates: dict[str, str] = {}
    if not isinstance(custom_headers, dict):
        return static_headers, request_templates

    provider_label = ctx.provider_id or "unknown"
    for raw_name, raw_template in custom_headers.items():
        name = str(raw_name)
        template = str(raw_template)
        try:
            nodes = _parse_checked_header(name, template)
            if nodes is None:
                continue
            if request_scoped and _is_request_time_dynamic(nodes):
                request_templates[name] = template
                continue
            rendered = _render_nodes(nodes, ctx)
            if rendered is None:
                request_variables = sorted(
                    _collect_variables(nodes) & _REQUEST_VARIABLES
                )
                if not request_scoped and request_variables:
                    raise HeaderTemplateError(
                        f"header '{name}' uses request-time variable "
                        f"'{request_variables[0]}', which only chat providers support"
                    )
                raise HeaderTemplateError(
                    f"header '{name}' has no value for its template variables"
                )
            _ensure_safe_header(name, rendered)
            static_headers[name] = rendered
        except HeaderTemplateError as exc:
            raise HeaderTemplateError(f"Provider '{provider_label}': {exc}") from exc
    return static_headers, request_templates


def validate_provider_headers(provider_config: dict) -> None:
    """Validate provider custom header templates before writing configuration.

    Args:
        provider_config: Merged provider configuration.

    Raises:
        HeaderTemplateError: If any template is invalid for the provider type.
    """
    provider_type = provider_config.get("provider_type")
    if provider_type:
        provider_type = getattr(provider_type, "value", provider_type)
        split_provider_headers(
            provider_config.get("custom_headers"),
            RequestHeaderContext(
                provider_id=str(provider_config.get("id") or ""),
                model=str(provider_config.get("model") or ""),
                version=__version__,
            ),
            request_scoped=provider_type == "chat_completion",
        )
    else:
        # Without provider_type only syntax can be checked; provider
        # initialization reports request-time variable misuse instead.
        validate_header_templates(
            provider_config.get("custom_headers"),
            provider_id=str(provider_config.get("id") or ""),
        )


def build_provider_headers(custom_headers: object = None) -> dict[str, str]:
    """Build provider headers with an overridable AstrBot user agent.

    Args:
        custom_headers: Optional, already rendered header mapping.

    Returns:
        A new header dictionary with string values and one User-Agent header.
    """
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if isinstance(custom_headers, dict):
        for name, value in custom_headers.items():
            name, value = str(name), str(value)
            if name.lower() == "user-agent":
                if value.strip():
                    headers["User-Agent"] = value
            else:
                headers[name] = value
    return headers


def build_conversation_headers(conversation_id: str | None) -> dict[str, str]:
    """Build request headers for a provider conversation.

    Args:
        conversation_id: AstrBot conversation ID to associate with the request.

    Returns:
        A header mapping containing the conversation ID, or an empty mapping when
        no conversation ID is available.
    """
    if not conversation_id:
        return {}
    return {CONVERSATION_ID_HEADER: str(conversation_id)}
