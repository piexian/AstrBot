import copy
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Generic

import jsonschema
import mcp
from deprecated import deprecated
from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass

from astrbot.core.message.message_event_result import MessageEventResult

from .run_context import ContextWrapper, TContext

ParametersType = dict[str, Any]
ToolExecResult = str | mcp.types.CallToolResult


@dataclass
class ToolSchema:
    """A class representing the schema of a tool for function calling."""

    name: str
    """The name of the tool."""

    description: str
    """The description of the tool."""

    parameters: ParametersType
    """The parameters of the tool, in JSON Schema format."""

    @model_validator(mode="after")
    def validate_parameters(self) -> "ToolSchema":
        jsonschema.validate(
            self.parameters, jsonschema.Draft202012Validator.META_SCHEMA
        )
        return self


@dataclass
class FunctionTool(ToolSchema, Generic[TContext]):
    """A callable tool, for function calling."""

    handler: (
        Callable[..., Awaitable[str | None] | AsyncGenerator[MessageEventResult, None]]
        | None
    ) = None
    """a callable that implements the tool's functionality. It should be an async function."""

    handler_module_path: str | None = None
    """
    The module path of the handler function. This is empty when the origin is mcp.
    This field must be retained, as the handler will be wrapped in functools.partial during initialization,
    causing the handler's __module__ to be functools
    """
    active: bool = True
    """
    Whether the tool is active. This field is a special field for AstrBot.
    You can ignore it when integrating with other frameworks.
    """
    is_background_task: bool = False
    """
    Declare this tool as a background task. Background tasks return immediately
    with a task identifier while the real work continues asynchronously.
    """

    def __repr__(self) -> str:
        return f"FuncTool(name={self.name}, parameters={self.parameters}, description={self.description})"

    async def call(self, context: ContextWrapper[TContext], **kwargs) -> ToolExecResult:
        """Run the tool with the given arguments. The handler field has priority."""
        raise NotImplementedError(
            "FunctionTool.call() must be implemented by subclasses or set a handler."
        )


@dataclass
class ToolSet:
    """A set of function tools that can be used in function calling.

    This class provides methods to add, remove, and retrieve tools, as well as
    convert the tools to different API formats (OpenAI, Anthropic, Google GenAI).
    """

    tools: list[FunctionTool] = Field(default_factory=list)

    def empty(self) -> bool:
        """Check if the tool set is empty."""
        return len(self.tools) == 0

    def add_tool(self, tool: FunctionTool) -> None:
        """Add a tool to the set.

        If a tool with the same name already exists:
        - Prefer the one that is active (active=True)
        - If both have the same active state, use the new one (overwrite)
        """
        for i, existing_tool in enumerate(self.tools):
            if existing_tool.name == tool.name:
                # Use getattr with a default of True for compatibility with tools
                # that may not define an `active` attribute (e.g., mocks).
                existing_active = bool(getattr(existing_tool, "active", True))
                new_active = bool(getattr(tool, "active", True))
                # Overwrite if new tool is active, or if existing tool is not active
                if new_active or not existing_active:
                    self.tools[i] = tool
                return
        self.tools.append(tool)

    def remove_tool(self, name: str) -> None:
        """Remove a tool by its name."""
        self.tools = [tool for tool in self.tools if tool.name != name]

    def get_tool(self, name: str) -> FunctionTool | None:
        """Get a tool by its name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def get_light_tool_set(self) -> "ToolSet":
        """Return a light tool set with only name/description."""
        light_tools = []
        for tool in self.tools:
            if hasattr(tool, "active") and not tool.active:
                continue
            light_params = {
                "type": "object",
                "properties": {},
            }
            light_tools.append(
                FunctionTool(
                    name=tool.name,
                    parameters=light_params,
                    description=tool.description,
                    handler=None,
                )
            )
        return ToolSet(light_tools)

    def get_param_only_tool_set(self) -> "ToolSet":
        """Return a tool set with name/parameters only (no description)."""
        param_tools = []
        for tool in self.tools:
            if hasattr(tool, "active") and not tool.active:
                continue
            params = (
                copy.deepcopy(tool.parameters)
                if tool.parameters
                else {"type": "object", "properties": {}}
            )
            param_tools.append(
                FunctionTool(
                    name=tool.name,
                    parameters=params,
                    description="",
                    handler=None,
                )
            )
        return ToolSet(param_tools)

    @deprecated(reason="Use add_tool() instead", version="4.0.0")
    def add_func(
        self,
        name: str,
        func_args: list,
        desc: str,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """Add a function tool to the set."""
        params = {
            "type": "object",  # hard-coded here
            "properties": {},
        }
        for param in func_args:
            params["properties"][param["name"]] = {
                "type": param["type"],
                "description": param["description"],
            }
        _func = FunctionTool(
            name=name,
            parameters=params,
            description=desc,
            handler=handler,
        )
        self.add_tool(_func)

    @deprecated(reason="Use remove_tool() instead", version="4.0.0")
    def remove_func(self, name: str) -> None:
        """Remove a function tool by its name."""
        self.remove_tool(name)

    @deprecated(reason="Use get_tool() instead", version="4.0.0")
    def get_func(self, name: str) -> FunctionTool | None:
        """Get all function tools."""
        return self.get_tool(name)

    @property
    def func_list(self) -> list[FunctionTool]:
        """Get the list of function tools."""
        return self.tools

    def openai_schema(self, omit_empty_parameter_field: bool = False) -> list[dict]:
        """Convert tools to OpenAI API function calling schema format."""
        result = []
        # Stable ordering preserves prompt-cache prefixes for compatible providers.
        for tool in sorted(self.tools, key=lambda tool: tool.name):
            func_def = {"type": "function", "function": {"name": tool.name}}
            if tool.description:
                func_def["function"]["description"] = tool.description

            if tool.parameters is not None:
                if (
                    tool.parameters and tool.parameters.get("properties")
                ) or not omit_empty_parameter_field:
                    func_def["function"]["parameters"] = tool.parameters

            result.append(func_def)
        return result

    def anthropic_schema(self) -> list[dict]:
        """Convert tools to Anthropic API format."""
        result = []
        for tool in self.tools:
            input_schema = {"type": "object"}
            if tool.parameters:
                input_schema["properties"] = tool.parameters.get("properties", {})
                input_schema["required"] = tool.parameters.get("required", [])
            tool_def = {"name": tool.name, "input_schema": input_schema}
            if tool.description:
                tool_def["description"] = tool.description
            result.append(tool_def)
        return result

    def google_schema(self) -> dict:
        """Build lossless JSON Schema declarations for Google GenerateContent.

        Returns:
            Declarations using parameters_json_schema, never both schema fields.

        Raises:
            ValueError: A local reference is invalid/cyclic or expansion exceeds
                the bounded budget. Remote references are never fetched.
        """
        declarations = []
        for tool in self.tools:
            definition: dict[str, Any] = {"name": tool.name}
            if tool.description:
                definition["description"] = tool.description
            if tool.parameters:
                root = tool.parameters
                nodes = 0

                def expand(
                    schema, path: str, refs: tuple[str, ...] = (), depth: int = 0
                ):
                    nonlocal nodes
                    nodes += 1
                    if nodes > 2048 or depth > 32:
                        raise ValueError(
                            f"Tool {tool.name} schema {path}: reference expansion limit exceeded."
                        )
                    if isinstance(schema, bool):
                        return schema
                    if not isinstance(schema, dict):
                        raise ValueError(
                            f"Tool {tool.name} schema {path}: expected a schema object or boolean."
                        )
                    if "$ref" in schema:
                        ref = schema["$ref"]
                        if not isinstance(ref, str) or not (
                            ref == "#" or ref.startswith("#/")
                        ):
                            raise ValueError(
                                f"Tool {tool.name} schema {path}: remote ref unsupported; supply a local definition."
                            )
                        if ref in refs:
                            raise ValueError(
                                f"Tool {tool.name} schema {path}: cyclic ref unsupported."
                            )
                        target = root
                        try:
                            for token in ref[2:].split("/") if ref != "#" else []:
                                target = target[
                                    token.replace("~1", "/").replace("~0", "~")
                                ]
                        except (KeyError, TypeError) as exc:
                            raise ValueError(
                                f"Tool {tool.name} schema {path}: unresolved local ref."
                            ) from exc
                        resolved = expand(target, path, (*refs, ref), depth + 1)
                        siblings = {
                            key: value for key, value in schema.items() if key != "$ref"
                        }
                        if siblings:
                            return {
                                "allOf": [
                                    resolved,
                                    expand(siblings, path, refs, depth + 1),
                                ]
                            }
                        return resolved
                    result = {}
                    for key, value in schema.items():
                        child_path = f"{path}/{key}"
                        if key in {"$defs", "definitions"}:
                            continue
                        if key in {
                            "properties",
                            "patternProperties",
                            "dependentSchemas",
                        }:
                            result[key] = {
                                name: expand(
                                    child, f"{child_path}/{name}", refs, depth + 1
                                )
                                for name, child in value.items()
                            }
                        elif key in {"allOf", "anyOf", "oneOf", "prefixItems"}:
                            result[key] = [
                                expand(child, f"{child_path}/{i}", refs, depth + 1)
                                for i, child in enumerate(value)
                            ]
                        elif key in {
                            "items",
                            "additionalProperties",
                            "not",
                            "if",
                            "then",
                            "else",
                            "contains",
                            "propertyNames",
                            "unevaluatedProperties",
                            "unevaluatedItems",
                            "additionalItems",
                        }:
                            result[key] = expand(value, child_path, refs, depth + 1)
                        else:
                            result[key] = copy.deepcopy(value)
                    return result

                definition["parameters_json_schema"] = expand(root, "#")
            declarations.append(definition)
        return {"function_declarations": declarations} if declarations else {}

    @deprecated(reason="Use openai_schema() instead", version="4.0.0")
    def get_func_desc_openai_style(self, omit_empty_parameter_field: bool = False):
        return self.openai_schema(omit_empty_parameter_field)

    @deprecated(reason="Use anthropic_schema() instead", version="4.0.0")
    def get_func_desc_anthropic_style(self):
        return self.anthropic_schema()

    @deprecated(reason="Use google_schema() instead", version="4.0.0")
    def get_func_desc_google_genai_style(self):
        return self.google_schema()

    def names(self) -> list[str]:
        """获取所有工具的名称列表"""
        return [tool.name for tool in self.tools]

    def merge(self, other: "ToolSet") -> None:
        """Merge another ToolSet into this one."""
        for tool in other.tools:
            self.add_tool(tool)

    def __len__(self) -> int:
        return len(self.tools)

    def __bool__(self) -> bool:
        return len(self.tools) > 0

    def __iter__(self):
        return iter(self.tools)

    def __repr__(self) -> str:
        return f"ToolSet(tools={self.tools})"

    def __str__(self) -> str:
        return f"ToolSet(tools={self.tools})"
