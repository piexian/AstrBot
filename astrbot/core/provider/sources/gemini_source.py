import asyncio
import base64
import json
import logging
import random
from collections.abc import AsyncGenerator
from typing import Literal, cast
from uuid import uuid4

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError

import astrbot.core.message.components as Comp
from astrbot import logger
from astrbot.api.provider import Provider
from astrbot.core.agent.message import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    TextPart,
    message_view_digest,
    protocol_content_digest,
)
from astrbot.core.exceptions import EmptyModelOutputError
from astrbot.core.message.message_event_result import MessageChain
from astrbot.core.provider.entities import LLMResponse, TokenUsage
from astrbot.core.provider.func_tool_manager import ToolSet
from astrbot.core.utils.media_utils import (
    describe_media_ref,
    resolve_media_ref_to_base64_data,
)
from astrbot.core.utils.network_utils import is_connection_error, log_connection_failure

from ..register import register_provider_adapter
from .request_retry import retry_provider_request


class SuppressNonTextPartsWarning(logging.Filter):
    """过滤 Gemini SDK 中的非文本部分警告"""

    def filter(self, record):
        return "there are non-text parts in the response" not in record.getMessage()


logging.getLogger("google_genai.types").addFilter(SuppressNonTextPartsWarning())


@register_provider_adapter(
    "googlegenai_chat_completion",
    "Google Gemini Chat Completion 提供商适配器",
)
class ProviderGoogleGenAI(Provider):
    preserve_native_message_state = True
    GENERATION_PARAMETERS = {
        "temperature": ("temperature",),
        "max_output_tokens": ("max_output_tokens", "max_tokens", "maxOutputTokens"),
        "top_p": ("top_p", "topP"),
        "top_k": ("top_k", "topK"),
        "frequency_penalty": ("frequency_penalty", "frequencyPenalty"),
        "presence_penalty": ("presence_penalty", "presencePenalty"),
        "stop_sequences": ("stop_sequences", "stop", "stopSequences"),
        "response_logprobs": ("response_logprobs", "responseLogprobs"),
        "logprobs": ("logprobs",),
        "seed": ("seed",),
    }
    CATEGORY_MAPPING = {
        "harassment": types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        "hate_speech": types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        "sexually_explicit": types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        "dangerous_content": types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    }

    THRESHOLD_MAPPING = {
        "BLOCK_NONE": types.HarmBlockThreshold.BLOCK_NONE,
        "BLOCK_ONLY_HIGH": types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        "BLOCK_MEDIUM_AND_ABOVE": types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "BLOCK_LOW_AND_ABOVE": types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    def __init__(
        self,
        provider_config,
        provider_settings,
    ) -> None:
        super().__init__(
            provider_config,
            provider_settings,
        )
        self.api_keys: list = super().get_keys()
        self.chosen_api_key: str = self.api_keys[0] if len(self.api_keys) > 0 else ""
        self.timeout: int = int(provider_config.get("timeout", 180))

        self.api_base: str | None = provider_config.get("api_base", None)
        if self.api_base and self.api_base.endswith("/"):
            self.api_base = self.api_base[:-1]

        self._http_client: httpx.AsyncClient | None = None
        self._stale_http_clients: list[httpx.AsyncClient] = []
        self._init_client()
        self.set_model(provider_config.get("model", "unknown"))
        self._init_safety_settings()

    def _init_client(self) -> None:
        """初始化Gemini客户端"""
        proxy = self.provider_config.get("proxy", "")
        http_options = types.HttpOptions(
            base_url=self.api_base,
            timeout=self.timeout * 1000,  # 毫秒
        )

        # 强制使用 httpx 作为异步 HTTP 后端，避免 aiohttp 响应类型兼容问题 (#7564)
        # httpx.AsyncClient 的 timeout 单位为秒（与 HttpOptions 的毫秒不同）
        async_client_kwargs: dict = {
            "base_url": self.api_base,
            "timeout": self.timeout,
        }
        if proxy:
            async_client_kwargs["proxy"] = proxy
            async_client_kwargs["trust_env"] = False
        else:
            async_client_kwargs["trust_env"] = True

        # Track the previous client so it can be closed in terminate() instead
        # of leaking when _init_client is called again (e.g. via set_key).
        # Only the most recent stale client is kept to avoid unbounded growth.
        if self._http_client is not None:
            self._stale_http_clients = [self._http_client]

        self._http_client = httpx.AsyncClient(**async_client_kwargs)
        http_options.httpx_async_client = self._http_client

        self.client = genai.Client(
            api_key=self.chosen_api_key,
            http_options=http_options,
        ).aio

    def _init_safety_settings(self) -> None:
        """初始化安全设置"""
        user_safety_config = self.provider_config.get("gm_safety_settings", {})
        self.safety_settings = [
            types.SafetySetting(
                category=harm_category,
                threshold=self.THRESHOLD_MAPPING[threshold_str],
            )
            for config_key, harm_category in self.CATEGORY_MAPPING.items()
            if (threshold_str := user_safety_config.get(config_key))
            and threshold_str in self.THRESHOLD_MAPPING
        ]

    async def _handle_api_error(self, e: APIError, keys: list[str]) -> bool:
        """处理API错误，返回是否需要重试"""
        if e.message is None:
            e.message = ""

        if e.code == 429 or "API key not valid" in e.message:
            keys.remove(self.chosen_api_key)
            if len(keys) > 0:
                self.set_key(random.choice(keys))
                logger.warning(
                    "Retrying Gemini request with another API key (status %s).",
                    e.code,
                )
                await asyncio.sleep(1)
                return True
            logger.error("No valid Gemini API keys remaining.")
            raise Exception("Gemini API rate limit reached or API key issue detected.")

        # 连接错误处理
        if is_connection_error(e):
            proxy = self.provider_config.get("proxy", "")
            log_connection_failure("Gemini", e, proxy)

        raise e

    async def _prepare_query_config(
        self,
        payloads: dict,
        tools: ToolSet | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
        system_instruction: str | None = None,
        modalities: list[str] | None = None,
        temperature: float | None = None,
    ) -> types.GenerateContentConfig:
        """Build shared generation configuration without leaking local controls.

        Args:
            payloads: Model, generation overrides and request metadata.
            tools: Current tool declaration set.
            tool_choice: Whether client tools are optional or required.
            system_instruction: The original system instruction.
            modalities: Requested response modalities.
            temperature: Optional bounded recitation-retry override.

        Returns:
            An SDK configuration with validated, prioritized generation fields.

        Raises:
            ValueError: An explicit setting or tool schema is unsupported.
        """
        if not modalities:
            modalities = ["TEXT"]

        # 流式输出不支持图片模态
        if (
            self.provider_settings.get("streaming_response", False)
            and "IMAGE" in modalities
        ):
            logger.warning(
                "Streaming responses do not support IMAGE modality, falling back to TEXT modality."
            )
            modalities = ["TEXT"]

        tool_list: list[types.Tool] | None = []
        model_name = cast(str, payloads.get("model", self.get_model()))
        native_coderunner = self.provider_config.get("gm_native_coderunner", False)
        native_search = self.provider_config.get("gm_native_search", False)
        url_context = self.provider_config.get("gm_url_context", False)

        if "gemini-2.0-lite" in model_name:
            if native_coderunner or native_search or url_context:
                logger.warning(
                    "gemini-2.0-lite does not support native code execution, search, or URL context tools. These settings will be ignored.",
                )
        else:
            if native_coderunner:
                tool_list.append(types.Tool(code_execution=types.ToolCodeExecution()))
            if native_search:
                tool_list.append(types.Tool(google_search=types.GoogleSearch()))
            if url_context:
                tool_list.append(types.Tool(url_context=types.UrlContext()))

        if tools:
            func_desc = tools.google_schema()
            tool_list.append(
                types.Tool(function_declarations=func_desc["function_declarations"]),
            )

        if not tool_list:
            tool_list = None

        tool_config = None
        has_func_decl = tool_list and any(t.function_declarations for t in tool_list)
        if has_func_decl:
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=(
                        types.FunctionCallingConfigMode.ANY
                        if tool_choice == "required"
                        else types.FunctionCallingConfigMode.AUTO
                    )
                )
            )

        # oper thinking config
        thinking_config = None
        if model_name in [
            "gemini-2.5-pro",
            "gemini-2.5-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-preview",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-lite-preview",
            "gemini-robotics-er-1.5-preview",
            "gemini-live-2.5-flash-preview-native-audio-09-2025",
        ]:
            # The thinkingBudget parameter, introduced with the Gemini 2.5 series
            thinking_budget = self.provider_config.get("gm_thinking_config", {}).get(
                "budget",
                None
                if model_name in {"gemini-2.5-pro", "gemini-2.5-pro-preview"}
                else 0,
            )
            if thinking_budget is not None:
                try:
                    thinking_config = types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )
                except ValueError as exc:
                    raise ValueError(
                        "Gemini thinking budget must be an integer."
                    ) from exc
                thinking_budget = thinking_config.thinking_budget
                if (
                    model_name in {"gemini-2.5-pro", "gemini-2.5-pro-preview"}
                    and thinking_budget != -1
                    and not 128 <= thinking_budget <= 32768
                ):
                    raise ValueError(
                        "Gemini 2.5 Pro thinking budget must be -1 or between 128 and 32768."
                    )
                if (
                    model_name in {"gemini-2.5-flash", "gemini-2.5-flash-preview"}
                    and thinking_budget != -1
                    and not 0 <= thinking_budget <= 24576
                ):
                    raise ValueError(
                        "Gemini 2.5 Flash thinking budget must be -1 or between 0 and 24576."
                    )
                if (
                    model_name
                    in {"gemini-2.5-flash-lite", "gemini-2.5-flash-lite-preview"}
                    and thinking_budget not in {-1, 0}
                    and not 512 <= thinking_budget <= 24576
                ):
                    raise ValueError(
                        "Gemini 2.5 Flash Lite thinking budget must be -1, 0 or between 512 and 24576."
                    )
        elif any(model_name.startswith(p) for p in ("gemini-3-", "gemini-3.")):
            # The thinkingLevel parameter, recommended for Gemini 3 models and onwards.
            # Use prefix match so new variants (3.1, 3-flash-lite-preview, etc.) are
            # covered without needing to keep an exhaustive list up to date.
            # Gemini 2.5 series models don't support thinkingLevel; use thinkingBudget instead.
            thinking_level = self.provider_config.get("gm_thinking_config", {}).get(
                "level", "HIGH"
            )
            if thinking_level and isinstance(thinking_level, str):
                thinking_level = thinking_level.upper()
                if (
                    model_name.startswith("gemini-3.1-pro")
                    and thinking_level == "MINIMAL"
                ):
                    raise ValueError(
                        "Gemini 3.1 Pro does not support MINIMAL thinking."
                    )
                allowed_levels = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
                fallback_level = "HIGH"
                if model_name.startswith("gemini-3.7"):
                    allowed_levels = {"LOW", "MEDIUM", "HIGH"}
                    fallback_level = "MEDIUM"
                if thinking_level not in allowed_levels:
                    logger.warning(
                        "Invalid thinking level %s for %s, using %s",
                        thinking_level,
                        model_name,
                        fallback_level,
                    )
                    thinking_level = fallback_level
                thinking_config = types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel(thinking_level)
                )
        else:
            requested = self.provider_config.get("gm_thinking_config") or {}
            if requested:
                if (
                    requested.get("budget") is not None
                    and requested.get("level") is not None
                ):
                    raise ValueError(
                        "Unknown Gemini model: configure either thinking budget or level, not both."
                    )
                logger.warning(
                    "Thinking capability for the configured model alias is unverified; preserving explicit configuration."
                )
                thinking_config = types.ThinkingConfig(
                    thinking_budget=requested.get("budget"),
                    thinking_level=requested.get("level"),
                )

        generation = {"temperature": 0.7}
        for source in (
            self.provider_config,
            payloads,
            payloads.get("generation_overrides", {}),
        ):
            for field, aliases in self.GENERATION_PARAMETERS.items():
                for alias in aliases:
                    if alias in source:
                        generation[field] = source[alias]
                        break
        if temperature is not None:
            generation["temperature"] = temperature
        if (
            generation["temperature"] is not None
            and not 0 <= generation["temperature"] <= 2
        ):
            raise ValueError("Gemini temperature must be between 0 and 2.")
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            **generation,
            response_modalities=modalities,
            tools=cast(types.ToolListUnion | None, tool_list),
            tool_config=tool_config,
            safety_settings=self.safety_settings if self.safety_settings else None,
            thinking_config=thinking_config,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )

    async def _prepare_conversation(self, payloads: dict) -> list[types.Content]:
        """Convert history without changing signed Parts or tool batch boundaries.

        Args:
            payloads: Logical messages and the target model.

        Returns:
            Explicit SDK Content objects in protocol order.

        Raises:
            ValueError: Native state is stale or a tool batch cannot be paired.
        """
        contents: list[types.Content] = []
        pending: list[tuple[str, types.FunctionCall]] = []
        results: dict[int, types.Part] = {}
        model = payloads.get("model", getattr(self, "model_name", ""))
        last_user_index = max(
            (
                index
                for index, message in enumerate(payloads["messages"])
                if message.get("role") == "user"
            ),
            default=-1,
        )
        for message_index, message in enumerate(payloads["messages"]):
            role = message["role"]
            if role == "system":
                continue
            if pending and role != "tool":
                raise ValueError(
                    f"Message {message_index}: tool batch is missing results."
                )
            if role == "tool":
                matches = [
                    i
                    for i, (call_id, _) in enumerate(pending)
                    if call_id == message.get("tool_call_id") and i not in results
                ]
                if not matches:
                    raise ValueError(
                        f"Message {message_index}: tool result does not match the current batch."
                    )
                # Legacy duplicate name IDs are recoverable only in original order.
                index = matches[0]
                if len(matches) > 1 and index != len(results):
                    raise ValueError(
                        f"Message {message_index}: ambiguous legacy tool batch."
                    )
                call = pending[index][1]
                result = types.FunctionResponse(
                    name=call.name,
                    response={"name": call.name, "content": message.get("content", "")},
                )
                if call.id is not None:
                    result.id = call.id
                results[index] = types.Part(function_response=result)
                if len(results) == len(pending):
                    contents.append(
                        types.UserContent(
                            parts=[results[i] for i in range(len(pending))]
                        )
                    )
                    pending = []
                    results = {}
                continue

            native = (message.get("provider_state") or {}).get("gemini")
            if native is not None:
                if native.get("version") != 1 or native.get("invalidated"):
                    raise ValueError(
                        f"Message {message_index}: unsupported or invalidated Gemini state."
                    )
                if native.get("display_only") is True:
                    if role != "assistant" or message.get("tool_calls"):
                        raise ValueError(
                            f"Message {message_index}: display-only state cannot contain protocol calls."
                        )
                    continue
                if native.get("view_digest") != message_view_digest(message):
                    raise ValueError(
                        f"Message {message_index}: Gemini snapshot revision changed; start a new turn without the edited signed history."
                    )
                if native.get("model") != model or native.get(
                    "backend"
                ) != protocol_content_digest(getattr(self, "api_base", None) or ""):
                    raise ValueError(
                        f"Message {message_index}: Gemini snapshot model/backend is incompatible; explicit migration is not enabled."
                    )
                if native.get("content_digest") != protocol_content_digest(
                    native.get("content")
                ):
                    raise ValueError(
                        f"Message {message_index}: native Part/signature integrity changed."
                    )
                try:
                    original = types.Content.model_validate(native["content"])
                except ValueError as exc:
                    raise ValueError(
                        f"Message {message_index}: invalid native Content state."
                    ) from exc
                if original.role is None:
                    original.role = "model"
                elif original.role != "model":
                    raise ValueError(
                        f"Message {message_index}: native response state is not model-owned."
                    )
                parts = original.parts or []
                calls = [
                    (i, p.function_call)
                    for i, p in enumerate(parts)
                    if p.function_call is not None
                ]
                bindings = native.get("calls", [])
                if len(calls) != len(bindings) or len(calls) != len(
                    message.get("tool_calls") or []
                ):
                    raise ValueError(
                        f"Message {message_index}: native tool batch bindings are incomplete."
                    )
                for (part_index, call), binding, tool in zip(
                    calls, bindings, message.get("tool_calls") or []
                ):
                    if (
                        binding["part_index"] != part_index
                        or binding["internal_id"] != tool["id"]
                        or call.name != tool["function"]["name"]
                        or (call.args or {})
                        != json.loads(tool["function"].get("arguments") or "{}")
                    ):
                        raise ValueError(
                            f"Message {message_index}: native tool batch bindings changed."
                        )
                    pending.append((binding["internal_id"], call))
                if (
                    calls
                    and native.get("requires_signature")
                    and not parts[calls[0][0]].thought_signature
                ):
                    raise ValueError(
                        f"Message {message_index}: Gemini state lost the required function-call signature; no placeholder will be injected."
                    )
                for instruction in native.get("request_suffix", []):
                    contents.append(
                        types.UserContent(
                            parts=[types.Part.from_text(text=instruction)]
                        )
                    )
                contents.append(original)
                continue

            parts: list[types.Part] = []
            content = message.get("content")
            if isinstance(content, str):
                if content or role == "user":
                    parts.append(types.Part.from_text(text=content or " "))
            elif isinstance(content, list):
                for part_index, item in enumerate(content):
                    kind = item.get("type")
                    if kind == "text":
                        if item.get("text") or role == "user":
                            parts.append(
                                types.Part.from_text(text=item.get("text") or " ")
                            )
                    elif kind == "think":
                        # Old ThinkPart has no reliable per-Part Google provenance.
                        # Tool signatures are restored only from their own metadata.
                        continue
                    elif kind in {"image_url", "audio_url"}:
                        media = await resolve_media_ref_to_base64_data(
                            item[kind]["url"],
                            media_type="image" if kind == "image_url" else "audio",
                            strict=True,
                        )
                        if media is None:
                            raise ValueError(
                                f"Message {message_index} Part {part_index}: media unavailable."
                            )
                        parts.append(
                            types.Part.from_bytes(
                                data=media.to_bytes(), mime_type=media.mime_type
                            )
                        )
                    else:
                        raise ValueError(
                            f"Message {message_index} Part {part_index}: unsupported content type."
                        )
            if role == "assistant":
                for call_index, tool in enumerate(message.get("tool_calls") or []):
                    name = tool["function"]["name"]
                    if not name:
                        raise ValueError(f"Message {message_index}: unnamed tool call.")
                    args = json.loads(tool["function"].get("arguments") or "{}")
                    call = types.FunctionCall(name=name, args=args)
                    # Legacy IDs have unknown origin; do not infer native IDs from strings.
                    part = types.Part(function_call=call)
                    signature = (
                        (tool.get("extra_content") or {}).get("google") or {}
                    ).get("thought_signature")
                    if signature:
                        try:
                            part.thought_signature = base64.b64decode(
                                signature, validate=True
                            )
                        except (ValueError, TypeError) as exc:
                            raise ValueError(
                                f"Message {message_index}: invalid tool signature encoding."
                            ) from exc
                    elif (
                        call_index == 0
                        and message_index > last_user_index
                        and model.startswith(("gemini-3-", "gemini-3."))
                    ):
                        raise ValueError(
                            f"Message {message_index}: current tool batch has no Gemini signature; source provenance is unknown and automatic migration is disabled."
                        )
                    parts.append(part)
                    pending.append((tool["id"], call))
                if not parts:
                    continue
                contents.append(types.ModelContent(parts=parts))
            elif role == "user" and parts:
                if (
                    contents
                    and isinstance(contents[-1], types.UserContent)
                    and not any(
                        p.function_response is not None
                        for p in contents[-1].parts or []
                    )
                ):
                    contents[-1].parts.extend(parts)
                else:
                    contents.append(types.UserContent(parts=parts))
        if pending:
            raise ValueError("Final tool batch is missing results.")
        if contents and contents[0].role == "model":
            contents.pop(0)
        return contents

    def _extract_reasoning_content(self, candidate: types.Candidate) -> str:
        """Extract reasoning content from candidate parts"""
        if not candidate.content or not candidate.content.parts:
            return ""

        thought_buf: list[str] = [
            (p.text or "") for p in candidate.content.parts if p.thought
        ]
        return "".join(thought_buf).strip()

    def _extract_usage(
        self, usage_metadata: types.GenerateContentResponseUsageMetadata
    ) -> TokenUsage:
        """Extract usage from response metadata.

        `prompt_token_count` includes tokens served from cache, so subtract
        `cached_content_token_count` to avoid double-counting cached input
        (matching the OpenAI provider's TokenUsage accounting).
        """
        prompt_tokens = usage_metadata.prompt_token_count or 0
        cached = usage_metadata.cached_content_token_count or 0
        return TokenUsage(
            input_other=max(0, prompt_tokens - cached),
            input_cached=cached,
            output=(usage_metadata.candidates_token_count or 0)
            + (getattr(usage_metadata, "thoughts_token_count", None) or 0),
        )

    @staticmethod
    def _ensure_usable_response(
        llm_response: LLMResponse,
        *,
        response_id: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        has_text_output = bool((llm_response.completion_text or "").strip())
        has_reasoning_output = bool((llm_response.reasoning_content or "").strip())
        has_tool_output = bool(llm_response.tools_call_args)
        has_media_output = bool(
            llm_response.result_chain
            and any(
                isinstance(part, (Comp.Image, Comp.Record))
                for part in llm_response.result_chain.chain
            )
        )
        if (
            has_text_output
            or has_reasoning_output
            or has_tool_output
            or has_media_output
        ):
            return
        raise EmptyModelOutputError(
            "Gemini completion has no usable output. "
            f"response_id={response_id}, finish_reason={finish_reason}"
        )

    def _process_content_parts(
        self,
        candidate: types.Candidate,
        llm_response: LLMResponse,
        *,
        validate_output: bool = True,
        model: str | None = None,
    ) -> MessageChain:
        """Parse one completed response and capture immutable protocol provenance.

        Args:
            candidate: Selected SDK candidate with original Part boundaries.
            llm_response: Destination for display content and executable calls.
            validate_output: Reject an otherwise unusable completion.
            model: Actual requested model, including per-request overrides.

        Returns:
            The user-facing result chain, separate from the native snapshot.

        Raises:
            ValueError: A blocked or incomplete tool response cannot be executed.
            EmptyModelOutputError: No usable content was returned.
        """
        finish_reason = candidate.finish_reason
        if finish_reason is not None and finish_reason in {
            types.FinishReason.SAFETY,
            types.FinishReason.PROHIBITED_CONTENT,
            types.FinishReason.SPII,
            types.FinishReason.BLOCKLIST,
            getattr(types.FinishReason, "IMAGE_SAFETY", None),
        }:
            raise ValueError(
                f"Gemini candidate was blocked; finish_reason={finish_reason}."
            )
        if not candidate.content:
            logger.warning("Gemini candidate has no content.")
            if validate_output:
                raise EmptyModelOutputError(
                    "Gemini candidate content is empty. "
                    f"finish_reason={candidate.finish_reason}"
                )
            llm_response.result_chain = MessageChain(chain=[])
            return llm_response.result_chain

        finish_reason = candidate.finish_reason
        result_parts: list[types.Part] | None = candidate.content.parts

        if not result_parts:
            logger.warning("Gemini candidate has no parts.")
            if validate_output:
                raise EmptyModelOutputError(
                    "Gemini candidate content parts are empty. "
                    f"finish_reason={candidate.finish_reason}"
                )
            llm_response.result_chain = MessageChain(chain=[])
            return llm_response.result_chain

        # 提取 reasoning content
        reasoning = self._extract_reasoning_content(candidate)
        if reasoning:
            llm_response.reasoning_content = reasoning

        chain = []
        part: types.Part

        # 暂时这样Fallback
        if all(
            part.inline_data
            and part.inline_data.mime_type
            and part.inline_data.mime_type.startswith("image/")
            for part in result_parts
        ):
            chain.append(Comp.Plain("这是图片"))
        batch_id = uuid4().hex
        target_model = model or getattr(self, "model_name", "")
        native = {
            "version": 1,
            "model": target_model,
            "backend": protocol_content_digest(getattr(self, "api_base", None) or ""),
            "batch_id": batch_id,
            "finish_reason": finish_reason.value if finish_reason is not None else None,
            "content": candidate.content.model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
            "calls": [],
            "requires_signature": target_model.startswith(("gemini-3-", "gemini-3.")),
        }
        call_parts = [part for part in result_parts if part.function_call is not None]
        native_ids = [
            part.function_call.id
            for part in call_parts
            if part.function_call.id is not None
        ]
        if len(native_ids) != len(set(native_ids)):
            raise ValueError(
                "Gemini response contains duplicate native call IDs; no tools were executed."
            )
        if (
            call_parts
            and native["requires_signature"]
            and not call_parts[0].thought_signature
        ):
            raise ValueError(
                "Gemini response lacks the required function-call signature; no tools were executed."
            )
        llm_response.provider_state = {"gemini": native}
        native["content_digest"] = protocol_content_digest(native["content"])
        for part_index, part in enumerate(result_parts):
            # Skip thinking parts — their text is already captured via
            # _extract_reasoning_content above.  Including them here would
            # leak the model's internal reasoning into the user-facing message,
            # which also causes duplicate/triple replies on some platforms.
            if part.text and not part.thought:
                chain.append(Comp.Plain(part.text))

            if part.function_call is not None:
                call = part.function_call
                if not call.name or not call.name.strip():
                    raise ValueError(
                        f"Gemini Part {part_index}: function call has no name."
                    )
                if getattr(call, "partial_args", None) is not None or getattr(
                    call, "will_continue", None
                ):
                    raise ValueError(
                        f"Gemini Part {part_index}: partial function arguments are unsupported; no tools were executed."
                    )
                if finish_reason and finish_reason != types.FinishReason.STOP:
                    raise ValueError(
                        "Gemini tool batch did not finish successfully; no tools were executed."
                    )
                llm_response.role = "tool"
                llm_response.tools_call_name.append(call.name)
                llm_response.tools_call_args.append(call.args or {})
                tool_call_id = f"gemini_{batch_id}_{part_index}"
                llm_response.tools_call_ids.append(tool_call_id)
                native["calls"].append(
                    {
                        "internal_id": tool_call_id,
                        "part_index": part_index,
                        "upstream_id_state": "present"
                        if call.id is not None
                        else "absent",
                        "upstream_id": call.id,
                    }
                )

            if (
                part.inline_data
                and part.inline_data.mime_type
                and part.inline_data.mime_type.startswith("image/")
                and part.inline_data.data
            ):
                chain.append(Comp.Image.fromBytes(part.inline_data.data))

            if (
                part.inline_data
                and part.inline_data.mime_type
                and part.inline_data.mime_type.startswith("audio/")
                and part.inline_data.data
            ):
                chain.append(
                    Comp.Record.fromBase64(
                        base64.b64encode(part.inline_data.data).decode("ascii")
                    )
                )

        chain_result = MessageChain(chain=chain)
        llm_response.result_chain = chain_result
        llm_response.provider_state = llm_response.to_assistant_message(
            bind_native_state=True
        ).provider_state
        if validate_output:
            self._ensure_usable_response(
                llm_response,
                response_id=None,
                finish_reason=str(finish_reason) if finish_reason is not None else None,
            )
        return chain_result

    async def _query(
        self,
        payloads: dict,
        tools: ToolSet | None,
        *,
        request_max_retries: int | None = None,
    ) -> LLMResponse:
        """Request one completion with bounded feature and transport retries.

        Args:
            payloads: Source messages, target model and generation overrides.
            tools: Current tool declarations.
            request_max_retries: Maximum transport attempts, including the first.

        Returns:
            Display output and native state for the completed model response.
        """
        system_instruction = next(
            (msg["content"] for msg in payloads["messages"] if msg["role"] == "system"),
            None,
        )

        model = payloads.get("model", self.get_model())

        modalities = ["TEXT"]
        if self.provider_config.get("gm_resp_image_modal", False):
            modalities.append("IMAGE")

        conversation = await self._prepare_conversation(payloads)
        if not conversation or conversation[-1].role == "model":
            raise ValueError(
                "Gemini request is empty or ends with a model turn; provide actual user input or tool results."
            )
        temperature = None

        result: types.GenerateContentResponse | None = None
        while True:
            try:
                config = await self._prepare_query_config(
                    payloads,
                    tools,
                    payloads.get("tool_choice", "auto"),
                    system_instruction,
                    modalities,
                    temperature,
                )
                result = await retry_provider_request(
                    "Gemini",
                    lambda: self.client.models.generate_content(
                        model=model,
                        contents=cast(types.ContentListUnion, conversation),
                        config=config,
                    ),
                    max_attempts=request_max_retries,
                )
                logger.debug(
                    "Gemini response received; candidates=%d",
                    len(result.candidates or []),
                )

                if not result.candidates:
                    logger.error("Gemini response has no candidates.")
                    reason = (
                        result.prompt_feedback.block_reason
                        if result.prompt_feedback
                        else None
                    )
                    if (
                        reason
                        and reason != types.BlockedReason.BLOCKED_REASON_UNSPECIFIED
                    ):
                        raise ValueError(
                            f"Gemini prompt was blocked; block_reason={reason}."
                        )
                    raise EmptyModelOutputError(
                        f"Gemini response has no candidates; block_reason={reason}."
                    )

                if result.candidates[0].finish_reason == types.FinishReason.RECITATION:
                    current_temperature = (
                        config.temperature if config.temperature is not None else 0.7
                    )
                    if current_temperature >= 2:
                        raise Exception(
                            "Temperature exceeded the maximum value of 2, but Gemini recitation still occurred."
                        )
                    temperature = min(2.0, round(current_temperature + 0.2, 10))
                    logger.warning(
                        f"Gemini recitation detected; increasing temperature to {temperature:.1f} and retrying...",
                    )
                    continue

                break

            except APIError as e:
                if e.message is None:
                    e.message = ""
                if "Developer instruction is not enabled" in e.message:
                    if system_instruction is None:
                        raise
                    logger.warning(
                        f"{model} does not support system prompts; removing it automatically. This may affect persona settings.",
                    )
                    system_instruction = None
                elif "Function calling is not enabled" in e.message:
                    if tools is None:
                        raise
                    logger.warning(
                        f"{model} does not support function calling; removing tools automatically."
                    )
                    tools = None
                elif (
                    "Multi-modal output is not supported" in e.message
                    or "Model does not support the requested response modalities"
                    in e.message
                    or "only supports text output" in e.message
                ):
                    if modalities == ["TEXT"]:
                        raise
                    logger.warning(
                        f"{model} does not support multimodal output; falling back to TEXT modality.",
                    )
                    modalities = ["TEXT"]
                else:
                    raise
                continue

        llm_response = LLMResponse("assistant", id=result.response_id)
        llm_response.raw_completion = result
        llm_response.result_chain = self._process_content_parts(
            result.candidates[0],
            llm_response,
            model=model,
        )
        llm_response.id = result.response_id
        if result.usage_metadata:
            llm_response.usage = self._extract_usage(result.usage_metadata)
            llm_response.provider_state["gemini"]["usage_metadata"] = (
                result.usage_metadata.model_dump(mode="json", exclude_none=True)
            )
        return llm_response

    async def _query_stream(
        self,
        payloads: dict,
        tools: ToolSet | None,
        *,
        request_max_retries: int | None = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Yield display deltas, then one validated complete protocol response.

        Args:
            payloads: Source messages, target model and generation overrides.
            tools: Current tool declarations.
            request_max_retries: Maximum transport attempts, including the first.

        Yields:
            Incremental display responses and a single final response at EOF.

        Raises:
            ValueError: A tool stream is partial, incomplete or blocked.
        """
        system_instruction = next(
            (msg["content"] for msg in payloads["messages"] if msg["role"] == "system"),
            None,
        )
        model = payloads.get("model", self.get_model())
        conversation = await self._prepare_conversation(payloads)
        if not conversation or conversation[-1].role == "model":
            raise ValueError(
                "Gemini request is empty or ends with a model turn; provide actual user input or tool results."
            )

        result = None
        while True:
            try:
                config = await self._prepare_query_config(
                    payloads,
                    tools,
                    payloads.get("tool_choice", "auto"),
                    system_instruction,
                )

                async def open_stream():
                    """Open the request through its first item before any display.

                    Returns:
                        The live iterator and its first response, if any.
                    """
                    stream = await self.client.models.generate_content_stream(
                        model=model,
                        contents=cast(types.ContentListUnion, conversation),
                        config=config,
                    )
                    try:
                        first = await anext(stream, None)
                    except BaseException:
                        if hasattr(stream, "aclose"):
                            await stream.aclose()
                        raise
                    return stream, first

                result, first_chunk = await retry_provider_request(
                    "Gemini",
                    open_stream,
                    max_attempts=request_max_retries,
                )
                break
            except APIError as e:
                if e.message is None:
                    e.message = ""
                if "Developer instruction is not enabled" in e.message:
                    if system_instruction is None:
                        raise
                    logger.warning(
                        f"{model} does not support system prompts; removing it automatically. This may affect persona settings.",
                    )
                    system_instruction = None
                elif "Function calling is not enabled" in e.message:
                    if tools is None:
                        raise
                    logger.warning(
                        f"{model} does not support function calling; removing tools automatically."
                    )
                    tools = None
                else:
                    raise
                continue

        parts: list[types.Part] = []
        usage = None
        response_id = None
        finish_reason = None
        selected_index = None
        try:
            chunk = first_chunk
            while chunk is not None:
                if (
                    chunk.prompt_feedback
                    and chunk.prompt_feedback.block_reason
                    and chunk.prompt_feedback.block_reason
                    != types.BlockedReason.BLOCKED_REASON_UNSPECIFIED
                ):
                    raise ValueError(
                        f"Gemini prompt was blocked; block_reason={chunk.prompt_feedback.block_reason}."
                    )
                if chunk.usage_metadata is not None:
                    usage = (
                        chunk.usage_metadata
                        if usage is None
                        else usage.model_copy(
                            update=chunk.usage_metadata.model_dump(exclude_none=True)
                        )
                    )
                if chunk.response_id:
                    response_id = chunk.response_id
                for candidate in chunk.candidates or []:
                    candidate_index = (
                        candidate.index if candidate.index is not None else 0
                    )
                    if selected_index is None:
                        selected_index = candidate_index
                    if candidate_index != selected_index:
                        continue
                    if candidate.finish_reason is not None:
                        finish_reason = candidate.finish_reason
                    display = []
                    thoughts = []
                    for part in (
                        candidate.content.parts if candidate.content else None
                    ) or []:
                        part = part.model_copy(deep=True)
                        call = part.function_call
                        if call is not None and (
                            getattr(call, "partial_args", None) is not None
                            or getattr(call, "will_continue", None)
                        ):
                            raise ValueError(
                                "Gemini stream contains unsupported partial function arguments; no tools were executed."
                            )
                        if part.text and part.thought:
                            thoughts.append(part.text)
                        elif part.text:
                            display.append(Comp.Plain(part.text))
                        # The SDK yields delta text and complete FC Parts by default.
                        # Only unsigned plain text deltas are concatenated. Signed,
                        # empty-metadata, media and call Parts retain their boundaries.
                        data = part.model_dump(exclude_none=True)
                        previous = (
                            parts[-1].model_dump(exclude_none=True) if parts else {}
                        )
                        if (
                            data.keys() <= {"text", "thought"}
                            and "text" in data
                            and previous.keys() <= {"text", "thought"}
                            and "text" in previous
                            and part.thought == parts[-1].thought
                        ):
                            parts[-1].text = (parts[-1].text or "") + (part.text or "")
                        else:
                            parts.append(part)
                    if display or thoughts:
                        yield LLMResponse(
                            "assistant",
                            is_chunk=True,
                            result_chain=MessageChain(chain=display),
                            reasoning_content="".join(thoughts) or None,
                        )
                chunk = await anext(result, None)
        finally:
            if hasattr(result, "aclose"):
                await result.aclose()
        if finish_reason is None and any(
            part.function_call is not None for part in parts
        ):
            raise ValueError(
                "Gemini tool stream ended without a completion reason; no tools were executed."
            )
        candidate = types.Candidate(
            content=types.Content(role="model", parts=parts),
            finish_reason=finish_reason,
        )
        final_response = LLMResponse("assistant", id=response_id)
        self._process_content_parts(candidate, final_response, model=model)
        final_response.raw_completion = types.GenerateContentResponse(
            candidates=[candidate], response_id=response_id, usage_metadata=usage
        )
        if usage is not None:
            final_response.usage = self._extract_usage(usage)
            final_response.provider_state["gemini"]["usage_metadata"] = (
                usage.model_dump(mode="json", exclude_none=True)
            )
        yield final_response

    async def text_chat(
        self,
        prompt=None,
        session_id=None,
        image_urls=None,
        audio_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        tool_choice: Literal["auto", "required"] = "auto",
        request_max_retries: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        if contexts is None:
            contexts = []
        new_record = None
        if prompt is not None:
            new_record = await self.assemble_context(
                prompt or "",
                image_urls,
                audio_urls,
                extra_user_content_parts,
            )
        context_query = self._ensure_message_to_dicts(contexts)
        if new_record:
            context_query.append(new_record)
        if system_prompt:
            context_query.insert(0, {"role": "system", "content": system_prompt})

        for part in context_query:
            if "_no_save" in part:
                del part["_no_save"]

        # tool calls result
        if tool_calls_result:
            if not isinstance(tool_calls_result, list):
                context_query.extend(tool_calls_result.to_openai_messages())
            else:
                for tcr in tool_calls_result:
                    context_query.extend(tcr.to_openai_messages())

        model = model or self.get_model()

        allowed_keys = {
            alias
            for aliases in self.GENERATION_PARAMETERS.values()
            for alias in aliases
        }
        payloads = {
            "messages": context_query,
            "model": model,
            "generation_overrides": {
                key: value for key, value in kwargs.items() if key in allowed_keys
            },
        }
        if func_tool and not func_tool.empty():
            payloads["tool_choice"] = tool_choice

        retry = 10
        keys = self.api_keys.copy()

        for _ in range(retry):
            try:
                return await self._query(
                    payloads,
                    func_tool,
                    request_max_retries=request_max_retries,
                )
            except APIError as e:
                if await self._handle_api_error(e, keys):
                    continue
                break

        raise Exception("Gemini request failed.")

    async def text_chat_stream(
        self,
        prompt=None,
        session_id=None,
        image_urls=None,
        audio_urls=None,
        func_tool=None,
        contexts=None,
        system_prompt=None,
        tool_calls_result=None,
        model=None,
        extra_user_content_parts=None,
        tool_choice: Literal["auto", "required"] = "auto",
        request_max_retries: int | None = None,
        **kwargs,
    ) -> AsyncGenerator[LLMResponse, None]:
        if contexts is None:
            contexts = []
        new_record = None
        if prompt is not None:
            new_record = await self.assemble_context(
                prompt or "",
                image_urls,
                audio_urls,
                extra_user_content_parts,
            )
        context_query = self._ensure_message_to_dicts(contexts)
        if new_record:
            context_query.append(new_record)
        if system_prompt:
            context_query.insert(0, {"role": "system", "content": system_prompt})

        for part in context_query:
            if "_no_save" in part:
                del part["_no_save"]

        # tool calls result
        if tool_calls_result:
            if not isinstance(tool_calls_result, list):
                context_query.extend(tool_calls_result.to_openai_messages())
            else:
                for tcr in tool_calls_result:
                    context_query.extend(tcr.to_openai_messages())

        model = model or self.get_model()

        allowed_keys = {
            alias
            for aliases in self.GENERATION_PARAMETERS.values()
            for alias in aliases
        }
        payloads = {
            "messages": context_query,
            "model": model,
            "generation_overrides": {
                key: value for key, value in kwargs.items() if key in allowed_keys
            },
        }
        if func_tool and not func_tool.empty():
            payloads["tool_choice"] = tool_choice

        retry = 10
        keys = self.api_keys.copy()

        for _ in range(retry):
            stream_started = False
            try:
                async for response in self._query_stream(
                    payloads,
                    func_tool,
                    request_max_retries=request_max_retries,
                ):
                    stream_started = True
                    yield response
                break
            except APIError as e:
                if stream_started:
                    raise
                if await self._handle_api_error(e, keys):
                    continue
                raise
        else:
            raise RuntimeError("Gemini streaming request exhausted its retry limit.")

    async def get_models(self):
        try:
            models = await retry_provider_request(
                "Gemini",
                lambda: self.client.models.list(),
            )
            return [
                m.name.replace("models/", "")
                for m in models
                if m.supported_actions
                and "generateContent" in m.supported_actions
                and m.name
            ]
        except APIError as e:
            raise Exception(f"Failed to fetch Gemini model list: {e.message}")

    def get_current_key(self) -> str:
        return self.chosen_api_key

    def get_keys(self) -> list[str]:
        return self.api_keys

    def set_key(self, key) -> None:
        self.chosen_api_key = key
        self._init_client()

    async def assemble_context(
        self,
        text: str,
        image_urls: list[str] | None = None,
        audio_urls: list[str] | None = None,
        extra_user_content_parts: list[ContentPart] | None = None,
    ):
        """组装上下文。"""

        async def resolve_image_part(image_url: str) -> dict | None:
            image_data = await resolve_media_ref_to_base64_data(
                image_url,
                media_type="image",
            )
            if not image_data:
                logger.warning("Image preprocessing returned no data; ignoring it.")
                return None
            return {
                "type": "image_url",
                "image_url": {"url": image_data.to_data_url()},
            }

        async def resolve_audio_part(audio_path: str) -> dict | None:
            try:
                audio_data = await resolve_media_ref_to_base64_data(
                    audio_path,
                    media_type="audio",
                    strict=True,
                )
            except Exception as exc:
                logger.warning(
                    "Audio preprocessing failed; ignoring it. Error: %s", exc
                )
                return None

            if not audio_data:
                logger.warning("Audio preprocessing returned no data; ignoring it.")
                return None
            return {
                "type": "audio_url",
                "audio_url": {"url": audio_data.to_data_url()},
            }

        # 构建内容块列表
        content_blocks = []

        # 1. 用户原始发言（OpenAI 建议：用户发言在前）
        if text:
            content_blocks.append({"type": "text", "text": text})
        elif image_urls:
            # 如果没有文本但有图片，添加占位文本
            content_blocks.append({"type": "text", "text": "[Image]"})
        elif audio_urls:
            content_blocks.append({"type": "text", "text": "[Audio]"})
        elif extra_user_content_parts:
            # 如果只有额外内容块，也需要添加占位文本
            content_blocks.append({"type": "text", "text": " "})

        # 2. 额外的内容块（系统提醒、指令等）
        if extra_user_content_parts:
            for part in extra_user_content_parts:
                if isinstance(part, TextPart):
                    content_blocks.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageURLPart):
                    image_part = await resolve_image_part(part.image_url.url)
                    if image_part:
                        content_blocks.append(image_part)
                elif isinstance(part, AudioURLPart):
                    audio_part = await resolve_audio_part(part.audio_url.url)
                    if audio_part:
                        content_blocks.append(audio_part)
                else:
                    raise ValueError(
                        f"Unsupported extra content part type: {type(part)}"
                    )

        # 3. 图片内容
        if image_urls:
            for image_url in image_urls:
                image_part = await resolve_image_part(image_url)
                if image_part:
                    content_blocks.append(image_part)

        if audio_urls:
            for audio_path in audio_urls:
                audio_part = await resolve_audio_part(audio_path)
                if audio_part:
                    content_blocks.append(audio_part)

        # 如果只有主文本且没有额外内容块和图片，返回简单格式以保持向后兼容
        if (
            text
            and not extra_user_content_parts
            and not image_urls
            and not audio_urls
            and len(content_blocks) == 1
            and content_blocks[0]["type"] == "text"
        ):
            return {"role": "user", "content": content_blocks[0]["text"]}

        # 否则返回多模态格式
        return {"role": "user", "content": content_blocks}

    async def encode_image_bs64(self, image_url: str) -> str:
        """将图片转换为 base64"""
        image_data = await resolve_media_ref_to_base64_data(
            image_url,
            media_type="image",
            strict=True,
        )
        if image_data is None:
            raise RuntimeError(
                f"Failed to encode image data: {describe_media_ref(image_url)}"
            )
        return image_data.to_data_url()

    async def _close_httpx_client(self, client: httpx.AsyncClient | None) -> None:
        """Safely close an httpx.AsyncClient, swallowing errors for idempotency."""
        if client is None:
            return
        try:
            await client.aclose()
        except Exception as e:
            # Idempotent: ignore errors from already-closed or broken clients,
            # but log at debug to aid diagnosing unexpected shutdown issues.
            logger.debug(f"[Gemini] Ignored error while closing httpx client: {e}")

    async def terminate(self) -> None:
        # Close the active Gemini client (external httpx client is managed
        # separately so genai.Client.aclose skips it).
        if self.client is not None:
            try:
                await self.client.aclose()
            except Exception:
                pass
            self.client = None

        # Close all tracked httpx clients (stale + current).
        for client in self._stale_http_clients:
            await self._close_httpx_client(client)
        self._stale_http_clients.clear()
        await self._close_httpx_client(self._http_client)
        self._http_client = None
