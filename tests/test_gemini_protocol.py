"""Gemini protocol tests through the real SDK HTTP transport (no live API)."""

import base64
import copy
import io
import json
from collections import Counter

import httpx
import pytest
import pytest_asyncio
from google import genai
from google.genai import types
from mcp.types import (
    AudioContent,
    CallToolResult,
    ImageContent,
    ResourceLink,
    TextContent,
)
from PIL import Image

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import Message, dump_messages_with_checkpoints
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI


@pytest_asyncio.fixture(params=[False, True], ids=["developer", "vertex"])
async def sdk_provider(request):
    requests = []
    replies = []

    def handle(request):
        requests.append(json.loads(request.content))
        reply = (
            replies.pop(0)
            if replies
            else {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "done"}]},
                        "finishReason": "STOP",
                    }
                ]
            }
        )
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, list):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in reply),
            )
        return httpx.Response(200, json=reply)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
        options = {"api_key": "offline-test"}
        if request.param:
            from google.auth.credentials import AnonymousCredentials

            credentials = AnonymousCredentials()
            credentials.token = "offline-token"
            options = {
                "vertexai": True,
                "project": "offline-project",
                "location": "global",
                "credentials": credentials,
            }
        client = genai.Client(
            **options,
            http_options=types.HttpOptions(
                httpx_async_client=transport,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        provider = ProviderGoogleGenAI.__new__(ProviderGoogleGenAI)
        provider.provider_config = {"id": "offline", "model": "gemini-2.5-flash"}
        provider.provider_settings = {}
        provider.model_name = "gemini-2.5-flash"
        provider.api_base = (
            "https://aiplatform.googleapis.com"
            if request.param
            else "https://generativelanguage.googleapis.com"
        )
        provider.api_keys = ["offline-test"]
        provider.chosen_api_key = "offline-test"
        provider.safety_settings = []
        provider.client = client.aio
        yield provider, requests, replies
        await client.aio.aclose()
        client.close()


@pytest.mark.asyncio
async def test_b_unusable_native_batch_is_rejected_before_side_effects(sdk_provider):
    provider, _, replies = sdk_provider
    provider.model_name = "gemini-3.1-pro-preview"
    replies.append(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "one", "args": {}}}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
    )
    executor = ResultExecutor([CallToolResult(content=[])])
    runner = await reset_runner(provider, executor)
    _ = [event async for event in runner.step_until_done(2)]
    assert executor.calls == []


@pytest.mark.asyncio
async def test_b_duplicate_native_ids_in_one_response_are_diagnostic(sdk_provider):
    provider, _, _ = sdk_provider
    with pytest.raises(ValueError, match="duplicate native"):
        provider._process_content_parts(
            types.Candidate(
                content=types.ModelContent(
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="one", id="duplicate", args={}
                            )
                        )
                        for _ in range(2)
                    ]
                ),
                finish_reason="STOP",
            ),
            LLMResponse("assistant"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("new_turn", [False, True])
async def test_m_legacy_signature_checks_are_scoped_to_current_turn(
    sdk_provider, new_turn
):
    provider, requests, _ = sdk_provider
    provider.model_name = "gemini-3.1-pro-preview"
    history = tool_history()
    if new_turn:
        history.append({"role": "user", "content": "new user turn"})
        await provider.text_chat(contexts=history)
        assert "skip_thought_signature_validator" not in json.dumps(requests)
    else:
        with pytest.raises(ValueError, match="signature"):
            await provider.text_chat(contexts=history)
        assert not requests


@pytest.mark.asyncio
async def test_b_roleless_sdk_content_is_still_model_owned(sdk_provider):
    provider, requests, _ = sdk_provider
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(content=types.Content(parts=[types.Part(text="answer")])),
        response,
    )
    await provider.text_chat(
        contexts=[
            Message(role="user", content="start"),
            response.to_assistant_message(),
            Message(role="user", content="continue"),
        ]
    )
    assert requests[-1]["contents"][1]["role"] == "model"


@pytest.mark.asyncio
async def test_r_mixed_batch_counts_and_orders_every_result(sdk_provider):
    provider, _, _ = sdk_provider

    class BatchExecutor(ResultExecutor):
        async def execute(self, tool, run_context, **kwargs):
            self.calls.append(tool.name)
            content = (
                TextContent(type="text", text="known")
                if len(self.calls) == 1
                else AudioContent(type="audio", data="AA==", mimeType="audio/wav")
            )
            yield CallToolResult(content=[content])

    executor = BatchExecutor([])
    runner = await reset_runner(provider, executor)
    response = LLMResponse(
        "tool",
        tools_call_name=["one", "one"],
        tools_call_args=[{}, {}],
        tools_call_ids=["a", "b"],
    )
    events = [
        event async for event in runner._handle_function_tools(runner.req, response)
    ]
    results = [
        block for event in events for block in event.tool_call_result_blocks or []
    ]
    assert [result.tool_call_id for result in results] == ["a", "b"]
    assert Counter(result.tool_call_id for result in results) == Counter(
        response.tools_call_ids
    )
    assert results[0].content == "known"
    assert "adapter" in results[1].content
    assert executor.calls == ["one", "one"]


@pytest.mark.asyncio
async def test_h_model_failure_fallback_does_not_repeat_completed_tool(sdk_provider):
    from test_tool_loop_agent_runner import MockProvider

    provider, _, replies = sdk_provider
    replies.extend(
        [
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"functionCall": {"name": "one", "args": {}}}],
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
            httpx.ConnectError("offline failure after tool"),
        ]
    )
    executor = ResultExecutor(
        [CallToolResult(content=[TextContent(type="text", text="completed")])]
    )
    runner = await reset_runner(provider, executor)
    fallback = MockProvider()
    fallback.should_call_tools = False
    runner.fallback_providers = [fallback]
    runner.request_max_retries = 1
    _ = [event async for event in runner.step_until_done(3)]
    assert len(executor.calls) == 1
    assert fallback.call_count == 1


@pytest.mark.asyncio
async def test_h_expired_legacy_media_fails_without_replaying_other_content(
    sdk_provider, tmp_path
):
    provider, requests, _ = sdk_provider
    contexts = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": str(tmp_path / "expired.png")},
                }
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    before = copy.deepcopy(contexts)
    with pytest.raises((ValueError, OSError)):
        await provider.text_chat(contexts=contexts)
    assert contexts == before and not requests


@pytest.mark.asyncio
async def test_c_signed_model_media_never_runs_input_transcoding(
    sdk_provider, monkeypatch
):
    provider, requests, _ = sdk_provider
    original = types.ModelContent(
        parts=[
            types.Part.from_bytes(
                data=base64.b64decode(png_data()), mime_type="image/png"
            )
        ]
    )
    original.parts[0].thought_signature = b"\xff\x01"
    response = LLMResponse("assistant")
    provider._process_content_parts(types.Candidate(content=original), response)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Model media must not enter the input resolver")

    monkeypatch.setattr(
        "astrbot.core.provider.sources.gemini_source.resolve_media_ref_to_base64_data",
        forbidden,
    )
    await provider.text_chat(
        contexts=[
            Message(role="user", content="start"),
            response.to_assistant_message(),
            Message(role="user", content="continue"),
        ]
    )
    actual = types.Content.model_validate(requests[-1]["contents"][1])
    assert actual.parts == original.parts


@pytest.mark.asyncio
async def test_b_edited_call_arguments_fail_before_execution(sdk_provider):
    provider, _, _ = sdk_provider
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(
            content=types.ModelContent(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="one", args={"x": 1})
                    )
                ]
            ),
            finish_reason="STOP",
        ),
        response,
    )
    response.tools_call_args[0] = {"x": 2}
    executor = ResultExecutor([CallToolResult(content=[])])
    runner = await reset_runner(provider, executor)
    with pytest.raises(ValueError, match="state|binding"):
        _ = [
            event async for event in runner._handle_function_tools(runner.req, response)
        ]
    assert executor.calls == []


@pytest.mark.asyncio
async def test_h_opaque_state_corruption_is_not_persisted(sdk_provider):
    provider, _, _ = sdk_provider
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(
            content=types.ModelContent(
                parts=[types.Part(text="answer", thought_signature=b"original")]
            )
        ),
        response,
    )
    response.provider_state["gemini"]["content"]["parts"][0]["thoughtSignature"] = (
        "Y2hhbmdlZA=="
    )
    saved = dump_messages_with_checkpoints([response.to_assistant_message()])
    assert saved[0]["provider_state"]["gemini"].get("invalidated") is True


@pytest.mark.asyncio
async def test_c_partial_usage_never_produces_negative_input(sdk_provider):
    provider, _, _ = sdk_provider
    usage = provider._extract_usage(
        types.GenerateContentResponseUsageMetadata(cached_content_token_count=5)
    )
    assert usage.input_other == 0 and usage.input_cached == 5


@pytest.mark.asyncio
async def test_s_partial_usage_tail_preserves_prior_fields(sdk_provider):
    provider, _, replies = sdk_provider
    replies.append(
        [
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "answer"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "cachedContentTokenCount": 2,
                    "candidatesTokenCount": 5,
                },
            },
            {"usageMetadata": {"candidatesTokenCount": 6}},
        ]
    )
    response = [item async for item in provider.text_chat_stream(prompt="test")][-1]
    assert response.usage.input_other == 8
    assert response.usage.input_cached == 2 and response.usage.output == 6


@pytest.mark.asyncio
async def test_h_provenance_contains_no_endpoint_credentials(sdk_provider):
    provider, _, _ = sdk_provider
    provider.api_base = (
        "https://user:private-password@gateway.invalid/api?token=private-token"
    )
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(content=types.ModelContent(parts=[types.Part(text="answer")])),
        response,
    )
    saved = response.to_assistant_message().model_dump_json()
    assert "private-password" not in saved and "private-token" not in saved


@pytest.mark.asyncio
async def test_c_flash_preserves_existing_disabled_thinking_default(sdk_provider):
    provider, _, _ = sdk_provider
    config = await provider._prepare_query_config({"model": "gemini-2.5-flash"})
    assert config.thinking_config.thinking_budget == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", [False, True])
async def test_b_legacy_duplicate_ids_require_original_order(sdk_provider, ambiguous):
    provider, requests, _ = sdk_provider
    calls = [
        {
            "id": call_id,
            "function": {"name": "one", "arguments": json.dumps({"x": index})},
        }
        for index, call_id in enumerate(["legacy", "legacy", "unique"])
    ]
    results = [
        {"role": "tool", "tool_call_id": call_id, "content": str(index)}
        for index, call_id in enumerate(["legacy", "legacy", "unique"])
    ]
    if ambiguous:
        results = [results[2], results[0], results[1]]
    contexts = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "tool_calls": calls},
        *results,
    ]
    if ambiguous:
        with pytest.raises(ValueError, match="ambiguous legacy"):
            await provider.text_chat(contexts=contexts)
        assert not requests
    else:
        await provider.text_chat(contexts=contexts)
        responses = [
            part["functionResponse"]
            for part in wire_contents(requests[-1])[-1]["parts"]
        ]
        assert [response["name"] for response in responses] == ["one"] * 3
        assert [response["response"]["content"] for response in responses] == [
            "0",
            "1",
            "2",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "skills_like"])
async def test_s_runner_streaming_history_replays_actual_execution(sdk_provider, mode):
    provider, requests, replies = sdk_provider
    first = [
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "preface"}]}}
            ]
        },
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {"name": "one", "args": {"x": 1}},
                                "thoughtSignature": "Zmlyc3Q=",
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        },
    ]
    replies.append(first)
    if mode == "skills_like":
        replies.append(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "execution"},
                                {
                                    "functionCall": {"name": "one", "args": {"x": 2}},
                                    "thoughtSignature": "c2Vjb25k",
                                },
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        )
    replies.append(
        [
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "final"}]},
                        "finishReason": "STOP",
                    }
                ]
            }
        ]
    )
    executor = ResultExecutor(
        [CallToolResult(content=[TextContent(type="text", text="result")])]
    )
    runner = await reset_runner(provider, executor)
    runner.streaming = True
    runner.tool_schema_mode = mode
    if mode == "skills_like":
        runner._skill_like_raw_tool_set = runner.req.func_tool
        runner._tool_schema_param_set = runner.req.func_tool
    events = [event async for event in runner.step_until_done(2)]
    assert (
        "".join(
            event.data["chain"].get_plain_text()
            for event in events
            if event.type == "streaming_delta"
        )
        == "prefacefinal"
    )
    assert executor.calls == [("one", {"x": 2 if mode == "skills_like" else 1})]
    saved = json.loads(
        json.dumps(dump_messages_with_checkpoints(runner.run_context.messages))
    )
    assert saved[1]["content"][0]["text"] == "preface"
    await provider.text_chat(
        contexts=[*saved, Message(role="user", content="continue")]
    )
    calls = [
        part
        for content in wire_contents(requests[-1])
        for part in content["parts"]
        if "functionCall" in part
    ]
    assert len(calls) == 1
    assert calls[0]["thoughtSignature"] == (
        "c2Vjb25k" if mode == "skills_like" else "Zmlyc3Q="
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("blocked", ["prompt", "candidate"])
async def test_c_blocked_responses_are_not_empty_output_retries(
    sdk_provider, streaming, blocked
):
    provider, _, replies = sdk_provider
    reply = (
        {"promptFeedback": {"blockReason": "SAFETY"}}
        if blocked == "prompt"
        else {"candidates": [{"finishReason": "SAFETY"}]}
    )
    replies.append([reply] if streaming else reply)
    with pytest.raises(ValueError, match="SAFETY"):
        if streaming:
            _ = [result async for result in provider.text_chat_stream(prompt="test")]
        else:
            await provider.text_chat(prompt="test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,budget",
    [
        ("gemini-2.5-flash", -2),
        ("gemini-2.5-flash", 24577),
        ("gemini-2.5-flash-lite", 128),
        ("gemini-2.5-flash", "invalid"),
    ],
)
async def test_c_known_thinking_budget_rejects_invalid_explicit_values(
    sdk_provider, model, budget
):
    provider, _, _ = sdk_provider
    provider.provider_config["gm_thinking_config"] = {"budget": budget}
    with pytest.raises(ValueError, match="thinking budget"):
        await provider._prepare_query_config({"model": model})


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "assistant"])
async def test_c_unknown_parts_are_not_reinterpreted_as_audio_or_text(
    sdk_provider, role
):
    provider, requests, _ = sdk_provider
    contexts = [
        {"role": "user", "content": "start"},
        {
            "role": role,
            "content": [
                {"type": "future", "audio_url": {"url": "data:audio/wav;base64,AA=="}}
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    with pytest.raises(ValueError, match="unsupported content type"):
        await provider.text_chat(contexts=contexts)
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,reason",
    [
        ("regular", "system"),
        ("regular", "tools"),
        ("regular", "modalities"),
        ("stream", "system"),
        ("stream", "tools"),
    ],
)
async def test_c_repeated_feature_errors_stop_after_one_adaptation(
    sdk_provider, mode, reason
):
    from google.genai.errors import APIError
    from unittest.mock import AsyncMock

    provider, _, _ = sdk_provider
    message = {
        "system": "Developer instruction is not enabled",
        "tools": "Function calling is not enabled",
        "modalities": "Multi-modal output is not supported",
    }[reason]
    request = AsyncMock(
        side_effect=[
            APIError(400, {"error": {"message": message}}),
            APIError(400, {"error": {"message": message}}),
            AssertionError("unbounded adaptation"),
        ]
    )
    provider.provider_config["gm_resp_image_modal"] = True
    tools = ToolSet(
        [FunctionTool(name="one", description="test", parameters={"type": "object"})]
    )
    payload = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "test"},
        ]
    }
    with pytest.raises(APIError):
        if mode == "regular":
            provider.client.models.generate_content = request
            await provider._query(payload, tools, request_max_retries=1)
        else:
            provider.client.models.generate_content_stream = request
            _ = [
                item
                async for item in provider._query_stream(
                    payload, tools, request_max_retries=1
                )
            ]
    assert request.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_c_production_sdk_retry_composition_honors_attempt_limit(
    monkeypatch, streaming
):
    from google.genai.errors import APIError
    from unittest.mock import AsyncMock

    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            503, json={"error": {"code": 503, "message": "offline overload"}}
        )

    original_client = genai.Client
    clients = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:

        def client_factory(*args, **kwargs):
            kwargs["http_options"].httpx_async_client = transport
            client = original_client(*args, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(
            "astrbot.core.provider.sources.gemini_source.genai.Client", client_factory
        )
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        provider = ProviderGoogleGenAI(
            {
                "id": "offline",
                "key": ["offline-key"],
                "model": "gemini-2.5-flash",
                "api_base": "https://offline.invalid",
            },
            {},
        )
        try:
            with pytest.raises(APIError):
                if streaming:
                    _ = [
                        item
                        async for item in provider.text_chat_stream(
                            prompt="test", request_max_retries=2
                        )
                    ]
                else:
                    await provider.text_chat(prompt="test", request_max_retries=2)
            assert len(requests) == 2
        finally:
            await provider.client.aclose()
            await provider._http_client.aclose()
            for client in clients:
                client.close()


def wire_contents(request):
    """Normalize only known protobuf aliases, retaining every field and value.

    SDK 1.56 Vertex serializes several Part fields in snake_case; newer paths
    use camelCase. No content, arguments, response data or unknown key is dropped.
    The captured request itself remains the actual HTTP JSON.
    """
    aliases = {
        "function_call": "functionCall",
        "function_response": "functionResponse",
        "thought_signature": "thoughtSignature",
        "inline_data": "inlineData",
        "mime_type": "mimeType",
        "file_data": "fileData",
        "file_uri": "fileUri",
    }

    def normalize(value):
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {
                aliases.get(key, key): item
                if key in {"args", "response"}
                else normalize(item)
                for key, item in value.items()
            }
        return value

    return normalize(request["contents"])


def tool_history():
    return [
        {"role": "user", "content": "compare"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": name,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
                for name in ("one", "two")
            ],
        },
        *[
            {"role": "tool", "tool_call_id": name, "content": name}
            for name in ("one", "two")
        ],
    ]


def png_data():
    buffer = io.BytesIO()
    with Image.new("RGB", (2, 2), "blue") as image:
        image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["text", "image", "requery"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_a_sdk_preserves_tool_content_boundary(sdk_provider, tail, streaming):
    provider, requests, replies = sdk_provider
    contexts = tool_history()
    if tail == "requery":
        runner = ToolLoopAgentRunner()
        runner.run_context = ContextWrapper(context=None)
        runner.run_context.messages = [
            Message.model_validate(message) for message in contexts
        ]
        contexts = runner._build_tool_requery_context(["one"])
    else:
        content = (
            "follow up"
            if tail == "text"
            else [
                {"type": "text", "text": "look"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{png_data()}"},
                },
            ]
        )
        contexts.append({"role": "user", "content": content})
    before = copy.deepcopy(contexts)
    if streaming:
        replies.append(
            [
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "done"}]},
                            "finishReason": "STOP",
                        }
                    ]
                }
            ]
        )
        assert [r async for r in provider.text_chat_stream(contexts=contexts)]
    else:
        await provider.text_chat(contexts=contexts)
    await provider._prepare_conversation({"messages": contexts})
    assert contexts == before
    contents = wire_contents(requests[0])
    assert [c["role"] for c in contents] == ["user", "model", "user", "user"]
    assert len(contents[-2]["parts"]) == 2
    assert all("functionResponse" in part for part in contents[-2]["parts"])
    assert all("functionResponse" not in part for part in contents[-1]["parts"])
    if tail == "image":
        assert contents[-1]["parts"][1]["inlineData"]["data"] == png_data()


@pytest.mark.asyncio
async def test_b_parallel_and_multistep_calls_execute_once_in_order(sdk_provider):
    provider, requests, replies = sdk_provider
    for offset in (0, 2):
        replies.append(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "one",
                                        "id": f"native-{i}",
                                        "args": {"x": offset + i},
                                    },
                                    "thoughtSignature": base64.b64encode(
                                        bytes([255, offset + i])
                                    ).decode(),
                                }
                                for i in (0, 1)
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        )
    executor = ResultExecutor(
        [CallToolResult(content=[TextContent(type="text", text="result")])]
    )
    runner = await reset_runner(provider, executor)
    _ = [event async for event in runner.step_until_done(3)]
    assert [args["x"] for _, args in executor.calls] == [0, 1, 2, 3]
    messages = runner.run_context.messages
    call_ids = [call.id for message in messages for call in message.tool_calls or []]
    result_ids = [
        message.tool_call_id for message in messages if message.role == "tool"
    ]
    assert len(set(call_ids)) == 4
    assert Counter(call_ids) == Counter(result_ids)
    calls = [
        p
        for c in wire_contents(requests[-1])
        for p in c["parts"]
        if "functionCall" in p
    ]
    assert [base64.urlsafe_b64decode(p["thoughtSignature"]) for p in calls] == [
        bytes([255, i]) for i in range(4)
    ]
    assert [p["functionCall"]["id"] for p in calls] == ["native-0", "native-1"] * 2


@pytest.mark.asyncio
async def test_h_two_sessions_do_not_share_call_bindings(sdk_provider):
    import asyncio

    provider, requests, _ = sdk_provider

    async def run_one(value):
        response = LLMResponse("assistant")
        provider._process_content_parts(
            types.Candidate(
                content=types.ModelContent(
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="one", args={"value": value}
                            )
                        )
                    ]
                ),
                finish_reason="STOP",
            ),
            response,
        )
        await provider.text_chat(
            contexts=[
                Message(role="user", content=f"session-{value}"),
                response.to_assistant_message(),
                Message(
                    role="tool",
                    tool_call_id=response.tools_call_ids[0],
                    content=f"result-{value}",
                ),
            ]
        )
        return response.tools_call_ids[0]

    ids = await asyncio.gather(run_one(1), run_one(2))
    assert ids[0] != ids[1]
    for request in requests:
        content = wire_contents(request)
        value = content[1]["parts"][0]["functionCall"]["args"]["value"]
        assert (
            content[2]["parts"][0]["functionResponse"]["response"]["content"]
            == f"result-{value}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["network", "cancelled"])
async def test_k_failed_requery_does_not_execute_candidate(sdk_provider, error):
    import asyncio

    provider, _, replies = sdk_provider
    replies.extend(
        [
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "first"},
                                {"functionCall": {"name": "one", "args": {}}},
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
            httpx.ConnectError("offline")
            if error == "network"
            else asyncio.CancelledError(),
        ]
    )
    executor = ResultExecutor([CallToolResult(content=[])])
    runner = await reset_runner(provider, executor)
    runner.request_max_retries = 1
    runner.tool_schema_mode = "skills_like"
    runner._skill_like_raw_tool_set = runner.req.func_tool
    runner._tool_schema_param_set = runner.req.func_tool
    with pytest.raises((httpx.ConnectError, asyncio.CancelledError)):
        _ = [event async for event in runner.step_until_done(2)]
    assert executor.calls == []
    saved = dump_messages_with_checkpoints(runner.run_context.messages)
    assert any(
        message["role"] == "assistant"
        and any(
            part.get("text") == "first"
            for part in message.get("content") or []
            if isinstance(part, dict)
        )
        for message in saved
    )
    assert not any(message.get("tool_calls") for message in saved)
    await provider.text_chat(
        contexts=[*saved, Message(role="user", content="continue")]
    )


@pytest.mark.asyncio
async def test_k_other_providers_keep_matching_first_thinking_signature(sdk_provider):
    from test_tool_loop_agent_runner import MockProvider
    from unittest.mock import AsyncMock

    provider = MockProvider()
    provider.text_chat = AsyncMock(
        side_effect=[
            LLMResponse(
                "tool",
                completion_text="first",
                reasoning_content="first reasoning",
                reasoning_signature="first signature",
                tools_call_name=["one"],
                tools_call_args=[{}],
                tools_call_ids=["first"],
            ),
            LLMResponse(
                "tool",
                completion_text="second",
                reasoning_content="second reasoning",
                reasoning_signature="second signature",
                tools_call_name=["one"],
                tools_call_args=[{}],
                tools_call_ids=["second"],
            ),
            LLMResponse("assistant", completion_text="done"),
        ]
    )
    runner = await reset_runner(provider, ResultExecutor([CallToolResult(content=[])]))
    runner.tool_schema_mode = "skills_like"
    runner._skill_like_raw_tool_set = runner.req.func_tool
    runner._tool_schema_param_set = runner.req.func_tool
    _ = [event async for event in runner.step_until_done(2)]
    thought = runner.run_context.messages[1].content[0]
    assert thought.think == "first reasoning" and thought.encrypted == "first signature"


@pytest.mark.asyncio
async def test_c_key_rotation_logs_no_key_prefix(sdk_provider, monkeypatch, caplog):
    from google.genai.errors import APIError
    from unittest.mock import AsyncMock

    provider, _, _ = sdk_provider
    provider.chosen_api_key = "offline-sensitive-first"
    monkeypatch.setattr(
        provider, "set_key", lambda key: setattr(provider, "chosen_api_key", key)
    )
    monkeypatch.setattr(
        "astrbot.core.provider.sources.gemini_source.asyncio.sleep", AsyncMock()
    )
    await provider._handle_api_error(
        APIError(429, {"error": {"message": "rate limited"}}),
        ["offline-sensitive-first", "offline-sensitive-second"],
    )
    assert "offline-sens" not in caplog.text


class ResultExecutor:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def execute(self, tool, run_context, **kwargs):
        self.calls.append((tool.name, kwargs))
        for result in self.results:
            if isinstance(result, BaseException):
                raise result
            yield result


async def reset_runner(provider, executor):
    runner = ToolLoopAgentRunner()
    tools = ToolSet(
        [
            FunctionTool(
                name="one",
                description="test",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )
    await runner.reset(
        provider=provider,
        request=ProviderRequest(prompt="test", func_tool=tools),
        run_context=ContextWrapper(context=None),
        tool_executor=executor,
        agent_hooks=BaseAgentRunHooks(),
        streaming=False,
    )
    return runner


@pytest.mark.asyncio
async def test_a_runner_image_and_max_steps_reach_sdk(
    sdk_provider, monkeypatch, tmp_path
):
    provider, requests, replies = sdk_provider
    monkeypatch.setattr(
        "astrbot.core.agent.tool_image_cache.tool_image_cache._cache_dir", str(tmp_path)
    )
    replies.append(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "one", "args": {}}}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
    )
    executor = ResultExecutor(
        [
            CallToolResult(
                content=[
                    ImageContent(type="image", data=png_data(), mimeType="image/png")
                ]
            )
        ]
    )
    runner = await reset_runner(provider, executor)
    assert [event async for event in runner.step_until_done(1)]
    assert len(executor.calls) == 1
    assert len(requests) == 2
    contents = wire_contents(requests[-1])
    assert any("functionResponse" in p for p in contents[-2]["parts"])
    assert any("inlineData" in p for p in contents[-1]["parts"])
    assert all("functionResponse" not in p for p in contents[-1]["parts"])
    assert not requests[-1]["generationConfig"].get("tools")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["audio", "resource_link", "mixed", "empty", "exhausted", "error"]
)
async def test_r_tool_result_has_exactly_one_final_response(sdk_provider, kind):
    provider, _, _ = sdk_provider
    audio = AudioContent(type="audio", data="AA==", mimeType="audio/wav")
    link = ResourceLink(
        type="resource_link", name="test", uri="https://example.invalid/private"
    )
    content = {
        "audio": [audio],
        "resource_link": [link],
        "mixed": [TextContent(type="text", text="known"), audio],
        "empty": [],
    }
    results = (
        []
        if kind == "exhausted"
        else [RuntimeError("executor failed")]
        if kind == "error"
        else [CallToolResult(content=content[kind])]
    )
    executor = ResultExecutor(results)
    runner = await reset_runner(provider, executor)
    response = LLMResponse(
        role="tool",
        tools_call_name=["one"],
        tools_call_args=[{}],
        tools_call_ids=["call-one"],
    )
    events = [
        event async for event in runner._handle_function_tools(runner.req, response)
    ]
    blocks = [
        block for event in events for block in (event.tool_call_result_blocks or [])
    ]
    assert Counter(block.tool_call_id for block in blocks) == Counter({"call-one": 1})
    assert len(executor.calls) == 1
    text = blocks[0].content
    if kind in {"audio", "resource_link", "mixed"}:
        assert "adapter" in text and "unsupported" in text
    if kind == "mixed":
        assert "known" in text
    if kind == "exhausted":
        assert "without returning" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_ids", [[None, None], ["one", "second"], ["upstream-one", "upstream-two"]]
)
async def test_b_call_identity_and_native_parts_roundtrip(sdk_provider, native_ids):
    provider, requests, _ = sdk_provider
    parts = [
        {"text": "before"},
        *[
            {
                "functionCall": {
                    "name": "one",
                    "args": {},
                    **({"id": call_id} if call_id is not None else {}),
                },
                **(
                    {
                        "thoughtSignature": base64.b64encode(
                            b"\xff\x00" + bytes([index])
                        ).decode()
                    }
                    if index == 0
                    else {}
                ),
            }
            for index, call_id in enumerate(native_ids)
        ],
    ]
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(
            content=types.Content.model_validate({"role": "model", "parts": parts}),
            finish_reason="STOP",
        ),
        response,
    )
    assert len(set(response.tools_call_ids)) == 2
    assert response.reasoning_signature is None
    message = response.to_assistant_message()
    restored = Message.model_validate(json.loads(message.model_dump_json()))
    contexts = [
        {"role": "user", "content": "start"},
        restored.model_dump(),
        *[
            {"role": "tool", "tool_call_id": call_id, "content": str(index)}
            for index, call_id in reversed(list(enumerate(response.tools_call_ids)))
        ],
    ]
    before = copy.deepcopy(contexts)
    await provider.text_chat(contexts=contexts)
    assert contexts == before
    actual = wire_contents(requests[-1])
    assert (
        actual[1]["parts"]
        == types.Content.model_validate({"parts": parts}).model_dump(
            mode="json", by_alias=True, exclude_none=True
        )["parts"]
    )
    assert (
        base64.urlsafe_b64decode(actual[1]["parts"][1]["thoughtSignature"])
        == b"\xff\x00\x00"
    )
    results = [p["functionResponse"] for p in actual[2]["parts"]]
    assert [p["response"]["content"] for p in results] == ["0", "1"]
    assert [p.get("id") for p in results] == native_ids
    assert [p["name"] for p in results] == ["one", "one"]
    assert "provider_state" not in json.dumps(requests[-1])


@pytest.mark.asyncio
async def test_b_no_argument_call_is_executable_without_mutating_original(sdk_provider):
    provider, _, _ = sdk_provider
    candidate = types.Candidate(
        content=types.ModelContent(
            parts=[types.Part(function_call=types.FunctionCall(name="one"))]
        )
    )
    response = LLMResponse("assistant")
    provider._process_content_parts(candidate, response)
    assert response.tools_call_args == [{}]
    assert (
        "args"
        not in response.provider_state["gemini"]["content"]["parts"][0]["functionCall"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["orphan", "duplicate", "missing"])
async def test_b_bad_tool_batches_fail_before_http(sdk_provider, fault):
    provider, requests, _ = sdk_provider
    contexts = tool_history()
    if fault == "orphan":
        contexts.insert(1, contexts.pop())
    elif fault == "duplicate":
        contexts.append(copy.deepcopy(contexts[-1]))
    else:
        contexts.pop()
        contexts.append({"role": "user", "content": "continue"})
    with pytest.raises(ValueError, match="[Bb]atch|[Tt]ool result"):
        await provider.text_chat(contexts=contexts)
    assert not requests


@pytest.mark.asyncio
async def test_b_mismatched_lists_fail_before_tool_side_effects(sdk_provider):
    provider, _, _ = sdk_provider
    executor = ResultExecutor([CallToolResult(content=[])])
    runner = await reset_runner(provider, executor)
    response = LLMResponse(
        "tool",
        tools_call_name=["one", "one"],
        tools_call_args=[{}],
        tools_call_ids=["one"],
    )
    with pytest.raises(ValueError, match="length"):
        _ = [r async for r in runner._handle_function_tools(runner.req, response)]
    assert not executor.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["edit", "filter", "version", "model"])
async def test_b_stale_or_incompatible_snapshot_is_not_replayed(sdk_provider, change):
    provider, requests, _ = sdk_provider
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(
            content=types.ModelContent(
                parts=[types.Part(text="private original", thought_signature=b"\xff")]
            )
        ),
        response,
    )
    message = response.to_assistant_message().model_dump()
    if change == "edit":
        message["content"] = [{"type": "text", "text": "redacted"}]
    elif change == "filter":
        message["content"] = []
    elif change == "version":
        message["provider_state"]["gemini"]["version"] = 999
    else:
        provider.model_name = "other-model"
    with pytest.raises(ValueError, match="state|snapshot|model|revision"):
        await provider.text_chat(
            contexts=[
                {"role": "user", "content": "start"},
                message,
                {"role": "user", "content": "continue"},
            ]
        )
    assert not requests


@pytest.mark.asyncio
async def test_s_sdk_stream_collects_parallel_calls_and_usage_tail(sdk_provider):
    provider, _, replies = sdk_provider
    replies.append(
        [
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "before "}]},
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {"name": "one", "args": {"x": 1}},
                                    "thoughtSignature": "/wA=",
                                }
                            ],
                        },
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [
                                {"functionCall": {"name": "one", "args": {"x": 2}}}
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
            {
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "cachedContentTokenCount": 2,
                    "thoughtsTokenCount": 3,
                }
            },
        ]
    )
    responses = [r async for r in provider.text_chat_stream(prompt="start")]
    final = [r for r in responses if not r.is_chunk]
    assert len(final) == 1
    assert final[0].tools_call_args == [{"x": 1}, {"x": 2}]
    assert final[0].completion_text == "before "
    assert final[0].usage.input == 10
    assert final[0].usage.output == 8
    assert len(final[0].provider_state["gemini"]["content"]["parts"]) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("second_id", ["first-id", "second-id"])
@pytest.mark.parametrize("no_tool", [False, True])
async def test_k_runner_preserves_display_but_commits_requery_protocol(
    sdk_provider, second_id, no_tool
):
    provider, requests, replies = sdk_provider

    def reply(text, call_id, value, signature):
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": text},
                            {
                                "functionCall": {
                                    "name": "one",
                                    "id": call_id,
                                    "args": {"x": value},
                                },
                                "thoughtSignature": signature,
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }

    replies.extend(
        [
            reply("first explanation", "first-id", 1, "Zmlyc3Q="),
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": "no tool needed",
                                    "thoughtSignature": "c2Vjb25k",
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
            if no_tool
            else reply("execution text", second_id, 2, "c2Vjb25k"),
        ]
    )
    executor = ResultExecutor(
        [CallToolResult(content=[TextContent(type="text", text="result")])]
    )
    runner = await reset_runner(provider, executor)
    runner.tool_schema_mode = "skills_like"
    runner._skill_like_raw_tool_set = runner.req.func_tool
    runner._tool_schema_param_set = runner.req.func_tool
    events = [event async for event in runner.step_until_done(2)]
    displayed = [
        event.data["chain"].get_plain_text()
        for event in events
        if event.type == "llm_result" and event.data and event.data["chain"]
    ]
    assert sum("first explanation" in text for text in displayed) == 1
    if no_tool:
        assert sum("no tool needed" in text for text in displayed) == 1
    saved = json.loads(
        json.dumps(dump_messages_with_checkpoints(runner.run_context.messages))
    )
    assert "first explanation" in json.dumps(saved)
    assert len(executor.calls) == (0 if no_tool else 1)
    if not no_tool:
        assert executor.calls[0][1] == {"x": 2}
    await provider.text_chat(contexts=saved + [{"role": "user", "content": "continue"}])
    serialized = json.dumps(requests[-1])
    assert "Zmlyc3Q=" not in serialized
    assert "first explanation" not in serialized
    assert "c2Vjb25k" in serialized
    assert "Please" in serialized or "tool" in serialized
    assert events


@pytest.mark.asyncio
async def test_b_multiple_executor_results_aggregate_once(sdk_provider):
    provider, _, _ = sdk_provider
    executor = ResultExecutor(
        [
            CallToolResult(content=[TextContent(type="text", text=text)])
            for text in ["first", "second"]
        ]
    )
    runner = await reset_runner(provider, executor)
    response = LLMResponse(
        "tool",
        tools_call_name=["one"],
        tools_call_args=[{}],
        tools_call_ids=["call-one"],
    )
    events = [r async for r in runner._handle_function_tools(runner.req, response)]
    blocks = [b for r in events for b in (r.tool_call_result_blocks or [])]
    assert len(blocks) == 1
    assert "first" in blocks[0].content and "second" in blocks[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parts,finish",
    [
        ([{"functionCall": {"name": "one", "willContinue": True}}], "STOP"),
        ([{"functionCall": {"name": "one", "args": {}}}], "MAX_TOKENS"),
    ],
)
async def test_s_incomplete_calls_do_not_commit(sdk_provider, parts, finish):
    provider, _, replies = sdk_provider
    replies.append(
        [
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": parts},
                        "finishReason": finish,
                    }
                ]
            }
        ]
    )
    with pytest.raises(ValueError):
        _ = [r async for r in provider.text_chat_stream(prompt="start")]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_c_generation_kwargs_reach_sdk_with_zero_values(sdk_provider, streaming):
    provider, requests, replies = sdk_provider
    provider.provider_config["temperature"] = 1.2
    provider.provider_config["presence_penalty"] = 1.0
    kwargs = {
        "temperature": 0,
        "top_p": 0,
        "topP": 0.9,
        "seed": 0,
        "max_output_tokens": 8,
        "response_logprobs": False,
        "frequency_penalty": 0,
        "presence_penalty": None,
        "abort_signal": object(),
    }
    if streaming:
        replies.append(
            [
                {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "ok"}]},
                            "finishReason": "STOP",
                        }
                    ]
                }
            ]
        )
        _ = [r async for r in provider.text_chat_stream(prompt="test", **kwargs)]
    else:
        await provider.text_chat(prompt="test", **kwargs)
    config = requests[0]["generationConfig"]
    assert config["temperature"] == 0
    assert config["topP"] == 0
    assert config["seed"] == 0
    assert config["maxOutputTokens"] == 8
    assert config["responseLogprobs"] is False
    assert config["frequencyPenalty"] == 0
    assert "presencePenalty" not in config
    assert "abort_signal" not in json.dumps(requests[0])


@pytest.mark.asyncio
async def test_c_recitation_does_not_send_temperature_above_two(sdk_provider):
    provider, requests, replies = sdk_provider
    replies.extend(
        [
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "blocked"}]},
                        "finishReason": "RECITATION",
                    }
                ]
            }
        ]
        * 5
    )
    with pytest.raises(Exception):
        await provider._query(
            {
                "model": provider.model_name,
                "messages": [{"role": "user", "content": "start"}],
                "temperature": 1.9,
            },
            None,
        )
    assert len(requests) <= 2
    assert all(0 <= r["generationConfig"]["temperature"] <= 2 for r in requests)


@pytest.mark.asyncio
async def test_c_pro_default_budget_is_not_invalid_zero(sdk_provider):
    provider, _, _ = sdk_provider
    provider.provider_config = {}
    config = await provider._prepare_query_config({"model": "gemini-2.5-pro"})
    assert config.thinking_config is None or config.thinking_config.thinking_budget in (
        None,
        -1,
    )


@pytest.mark.asyncio
async def test_c_history_audio_and_model_media_keep_roles(sdk_provider, tmp_path):
    import wave

    provider, requests, _ = sdk_provider
    audio = tmp_path / "history.wav"
    with wave.open(str(audio), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\x00\x00" * 16)
    history = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{png_data()}"},
                },
                {"type": "audio_url", "audio_url": {"url": str(audio)}},
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    await provider.text_chat(contexts=history)
    model = wire_contents(requests[0])[1]
    assert model["role"] == "model"
    assert [
        types.Part.model_validate(p).inline_data.mime_type for p in model["parts"]
    ] == ["image/png", "audio/wav"]


@pytest.mark.parametrize("tool_ids", [["a"], ["a", "a"], ["a", "wrong"]])
def test_h_truncation_does_not_keep_partial_tool_batch(tool_ids):
    from astrbot.core.agent.context.truncator import ContextTruncator

    messages = [
        Message.model_validate(
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": name, "function": {"name": "one", "arguments": "{}"}}
                    for name in ["a", "b"]
                ],
            }
        )
    ]
    messages.extend(
        Message(role="tool", tool_call_id=name, content="result") for name in tool_ids
    )
    assert ContextTruncator().fix_messages(messages) == []


@pytest.mark.asyncio
async def test_c_known_pro_minimal_and_unknown_alias_thinking(sdk_provider):
    provider, _, _ = sdk_provider
    provider.provider_config["gm_thinking_config"] = {"level": "MINIMAL"}
    with pytest.raises(ValueError, match="thinking"):
        await provider._prepare_query_config({"model": "gemini-3.1-pro-preview"})
    provider.provider_config["gm_thinking_config"] = {"budget": 1024}
    config = await provider._prepare_query_config({"model": "gateway-alias"})
    assert config.thinking_config.thinking_budget == 1024


@pytest.mark.asyncio
async def test_h_runner_real_database_checkpoint_and_sdk_replay(sdk_provider, temp_db):
    from types import SimpleNamespace
    from astrbot.core.conversation_mgr import ConversationManager
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
        InternalAgentSubStage,
    )
    from astrbot.core.agent.message import bind_checkpoint_messages

    provider, requests, replies = sdk_provider
    replies.append(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "working"},
                            {
                                "functionCall": {
                                    "name": "one",
                                    "id": "native-one",
                                    "args": {},
                                },
                                "thoughtSignature": "/wA=",
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
    )
    executor = ResultExecutor(
        [CallToolResult(content=[TextContent(type="text", text="result")])]
    )
    runner = await reset_runner(provider, executor)
    _ = [event async for event in runner.step_until_done(2)]
    await temp_db.initialize()
    row = await temp_db.create_conversation(
        user_id="test:FriendMessage:offline", platform_id="test", content=[]
    )
    manager = ConversationManager(temp_db)
    runner.req.conversation = await manager.get_conversation(
        "test:FriendMessage:offline", row.conversation_id
    )
    stage = InternalAgentSubStage()
    stage.conv_manager = manager
    event = SimpleNamespace(
        unified_msg_origin="test:FriendMessage:offline",
        get_extra=lambda key: (
            "checkpoint-offline" if key == "llm_checkpoint_id" else None
        ),
    )
    await stage._save_to_history(
        event, runner.req, runner.final_llm_resp, runner.run_context.messages, None
    )
    loaded = await manager.get_conversation(
        event.unified_msg_origin, row.conversation_id
    )
    restored = bind_checkpoint_messages(json.loads(loaded.history))
    assert any(m.provider_state for m in restored)
    assert restored[-1]._checkpoint_after.id == "checkpoint-offline"
    before = len(executor.calls)
    await provider.text_chat(
        contexts=[*restored, Message(role="user", content="continue")]
    )
    model = wire_contents(requests[-1])[1]
    assert (
        base64.urlsafe_b64decode(model["parts"][1]["thoughtSignature"]) == b"\xff\x00"
    )
    assert model["parts"][1]["functionCall"]["id"] == "native-one"
    assert len(executor.calls) == before == 1


@pytest.mark.asyncio
async def test_h_temp_parts_and_modality_filter_cannot_restore_native_media(
    sdk_provider,
):
    from astrbot.core.provider.modalities import sanitize_contexts_by_modalities

    provider, requests, _ = sdk_provider
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(
            content=types.ModelContent(
                parts=[
                    types.Part(text="caption"),
                    types.Part.from_bytes(
                        data=base64.b64decode(png_data()), mime_type="image/png"
                    ),
                ]
            )
        ),
        response,
    )
    message = response.to_assistant_message()
    filtered, _ = sanitize_contexts_by_modalities([message], ["text"])
    with pytest.raises(ValueError, match="revision"):
        await provider.text_chat(
            contexts=[
                {"role": "user", "content": "start"},
                *filtered,
                {"role": "user", "content": "continue"},
            ]
        )
    message.content[-1].mark_as_temp()
    saved = dump_messages_with_checkpoints([message])
    assert png_data() not in json.dumps(saved)
    with pytest.raises(ValueError, match="invalidated"):
        await provider.text_chat(
            contexts=[{"role": "user", "content": "start"}, *saved]
        )
    assert not requests


@pytest.mark.asyncio
async def test_h_generic_and_third_party_provider_never_receive_native_state(
    sdk_provider,
):
    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
    from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic

    provider, _, _ = sdk_provider
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(
            content=types.ModelContent(
                parts=[
                    types.Part(
                        text="safe display", thought_signature=b"private-signature"
                    )
                ]
            )
        ),
        response,
    )
    message = response.to_assistant_message()
    for cls in [ProviderOpenAIOfficial, ProviderAnthropic]:
        other = cls.__new__(cls)
        assert all(
            "provider_state" not in item
            for item in other._ensure_message_to_dicts([message, message.model_dump()])
        )
    runner = await reset_runner(provider, ResultExecutor([]))
    runner.provider = ProviderOpenAIOfficial.__new__(ProviderOpenAIOfficial)
    runner.provider.provider_config = {}
    sanitized = runner._sanitize_contexts_for_provider([message])
    assert not any(getattr(item, "provider_state", None) for item in sanitized)
    assert message.provider_state is not None


@pytest.mark.asyncio
async def test_c_model_tail_and_blocked_response_are_diagnostic(sdk_provider):
    provider, requests, replies = sdk_provider
    with pytest.raises(ValueError, match="model"):
        await provider.text_chat(
            contexts=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "model tail"},
            ]
        )
    assert not requests
    replies.append({"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(Exception, match="SAFETY"):
        await provider.text_chat(prompt="test")


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_c_pure_media_is_usable_and_persistable(sdk_provider, streaming):
    provider, requests, replies = sdk_provider
    reply = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"inlineData": {"mimeType": "audio/wav", "data": "AA=="}}
                    ],
                },
                "finishReason": "STOP",
            }
        ]
    }
    replies.append([reply] if streaming else reply)
    if streaming:
        result = [r async for r in provider.text_chat_stream(prompt="test")][-1]
    else:
        result = await provider.text_chat(prompt="test")
    message = result.to_assistant_message()
    assert message.content[0].type == "audio_url"
    await provider.text_chat(
        contexts=[
            {"role": "user", "content": "start"},
            message,
            {"role": "user", "content": "continue"},
        ]
    )
    assert [
        types.Part.model_validate(p) for p in wire_contents(requests[-1])[1]["parts"]
    ] == [
        types.Part.model_validate(p) for p in reply["candidates"][0]["content"]["parts"]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["missing_finish", "truncated", "cancelled"])
async def test_s_interrupted_stream_cannot_submit_calls(sdk_provider, error):
    import asyncio

    provider, _, _ = sdk_provider
    closed = []

    async def chunks():
        try:
            yield types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.ModelContent(
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name="one", args={}
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
            if error == "truncated":
                raise httpx.ReadError("truncated stream")
            if error == "cancelled":
                raise asyncio.CancelledError()
        finally:
            closed.append(True)

    # Resource-failure fixture; successful streaming tests use actual SSE parsing.
    from unittest.mock import AsyncMock

    provider.client.models.generate_content_stream = AsyncMock(return_value=chunks())
    expected = {
        "missing_finish": ValueError,
        "truncated": httpx.ReadError,
        "cancelled": asyncio.CancelledError,
    }[error]
    received = []
    with pytest.raises(expected):
        async for item in provider._query_stream(
            {"messages": [{"role": "user", "content": "start"}]},
            None,
            request_max_retries=1,
        ):
            received.append(item)
    assert not any(not item.is_chunk for item in received)
    assert closed == [True]


@pytest.mark.asyncio
async def test_m_local_signature_loss_is_never_replaced_with_sentinel(sdk_provider):
    provider, requests, _ = sdk_provider
    provider.model_name = "gemini-3.1-pro-preview"
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(
            content=types.ModelContent(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name="one", args={}),
                        thought_signature=b"original",
                    )
                ]
            ),
            finish_reason="STOP",
        ),
        response,
    )
    response.provider_state["gemini"]["content"]["parts"][0].pop("thoughtSignature")
    history = [
        {"role": "user", "content": "start"},
        response.to_assistant_message(),
        {
            "role": "tool",
            "tool_call_id": response.tools_call_ids[0],
            "content": "result",
        },
    ]
    with pytest.raises(ValueError, match="signature|invalidated Gemini state"):
        await provider.text_chat(contexts=history)
    assert not requests


@pytest.mark.asyncio
async def test_s_chunk_layouts_preserve_signatures_and_candidate_isolation(
    sdk_provider,
):
    provider, _, replies = sdk_provider
    replies.append(
        [
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": "one"}]},
                    },
                    {
                        "index": 1,
                        "content": {
                            "role": "model",
                            "parts": [{"functionCall": {"name": "wrong", "args": {}}}],
                        },
                    },
                ]
            },
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {"role": "model", "parts": [{"text": " two"}]},
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            },
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [{"text": "", "thoughtSignature": "/wA="}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            },
        ]
    )
    responses = [r async for r in provider.text_chat_stream(prompt="test")]
    assert (
        "".join(r.completion_text or "" for r in responses if r.is_chunk) == "one two"
    )
    final = responses[-1]
    assert final.completion_text == "one two" and final.tools_call_ids == []
    assert final.usage.total == 15
    parts = types.Content.model_validate(
        final.provider_state["gemini"]["content"]
    ).parts
    assert len(parts) == 2
    assert parts[1].text == "" and parts[1].thought_signature == b"\xff\x00"


@pytest.mark.asyncio
async def test_c_media_only_history_is_saved(sdk_provider):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
        InternalAgentSubStage,
    )
    from astrbot.core.db.po import Conversation

    provider, _, _ = sdk_provider
    response = LLMResponse("assistant")
    provider._process_content_parts(
        types.Candidate(
            content=types.ModelContent(
                parts=[types.Part.from_bytes(data=b"\x00", mime_type="audio/wav")]
            )
        ),
        response,
        validate_output=False,
    )
    stage = InternalAgentSubStage()
    stage.conv_manager = AsyncMock()
    request = ProviderRequest(
        conversation=Conversation(platform_id="test", user_id="test", cid="offline")
    )
    event = SimpleNamespace(unified_msg_origin="test", get_extra=lambda key: None)
    await stage._save_to_history(
        event,
        request,
        response,
        [Message(role="user", content="start"), response.to_assistant_message()],
        None,
    )
    stage.conv_manager.update_conversation.assert_awaited_once()


@pytest.mark.asyncio
async def test_s_key_retry_does_not_repeat_started_output(sdk_provider):
    from google.genai.errors import APIError
    from unittest.mock import AsyncMock

    provider, _, _ = sdk_provider
    attempts = []

    async def fail_after_text(*args, **kwargs):
        attempts.append(True)
        yield LLMResponse("assistant", completion_text="already shown", is_chunk=True)
        raise APIError(429, {"error": {"message": "offline failure"}})

    provider._query_stream = fail_after_text
    provider._handle_api_error = AsyncMock(return_value=True)
    with pytest.raises(APIError):
        _ = [r async for r in provider.text_chat_stream(prompt="test")]
    assert attempts == [True]
    provider._handle_api_error.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("edit_stage", ["response", "message", "response_media"])
async def test_h_redaction_removes_raw_snapshot_before_persistence(
    sdk_provider, edit_stage
):
    provider, requests, _ = sdk_provider
    response = LLMResponse("assistant")
    parts = [types.Part(text="private-original")]
    if edit_stage == "response_media":
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(png_data()), mime_type="image/png"
            )
        )
    provider._process_content_parts(
        types.Candidate(content=types.ModelContent(parts=parts)), response
    )
    if edit_stage == "response":
        response.completion_text = "redacted"
    elif edit_stage == "response_media":
        response.result_chain.chain = [
            p for p in response.result_chain.chain if p.type.value != "Image"
        ]
    message = response.to_assistant_message()
    if edit_stage == "message":
        message.content[0].text = "redacted"
    saved = dump_messages_with_checkpoints([message])
    if edit_stage != "response_media":
        assert "private-original" not in json.dumps(saved)
    else:
        assert png_data() not in json.dumps(saved)
    assert saved[0]["provider_state"]["gemini"]["invalidated"]
    with pytest.raises(ValueError):
        await provider.text_chat(
            contexts=[
                Message(role="user", content="start"),
                *saved,
                Message(role="user", content="continue"),
            ]
        )
    assert not requests
