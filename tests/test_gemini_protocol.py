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
