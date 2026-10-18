import hashlib

import pytest

from astrbot import __version__
from astrbot.core.provider.headers import (
    DEFAULT_USER_AGENT,
    HeaderTemplateError,
    RequestHeaderContext,
    build_provider_headers,
    render_header_template,
    render_request_headers,
    split_provider_headers,
    validate_header_templates,
    validate_provider_headers,
)

FULL_CONTEXT = RequestHeaderContext(
    conversation_id="conversation-1",
    umo="aiocqhttp:GroupMessage:100",
    request_id="request-1",
    model="gpt-test",
    provider_id="provider-1",
    version=__version__,
)


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        ("literal", "literal"),
        ("", ""),
        ("前缀-中文", "前缀-中文"),
        ("{conversation_id}", "conversation-1"),
        ("{umo}", "aiocqhttp:GroupMessage:100"),
        ("{request_id}", "request-1"),
        ("{model}", "gpt-test"),
        ("{provider_id}", "provider-1"),
        ("{version}", __version__),
        ("pre-{umo}-post", "pre-aiocqhttp:GroupMessage:100-post"),
        ("{{literal}}", "{literal}"),
        ("a}}b", "a}b"),
    ],
)
def test_render_header_template_values(template, expected):
    assert render_header_template(template, FULL_CONTEXT) == expected


def test_render_header_template_hashes_single_and_nested_templates():
    assert render_header_template("{sha256:{conversation_id}}", FULL_CONTEXT) == (
        hashlib.sha256(b"conversation-1").hexdigest()
    )
    assert render_header_template("{sha256:static}", FULL_CONTEXT) == (
        hashlib.sha256(b"static").hexdigest()
    )
    nested = render_header_template("{sha256:{sha256:{umo}}}", FULL_CONTEXT)
    inner = hashlib.sha256(FULL_CONTEXT.umo.encode()).hexdigest()
    assert nested == hashlib.sha256(inner.encode()).hexdigest()


@pytest.mark.parametrize(
    "template",
    [
        "{unknown}",
        "{unknown_function:{umo}}",
        "{Conversation_Id}",
        "{}",
        "{sha256:}",
        "{conversation_id",
        "{sha256:{conversation_id}",
        "a}b",
        "{umo}}",
        "{conversation_id}suffix{",
    ],
)
def test_render_header_template_rejects_invalid_templates(template):
    with pytest.raises(HeaderTemplateError):
        render_header_template(template, FULL_CONTEXT)


def test_render_header_template_rejects_excessive_nesting():
    template = "x"
    for _ in range(9):
        template = "{sha256:" + template + "}"
    with pytest.raises(HeaderTemplateError, match="deeper than"):
        render_header_template(template, FULL_CONTEXT)


@pytest.mark.parametrize("template", ["{umo}", "{model}", "{sha256:{umo}}"])
def test_render_header_template_omits_missing_values(template):
    assert render_header_template(template, RequestHeaderContext()) is None


def test_render_header_template_falls_back_to_request_id():
    ctx = RequestHeaderContext(request_id="request-1")
    assert render_header_template("{conversation_id}", ctx) == "request-1"
    assert render_request_headers({"X-Session": "{conversation_id}"}, ctx) == {
        "X-Session": "request-1"
    }


@pytest.mark.parametrize("control", ["\r", "\n", "\0"])
def test_render_request_headers_rejects_control_characters(control):
    with pytest.raises(HeaderTemplateError):
        render_request_headers({"X-Test": f"a{control}b"}, FULL_CONTEXT)
    with pytest.raises(HeaderTemplateError):
        render_request_headers({f"X{control}Test": "a"}, FULL_CONTEXT)


def test_render_request_headers_omits_rendered_control_characters():
    ctx = RequestHeaderContext(umo="aiocqhttp:GroupMessage:100\r\nx")
    assert render_request_headers({"X-Umo": "{umo}"}, ctx) == {}


def test_render_header_template_error_does_not_echo_template_values():
    with pytest.raises(HeaderTemplateError) as exc_info:
        render_header_template("Bearer {sk_live_xxx}", FULL_CONTEXT)
    message = str(exc_info.value)
    assert "sk_live_xxx" not in message
    assert "Bearer" not in message
    assert "umo" in message


def test_render_request_headers_drops_missing_values_and_reserved_header():
    headers = render_request_headers(
        {
            "X-Umo": "{umo}",
            "X-Missing": "{umo}",
            "X-Model": "{model}",
            "X-Astrbot-Conversation-Id": "spoofed",
        },
        RequestHeaderContext(model="gpt-test"),
    )
    assert headers == {"X-Model": "gpt-test"}


def test_split_provider_headers_keeps_request_templates_out_of_static_headers():
    static, templates = split_provider_headers(
        {
            "X-Static": "value",
            "X-Escaped": "{{value}}",
            "X-Sha": "{sha256:value}",
            "X-Provider": "{provider_id}",
            "X-Session": "{sha256:{conversation_id}}",
            "X-Astrbot-Conversation-Id": "spoofed",
        },
        RequestHeaderContext(provider_id="provider-1", model="gpt-test"),
        request_scoped=True,
    )
    assert static == {
        "X-Static": "value",
        "X-Escaped": "{value}",
        "X-Sha": hashlib.sha256(b"value").hexdigest(),
        "X-Provider": "provider-1",
    }
    assert templates == {"X-Session": "{sha256:{conversation_id}}"}


def test_split_provider_headers_rejects_request_variables_for_static_providers():
    with pytest.raises(HeaderTemplateError, match="only chat providers support"):
        split_provider_headers(
            {"X-Session": "{umo}"},
            RequestHeaderContext(provider_id="tts-1"),
            request_scoped=False,
        )


def test_split_provider_headers_rejects_missing_static_values():
    with pytest.raises(HeaderTemplateError, match="no value for its template"):
        split_provider_headers(
            {"X-Model": "{model}"},
            RequestHeaderContext(provider_id="tts-1"),
            request_scoped=False,
        )


def test_split_provider_headers_reports_provider_and_header_name_only():
    with pytest.raises(HeaderTemplateError) as exc_info:
        split_provider_headers(
            {"X-Secret": "Bearer {unknown}"},
            RequestHeaderContext(provider_id="provider-1"),
            request_scoped=True,
        )
    message = str(exc_info.value)
    assert "provider-1" in message
    assert "X-Secret" in message
    assert "Bearer" not in message


def test_validate_provider_headers_accepts_request_variables_for_chat():
    validate_provider_headers(
        {
            "id": "provider-1",
            "type": "openai_chat_completion",
            "provider_type": "chat_completion",
            "custom_headers": {"X-Session": "{sha256:{conversation_id}}"},
        }
    )


def test_validate_provider_headers_syntax_only_without_provider_type():
    # Without provider_type only template syntax is checked; provider
    # initialization reports request-time variable misuse instead.
    validate_provider_headers(
        {"id": "tts-1", "type": "openai_tts_api", "custom_headers": {"X": "{umo}"}}
    )

    with pytest.raises(HeaderTemplateError, match="Unknown header template variable"):
        validate_provider_headers(
            {"id": "tts-1", "type": "openai_tts_api", "custom_headers": {"X": "{bad}"}}
        )


def test_validate_provider_headers_rejects_request_variable_for_non_chat():
    with pytest.raises(HeaderTemplateError, match="only chat providers support"):
        validate_provider_headers(
            {
                "id": "tts-1",
                "type": "openai_tts_api",
                "provider_type": "text_to_speech",
                "custom_headers": {"X": "{umo}"},
            }
        )


@pytest.mark.parametrize(
    "custom_headers",
    [
        {"X-Model": "{model}"},
        {"X-Session": "{umo}"},
        {"X-Session": "{sha256:{conversation_id}}"},
        {"X-Static": "value", "X-Escaped": "{{value}}"},
        {"X-Astrbot-Conversation-Id": "spoofed"},
        None,
        {},
    ],
)
def test_validate_header_templates_accepts_unresolvable_values(custom_headers):
    validate_header_templates(custom_headers, provider_id="source-1")


@pytest.mark.parametrize(
    "custom_headers",
    [
        {"X-Bad": "{unknown}"},
        {"X-Bad": "{sha256:{conversation_id}"},
        {"X-Bad": "a}b"},
        {"X-Bad": "a\rb"},
        {"X-Bad\n": "value"},
    ],
)
def test_validate_header_templates_rejects_invalid_templates(custom_headers):
    with pytest.raises(HeaderTemplateError, match="source-1"):
        validate_header_templates(custom_headers, provider_id="source-1")


@pytest.mark.parametrize(
    "custom_headers", [None, {}, [], "invalid", {"User-Agent": " "}]
)
def test_build_provider_headers_keeps_default_user_agent(custom_headers):
    assert build_provider_headers(custom_headers) == {"User-Agent": DEFAULT_USER_AGENT}
