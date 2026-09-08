# Gemini / Vertex GenerateContent compatibility audit

## Baseline and authorization

Implementation baseline: `6748372ba9a63ceef2028da9a34e78b5c2ed53fc`, branch `fix/gemini-vertex-model-turn-content`, inspected on 2026-09-08. The working tree originally had no tracked changes; the pre-existing `.codegraph/` remains local and excluded. The v2 plan supplied in the conversation is authoritative. The branch was not reset to the plan's master revision and upstream master was not merged.

The actual project version is **4.28.0-beta.1**, not the plan's release label. `pyproject.toml` requires Python >=3.12. Installed Python is 3.12.3; the local lock and environment have google-genai 2.12.1, Pydantic 2.13.4 and httpx 0.28.1. Both dependency declarations require google-genai >=1.56.0, Pydantic >=2.12.5 and httpx >=0.28.1. No dependency declaration, lockfile or service environment was upgraded. Minimum-SDK tests use a temporary uv overlay with google-genai 1.56.0.

CI uses Python 3.12 for unit tests on Linux/macOS/Windows (Windows best effort), Ruff 0.15.22, and a separate Python 3.10–3.14 startup matrix. The existing test dependencies were reused with `--no-sync`; no startup smoke or deployment was run because this task does not authorize starting configured channels. All new network tests use HTTP stubs and synthetic signatures. No live credentials were read or real model request made.

## PR adoption record

Metadata and current diffs were queried with `gh` on 2026-09-08. Every merged commit below was confirmed as an ancestor of HEAD. Author test reports and CI status are not counted as local verification.

| PR | State; merged at (UTC) | Queried head SHA | Adoption / preservation | Tests |
| --- | --- | --- | --- | --- |
| #9899 | merged; 2026-09-04 02:28:40 | `529ab943f42c35a02004846f38361403ca8aa2e8` | Keep per-instance thinking serialization and the 3.7 fallback; merge `6b848090525a0eebe622a9d46b032a6c5148c4a3` | U01 |
| #9880 | merged; 2026-09-01 15:48:13 | `5a6be73353901e310e97399fb70fdbda993c7de1` | Keep prompt minus cached accounting; merge `5ad953611f9bd1a093c128fc8548bd657412c05e` | U02, C10 |
| #9874 | merged; 2026-08-31 01:19:31 | `88768d0b17dadfb24b53f8111f012f45c82b067e` | Keep shared async media resolution; merge `5aca0c9847c2a50bc2e146b696149514845033ac` | U03 |
| #9738 | merged; 2026-08-26 16:57:35 | `b2e4b3d2e70233c4aea3ecc7e540643c5a70e418` | Keep leading-model handling; merge `8ea8ce613a0bee4ddb48b21490afe23418277c75` | U04 |
| #9957 | merged; 2026-09-06 10:48:56 | `b80e3555abe9353dec549389791fed7dea0bdd30` | Keep prefix copying and appended requery instruction; merge `1fc3d33455cd23f3bf68e2ee4f9375004069f9e1` | U05, A09, K05 |
| #8240 | merged; 2026-05-22 12:32:29 | `f9e97ed46378c4d6bd359d6a088f285ccc80d7db` | Keep the first displayed explanation; do not attach second-stage signatures to it; merge `f5bd4f30e578e77def89a3f9c2c934b7542aad0f` | K03 |
| #9761 | open; not merged | `21358a87e997b50586a79ff93a6b73a7e3b99f57` | Adopt name/native-ID separation. Reject whole-history lookup and `id != name` provenance inference | B01–B16 |
| #9691 | open; not merged | `e11ba4936dccf54daa463308ed078e8bd98623bf` | Adopt outer content fallback with an explicit adapter limitation, plus exhausted-executor handling | R01–R06 |
| #9853 | open; not merged | `821f2c79b1807a8c6af34daf6a0506474e10d1dc` | No automatic sentinel injection or foreign-origin inference from missing signatures | M01, M05 |
| #9969 | open; not merged | `89e41792bc539e5a1731ba4c2dab6bab2380604f` | Do not globally replace GIFs with first-frame PNGs | C12 excluded |

The image-format feature branch (#9703 work) is not merged into this Gemini branch. Its separate Windows deployment is not changed by this task.

## Patch boundaries and contracts

### Q1 — Content boundaries

Production boundary handling already exists in baseline commit `6748372ba`. The additional patch supplies real SDK HTTP assertions for plain text, images, parallel results, max-step termination and the actual requery constructor. Function responses remain grouped together and separate from subsequent ordinary user Parts. Existing normal-chat and leading-model behavior is retained. No workaround assistant turn or empty trailing user message is inserted.

### Q2 — Tool result fallback

The independent runner patch handles unsupported MCP content in the outer `CallToolResult.content` dispatch. Known content remains, unknown content produces an adapter-level explanation, and normal exhaustion without a result gets a factual final result. Audio/link contents are not downloaded, transcribed or interpreted. Cancellation remains distinct from normal completion. B subsequently adds aggregation of multiple executor events into one final protocol result.

### Q3–Q5 — Native protocol, persistence and execution

`Message.provider_state` and `LLMResponse.provider_state` are optional. Gemini state version 1 stores the requested model, an endpoint fingerprint (never literal credentials), a stable batch ID, JSON-safe original Content, call bindings, visible-view and native-content digests, finish reason and available raw usage metadata. Bindings distinguish internal identity, original Part index, upstream ID and whether it was present or absent. Old history without state has unknown provenance and uses a batch-local compatibility path; it is not rewritten.

Internal IDs are allocated once when a complete response is parsed. Native IDs, including IDs equal to function names, are replayed unchanged. Absent native IDs stay absent. Results use the original function name and original call order. Same-name/same-argument calls are independent. Duplicate native IDs within a response, malformed list lengths, edited arguments, orphan/duplicate/missing results and ambiguous old duplicate IDs are diagnosed before dispatch or HTTP submission. Complete no-argument calls execute with an empty object without adding that object to the original Part.

The two digests are established at parsing. Normal response hooks, history edits, modality filtering and persistence exclusions cannot reauthorize an old snapshot. Saving an edited message removes raw content and retains only an invalidation marker. SDK bytes are stored in its JSON representation and restored through SDK types; unsigned display text can be joined, but signed Part boundaries and signatures are not moved. Native model media bypasses input transcoding and remains model-owned. Optional omitted SDK fields are normalized by the SDK rather than replaced with fabricated values.

Skills-like execution uses the second response's calls, metadata and native Content, while display history keeps the first explanation. The actual appended requery instruction is stored as a request suffix and restored before that Content. On failed/cancelled Gemini requery, the already displayed explanation is saved as a minimal `display_only` record: it contains no unexecuted calls or native snapshot, and Gemini does not send it as protocol history. Other providers retain their existing first-stage thinking/signature pairing. Already yielded result chains are not mutated when composing display history.

Streaming consumes the selected candidate through stream end, retains signature-only Parts and late usage, and submits exactly one final response. Opening the HTTP stream and reading its first SDK item form one retryable operation, so the configured attempt limit covers actual HTTP failures before any display. Only unsigned plain text deltas are joined. Partial argument mode is explicitly unsupported. Duplicate native call IDs and streams ending without completion evidence for tool calls are rejected before tool execution. Normal text-only EOF remains supported. Started stream output is not replayed by key rotation. Existing timeout and cancellation mechanisms remain in place; generator resources close on exit.

Generic provider conversion strips native state for both Message and dict histories. Runner and summary-compressor boundaries also strip it for providers that do not opt in. Context truncation retains only complete result batches, using multiplicities rather than ID sets. No database schema migration is needed: existing conversation JSON persistence carries the optional state. A real SQLite save/checkpoint/reload/SDK replay test exercises the actual save entry.

### Q6 — Configuration, Schema and accounting

Generation parameters use one allowlisted builder. Per-call overrides win over payload/provider defaults; canonical snake_case wins over aliases in the same layer. Zero and false values survive; explicit null clears a configured value to SDK omission. Local control kwargs do not enter HTTP JSON.

Thinking preserves #9899. Missing 2.5 Pro budget is omitted; explicit invalid Pro/Flash/Flash-Lite budgets produce local errors after SDK numeric normalization. Flash's existing default budget zero is preserved. Gemini 3.1 Pro MINIMAL is diagnosed; 3.7's existing MEDIUM fallback remains. Unknown gateway aliases retain one explicit budget or level with an unverified-capability warning. This is not a complete capability catalog for every dynamically offered model.

Tool declarations now use `parameters_json_schema`, mutually exclusive with `parameters`. Nullable unions, union siblings, oneOf/allOf and additional-properties semantics are preserved. Local JSON Pointer references are expanded without network access, with depth 32 and 2048-node limits; unresolved, cyclic or remote references report the tool and schema path. Reference siblings remain an intersection. Untyped arrays are no longer narrowed to strings. This deliberately changes the public Google-schema dictionary shape and its old missing-items test; callers indexing `parameters` must adapt. MCP normalization and other providers' schema builders are unchanged.

Shared parsing/streaming changes also cover the C media and error invariants: unsupported user Parts are not treated as audio, model media is not converted to `None`, supported media-only output can be saved, blocked responses are distinguished from retryable empty output, and recitation temperature cannot exceed two. Output token usage means generated candidate plus thought tokens, not only visible text and not a billing guarantee. Cached input is subtracted once, partial usage snapshots retain prior fields, and cumulative snapshots are not summed.

## Evidence and executed validation

- User-reported live evidence: separate functionResponse/user tails worked on the supplied new-api/Vertex path; mixed tails returned 400. This task did not independently replay that service.
- Deterministic baseline: the original target suite returned 72 passed. In an independent worktree at `6748372ba`, the added A/R subset returned 18 passed and 8 failed: all A cases and existing empty/error results passed; audio, link, mixed and exhausted results failed on both SDK backends.
- New defects were tested red before their fixes. Additional state-integrity, display-event and pre-dispatch guards were exercised red during implementation and then corrected; these are implementation safeguards, not falsely attributed upstream defects.
- SDK tests use actual `genai.Client`, AsyncClient, HTTP JSON serialization and SSE parsing on Developer and Vertex SDK paths. Vertex uses anonymous fake credentials and MockTransport only. Fault injection tests additionally stub stream generators to verify closure/cancellation.
- Known protobuf aliases are normalized for comparison only; captured raw requests are retained and no fields, arguments, result values or Part order are dropped. SDK 1.56 Vertex uses snake_case for some Parts, and the raw schema key varies between SDK/backend versions.
- Baseline reader probe: the old Message class reads a new record as assistant with one tool call, but discards native state. It cannot be assumed to continue new signed tool trajectories safely. Estimated tokens were 41 with and without native state; the counter does not traverse the snapshot.

Commands actually used (from repository root unless specified):

```bash
uv run --no-sync pytest tests/test_gemini_source.py tests/test_tool_loop_agent_runner.py tests/test_conversation_checkpoint.py -q --tb=short
uv run --no-sync pytest tests --test-profile blocking -q --tb=short --show-capture=no
uv run --with google-genai==1.56.0 --no-sync pytest tests/test_gemini_protocol.py tests/test_gemini_source.py -q --tb=short --show-capture=no
uv run --no-sync ruff format .
uv run --no-sync ruff check .
git diff --check
```

Patch application was checked in detached worktrees rooted at the exact baseline, without stashing or altering user changes. All four patches were also reverse-checked and reversed in order; the verification worktree returned to a clean baseline.

| Actual run | Result | Notes |
| --- | --- | --- |
| Original G + R + CP baseline command above | 72 passed | before implementation |
| Added A/R subset in baseline worktree, `python -m pytest tests/test_gemini_protocol.py -q -k 'test_a or test_r'` | 18 passed, 8 failed | baseline failure proof, not a final failure |
| Q1 alone, `python -m pytest tests/test_gemini_protocol.py -q` | 14 passed | independently applicable |
| Q1 + Q2, same command | 28 passed | independent result fallback |
| Q1 + Q2 + Q3–Q5, same command | 135 passed | complete B before C configuration/Schema changes |
| Complete series, `python -m pytest tests/test_gemini_protocol.py tests/test_gemini_source.py tests/unit/test_tool_google_schema.py -q --tb=short --show-capture=no` | 226 passed | independent final checkout |
| Final main checkout, blocking suite command above | **2585 passed**, 28 warnings | 197.79 seconds, exit 0 |
| Minimum google-genai 1.56.0 command above | **224 passed**, 1 warning | 16.93 seconds, exit 0 |
| Ruff format / check and `git diff --check` | passed | no dependency changes |
| Four reverse checks and reverse applications | passed | verification checkout clean afterward |

Warnings include deprecations and asynchronous SQLite teardown warnings reported in other test modules; their root cause was not changed or claimed resolved. No test failure was hidden or suppressed. Final logs and the ordered patches are included in the delivery archive. Startup, live API and production acceptance are not included in these pass counts.

## Mapping of all 80 plan specifications

`P` = `tests/test_gemini_protocol.py`; `G` = `tests/test_gemini_source.py`; `R` = `tests/test_tool_loop_agent_runner.py`; `CP` = `tests/test_conversation_checkpoint.py`. Names below are actual test functions. PASS means offline coverage only; it does not mean service acceptance. A row can use several existing tests rather than a new duplicate function.

| ID | Actual test / evidence | Scope |
| --- | --- | --- |
| A01 | P.test_a_sdk_preserves_tool_content_boundary | SDK HTTP, successful FC/FR |
| A02 | P.test_a_sdk_preserves_tool_content_boundary, tail=text | SDK HTTP |
| A03 | P.test_a_sdk_preserves_tool_content_boundary, tail=image | SDK HTTP and bytes |
| A04 | P.test_a_sdk_preserves_tool_content_boundary | two calls/results |
| A05 | P.test_a_runner_image_and_max_steps_reach_sdk | real image cache path |
| A06 | P.test_a_runner_image_and_max_steps_reach_sdk | max steps=1, tools removed |
| A07 | G.test_gemini_prepare_conversation_keeps_normal_user_first_history | preservation |
| A08 | P.test_a_sdk_preserves_tool_content_boundary; P.test_b_call_identity_and_native_parts_roundtrip | input unchanged / stable IDs |
| A09 | P.test_a_sdk_preserves_tool_content_boundary, tail=requery | real requery builder |
| R01 | P.test_r_tool_result_has_exactly_one_final_response, audio | unsupported adapter result |
| R02 | same function, resource_link | no link fetch |
| R03 | same function, mixed | known text plus explanation |
| R04 | P.test_r_mixed_batch_counts_and_orders_every_result | ordered Counter comparison |
| R05 | P.test_r_tool_result_has_exactly_one_final_response, empty/exhausted/error | distinct final facts |
| R06 | R.test_stop_interrupts_pending_regular_tool; P.test_h_model_failure_fallback_does_not_repeat_completed_tool | cancellation / side effects |
| B01 | P.test_b_call_identity_and_native_parts_roundtrip | native ID differs from name |
| B02 | same function, absent IDs | unique internal IDs, no injected native IDs |
| B03 | P.test_b_parallel_and_multistep_calls_execute_once_in_order | same name, different args |
| B04 | P.test_b_call_identity_and_native_parts_roundtrip; P.test_r_mixed_batch_counts_and_orders_every_result | same name/args are independent |
| B05 | P.test_b_call_identity_and_native_parts_roundtrip | only first Part signed |
| B06 | P.test_b_parallel_and_multistep_calls_execute_once_in_order | independent signatures |
| B07 | P.test_b_call_identity_and_native_parts_roundtrip | no tool signature on prose |
| B08 | P.test_b_parallel_and_multistep_calls_execute_once_in_order | multiple steps |
| B09 | P.test_b_no_argument_call_is_executable_without_mutating_original | absent args execute as object |
| B10 | P.test_b_bad_tool_batches_fail_before_http | orphan/duplicate/missing results |
| B11 | P.test_b_multiple_executor_results_aggregate_once | one final protocol result |
| B12 | P.test_b_legacy_duplicate_ids_require_original_order | finite recovery / ambiguity error |
| B13 | P.test_b_call_identity_and_native_parts_roundtrip | native ID equal to name |
| B14 | same function, absent native IDs | internal IDs never injected |
| B15 | P.test_b_parallel_and_multistep_calls_execute_once_in_order; P.test_b_bad_tool_batches_fail_before_http | batch-local ID reuse / bad order |
| B16 | P.test_b_mismatched_lists_fail_before_tool_side_effects; P.test_b_edited_call_arguments_fail_before_execution | validation before execution |
| S01 | P.test_s_chunk_layouts_preserve_signatures_and_candidate_isolation | deltas / final text |
| S02 | P.test_s_sdk_stream_collects_parallel_calls_and_usage_tail | separate FC chunks |
| S03 | P.test_s_runner_streaming_history_replays_actual_execution | preface and history |
| S04 | P.test_s_chunk_layouts_preserve_signatures_and_candidate_isolation | empty signed text Part |
| S05 | P.test_s_sdk_stream_collects_parallel_calls_and_usage_tail | metadata-only tail |
| S06 | P.test_s_chunk_layouts_preserve_signatures_and_candidate_isolation; P.test_s_partial_usage_tail_preserves_prior_fields | snapshots, not sums |
| S07 | P.test_s_interrupted_stream_cannot_submit_calls; P.test_s_key_retry_does_not_repeat_started_output | interruption / close / retry |
| S08 | P.test_s_incomplete_calls_do_not_commit | partial mode explicitly rejected |
| S09 | P.test_s_chunk_layouts_preserve_signatures_and_candidate_isolation; P.test_s_runner_streaming_history_replays_actual_execution | shared complete-response parser |
| K01 | P.test_k_runner_preserves_display_but_commits_requery_protocol | changed native ID |
| K02 | same function | same native ID, new args/signature |
| K03 | same function; P.test_k_other_providers_keep_matching_first_thinking_signature | display once, preserve other providers |
| K04 | same function, no_tool; P.test_k_failed_requery_does_not_execute_candidate | no old calls, error/cancel display retained |
| K05 | P.test_a_sdk_preserves_tool_content_boundary, tail=requery; P.test_s_runner_streaming_history_replays_actual_execution | independent appended instruction |
| H01 | P.test_h_runner_real_database_checkpoint_and_sdk_replay | actual SQLite save/reload |
| H02 | G existing history tests; CP round trips; baseline-reader probe | no DB migration |
| H03 | P.test_b_call_identity_and_native_parts_roundtrip; P.test_h_runner_real_database_checkpoint_and_sdk_replay | Message/JSON/DB state survives |
| H04 | P.test_h_redaction_removes_raw_snapshot_before_persistence | response and Message edits |
| H05 | P.test_h_temp_parts_and_modality_filter_cannot_restore_native_media | exclusions cannot revive raw bytes |
| H06 | P.test_h_truncation_does_not_keep_partial_tool_batch; P.test_h_runner_real_database_checkpoint_and_sdk_replay; tests/agent | batches / checkpoints |
| H07 | P.test_h_generic_and_third_party_provider_never_receive_native_state; P.test_b_stale_or_incompatible_snapshot_is_not_replayed | no leakage / wrong-model replay |
| H08 | P.test_h_expired_legacy_media_fails_without_replaying_other_content | clear failure, no role substitution |
| H09 | P.test_h_two_sessions_do_not_share_call_bindings | concurrent sessions |
| H10 | P.test_h_model_failure_fallback_does_not_repeat_completed_tool | one completed side effect |
| H11 | P.test_b_stale_or_incompatible_snapshot_is_not_replayed, version; P.test_h_opaque_state_corruption_is_not_persisted | unknown / corrupt state |
| C01 | P.test_c_generation_kwargs_reach_sdk_with_zero_values | both request paths |
| C02 | same function | zero/false/null omission, canonical alias precedence |
| C03 | P.test_c_pro_default_budget_is_not_invalid_zero; P.test_c_known_pro_minimal_and_unknown_alias_thinking; P.test_c_known_thinking_budget_rejects_invalid_explicit_values; P.test_c_flash_preserves_existing_disabled_thinking_default | verified model rules only |
| C04 | P.test_c_json_schema_semantics_reach_http; P.test_c_local_ref_keeps_sibling_constraints | unions / refs |
| C05 | P.test_c_bad_schema_references_fail_locally | remote/cyclic/unresolved refs, no fetch |
| C06 | P.test_c_json_schema_semantics_reach_http | siblings / untyped arrays |
| C07 | P.test_s_chunk_layouts_preserve_signatures_and_candidate_isolation; G.test_gemini_empty_output_raises_empty_model_output_error | empty text vs signature |
| C08 | P.test_c_history_audio_and_model_media_keep_roles; P.test_c_unknown_parts_are_not_reinterpreted_as_audio_or_text | model image/audio history and unknown types |
| C09 | P.test_c_pure_media_is_usable_and_persistable; P.test_c_media_only_history_is_saved; P.test_c_blocked_responses_are_not_empty_output_retries | media / blocking / emptiness |
| C10 | P.test_c_partial_usage_never_produces_negative_input; P.test_s_partial_usage_tail_preserves_prior_fields; G cache tests | cache and generated thoughts |
| C11 | P.test_c_recitation_does_not_send_temperature_above_two; P.test_s_key_retry_does_not_repeat_started_output; P.test_c_blocked_responses_are_not_empty_output_retries; P.test_c_repeated_feature_errors_stop_after_one_adaptation; P.test_c_production_sdk_retry_composition_honors_attempt_limit | bounded retry, actual SDK/wrapper composition and parameters |
| C12 | Not applicable | no image-format/GIF conversion change |
| C13 | P.test_c_signed_model_media_never_runs_input_transcoding | original signed bytes |
| U01 | G.test_gemini_thinking_level_is_serialized_on_every_request; G.test_gemini_37_minimal_thinking_level_falls_back_to_medium | preserved |
| U02 | G.test_gemini_extract_usage_excludes_cached_tokens_from_input_other; G.test_gemini_extract_usage_without_cache_keeps_full_prompt_tokens | preserved |
| U03 | G.test_gemini_prepare_conversation_resolves_local_history_image; P image HTTP cases | shared resolver preserved |
| U04 | G.test_gemini_prepare_conversation_removes_leading_model_content | current user retained |
| U05 | R.test_skills_like_requery_preserves_existing_context_prefix; P requery HTTP cases | system/prefix unchanged |
| M01 | P.test_m_local_signature_loss_is_never_replaced_with_sentinel; P.test_b_unusable_native_batch_is_rejected_before_side_effects | no sentinel / early failure |
| M02 | Excluded | optional explicit migration not implemented |
| M03 | Excluded | no migration of unknown provenance |
| M04 | Excluded | no sentinel replacement path |
| M05 | P.test_m_legacy_signature_checks_are_scoped_to_current_turn | old unsigned turns are not blanket-blocked |
| M06 | Excluded | optional migration persistence not implemented |

## Remaining validation limits

All specified live new-api/Vertex, direct Vertex and Developer API service checks are SKIP: no explicit live authorization or credentials were supplied. HTTP 200 responses in transport fixtures prove code and wire structure only. No Windows/macOS or alternate-Python execution was performed in this task. Minimum google-genai coverage uses the current Pydantic/httpx versions, not every dependency combination.

B6 and D are not implemented: no migration switch/sentinel, native functionResponse media rollout, image streaming feature, Live/Interactions API or authentication migration. The collector supports default SDK delta text and complete function-call Parts; nonstandard cumulative gateway snapshots without identity/continuation metadata are not claimed compatible. Dynamic model capability catalogs and real audio playback across platforms are not exhaustively verified.

Storage is bounded by retained conversation content, without a parallel unbounded cache. Native snapshots and display media references can store two encodings of a media payload; digesting and serialization are linear in retained content size. No timing SLA or cloud-billing equivalence is claimed. The native snapshot is not counted again by the token estimator.

## Rollback and application

Apply the four patches in order on baseline `6748372ba`: Q1 boundary regressions, Q2 tool result fallback, Q3–Q5 native protocol, then Q6 configuration/Schema. The separate `Q1-existing-6748372.patch` documents the already-present production fix and must not be applied twice. The B patch includes shared media/error validation needed by its persistence/streaming contract; Q6 contains the independently reviewable configuration, Schema and usage changes.

Before any later deployment, save configuration and conversation backups using the site's normal process. Deployment remains unperformed here. To roll back, first run `git apply --reverse --check <patch>` in a review checkout, then reverse Q6, Q3–Q5, Q2 and Q1 in that order. Do not apply reverse patches blindly over later user edits. Q6 can be reverted without a database migration, but complex schemas and accounting semantics revert too. Q2 independently restores the old unsupported-result limitation. Reverting B makes old readers discard native state: restore pre-deployment conversations or start fresh conversations for signed tool trajectories; do not rewrite stored signatures or perform a whole-release downgrade as a substitute. The original boundary commit should only be reverted after dependent protocol changes are removed.

The implementation is organized into the following local functional commits, with this audit committed separately. Nothing has been pushed. No production channels, configuration, credentials, databases or running instances were changed.

| Commit | Function |
| --- | --- |
| `0ed726922` | SDK tool-response boundary regressions |
| `a17faeb66` | Unsupported and empty tool-result fallback |
| `d99521cd4` | Native protocol state, runner execution and history persistence |
| `4709593d2` | Generation options, thinking validation and token accounting |
| `d8cbe3b32` | Lossless Google JSON Schema declarations |

The generation/options commit was additionally checked before applying the Schema commit in an independent checkout: `python -m pytest tests/test_gemini_protocol.py tests/test_gemini_source.py tests/unit/test_tool_google_schema.py -q --tb=short --show-capture=no` returned **210 passed**, one deprecation warning, exit 0. The final implementation and test files are byte-identical to the previously validated worktree; only commit organization and this audit changed.

## Sources checked

The native identity and JSON Schema paths were checked against the [Content REST definition](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/v1/Content) and [FunctionDeclaration definition](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/Shared.Types/FunctionDeclaration). Signature constraints were checked against [GenerateContent thought signatures](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures), and verified thinking ranges against [GenerateContent thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking), on 2026-09-08. These document the target protocol; they do not replace live channel verification.
