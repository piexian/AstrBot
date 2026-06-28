# QQ 官方机器人「回复消息丢图 / 回复不激活」调研

> 调研日期：2026-06-28
> 对照对象：腾讯官方 [`tencent-connect/openclaw-qqbot`](https://github.com/tencent-connect/openclaw-qqbot) v1.7.x（TypeScript，QQ 龙虾机器人 OpenClaw 接入插件）
> 本仓库现状：`astrbot/core/platform/sources/qqofficial/`（基于旧 `botpy` SDK / 旧 QQ 官方机器人 API）

## 一、结论先行

AstrBot 的 `qq_official` 适配器在「用户回复一条带图消息时图片丢失」「回复机器人消息时唤醒词/激活可能失效」的根因是：**它抱的是旧 `botpy` SDK / 旧 QQ 官方机器人 API**。

- 旧 API 引用事件里 `message_reference` **只有 `message_id`**，群 / C2C **没有任何回查消息内容的 API**。
- 新 QQ 龙虾 API（openclaw 用的）在引用事件里**通过 `msg_elements[0]` 整条内联了被引用消息的 `content` + `attachments`**。openclaw 能拿到引用图片，根因在此；它的本地缓存只是优化，不是解法本体。

真根治只有一条路：**对接新 QQ 龙虾 API**（拿到 `msg_elements[0]` 内联内容）。在旧 botpy 上做本地缓存只是有死角的过渡方案。

## 二、新旧协议对比

| 维度 | 旧 botpy（AstrBot 现状） | 新 QQ 龙虾 API（openclaw） |
|---|---|---|
| 引用事件被引用消息内容 | ❌ 仅 `message_id` | ✅ `msg_elements[0]` 内联 content+attachments |
| 群 / C2C 取消息 API | ❌ 无（只有只写 POST） | — 不需要，事件已内联 |
| 频道取消息 API | ✅ `GET /channels/{cid}/messages/{mid}` | — |
| 丢图能否解决 | ⚠️ 只能本地缓存，且**只能恢复 bot 见过的消息** | ✅ 协议层根治，缓存非必需 |
| 缓存的作用 | 旧协议下的唯一救命药 | 优化（省下载）+ 隐式唤醒 + 兜底 |

旧 API 证据（本仓库依赖 `botpy`）：
- `botpy/message.py` `_MessageRef` 仅解析 `message_id`（`data.get("message_reference", {}).message_id`），不含内容。
- `botpy/api.py` 群/C2C 仅 `POST /v2/groups/{group_openid}/messages`、`POST /v2/users/{openid}/messages`（只写）；频道有 `get_message`（`GET /channels/{cid}/messages/{mid}`）。

## 三、新 QQ 龙虾 API 引用事件结构

`openclaw-qqbot` 的 `src/types.ts`：

```ts
export const MSG_TYPE_QUOTE = 103;   // 引用（回复）消息

/** 消息元素结点，引用消息时 msg_elements[0] 为被引用的原始消息 */
export interface MsgElement {
  msg_idx?: string;            // REFIDX_xxx 索引
  message_type?: number;
  content?: string;            // ← 被引用消息文本，内联
  attachments?: MessageAttachment[];  // ← 被引用消息附件（含 url），内联
  msg_elements?: MsgElement[];
}

// C2CMessageEvent / GroupMessageEvent 均含：
//   msg_elements?: MsgElement[]   // [0] = 被引用的原始消息
//   message_scene?: { ext?: string[] }  // ["ref_msg_idx=REFIDX_xxx", "msg_idx=REFIDX_yyy"]
```

## 四、openclaw 的处理流水线（已逐行确证）

REFIDX 提取（`src/utils/text-parsing.ts` `parseRefIndices`）：从 `ext` 解析 `ref_msg_idx=`/`msg_idx=`；当 `message_type==MSG_TYPE_QUOTE` 时用 `msg_elements[0].msg_idx` 覆盖（更权威）。

回填两条路径（`src/gateway.ts`）：
- **缓存命中**（行 925）：`getRefIndex(refMsgIdx)` → 行 931 `formatRefEntryForAgent`（用已下载 `localPath`，同步不下载）。
- **缓存未命中且 `msgType==103`**（行 934-939）：从 `msg_elements[0]` 取 `{content, attachments}` → `formatMessageReferenceForAgent` 现场下载附件、语音转录。
- 未命中且非引用类型 / `msg_elements` 为空（行 941-948）：拿不到内容，AI 仅知「用户引用了一条消息」。

索引写入：
- 入站消息处理完写缓存（行 972-975）。
- **bot 出站消息也写缓存**，带 `isBot: true`（行 496-503）—— 隐式唤醒的基础。

隐式唤醒（行 296-303 + 1116-1119）：`resolveImplicitMention` 只查缓存 `getRefIndex`，`refEntry?.isBot === true` 即「回复 bot 消息 = 隐式唤醒」。因 bot 出站必缓存，故回复 bot 消息一定命中。

附件落地：`formatAttachmentTags`（`src/group-history.ts`）有本地路径 → `MEDIA:/path`；语音+转录 → `MEDIA:path（内容:"…"）`；无路径 → `[图片]`/`[语音消息]`。下载到本地是因为 QQ 附件 URL 会过期。

缓存持久化（`src/ref-index-store.ts`）：JSONL 追加写 `~/.openclaw/qqbot/data/ref-index.jsonl`，7 天 TTL，5 万条上限，LRU + compact，跨重启不丢。

## 五、对 AstrBot 的启示

1. **根因**：`_parse_from_qqofficial`（`qqofficial_platform_adapter.py`）完全忽略 `message_reference`，不解析、不构造 `Reply` 组件、不缓存历史消息。`waking_check/stage.py:130-134` 本有「Reply 组件且 `sender_id==self_id` 即唤醒」路径，但因 QQ 适配器从不构造 `Reply` 而恒失效。
2. **过渡方案（旧 botpy + 本地缓存）**：建 `message_id → {文本, 图片(localPath), sender_id, is_bot}` 缓存（入站+出站都写），回复时查缓存回填图片并构造 `Reply` 组件。局限：只能恢复 bot 见过的消息，用户回复 bot 不在场的历史消息图片仍丢。
3. **根治方案（对接新 API）**：重写传输层对接新 QQ 龙虾 API，解析 `msg_elements[0]` 内联内容。改动大但根治，且顺带获得 STT、流式、大文件、键盘按钮等能力。

## 六、相关参考

- openclaw-qqbot 仓库：https://github.com/tencent-connect/openclaw-qqbot
- QQ 龙虾机器人文档（腾讯文档，需登录）：https://docs.qq.com/doc/DTFRyQURhT2ZvQ25J
- 本仓库上游（截至 2026-06-28）**未修复此问题**，无对应 issue/PR。
