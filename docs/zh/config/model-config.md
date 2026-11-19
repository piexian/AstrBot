# 配置自定义模型参数

可在 WebUI 中配置模型参数：

1. 打开「模型提供商」，选择包含目标模型的提供商。
2. 在「已配置的模型」列表中，点击模型名称或其设置图标，打开模型配置对话框。
3. 在「自定义请求体参数」（`custom_extra_body`）中添加模型所需的参数，例如 `temperature`、`top_p` 或 `max_tokens`。
4. 点击「保存」。

可用的字段取决于提供商，具体支持的参数和取值请参考对应提供商的文档。

## DeepSeek 服务端联网工具

使用 DeepSeek（Chat Completions）提供商时，提供商配置里还有两个专属开关：

- `deepseek_web_search`：注入 `deepseek_web_search`（联网、图片与深度思考搜索）和 `deepseek_read_url`（阅读链接）两个服务端工具，在 DeepSeek 服务端执行，不占用本地工具循环；建议使用 `deepseek-flash` 模型，详见[官方文档](https://api-docs.deepseek.com/zh-cn/guides/web_tools/)。
- `deepseek_thinking`：在请求体加入 `thinking: {"type": "enabled"}` 开启思考模式；与联网工具同时开启即为深度思考搜索。
