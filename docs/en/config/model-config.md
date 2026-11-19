# Configure Custom Model Parameters

You can configure model parameters in the WebUI:

1. Open `Providers` and select the provider containing your model.
2. In its configured models list, click the model name or its settings icon to open the model configuration dialog.
3. Under `Custom request body parameters` (`custom_extra_body`), add the parameters required by your model, such as `temperature`, `top_p`, or `max_tokens`.
4. Click `Save`.

Available fields depend on the provider. Refer to your provider's documentation for supported parameters and values.

## DeepSeek Server-Side Web Tools

The DeepSeek (Chat Completions) provider additionally offers two dedicated switches in its provider configuration:

- `deepseek_web_search`: injects the `deepseek_web_search` (web, image, and deep-thinking search) and `deepseek_read_url` (read links) server-side tools, executed on DeepSeek's servers without occupying the local tool loop; use the `deepseek-flash` model and see the [official docs](https://api-docs.deepseek.com/guides/web_tools/).
- `deepseek_thinking`: adds `thinking: {"type": "enabled"}` to the request body to enable thinking mode; combined with the web tools it enables deep-thinking search.
