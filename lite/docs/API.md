# The API

Strata-Lite speaks llama-server's API so existing clients move over by changing a port. Verified by `lite/tests/api_parity.py`
against a llama-server on the same model: 63 checks pass for each model, and the shape differences are the ones listed below
(`lite/results/api/parity-5070-auto.txt` for Gemma 4, `lite/results/api/parity-qwen36-5070.txt` for Qwen3.6).

## Endpoints
- `POST /v1/chat/completions`: OpenAI chat; streaming and `stream_options.include_usage`; `cache_prompt`; `chat_template_kwargs`
  (for Gemma 4, `{"enable_thinking": false}` turns thinking off; it is on by default and the thoughts arrive as `reasoning_content`);
  `response_format: {"type": "json_object"}`; image parts as `image_url` (data URLs) when `--mmproj` is loaded.
- `POST /completion`: llama-server's raw completion, including prompts as token ids (what the benchmarks use).
- `GET /health`, `GET /props`, `GET /slots`, `GET /v1/models`, `GET /metrics`.
- The response carries llama-server's `timings` (prompt and predicted counts and per-second rates) and `usage`.

## Known differences from llama-server
| where | difference |
|---|---|
| `/v1/models` | `meta.ftype`, `meta.n_params`, `meta.size` and `tags` are not reported |
| `timings` | two extra fields when MTP is on: `draft_n` and `draft_n_accepted` (drafts proposed and accepted in the request) |

Everything a client reads today (`model`, `choices`, `usage`, `timings.prompt_n`, `timings.predicted_per_second`, `reasoning_content`,
`/health`, `/props.model_path`) is byte-compatible in shape.

## Security
The server has no authentication. Bind to loopback (the default) and put a reverse proxy with auth in front if you expose it.
