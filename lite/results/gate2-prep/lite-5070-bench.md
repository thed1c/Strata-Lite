## S: speed through the API (frozen requests as captured, max_tokens 300; cold, then the same request again)

| server | request | prompt n | prefill tok/s | warm cache n | decode tok/s cold / warm | MTP accepted | VRAM MiB |
|---|---|---|---|---|---|---|---|
| lite-5070 | p1 | 603 | 411.6 | 602 | 60.0 / 78.7 | 134/159 | 11130 |
| lite-5070 | p4 | 6766 | 1854.0 | 6769 | 51.0 / 58.1 | 146/318 | 11244 |
| lite-5070 | long-a | — | HTTP 400 exceed_context_size_error | | | | |
| lite-5070 | image-x1 | 289 | 188.5 |  | 29.2 /  | 6/9 | 11636 |
| lite-5070 | image-x8 | 1885 | 277.0 |  | 39.4 /  | 7/12 | 11646 |
