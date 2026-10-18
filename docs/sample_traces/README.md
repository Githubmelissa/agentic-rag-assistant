# Sample traces

Real traces copied from `logs/` (not committed), recorded on 2026-10-18 with qwen2.5:3b
via Ollama on a laptop CPU. Each file shows the route, every node with its latency and
token counts, the passages retrieved, the SQL attempts, and the final answer.

| File | Route | Notes |
|---|---|---|
| `documents_route.json` | documents | Hermès 2025 revenue, correct, cited PDF p.8. The 158 s total includes loading the models on the first question. |
| `sales_data_route.json` | sales_data | Country with most orders in 2011: United Kingdom (16,461 orders, `COUNT(DISTINCT invoice)`). |
| `out_of_scope_route.json` | out_of_scope | Fixed refusal, no tool called, 3.3 s. |
| `both_route_llm_merge_error.json` | both | Kept as a documented failure: `split` and both tools were correct (EUR 14,675 million, cited p.13; GBP 9,471,580.96), but the old LLM merge step wrote "approximately GBP 947,158". The merge is now a fixed template (failure log #14). |
