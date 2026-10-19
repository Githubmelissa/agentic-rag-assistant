Only questions verified by hand are used; unverified items are excluded.

| set | drafted | verified (used) |
|---|---|---|
| doc | 40 | 40 |
| sql | 20 | 20 |
| mixed | 10 | 10 |
| oos | 10 | 10 |

### Retrieval (40 verified document questions, page level)

| chunk size | mode | recall@3 | recall@5 | recall@10 | MRR | median latency |
|---|---|---|---|---|---|---|
| 300 | dense | 72% | 75% | 85% | 0.549 | 0.05 s |
| 300 | bm25 | 70% | 85% | 98% | 0.522 | 0.01 s |
| 300 | hybrid | 82% | 90% | 95% | 0.719 | 0.06 s |
| 300 | hybrid_rerank | 82% | 92% | 100% | 0.741 | 2.08 s |
| 500 | dense | 62% | 62% | 80% | 0.577 | 0.05 s |
| 500 | bm25 | 80% | 88% | 100% | 0.659 | 0.01 s |
| 500 | hybrid | 85% | 90% | 95% | 0.743 | 0.06 s |
| 500 | hybrid_rerank | 82% | 92% | 100% | 0.694 | 3.61 s |
| 1000 | dense | 62% | 62% | 78% | 0.577 | 0.05 s |
| 1000 | bm25 | 80% | 92% | 98% | 0.652 | 0.00 s |
| 1000 | hybrid | 82% | 85% | 95% | 0.710 | 0.05 s |
| 1000 | hybrid_rerank | 85% | 98% | 98% | 0.757 | 3.58 s |

![Recall@5 by retrieval mode](results/recall_at_5_by_mode.png)

### Text-to-SQL (20 verified questions, ollama:qwen2.5:3b)

| max retries | retry on empty result | execution accuracy | query ran | guard rejections (share of attempts) | questions with a retry | median latency |
|---|---|---|---|---|---|---|
| 2 | no | 55% | 100% | 0% | 0 | 11.16 s |
| 0 | no | 55% | 100% | 0% | 0 | 10.93 s |
| 2 | yes | 55% | 100% | 0% | 2 | 11.23 s |

The first two rows are the baseline. Retry-on-empty-result (last row) was added after looking at the baseline failures, on the same questions, so its gain is optimistic: there is no separate held-out set.

### Routing (80 verified questions, ollama:qwen2.5:3b)

Overall accuracy: **75%** (unparseable router replies: 0).

| expected route | n | accuracy |
|---|---|---|
| documents | 40 | 70% |
| sales_data | 20 | 75% |
| both | 10 | 100% |
| out_of_scope | 10 | 70% |

### End-to-end answers (verified questions, ollama:qwen2.5:3b)

| metric | n | value |
|---|---|---|
| Document questions: gold number/name in the answer | 40 | 60% |
| Document questions: cites a gold page | 40 | 57% |
| Document questions: answered "not found" | 40 | 5% |
| Mixed questions: routed to both | 10 | 100% |
| Mixed questions: reports part correct | 10 | 80% |
| Mixed questions: SQL part correct | 10 | 70% |
| Out-of-scope questions: refused | 10 | 100% |
| LLM judge: correct (answered questions) | 35 | 63% |
| LLM judge: faithful to cited pages | 35 | 54% |

The LLM judge is the same small local model as the system; it is imperfect (it can be lenient or misread numbers). Hand check of 20 judgements: agreement 80% (correctness 90%, faithfulness 70%).

#### Latency per route (laptop CPU, local LLM)

| route | n | median | p95 |
|---|---|---|---|
| both | 19 | 133.2 s | 180.85 s |
| documents | 31 | 117.61 s | 155.32 s |
| out_of_scope | 9 | 3.48 s | 4.26 s |
| sales_data | 1 | 25.41 s | 25.41 s |

Tokens per query (input + output, all LLM calls): median 2929.0, mean 2697.6 (n = 60).

First run, before the split-validation fix (ollama:qwen2.5:3b, same questions): document answers 52%, mixed reports part 80%, mixed SQL part 70%, out-of-scope refusals 100%. The fix drops sub-questions that the split model invents (see docs/decisions.md); it was written after reading the first run's traces, so only the questions routed to `both` were re-run.
