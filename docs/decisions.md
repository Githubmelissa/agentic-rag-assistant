# Design decisions

Short notes on each choice, the reason, and the alternative. Updated at every stage.

## Stage 0: skeleton

- **`src/` layout + `pyproject.toml`.** Tests import the installed package, which catches
  packaging mistakes early. Alternative: flat layout (simpler, but hides import errors).
- **Dependency groups** (`rag`, `sql`, `agent`, `api`, `dev`). Each stage installs only what
  it needs; CI does not need PyTorch for lint. Alternative: one `requirements.txt`.
- **pydantic-settings for config.** One typed, validated object; the API key is hidden from
  `repr`. Alternative: `os.getenv` everywhere (untyped, scattered).

## Stage 1: ingestion and indexes

- **Page numbers = PDF page numbers (1-based)**, the number shown in the PDF viewer's page
  box. Printed page numbers (e.g. "46" at the bottom of Kering's p.48) are stored as
  `page_label` when the PDF defines them, but Hermès and LVMH do not, so citations use the
  PDF page number to stay consistent and easy to check.
- **Text extracted by blocks** (PyMuPDF), so paragraphs stay together. Alternative:
  plain page text (loses paragraph boundaries) or layout-aware parsers such as `unstructured`
  (heavier, slower).
- **Repeated headers/footers removed**: a block whose digit-free text appears on ≥30% of a
  document's pages is boilerplate (e.g. "KERING — 2025 Universal Registration Document").
  Removing it keeps it from matching every query.
- **Pages with < 30 characters skipped** (covers, full-page photos).
- **Chunk size in tokens of the embedding model's tokenizer, default 500 (overlap 80).**
  The brief suggested ~600, but `bge-small-en-v1.5` reads at most 512 tokens: anything
  beyond is silently cut off before embedding. 500 keeps every chunk fully visible to the
  model. The ingest CLI prints a warning if a chunk size above the model limit is used
  (relevant for the 1000-token ablation in Stage 6, where dense search will only see the
  first 512 tokens of each chunk; BM25 sees everything). Alternative: a long-context
  embedding model (e.g. 8k tokens), at the cost of size and speed.
- **Sentence-aware greedy packing.** Chunks end on sentence boundaries; a sentence longer
  than a chunk (tables, lists) is split into word windows. **Overlap** repeats the last
  sentences of the previous chunk so a fact on a boundary appears whole somewhere.
- **Chunks never cross pages**, so each chunk has exactly one page to cite. Trade-off: a
  paragraph continuing on the next page is split in two.
- **Token counter injected** into the chunker: tests use a word counter, real runs use the
  model tokenizer. Tests stay fast and never download a model.
- **`chunks.jsonl` is the single source of truth.** Both indexes are built from it, so dense
  and BM25 always search the same chunks (needed for fair comparison and for fusion).
- **Chroma with our own embeddings** (`embedding_function=None`) and cosine distance;
  embeddings are normalised. Chroma never downloads its own default model.
- **BM25 rebuilt in memory** from `chunks.jsonl` at load time (about a second) instead of a
  pickle: no stale or version-fragile files, nothing unsafe to unpickle.
- **BM25 tokenizer**: lowercase, words and numbers (keeps "17.2" and "1,234"), small
  stop-word list, no stemming. Alternative: stemming (matches "emission"/"emissions",
  but can merge unrelated words).
- **One index folder per chunk size** (`indexes/cs500/`), so the three chunk-size variants
  for the ablation can coexist.
- **Query instruction for BGE**: queries (not documents) get the prefix recommended by the
  model authors, "Represent this sentence for searching relevant passages: ".

### Known issues seen in the real PDFs (inputs for the failure analysis)

- The LVMH annual report is a magazine-style document: 58 of its 184 pages have almost no
  text, and some titles are extracted with spaced letters ("A n n u a l R e p o r t").
- Tables are extracted as flat text, row by row, so numbers lose their column headers.

## Stage 2: hybrid retrieval and reranking

- **Reciprocal Rank Fusion, implemented by hand** (`retrieval/hybrid.py`):
  `score(d) = Σ 1 / (k + rank)`, ranks starting at 1. It uses positions only, because
  cosine scores (~0.7–0.8) and BM25 scores (0–40) are on different scales and cannot be
  added. A chunk found by both retrievers gets two contributions and rises.
- **k = 60**, the value from the original RRF paper (Cormack, Clarke & Büttcher, 2009).
  A large k makes rank 1 and rank 5 count almost the same (1/61 vs 1/65), so *agreement
  between retrievers* matters more than being first in one list. A unit test shows the
  difference with k = 0.
- **Each retriever contributes its top 20** before fusion. More candidates raise recall but
  give the reranker more work. Configurable (`RETRIEVAL_CANDIDATES`).
- **Cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the **top 20
  fused** results. A bi-encoder (the dense index) embeds query and chunk separately: fast,
  but coarse. A cross-encoder reads them together and judges whether the chunk answers the
  question: precise, but it costs one model call per (query, chunk) pair, so it is only
  applied to a short list. Its latency is measured in Stage 6. Alternative: a larger
  reranker (e.g. bge-reranker-base), slower on CPU.
- **One `retrieve(query, mode=...)` entry point** with four modes, so the ablation runs the
  exact same code path for each mode. Models are loaded once and reused.
- **Deterministic tie-breaking** (first appearance in RRF; stable sort in reranking), so
  evaluation runs are reproducible.
- **Deferred: removing section headings** that start many chunks ("6 FINANCIAL STATEMENTS
  AS OF DECEMBER 31, 2025"). They may make chunks look alike to the dense model, but the
  change will only be made if the Stage 6 evaluation shows it helps.

## Stage 3: answers with citations

- **LLM: Ollama with `qwen2.5:3b` by default (free, local).** No account or API key, and
  data never leaves the laptop. Trade-off: on a laptop CPU it is slow (measured in Stage 6)
  and a 3B model follows instructions less reliably than large hosted models. Alternatives
  with the same code: `groq` (free hosted tier, e.g. Llama 3.3 70B, fast; needs a key)
  or any OpenAI-compatible endpoint. Switching is one line in `.env`.
- **Provider-agnostic wrapper with plain `httpx`** (`llm.py`): one `complete(system, user)`
  method, returning text, token counts and latency. No vendor SDKs: one small dependency and
  the request format is visible. A `MockLLM` makes tests and CI run without any model.
- **Ollama context window set explicitly** (`num_ctx = 4096`). Ollama's default window is
  small and silently cuts off long prompts; 5 passages of ≤ 500 tokens plus instructions
  need ~3,000 tokens. 8192 was tried first but crashed Ollama once for lack of memory on a 16 GB laptop with other apps open; 4096 fits our prompts and uses less RAM.
- **Temperature 0**: answers as deterministic as possible, so evaluation runs are
  repeatable.
- **Grounding prompt** (`agent/prompts.py`): use only the context, cite every factual
  sentence with the passage label, copy numbers exactly, and reply with one fixed sentence
  ("I could not find this in the documents.") when the answer is missing. A fixed sentence
  makes refusals easy to detect and to score.
- **Citations use the PDF page number** (`[kering_urd_2025 p.47]`); the printed page is
  shown next to it in the output (`printed p.45`) when the PDF defines one.
- **Citation checking**: every label cited by the model is matched against the passages it
  was actually given. Labels that were not in the context are reported as
  `invalid_citations` (a sign of hallucination). This is a cheap, automatic faithfulness
  signal; the LLM-as-judge in Stage 6 checks whether the cited passage really supports the
  sentence.
- **Top 5 passages as context** (`ANSWER_TOP_K`). More passages raise the chance the answer
  is included, but make prompts longer (slower on CPU) and can distract a small model.

## Stage 4: sales database and SQL tool

- **One table plus one view.** `sales` keeps every line with flags (`is_cancelled`,
  `is_product`); the view `sales_clean` keeps only real product sales. The LLM writes
  simpler, safer SQL against the view, and questions about cancellations remain possible.
  Cleaning decisions: `docs/data_notes.md`.
- **Precomputed `line_total`** and ISO-text dates: fewer chances for the LLM to get revenue
  or date filters wrong.
- **Guard in four layers** (`tools/sql_guard.py`): (1) sqlglot parse, exactly one
  SELECT/UNION statement, no write or admin node anywhere in the tree (catches
  `WITH ... DELETE`); (2) table allow-list (`sales`, `sales_clean`, CTEs) and a deny-list
  of file/extension functions; (3) LIMIT added or lowered to `SQL_ROW_LIMIT`; (4) read-only
  connection (`mode=ro` + `PRAGMA query_only`) with a time limit. The read-only connection
  alone already blocks writes (a test proves it); the other layers block reads of other
  tables, runaway queries, and give the LLM precise error messages.
- **We execute the SQL that sqlglot regenerates**, not the raw text: what runs is exactly
  what was validated. Checked that regeneration only changes cosmetics (e.g. `strftime` →
  `STRFTIME`), not results.
- **Indexes on date, country and customer only; none on `stock_code`.** With a
  stock_code index, SQLite answered "top products" (GROUP BY stock_code over 1M rows) by
  walking the index with a random row lookup each time: 2.6 s instead of 1.0 s for a plain
  scan, which exceeded the time limit on a laptop. `ANALYZE` did not change the plan. A
  single-product lookup without the index costs ~0.2 s, an acceptable trade-off.
- **Time limit 15 s** (`SQL_TIMEOUT_SECONDS`): 5 s stopped legitimate full-table scans on a
  laptop CPU (2-3x slower than the development machine). The limit exists to stop runaway
  queries (e.g. infinite recursive CTEs), not normal aggregations.
- **Time limit with a progress handler**: SQLite has no query timeout, so a callback runs
  every 10,000 VM steps and aborts after `SQL_TIMEOUT_SECONDS`. A test stops an infinite
  recursive CTE.
- **Retry loop**: the guard's or SQLite's error message is sent back to the LLM with its
  previous query, at most `SQL_MAX_RETRIES = 2` more attempts. Setting it to 0 gives the
  "without retry" ablation.
- **Schema in a Markdown file** (`tools/schema.md`) inserted into the prompt, plus three
  worked examples (few-shot). Alternative: introspect the schema automatically; a written
  description can also carry business rules (e.g. Ireland is 'EIRE').
- **Two LLM calls per question**: one writes the SQL, one turns the result rows into a
  sentence. The rows themselves are always returned too, so the answer can be checked.
- **Mock LLM writes a fixed SQL query**, so the full pipeline runs in CI without a model.

## Stage 5: router agent and traces

- **LangGraph state graph** with four nodes (`route`, `doc_rag`, `sql_agent`,
  `synthesize`) and conditional edges. The state is a typed dict carried between nodes.
  Alternative: a plain Python `if/else`; LangGraph makes the flow explicit, drawable
  (Mermaid) and easy to extend (e.g. a "clarify" node), which is what agent frameworks
  are for. Diagram: `docs/architecture.md`.
- **LLM router with four labels** and one example per label. The reply is parsed
  tolerantly ("Sales data", "out-of-scope.", "Category: both"); the first label found wins.
- **Fallback route `both`** when the reply has no valid label: searching both sources costs
  time, but refusing a valid question is worse. The trace records `parsed_ok: false`, so
  these cases are visible and counted in the evaluation.
- **Query decomposition for `both`** (`split` node): added after a real failure. For
  "Compare Kering's revenue in 2025 with the gift shop's total revenue in 2011", retrieval
  found the right pages (p.13 and p.47 in the top 2), but the document tool received the
  whole question, saw that the gift shop was not in the reports, and refused everything.
  Now an extra LLM call writes one self-contained sub-question per source ("REPORTS: ...",
  "SALES: ..."); if the reply cannot be parsed, both tools get the original question.
  **Hardened after the Stage 6 agent run** (see the failure log): a sub-question may be
  `NONE` (that source is not needed, so the tool is skipped), and a sub-question that
  mentions a number or a company absent from the user's question is dropped, because the
  3B model sometimes answers with the example from its own prompt. When nothing valid is
  left, only the document tool runs, on the original question: it can answer "not found",
  whereas the SQL tool always returns some number. The check compares only numbers and a
  fixed list of company names, so it cannot judge wording; a legitimate sub-question that
  adds an implicit year (e.g. "in 2025" when the user only said "last year") would be
  dropped too, which costs that half of the answer but never invents one.
- **`both` runs the two tools one after the other**, not in parallel: on a laptop CPU the
  local LLM is the bottleneck, so parallel calls would not be faster.
- **No LLM in `synthesize`.** Single-source answers are returned as they are. For `both`,
  the two answers are placed side by side with a fixed template ("From the company
  reports: ... From the gift-shop sales data: ..."). An LLM merge was tried first, but on a
  real run it turned a correct GBP 9,471,580.96 into "approximately GBP 947,158": letting a
  small model rewrite numbers that are already verified adds risk and ~15 s for no gain.
  Trade-off: the combined answer reads less fluently.
- **Fixed refusal for out-of-scope questions**, written in the code, not generated: a
  refusal should never invent anything and costs no LLM call.
- **Traces as JSON files** in `logs/` (gitignored): node, latency, inputs/outputs summary,
  token counts, retrieved chunk IDs and scores, SQL attempts and errors. Full passages and
  full result tables are not copied, to keep files small. These traces feed the latency
  and token metrics in Stage 6.
- **Dependencies injected into `build_graph`** (retriever, LLM, database path), so tests run
  every route with fakes in a fraction of a second.

## Stage 6: evaluation

- **Evaluation sets drafted by Claude, verified by Melissa** (`python -m eval.verify`).
  Every item starts as `"verified": false`; only verified items are scored. Results made
  with `--include-unverified` (dry runs) are labelled and never written to the README.
- **Several gold pages per question.** A figure such as Kering's 2025 revenue appears on
  about ten pages (key figures, income statement, segment notes). Retrieval is counted as
  a hit when *any* page that states the fact is retrieved. The draft lists every page
  where the answer appears within ~200 characters of its context; the verifier removes
  pages that do not really answer.
- **Page-level retrieval metrics**: chunks are reduced to unique (report, page) pairs
  before computing recall@k and MRR, so two chunks from the same page do not use two slots.
- **SQL execution accuracy, column-subset match**: both queries are executed; same number
  of rows, and every gold column must appear among the predicted columns with the same
  values (floats compared to the cent). A query returning (month, revenue) when the gold
  returns (month) is correct. Row order matters only for questions that ask for an order.
  String match on SQL text would reject correct queries written differently.
- **Evaluation questions differ from the prompt's few-shot examples** and from the
  questions used while building, so the scores are not inflated by tuning on them.
- **Answer correctness, two ways**: (a) automatic, the gold headline number (or name)
  must appear in the answer; (b) LLM-as-judge with a fixed rubric (correct, faithful to
  cited pages). The judge is the same 3B local model: free but weak, so 20 judgements are
  checked by hand and the agreement rate is reported.
- **Resumable runs**: each step caches per-question results in `results/raw/` (not
  committed: it contains report text). A laptop run interrupted after an hour restarts
  where it stopped.
- **The SQL evaluation skips the answer sentence** (`generate_answer=False`): execution
  accuracy only needs the rows, and it halves the LLM calls.
- **Verification found 4 flawed SQL items (out of 20), fixed before any run:** three
  ambiguous questions ("different countries" when the field contains 'Unspecified' and
  'European Community'; a cancellation total whose sign was not stated; "average unit
  price", which can mean per line 3.23, per unit sold 1.82 or per product 3.80) were
  reworded; and the schema notes told the model to subtract *all* cancellation lines for
  net revenue, although some are fees (AMAZONFEE -£182k) rather than cancelled sales:
  the rule now says product cancellations only. Also, the scoring now treats '12' and 12
  as equal (strftime returns text).
- **Document set verification (40 items):** 13 gold-page lists corrected by hand: wrong
  pages removed (e.g. doc_02 p.66, where 1,631 is Eyewear *revenue*, not operating
  income; doc_06 p.393, which shows the dividend split by payment date, not the €3.00
  ordinary dividend; name pages that only mention a person in passing or in a former
  role) and missing pages added where the fact is worded differently ("75% of
  production" vs "75% of objects"; "€80.8bn" vs "€80.8 billion"). Two questions were
  ambiguous between growth at constant and at current exchange rates and were reworded.
  The mixed questions reuse the verified document answers and pages.
- **Billions count in the automatic answer check:** the reports write "€16 billion" for
  €16,002 million, so an amount in billions matches when, converted to millions, it is
  within 1% of the gold figure. Approximate statements ("over €11 billion" for €11,333
  million, 3% off) still do not match; small numbers (percentages, per-share amounts)
  need an exact match.
- **Chunk size stays 500 although 1000 scored best in retrieval** (hybrid_rerank recall@5
  0.975 vs 0.925, i.e. 2 of 40 questions). bge-small reads at most 512 tokens, so the end
  of a 1000-token chunk is not embedded (dense recall is the same at 500 and 1000); and
  5 chunks of 1000 tokens (~5,000 tokens) exceed the 4,096-token context used with Ollama
  on a 16 GB laptop. The gain comes from BM25 and the reranker, not from better
  embeddings. With 40 questions, one question is 2.5 points: small differences are noise.
- **Retry on empty result (added after the baseline SQL run).** The baseline retry loop
  only reacts to errors and guard rejections; in the first run every wrong query ran
  without error, so retries changed nothing (55% with and without). A query that returns
  no rows or only NULLs is now also sent back to the model with a generic hint (check the
  table, the filters, the exact spelling of text values). If every attempt is empty, the
  empty result is kept, since "no data" can be the true answer. The idea came from
  looking at the evaluation failures, so it is reported as an extra row next to the
  baseline (`r2e`), never in place of it, and its gain is optimistic (no held-out set).
  Setting: `SQL_RETRY_ON_EMPTY`.
  **Result: no gain** (55% in all three configurations). It fired on 2 questions:
  sql_06 corrected 'GERMANY' to UPPER(country) = 'GERMANY' but then added an unwanted
  GROUP BY month (12 rows instead of one total); sql_12 kept querying `sales_clean` for
  cancellations three times. It also raised p95 latency from 18 s to 42 s. The option
  stays in the code (tested) but is **off by default**: the evidence does not support
  turning it on. The remaining errors are reasoning errors of the 3B model; the next
  lever would be a larger model, not more retries.
- **Validating the sub-questions of the `split` node (after the first agent run).** The
  router sends an unsure question to `both`; `split` then has to invent a sales
  sub-question for a question that has none, and the 3B model copies the example in its
  prompt instead (7 wrong answers, see failure log 16). Two changes, both general: the
  prompt now allows `NONE` for a source and forbids inventing a topic, company or year;
  and `drop_reason()` rejects a sub-question that is not a question, or that contains a
  number or a company name absent from the user's question (fixed company list, so the
  check cannot raise a false alarm on wording). A dropped reports sub-question falls back
  to the user's own question, because the reports tool can answer "not found", whereas
  the SQL tool always returns some number. Replaying the check on the 19 recorded splits
  of the first run: all 10 mixed questions keep both halves (no false rejection) and 8 of
  the 9 misrouted document questions lose their invented sales half. The ninth (doc_39,
  "What share of LVMH's 2025 revenue came from the United States?") survives, because its
  invented sales sub-question reuses words from the question: the check catches invented
  facts, not invented intent.
- **Keeping the first agent run.** The fix above was written after reading the first
  run's traces, so the first results are kept in `results/agent_v1.json` and printed in
  the README next to the new ones. Only the 19 questions routed to `both` were re-run
  (~35 min); the cached results of the other 41 questions are untouched, so the two runs
  differ only where the fix applies.
- **Hand-check sheet for the judge.** The 20 sampled judgements are graded from
  `results/raw/judge_review.md`, written by `eval.judge`: it quotes the text of the pages
  the answer cited (with the reference number marked), so faithfulness can be judged
  without opening the PDFs. The sheet deliberately omits the judge's own scores, so the
  human labels are not anchored on them; the CSV still shows them next to the cells to
  fill, so the check is blind only if the sheet is used first. The sheet lives in
  `results/raw/` (not committed) because it quotes report text.

## Stage 7 - service, container, CI

- **FastAPI over Flask**: the request and response models are the pydantic models already
  used for the settings, so validation and the OpenAPI page at `/docs` come for free. The
  interactive page is also the cheapest demo of the project for someone who will not clone
  it.
- **The agent is built once, at startup** (FastAPI `lifespan`), not per request: loading
  the embedding and reranking models takes several seconds and a few hundred MB of RAM.
- **A broken setup does not kill the app**: if the indexes or the LLM are missing, the API
  still starts, `/health` reports the exact error and `/ask` answers 503. A container that
  exits at boot tells the user nothing; this one says "build the indexes first".
- **One question at a time**, with a lock, and `429` for a second question instead of a
  queue: with a 3B model on a laptop CPU, two parallel answers are not twice as fast, they
  are both slow, and Ollama would load a second context (the out-of-memory crash of Stage
  3). The endpoints are plain `def`, so FastAPI runs them in a worker thread and `/health`
  stays responsive while an answer is being produced.
- **`extra="forbid"` on the request**: a misspelled field (`"questions"`) is a 422 error,
  not a silently ignored parameter.
- **No authentication**: a local demo. Exposing it would need at least an API key, a rate
  limit and a request-size limit; this is written in the README rather than half-built.
- **The image carries the code only.** The PDFs, the indexes and the sales database are
  mounted from the host (read-only for the data, so the container cannot modify them), and
  the CPU-only PyTorch index is used in the build: the default Linux wheels carry CUDA
  (~2.5 GB) for a project that runs on a laptop CPU.
- **Ollama stays on the host** rather than being a second container: the model is already
  downloaded on the machine, and a container would download its own copy (2 GB) and
  compete for the same RAM. The API reaches it through `host.docker.internal`.
- **CI installs a `ci` extra without PyTorch.** The test suite never loads a real model
  (fakes follow the `Embedder` and `Reranker` protocols), so the job needs no torch, no
  network and about a minute. The trade-off is that a missing real-model dependency would
  not be caught by CI; it is caught by the Docker build, which installs `[all]`.
- **Tests ignore the local `.env`.** `tests/conftest.py` builds every `Settings` with
  `_env_file=None`. Without it the suite reads whatever the developer configured, so it
  can pass on a fresh clone (and in CI) and fail on the machine where the project actually
  runs - which is exactly what happened when the API tests were added (failure log 19).
- **The Dockerfile is verified in CI, not locally.** The development machine has no
  Docker engine, so a second CI job builds the image and starts it with the fake LLM and
  calls `/health`. Shipping an unbuilt Dockerfile in a portfolio project would be worse
  than having none; this way every push proves it still builds. The limit is stated in
  the README: the container has never answered a real question.
