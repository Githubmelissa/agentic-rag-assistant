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
