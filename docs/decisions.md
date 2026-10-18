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
