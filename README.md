# Agentic RAG Assistant for Company Reports and Sales Data

> Status: **in progress** (Stage 1 of 8: ingestion and indexes).

An assistant that answers business questions from two sources and shows its work:

1. **Document RAG** over public company reports, with cited page numbers.
2. **Data agent** that answers questions about a retail sales table with safe, read-only SQL.
3. **Router agent (LangGraph)** that picks the right tool (documents, SQL, both, or refuse).

## Quickstart (development)

PowerShell (Windows):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
python -m pytest
python -m ruff check .
```

bash (Linux/macOS):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env
python -m pytest
python -m ruff check .
```

Install every dependency group (needed from Stage 1 on): `pip install -e ".[all,dev]"`.

### Build the search indexes

Put the report PDFs in `data/raw/` (see `data/README.md`), then:

```
python -m assistant.ingest                     # chunk + embed + index (first run downloads the model)
python -m assistant.retrieval "What was Kering's revenue in 2025?"
python -m assistant.retrieval "What was Kering's revenue in 2025?" --mode bm25
```

Design choices are explained in `docs/decisions.md`.

## Evaluation

TBD. All numbers will come from scripts in `eval/`, saved to `results/`.

## License

MIT. See `LICENSE`. Data sources and their licenses are listed in `data/README.md`.
