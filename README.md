# Agentic RAG Assistant for Company Reports and Sales Data

> Status: **in progress** (Stage 0 of 8: project skeleton).

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

## Evaluation

TBD. All numbers will come from scripts in `eval/`, saved to `results/`.

## License

MIT. See `LICENSE`. Data sources and their licenses are listed in `data/README.md`.
