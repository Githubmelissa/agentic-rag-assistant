# Agentic RAG Assistant for Company Reports and Sales Data

> Status: **in progress** (Stage 2 of 8: hybrid retrieval and reranking).

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
python -m assistant.retrieval "What was Kering's revenue in 2025?"                 # hybrid + rerank
python -m assistant.retrieval "What was Kering's revenue in 2025?" --mode bm25     # dense | bm25 | hybrid | hybrid_rerank
python -m assistant.retrieval "What was Kering's revenue in 2025?" --compare       # all four modes side by side
```

Design choices are explained in `docs/decisions.md`.

### Troubleshooting (Windows)

`OSError: [WinError 1114] ... torch\lib\c10.dll`, or Python closing silently with exit code
`-1073741819` when importing `onnxruntime`: the virtual environment was created from an
**Anaconda** Python. Anaconda ships older copies of the Visual C++ runtime DLLs that recent
PyTorch and ONNX Runtime cannot use. Create the environment from the official python.org
installer instead (check with `python -c "import sys; print(sys.version)"`: it must not say
"Anaconda"), and install the CPU build of PyTorch first:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all,dev]"
```

## Evaluation

TBD. All numbers will come from scripts in `eval/`, saved to `results/`.

## License

MIT. See `LICENSE`. Data sources and their licenses are listed in `data/README.md`.
