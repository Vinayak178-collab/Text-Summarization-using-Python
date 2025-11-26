# Text Summarizer (Python)

A Python-based text summarization system supporting **extractive** and **abstractive** methods, with a FastAPI backend and simple web UI.

## Features

- Extractive summarization using sentence embeddings (Sentence-Transformers) and centroid-based ranking.
- Placeholders for TextRank and additional extractive strategies.
- Abstractive summarization using pretrained transformer models (e.g., BART/T5 via HuggingFace).
- Support for plain text input; hooks for PDF upload and URL ingestion.
- REST API with FastAPI and a minimal HTML UI.

## Project structure

```text
text-summarizer/
├─ data/                   # sample datasets, small examples
├─ notebooks/              # experiments, EDA (optional, empty by default)
├─ src/
│  ├─ api/                 # FastAPI app and routes
│  ├─ models/              # model loading & wrappers
│  ├─ preprocessing/       # text cleaning, pdf parsing, sentence splitting
│  ├─ summarizers/         # extractive.py, abstractive.py
│  └─ utils/               # evaluation, helpers
├─ tests/                  # unit tests (basic examples)
├─ examples/               # sample article/summary files
├─ requirements.txt
└─ README.md
```

## Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK punkt data (for sentence tokenization) the first time you run:

```python
import nltk
nltk.download("punkt")
```

4. Run the API with uvicorn:

```bash
uvicorn src.api.main:app --reload
```

5. Open the browser at `http://127.0.0.1:8000` to use the simple UI, or call the API directly:

```bash
curl -X POST "http://127.0.0.1:8000/summarize" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"<long article text>\", \"mode\":\"extractive\", \"num_sentences\":3}"
```

## Main API endpoints

- `POST /summarize` – summarize provided text.
  - Body: `{ "text": "...", "mode": "extractive"|"abstractive", "num_sentences": 3, "max_length": 150, "min_length": 30 }`
  - Returns: `{ "summary": "...", "mode": "...", "details": {...} }`
- `GET /health` – simple health check.
- `GET /` – minimal HTML UI.

## Evaluation (ROUGE)

You can compute ROUGE scores between reference and generated summaries using `src.utils.evaluation.compute_rouge`:

```python
from src.utils.evaluation import compute_rouge

references = ["This is the reference summary."]
candidates = ["This is the generated summary."]
scores = compute_rouge(references, candidates)
print(scores)
```

Pair the files in `examples/sample_article.txt` and `examples/sample_summary.txt` with your model's outputs to sanity-check performance.

## Development milestones

- Basic preprocessing and extractive summarization.
- Abstractive summarization via pretrained models.
- Long-document handling via chunking / hierarchical summarization.
- Evaluation using ROUGE and human-readable examples.
- Dockerization and optional deployment.

## Notes

- GPU is recommended for abstractive summarization with large transformer models, but CPU-only setups work for extractive methods and small demos.
- Check model licenses (HuggingFace models) if you plan to use this system commercially.


