# RAG-based ARASAAC Pictogram Retrieval for AAC Communication

Project for the course "Big Data and Text Mining" — Giorgio Scavello (giorgio.scavello@studio.unibo.it).

## Overview

This project implements an end-to-end pipeline for retrieving [ARASAAC](https://arasaac.org/) pictograms from a natural-language query, for use in Augmentative and Alternative Communication (AAC) tools. The pipeline is:

```
user query → LLM query simplification (noise removal) → embedding-based retrieval → cross-encoder / VL reranking
```

A side goal of the project is comparing a local LLM (via [Ollama](https://ollama.com/)) against an in-browser LLM (via [WebLLM](https://github.com/mlc-ai/web-llm)) for the query-simplification step, to see whether the pipeline can run entirely client-side without a local LLM install.

All experimentation, analysis and narrative live in [`notebook_explained.ipynb`](notebook_explained.ipynb); shared helper code lives in [`BDATM/utils.py`](BDATM/utils.py).

## Repository structure

```
BDATM-project/
├── notebook_explained.ipynb   # main notebook: data, embeddings, retrieval, reranking, webapp
├── BDATM/
│   ├── utils.py                  # retrieval / reranking / evaluation / server helpers
│   ├── index.html                # frontend for the pictogram web app
│   ├── df_noisy_sentences.parquet  # LLM-generated "noisy" queries (see below)
│   ├── embeddings/                # local Milvus-lite vector DBs (Yuan, Jina text, Jina image)
│   ├── qwen_vl_reranker_repo/     # vendored Qwen3-VL-Reranker scripts
│   ├── scripts/                  # copy of the Qwen3-VL-Reranker inference code
│   └── jsons/
│       ├── retrieval/            # checkpoints for the retrieval-stage evaluation (Recall@k, etc.)
│       └── reranking/            # checkpoints for the 10 reranking pipeline comparisons
└── static/                       # served copy of BDATM/index.html for the web app
```

## Datasets

- **Sentences & concepts**: [`disi-unibo-nlp-students/aac_database`](https://huggingface.co/datasets/disi-unibo-nlp-students/aac_database) — sentences paired with their gold pictogram concepts (plus alternate acceptable candidates per concept). Since these sentences are direct pictogram→text translations rather than natural queries, an LLM (Qwen-3.5 9B) is used to generate more realistic "noisy" queries from them (`df_noisy_sentences.parquet`).
- **Pictograms**: [`disi-unibo-nlp-students/ARASAAC-Pictograms`](https://huggingface.co/datasets/disi-unibo-nlp-students/ARASAAC-Pictograms) — the full ARASAAC pictogram set (image, id, categories, keywords), used for building the retrieval passages/embeddings and for the web app's candidate pool.

## Setup

The notebook is written for a Colab-style environment but installs everything it needs:

```bash
pip install "transformers>=4.57.0" "qwen-vl-utils>=0.0.14" pandas numpy matplotlib scipy Pillow \
    huggingface-hub datasets sentence-transformers accelerate \
    langchain-core langchain-ollama langchain-community langchain-huggingface \
    "pymilvus==2.4.9" "milvus-lite==2.4.9" tqdm ollama flask flask-cors \
    fastapi "uvicorn[standard]" pydantic gdown "marshmallow==3.20.0" "torchao>=0.16.0" pyngrok
```

`torch`/`torchvision` are intentionally not pinned — use whatever build (CUDA/MPS/CPU) is already available in your environment.

You'll also need:
- **Ollama** running locally, with `qwen3.5:4b` pulled (used for query simplification throughout retrieval and reranking).
- A **Hugging Face token** with access to the two datasets above, set via an environment variable / secrets manager rather than hardcoded in the notebook.
- A **ngrok token** is necessary to run the web app and the WebLLM in Colab; set it via an environment variable / secrets manager rather than hardcoded in the notebook.

Run the notebook from the repository root (`BDATM-project/`) — all paths (e.g. `./BDATM/jsons/...`) are relative to that working directory.

## Pipeline stages (as evaluated in the notebook)

### 1. Embeddings (Section 3)
Two embedding models are compared:
- **Yuan-embedding-2.0-en** — text-only, symmetric embeddings (top of the MTEB retrieval leaderboard at the time of writing).
- **jina-embeddings-v5-omni-nano** — multimodal (image+text), asymmetric (separate query/document encoding), used both in a text-only mode and an image+text mode so pictograms without usable text metadata can still be embedded.

### 2. Retrieval (Section 4)
Each embedding model is evaluated with `Recall@k` (strict: exact gold pictogram id; relaxed: any accepted candidate id) under three query variants: the noisy sentence as-is, LLM-simplified (Ollama, Qwen3.5:4B), and WebLLM-simplified (Qwen2.5-7B-Instruct). **Result:** `jina-embeddings-v5-omni-nano` in image+text mode, used with no LLM polishing at all, performs best; LLM simplification helps the symmetric Yuan model but doesn't help (and can hurt) the already-asymmetric Jina model. WebLLM doesn't improve results enough to justify its much higher latency, so it's dropped from later steps.

### 3. Reranking (Section 5)
The top-2 retrievers (Yuan text, Jina image+text) are each reranked with:
- **ms-marco-MiniLM-L12-v2** — small, fast, pointwise cross-encoder.
- **jina-reranker-v3** — larger listwise reranker (limited to 64-document joint attention windows).
- **Qwen3-VL-Reranker-2B** — vision-language reranker that scores pictogram *images*, not just text (evaluated on a smaller sample due to cost).

Nine pipelines (`BDATM/jsons/reranking/*_checkpoint_1.json` … `_9.json`) cross these rerankers with the retrievers and LLM-query-simplification on/off; a tenth pipeline (`_checkpoint_10.json`) mixes the text and VL rerankers (50/50 blended scores) on top of the Jina image+text retriever. **Result:** for Jina image+text retrieval, neither reranker nor LLM polishing beats plain noisy retrieval; for Yuan retrieval, ms-marco-MiniLM gives a significant strict-recall improvement, and the VL reranker looks promising on the (small) sample tested, warranting further investigation.

### 4. Web app (Section 6)
A small FastAPI + HTML app (`create_pictogram_server` in `utils.py`, frontend in `BDATM/index.html`) lets a user type a query, retrieves and reranks candidate pictograms, and lets them pick which ones to keep for their own AAC boards.

## Evaluation checkpoints

Long-running evaluations (LLM calls, cross-encoder scoring) are checkpointed to JSON so they don't need to be recomputed:
- `BDATM/jsons/retrieval/` — Recall@k and retrieval-stage evaluation checkpoints.
- `BDATM/jsons/reranking/` — one checkpoint file per pipeline (see `run_configs`/`plot_pipeline_comparison` in `utils.py`), which can be reloaded and compared without rerunning any model.
If wanted, these checkpoints can be regenerated by running the notebook from scratch by just changing the name of the json, in the related cell of the notebook. However the checkpoints were included here to save time.