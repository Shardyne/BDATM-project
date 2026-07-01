import random
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer as ST
import pandas as pd
import warnings
import socket
import threading
from IPython.display import display, HTML
import time
from tqdm.notebook import tqdm
import ast


# check for unusable description text (mojibake, etc.)
def is_unusable_description(text):
    """Detect mojibake: strings where >40% of characters are non-ASCII."""
    if not isinstance(text, str) or len(text) == 0:
        return False
    return sum(1 for c in text if ord(c) > 127) / len(text) > 0.4

# checks for null concepts (keywords, descriptions or candidate pictograms) in a dataframe
def check_null_concepts(df):
    def analyze_row(concepts):
        has_null = False
        has_unusable = False
        if concepts is None or (hasattr(concepts, '__len__') and len(concepts) == 0):
            return True, False
        try:
            for entry in concepts:
                if entry is None:
                    has_null = True
                    continue
                if not isinstance(entry, dict):
                    continue
                # check concept keyword
                if is_unusable_description(entry.get('text', '')):
                    has_unusable = True
                # check selected pictogram description and keywords
                pictogram = entry.get('pictogram') or {}
                if is_unusable_description(pictogram.get('description', '')):
                    has_unusable = True
                for kw in pictogram.get('keywords', []):
                    if isinstance(kw, str) and is_unusable_description(kw):
                        has_unusable = True
                # check all candidate descriptions and keywords
                candidates = entry.get('candidates')
                if candidates is not None:
                    for candidate in candidates:
                        if not isinstance(candidate, dict):
                            continue
                        if is_unusable_description(candidate.get('description', '')):
                            has_unusable = True
                        for kw in candidate.get('keywords', []):
                            if isinstance(kw, str) and is_unusable_description(kw):
                                has_unusable = True
        except TypeError:
            has_null = True
        return has_null, has_unusable

    results = df['concepts'].apply(analyze_row)
    null_mask     = results.apply(lambda x: x[0])
    unusable_mask = results.apply(lambda x: x[1])

    print(f"Rows with at least one null concept entry:       {null_mask.sum()} / {len(df)}")
    print(f"Rows with at least one unusable description:     {unusable_mask.sum()} / {len(df)}")
    return null_mask, unusable_mask

# could be removed maybe
#def show_pictures(df_merged, n=3):
#
#    # indexes
#    ind=random.sample(list(range(len(df_merged))), k=n)
#    print(ind)
#
#    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
#
#    for ax, idx in zip(axes, ind):
#
#        image_bytes = df_merged['image'][idx]['bytes']
#        image = Image.open(io.BytesIO(image_bytes))
#        ax.imshow(image)
#        ax.axis('off')
#        title = df_merged['keywords'][idx]
#        ax.set_title(title)

# shows the pictograms with their keywords and meanings
def show_pictogram(df_images, n=3):
    """Show n random pictograms from df_images with all their meanings."""
    ind = random.sample(list(range(len(df_images))), k=n)
    rows = [df_images.iloc[i] for i in ind]
    print([int(r['pictogram_id']) for r in rows])

    _, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        image_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(image_bytes))

        raw = row['keywords']
        kws = list(raw) if raw is not None else []

        words    = ' / '.join(kw['keyword'] for kw in kws if kw.get('keyword')) or str(int(row['pictogram_id']))
        meanings = '\n'.join(kw['meaning'].rstrip('.') for kw in kws if kw.get('meaning'))

        ax.imshow(image)
        ax.axis('off')
        ax.set_title(words, fontsize=13, fontweight='bold')
        ax.set_xlabel(f"id:{int(row['pictogram_id'])}\n{meanings}", fontsize=8, labelpad=6)

    plt.tight_layout()
    plt.show()

# creates the embeddings for a list of texts and inserts them into Milvus in batches
def milvus_insert(milvus_client,
    collection_name: str,
    text_list: list[str],
    embedding_model,
    embeddings: list = None,
    checkpoint_file: str = None,
):
    embedding_batch_size = 512
    insert_batch_size = 64
    if checkpoint_file is None:
        checkpoint_file = f"{collection_name}_embeddings.npy"  # ← derive from collection


    if embeddings is None:
        all_texts = [t["text"] for t in text_list]
        
        # Resume from checkpoint if exists
        if os.path.exists(checkpoint_file):
            embeddings = np.load(checkpoint_file).tolist()
            start = len(embeddings)
            print(f"Resuming from {start}/{len(all_texts)}")
        else:
            embeddings = []
            start = 0

        for i in tqdm(range(start, len(all_texts), embedding_batch_size), desc="Embedding"):
            batch = all_texts[i : i + embedding_batch_size]
            batch_embeddings = embedding_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)

            # Save every 20 batches (~10k texts)
            if (i // embedding_batch_size) % 20 == 0:
                np.save(checkpoint_file, np.array(embeddings))

        np.save(checkpoint_file, np.array(embeddings))
        print("Embeddings saved!")

    # Insert into Milvus
    for row_id in tqdm(range(0, len(text_list), insert_batch_size), desc="Inserting"):
        batch_dicts = text_list[row_id : row_id + insert_batch_size]
        batch_texts = [t["text"] for t in batch_dicts]
        batch_embeddings = embeddings[row_id : row_id + insert_batch_size]
        batch_ids = [t["id"] for t in batch_dicts]
        batch_data = [
            {
                "id": id_,
                "text": text,
                "vector": vector,
            }
            for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
        ]
        milvus_client.insert(
            collection_name=collection_name,
            data=batch_data,
        )

# prompt for the LLMs to simplify a sentence into 3-5 core keywords for AAC pictograms
_SIMPLIFY_SYSTEM = (
    "You are an AAC communication expert. "
    "Extract the 3-5 CORE keywords that directly express the main meaning of the sentence as pictograms. "
    "Output ONLY the keywords, space-separated, no punctuation, no explanation, no extras. "
    "Do NOT brainstorm or list related concepts — keep ONLY what is explicitly stated or directly implied. "
    "Examples: 'I am tired of sitting down' → 'person tired sit'\n"
    "'pictograms for going to school' → 'person go school'\n"
    "'pictograms for math class at school' → 'person school math'"
)

# function used to call the LLMs via either ollama or webllm (browser-based) 
# backends to simplify a sentence into AAC keywords
def llm_simplify_query(sentence, model='qwen3.5:4b', backend='ollama', bridge=None, seed=42):
    """Simplify a sentence into AAC keywords using an LLM. Falls back to original on failure."""
    try:
        if backend == 'ollama':
            import ollama as _ollama
            resp = _ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": _SIMPLIFY_SYSTEM},
                    {"role": "user",   "content": f"/no_think\n{sentence}"},
                ],
                think=False,
                options={"temperature": 0, "seed": seed},
            )
            raw = re.sub(r'<think>[\s\S]*?</think>\s*', '', resp.message.content).strip()
            return raw if raw else sentence
        elif backend == 'webllm':
            bridge_data = _bridge_registry[bridge] if isinstance(bridge, int) else bridge
            full_prompt = f"/no_think\n\n{sentence}"
            raw = call_browser_llm(bridge_data, full_prompt, timeout=600)
            print(f"[webllm] got response: {raw[:80]!r}", flush=True)
            raw = re.sub(r'<think>[\s\S]*?</think>\s*', '', raw).strip()
            return raw if raw else sentence
    except Exception as e:
        print(f"[llm_simplify_query] fallback to original: {e}")
    return sentence

# remove the stopwords and normalize the phrase for better matching
def normalize_phrase(s):
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the|some|your|my|his|her|its|our|their)\b', '', s)
    return re.sub(r'\s+', ' ', s).strip()

# extract the 'keyword' strings from a pictogram's keywords list
def extract_keywords_list(keywords):
    if keywords is None:
        return []
    if isinstance(keywords, np.ndarray):
        keywords = keywords.tolist()
    if not isinstance(keywords, list):
        return []
    return [k['keyword'] for k in keywords if isinstance(k, dict) and 'keyword' in k]

# create a class to wrap the Jina embeddings model for text and image encoding
class JinaEmbeddings:
    def __init__(self, model_name):
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.model = ST(model_name, trust_remote_code=True, device=device,
                        model_kwargs={"default_task": "retrieval"})
        print(f"JinaEmbeddings loaded on {device}")

    def encode(self, inputs, task="retrieval"):
        """Encode text (str/list[str]) or images (bytes/PIL/list) into the shared embedding space."""
        if isinstance(inputs, str):
            inputs = [inputs]
        elif not isinstance(inputs, list):
            inputs = [inputs]
        inputs = [
            Image.open(io.BytesIO(x)).convert("RGB") if isinstance(x, bytes) else x
            for x in inputs
        ]
        return self.model.encode(inputs, task=task, normalize_embeddings=True)

    def embed_documents(self, texts):
        return self.model.encode_document(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode_query([text], normalize_embeddings=True).tolist()[0]
    
# sets the random seed for reproducibility across random, numpy, torch, and transformers libraries
def set_seed(seed=42):
    import torch
    from transformers import set_seed as hf_set_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    hf_set_seed(seed)

# a hit requires an exact gold pictogram match; recall, precision, and F1 are computed accordingly
def strict_retrieval_metrics(gt_ids, retrieved_ids):
    """Strict: gold = one pictogram per concept. Hit if gold pictogram is retrieved."""
    ret_set  = {int(r) for r in retrieved_ids}
    gold_set = {int(g) for g in gt_ids}
    n = len(gold_set)
    if n == 0:
        return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
    hits      = len(gold_set & ret_set)
    recall    = hits / n
    precision = hits / len(ret_set) if ret_set else 0.0
    f1 = 2*recall*precision/(recall+precision) if (recall+precision) > 0 else 0.0
    return {'recall': recall, 'precision': precision, 'f1': f1}

# a hit requires any valid pictogram (gold or candidate) to be retrieved; recall, precision, and F1 are computed accordingly
def relaxed_retrieval_metrics(gt_ids, retrieved_ids, candidate_id_sets):
    """Relaxed: gold = pictogram + all candidates per concept. Hit if any valid is retrieved."""
    ret_set = {int(r) for r in retrieved_ids}
    n = len(gt_ids)
    if n == 0:
        return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
    # per-concept valid set = gold pictogram ∪ all candidates
    valid_per_concept = [{int(g)} | {int(c) for c in cands}
                         for g, cands in zip(gt_ids, candidate_id_sets)]
    hits      = sum(1 for valid in valid_per_concept if valid & ret_set)
    recall    = hits / n
    all_valid = set().union(*valid_per_concept)
    precision = len(all_valid & ret_set) / len(ret_set) if ret_set else 0.0
    f1 = 2*recall*precision/(recall+precision) if (recall+precision) > 0 else 0.0
    return {'recall': recall, 'precision': precision, 'f1': f1}

# it embeds the query and then retrieves the top_k pictogram ids from Milvus using the provided client, collection, and embedding function
def retrieve(query, top_k, client, collection, embed_fn):
    vec = embed_fn(query)
    results = client.search(
        collection_name=collection,
        data=[vec],
        limit=top_k,
        output_fields=['id'],
    )[0]
    return [r['entity']['id'] for r in results]

# the function takes the dataframe of sentences and concepts, a dictionary of retrieval configurations,
#  and evaluates the retrieval performance using strict and relaxed metrics. 
# It samples a subset of sentences, retrieves pictograms for each sentence using the specified 
# retrieval configurations, and computes recall, precision, and F1 scores. The results are aggregated 
# across multiple random states, and confidence intervals are calculated. It supports a custom retrieval function for each configuration, 
# allowing for flexibility in how the retrieval is performed. Optionally, it can plot the results.
def evaluate_retrieval(df, retrieval_configs, n_samples=100, random_states=[42, 10, 99], k=200, plot=False, checkpoint_path=None):
    import json as _json
    from scipy.stats import t as t_dist

    _ckpt_done = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as _f:
            _ckpt_done = _json.load(_f)
        print(f"Checkpoint loaded: {len(_ckpt_done)} entries done")

    def _ckpt_save():
        if checkpoint_path:
            with open(checkpoint_path, 'w') as _f:
                _json.dump(_ckpt_done, _f, indent=2)

    all_runs = []

    for random_state in random_states:
        set_seed(random_state)
        print(f'\n{"="*60}')
        print(f'Random state: {random_state}')
        print(f'{"="*60}')
        sample = df.sample(n=min(n_samples, len(df)), random_state=random_state).reset_index(drop=True)

        for cfg in retrieval_configs:
            _key = f"{random_state}__{cfg['name']}"
            if _key in _ckpt_done:
                print(f"  [checkpoint] skipping {cfg['name']} (seed={random_state})")
                records = _ckpt_done[_key]
            else:
                records = []
                pbar = tqdm(sample.iterrows(), total=len(sample), desc=cfg['name'], unit='sent')
                for _, row in pbar:
                    concept_data      = row['concepts']
                    gt_ids            = [int(cd['pictogram']['id']) for cd in concept_data]
                    candidate_id_sets = [{int(c['id']) for c in cd['candidates']} for cd in concept_data]

                    query = row['sentence']
                    if 'llm_model' in cfg and 'retrieve_fn' not in cfg:
                        query = llm_simplify_query(
                            query,
                            model=cfg['llm_model'],
                            backend=cfg.get('backend', 'ollama'),
                            bridge=cfg.get('bridge'),
                            seed=random_state,
                        )
                        print(f"[llm_simplify] sentence={row['sentence']!r}  query={query!r}", flush=True)

                    if 'retrieve_fn' in cfg:
                        retrieved = cfg['retrieve_fn'](query, top_k=k)
                    else:
                        retrieved = retrieve(query, top_k=k,
                                             client=cfg['client'],
                                             collection=cfg['collection'],
                                             embed_fn=cfg['embed_fn'])

                    strict  = strict_retrieval_metrics(gt_ids, retrieved)
                    relaxed = relaxed_retrieval_metrics(gt_ids, retrieved, candidate_id_sets)
                    records.append({
                        'recall':            strict['recall'],
                        'precision':         strict['precision'],
                        'f1':                strict['f1'],
                        'relaxed_recall':    relaxed['recall'],
                        'relaxed_precision': relaxed['precision'],
                        'relaxed_f1':        relaxed['f1'],
                    })
                    avg = pd.DataFrame(records).mean()
                    pbar.set_postfix(rec=f"{avg['recall']:.2f}", rel=f"{avg['relaxed_recall']:.2f}")

                _ckpt_done[_key] = records
                _ckpt_save()

            rdf = pd.DataFrame(records)
            all_runs.append({
                'random_state': random_state, 'Model': cfg['name'],
                'recall':            rdf['recall'].mean(),
                'precision':         rdf['precision'].mean(),
                'f1':                rdf['f1'].mean(),
                'relaxed_recall':    rdf['relaxed_recall'].mean(),
                'relaxed_precision': rdf['relaxed_precision'].mean(),
                'relaxed_f1':        rdf['relaxed_f1'].mean(),
            })

    runs_df = pd.DataFrame(all_runs)
    n_runs  = len(random_states)
    # t critical value for 95% CI (two-tailed), df = n_runs - 1
    t_crit  = t_dist.ppf(0.975, df=max(n_runs - 1, 1))

    def ci95(mean, std):
        half = t_crit * std / np.sqrt(n_runs)
        return f"{mean:.3f} ± {half:.3f}"

    agg = runs_df.groupby('Model').agg(
        recall_mean=('recall','mean'),                       recall_std=('recall','std'),
        precision_mean=('precision','mean'),                 precision_std=('precision','std'),
        f1_mean=('f1','mean'),                               f1_std=('f1','std'),
        relaxed_recall_mean=('relaxed_recall','mean'),       relaxed_recall_std=('relaxed_recall','std'),
        relaxed_precision_mean=('relaxed_precision','mean'), relaxed_precision_std=('relaxed_precision','std'),
        relaxed_f1_mean=('relaxed_f1','mean'),               relaxed_f1_std=('relaxed_f1','std'),
    ).reset_index()

    summary = pd.DataFrame({
        'Model':             agg['Model'],
        'Recall':            agg.apply(lambda r: ci95(r.recall_mean,            r.recall_std),            axis=1),
        'Precision':         agg.apply(lambda r: ci95(r.precision_mean,         r.precision_std),         axis=1),
        'F1':                agg.apply(lambda r: ci95(r.f1_mean,                r.f1_std),                axis=1),
        'Relaxed Recall':    agg.apply(lambda r: ci95(r.relaxed_recall_mean,    r.relaxed_recall_std),    axis=1),
        'Relaxed Precision': agg.apply(lambda r: ci95(r.relaxed_precision_mean, r.relaxed_precision_std), axis=1),
        'Relaxed F1':        agg.apply(lambda r: ci95(r.relaxed_f1_mean,        r.relaxed_f1_std),        axis=1),
    })
    print(f'\nRetrieval Evaluation (mean ± 95% CI, t-dist df={n_runs-1}):')
    display(summary)

    if plot:
        models  = agg['Model'].tolist()
        x       = np.arange(len(models))
        width   = 0.5
        ci_s    = t_crit * agg['recall_std'].values        / np.sqrt(n_runs)
        ci_r    = t_crit * agg['relaxed_recall_std'].values / np.sqrt(n_runs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.bar(x, agg['recall_mean'],         width, yerr=ci_s, capsize=6, color='steelblue', alpha=0.8)
        ax2.bar(x, agg['relaxed_recall_mean'], width, yerr=ci_r, capsize=6, color='seagreen',  alpha=0.8)
        for ax, title in zip([ax1, ax2], ['Strict Recall (95% CI)', 'Relaxed Recall (95% CI)']):
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha='right')
            ax.set_ylabel('Recall'); ax.set_title(title); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    return runs_df, summary

# wraps a retrieval config with LLM query simplification at retrieval time, where the LLM simplifies the user query 
# into AAC keywords before embedding, improving retrieval quality for short/informal queries
def make_llm_simplify_config(base_cfg, llm_model='qwen3.5:4b', backend='ollama', bridge=None, seed=42, **kwargs):
    """Wrap a retrieval config with LLM query simplification at retrieval time.

    The LLM strips noise from the user query into AAC keywords before embedding,
    improving retrieval quality for short/informal user queries.
    """
    def retrieve_fn(sentence, top_k):
        simplified = llm_simplify_query(sentence, model=llm_model, backend=backend, bridge=bridge, seed=seed)
        print(f"[llm_simplify] sentence={sentence!r}  query={simplified!r}", flush=True)
        return retrieve(simplified, top_k=top_k,
                        client=base_cfg['client'], collection=base_cfg['collection'],
                        embed_fn=base_cfg['embed_fn'])
    retrieve_fn.__name__ = f"llm_simplify+{base_cfg['name']}"
    return {
        'name':        f"{base_cfg['name']} + LLM simplified ({llm_model})",
        'retrieve_fn': retrieve_fn,
        'llm_model':   llm_model,
        'backend':     backend,
        'bridge':      bridge,
        'seed':        seed,
    }

# plots the strict and relaxed Recall@k for each retrieval configuration, querying each model once at max(k_values) and truncating for efficiency
def plot_recall_at_k(df, retrieval_configs, k_values, n_samples=100, random_state=42, checkpoint_path=None):
    """Elbow plot of strict and relaxed Recall@k for each retrieval config.
    Queries each model once at max(k_values) then truncates — much faster than re-querying per k.
    """
    import json as _json
    set_seed(random_state)
    max_k = max(k_values)
    sample = df.sample(n=min(n_samples, len(df)), random_state=random_state).reset_index(drop=True)

    _ckpt_done = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as _f:
            _ckpt_done = _json.load(_f)
        print(f"Checkpoint loaded: {len(_ckpt_done)} configs done")

    def _ckpt_save():
        if checkpoint_path:
            with open(checkpoint_path, 'w') as _f:
                _json.dump(_ckpt_done, _f, indent=2)

    # ── fetch retrieved lists once at max_k ──────────────────────────────
    config_data = {}
    for cfg in retrieval_configs:
        if cfg['name'] in _ckpt_done:
            print(f"  [checkpoint] skipping {cfg['name']}")
            raw = _ckpt_done[cfg['name']]
            config_data[cfg['name']] = [
                (gt_ids, [set(c) for c in cands], retrieved)
                for gt_ids, cands, retrieved in raw
            ]
        else:
            rows_data = []
            for _, row in tqdm(sample.iterrows(), total=len(sample), desc=cfg['name']):
                gt_ids            = [int(cd['pictogram']['id']) for cd in row['concepts']]
                candidate_id_sets = [{int(c['id']) for c in cd['candidates']} for cd in row['concepts']]
                query = row['sentence']
                if 'llm_model' in cfg and 'retrieve_fn' not in cfg:
                    query = llm_simplify_query(
                        query,
                        model=cfg['llm_model'],
                        backend=cfg.get('backend', 'ollama'),
                        bridge=cfg.get('bridge'),
                        seed=random_state,
                    )
                    print(f"[llm_simplify] sentence={row['sentence']!r}  query={query!r}", flush=True)

                if 'retrieve_fn' in cfg:
                    retrieved = cfg['retrieve_fn'](query, top_k=max_k)
                else:
                    retrieved = retrieve(query, top_k=max_k,
                                         client=cfg['client'],
                                         collection=cfg['collection'],
                                         embed_fn=cfg['embed_fn'])
                rows_data.append((gt_ids, candidate_id_sets, retrieved))
            config_data[cfg['name']] = rows_data
            _ckpt_done[cfg['name']] = [
                (gt_ids, [list(c) for c in cands], retrieved)
                for gt_ids, cands, retrieved in rows_data
            ]
            _ckpt_save()

    # ── compute recall@k for each k by truncating ────────────────────────
    strict_results  = {cfg['name']: [] for cfg in retrieval_configs}
    relaxed_results = {cfg['name']: [] for cfg in retrieval_configs}

    for k in k_values:
        for cfg in retrieval_configs:
            s_recalls, r_recalls = [], []
            for gt_ids, candidate_id_sets, retrieved in config_data[cfg['name']]:
                truncated = retrieved[:k]
                s_recalls.append(strict_retrieval_metrics(gt_ids, truncated)['recall'])
                r_recalls.append(relaxed_retrieval_metrics(gt_ids, truncated, candidate_id_sets)['recall'])
            strict_results[cfg['name']].append(np.mean(s_recalls))
            relaxed_results[cfg['name']].append(np.mean(r_recalls))

    # ── plot ──────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for cfg in retrieval_configs:
        ax1.plot(k_values, strict_results[cfg['name']],  marker='o', label=cfg['name'])
        ax2.plot(k_values, relaxed_results[cfg['name']], marker='o', label=cfg['name'])
    for ax, title in zip([ax1, ax2], ['Strict Recall@k', 'Relaxed Recall@k']):
        ax.set_xlabel('k'); ax.set_ylabel('Recall'); ax.set_title(title)
        ax.grid(alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(retrieval_configs), 4),
               bbox_to_anchor=(0.5, -0.05), frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    return pd.DataFrame(strict_results, index=k_values), pd.DataFrame(relaxed_results, index=k_values)

_bridge_registry: dict = {}
# finds a free port on the local machine by binding a socket to port 0, which tells the OS to select an available port, and then returns that port number.
def find_free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# creates a Flask web server that acts as a bridge for WebLLM, allowing communication 
# between a browser-based LLM and a Python backend. It provides endpoints to get prompts,
#  send responses, list available models, and serve a worker page that runs the model in the 
# browser using WebGPU. The server listens on a specified port (or finds a free one) and uses 
# CORS to allow cross-origin requests.
def create_webllm_bridge(model_id="Qwen2.5-7B-Instruct-q4f16_1-MLC", port=None):
    if port is None:
        port = find_free_port()

    from flask import Flask, request, jsonify
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)

    bridge_data = {"prompt": None, "response": None, "metrics": None, "gen_id": 0}

    @app.route('/get_prompt')
    def get_prompt():
        p = bridge_data.get("prompt")
        bridge_data["prompt"] = None
        return jsonify({"prompt": p, "gen_id": bridge_data["gen_id"]})

    @app.route('/send_response', methods=['POST'])
    def send_response():
        data = request.json
        if data.get("gen_id") != bridge_data["gen_id"]:
            return "STALE"   # notebook already moved on, discard
        bridge_data["response"] = data.get("text")
        bridge_data["metrics"] = {
            "model":          data.get("model"),
            "latency_ms":     data.get("latency_ms"),
            "tokens_per_sec": data.get("tokens_per_sec")
        }
        return "OK"

    @app.route('/models')
    def list_models():
        return """
        <script type="module">
            import { prebuiltAppConfig } from "https://unpkg.com/@mlc-ai/web-llm?module";
            const ids = prebuiltAppConfig.model_list.map(m => m.model_id).sort();
            document.body.innerText = ids.join("\\n");
        </script>
        <pre>loading...</pre>
        """

    @app.route('/worker')
    def worker_page():
        return f"""
        <html>
        <body style="font-family: sans-serif; padding: 30px; background: #f4f4f9;">
            <h2>🖥️ WebLLM Browser Worker</h2>
            <p><b>Model:</b> <code>{model_id}</code></p>
            <div id="status" style="font-weight: bold; color: blue;">Initializing...</div>
            <div id="log" style="font-size: 0.8em; color: #666; margin-top: 10px; white-space: pre-wrap;"></div>
            <hr>
            <p>This tab uses your GPU (WebGPU) to run the model. Keep it open!</p>
            
            <script type="module">
                import {{ CreateMLCEngine }} from "https://unpkg.com/@mlc-ai/web-llm?module";
                
                const modelId = "{model_id}";

                async function init() {{
                    try {{
                        const engine = await CreateMLCEngine(modelId, {{ 
                            initProgressCallback: (p) => {{ 
                                document.getElementById('status').innerText = "Loading: " + Math.round(p.progress * 100) + "%";
                                document.getElementById('log').innerText = p.text;
                            }}
                        }});

                        document.getElementById('status').innerText = "🔥 Warming up GPU...";
                        document.getElementById('status').style.color = "orange";
                        await engine.chat.completions.create({{
                            messages: [{{ role: "user", content: "hi" }}],
                            max_tokens: 1
                        }});

                        document.getElementById('status').innerText = "🟢 Ready! Waiting for Notebook...";
                        document.getElementById('status').style.color = "green";

                        async function poll() {{
                            try {{
                                const res = await fetch('/get_prompt');
                                const data = await res.json();
                                if (data.prompt) {{
                                    const genId = data.gen_id;
                                    document.getElementById('status').innerText = "✍️ Generating...";
                                    document.getElementById('status').style.color = "orange";

                                    const start = performance.now();

                                    const reply = await engine.chat.completions.create({{
                                        messages: [
                                            {{ role: "system", content: "{_SIMPLIFY_SYSTEM.replace(chr(10), ' ')}" }},
                                            {{ role: "user", content: data.prompt }}
                                        ],
                                        max_tokens: 150,
                                        temperature: 0,
                                        enable_thinking: false
                                    }});

                                    const elapsed = performance.now() - start;
                                    const tokens = reply.usage?.completion_tokens || 0;

                                    const payload = JSON.stringify({{
                                        text: reply.choices[0].message.content,
                                        model: modelId,
                                        gen_id: genId,
                                        latency_ms: elapsed.toFixed(0),
                                        tokens_per_sec: (tokens / (elapsed / 1000)).toFixed(1)
                                    }});
                                    let sent = false;
                                    for (let attempt = 0; attempt < 10 && !sent; attempt++) {{
                                        try {{
                                            const sr = await fetch('/send_response', {{
                                                method: 'POST',
                                                headers: {{'Content-Type': 'application/json'}},
                                                body: payload
                                            }});
                                            const txt = await sr.text();
                                            if (txt === "STALE") {{
                                                console.warn("Response was stale, discarding.");
                                            }}
                                            sent = true;  // either OK or STALE, stop retrying
                                        }} catch (e) {{
                                            console.warn(`send_response attempt ${{attempt + 1}} failed, retrying...`, e);
                                            await new Promise(r => setTimeout(r, 2000));
                                        }}
                                    }}
                                    if (!sent) {{ console.error("send_response failed after 10 attempts"); }}

                                    document.getElementById('status').innerText = "🟢 Ready!";
                                    document.getElementById('status').style.color = "green";
                                }}
                            }} catch (e) {{ console.error("Poll error", e); }}
                            setTimeout(poll, 1000);
                        }}
                        poll();
                    }} catch (err) {{
                        document.getElementById('status').innerText = "❌ Error: " + err.message;
                        document.getElementById('status').style.color = "red";
                        document.getElementById('log').innerText = "Check if WebGPU is enabled in your browser settings.";
                    }}
                }}
                init();
            </script>
        </body>
        </html>
        """

    threading.Thread(
        target=lambda: app.run(port=port, debug=False, use_reloader=False),
        daemon=True
    ).start()

    from pyngrok import ngrok as _ngrok
    tunnel = _ngrok.connect(port)
    public_url = tunnel.public_url
    print(f"✅ Bridge Server active — model: {model_id}")
    print(f"   ngrok public URL: {public_url}")

    display(HTML(f"""
        <div style="padding: 20px; border: 3px solid #007acc; border-radius: 10px; background: #eef6ff;">
            <h3 style="margin-top:0;">Step 1: Open the Worker</h3>
            <p><b>Model:</b> <code>{model_id}</code></p>
            <p>Click this link to open the worker in a new tab:</p>
            <a href="{public_url}/worker" target="_blank"
               style="background: #007acc; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">
               OPEN BROWSER WORKER
            </a>
            <p style="margin-top: 15px; font-size: 0.85em; color: #555;">
                Browse available models:
                <a href="{public_url}/models" target="_blank">{public_url}/models</a>
            </p>
        </div>
    """))

    _bridge_registry[port] = bridge_data
    return app, bridge_data, port

# sends the prompt to the browser-based LLM via the bridge and waits for a response, with a timeout.
def call_browser_llm(bridge_data, prompt, timeout=300):
    bridge_data["gen_id"] = bridge_data.get("gen_id", 0) + 1
    bridge_data["response"] = None
    bridge_data["metrics"] = None
    bridge_data["prompt"] = prompt

    start = time.time()
    while True:
        if bridge_data["response"] is not None:
            break
        if time.time() - start > timeout:
            raise TimeoutError("LLM did not respond in time")
        time.sleep(0.2)

    m = bridge_data.get("metrics") or {}
    print(f"Model:     {m.get('model')}")
    print(f"Latency:   {m.get('latency_ms')} ms")
    print(f"Tokens/s:  {m.get('tokens_per_sec')}")

    return bridge_data["response"]

#def call_llm(sent, model="qwen2.5:7b-instruct", backend="ollama", bridge=None, context=None):
#
#    import textwrap
#
#    board_line = f"Symbols already on the board: {', '.join(context)}.\n" if context else ""
#    system = (
#        "You are an AAC pictogram retrieval assistant.\n"
#        "Extract 3-4 short concept queries from the user message for pictogram search.\n"
#        "Concepts can overlap or be variations of the same idea — prefer broader coverage.\n"
#        + (board_line if board_line else "") +
#        "If the user message is generic or vague (e.g. \"more\", \"add more\"), "
#        "use the board context above to infer related or complementary concepts.\n"
#        'Output ONLY valid JSON (no markdown): {"queries": ["concept 1", "concept 2", "concept 3"]}\n'
#        "Always output at least 3 queries.\n"
#        'Examples:\n'
#        '  message: "I want to go to the swimming pool"\n'
#        '  output: {"queries": ["swimming pool", "swimming", "water sport", "go"]}\n'
#        '  message: "she is eating pizza with friends"\n'
#        '  output: {"queries": ["eating", "pizza", "food", "friends"]}\n'
#        'Each query must be a short phrase or single concept.'
#    )
#    prompt = f"/no_think\nNow extract concepts for this query:\nQuery: \"{sent}\"\nOutput:"
#
#    if backend == "ollama":
#        import ollama
#        response = ollama.chat(
#            model=model,
#            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
#            think=False
#        )
#        return response.message.content, {
#            "model": model,
#            "latency_ms": round(response.eval_duration / 1e6, 0),
#            "tokens_per_sec": round(response.eval_count / (response.eval_duration / 1e9), 1)
#        }
#
#    elif backend == "webllm":
#        bridge_data = _bridge_registry[bridge] if isinstance(bridge, int) else bridge
#        full_prompt = "/no_think\n" + system + "\n\n" + f"Now extract concepts for this query:\nQuery: \"{sent}\"\nOutput:"
#        response = call_browser_llm(bridge_data, full_prompt)
#        response = re.sub(r'<think>[\s\S]*?</think>\s*', '', response).strip()
#        return response, bridge_data.get("metrics")

def _parse_llm_list(raw: str):
    import json as _json
    clean = re.sub(r'<think>[\s\S]*?</think>\s*', '', raw).strip()
    # try {"queries": [...]} JSON format (matches frontend output)
    try:
        clean_json = clean.replace('```json', '').replace('```', '').strip()
        parsed = _json.loads(clean_json)
        if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
            return [str(q) for q in parsed["queries"] if q]
    except (ValueError, KeyError):
        pass
    # fall back to bracket list format ["a", "b"]
    match = re.search(r'\[([^\[\]]*)\]', clean, re.DOTALL)
    if match:
        try:
            result = ast.literal_eval('[' + match.group(1) + ']')
            if isinstance(result, list):
                return [str(x) for x in result]
        except (ValueError, SyntaxError):
            pass
        items = [s.strip().strip('"\'\'') for s in match.group(1).split(',')]
        items = [s for s in items if s]
        if items:
            return items
    return None

def _run_baselines(retrieve_fn, reranker, sample, pool_k, rerank_n, compute_metrics=True):
    """Compute pool/sent/oracle modes once; cache per-row pools for LLM reuse.

    When compute_metrics=False, skips CE reranker calls (sent/oracle) and only
    builds rows_cache — use this when show_baselines=False to save time.
    """
    rows_cache = []  # (gt_ids, candidate_id_sets, sentence, pool, oracle_concepts)
    results = {k: {"strict": [], "relaxed": [], "strict_prec": [], "relaxed_prec": [], "strict_f1": [], "relaxed_f1": []}
               for k in ("pool", "sent", "oracle")}

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="baselines" if compute_metrics else "building cache"):
        concept_data      = row["concepts"]
        gt_ids            = [int(cd["pictogram"]["id"]) for cd in concept_data]
        candidate_id_sets = [{int(c["id"]) for c in cd["candidates"]} for cd in concept_data]
        sentence          = row["sentence"]
        oracle_concepts   = [cd["text"] for cd in concept_data]

        pool = retrieve_fn(sentence, top_k=pool_k)
        rows_cache.append((gt_ids, candidate_id_sets, sentence, pool, oracle_concepts))

        if compute_metrics:
            reranked_sent = reranker(sentence, pool, top_n=rerank_n)

            import math
            n_per_oracle = math.ceil(rerank_n / len(oracle_concepts))
            oracle_ids: set[int] = set()
            for concept in oracle_concepts:
                oracle_ids.update(reranker(concept, pool, top_n=n_per_oracle))

            for key, ranked in (("pool", pool[:rerank_n]), ("sent", reranked_sent),
                                 ("oracle", list(oracle_ids))):
                sm = strict_retrieval_metrics(gt_ids, ranked)
                rm = relaxed_retrieval_metrics(gt_ids, ranked, candidate_id_sets)
                results[key]["strict"].append(sm["recall"])
                results[key]["relaxed"].append(rm["recall"])
                results[key]["strict_prec"].append(sm["precision"])
                results[key]["relaxed_prec"].append(rm["precision"])
                results[key]["strict_f1"].append(sm["f1"])
                results[key]["relaxed_f1"].append(rm["f1"])

    return rows_cache, results


def _run_llm_simplify_retrieve_only_mode(rows_cache, retrieve_fn, reranker, pool_k, rerank_n, llm_model, backend, bridge, desc, seed=42):
    """LLM simplified keywords for retrieval only; original sentence used for CE reranking.

    Isolates the retrieval benefit of LLM simplification while keeping the CE
    query faithful to the original sentence for scoring.
    """
    results = {"strict": [], "relaxed": []}
    for gt_ids, candidate_id_sets, sentence, _, _ in tqdm(rows_cache, desc=desc):
        simplified = llm_simplify_query(sentence, model=llm_model, backend=backend, seed=seed)
        pool = retrieve_fn(simplified, top_k=pool_k)
        ranked = reranker(sentence, pool, top_n=rerank_n)
        sm = strict_retrieval_metrics(gt_ids, ranked)
        rm = relaxed_retrieval_metrics(gt_ids, ranked, candidate_id_sets)
        results["strict"].append(sm["recall"])
        results["relaxed"].append(rm["recall"])
    return results


def _run_llm_simplify_retrieve_mode(rows_cache, retrieve_fn, reranker, pool_k, rerank_n, llm_model, backend, bridge, desc, seed=42):
    """LLM simplifies the sentence; simplified query used for BOTH retrieval and CE reranking.

    Compared to _run_llm_simplify_mode (rerank only), this also changes the pool —
    so we can isolate how much LLM simplification helps at the retrieval stage.
    """
    results = {"strict": [], "relaxed": [], "strict_prec": [], "relaxed_prec": [], "strict_f1": [], "relaxed_f1": [],
               "pool_strict": [], "pool_relaxed": []}
    for gt_ids, candidate_id_sets, sentence, _, _ in tqdm(rows_cache, desc=desc):
        simplified = llm_simplify_query(sentence, model=llm_model, backend=backend, seed=seed)
        pool = retrieve_fn(simplified, top_k=pool_k)

        # pool recall with simplified retrieval
        pm = strict_retrieval_metrics(gt_ids, pool[:rerank_n])
        prm = relaxed_retrieval_metrics(gt_ids, pool[:rerank_n], candidate_id_sets)
        results["pool_strict"].append(pm["recall"])
        results["pool_relaxed"].append(prm["recall"])

        # CE reranks simplified pool with simplified query (full pipeline)
        ranked = reranker(simplified, pool, top_n=rerank_n)
        sm = strict_retrieval_metrics(gt_ids, ranked)
        rm = relaxed_retrieval_metrics(gt_ids, ranked, candidate_id_sets)
        results["strict"].append(sm["recall"])
        results["relaxed"].append(rm["recall"])
        results["strict_prec"].append(sm["precision"])
        results["relaxed_prec"].append(rm["precision"])
        results["strict_f1"].append(sm["f1"])
        results["relaxed_f1"].append(rm["f1"])
    return results


def _run_llm_simplify_mode(rows_cache, reranker, rerank_n, llm_model, backend, bridge, desc, seed=42):
    """LLM simplifies the sentence to AAC keywords; CE reranks the pool with that query.

    The simplified keyword query matches pictogram descriptions better than the
    full sentence, since AAC descriptions are short and keyword-style.
    One LLM call per sentence, then standard CE reranking.
    """
    results = {"strict": [], "relaxed": [], "strict_prec": [], "relaxed_prec": [], "strict_f1": [], "relaxed_f1": []}
    for gt_ids, candidate_id_sets, sentence, pool, _ in tqdm(rows_cache, desc=desc):
        simplified = llm_simplify_query(sentence, model=llm_model, backend=backend, seed=seed)
        print(f"[llm_simplify] sentence={sentence!r}  rerank_query={simplified!r}", flush=True)
        ranked = reranker(simplified, pool, top_n=rerank_n)
        sm = strict_retrieval_metrics(gt_ids, ranked)
        rm = relaxed_retrieval_metrics(gt_ids, ranked, candidate_id_sets)
        results["strict"].append(sm["recall"])
        results["relaxed"].append(rm["recall"])
        results["strict_prec"].append(sm["precision"])
        results["relaxed_prec"].append(rm["precision"])
        results["strict_f1"].append(sm["f1"])
        results["relaxed_f1"].append(rm["f1"])
    return results


def _run_llm_mode(rows_cache, reranker, rerank_n, llm_model, backend, bridge, desc):
    """Rerank pool candidates using per-concept queries (budget split across concepts)."""
    import math
    results = {"strict": [], "relaxed": []}
    for gt_ids, candidate_id_sets, sentence, pool, _ in tqdm(rows_cache, desc=desc):
        raw, _ = call_llm(sentence, model=llm_model, backend=backend, bridge=bridge)
        llm_concepts = _parse_llm_list(raw) or [sentence]
        tqdm.write(f"  [{sentence[:50]}] → {llm_concepts}")
        n_per = math.ceil(rerank_n / len(llm_concepts))
        llm_ids: set[int] = set()
        for concept in llm_concepts:
            llm_ids.update(reranker(concept, pool, top_n=n_per))
        ranked = list(llm_ids)
        results["strict"].append(strict_retrieval_metrics(gt_ids, ranked)["recall"])
        results["relaxed"].append(relaxed_retrieval_metrics(gt_ids, ranked, candidate_id_sets)["recall"])
    return results


def _run_llm_listwise_mode(rows_cache, reranker, rerank_n, pid_to_description, llm_model, backend, bridge, desc, ce_n=60):
    """Cross-encoder shortlists ce_n candidates, LLM reorders and we keep top rerank_n.

    ce_n > rerank_n means the LLM can surface items the CE ranked below rerank_n,
    actually changing the returned set and improving recall — not just reordering.
    One LLM call per sentence.
    """
    import json as _json
    import ollama as _ollama

    LISTWISE_SYSTEM = (
        "You are a pictogram reranker for an AAC communication system.\n"
        "Given a user query and a numbered list of pictogram descriptions, "
        "reorder them from most to least relevant for the query.\n"
        f"Output ONLY a JSON array of indices in your preferred order, e.g. [2, 0, 5, 1, ...].\n"
        "Include ALL indices exactly once. No explanation."
    )

    results = {"strict": [], "relaxed": [], "strict_prec": [], "relaxed_prec": [], "strict_f1": [], "relaxed_f1": []}
    for gt_ids, candidate_id_sets, sentence, pool, _ in tqdm(rows_cache, desc=desc):
        shortlist = reranker(sentence, pool, top_n=ce_n)
        entries = [f"{i}: {pid_to_description.get(pid, f'pictogram {pid}')}"
                   for i, pid in enumerate(shortlist)]
        user_prompt = f"/no_think\nQuery: \"{sentence}\"\n\n" + "\n".join(entries)

        try:
            if backend == "ollama":
                resp = _ollama.chat(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": LISTWISE_SYSTEM},
                        {"role": "user",   "content": user_prompt},
                    ],
                    think=False,
                )
                raw = resp.message.content.strip()
            else:
                raw = ""

            raw = re.sub(r'<think>[\s\S]*?</think>\s*', '', raw).strip()
            raw = raw.replace('```json', '').replace('```', '').strip()
            order = _json.loads(raw)
            if isinstance(order, list) and len(order) > 0:
                valid = [i for i in order if isinstance(i, int) and 0 <= i < len(shortlist)]
                seen = set(valid)
                for i in range(len(shortlist)):
                    if i not in seen:
                        valid.append(i)
                ranked = [shortlist[i] for i in valid][:rerank_n]
            else:
                ranked = shortlist[:rerank_n]
        except Exception:
            ranked = shortlist[:rerank_n]

        sm = strict_retrieval_metrics(gt_ids, ranked)
        rm = relaxed_retrieval_metrics(gt_ids, ranked, candidate_id_sets)
        results["strict"].append(sm["recall"])
        results["relaxed"].append(rm["recall"])
        results["strict_prec"].append(sm["precision"])
        results["relaxed_prec"].append(rm["precision"])
        results["strict_f1"].append(sm["f1"])
        results["relaxed_f1"].append(rm["f1"])
    return results


def _run_llm_rag_mode(rows_cache, reranker, pid_to_description, llm_model, backend, bridge, desc, ce_n=60):
    """RAG-style: CE shortlists ce_n candidates, LLM selects the relevant subset.

    The LLM decides how many pictograms to return (variable size), not constrained
    to rerank_n. Evaluated with strict/relaxed recall on the LLM's selection.
    """
    import json as _json
    import ollama as _ollama

    RAG_SYSTEM = (
        "You are an AAC pictogram selector. "
        "Given a sentence and a numbered list of pictogram descriptions, "
        "select ALL pictograms needed to visually communicate the sentence. "
        "AAC communication requires pictograms for EVERY concept: person, action, object, place, feeling, modifier. "
        "Select 5 to 10 pictograms. Err on the side of including more rather than fewer. "
        "Output ONLY a JSON array of indices, e.g. [2, 5, 11, 3, 8]. No explanation."
    )

    results = {"strict": [], "relaxed": [], "strict_prec": [], "relaxed_prec": [], "strict_f1": [], "relaxed_f1": [], "n_selected": []}
    for gt_ids, candidate_id_sets, sentence, pool, _ in tqdm(rows_cache, desc=desc):
        shortlist = reranker(sentence, pool, top_n=ce_n)
        entries = [f"{i}: {pid_to_description.get(pid, str(pid))}"
                   for i, pid in enumerate(shortlist)]
        user_prompt = f"/no_think\nSentence: \"{sentence}\"\n\n" + "\n".join(entries)

        selected = shortlist[:10]  # fallback
        try:
            if backend == "ollama":
                resp = _ollama.chat(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": RAG_SYSTEM},
                        {"role": "user",   "content": user_prompt},
                    ],
                    think=False,
                )
                raw = resp.message.content.strip()
            else:
                raw = "[]"

            raw = re.sub(r'<think>[\s\S]*?</think>\s*', '', raw).strip()
            raw = raw.replace('```json', '').replace('```', '').strip()
            indices = _json.loads(raw)
            if isinstance(indices, list) and len(indices) > 0:
                valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(shortlist)]
                selected = [shortlist[i] for i in valid]
        except Exception:
            pass

        sm = strict_retrieval_metrics(gt_ids, selected)
        rm = relaxed_retrieval_metrics(gt_ids, selected, candidate_id_sets)
        tqdm.write(f"  [{sentence[:50]}] GT={len(gt_ids)} sel={len(selected)} R={rm['recall']:.2f} P={rm['precision']:.2f} F1={rm['f1']:.2f}")
        results["strict"].append(sm["recall"])
        results["relaxed"].append(rm["recall"])
        results["strict_prec"].append(sm["precision"])
        results["relaxed_prec"].append(rm["precision"])
        results["strict_f1"].append(sm["f1"])
        results["relaxed_f1"].append(rm["f1"])
        results["n_selected"].append(len(selected))
    return results


def _run_llm_expand_mode(rows_cache, retrieve_fn, reranker, rerank_n, llm_model, backend, bridge, desc, concept_k=50):
    """LLM concepts expand the retrieval pool; original sentence reranks the union.

    Uses LLM concepts to retrieve additional candidates beyond the sentence pool,
    then reranks the combined pool with the original sentence query. This separates
    the LLM's role (diversifying candidates) from the reranker's role (scoring).
    """
    results = {"strict": [], "relaxed": []}
    for gt_ids, candidate_id_sets, sentence, pool, _ in tqdm(rows_cache, desc=desc):
        raw, _ = call_llm(sentence, model=llm_model, backend=backend, bridge=bridge)
        llm_concepts = _parse_llm_list(raw) or []
        tqdm.write(f"  [{sentence[:50]}] → {llm_concepts}")

        seen = set(pool)
        expanded = list(pool)
        for concept in llm_concepts:
            for pid in retrieve_fn(concept, top_k=concept_k):
                if pid not in seen:
                    expanded.append(pid)
                    seen.add(pid)

        ranked = reranker(sentence, expanded, top_n=rerank_n)
        results["strict"].append(strict_retrieval_metrics(gt_ids, ranked)["recall"])
        results["relaxed"].append(relaxed_retrieval_metrics(gt_ids, ranked, candidate_id_sets)["recall"])
    return results


def print_retrieval_results(results, rerank_n, name=None, modes=None, show_baselines=True):
    def avg(lst): return sum(lst) / len(lst) if lst else None
    header = f"── {name} " if name else "── "
    print(f"\n{header}{'─' * (60 - len(header))}")
    print(f'{"":50} {"Strict":>7} {"Relaxed":>8}')
    # (result_key, label, is_baseline, required_mode_in_modes)
    label_specs = [
        ("pool",                       f"Recall@{rerank_n} (pool — noisy)                        ", True,  None),
        ("sent",                       f"Recall@{rerank_n} (CE  — noisy, pool+rerank)             ", True,  None),
        ("llm_simplify",               f"Recall@{rerank_n} (CE  — LLM rerank, noisy pool)         ", False, "llm_simplify"),
        ("llm_simplify_retrieve:pool", f"Recall@{rerank_n} (pool — LLM polished)                  ", False, "llm_simplify_retrieve"),
        ("llm_simplify_retrieve_only", f"Recall@{rerank_n} (CE  — LLM pool, noisy rerank)         ", False, "llm_simplify_retrieve_only"),
        ("llm_simplify_retrieve",      f"Recall@{rerank_n} (CE  — LLM polished, full pipeline)    ", False, "llm_simplify_retrieve"),
    ]
    for key, label, is_baseline, req_mode in label_specs:
        if is_baseline and not show_baselines:
            continue
        if req_mode is not None and modes is not None and req_mode not in modes:
            continue
        if ":" in key:
            base, subkey = key.split(":", 1)
            d = results.get(base, {})
            s = avg(d.get(f"{subkey}_strict") or [])
            r = avg(d.get(f"{subkey}_relaxed") or []) or 0
        else:
            d = results.get(key, {})
            s = avg(d.get("strict") or [])
            r = avg(d.get("relaxed") or []) or 0
        if s is None:
            continue
        print(f"{label}: {s:.3f}   {r:.3f}")

VL_RERANK_PROMPT = "Retrieve images or text relevant to the user's query."


class QwenVLReranker:
    """Wraps Qwen3VLReranker (from the model's scripts/) to expose a .predict(pairs) interface.

    pairs: list of (query_str, {'text': keywords_str, 'image': PIL.Image})
    All pairs must share the same query (as in rerank_pictograms).
    Load the model scripts by cloning the repo or passing scripts_dir.
    """

    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-VL-Reranker-2B",
                 scripts_dir: str = ".", **model_kwargs):
        import sys
        sys.path.insert(0, scripts_dir)
        from scripts.qwen3_vl_reranker import Qwen3VLReranker as _Qwen3VLReranker
        self._model = _Qwen3VLReranker(model_name_or_path=model_name_or_path, **model_kwargs)
        print(f"QwenVLReranker loaded from {model_name_or_path}")

    def predict(self, pairs: list, prompt: str = VL_RERANK_PROMPT, fps: float = 1.0,
                batch_size: int = 16, **kwargs) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        documents = []
        for _, doc in pairs:
            if isinstance(doc, dict):
                entry = {}
                if doc.get('text'):
                    entry['text'] = doc['text']
                if doc.get('image') is not None:
                    entry['image'] = doc['image']
                documents.append(entry)
            else:
                documents.append({'text': str(doc)})

        scores = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            inputs = {
                'instruction': prompt,
                'query':       {'text': query},
                'documents':   batch,
                'fps':         fps,
            }
            scores.extend(self._model.process(inputs))
        return scores

def rerank_pictograms(
    query: str,
    candidate_ids: list[int],
    top_n: int,
    pid_to_description: dict = None,
    text_reranker=None,
    pid_to_image: dict = None,
    image_reranker=None,
    image_weight: float = 0.4,
    batch_size: int = 64,
) -> list[int]:
    """Rerank candidates using a text cross-encoder and/or a VL cross-encoder.

    - text_reranker:  CrossEncoder scoring (query, keyword_text) pairs
    - image_reranker: CrossEncoder (e.g. Qwen3-VL-Reranker-2B) scoring
                      (query, {'text': keywords, 'image': PIL}) pairs
    - pid_to_image values: {'bytes': bytes, 'keywords': str}
    - When both rerankers provided, scores are min-max normalised then combined.
    """
    scored_ids, text_pairs, image_pairs, unscored_ids = [], [], [], []

    for pid in candidate_ids:
        desc  = pid_to_description.get(pid, '') if pid_to_description else ''
        entry = pid_to_image.get(pid)             if pid_to_image      else None
        if desc or entry:
            scored_ids.append(pid)
            text_pairs.append((query, desc) if desc else None)
            if entry:
                img = Image.open(io.BytesIO(entry['bytes'])).convert("RGB")
                image_pairs.append((query, {'text': entry.get('keywords', ''), 'image': img}))
            else:
                image_pairs.append(None)
        else:
            unscored_ids.append(pid)

    if not scored_ids:
        return candidate_ids[:top_n]

    text_scores  = np.zeros(len(scored_ids))
    image_scores = np.zeros(len(scored_ids))

    if text_reranker is not None:
        valid = [(i, p) for i, p in enumerate(text_pairs)
                 if p is not None and p[1].strip() and not is_unusable_description(p[1])]
        if valid:
            idxs, pairs = zip(*valid)
            pairs_list = list(pairs)
            try:
                scores = text_reranker.predict(pairs_list, batch_size=batch_size)
            except RuntimeError:
                # fall back to per-pair scoring to avoid 0-length tensor batches
                scores = []
                for pair in pairs_list:
                    try:
                        scores.append(text_reranker.predict([pair], batch_size=1)[0])
                    except RuntimeError:
                        scores.append(0.0)
            raw = np.array(scores, dtype=float)
            raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            for i, s in zip(idxs, raw):
                text_scores[i] = s

    if image_reranker is not None:
        valid = [(i, p) for i, p in enumerate(image_pairs) if p is not None]
        if valid:
            idxs, pairs = zip(*valid)
            if hasattr(image_reranker, 'process'):
                documents = [{'text': p[1].get('text', ''), 'image': p[1].get('image')} for p in pairs]
                scores = image_reranker.process({
                    'instruction': VL_RERANK_PROMPT,
                    'query': {'text': pairs[0][0]},
                    'documents': documents,
                })
                raw = np.array(scores, dtype=float)
            else:
                raw = np.array(image_reranker.predict(list(pairs), prompt=VL_RERANK_PROMPT,
                                                       batch_size=batch_size), dtype=float)
            raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            for i, s in zip(idxs, raw):
                image_scores[i] = s

    if text_reranker is not None and image_reranker is not None:
        final = (1 - image_weight) * text_scores + image_weight * image_scores
    elif text_reranker is not None:
        final = text_scores
    else:
        final = image_scores

    ranked = sorted(zip(scored_ids, final), key=lambda x: x[1], reverse=True)
    result = [pid for pid, _ in ranked]
    result.extend(unscored_ids)
    return result[:top_n]


def _plot_run_configs(agg, t_crit, n_seeds):
    configs = agg['config'].unique().tolist()
    modes   = agg['mode'].unique().tolist()
    x       = np.arange(len(modes))
    n_cfg   = len(configs)
    width   = 0.8 / max(n_cfg, 1)
    colors  = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, 2.5 * len(modes)), 5))
    for i, cfg in enumerate(configs):
        sub  = agg[agg['config'] == cfg].set_index('mode').reindex(modes)
        ci_s = t_crit * sub['strict_std'].fillna(0).values  / np.sqrt(n_seeds)
        ci_r = t_crit * sub['relaxed_std'].fillna(0).values / np.sqrt(n_seeds)
        off  = (i - (n_cfg - 1) / 2) * width
        kw   = dict(width=width, capsize=5, color=colors[i % len(colors)], alpha=0.8, label=cfg)
        ax1.bar(x + off, sub['strict_mean'].fillna(0).values,  yerr=ci_s, **kw)
        ax2.bar(x + off, sub['relaxed_mean'].fillna(0).values, yerr=ci_r, **kw)

    for ax, title in zip([ax1, ax2], ['Strict Recall (95% CI)', 'Relaxed Recall (95% CI)']):
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=20, ha='right')
        ax.set_ylabel('Recall')
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_configs(
    configs,
    df,
    retrieve_fn=None,
    reranker=None,
    pid_to_description=None,
    n_samples=200,
    pool_k=200,
    rerank_n=30,
    random_state=42,
    random_states=None,
    modes=None,
    show_baselines=True,
    plot=False,
    checkpoint_path=None,
):
    """Compute pool/sent/oracle baselines once, then run the selected LLM modes per config.

    random_states: list of seeds to average over (supersedes random_state when given).
    modes: list of modes to run. Supported values:
      'llm_simplify'               – LLM rerank only, noisy pool
      'llm_simplify_retrieve_only' – LLM pool, noisy rerank
      'llm_simplify_retrieve'      – LLM pool + LLM rerank (full pipeline)
    Defaults to all three when None.
    show_baselines: whether to include pool/sent rows in the results table and plot.
    plot: if True, show a bar chart (modes on x-axis, error bars = 95% CI across seeds).

    Returns: (runs_df, summary)
    """
    from scipy.stats import t as t_dist

    if retrieve_fn is None:
        retrieve_fn = retrieve
    if reranker is None:
        reranker = rerank_description_crossencoder
    if modes is None:
        modes = ["llm_simplify", "llm_simplify_retrieve_only", "llm_simplify_retrieve"]
    if random_states is None:
        random_states = [random_state]

    _mode_labels = {
        "pool":                       f"Pool@{rerank_n}",
        "sent":                       f"CE sent@{rerank_n}",
        "llm_simplify":               "LLM rerank",
        "llm_simplify_retrieve_only": "LLM pool",
        "llm_simplify_retrieve":      "LLM full",
    }

    import json as _json

    # ── checkpoint setup ─────────────────────────────────────────────────────
    def _ckpt_key(seed, cfg_name, mode):
        return f"{seed}__{cfg_name}__{mode}"

    _ckpt_done = {}
    if checkpoint_path:
        import os as _os
        if _os.path.exists(checkpoint_path):
            with open(checkpoint_path) as _f:
                _ckpt_done = _json.load(_f)
            print(f"Checkpoint loaded: {len(_ckpt_done)} entries done")

    def _ckpt_save():
        if checkpoint_path:
            with open(checkpoint_path, "w") as _f:
                _json.dump(_ckpt_done, _f, indent=2)

    all_runs      = []
    all_sentences = []

    # restore all_runs from checkpoint
    if _ckpt_done:
        for key, entry in _ckpt_done.items():
            all_runs.append(entry)

    for seed in random_states:
        set_seed(seed)
        print(f'\n{"="*60}')
        print(f'Seed: {seed}')
        print(f'{"="*60}')
        sample = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

        _baseline_key = _ckpt_key(seed, "__baselines__", "all")
        if show_baselines and _baseline_key in _ckpt_done:
            print(f"  [checkpoint] skipping baselines (seed={seed})")
            base_results = _ckpt_done[_baseline_key]["_raw"]
            rows_cache, _ = _run_baselines(retrieve_fn, reranker, sample, pool_k, rerank_n, compute_metrics=False)
        else:
            print("Computing baselines...")
            rows_cache, base_results = _run_baselines(retrieve_fn, reranker, sample, pool_k, rerank_n, compute_metrics=show_baselines)
            if show_baselines:
                _ckpt_done[_baseline_key] = {"_raw": base_results}
                _ckpt_save()

        for cfg in configs:
            name      = cfg.get("name", "config")
            llm_model = cfg.get("llm_model", "qwen2.5:7b-instruct")
            backend   = cfg.get("backend", "ollama")
            bridge    = cfg.get("bridge")

            results = dict(base_results)

            if "llm_simplify" in modes:
                _key = _ckpt_key(seed, name, "llm_simplify")
                if _key in _ckpt_done:
                    print(f"  [checkpoint] skipping llm_simplify ({name}, seed={seed})")
                    results["llm_simplify"] = _ckpt_done[_key]["_raw"]
                else:
                    results["llm_simplify"] = _run_llm_simplify_mode(
                        rows_cache, reranker, rerank_n, llm_model, backend, bridge,
                        desc=f"LLM rerank ({name})", seed=seed
                    )
                    s_vals = results["llm_simplify"].get("strict",  [])
                    r_vals = results["llm_simplify"].get("relaxed", [])
                    entry = {
                        "seed": seed, "config": name,
                        "mode": _mode_labels.get("llm_simplify", "llm_simplify"),
                        "strict": float(np.mean(s_vals)) if s_vals else None,
                        "relaxed": float(np.mean(r_vals)) if r_vals else None,
                        "_raw": results["llm_simplify"],
                    }
                    _ckpt_done[_key] = entry
                    all_runs.append(entry)
                    _ckpt_save()

            if "llm_simplify_retrieve_only" in modes:
                _key = _ckpt_key(seed, name, "llm_simplify_retrieve_only")
                if _key in _ckpt_done:
                    print(f"  [checkpoint] skipping llm_simplify_retrieve_only ({name}, seed={seed})")
                    results["llm_simplify_retrieve_only"] = _ckpt_done[_key]["_raw"]
                else:
                    results["llm_simplify_retrieve_only"] = _run_llm_simplify_retrieve_only_mode(
                        rows_cache, retrieve_fn, reranker, pool_k, rerank_n, llm_model, backend, bridge,
                        desc=f"LLM pool, noisy rerank ({name})", seed=seed
                    )
                    s_vals = results["llm_simplify_retrieve_only"].get("strict",  [])
                    r_vals = results["llm_simplify_retrieve_only"].get("relaxed", [])
                    entry = {
                        "seed": seed, "config": name,
                        "mode": _mode_labels.get("llm_simplify_retrieve_only", "llm_simplify_retrieve_only"),
                        "strict": float(np.mean(s_vals)) if s_vals else None,
                        "relaxed": float(np.mean(r_vals)) if r_vals else None,
                        "_raw": results["llm_simplify_retrieve_only"],
                    }
                    _ckpt_done[_key] = entry
                    all_runs.append(entry)
                    _ckpt_save()

            if "llm_simplify_retrieve" in modes:
                _key = _ckpt_key(seed, name, "llm_simplify_retrieve")
                if _key in _ckpt_done:
                    print(f"  [checkpoint] skipping llm_simplify_retrieve ({name}, seed={seed})")
                    results["llm_simplify_retrieve"] = _ckpt_done[_key]["_raw"]
                else:
                    results["llm_simplify_retrieve"] = _run_llm_simplify_retrieve_mode(
                        rows_cache, retrieve_fn, reranker, pool_k, rerank_n, llm_model, backend, bridge,
                        desc=f"LLM pool+rerank ({name})", seed=seed
                    )
                    s_vals = results["llm_simplify_retrieve"].get("strict",  [])
                    r_vals = results["llm_simplify_retrieve"].get("relaxed", [])
                    entry = {
                        "seed": seed, "config": name,
                        "mode": _mode_labels.get("llm_simplify_retrieve", "llm_simplify_retrieve"),
                        "strict": float(np.mean(s_vals)) if s_vals else None,
                        "relaxed": float(np.mean(r_vals)) if r_vals else None,
                        "_raw": results["llm_simplify_retrieve"],
                    }
                    _ckpt_done[_key] = entry
                    all_runs.append(entry)
                    _ckpt_save()

            print_retrieval_results(results, rerank_n, name=name, modes=modes, show_baselines=show_baselines)

            # Collect per-seed per-mode means for entries not yet in all_runs (baselines)
            tracked = []
            if show_baselines:
                tracked += [("pool", base_results["pool"]), ("sent", base_results["sent"])]
            for key, d in tracked:
                s_vals = d.get("strict",  [])
                r_vals = d.get("relaxed", [])
                all_runs.append({
                    "seed":    seed,
                    "config":  name,
                    "mode":    _mode_labels.get(key, key),
                    "strict":  float(np.mean(s_vals)) if s_vals else None,
                    "relaxed": float(np.mean(r_vals)) if r_vals else None,
                })

    # ── aggregate across seeds ────────────────────────────────────────────
    runs_df = pd.DataFrame(all_runs)
    n_seeds = len(random_states)
    t_crit  = t_dist.ppf(0.975, df=max(n_seeds - 1, 1))

    def _ci95(mean, std):
        if pd.isna(std) or n_seeds < 2:
            return f"{mean:.3f}"
        half = t_crit * std / np.sqrt(n_seeds)
        return f"{mean:.3f} ± {half:.3f}"

    agg = runs_df.groupby(["config", "mode"], sort=False).agg(
        strict_mean=("strict",  "mean"), strict_std=("strict",  "std"),
        relaxed_mean=("relaxed", "mean"), relaxed_std=("relaxed", "std"),
    ).reset_index()

    summary = pd.DataFrame({
        "Config":         agg["config"],
        "Mode":           agg["mode"],
        "Strict Recall":  agg.apply(lambda r: _ci95(r.strict_mean,  r.strict_std),  axis=1),
        "Relaxed Recall": agg.apply(lambda r: _ci95(r.relaxed_mean, r.relaxed_std), axis=1),
    })
    print(f'\nResults (mean ± 95% CI across {n_seeds} seed(s), t df={n_seeds - 1}):')
    display(summary)

    if plot:
        _plot_run_configs(agg, t_crit, n_seeds)

    return runs_df, summary