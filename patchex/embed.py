"""ESM-2 embedding, matching the regime PatchEX was trained under.

Training used a precomputed cache built by `extract_esm_cache_padded.py`. Three
properties of that cache are load-bearing and are reproduced exactly here:

1. `padding="max_length"` to MAX_LEN=1000 — NOT per-batch dynamic padding. The
   patch stack pools over all 40 patch slots, so the padded positions are part
   of the computation, and dynamic padding changes the answer.
2. ESM weights in fp32, forward under autocast(fp16) on CUDA, hidden states
   stored as fp16 — the HF Trainer `fp16=True` path.
3. `truncation=True` at 1000 residues; longer proteins are truncated, not
   chunked.

Deviating from any of these silently shifts predictions, so `embed_sequences`
is the only supported way to featurise input for the released checkpoints.
"""
from __future__ import annotations

import numpy as np
import torch

ESM_ID = "facebook/esm2_t30_150M_UR50D"
D_ESM = 640
MAX_LEN = 1000

_CACHE: dict = {}


def load_esm(esm: str = ESM_ID, device: str | None = None):
    """Load (tokenizer, model) once per process; cached across calls."""
    from transformers import AutoTokenizer, EsmModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    key = (esm, device)
    if key not in _CACHE:
        tok = AutoTokenizer.from_pretrained(esm)
        model = EsmModel.from_pretrained(esm).eval().to(device)
        _CACHE[key] = (tok, model, device)
    return _CACHE[key]


@torch.no_grad()
def embed_sequences(sequences, esm: str = ESM_ID, device: str | None = None,
                    batch_size: int = 8, max_length: int = MAX_LEN,
                    fp16: bool = True, progress: bool = False):
    """Embed protein sequences into the padded ESM-2 features PatchEX expects.

    Args:
        sequences: list of amino-acid strings.
        batch_size: lower this if you run out of GPU memory; results are
            batch-size independent because padding is to a fixed length.

    Returns:
        embeds [N, max_length, 640] float32, mask [N, max_length] int64.
    """
    tok, model, device = load_esm(esm, device)
    seqs = [str(s).strip().upper() for s in sequences]
    n = len(seqs)
    embeds = np.zeros((n, max_length, D_ESM), dtype=np.float16)
    masks = np.zeros((n, max_length), dtype=np.uint8)
    use_ac = bool(fp16) and device.startswith("cuda")

    for st in range(0, n, batch_size):
        chunk = seqs[st:st + batch_size]
        b = tok(chunk, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_length)
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_ac):
            out = model(**b).last_hidden_state
        embeds[st:st + len(chunk)] = out.detach().to(torch.float16).cpu().numpy()
        masks[st:st + len(chunk)] = b["attention_mask"].to(torch.uint8).cpu().numpy()
        if progress:
            print(f"  embedded {min(st + batch_size, n)}/{n}", flush=True)

    return embeds.astype(np.float32), masks.astype(np.int64)


def read_fasta(path):
    """Minimal FASTA reader. Returns (ids, sequences); ids are the token after '>'."""
    ids, seqs, cur, buf = [], [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    ids.append(cur); seqs.append("".join(buf))
                cur, buf = line[1:].split()[0] if len(line) > 1 else f"seq{len(ids)}", []
            else:
                buf.append(line)
    if cur is not None:
        ids.append(cur); seqs.append("".join(buf))
    if not ids:
        raise ValueError(f"no sequences found in {path}")
    return ids, seqs
