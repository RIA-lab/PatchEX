"""PatchEX inference: sequence in, property + per-patch weights out.

    from patchex import PatchEX
    mdl = PatchEX.load("topt")                 # 3-seed ensemble by default
    out = mdl.predict(["MKT..."])              # -> list[Prediction]
    out[0].value                               # Topt in C (or pH)
    out[0].w_A, out[0].w_B                     # per-patch arrays, length P

Two weights are returned per protein and they answer different questions:

    w_A  site map   — supervised by functional-site annotations. Sigmoid, so each
                      patch is scored independently and the values do NOT sum to 1.
                      Use this to ask "which regions are functionally important?"
    w_B  mixture    — the weights that FORM the prediction (y = sum_i w_B,i * p_i).
                      Softmax, sums to 1. Use this to ask "what did the model
                      actually use to reach this number?"

They are near-orthogonal in practice (mean per-protein correlation 0.10), so
they are not interchangeable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
import torch

from .embed import MAX_LEN, embed_sequences
from .model import Model

PATCH_LEN = 25
TASKS = {"topt": dict(arm="opt", unit="C", label="Topt"),
         "ph": dict(arm="ph", unit="pH", label="pHopt")}

BASE_CONFIG = dict(context_window=1000, target_window=640, patch_len=PATCH_LEN,
                   max_seq_len=1000, d_model=640, n_layers=3, patch_inter_kernel=2,
                   n_patch_inter_heads=4, n_RD=4, pred_mode="mixture",
                   site_use_esm=True, use_species=False)


@dataclass
class Prediction:
    """One protein's result."""
    id: str
    value: float                 # Topt (C) or optimal pH
    unit: str
    n_patches: int
    length: int
    w_A: list                    # site map, per patch (sigmoid, independent)
    w_B: list                    # mixture weights, per patch (softmax, sums to 1)
    patch_preds: list            # per-patch scalar p_i, physical units
    per_seed: list | None = None  # per-seed values when ensembling
    std: float | None = None      # across-seed sd when ensembling

    def top_patches(self, k: int = 5, weight: str = "w_A"):
        """Highest-weighted patches as (patch_index, residue_start, residue_end, weight)."""
        w = np.asarray(getattr(self, weight))
        order = np.argsort(-w)[:k]
        return [(int(i), int(i) * PATCH_LEN + 1,
                 min((int(i) + 1) * PATCH_LEN, self.length), float(w[i])) for i in order]

    def to_dict(self):
        return asdict(self)


def _default_ckpt_dir():
    return os.environ.get("PATCHEX_CHECKPOINTS",
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "checkpoints"))


class PatchEX:
    """Loaded PatchEX model(s) for one task."""

    def __init__(self, models, task: str, device: str):
        self.models = models
        self.task = task
        self.device = device
        self.unit = TASKS[task]["unit"]

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, task: str = "topt", seeds: Sequence[int] | None = None,
             checkpoint_dir: str | None = None, device: str | None = None):
        """Load checkpoints for `task` ('topt' or 'ph').

        seeds=None loads every available seed and ensembles them (recommended —
        it is 3-5% more accurate than a single seed at 3x the inference cost).
        Pass e.g. seeds=[0] for a single model.
        """
        task = task.lower()
        if task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)}, got {task!r}")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cdir = checkpoint_dir or _default_ckpt_dir()
        arm = TASKS[task]["arm"]

        found = []
        for s in (seeds if seeds is not None else range(8)):
            p = os.path.join(cdir, f"patchex_{arm}_s{s}.pt")
            if os.path.exists(p):
                found.append((s, p))
        if not found:
            raise FileNotFoundError(
                f"no checkpoints for task={task!r} in {cdir}. Expected files named "
                f"patchex_{arm}_s<seed>.pt — download them (see README) or set "
                f"PATCHEX_CHECKPOINTS.")

        models = []
        for s, p in found:
            blob = torch.load(p, map_location="cpu", weights_only=False)
            cfg = dict(BASE_CONFIG); cfg.update(blob.get("config", {}))
            m = Model(cfg)
            sd = {k: v.float() for k, v in blob["state_dict"].items()}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            # direct_head is an auxiliary output stripped from release checkpoints;
            # anything else missing means the checkpoint does not match this code.
            unexpected_real = [k for k in unexpected]
            missing_real = [k for k in missing if not k.startswith("direct_head")]
            if missing_real or unexpected_real:
                raise RuntimeError(f"checkpoint {p} does not match the model definition "
                                   f"(missing={missing_real[:4]}, unexpected={unexpected_real[:4]})")
            m.inference = True
            models.append((s, m.eval().to(device)))
        return cls(models, task, device)

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, sequences: Iterable[str], ids: Sequence[str] | None = None,
                batch_size: int = 8, esm_batch_size: int = 8,
                progress: bool = False) -> list[Prediction]:
        """Predict for one or many sequences. Accepts a bare string for convenience."""
        if isinstance(sequences, str):
            sequences = [sequences]
        seqs = [str(s).strip().upper() for s in sequences]
        ids = list(ids) if ids is not None else [f"seq{i+1}" for i in range(len(seqs))]
        if len(ids) != len(seqs):
            raise ValueError(f"got {len(ids)} ids for {len(seqs)} sequences")

        if progress:
            print(f"embedding {len(seqs)} sequence(s) with ESM-2 ...", flush=True)
        emb, msk = embed_sequences(seqs, device=self.device, batch_size=esm_batch_size,
                                   progress=progress)

        n = len(seqs)
        vals = np.zeros((len(self.models), n), dtype=np.float64)
        wA_acc, wB_acc, pp_acc, pm = None, None, None, None
        for mi, (seed, m) in enumerate(self.models):
            if progress:
                print(f"running seed {seed} ...", flush=True)
            outs_v, outs_a, outs_b, outs_p, outs_m = [], [], [], [], []
            for st in range(0, n, batch_size):
                E = torch.from_numpy(emb[st:st + batch_size]).to(self.device)
                A = torch.from_numpy(msk[st:st + batch_size]).to(self.device)
                o = m(E, A)
                outs_v.append(o.pred.float().cpu().numpy())
                outs_a.append(o.w_A.float().cpu().numpy())
                outs_b.append(o.w_B.float().cpu().numpy())
                outs_p.append(o.patch_preds.float().cpu().numpy())
                outs_m.append(o.patch_mask.float().cpu().numpy())
            vals[mi] = np.concatenate(outs_v)
            a, b, p, k = (np.concatenate(x) for x in (outs_a, outs_b, outs_p, outs_m))
            wA_acc = a if wA_acc is None else wA_acc + a
            wB_acc = b if wB_acc is None else wB_acc + b
            pp_acc = p if pp_acc is None else pp_acc + p
            pm = k
        nm = len(self.models)
        wA_acc, wB_acc, pp_acc = wA_acc / nm, wB_acc / nm, pp_acc / nm

        results = []
        for i, (sid, s) in enumerate(zip(ids, seqs)):
            P = int(pm[i].sum())
            results.append(Prediction(
                id=sid, value=float(vals[:, i].mean()), unit=self.unit,
                n_patches=P, length=min(len(s), MAX_LEN),
                w_A=[float(x) for x in wA_acc[i][:P]],
                w_B=[float(x) for x in wB_acc[i][:P]],
                patch_preds=[float(x) for x in pp_acc[i][:P]],
                per_seed=[float(v) for v in vals[:, i]] if nm > 1 else None,
                std=float(vals[:, i].std(ddof=1)) if nm > 2 else None))
        return results

    def predict_fasta(self, path: str, **kw) -> list[Prediction]:
        from .embed import read_fasta
        ids, seqs = read_fasta(path)
        return self.predict(seqs, ids=ids, **kw)

    @property
    def seeds(self):
        return [s for s, _ in self.models]

    def __repr__(self):
        return (f"PatchEX(task={self.task!r}, seeds={self.seeds}, "
                f"device={self.device!r}, ensemble={len(self.models) > 1})")


def predictions_to_frame(preds: list[Prediction]):
    """Summary table: one row per protein (weights excluded; see write_weights)."""
    import pandas as pd
    rows = []
    for p in preds:
        r = dict(id=p.id, prediction=round(p.value, 4), unit=p.unit,
                 length=p.length, n_patches=p.n_patches)
        if p.std is not None:
            r["seed_sd"] = round(p.std, 4)
        if p.per_seed:
            for i, v in enumerate(p.per_seed):
                r[f"seed{i}"] = round(v, 4)
        top = p.top_patches(3, "w_A")
        r["top_site_patches"] = "; ".join(f"{a}-{b}({w:.2f})" for _, a, b, w in top)
        rows.append(r)
    return pd.DataFrame(rows)


def write_weights(preds: list[Prediction], path: str):
    """Long-format per-patch weights: one row per (protein, patch)."""
    import pandas as pd
    rows = []
    for p in preds:
        for i, (a, b, pp) in enumerate(zip(p.w_A, p.w_B, p.patch_preds)):
            rows.append(dict(id=p.id, patch=i, res_start=i * PATCH_LEN + 1,
                             res_end=min((i + 1) * PATCH_LEN, p.length),
                             w_A=round(a, 6), w_B=round(b, 6), patch_pred=round(pp, 4)))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df
