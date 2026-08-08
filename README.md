# PatchEX

**Interpretable prediction of enzyme optimal temperature (T<sub>opt</sub>) and optimal pH.**

PatchEX predicts an enzyme property from sequence alone and, in the same forward pass,
returns **two per-patch weight maps** that answer different questions:

| | what it is | what it answers | how to read it |
|---|---|---|---|
| **w<sub>A</sub>** — site map | supervised by functional-site annotations (UniProt features + M-CSA); **sigmoid**, independent per patch | *Where are the functional sites?* | values in [0,1]; they do **not** sum to 1 |
| **w<sub>B</sub>** — mixture | the weights that **form** the prediction, ŷ = Σ<sub>i</sub> w<sub>B,i</sub> · p<sub>i</sub>; **softmax** | *What did the model actually use?* | sums to 1 across patches |

The two are near-orthogonal (mean per-protein correlation 0.10), so they are not
interchangeable. w<sub>B</sub> is causal by construction — it needs no backward pass and no
reference baseline, because it literally produces the number.

Sequences are split into **25-residue patches** (up to 40 patches / 1000 residues).

---

## Installation

```bash
git clone https://github.com/RIA-lab/PatchEX.git
cd PatchEX

conda create -n patchex python=3.10 -y
conda activate patchex

# PyTorch — pick the build that matches your CUDA (see pytorch.org).
# CPU-only works for inference; training needs a GPU.
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -e .
```

Verify:

```bash
python -c "import patchex; print(patchex.__version__)"
```

### Checkpoints

Download the six released checkpoints (3 seeds × 2 tasks, ~75 MB each) into `checkpoints/`:
https://doi.org/10.5281/zenodo.21847056
```
checkpoints/
├── patchex_opt_s0.pt   patchex_opt_s1.pt   patchex_opt_s2.pt    # T_opt
└── patchex_ph_s0.pt    patchex_ph_s1.pt    patchex_ph_s2.pt     # optimal pH
```

Set `PATCHEX_CHECKPOINTS=/path/to/checkpoints` if you keep them elsewhere.

ESM-2 (`facebook/esm2_t30_150M_UR50D`, ~600 MB) is downloaded automatically from
Hugging Face on first use.

---

## Inference

### Command line

```bash
# single sequence — 3-seed ensemble by default
patchex-predict --task topt --seq MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ...

# batch from FASTA: summary table + per-patch weights + figure
patchex-predict --task ph \
    --fasta examples/example.fasta \
    --out results.csv \
    --weights patch_weights.csv \
    --plot patch_weights.png

# single seed — ~3x faster, ~3-5% higher error
patchex-predict --task topt --fasta examples/example.fasta --seeds 0
```

`--task topt` returns °C; `--task ph` returns pH units.

**Outputs**

- `--out` — one row per protein: prediction, per-seed values, across-seed sd, top site patches.
- `--weights` — long format, one row per (protein, patch): `w_A`, `w_B`, `patch_pred`, residue range.
- `--json` — everything, including full weight vectors.
- `--plot` — w<sub>A</sub> and w<sub>B</sub> along the sequence, one panel per protein.

### Python

```python
from patchex import PatchEX

mdl = PatchEX.load("topt")                      # all available seeds -> ensemble
preds = mdl.predict_fasta("examples/example.fasta")

p = preds[0]
print(p.id, p.value, p.unit, "±", p.std)        # 3-seed mean and sd
print(p.top_patches(5, "w_A"))                  # [(patch, res_start, res_end, weight), ...]

mdl_single = PatchEX.load("ph", seeds=[0])      # one model instead of the ensemble
```

Colour a structure by the site map:

```python
from patchex.viz import weights_to_pdb_bfactor
weights_to_pdb_bfactor(p, "AF-P33247-F1.pdb", "painted.pdb", weight="w_A")
# PyMOL:  load painted.pdb; spectrum b, white_salmon_red, minimum=0, maximum=1
```

Use a **fixed 0–1 range** for w<sub>A</sub>. Per-structure auto-scaling saturates every
protein to the top colour and destroys the comparison between them.

### Ensembling

`PatchEX.load(task)` loads every checkpoint it finds and averages the predictions
(weights are averaged too). This lowers error by 3–5% at 3× the inference cost:

| task | single model | 3-seed ensemble |
|---|---|---|
| T<sub>opt</sub> | 9.208 MAE (°C) | **8.898** |
| optimal pH | 0.615 MAE | **0.585** |

When comparing against another published method, compare **like with like** — an
ensemble against another method's single model overstates the difference.

---

## Training

Training reads a precomputed padded ESM-2 cache. Building it is a one-off GPU job;
the cache is large (11 GB for T<sub>opt</sub>, 12 GB for pH) because every sequence is
stored padded to 1000 × 640 in fp16.

```bash
# 1. build the ESM cache (GPU strongly recommended)
python scripts/extract_esm_cache.py \
    --data data/opt --out cache/esm_opt --esm facebook/esm2_t30_150M_UR50D --fp16

python scripts/extract_esm_cache.py \
    --data data/ph  --out cache/esm_ph  --esm facebook/esm2_t30_150M_UR50D --fp16

# 2. train (defaults = the published configuration)
python scripts/train.py --task opt --seed 0
python scripts/train.py --task ph  --seed 0
```

On completion the script prints a **reproducibility report**:

```
==========================================================================
REPRODUCIBILITY REPORT — optimal temperature (Topt), seed 0
published reference: 3 seeds, test n=857, lambda_site=10, lambda_anchor=1.0, ...
--------------------------------------------------------------------------
metric         this run          published        delta   status
MAE              9.2125      9.208 ± 0.044      +0.0045   OK
RMSE            13.0766     12.859 ± 0.248      +0.2176   OK
...
--------------------------------------------------------------------------
ALL METRICS WITHIN ±2 sd OF PUBLISHED — reproduction successful.
==========================================================================
```

and writes `results/<task>/A_esmhead_s<seed>/reproducibility_report.json`.

For the published numbers, train seeds 0, 1, 2 and average — the reference values are
3-seed means, and a single seed is naturally noisier.

**Published results** (test split, mean ± sd over 3 seeds):

| task | MAE | RMSE | R² | Pearson | site AUROC |
|---|---|---|---|---|---|
| T<sub>opt</sub> (n=857) | 9.208 ± 0.044 | 12.859 ± 0.248 | 0.347 ± 0.025 | 0.612 ± 0.010 | 0.823 ± 0.013 |
| optimal pH (n=1971) | 0.615 ± 0.008 | 0.857 ± 0.008 | 0.449 ± 0.010 | 0.686 ± 0.004 | 0.855 ± 0.003 |

Site AUROC is the recovery of annotated functional sites by w<sub>A</sub>. For reference,
the best post-hoc attribution method applied to the same model reaches 0.57 (T<sub>opt</sub>)
and 0.63 (pH); exact input-space occlusion reaches 0.54 and 0.57; random is 0.50.

### Data

| task | directory | train / val / test | label column |
|---|---|---|---|
| T<sub>opt</sub> | `data/opt/` | 6 870 / 857 / 857 | `temperature_optimum` |
| optimal pH | `data/ph/` | 7 124 / 760 / 1 971 | `phopt` |

Each directory holds `train.csv`, `val.csv`, `test.csv` (columns `accession`,
`sequence`, `organism`, label) plus patch-level site labels in `.npz` form.

The pH splits are the **official EpHod splits** (Gado et al., *Nat. Mach. Intell.* 2025;
Zenodo 10.5281/zenodo.14252615), built with MMseqs2 at <20% identity between train and
test. Keep them as-is when comparing to published pH predictors — some published methods
train on the official validation split, which inflates their test numbers.

---

## Repository layout

```
patchex/            installable package
  model.py            PatchEX architecture (trunk + 3 heads)
  backbone.py         patch encoder (PatchTST-derived, via PatchET)
  layers.py           transformer layers, RevIN
  embed.py            ESM-2 embedding, matching the training regime
  infer.py            PatchEX.load / .predict / .predict_fasta
  viz.py              weight plots, PDB B-factor painting
  reference.py        published numbers for the reproducibility check
  cli.py              patchex-predict
scripts/
  train.py            training + reproducibility report
  extract_esm_cache.py  one-off ESM cache builder
data/               splits for both tasks
checkpoints/        released model weights (downloaded separately)
examples/           example FASTA
```

---

## Notes and caveats

- **Sequences longer than 1000 residues are truncated**, not chunked. Predictions for
  very long proteins reflect the first 1000 residues only.
- **The embedding regime is load-bearing.** ESM-2 is run with `padding="max_length"`
  to 1000, fp32 weights, fp16 storage. `patchex.embed` reproduces this exactly;
  substituting dynamic padding or a different ESM size will silently shift predictions.
- **Released checkpoints are fp16 and omit `direct_head`**, an auxiliary output that is
  trained but never produces the reported prediction. Verified against the full fp32
  training checkpoints: max deviation 0.018 °C (T<sub>opt</sub>) and 0.0012 pH units,
  Pearson r = 1.000000.
- **w<sub>A</sub> is not an attention map.** It is an independent per-patch probability,
  and it is not used to compute the prediction. If you want to know what drove a
  specific number, read w<sub>B</sub>.
- Training used no species input; the optional species head is disabled in the
  released configuration.

