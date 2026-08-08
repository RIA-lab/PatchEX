
"""Faithful padded ESM-2 cache for baseline reproduction.

Stores the FULL padded [max_length, 640] ESM last_hidden_state per sequence,
plus the attention_mask, so that a cached baseline forward is numerically
identical to running ESM in-forward with padding='max_length'. This matters
because every ESM baseline in the PatchET paper pools over ALL positions
(including padding), so the padded outputs are load-bearing.

ESM is run under the SAME regime training uses: model in fp32, forward under
torch.autocast(fp16) + no_grad, last_hidden_state stored as fp16 -- matching
the HF Trainer fp16=True autocast path.
"""
import argparse, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, EsmModel

D_ESM = 640

def build(a):
    tok = AutoTokenizer.from_pretrained(a.esm)
    model = EsmModel.from_pretrained(a.esm).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    use_ac = a.fp16 and dev == "cuda"
    print(f"device={dev} autocast_fp16={use_ac} max_length={a.max_length}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    manifest = {}
    for split in a.splits.split(","):
        df = pd.read_csv(os.path.join(a.data, f"{split}.csv"))
        seqs = [str(s) for s in df[a.seq_col]]
        accs = [str(x) for x in df[a.acc_col]]
        n = len(seqs); L = a.max_length
        emb_path = os.path.join(a.out, f"{split}_emb.f16")
        msk_path = os.path.join(a.out, f"{split}_mask.u8")
        emb = np.memmap(emb_path, dtype=np.float16, mode="w+", shape=(n, L, D_ESM))
        msk = np.memmap(msk_path, dtype=np.uint8,   mode="w+", shape=(n, L))
        gb = n * L * D_ESM * 2 / 1e9
        print(f"{split}: {n} seqs -> {gb:.2f} GB padded", flush=True)
        with torch.no_grad():
            for st in range(0, n, a.batch_size):
                chunk = seqs[st:st + a.batch_size]
                b = tok(chunk, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=L)
                b = {k: v.to(dev) for k, v in b.items()}
                if use_ac:
                    with torch.autocast("cuda", dtype=torch.float16):
                        hs = model(**b).last_hidden_state
                else:
                    hs = model(**b).last_hidden_state
                hs = hs.float().cpu().numpy().astype(np.float16)          # [b, L, 640]
                am = b["attention_mask"].cpu().numpy().astype(np.uint8)    # [b, L]
                emb[st:st + len(chunk)] = hs
                msk[st:st + len(chunk)] = am
                if (st // a.batch_size) % 20 == 0:
                    print(f"  {min(st + len(chunk), n)}/{n}", flush=True)
        emb.flush(); msk.flush(); del emb, msk
        np.savez(os.path.join(a.out, f"{split}_index.npz"),
                 n=n, accessions=np.array(accs), max_length=L)
        manifest[split] = dict(n_seq=n, gb=round(gb, 3),
                               emb=os.path.basename(emb_path),
                               mask=os.path.basename(msk_path))
    manifest["meta"] = dict(esm=a.esm, d_esm=D_ESM, max_length=a.max_length,
                            dtype="float16", padding="max_length",
                            includes_special_tokens=True, autocast_fp16=use_ac,
                            data=a.data)
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2), flush=True)

def verify(a):
    """Re-encode a few seqs in-forward-style and compare against the padded cache."""
    tok = AutoTokenizer.from_pretrained(a.esm)
    model = EsmModel.from_pretrained(a.esm).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
    use_ac = a.fp16 and dev == "cuda"
    split = a.splits.split(",")[0]
    ix = np.load(os.path.join(a.out, f"{split}_index.npz"), allow_pickle=True)
    n = int(ix["n"]); L = int(ix["max_length"])
    emb = np.memmap(os.path.join(a.out, f"{split}_emb.f16"), dtype=np.float16, mode="r", shape=(n, L, D_ESM))
    msk = np.memmap(os.path.join(a.out, f"{split}_mask.u8"), dtype=np.uint8, mode="r", shape=(n, L))
    df = pd.read_csv(os.path.join(a.data, f"{split}.csv"))
    rng = np.random.default_rng(0)
    picks = rng.choice(n, size=min(6, n), replace=False)
    errs = []
    with torch.no_grad():
        for i in picks:
            s = str(df[a.seq_col].iloc[i])
            b = tok([s], return_tensors="pt", padding="max_length", truncation=True, max_length=L)
            b = {k: v.to(dev) for k, v in b.items()}
            if use_ac:
                with torch.autocast("cuda", dtype=torch.float16):
                    ref = model(**b).last_hidden_state[0].float().cpu().numpy()
            else:
                ref = model(**b).last_hidden_state[0].float().cpu().numpy()
            got = np.asarray(emb[i], dtype=np.float32)
            assert got.shape == ref.shape, f"row {i}: {got.shape} vs {ref.shape}"
            assert np.array_equal(np.asarray(msk[i]), b["attention_mask"][0].cpu().numpy().astype(np.uint8))
            errs.append(float(np.abs(got - ref).max()))
    print(f"VERIFY n={len(errs)} max_abs_err={max(errs):.5f} mean={np.mean(errs):.5f}", flush=True)
    print("fp16 autocast round-trip tolerance ~1e-2; larger means a layout bug.", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/opt_v3id_final")
    p.add_argument("--out",  default="cache/esm_opt_v3id_final")
    p.add_argument("--esm",  default="esm150")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--seq_col", default="sequence")
    p.add_argument("--acc_col", default="accession")
    p.add_argument("--max_length", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--verify_only", action="store_true")
    a = p.parse_args()
    if a.verify_only: verify(a)
    else: build(a); verify(a)

if __name__ == "__main__":
    main()
