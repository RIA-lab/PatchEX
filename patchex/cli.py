"""Command-line interface: `patchex-predict`.

Examples
--------
# single sequence, ensemble of all available seeds (default)
patchex-predict --task topt --seq MKTAYIAKQRQISFVKSHFSRQ...

# batch from FASTA, write summary + per-patch weights + figure
patchex-predict --task ph --fasta proteins.fasta \\
    --out results.csv --weights weights.csv --plot weights.png

# single seed (faster, ~3-5% less accurate)
patchex-predict --task topt --fasta proteins.fasta --seeds 0
"""
from __future__ import annotations

import argparse
import json
import sys


def build_parser():
    p = argparse.ArgumentParser(
        prog="patchex-predict",
        description="Predict enzyme optimal temperature or optimal pH, with per-patch weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", help="input FASTA (one or many sequences)")
    src.add_argument("--seq", help="a single amino-acid sequence")
    p.add_argument("--task", default="topt", choices=["topt", "ph"],
                   help="topt = optimal temperature (C); ph = optimal pH (default: topt)")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds, e.g. '0' or '0,1,2'. Default: all "
                        "available (ensemble; ~3-5%% more accurate, 3x slower)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="directory of patchex_<arm>_s<seed>.pt files "
                        "(default: ./checkpoints or $PATCHEX_CHECKPOINTS)")
    p.add_argument("--out", default=None, help="summary CSV (default: print to stdout)")
    p.add_argument("--weights", default=None, help="per-patch weights CSV (long format)")
    p.add_argument("--json", dest="json_out", default=None, help="full results as JSON")
    p.add_argument("--plot", default=None, help="patch-weight figure (PNG)")
    p.add_argument("--max-panels", type=int, default=6, help="proteins to draw (default: 6)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=8, help="PatchEX forward batch size")
    p.add_argument("--esm-batch-size", type=int, default=8,
                   help="ESM-2 batch size; lower this if you hit GPU OOM")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    from .infer import PatchEX, predictions_to_frame, write_weights

    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    mdl = PatchEX.load(a.task, seeds=seeds, checkpoint_dir=a.checkpoint_dir,
                       device=a.device)
    if not a.quiet:
        print(f"[patchex] {mdl}", file=sys.stderr)

    kw = dict(batch_size=a.batch_size, esm_batch_size=a.esm_batch_size,
              progress=not a.quiet)
    preds = mdl.predict_fasta(a.fasta, **kw) if a.fasta else mdl.predict([a.seq], **kw)

    df = predictions_to_frame(preds)
    if a.out:
        df.to_csv(a.out, index=False)
        if not a.quiet:
            print(f"[patchex] wrote {a.out}", file=sys.stderr)
    else:
        print(df.to_string(index=False))

    if a.weights:
        write_weights(preds, a.weights)
        if not a.quiet:
            print(f"[patchex] wrote {a.weights}", file=sys.stderr)
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump([p.to_dict() for p in preds], fh, indent=1)
        if not a.quiet:
            print(f"[patchex] wrote {a.json_out}", file=sys.stderr)
    if a.plot:
        from .viz import plot_weights
        plot_weights(preds, a.plot, max_panels=a.max_panels)
        if not a.quiet:
            print(f"[patchex] wrote {a.plot}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
