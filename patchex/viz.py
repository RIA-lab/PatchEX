"""Patch-weight visualisation.

One panel per protein: the site map (w_A) and the mixture weights (w_B) along
the sequence. The two are plotted on separate axes because they are different
quantities on different scales — w_A is an independent per-patch probability,
w_B is a softmax that sums to 1 and therefore shrinks as the protein gets longer.
"""
from __future__ import annotations

import numpy as np

PATCH_LEN = 25
COL_A, COL_B = "#B03A2E", "#2C6E9B"


def plot_weights(preds, path: str = "patch_weights.png", max_panels: int = 6,
                 dpi: int = 200, highlight=None):
    """Plot per-patch weights for up to `max_panels` proteins.

    Args:
        preds: list of Prediction objects from PatchEX.predict.
        highlight: optional {protein_id: [residue positions]} to mark known
            sites (e.g. catalytic residues) as ticks under the axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = list(preds)[:max_panels]
    n = len(preds)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.75 * n + 0.4), squeeze=False)
    axes = axes[:, 0]
    legend_handles = []

    for ax, p in zip(axes, preds):
        P = p.n_patches
        centres = np.arange(P) * PATCH_LEN + PATCH_LEN / 2
        wa, wb = np.asarray(p.w_A), np.asarray(p.w_B)

        ax.fill_between(centres, 0, wa, step="mid", color=COL_A, alpha=0.30, lw=0)
        ax.plot(centres, wa, drawstyle="steps-mid", color=COL_A, lw=1.4,
                label="$w_A$  site map (sigmoid)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("$w_A$", color=COL_A, fontsize=9)
        ax.tick_params(axis="y", labelcolor=COL_A, labelsize=8)

        ax2 = ax.twinx()
        ax2.plot(centres, wb, drawstyle="steps-mid", color=COL_B, lw=1.2, ls=(0, (4, 2)),
                 label="$w_B$  mixture (softmax)")
        ax2.set_ylim(0, max(float(wb.max()) * 1.35, 1e-3))
        ax2.set_ylabel("$w_B$", color=COL_B, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=COL_B, labelsize=8)

        if not legend_handles:
            # capture handles from BOTH axes of the first panel. fig.axes puts every
            # primary axis before the twins, so indexing into it picks the wrong one.
            legend_handles = ax.get_lines()[:1] + ax2.get_lines()[:1]

        if highlight and p.id in highlight:
            for r in highlight[p.id]:
                ax.plot([r], [0.02], marker="^", ms=5, color="#1f77d0",
                        clip_on=False, zorder=5)

        ax.set_xlim(0, max(p.length, P * PATCH_LEN))
        ax.set_xlabel("residue position", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        val = f"{p.value:.2f} {p.unit}" + (f" ± {p.std:.2f}" if p.std is not None else "")
        ax.set_title(f"{p.id}   —   predicted {val}", loc="left", fontsize=9.5)
        for sp in ("top",):
            ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)

    axes[0].legend(legend_handles, [h.get_label() for h in legend_handles],
                   frameon=False, fontsize=8, loc="upper right", ncol=2,
                   bbox_to_anchor=(1.0, 1.42))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def weights_to_pdb_bfactor(pred, pdb_in: str, pdb_out: str, weight: str = "w_A"):
    """Write `weight` into the B-factor column of a PDB so it can be coloured in 3D.

    In PyMOL:  load out.pdb; spectrum b, white_salmon_red, minimum=0, maximum=1
    Use a FIXED 0-1 range for w_A — per-structure auto-scaling saturates every
    protein and destroys the comparison between them.
    """
    w = np.asarray(getattr(pred, weight))
    lines = []
    with open(pdb_in) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                resi = int(line[22:26])
                idx = min(max((resi - 1) // PATCH_LEN, 0), len(w) - 1)
                line = f"{line[:60]}{float(w[idx]):6.2f}{line[66:]}"
            lines.append(line)
    with open(pdb_out, "w") as fh:
        fh.writelines(lines)
    return pdb_out
