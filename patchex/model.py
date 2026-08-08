"""PatchEX v2: two-weight, species-optional patch predictor.

Weight B (impact)  : learned mixture weight; y_hat = sum_i w_B,i * p_i  -> causal by construction
Weight A (site)    : independent per-patch sigmoid, supervised by functional-site labels
Species (optional) : bounded residual correction, dropout-trained, UNKNOWN-safe

Consumes PRECOMPUTED padded ESM embeddings [B,L,640] + mask (same cache as the baselines),
runs the PatchET patch stack (optionally frozen), and adds the v2 heads on top.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from .types import ModelOutput


class RDBlock(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dense = nn.Linear(dim, dim)
    def forward(self, x):
        return x + F.leaky_relu(self.dense(x))


class PatchETTrunk(nn.Module):
    """PatchET patch-feature trunk — module names match models/patchet_cached.py EXACTLY
    so a trained PatchET checkpoint loads straight in (that is the 'inherit PatchET' step).

    Returns:
      local  [B, P, tw]  per-patch intra features   (== PatchEX's local_feats)
      inter  [B, tw, P]  per-patch inter features   (== PatchEX's global_feats)
      pooled [B, 2*H*tw] the original global pooled vector (feeds the direct head)
    """
    def __init__(self, cfg):
        super().__init__()
        from .backbone import PatchIntraBackbone
        tw = cfg['target_window']; k = cfg['patch_inter_kernel']
        self.patch_inter_heads = cfg['n_patch_inter_heads']
        self.patch_intra_conv = nn.Conv1d(640, tw, kernel_size=cfg['patch_len'], stride=cfg['patch_len'])
        self.patch_intra_layers = PatchIntraBackbone(
            c_in=int(cfg['context_window'] / cfg['patch_len']),
            context_window=cfg['context_window'], target_window=tw,
            patch_len=cfg['patch_len'], stride=cfg['patch_len'],
            max_seq_len=cfg['max_seq_len'], n_layers=cfg['n_layers'], d_model=cfg['d_model'])
        self.patch_inter_conv = nn.Conv1d(tw, tw, kernel_size=2*k+1, padding=k)
        self.patch_inter_layers = nn.ModuleList([
            nn.Conv1d(tw, tw, kernel_size=2*k+1, padding=k) for _ in range(self.patch_inter_heads)])

    def forward(self, embeds):
        hidden_state = embeds
        patch_intra_values = self.patch_intra_conv(hidden_state.transpose(1, 2))   # [B,tw,P]
        hidden_state = self.patch_intra_layers(hidden_state)                       # [B,P,tw]
        hidden_state = F.softmax(hidden_state, dim=-1)
        hidden_state = hidden_state * patch_intra_values.transpose(1, 2)           # [B,P,tw] local
        patch_inter_values = self.patch_inter_conv(hidden_state.transpose(1, 2))   # [B,tw,P]
        outs_sum, outs_max = [], []
        for i in range(self.patch_inter_heads):
            w = F.softmax(self.patch_inter_layers[i](hidden_state.transpose(1, 2)), dim=-1)
            outs_sum.append(torch.sum(patch_inter_values * w, dim=-1))
            outs_max.append(torch.max(patch_inter_values * w, dim=-1)[0])
        pooled = torch.cat(outs_sum + outs_max, dim=1)                             # [B,2*H*tw]
        return hidden_state, patch_inter_values, pooled


class MixGate(nn.Module):
    """Weight B: logits for the mixture weights that PRODUCE the prediction."""
    def __init__(self, dim, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, local, inter):
        z = torch.cat([local, inter.transpose(1, 2)], dim=-1)   # [B,P,2*tw]
        return self.net(z).squeeze(-1)                          # [B,P]


class PatchScalarHead(nn.Module):
    """Per-patch scalar prediction p_i, in physical units.

    bias_init: final-layer bias is set to the training label mean so that the
    convex mixture sum(w_i p_i) STARTS inside the target range. Without this the
    mixture cannot reach e.g. 20-90 C from near-zero p_i, because a convex
    combination is bounded by min/max of the p_i.
    """
    def __init__(self, dim, hidden=128, dropout=0.1, bias_init=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1))
        nn.init.constant_(self.net[-1].bias, float(bias_init))
    def forward(self, local, inter):
        z = torch.cat([local, inter.transpose(1, 2)], dim=-1)
        return self.net(z).squeeze(-1)                          # [B,P]


class SiteHead(nn.Module):
    """Weight A: independent per-patch functional-site logit (NOT a softmax).

    Takes RAW mean-pooled ESM patch features alongside the trunk features. The trunk
    applies softmax over the feature axis and multiplies by a conv output, which is
    lossy for site detection: a logistic probe on raw ESM patch means reaches AUROC
    0.804 on Topt while a head reading only trunk features reached 0.622. Concatenating
    the raw ESM view means this head is a strict superset of that probe -- it can
    recover the probe's solution by zeroing the trunk block.
    """
    def __init__(self, dim, hidden=256, dropout=0.1, esm_dim=640, use_esm=True):
        super().__init__()
        self.use_esm = use_esm
        in_dim = dim * 2 + (esm_dim if use_esm else 0)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1))
    def forward(self, local, inter, esm_patch=None):
        zs = [local, inter.transpose(1, 2)]
        if self.use_esm and esm_patch is not None:
            zs.append(esm_patch)
        z = torch.cat(zs, dim=-1)
        return self.net(z).squeeze(-1)                          # [B,P] logits


class SpeciesHead(nn.Module):
    """Optional bounded residual. id 0 == UNKNOWN and is pinned to zero output."""
    def __init__(self, n_species, emb_dim=32, hidden=64, max_delta=15.0):
        super().__init__()
        self.emb = nn.Embedding(n_species + 1, emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.max_delta = max_delta
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, species_id):
        e = self.emb(species_id)                                 # [B,emb]
        d = torch.tanh(self.mlp(e).squeeze(-1)) * self.max_delta # [B]
        return d * (species_id > 0).float()                      # UNKNOWN -> exactly 0


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        tw = config['target_window']
        self.patch_len = config['patch_len']
        self.trunk = PatchETTrunk(config)
        self.label_mean = float(config.get('label_mean', 0.0))
        self.patch_scalar = PatchScalarHead(tw, bias_init=self.label_mean)
        self.mix_gate     = MixGate(tw)
        self.site_head    = SiteHead(tw, use_esm=bool(config.get('site_use_esm', True)))
        self.direct_head  = nn.Sequential(
            *[RDBlock(tw * 2 * config['n_patch_inter_heads']) for _ in range(config['n_RD'])],
            nn.Linear(tw * 2 * config['n_patch_inter_heads'], 1))
        # optional learned affine on the mixture output (extra freedom beyond the convex hull)
        self.mix_affine = bool(config.get('mix_affine', False))
        if self.mix_affine:
            self.out_scale = nn.Parameter(torch.tensor(1.0))
            self.out_bias  = nn.Parameter(torch.tensor(0.0))
        self.use_species = bool(config.get('use_species', False))
        if self.use_species:
            self.species_head = SpeciesHead(config['n_species'],
                                            max_delta=config.get('species_max_delta', 15.0))
        # objective weights
        self.lambda_site   = float(config.get('lambda_site', 0.0))
        self.lambda_anchor = float(config.get('lambda_anchor', 0.0))
        self.lambda_ent    = float(config.get('lambda_ent', 0.0))
        self.lambda_direct = float(config.get('lambda_direct', 0.0))
        self.site_pos_weight = float(config.get('site_pos_weight', 1.0))
        self.pred_mode = config.get('pred_mode', 'mixture')   # 'mixture' | 'direct'
        self.loss_fct = None      # set by the trainer (weighted RMSE)
        self.inference = False

    def patch_mask_from_attention(self, attention_mask, P):
        seq_lens = attention_mask.sum(dim=1).long()
        nvalid = ((seq_lens + self.patch_len - 1) // self.patch_len).clamp(min=1)
        nvalid = torch.minimum(nvalid, torch.tensor(P, device=nvalid.device))
        ar = torch.arange(P, device=attention_mask.device).unsqueeze(0)
        return (ar < nvalid.unsqueeze(1)).float()               # [B,P]

    def forward(self, embeds, attention_mask, labels=None,
                site_labels=None, site_mask=None, species_id=None):
        local, inter, pooled = self.trunk(embeds)
        B, P, _ = local.shape
        # raw ESM patch means [B,P,640] -- the exact feature the logistic probe uses
        pl = self.patch_len
        esm_patch = embeds[:, :P * pl, :].reshape(B, P, pl, embeds.shape[-1]).mean(2)
        pmask = self.patch_mask_from_attention(attention_mask, P)

        p     = self.patch_scalar(local, inter)                 # [B,P] per-patch scalar
        b_lg  = self.mix_gate(local, inter)                     # [B,P] mixture logits
        b_lg  = b_lg.masked_fill(pmask == 0, float('-inf'))
        w_B   = torch.softmax(b_lg, dim=-1)                     # [B,P] sums to 1
        a_lg  = self.site_head(local, inter, esm_patch)                    # [B,P] site logits
        w_A   = torch.sigmoid(a_lg) * pmask

        y_mix    = (w_B * p).sum(dim=-1)                        # THE prediction (causal)
        if self.mix_affine:
            y_mix = self.out_scale * y_mix + self.out_bias      # d y/d p_i = out_scale * w_B,i
        y_direct = self.direct_head(pooled).squeeze(-1)
        pred = y_mix if self.pred_mode == 'mixture' else y_direct

        if self.use_species and species_id is not None:
            pred = pred + self.species_head(species_id)

        if self.inference:
            return ModelOutput(pred=pred, w_A=w_A, w_B=w_B, patch_preds=p,
                               patch_mask=pmask, y_mix=y_mix, y_direct=y_direct)

        loss = None
        if labels is not None:
            loss = self.loss_fct(pred, labels)
            diag = {}
            # --- site supervision: masked BCE, OUTSIDE any error gate ---
            if self.lambda_site > 0 and site_labels is not None:
                m = pmask * (site_mask if site_mask is not None else 1.0)
                bce = F.binary_cross_entropy_with_logits(
                    a_lg, site_labels, reduction='none',
                    pos_weight=torch.tensor(self.site_pos_weight, device=a_lg.device))
                denom = m.sum().clamp(min=1.0)
                loss = loss + self.lambda_site * (bce * m).sum() / denom
            # --- weak anchor: keep p_i in physical units (NOT the old all->y term) ---
            if self.lambda_anchor > 0:
                anch = ((p - labels.unsqueeze(1)) ** 2 * pmask).sum() / pmask.sum().clamp(min=1.0)
                loss = loss + self.lambda_anchor * anch
            # --- entropy: prevent uniform / single-patch collapse ---
            if self.lambda_ent != 0:
                ent = -(w_B.clamp_min(1e-9).log() * w_B * pmask).sum(-1).mean()
                loss = loss + self.lambda_ent * ent
            # --- keep the direct head trained as an auxiliary output ---
            if self.lambda_direct > 0:
                loss = loss + self.lambda_direct * self.loss_fct(y_direct, labels)
        return ModelOutput(loss=loss, pred=pred, w_A=w_A, w_B=w_B)
