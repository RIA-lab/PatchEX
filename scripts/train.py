"""Train PatchEX on the optimal-temperature (Topt) or optimal-pH dataset.

    python scripts/train.py --task opt --seed 0     # Topt, shipped config
    python scripts/train.py --task ph  --seed 0     # optimal pH

Defaults reproduce the published `A_esmhead` configuration. On completion the
script prints a REPRODUCIBILITY REPORT comparing the achieved test metrics
against the published reference values (patchex/reference.py) and writes it to
<results_root>/<task>/<variant>_s<seed>/reproducibility_report.json.

Requires the precomputed ESM cache — see README "Training" for how to build it.
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, torch, yaml
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from scipy.stats import rankdata
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score

D_ESM, MAXLEN = 640, 1000

TASKS = {
  'opt': dict(data='data/opt_v3id_final', cache='cache/esm_opt_v3id_final',
              label='temperature_optimum', patch_labels='{split}_patch_labels.npz',
              ranges=[(None,25),(25,50),(50,80),(80,None)], max_delta=15.0),
  'ph':  dict(data='data/ph', cache='cache/esm_ph_padded',
              label='phopt', patch_labels='patch_site_labels.npz',
              ranges=[(None,5),(5,7),(7,9),(9,None)], max_delta=2.0),
}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def range_weights(labels, ranges):
    n=len(labels); ws=[]
    for lo,hi in ranges:
        if lo is None: m = labels < hi
        elif hi is None: m = labels >= lo
        else: m = (labels>=lo)&(labels<hi)
        cnt=int(m.sum()); ws.append(0.0 if cnt==0 else n/cnt)
    s=sum(ws); return [w/s for w in ws] if s>0 else ws

class WeightedRMSE:
    def __init__(self, ranges, weights):
        self.ranges, self.weights = ranges, weights
    def __call__(self, pred, y):
        tot = pred.new_tensor(0.0); acc = pred.new_tensor(0.0)
        for (lo,hi),w in zip(self.ranges, self.weights):
            if lo is None: m = y < hi
            elif hi is None: m = y >= lo
            else: m = (y>=lo)&(y<hi)
            if m.any():
                tot = tot + w*torch.sqrt(F.mse_loss(pred[m], y[m])+1e-8); acc = acc + w
        return tot/acc.clamp(min=1e-8) if float(acc)>0 else torch.sqrt(F.mse_loss(pred,y)+1e-8)

class V2Dataset(TorchDataset):
    def __init__(self, split, tcfg, sp2id=None, use_species=False, species_dropout=0.0):
        df = pd.read_csv(os.path.join(tcfg['data'], f'{split}.csv'))
        self.df = df
        self.labels = df[tcfg['label']].astype('float32').values
        idx = np.load(os.path.join(tcfg['cache'], f'{split}_index.npz'), allow_pickle=True)
        self.n = int(idx['n'])
        self.emb = np.memmap(os.path.join(tcfg['cache'], f'{split}_emb.f16'),
                             dtype=np.float16, mode='r', shape=(self.n, MAXLEN, D_ESM))
        self.msk = np.memmap(os.path.join(tcfg['cache'], f'{split}_mask.u8'),
                             dtype=np.uint8, mode='r', shape=(self.n, MAXLEN))
        # --- site labels, keyed by accession ---
        pl = tcfg['patch_labels'].replace('{split}', split)
        z = np.load(os.path.join(tcfg['data'], pl), allow_pickle=True)
        self.sites = {k: z[k] for k in z.files}
        self.acc = df['accession'].astype(str).values
        self.P = MAXLEN // 25
        self.use_species, self.species_dropout, self.sp2id = use_species, species_dropout, sp2id
        self.org = df['organism'].astype(str).values if 'organism' in df else None
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        P = self.P
        sl = np.zeros(P, dtype=np.float32); sm = np.zeros(P, dtype=np.float32)
        a = self.acc[i]
        if a in self.sites:                       # proteins WITHOUT labels contribute 0 to L_site
            v = np.asarray(self.sites[a], dtype=np.float32)
            k = min(len(v), P); sl[:k] = v[:k]; sm[:k] = 1.0
        sid = 0
        if self.use_species and self.sp2id is not None and self.org is not None:
            sid = self.sp2id.get(self.org[i], 0)
            if self.species_dropout > 0 and random.random() < self.species_dropout: sid = 0
        return {"embeds": torch.from_numpy(np.asarray(self.emb[i], dtype=np.float32)),
                "attention_mask": torch.from_numpy(np.asarray(self.msk[i], dtype=np.int64)),
                "labels": torch.tensor(self.labels[i], dtype=torch.float32),
                "site_labels": torch.from_numpy(sl), "site_mask": torch.from_numpy(sm),
                "species_id": torch.tensor(sid, dtype=torch.long)}

def metrics_reg(p):
    pred, y = p.predictions, p.label_ids
    if isinstance(pred, tuple): pred = pred[0]
    pred = np.asarray(pred).squeeze(); y = np.asarray(y).squeeze()
    pe = float(np.corrcoef(pred,y)[0,1])
    sp = float(np.corrcoef(rankdata(pred), rankdata(y))[0,1])
    mae = float(np.abs(pred-y).mean()); rmse = float(np.sqrt(((pred-y)**2).mean()))
    return {"Pearson Correlation":pe, "Spearman Correlation":sp, "MAE":mae, "RMSE":rmse,
            "R2":float(r2_score(y,pred))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='opt', choices=list(TASKS))
    ap.add_argument('--variant', default='A_esmhead')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init_from', default='')          # trained PatchET ckpt (same split!)
    ap.add_argument('--freeze', default='top', choices=['all','top','none'])
    ap.add_argument('--pred_mode', default='mixture', choices=['mixture','direct'])
    ap.add_argument('--lambda_site', type=float, default=None,
                    help='default: 10 for --task opt, 1 for --task ph (published values)')
    ap.add_argument('--lambda_anchor', type=float, default=1.0)
    ap.add_argument('--lambda_ent', type=float, default=0.0)
    ap.add_argument('--lambda_direct', type=float, default=0.1)
    ap.add_argument('--mix_affine', action='store_true')
    ap.add_argument('--two_stage', action='store_true',
                    help='Stage 1: train value path (trunk+patch_scalar+mix_gate+direct). '
                         'Stage 2: FREEZE all of that, train ONLY site_head (and mix_gate if --stage2_gate).')
    ap.add_argument('--stage2_gate', action='store_true')
    ap.add_argument('--stage2_epochs', type=int, default=30)
    ap.add_argument('--use_species', action='store_true')
    ap.add_argument('--species_dropout', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--results_root', default='results')
    a = ap.parse_args()
    if a.lambda_site is None:
        a.lambda_site = 10.0 if a.task == 'opt' else 1.0
    set_seed(a.seed)
    HAS_CUDA = torch.cuda.is_available()
    if not HAS_CUDA:
        # CPU node: fp16 autocast is unavailable; 32 threads measured optimal (64 regresses)
        torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '32')))
    t = TASKS[a.task]

    tr_df = pd.read_csv(os.path.join(t['data'],'train.csv'))
    sp2id = None
    if a.use_species:
        sps = sorted(tr_df['organism'].astype(str).unique())
        sp2id = {s:i+1 for i,s in enumerate(sps)}
    ds = {s: V2Dataset(s, t, sp2id, a.use_species, a.species_dropout if s=='train' else 0.0)
          for s in ['train','val','test']}

    # pos_weight for the site BCE, from the train split
    allv = np.concatenate([np.asarray(v).ravel() for v in ds['train'].sites.values()])
    pos = float(allv.mean()); pw = (1-pos)/max(pos,1e-6)

    cfg = dict(context_window=1000, target_window=640, patch_len=25, max_seq_len=1000,
               d_model=640, n_layers=3, patch_inter_kernel=2, n_patch_inter_heads=4, n_RD=4,
               lambda_site=a.lambda_site, lambda_anchor=a.lambda_anchor,
               lambda_ent=a.lambda_ent, lambda_direct=a.lambda_direct,
               site_pos_weight=pw, pred_mode=a.pred_mode,
               mix_affine=a.mix_affine, two_stage=a.two_stage, label_mean=float(np.mean(ds['train'].labels)),
               use_species=a.use_species, n_species=len(sp2id) if sp2id else 0,
               species_max_delta=t['max_delta'])
    from patchex.model import Model
    model = Model(cfg)

    # --- inherit PatchET (trained on THIS split — never the released ckpt) ---
    if a.init_from:
        sd = torch.load(a.init_from, map_location='cpu')
        sd = sd.get('state_dict', sd)
        tsd = {k.replace('patch_','trunk.patch_'): v for k,v in sd.items() if k.startswith('patch_')}
        missing, unexpected = model.load_state_dict(tsd, strict=False)
        print(f"[init] loaded {len(tsd)} trunk tensors from {a.init_from}", flush=True)
    if a.freeze == 'all':
        for p in model.trunk.parameters(): p.requires_grad = False
    elif a.freeze == 'top':
        for p in model.trunk.parameters(): p.requires_grad = False
        for p in model.trunk.patch_intra_layers.parameters(): p.requires_grad = True

    w = range_weights(ds['train'].labels, t['ranges'])
    model.loss_fct = WeightedRMSE(t['ranges'], w)

    tag = f"{a.variant}_s{a.seed}"
    out = os.path.join(a.results_root, a.task, tag); os.makedirs(out, exist_ok=True)
    args = TrainingArguments(
        output_dir=os.path.join('output_v2', a.task, tag), overwrite_output_dir=True,
        num_train_epochs=a.epochs, per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        learning_rate=a.lr, weight_decay=1e-4, fp16=HAS_CUDA, max_grad_norm=1.0,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=1,
        logging_strategy='epoch', report_to=[], dataloader_num_workers=2,
        load_best_model_at_end=True, metric_for_best_model='eval_RMSE', greater_is_better=False,
        seed=a.seed, save_safetensors=False,
        label_names=['labels'], remove_unused_columns=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds['train'], eval_dataset=ds['val'],
                      compute_metrics=metrics_reg,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)])
    trainer.train()

    if a.two_stage:
        # ---- Stage 2: freeze the value path; train the weight head(s) alone ----
        # This decouples "what predicts" from "what explains": the prediction is fixed,
        # so weight-head gradients cannot change accuracy, only the explanation.
        for p in model.parameters(): p.requires_grad = False
        for p in model.site_head.parameters(): p.requires_grad = True
        if a.stage2_gate:
            for p in model.mix_gate.parameters(): p.requires_grad = True
        args2 = TrainingArguments(
            output_dir=os.path.join('output_v2', a.task, tag + '_s2'), overwrite_output_dir=True,
            num_train_epochs=a.stage2_epochs, per_device_train_batch_size=a.bs,
            per_device_eval_batch_size=a.bs, learning_rate=a.lr, weight_decay=1e-4,
            fp16=HAS_CUDA, max_grad_norm=1.0, eval_strategy='epoch', save_strategy='epoch',
            save_total_limit=1, logging_strategy='epoch', report_to=[], dataloader_num_workers=2,
            load_best_model_at_end=True, metric_for_best_model='eval_RMSE', greater_is_better=False,
            seed=a.seed, save_safetensors=False, label_names=['labels'], remove_unused_columns=False)
        trainer = Trainer(model=model, args=args2, train_dataset=ds['train'], eval_dataset=ds['val'],
                          compute_metrics=metrics_reg,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)])
        trainer.train()
        print('[two_stage] stage 2 complete (value path frozen)', flush=True)

    for split in ['val','test']:
        pr = trainer.predict(ds[split])
        m = {k.replace('test_',''):v for k,v in pr.metrics.items()}
        m['config'] = {k:v for k,v in vars(a).items()}
        json.dump(m, open(os.path.join(out, f'{split}_metrics.json'),'w'), indent=2)

    # --- dump weights for WP2/WP3 (test split, inference mode) ---
    dev = 'cuda' if HAS_CUDA else 'cpu'
    model.inference = True; model.eval().to(dev)
    WA, WB, PP, PM, YH = [],[],[],[],[]
    with torch.no_grad():
        for i in range(0, len(ds['test']), 32):
            batch = [ds['test'][j] for j in range(i, min(i+32, len(ds['test'])))]
            eb = torch.stack([b['embeds'] for b in batch]).to(dev)
            am = torch.stack([b['attention_mask'] for b in batch]).to(dev)
            si = torch.stack([b['species_id'] for b in batch]).to(dev)
            if HAS_CUDA:
                with torch.autocast('cuda', dtype=torch.float16):
                    o = model(eb, am, species_id=si if a.use_species else None)
            else:
                o = model(eb, am, species_id=si if a.use_species else None)
            WA.append(o.w_A.float().cpu().numpy()); WB.append(o.w_B.float().cpu().numpy())
            PP.append(o.patch_preds.float().cpu().numpy()); PM.append(o.patch_mask.cpu().numpy())
            YH.append(o.pred.float().cpu().numpy())
    np.savez_compressed(os.path.join(out,'weights_test.npz'),
        w_A=np.concatenate(WA), w_B=np.concatenate(WB), patch_preds=np.concatenate(PP),
        patch_mask=np.concatenate(PM), pred=np.concatenate(YH),
        labels=ds['test'].labels, accession=ds['test'].acc)
    print("DONE", tag, json.load(open(os.path.join(out,'test_metrics.json')))['RMSE'], flush=True)

    # ---------------- reproducibility report ----------------
    tm = json.load(open(os.path.join(out, 'test_metrics.json')))
    achieved = {k: tm[k] for k in ('MAE', 'RMSE', 'R2', 'Pearson', 'Spearman') if k in tm}
    try:
        from patchex.reference import REFERENCE, compare
        rows, ok = compare(a.task, achieved)
        ref = REFERENCE[a.task]
        print('', flush=True)
        print('=' * 74, flush=True)
        print(f'REPRODUCIBILITY REPORT — {ref["name"]}, seed {a.seed}', flush=True)
        print(f'published reference: {ref["n_seeds"]} seeds, test n={ref["test_n"]}, {ref["config"]}', flush=True)
        print('-' * 74, flush=True)
        print(f'{"metric":10s} {"this run":>12s} {"published":>18s} {"delta":>10s}   status', flush=True)
        for r in rows:
            print(f'{r["metric"]:10s} {r["achieved"]:>12.4f} {r["published"]:>18s} '
                  f'{r["delta"]:>+10.4f}   {r["status"]}', flush=True)
        print('-' * 74, flush=True)
        print(('ALL METRICS WITHIN ±2 sd OF PUBLISHED — reproduction successful.' if ok else
               'SOME METRICS OUTSIDE ±2 sd. A single seed is noisier than the 3-seed mean;\n'
               'if the gap is large, check the data split, the ESM checkpoint, and lambda values.'), flush=True)
        print(f'Note: published values are the mean of {ref["n_seeds"]} seeds. For the paper number,\n'
              f'train seeds 0,1,2 and average. The 3-seed ENSEMBLE reaches MAE {ref["ensemble_MAE"]}.', flush=True)
        print('=' * 74, flush=True)
        json.dump(dict(task=a.task, seed=a.seed, achieved=achieved, rows=rows,
                       within_tolerance=bool(ok), reference=ref),
                  open(os.path.join(out, 'reproducibility_report.json'), 'w'), indent=2)
    except Exception as e:
        print(f'[warn] reproducibility report unavailable: {type(e).__name__}: {e}', flush=True)

if __name__ == '__main__':
    main()
