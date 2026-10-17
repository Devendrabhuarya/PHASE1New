"""CICIoT2023 sweep.  python run_ciciot.py --csv /content/merge.csv --cache /content/ciciot.npz
Attribute (device_type) is SYNTHETIC: CICIoT2023 has no per-row device id. rho controls how strongly
device_type shapes a client's class mix (rho=0: none, rho=1: strong)."""
import argparse, itertools, numpy as np
from abacfl.ciciot import load_ciciot
from abacfl.data import FedConfig, ArraySource
from abacfl.runner import RunCfg, run

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="one CSV, a folder of CSVs, or a glob e.g. /content/ciciot/Merged*.csv"); ap.add_argument("--cache", default=None)
ap.add_argument("--cap", type=int, default=20000, help="max rows kept per fine-grained label")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--attacks", default="sign_flip,label_flip"); ap.add_argument("--rhos", default="1.0,0.0")
ap.add_argument("--variants", default="fedavg,median,krum,A0,A2,A3")
ap.add_argument("--frac", type=float, default=0.2); ap.add_argument("--placement", default="random")
ap.add_argument("--n_per_client", type=int, default=500)
a = ap.parse_args()

Xtr, ytr, Xte, yte, classes = load_ciciot(a.csv, cap_per_label=a.cap, cache=a.cache)
n_cls, d = len(classes), Xtr.shape[1]
src_tr, src_te = ArraySource(Xtr, ytr, n_cls), ArraySource(Xte, yte, n_cls)
print(f"{n_cls} classes, {d} features, train {len(Xtr):,} test {len(Xte):,}")
print(f"{'attack':11}{'rho':>4} {'variant':8}{'F1':>7}{'TPR':>6}{'FPR':>6}{'det.d':>6}{'rev.d':>6}{'hRev':>5}")
for atk, rho in itertools.product(a.attacks.split(","), map(float, a.rhos.split(","))):
    for v in a.variants.split(","):
        rs = [run(RunCfg(variant=v, rounds=a.rounds, seed=s, attack=atk, frac_attackers=a.frac,
                         attacker_placement=a.placement,
                         fed=FedConfig(rho=rho, n_cls=n_cls, d=d, type_shift=0.0, n_per_client=a.n_per_client)),
                  src_tr, src_te) for s in range(a.seeds)]
        m = lambda key: np.nanmean([r.get(key, np.nan) for r in rs])
        print(f"{atk:11}{rho:4.1f} {v:8}{m('final_f1'):7.3f}{m('tpr'):6.2f}{m('fpr'):6.2f}"
              f"{m('detect_delay'):6.1f}{m('revoke_delay'):6.1f}{m('honest_revoked'):5.1f}", flush=True)
