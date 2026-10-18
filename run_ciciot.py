"""CICIoT2023 sweep.  python run_ciciot.py --csv /content/merge.csv --cache /content/ciciot.npz
Attribute (device_type) is SYNTHETIC: CICIoT2023 has no per-row device id. rho controls how strongly
device_type shapes a client's class mix (rho=0: none, rho=1: strong)."""
import argparse, itertools, numpy as np
from abacfl.ciciot import load_ciciot
from abacfl.data import FedConfig, ArraySource
from abacfl.runner import RunCfg, run
from abacfl.detector import DetCfg
from abacfl.ledger import RepCfg

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="one CSV, a folder of CSVs, or a glob e.g. /content/ciciot/Merged*.csv"); ap.add_argument("--cache", default=None)
ap.add_argument("--cap", type=int, default=20000, help="max rows kept per fine-grained label")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--attacks", default="sign_flip,label_flip"); ap.add_argument("--rhos", default="1.0,0.0")
ap.add_argument("--variants", default="fedavg,median,krum,A0,A2,A3")
ap.add_argument("--frac", type=float, default=0.2); ap.add_argument("--placement", default="random")
ap.add_argument("--n_per_client", type=int, default=500)
ap.add_argument("--attack_scale", type=float, default=3.0, help="1.0 = stealthier attackers")
ap.add_argument("--alpha_type", type=float, default=0.5); ap.add_argument("--alpha_client", type=float, default=5.0)
ap.add_argument("--cluster_strength", type=float, default=0.0,
                help="0..1: device types draw rows from distinct k-means traffic profiles (covariate shift)")
ap.add_argument("--local_epochs", type=int, default=1)
ap.add_argument("--z0", type=float, default=3.0, help="anomaly threshold centre; lower = more sensitive")
ap.add_argument("--gamma", type=float, default=1.5)
ap.add_argument("--phi_mode", default="cos_or_D", choices=["cos_or_D", "D"], help="capture-guard trigger")
ap.add_argument("--phi_combine", default="blend", choices=["blend", "max"])
ap.add_argument("--tau_rep", type=float, default=None, help="stricter evidence threshold for reputation penalties (e.g. 0.7)")
ap.add_argument("--M", type=int, default=3, help="revoke if low reputation in M of last N rounds")
ap.add_argument("--N", type=int, default=5)
ap.add_argument("--eta_down", type=float, default=0.3)
ap.add_argument("--on_off", default=None, help="period,on_len,start e.g. 6,2,5 (adaptive on-off attacker)")
a = ap.parse_args()

Xtr, ytr, Xte, yte, classes = load_ciciot(a.csv, cap_per_label=a.cap, cache=a.cache)
n_cls, d = len(classes), Xtr.shape[1]
cfg_fed = FedConfig()
n_types = len(cfg_fed.clients_per_type)
cl = None
if a.cluster_strength > 0:
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_types, random_state=0, n_init=3).fit(
        Xtr[np.random.default_rng(0).choice(len(Xtr), min(100_000, len(Xtr)), replace=False)])
    cl = km.predict(Xtr)
    print("traffic-profile cluster sizes:", np.bincount(cl, minlength=n_types))
src_tr, src_te = ArraySource(Xtr, ytr, n_cls, cluster=cl), ArraySource(Xte, yte, n_cls)
print(f"{n_cls} classes, {d} features, train {len(Xtr):,} test {len(Xte):,}")
print(f"{'attack':11}{'rho':>4} {'variant':8}{'F1':>7}{'sd':>6}{'AUC':>6}{'sd':>5}{'T@1%':>6}{'sd':>5}{'TPR':>6}{'FPR':>6}{'det.d':>6}{'rev.d':>6}{'hRev':>5}{'hCli':>6}")
for atk, rho in itertools.product(a.attacks.split(","), map(float, a.rhos.split(","))):
    for v in a.variants.split(","):
        rs = [run(RunCfg(variant=v, rounds=a.rounds, seed=s, attack=atk, frac_attackers=a.frac,
                         attacker_placement=a.placement,
                         attack_scale=a.attack_scale, local_epochs=a.local_epochs,
                         det=DetCfg(z0=a.z0, gamma=a.gamma, phi_mode=a.phi_mode, phi_combine=a.phi_combine),
                         rep=RepCfg(tau_rep=a.tau_rep, M=a.M, N=a.N, eta_down=a.eta_down),
                         on_off=tuple(int(x) for x in a.on_off.split(',')) if a.on_off else None,
                         fed=FedConfig(rho=rho, n_cls=n_cls, d=d, type_shift=0.0, n_per_client=a.n_per_client,
                                       alpha_type=a.alpha_type, alpha_client=a.alpha_client,
                                       cluster_strength=a.cluster_strength)),
                  src_tr, src_te) for s in range(a.seeds)]
        m = lambda key: np.nanmean([r.get(key, np.nan) for r in rs]) if any(key in r for r in rs) else float("nan")
        sd = lambda key: np.nanstd([r.get(key, np.nan) for r in rs]) if any(key in r for r in rs) else float("nan")
        print(f"{atk:11}{rho:4.1f} {v:8}{m('final_f1'):7.3f}{sd('final_f1'):6.3f}{m('auc'):6.2f}{sd('auc'):5.2f}"
              f"{m('tpr_at_1pct'):6.2f}{sd('tpr_at_1pct'):5.2f}{m('tpr'):6.2f}{m('fpr'):6.2f}"
              f"{m('detect_delay'):6.1f}{m('revoke_delay'):6.1f}{m('honest_revoked'):5.1f}{m('honest_clients_revoked'):6.2f}", flush=True)
