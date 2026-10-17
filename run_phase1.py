"""Sweep runner. Usage: python run_phase1.py [--seeds 3] [--rounds 30]"""
import argparse, itertools, numpy as np
from abacfl.runner import RunCfg, run
from abacfl.data import FedConfig

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--attacks", default="sign_flip,label_flip,scale")
ap.add_argument("--rhos", default="1.0,0.0")
ap.add_argument("--variants", default="fedavg,median,krum,A0,A1,A2,A3")
ap.add_argument("--frac", type=float, default=0.2)
ap.add_argument("--placement", default="random")
a = ap.parse_args()

print(f"{'attack':11}{'rho':>4} {'variant':8}{'F1':>7}{'TPR':>6}{'FPR':>6}{'det.d':>6}{'rev.d':>6}{'hRev':>5}")
for atk, rho in itertools.product(a.attacks.split(","), map(float, a.rhos.split(","))):
    for v in a.variants.split(","):
        rs = [run(RunCfg(variant=v, rounds=a.rounds, seed=s, attack=atk, frac_attackers=a.frac,
                         attacker_placement=a.placement, fed=FedConfig(rho=rho))) for s in range(a.seeds)]
        m = lambda key: np.nanmean([r.get(key, np.nan) for r in rs])
        print(f"{atk:11}{rho:4.1f} {v:8}{m('final_f1'):7.3f}{m('tpr'):6.2f}{m('fpr'):6.2f}"
              f"{m('detect_delay'):6.1f}{m('revoke_delay'):6.1f}{m('honest_revoked'):5.1f}")
