"""Phase 2 runner (Colab). Runs the frozen configuration twice per seed:
  (1) reference with the in-memory ledger, (2) with every gate / reputation update / Merkle commit going through the
Fabric gateway. Reports accuracy parity, on-chain latency per operation, and audit verification.
  Real data : python run_phase2.py --gateway https://<tunnel> --token $TOK --csv /content/ciciot/Merged*.csv --cache /content/ciciot.npz
  Quick test: python run_phase2.py --gateway http://127.0.0.1:8080 --token $TOK --synthetic"""
import argparse, json, numpy as np
from abacfl.data import FedConfig, ArraySource
from abacfl.runner import RunCfg, run
from abacfl.detector import DetCfg
from abacfl.ledger import RepCfg, InMemoryLedger
from abacfl.fabric_ledger import FabricLedger

ap = argparse.ArgumentParser()
ap.add_argument("--gateway", required=True); ap.add_argument("--token", required=True)
ap.add_argument("--csv"); ap.add_argument("--cache"); ap.add_argument("--cap", type=int, default=20000)
ap.add_argument("--synthetic", action="store_true", help="skip CICIoT2023 (plumbing test only)")
ap.add_argument("--attacks", default="sign_flip,label_flip"); ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--rounds", type=int, default=30); ap.add_argument("--variant", default="A3")
ap.add_argument("--out", default="phase2_audit.json")
a = ap.parse_args()

fed = FedConfig(rho=1.0, cluster_strength=0.9, alpha_client=1.0, n_per_client=500)
src_tr = src_te = None
if not a.synthetic:
    from abacfl.ciciot import load_ciciot
    Xtr, ytr, Xte, yte, classes = load_ciciot(a.csv, cap_per_label=a.cap, cache=a.cache)
    from sklearn.cluster import MiniBatchKMeans
    k = len(fed.clients_per_type)
    cl = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit(
        Xtr[np.random.default_rng(0).choice(len(Xtr), min(100_000, len(Xtr)), replace=False)]).predict(Xtr)
    src_tr, src_te = ArraySource(Xtr, ytr, len(classes), cluster=cl), ArraySource(Xte, yte, len(classes))
    fed = FedConfig(rho=1.0, n_cls=len(classes), d=Xtr.shape[1], type_shift=0.0, n_per_client=500, alpha_type=0.5,
                    alpha_client=1.0, cluster_strength=0.9)

summary, audit = [], []
for atk in a.attacks.split(","):
    for seed in range(a.seeds):
        cfg = RunCfg(variant=a.variant, rounds=a.rounds, seed=seed, attack=atk, attack_scale=1.0, fed=fed,
                     det=DetCfg(z0=2.0), rep=RepCfg(tau_rep=0.7, M=4, N=6))      # frozen settings
        ref = run(cfg, src_tr, src_te, ledger_factory=InMemoryLedger)
        fab = run(cfg, src_tr, src_te, ledger_factory=lambda ids, rep: FabricLedger(
            a.gateway, a.token, ids, rep, run_id=f"{atk}-s{seed}"))
        L, RL = fab["ledger"], ref["ledger"]
        v = L.verify()
        row = dict(attack=atk, seed=seed, f1_ref=ref["final_f1"], f1_fabric=fab["final_f1"],
                   events_equal=RL.events == L.events, chain_equal=[c["h"] for c in RL.chain] == [c["h"] for c in L.chain],
                   chain_ok=v["chain_ok"], root_mismatch=v["local_root_mismatch"], head=v["head"],
                   revoked=sorted({k for (_, k, e) in L.events if e == "REVOKED"}), timings=L.timing_summary())
        summary.append(row)
        audit.append(dict(attack=atk, seed=seed, events=L.events, chain=L.chain))
        print(f"{atk:11} seed{seed} F1 ref={row['f1_ref']:.4f} fabric={row['f1_fabric']:.4f} | events equal={row['events_equal']} "
              f"chain equal={row['chain_equal']} chain_ok={row['chain_ok']} | revoked clients={row['revoked']}", flush=True)

ops = sorted({o for r in summary for o in r["timings"]})
print("\nOn-chain latency per call (ms)  mean / p95 / calls-per-run")
for o in ops:
    m = np.mean([r["timings"][o]["mean_s"] for r in summary if o in r["timings"]]) * 1000
    p = np.mean([r["timings"][o]["p95_s"] for r in summary if o in r["timings"]]) * 1000
    n = np.mean([r["timings"][o]["n"] for r in summary if o in r["timings"]])
    print(f"  {o:16}{m:9.1f}{p:9.1f}{n:8.0f}")
json.dump(dict(summary=summary, audit=audit), open(a.out, "w"), indent=1, default=str)
print("saved", a.out)
