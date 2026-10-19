"""Parity: Python reputation transition vs the chaincode's JS transition on random score sequences.
Usage (repo root): python3 fabric/tests/parity_test.py"""
import json, os, random, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from abacfl.ledger import RepCfg, transition
from dataclasses import asdict

random.seed(7)
cfgs = [RepCfg(), RepCfg(tau_rep=0.7, M=4, N=6), RepCfg(eta_down=0.5, Tc=3, P=4)]
bad = 0
for cfg in cfgs:
    cases = []
    for _ in range(300):
        scores = [None if random.random() < 0.1 else round(random.random() ** random.choice([1, 2, 0.5]), 4) for _ in range(25)]
        cases.append(dict(state=dict(R=cfg.R0, status="ACTIVE", window=[], cooldown=0, probation_left=0), scores=scores))
    out = subprocess.run(["node", os.path.join(ROOT, "fabric/tests/transition_cli.js")], input=json.dumps(dict(cfg=asdict(cfg), cases=cases)),
                         capture_output=True, text=True, cwd=ROOT)
    if out.returncode: print(out.stderr); sys.exit(2)
    js = json.loads(out.stdout)
    for c, traj in zip(cases, js):
        s = dict(c["state"]); ev = []; nev = 0
        for i, a in enumerate(c["scores"]):
            s = transition(s, a, cfg, ev, i, 0)
            j = traj[i]["state"]
            je = traj[i]["event"]
            pe = ev[-1][2] if ev and ev[-1][0] == i and len(ev) > nev else None
            nev = len(ev)
            if abs(s["R"] - j["R"]) > 1e-9 or s["status"] != j["status"] or pe != je:
                bad += 1; print("MISMATCH", cfg, i, s, j); break
print("parity:", "FAIL" if bad else "ok (900 sequences x 25 rounds)")
sys.exit(1 if bad else 0)
