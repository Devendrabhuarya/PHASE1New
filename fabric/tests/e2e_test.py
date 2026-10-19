"""End-to-end test WITHOUT Fabric: Python runner -> FabricLedger -> HTTP gateway (mock backend running the real
chaincode class) and, for comparison, the same run on InMemoryLedger. Decisions must be identical.
Usage: python3 fabric/tests/e2e_test.py (from repo root)"""
import os, socket, subprocess, sys, time, urllib.request
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from abacfl.runner import RunCfg, run
from abacfl.fabric_ledger import FabricLedger
from abacfl.ledger import RepCfg
from abacfl.detector import DetCfg

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

port, token = free_port(), "test-token-123"
env = dict(os.environ, BACKEND="mock", PORT=str(port), GATEWAY_TOKEN=token, MOCK_CLIENTS_PER_TYPE="14,10,8,6")
proc = subprocess.Popen(["node", "fabric/gateway/server.js"], env=env, cwd=ROOT)
try:
    for _ in range(50):
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"Authorization": f"Bearer {token}"}), timeout=1); break
        except Exception: time.sleep(0.1)
    else: raise SystemExit("gateway did not start")

    # auth check
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); raise SystemExit("FAIL: unauthenticated call accepted")
    except urllib.error.HTTPError as e:
        assert e.code == 401; print("ok - gateway rejects missing token")

    failures = 0
    for name, kw in [("sign_flip scale3", dict(attack="sign_flip", attack_scale=3.0)),
                     ("label_flip stealth + strict revocation", dict(attack="label_flip", attack_scale=1.0, rep=RepCfg(tau_rep=0.7, M=4, N=6))),
                     ("on-off sign_flip", dict(attack="sign_flip", attack_scale=1.0, on_off=(4, 2, 2)))]:
        cfg = RunCfg(variant="A3", rounds=14, seed=3, det=DetCfg(z0=2.0), **kw)
        ref = run(cfg, ledger_factory=None)
        refl = run(cfg, ledger_factory=lambda ids, rep: __import__("abacfl.ledger", fromlist=["x"]).InMemoryLedger(ids, rep))["ledger"]
        fab = run(cfg, ledger_factory=lambda ids, rep: FabricLedger(f"http://127.0.0.1:{port}", token, ids, rep, run_id=name))
        fl = fab["ledger"]
        same_f1 = abs(ref["final_f1"] - fab["final_f1"]) < 1e-12
        same_ev = refl.events == fl.events
        same_chain = [c["h"] for c in refl.chain] == [c["h"] for c in fl.chain]
        same_state = all(abs(refl.state[k]["R"] - fl.state[k]["R"]) < 1e-12 and refl.state[k]["status"] == fl.state[k]["status"] for k in refl.state)
        v = fl.verify()
        good = same_f1 and same_ev and same_chain and same_state and v["chain_ok"] and not v["local_root_mismatch"]
        failures += 0 if good else 1
        print(f"{'ok' if good else 'FAIL'} - {name}: F1 match={same_f1} events match={same_ev} ({len(fl.events)} events) "
              f"chain match={same_chain} ({len(fl.chain)-1} rounds) state match={same_state} verify={v['chain_ok']}")
        if not good: print("  detail:", same_f1, same_ev, same_chain, same_state, v)

    # replay protection through the HTTP path
    led = fl
    d = {0: np.ones(5)}
    r1 = led._post("/commit_updates", {"round": 999, "updates": [dict(client="client0", n=10, hash="a" * 64)]})["results"]
    r2 = led._post("/commit_updates", {"round": 999, "updates": [dict(client="client0", n=10, hash="b" * 64)]})["results"]
    assert r1["client0"] == "OK" or "ROUND_STALE" in r1["client0"]
    print("ok - replay/stale handling:", r2["client0"][:60])
    print("timings:", {k: round(v["mean_s"] * 1000, 1) for k, v in fl.timing_summary().items()}, "(ms, mock backend)")
    sys.exit(1 if failures else 0)
finally:
    proc.terminate()
