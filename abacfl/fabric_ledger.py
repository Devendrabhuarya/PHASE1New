"""FabricLedger: same interface as InMemoryLedger, but every gate decision, update commitment, reputation
transition and round commit goes through the REST gateway to the Hyperledger Fabric chaincode.

Scoring stays off-chain (in this process). The chain stores state, enforces the gate, applies the reputation
transition rule, and keeps a Merkle-root hash chain of per-round records.
"""
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from .ledger import merkle_root_str


class FabricLedger:
    def __init__(self, url, token, client_ids, rep_cfg, run_id="run", timeout=180, policy=None):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout
        self.ids = list(client_ids)
        self.cfg = rep_cfg
        self.timings = defaultdict(list)       # call name -> [seconds, ...]
        self.events, self.records = [], {}
        self.chain = [dict(h="0" * 64, root=None, t=-1)]
        self._gate_cache = None
        cfg = asdict(rep_cfg)
        cfg.update(policy or {})               # e.g. {"allowed_device_types": ["type0", ...]}
        self._post("/init", {"config": cfg}, "init")
        self._post("/reset", {"runId": run_id}, "reset")
        res = self._post("/register_batch", {"clients": [self._cid(k) for k in self.ids]}, "register")["results"]
        bad = {k: v for k, v in res.items() if v != "OK"}
        if bad:
            raise RuntimeError(f"client registration failed: {bad}")
        self.state = {}
        self._refresh()

    # ---- plumbing ----
    @staticmethod
    def _cid(k):
        return f"client{k}"

    def _post(self, path, payload, name=None):
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"gateway {path} -> HTTP {e.code}: {e.read().decode()[:500]}") from None
        self.timings[name or path.strip("/")].append(time.perf_counter() - t0)
        return out

    def _refresh(self):
        st = self._post("/states", {}, "states")
        self.state = {k: self._pick(st[self._cid(k)]) for k in self.ids}

    @staticmethod
    def _pick(s):
        return dict(R=float(s["R"]), status=s["status"], window=[int(x) for x in s["window"]], cooldown=s["cooldown"], probation_left=s["probation_left"])

    # ---- interface used by runner.py ----
    def _ensure_gate(self):
        if self._gate_cache is None:               # one batched on-chain Gate evaluation per round
            r = self._post("/gate_batch", {"clients": [self._cid(k) for k in self.ids]}, "gate")["results"]
            self._gate_cache = {k: r[self._cid(k)] for k in self.ids}

    def gate(self, k):
        self._ensure_gate()
        return bool(self._gate_cache[k]["allow"])

    def gate_reason(self, k):
        self._ensure_gate()
        return self._gate_cache[k].get("reason")

    def weight_cap(self, k):
        self._ensure_gate()
        return float(self._gate_cache[k].get("cap", 1.0))

    def reputation(self, k):
        return self.state[k]["R"]

    def commit_updates(self, t, deltas, n_k):
        ups = [dict(client=self._cid(k), n=int(n_k[k]),
                    hash=hashlib.sha256(np.ascontiguousarray(d, dtype=np.float64).tobytes()).hexdigest())
               for k, d in deltas.items()]
        res = self._post("/commit_updates", {"round": t, "updates": ups}, "commit_updates")["results"]
        self.last_update_errors = {k: v for k, v in res.items() if v != "OK"}
        return {k: res[self._cid(k)] == "OK" for k in deltas}

    def apply_round(self, t, a_by_client):
        scores = {self._cid(k): float(a) for k, a in a_by_client.items()}
        r = self._post("/apply_round", {"round": t, "scores": scores}, "apply_round")
        for k in self.ids:
            self.state[k] = self._pick(r["states"][self._cid(k)])
        # chaincode iterates keys lexicographically ("client10" < "client7"); canonicalise to numeric id order
        for (rt, cid, ev) in sorted(r["events"], key=lambda e: int(e[1][len("client"):])):
            self.events.append((rt, int(cid[len("client"):]), ev))
        self._gate_cache = None

    def commit(self, t, records, model_hash=None):
        recs = [json.dumps(x, sort_keys=True) for x in records]
        r = self._post("/commit_round", {"round": t, "records": recs, "modelHash": model_hash or ""}, "commit_round")
        self.records[t] = recs
        self.chain.append(dict(h=r["hash"], root=r["root"], t=t))
        return r["hash"]

    # ---- audit ----
    def verify(self):
        """Chain-side verification of the hash chain + local recomputation of every round's Merkle root."""
        v = self._post("/verify", {}, "verify")
        bad = []
        for t, recs in self.records.items():
            on = self._post("/round", {"round": t}, "get_round")
            if on is None or on["root"] != merkle_root_str(recs):
                bad.append(t)
        return dict(chain_ok=bool(v.get("ok")), rounds=v.get("rounds"), local_root_mismatch=bad, head=v.get("head"))

    def timing_summary(self):
        return {k: dict(n=len(v), mean_s=float(np.mean(v)), p95_s=float(np.percentile(v, 95))) for k, v in self.timings.items()}
