"""Reputation/status state machine + ledger interface.

InMemoryLedger  : used in Phase 1 (same interface the Fabric client will implement in Phase 2).
Rule of the design: statistical scoring is OFF-chain; the chain stores state, enforces the gate,
and commits per-round records (hash chain + Merkle root).
"""
from dataclasses import dataclass
import hashlib, json


@dataclass
class RepCfg:
    R0: float = 0.5
    R_min: float = 0.2       # gate threshold
    R_th: float = 0.3        # "low reputation" threshold for revocation window
    tau_a: float = 0.5       # flag threshold used for TPR/FPR bookkeeping
    tau_rep: float = None    # threshold used for reputation penalties (None -> tau_a). Higher = stricter evidence
    eta_up: float = 0.05
    eta_down: float = 0.3
    N: int = 5
    M: int = 3
    Tc: int = 5              # cooldown rounds before re-enrolment
    P: int = 3               # probation rounds
    probation_cap: float = 0.5


def new_state(cfg):
    return dict(R=cfg.R0, status="ACTIVE", window=[], cooldown=0, probation_left=0)


def transition(s, a, cfg, event_log=None, t=None, k=None):
    """a = anomaly score of this round (None if client did not participate)."""
    s = dict(s, window=list(s["window"]))
    if s["status"] == "REVOKED":
        s["cooldown"] -= 1
        if s["cooldown"] <= 0:
            s.update(status="PROBATION", R=cfg.R0, window=[], probation_left=cfg.P)
            if event_log is not None:
                event_log.append((t, k, "PROBATION"))
        return s
    if a is not None:
        thr = cfg.tau_a if cfg.tau_rep is None else cfg.tau_rep
        if a < thr:
            s["R"] = min(1.0, s["R"] + cfg.eta_up * (1 - a))
        else:
            s["R"] = max(0.0, s["R"] - cfg.eta_down * a)
    s["window"] = (s["window"] + [int(s["R"] < cfg.R_th)])[-cfg.N:]
    if sum(s["window"]) >= cfg.M:
        s.update(status="REVOKED", cooldown=cfg.Tc, window=[])
        if event_log is not None:
            event_log.append((t, k, "REVOKED"))
    elif s["status"] == "PROBATION" and a is not None:
        s["probation_left"] -= 1
        if s["probation_left"] <= 0:
            s["status"] = "ACTIVE"
    return s


def merkle_root(items):
    layer = [hashlib.sha256(json.dumps(x, sort_keys=True).encode()).digest() for x in items] or [b"\0" * 32]
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [hashlib.sha256(layer[i] + layer[i + 1]).digest() for i in range(0, len(layer), 2)]
    return layer[0].hex()


class InMemoryLedger:
    def __init__(self, client_ids, cfg: RepCfg):
        self.cfg = cfg
        self.state = {k: new_state(cfg) for k in client_ids}
        self.chain = [dict(h="0" * 64, root=None, t=-1)]
        self.events = []

    # --- what chaincode will do at the gate (Phase 2: read from world state) ---
    def gate(self, k):
        s = self.state[k]
        return s["status"] in ("ACTIVE", "PROBATION") and s["R"] >= self.cfg.R_min - 1e-9

    def weight_cap(self, k):
        return self.cfg.probation_cap if self.state[k]["status"] == "PROBATION" else 1.0

    def reputation(self, k):
        return self.state[k]["R"]

    # --- end-of-round commit (one tx per round) ---
    def apply_round(self, t, a_by_client):
        for k in self.state:
            self.state[k] = transition(self.state[k], a_by_client.get(k), self.cfg, self.events, t, k)

    def commit(self, t, records):
        root = merkle_root(records)
        prev = self.chain[-1]["h"]
        h = hashlib.sha256((prev + root + str(t)).encode()).hexdigest()
        self.chain.append(dict(h=h, root=root, t=t))
        return h
