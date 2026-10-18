"""One FL experiment: baselines (fedavg/median/trimmed/krum) and A0-A3 detector variants."""
from dataclasses import dataclass, field
import numpy as np
from .core import MLP, macro_f1
from .data import FedConfig, make_federation
from .attacks import Attack
from .detector import DetCfg, score_round, risk, g_weight, robust_aggregate
from .ledger import RepCfg, InMemoryLedger

def auroc(pos, neg):
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def _tpr_at_fpr(pos, neg, fpr):
    """TPR when the threshold is set so that honest clients are flagged at rate `fpr` (matched-FPR comparison)."""
    if not pos or not neg:
        return float('nan')
    pos, neg = np.asarray(pos), np.asarray(neg)
    thr = np.quantile(neg, 1 - fpr)
    return float((pos > thr).mean() + 0.5 * (pos == thr).mean())


VARIANTS = {"A0": "global", "A1": "cohort", "A2": "cohort_shrunk", "A3": "full"}


@dataclass
class RunCfg:
    variant: str = "A3"            # fedavg | median | trimmed | krum | A0 | A1 | A2 | A3
    rounds: int = 30
    seed: int = 0
    d_h: int = 32
    local_epochs: int = 1
    lr: float = 0.1
    bs: int = 32
    attack: str = "sign_flip"      # none | label_flip | sign_flip | noise | scale
    attack_scale: float = 3.0
    on_off: tuple = None           # (period, on_len, start)
    frac_attackers: float = 0.2
    attacker_placement: str = "random"   # random | cohort (all attackers inside ONE device type)
    alpha_risk: float = 0.7
    omega_max: float = 0.5
    fed: FedConfig = field(default_factory=FedConfig)
    det: DetCfg = field(default_factory=DetCfg)
    rep: RepCfg = field(default_factory=RepCfg)


def _pick_attackers(clients, cfg, rng):
    n = len(clients)
    na = int(round(cfg.frac_attackers * n))
    if cfg.attack == "none" or na == 0:
        return set()
    if cfg.attacker_placement == "cohort":
        by = {}
        for c in clients:
            by.setdefault(c["attrs"]["device_type"], []).append(c["id"])
        biggest = max(by.values(), key=len)
        return set(rng.choice(biggest, min(na, len(biggest)), replace=False).tolist())
    return set(rng.choice(n, na, replace=False).tolist())


def run(cfg: RunCfg, src_train=None, src_test=None):
    rng = np.random.default_rng(cfg.seed)
    clients, Xt, yt = make_federation(cfg.fed, rng, src_train, src_test)
    n_cls, d = cfg.fed.n_cls, Xt.shape[1]
    model = MLP(d, cfg.d_h, n_cls)
    w = model.init(rng)
    byid = {c["id"]: c for c in clients}
    ids = list(byid)
    attackers = _pick_attackers(clients, cfg, rng)
    atk = Attack(cfg.attack, cfg.attack_scale, cfg.on_off)
    is_A = cfg.variant in VARIANTS
    det = DetCfg(**{**cfg.det.__dict__, "mode": VARIANTS.get(cfg.variant, "full")})
    ledger = InMemoryLedger(ids, cfg.rep) if is_A else None
    keys = {k: byid[k]["attrs"]["device_type"] for k in ids}
    parents = {}
    n_cap = 2 * np.median([byid[k]["n"] for k in ids])

    H = dict(f1=[], acc=[], tp=0, fn=0, fp=0, tn=0, fp_by_type={}, n_by_type={}, first_active={},
             first_flag={}, first_revoke={}, honest_revoked=0, honest_rev_set=set(), pre_attack_revoked=0, blocked_attackers=[], allowed=[])
    for t in range(cfg.rounds):
        allowed = [k for k in ids if (ledger.gate(k) if is_A else True)]
        H["allowed"].append(len(allowed))
        H["blocked_attackers"].append(sum(1 for k in attackers if k not in allowed))
        deltas, active = {}, {}
        for k in allowed:
            c = byid[k]
            bad = k in attackers and atk.active(t)
            active[k] = bad
            y = atk.poison_labels(c["y"], n_cls) if bad else c["y"]
            wk = model.local_train(w, c["X"], y, cfg.local_epochs, cfg.lr, cfg.bs, rng)
            dlt = wk - w
            deltas[k] = atk.poison_update(dlt, rng) if bad else dlt
            if bad:
                H["first_active"].setdefault(k, t)
        if not deltas:
            H["f1"].append(H["f1"][-1] if H["f1"] else 0.0); H["acc"].append(H["acc"][-1] if H["acc"] else 0.0)
            if is_A:
                ledger.apply_round(t, {})
            continue

        V = np.stack(list(deltas.values()))
        if cfg.variant == "fedavg":
            wt = np.array([byid[k]["n"] for k in deltas], float)
            agg = (V * (wt / wt.sum())[:, None]).sum(0)
        elif cfg.variant == "median":
            agg = np.median(V, axis=0)
        elif cfg.variant == "trimmed":
            b = int(0.2 * len(V)); S = np.sort(V, axis=0)
            agg = S[b:len(V) - b].mean(0) if len(V) > 2 * b else S.mean(0)
        elif cfg.variant == "krum":      # multi-Krum, given the TRUE f (advantage to baseline)
            n = len(V); f = int(round(cfg.frac_attackers * n)); m = max(1, n - f)
            D = ((V[:, None, :] - V[None, :, :]) ** 2).sum(-1)
            sc = np.sort(D, axis=1)[:, 1:max(2, n - f - 1)].sum(1)
            agg = V[np.argsort(sc)[:m]].mean(0)
        else:
            sc = score_round(deltas, keys, parents, det)
            a = sc["a"]
            rho = {k: risk(a[k], ledger.reputation(k), cfg.alpha_risk) for k in deltas}
            g = {k: g_weight(rho[k]) for k in deltas}
            caps = {k: ledger.weight_cap(k) for k in deltas}
            nk = {k: min(byid[k]["n"], n_cap) for k in deltas}
            omega = cfg.omega_max if cfg.variant == "A3" else None
            agg, wts = robust_aggregate(deltas, nk, g, caps, keys, omega_max=omega)
            if agg is None:
                agg = np.zeros_like(w)
            pos = [a[k] for k in deltas if k in attackers and active[k]]
            neg = [a[k] for k in deltas if k not in attackers]
            if pos and neg:
                H.setdefault('aucs', []).append(auroc(pos, neg))   # threshold-free separability this round
            H.setdefault('pos_s', []).extend(a[k] for k in deltas if k in attackers and active[k])
            H.setdefault('neg_s', []).extend(a[k] for k in deltas if k not in attackers)
            for k in deltas:                         # detection bookkeeping (flag = a_tilde >= tau_a)
                flag = a[k] >= cfg.rep.tau_a
                if k in attackers:
                    if active[k]:
                        H["tp" if flag else "fn"] += 1
                        if flag:
                            H["first_flag"].setdefault(k, t)
                else:
                    ty = keys[k]
                    H["n_by_type"][ty] = H["n_by_type"].get(ty, 0) + 1
                    if flag:
                        H["fp"] += 1; H["fp_by_type"][ty] = H["fp_by_type"].get(ty, 0) + 1
                    else:
                        H["tn"] += 1
            n_ev = len(ledger.events)
            ledger.apply_round(t, a)
            for (tt, k, ev) in ledger.events[n_ev:]:
                if ev == "REVOKED":
                    if k in attackers:
                        if k in H["first_active"] and tt >= H["first_active"][k]:
                            H["first_revoke"].setdefault(k, tt)
                        else:                      # revoked before it ever attacked (dormant on-off attacker)
                            H["pre_attack_revoked"] += 1
                    else:
                        H["honest_revoked"] += 1; H["honest_rev_set"].add(k)
            ledger.commit(t, [dict(k=k, a=round(float(a[k]), 4), rho=round(float(rho[k]), 4),
                                   w=round(float(wts[k]), 4), R=round(ledger.reputation(k), 4),
                                   s=ledger.state[k]["status"]) for k in deltas])
        w = w + agg
        pred = model.predict(w, Xt)
        H["f1"].append(macro_f1(yt, pred, n_cls)); H["acc"].append(float((pred == yt).mean()))

    out = dict(final_f1=float(np.mean(H["f1"][-5:])), final_acc=float(np.mean(H["acc"][-5:])),
               f1_curve=H["f1"], n_attackers=len(attackers))
    if is_A:
        P, N = H["tp"] + H["fn"], H["fp"] + H["tn"]
        out.update(tpr=H["tp"] / P if P else float("nan"), fpr=H["fp"] / N if N else float("nan"),
                   fpr_by_type={ty: H["fp_by_type"].get(ty, 0) / n for ty, n in H["n_by_type"].items()},
                   detect_delay=float(np.mean([H["first_flag"][k] - H["first_active"][k] for k in H["first_flag"]]))
                   if H["first_flag"] else float("nan"),
                   revoke_delay=float(np.mean([H["first_revoke"][k] - H["first_active"][k] for k in H["first_revoke"]]))
                   if H["first_revoke"] else float("nan"),
                   pre_attack_revoked=H["pre_attack_revoked"], attackers_revoked=len(H["first_revoke"]), honest_revoked=H["honest_revoked"],
                   honest_clients_revoked=len(H["honest_rev_set"]) / max(1, len(ids) - len(attackers)),
                   auc=float(np.mean(H['aucs'])) if H.get('aucs') else float('nan'),
                   tpr_at_1pct=_tpr_at_fpr(H.get('pos_s'), H.get('neg_s'), 0.01),
                   chain_len=len(ledger.chain))
    return out
