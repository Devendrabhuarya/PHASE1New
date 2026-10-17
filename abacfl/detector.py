"""Cohort-conditioned anomaly scoring (Section: C1-C6 of the design).

modes
  'global'        A0  one population, global median reference + global baselines
  'cohort'        A1  cohort reference + raw cohort baselines
  'cohort_shrunk' A2  A1 + shrinkage of cohort baselines toward global (C3)
  'full'          A3  A2 + cohort-capture guard (C5)
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np

EPS = 1e-8
GLOBAL = "GLOBAL"


@dataclass
class DetCfg:
    mode: str = "full"
    m_min: int = 4          # min cohort size before backing off to parent/global
    kappa: float = 4.0      # shrinkage strength: lambda_c = n_c / (n_c + kappa)
    z0: float = 3.0
    gamma: float = 1.5
    rel_floor: float = 0.10  # floor on baseline scale: s >= rel_floor*|m| (guards tiny-MAD cohorts)
    lam: tuple = (1 / 3, 1 / 3, 1 / 3)
    phi_combine: str = "blend"  # 'blend': (1-phi)*a_coh + phi*a_glob | 'max': max(a_coh, phi*a_glob)
    phi_mode: str = "cos_or_D"  # 'cos_or_D' = max(1-cos(r_c,r_g), D_c) (C5) | 'D' = cross-cohort outlier only


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def _feats(d, r):
    return np.array([np.linalg.norm(d), _cos(d, r), np.linalg.norm(d - r)])


def _stats(F, rel_floor):
    m = np.median(F, axis=0)
    s = 1.4826 * np.median(np.abs(F - m), axis=0)
    return m, np.maximum(s, rel_floor * np.abs(m) + EPS)


def _z(f, m, s):
    return np.array([abs(f[0] - m[0]) / s[0],
                     max(0.0, (m[1] - f[1]) / s[1]),      # one-sided: LOW cosine is bad
                     abs(f[2] - m[2]) / s[2]])


def _sig(x, g, z0):
    return 1.0 / (1.0 + np.exp(-g * (x - z0)))


def score_round(deltas, keys, parents, cfg: DetCfg):
    """deltas: {client: vec}; keys: {client: cohort key (from CERT attrs)};
    parents: {cohort key: parent key} for back-off. Returns dict with a_tilde etc."""
    ids = list(deltas)
    V = np.stack([deltas[k] for k in ids])
    r_g = np.median(V, axis=0)
    lam = np.array(cfg.lam)

    # ---- effective cohorts with back-off (C1) ----
    eff = {}
    if cfg.mode == "global":
        eff = {k: GLOBAL for k in ids}
    else:
        cnt = Counter(keys[k] for k in ids)
        pcnt = Counter(parents.get(keys[k], keys[k]) for k in ids)
        for k in ids:
            key = keys[k]
            p = parents.get(key, key)
            eff[k] = key if cnt[key] >= cfg.m_min else (p if pcnt[p] >= cfg.m_min else GLOBAL)
    members = {}
    for k in ids:
        members.setdefault(eff[k], []).append(k)
    ref = {e: (r_g if e == GLOBAL else np.median(np.stack([deltas[k] for k in m]), axis=0))
           for e, m in members.items()}

    Fc = {k: _feats(deltas[k], ref[eff[k]]) for k in ids}     # vs (effective) cohort reference
    Fg = {k: _feats(deltas[k], r_g) for k in ids}             # vs global reference
    Gc = np.stack([Fc[k] for k in ids])
    Gg = np.stack([Fg[k] for k in ids])
    mG, sG = _stats(Gc, cfg.rel_floor)
    mGg, sGg = _stats(Gg, cfg.rel_floor)

    a, a_glob, a_coh = {}, {}, {}
    for k in ids:
        a_glob[k] = _sig(float(lam @ _z(Fg[k], mGg, sGg)), cfg.gamma, cfg.z0)
        if cfg.mode == "global":
            a_coh[k] = a_glob[k]
            continue
        e = eff[k]
        if e == GLOBAL:
            m, s = mG, sG
        else:
            F = np.stack([Fc[j] for j in members[e]])
            m, s = _stats(F, cfg.rel_floor)
            if cfg.mode in ("cohort_shrunk", "full"):
                n = len(members[e])
                lc = n / (n + cfg.kappa)
                m, s = lc * m + (1 - lc) * mG, lc * s + (1 - lc) * sG
        a_coh[k] = _sig(float(lam @ _z(Fc[k], m, s)), cfg.gamma, cfg.z0)

    # ---- capture guard (C5) ----
    phi = {}
    if cfg.mode == "full":
        es = [e for e in members if e != GLOBAL]
        dist = {e: float(np.linalg.norm(ref[e] - r_g)) for e in es}
        if len(es) >= 3:
            dv = np.array(list(dist.values()))
            med, mad = np.median(dv), 1.4826 * np.median(np.abs(dv - np.median(dv)))
            mad = max(mad, 0.1 * abs(med) + EPS)
        for e in es:
            one_minus_cos = float(np.clip(1 - _cos(ref[e], r_g), 0, 1))
            D = _sig((dist[e] - med) / mad, cfg.gamma, cfg.z0) if len(es) >= 3 else 0.0
            phi[e] = D if cfg.phi_mode == "D" else max(one_minus_cos, D)
        for k in ids:
            ph = phi.get(eff[k], 0.0)
            a[k] = ((1 - ph) * a_coh[k] + ph * a_glob[k]) if cfg.phi_combine == "blend" \
                else max(a_coh[k], ph * a_glob[k])
    else:
        a = dict(a_coh)
    return dict(a=a, a_coh=a_coh, a_glob=a_glob, eff=eff, phi=phi, ref_norm=float(np.linalg.norm(r_g)))


# ---------- risk, weights, aggregation ----------
def risk(a, R, alpha=0.7):
    return alpha * a + (1 - alpha) * (1 - R)


def g_weight(rho, lo=0.3, hi=0.7):
    if rho < lo:
        return 1.0
    if rho >= hi:
        return 0.0
    return (hi - rho) / (hi - lo)


def robust_aggregate(deltas, n_k, g, caps, cohort_of, clip_c=1.5, omega_max=None):
    ids = list(deltas)
    norms = np.array([np.linalg.norm(deltas[k]) for k in ids])
    m1 = np.median(norms)
    w = {}
    for k, nk in zip(ids, norms):
        w[k] = n_k[k] * g[k] * caps[k]
    if omega_max is not None:
        for _ in range(3):
            tot = sum(w.values())
            if tot <= 0:
                break
            byc = {}
            for k in ids:
                byc.setdefault(cohort_of[k], []).append(k)
            if len(byc) * omega_max < 1:
                break
            for c, ks in byc.items():
                share = sum(w[k] for k in ks) / tot
                if share > omega_max:
                    for k in ks:
                        w[k] *= omega_max / share
    tot = sum(w.values())
    if tot <= 0:
        return None, w
    out = np.zeros_like(next(iter(deltas.values())))
    for k, nk in zip(ids, norms):
        d = deltas[k] * min(1.0, clip_c * m1 / (nk + EPS))
        out += (w[k] / tot) * d
    return out, {k: w[k] / tot for k in ids}
