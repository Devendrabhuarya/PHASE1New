"""Federation builder with a tunable attribute<->data correlation (rho).

rho = 0 : device_type tells you nothing about a client's data (cohorts should not help).
rho = 1 : device_type strongly shapes class mix (and, for synthetic data, a feature shift).

Real datasets: wrap pooled arrays in ArraySource (train/test split done BEFORE this step).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class FedConfig:
    clients_per_type: tuple = (14, 10, 8, 6)
    n_cls: int = 6
    d: int = 32
    n_per_client: int = 300
    rho: float = 1.0           # attribute-data correlation
    alpha_type: float = 0.5    # skew of type-level class mix (small = more skew)
    alpha_client: float = 5.0  # client-level variation around its type (large = tighter)
    type_shift: float = 0.6    # covariate shift per type (synthetic only), scaled by rho
    n_test_per_class: int = 240
    cluster_strength: float = 0.0  # real-data only: prob. a client's rows come from its type's traffic-profile cluster (x rho)


class SyntheticSource:
    def __init__(self, rng, d, n_cls, sep=0.4):
        self.mu = rng.normal(0, sep, (n_cls, d))
        self.d, self.n_cls = d, n_cls

    def sample(self, counts, rng, cluster=None, p_cluster=0.0):
        ys = np.repeat(np.arange(self.n_cls), counts)
        X = self.mu[ys] + rng.normal(0, 1, (len(ys), self.d))
        return X.astype(np.float64), ys


class ArraySource:
    """Sample rows with replacement per class from pooled arrays (e.g. CICIoT2023, N-BaIoT)."""
    def __init__(self, X, y, n_cls, cluster=None):
        """cluster: optional per-row traffic-profile id (e.g. k-means on features). Lets a device
        type see a genuinely different feature distribution (covariate shift), not only a class mix."""
        self.X, self.n_cls = X, n_cls
        self.idx = [np.where(y == c)[0] for c in range(n_cls)]
        self.d = X.shape[1]
        self.by_cluster = None
        if cluster is not None:
            ks = np.unique(cluster)
            self.by_cluster = [{int(k): ix[cluster[ix] == k] for k in ks} for ix in self.idx]

    def sample(self, counts, rng, cluster=None, p_cluster=0.0):
        rows, ys = [], []
        for c, k in enumerate(counts):
            if k > 0 and len(self.idx[c]) > 0:
                pool = self.idx[c]
                if cluster is not None and self.by_cluster is not None and p_cluster > 0:
                    cp = self.by_cluster[c].get(int(cluster), pool[:0])
                    if len(cp) >= 20:
                        from_c = rng.random(k) < p_cluster
                        r = np.where(from_c, rng.choice(cp, k, replace=True), rng.choice(pool, k, replace=True))
                        rows.append(r); ys.append(np.full(k, c)); continue
                rows.append(rng.choice(pool, k, replace=True))
                ys.append(np.full(k, c))
        r = np.concatenate(rows)
        return self.X[r], np.concatenate(ys)


def make_federation(cfg: FedConfig, rng, src_train=None, src_test=None):
    n_types = len(cfg.clients_per_type)
    src_train = src_train or SyntheticSource(rng, cfg.d, cfg.n_cls)
    src_test = src_test or src_train
    uniform = np.ones(cfg.n_cls) / cfg.n_cls
    p_type = [rng.dirichlet(cfg.alpha_type * np.ones(cfg.n_cls)) for _ in range(n_types)]
    shifts = rng.normal(0, 1, (n_types, cfg.d)) * cfg.type_shift * cfg.rho
    clients, cid = [], 0
    for t in range(n_types):
        base = cfg.rho * p_type[t] + (1 - cfg.rho) * uniform
        for _ in range(cfg.clients_per_type[t]):
            q = rng.dirichlet(cfg.alpha_client * cfg.n_cls * base + 1e-3)
            counts = rng.multinomial(cfg.n_per_client, q)
            X, y = src_train.sample(counts, rng, cluster=t, p_cluster=cfg.rho * cfg.cluster_strength)
            clients.append(dict(id=cid, X=X + shifts[t], y=y, n=len(y),
                                attrs=dict(device_type=f"type{t}", org=f"org{cid % 2}",
                                           location_class=f"loc{cid % 3}")))
            cid += 1
    Xt, yt = [], []
    per = max(1, cfg.n_test_per_class // n_types)
    for t in range(n_types):
        X, y = src_test.sample(np.full(cfg.n_cls, per), rng)
        Xt.append(X + shifts[t]); yt.append(y)
    return clients, np.concatenate(Xt), np.concatenate(yt)
