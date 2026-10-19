"""Virtual-client partitioners for the cohort-conditioned ABAC-FL project.

Two modes, same output format:

* partition_natural(): dataset has a real device id per row (N-BaIoT). Each
  device's rows are split into several virtual clients with mild within-device
  class skew. Cohort key = device type (real).

* partition_synthetic(): dataset has no device id (CICIoT2023). Clients get a
  synthetic device_type; `corr` in [0, 1] controls how strongly device_type
  predicts the client's class mix (corr=0: attributes carry no information).

Each client is a dict:
    {"client_id": int, "idx": np.ndarray of row indices (disjoint across clients),
     "attrs": {"client_id", "device_type", "org", "location_class"}}
The attrs mimic the static certificate attributes issued by the CA.
"""
from __future__ import annotations

import numpy as np

ORGS = ["org1", "org2"]
LOCATIONS = ["home", "remote"]


def _attrs(cid: int, device_type: str, rng: np.random.Generator) -> dict:
    return {
        "client_id": f"c{cid:03d}",
        "device_type": device_type,
        "org": ORGS[int(rng.integers(len(ORGS)))],
        "location_class": LOCATIONS[int(rng.integers(len(LOCATIONS)))],
    }


def _allocate(y: np.ndarray, q: np.ndarray, rng: np.random.Generator,
              min_size: int) -> list[np.ndarray]:
    """Split rows among K clients so each class is divided in proportion to
    q[k, c] (column-normalised). Disjoint, covers every row. q: (K, C)."""
    K, C = q.shape
    classes = np.unique(y)
    assert C == len(classes)
    parts: list[list[np.ndarray]] = [[] for _ in range(K)]
    for ci, c in enumerate(classes):
        rows = np.flatnonzero(y == c)
        rng.shuffle(rows)
        w = q[:, ci] + 1e-12
        w = w / w.sum()
        cuts = (np.cumsum(w)[:-1] * len(rows)).astype(int)
        for k, chunk in enumerate(np.split(rows, cuts)):
            parts[k].append(chunk)
    out = [np.concatenate(p) if p else np.array([], dtype=int) for p in parts]
    # top up tiny clients by borrowing from the largest one (keeps disjointness)
    for k in range(K):
        while len(out[k]) < min_size:
            donor = int(np.argmax([len(o) for o in out]))
            take = min(min_size - len(out[k]), max(1, len(out[donor]) // 10))
            out[k] = np.concatenate([out[k], out[donor][-take:]])
            out[donor] = out[donor][:-take]
    for o in out:
        rng.shuffle(o)
    return out


def partition_natural(y: np.ndarray, device_ids: np.ndarray,
                      device_types: dict, clients_per_device: int = 5,
                      alpha_within: float = 5.0, min_size: int = 50,
                      seed: int = 0) -> list[dict]:
    """device_ids: per-row device id; device_types: {device_id: type string}."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    clients, cid = [], 0
    for dev in sorted(device_types, key=str):
        rows = np.flatnonzero(device_ids == dev)
        yd = y[rows]
        present = np.unique(yd)
        # large alpha_within -> mild skew; small -> strong skew
        q = np.zeros((clients_per_device, len(present)))
        for k in range(clients_per_device):
            q[k] = rng.dirichlet(alpha_within * np.ones(len(present)))
        # allocate within this device (relabel classes locally)
        remap = {c: i for i, c in enumerate(present)}
        yl = np.array([remap[c] for c in yd])
        parts = _allocate(yl, q, rng, min_size)
        for p in parts:
            clients.append({"client_id": cid, "idx": rows[p],
                            "attrs": _attrs(cid, device_types[dev], rng)})
            cid += 1
    return clients


def partition_synthetic(y: np.ndarray, n_clients: int = 40,
                        device_types: tuple = ("camera", "doorbell", "thermostat", "hub"),
                        type_probs: tuple | None = None, corr: float = 0.7,
                        alpha_type: float = 0.5, alpha_client: float = 5.0,
                        min_size: int = 50, seed: int = 0) -> list[dict]:
    """corr=0: device_type is independent of data. corr=1: clients of a type
    share that type's class mix (plus client-level noise from alpha_client)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    C = len(classes)
    T = len(device_types)
    type_probs = np.ones(T) / T if type_probs is None else np.asarray(type_probs, float)
    types = rng.choice(T, size=n_clients, p=type_probs / type_probs.sum())
    # make sure every type has at least one client
    for t in range(T):
        if not np.any(types == t):
            types[rng.integers(n_clients)] = t
    p_type = rng.dirichlet(alpha_type * np.ones(C), size=T)        # (T, C)
    q = np.zeros((n_clients, C))
    for k in range(n_clients):
        u_k = rng.dirichlet(alpha_type * np.ones(C))                # attribute-free
        target = corr * p_type[types[k]] + (1 - corr) * u_k
        q[k] = rng.dirichlet(alpha_client * C * target + 1e-3)
    parts = _allocate(y, q, rng, min_size)
    return [{"client_id": k, "idx": parts[k],
             "attrs": _attrs(k, device_types[types[k]], rng)}
            for k in range(n_clients)]


def cohort_key(attrs: dict, level: str = "device_type") -> str:
    """Cohort key derived ONLY from CA-signed attributes."""
    if level == "device_type":
        return attrs["device_type"]
    if level == "device_type+org":
        return f'{attrs["device_type"]}|{attrs["org"]}'
    return "global"


def class_dist(y: np.ndarray, idx: np.ndarray, n_classes: int) -> np.ndarray:
    h = np.bincount(y[idx], minlength=n_classes).astype(float)
    return h / max(h.sum(), 1.0)
