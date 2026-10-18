"""CICIoT2023 loader for the merged CSV (39 numeric features + 'Label').

Memory-safe: streams the CSV in chunks and keeps a uniform random sample of at most `cap_per_label`
rows per fine-grained label (reservoir by random keys), then de-duplicates, groups labels into
8 classes (Benign + 7 attack categories), splits train/test BEFORE any normalisation, and
normalises with TRAIN statistics only (log1p for heavy-tailed columns, then z-score, clip).
"""
import glob, os
import numpy as np
import pandas as pd

CLASSES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
_WEB = {"browserhijacking", "xss", "uploading_attack", "sqlinjection", "backdoor_malware", "commandinjection"}


def coarse_label(s):
    l = str(s).strip().lower()
    if l.startswith("benign"): return "Benign"
    if l.startswith("ddos"): return "DDoS"
    if l.startswith("dos"): return "DoS"
    if l.startswith("mirai"): return "Mirai"
    if l.startswith("recon") or l == "vulnerabilityscan": return "Recon"
    if l in ("mitm-arpspoofing", "dns_spoofing"): return "Spoofing"
    if l in _WEB: return "Web"
    if l == "dictionarybruteforce": return "BruteForce"
    return None


def list_csvs(path):
    """path may be one CSV, a directory of CSVs, or a glob like /content/Merged*.csv"""
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
    elif any(ch in path for ch in "*?["):
        files = glob.glob(path)
    else:
        files = [path]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"no CSV files found for {path!r}")
    return files


def _chunks(files, chunksize, verbose):
    for i, f in enumerate(files, 1):
        if verbose:
            print(f"file {i}/{len(files)}: {os.path.basename(f)}", flush=True)
        yield from pd.read_csv(f, chunksize=chunksize)


def load_ciciot(csv_path, cap_per_label=20000, chunksize=500_000, seed=0, test_frac=0.2,
                label_col="Label", cache=None, verbose=True):
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"], list(z["classes"])
    rng = np.random.default_rng(seed)
    keep = {}                                    # fine label -> DataFrame (with random key 'u__')
    n_read = 0
    files = list_csvs(csv_path)
    for ch in _chunks(files, chunksize, verbose):
        ch.columns = [c.strip() for c in ch.columns]
        n_read += len(ch)
        ch["u__"] = rng.random(len(ch))
        for lab, g in ch.groupby(label_col):
            prev = keep.get(lab)
            g = g if prev is None else pd.concat([prev, g], ignore_index=True)
            keep[lab] = g.nsmallest(cap_per_label, "u__") if len(g) > cap_per_label else g
        if verbose:
            print(f"  total read {n_read:,} rows; {len(keep)} fine labels", flush=True)
    df = pd.concat(keep.values(), ignore_index=True).drop(columns="u__")
    feats = [c for c in df.columns if c != label_col]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats)
    n0 = len(df)
    df = df.drop_duplicates()
    df["cls"] = df[label_col].map(coarse_label)
    unmapped = sorted(df.loc[df["cls"].isna(), label_col].unique())
    if unmapped:
        print("WARNING unmapped labels dropped:", unmapped)
    df = df.dropna(subset=["cls"])
    y = df["cls"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X = df[feats].to_numpy(dtype=np.float64)
    if verbose:
        print(f"after dedup {n0:,}->{len(df):,}; features={len(feats)}")
        print({c: int((y == i).sum()) for i, c in enumerate(CLASSES)})
    # stratified split BEFORE normalisation
    tr, te = [], []
    for c in range(len(CLASSES)):
        idx = rng.permutation(np.where(y == c)[0])
        k = int(round(len(idx) * (1 - test_frac)))
        tr.append(idx[:k]); te.append(idx[k:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    Xtr, Xte = X[tr], X[te]
    q99, q1, med = np.percentile(Xtr, 99, 0), np.percentile(Xtr, 1, 0), np.median(Xtr, 0)
    heavy = (Xtr.min(0) >= 0) & ((q99 - q1) > 100)        # non-negative and wide dynamic range
    for M in (Xtr, Xte):
        M[:, heavy] = np.log1p(M[:, heavy])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    Xtr = np.clip((Xtr - mu) / sd, -10, 10)
    Xte = np.clip((Xte - mu) / sd, -10, 10)
    if verbose:
        print("log1p applied to:", [f for f, h in zip(feats, heavy) if h])
    out = (Xtr, y[tr], Xte, y[te], CLASSES)
    if cache:
        np.savez_compressed(cache, Xtr=Xtr, ytr=y[tr], Xte=Xte, yte=y[te], classes=np.array(CLASSES))
    return out
