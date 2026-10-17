# ABAC-aware cohort-conditioned malicious-client detection — Phase 1

Status: runs end to end on SYNTHETIC non-IID data (numpy only, no GPU, no blockchain).
Real datasets (N-BaIoT, CICIoT2023) are the next step: wrap pooled train/test arrays in
`abacfl.data.ArraySource` and pass them to `run(cfg, src_train, src_test)`; set `FedConfig.type_shift=0`
and `FedConfig.d` to the feature count. Standardise features and split train/test BEFORE wrapping.

## Run on Colab
    !unzip -q abacfl_phase1.zip -d /content/p1 && cd /content/p1
    !python run_phase1.py --seeds 3 --rounds 30                    # random attackers
    !python run_phase1.py --seeds 3 --placement cohort --attacks sign_flip --rhos 1.0   # cohort-capture attack

## Layout
    abacfl/core.py      numpy MLP, local training, macro-F1
    abacfl/data.py      federation builder; rho = attribute<->data correlation knob
    abacfl/attacks.py   label_flip, sign_flip, noise, scale, on-off (all credentialed)
    abacfl/detector.py  cohorts + back-off, shrunk baselines, capture guard, risk, weights, aggregation
    abacfl/ledger.py    reputation/status state machine + InMemoryLedger (hash chain + Merkle root)
    abacfl/runner.py    one experiment; variants fedavg|median|trimmed|krum|A0|A1|A2|A3

## Phase 2 hook (real Fabric, option A)
`InMemoryLedger` defines the interface the Fabric client must implement:
`gate(k)`, `weight_cap(k)`, `reputation(k)`, `apply_round(t, a_by_client)`, `commit(t, records)`.
Phase 2 = a `FabricLedger` that calls a REST gateway (laptop, tunnelled) which invokes chaincode.
Scoring stays off-chain in Colab; the chain holds state, enforces the gate, stores per-round Merkle commits.

## Variants
A0 global baseline | A1 cohort baseline | A2 cohort + shrinkage | A3 A2 + capture guard (blend)
Metrics: final macro-F1 (last 5 rounds), TPR/FPR per client-round, detect/revoke delay (rounds),
honest revocation events (hRev; counts repeated revoke->probation->revoke cycles).

## CICIoT2023 (merge.csv: 39 numeric features + Label)
    # Colab: copy merge.csv to /content (or mount Drive), then
    !python run_ciciot.py --csv /content/merge.csv --cache /content/ciciot.npz --cap 20000 --seeds 3 --rounds 30
The loader streams the CSV in chunks, keeps <= `--cap` random rows per fine label, drops duplicates,
groups labels into 8 classes (Benign, DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce), splits
train/test BEFORE normalising, then log1p (heavy-tailed cols) + z-score using train statistics only.
`--cache` stores the processed arrays so later runs start in seconds.
device_type is synthetic (no per-row device id in the dataset); rho controls how much it shapes class mix.
