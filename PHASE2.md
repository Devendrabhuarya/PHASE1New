# Phase 2 — Real Hyperledger Fabric + Colab integration

Architecture: Colab (training, detector, risk) --HTTPS+bearer token--> tunnel --> REST gateway (WSL2) --Fabric Gateway SDK--> test-network (Org1+Org2, Fabric CA) running the `abacfl` chaincode.
Heavy maths stays off-chain. On-chain: ABAC gate (certificate attributes), rule-based reputation transitions, update-hash binding, per-round Merkle root + hash chain.

## What is tested in the sandbox vs. what you must run
| Component | Status |
|---|---|
| Chaincode logic + contract (25 checks, mock stub) | tested |
| Python <-> JS reputation parity (900 random sequences, incl. events) | tested |
| Gateway server (mock backend), Python `FabricLedger`, full FL run: F1, events, hash chain identical to in-memory | tested |
| Real Fabric network, `enroll_clients.sh`, `backend_fabric.js` SDK code, tunnel, endorsement policy | NOT tested (no Docker/npm here). Expect small fixes. |

Re-run the tested part anywhere with Node 18+: `NODE_PATH=fabric/tests/stubs node fabric/tests/test_contract.js; python3 fabric/tests/parity_test.py; python3 fabric/tests/e2e_test.py`

## Steps (Windows laptop)
1. Docker Desktop (WSL2 backend) + Ubuntu in WSL2. In WSL2 install Node 18+, Git, curl, jq.
2. `curl -sSLO https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh && chmod +x install-fabric.sh && ./install-fabric.sh docker binary samples` (clones ~/fabric-samples).
3. Network with CAs and channel: `cd ~/fabric-samples/test-network && ./network.sh up createChannel -ca -c abacfl`
4. Deploy chaincode (copy `fabric/chaincode/abacfl-js` into WSL2; `npm install` inside it):
   `./network.sh deployCC -c abacfl -ccn abacfl -ccp /path/to/abacfl-js -ccl javascript -ccep "OR('Org1MSP.peer','Org2MSP.peer')"`
5. Identities with ABAC attributes: `FABRIC_SAMPLES=~/fabric-samples bash fabric/scripts/enroll_clients.sh` (CLIENTS_PER_TYPE must equal `FedConfig.clients_per_type`, default 14 10 8 6).
6. Gateway: `export GATEWAY_TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))'); bash fabric/scripts/start_gateway.sh` (binds 127.0.0.1:8080).
7. Tunnel in a second terminal: `cloudflared tunnel --url http://127.0.0.1:8080` (or ngrok); copy the https URL.
8. Colab: upload the package, then
   `!python run_phase2.py --gateway https://<tunnel> --token <TOKEN> --csv "/content/ciciot/Merged*.csv" --cache /content/ciciot.npz --seeds 2`
   Smoke test first with `--synthetic --rounds 5`.
9. Expected output: per run F1 ref vs fabric (identical), `events equal=True chain equal=True chain_ok=True`, and a latency table (submit calls ~0.5–3 s on test-network; the paper's overhead figure). Audit trail in `phase2_audit.json`.

## First things to check if something fails
- `ROUND_STALE`/`REPLAY`: re-run uses a new `run_id`; the gateway `/reset` clears state (aggregator only).
- `NO_ATTRIBUTES`: identity enrolled without `--enrollment.attrs` (re-run script).
- Endorsement failures: policy needs peers of both orgs reachable; check `docker ps`.
- Enrolment CA path differs by fabric-samples version: edit `CA_CERT` in `enroll_clients.sh`.

## Security
Token is mandatory; gateway listens on localhost only; tunnel URL + token = aggregator power, so treat both as secrets and stop the tunnel when done. Do not commit `abacfl_identities/`.
Trust assumption (state in paper): the CA assigns attributes honestly and limits attacker concentration per cohort.
