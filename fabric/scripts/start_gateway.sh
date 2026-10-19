#!/usr/bin/env bash
# Starts the REST gateway against the real Fabric test-network. Run in WSL2 from the repo root.
set -euo pipefail
FS="${FABRIC_SAMPLES:-$HOME/fabric-samples}"
: "${GATEWAY_TOKEN:?set GATEWAY_TOKEN to a long random string (python3 -c 'import secrets;print(secrets.token_urlsafe(32))')}"
export BACKEND=fabric PORT="${PORT:-8080}"
export CRYPTO_DIR="${CRYPTO_DIR:-$HOME/abacfl_identities}"
export TLS_CERT_PATH="$FS/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
export PEER_ENDPOINT=localhost:7051 PEER_HOST_ALIAS=peer0.org1.example.com
export MSP_ID=Org1MSP CHANNEL=abacfl CHAINCODE=abacfl CONTRACT=AbacFl
cd "$(dirname "$0")/../gateway"
[ -d node_modules ] || npm install
exec node server.js
