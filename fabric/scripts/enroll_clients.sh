#!/usr/bin/env bash
# Registers + enrols N clients and 1 aggregator on the test-network Org1 CA, with ABAC attributes baked into the
# X.509 certificate (":ecert"). Run from WSL2 AFTER:  ./network.sh up createChannel -ca -c abacfl
# Usage: FABRIC_SAMPLES=~/fabric-samples ./enroll_clients.sh [out_dir]
set -euo pipefail
FS="${FABRIC_SAMPLES:-$HOME/fabric-samples}"
OUT="$(realpath -m "${1:-$HOME/abacfl_identities}")"
CLIENTS_PER_TYPE=(${CLIENTS_PER_TYPE:-14 10 8 6})      # must equal FedConfig.clients_per_type
TYPES=(${DEVICE_TYPES:-camera sensor plug hub})
export PATH="$FS/bin:$PATH"
export FABRIC_CFG_PATH="$FS/config"
CA_URL="https://localhost:7054"
CA_CERT="$FS/test-network/organizations/fabric-ca/org1/ca-cert.pem"
ADMIN_HOME="$FS/test-network/organizations/peerOrganizations/org1.example.com"   # CA admin was enrolled here by network.sh -ca
mkdir -p "$OUT"

# CA admin (bootstrap identity of the test-network CA)
export FABRIC_CA_CLIENT_HOME="$OUT/_cadmin"
fabric-ca-client enroll -u "https://admin:adminpw@localhost:7054" --caname ca-org1 --tls.certfiles "$CA_CERT" >/dev/null

enroll_one () {   # name attrs
  local name="$1" attrs="$2" pw="${1}pw"
  FABRIC_CA_CLIENT_HOME="$OUT/_cadmin" fabric-ca-client register --caname ca-org1 --id.name "$name" --id.secret "$pw" \
      --id.type client --id.attrs "$attrs" --tls.certfiles "$CA_CERT" >/dev/null 2>&1 || true   # already registered -> ok
  FABRIC_CA_CLIENT_HOME="$OUT/$name" fabric-ca-client enroll -u "https://$name:$pw@localhost:7054" --caname ca-org1 \
      -M "$OUT/$name/msp" --enrollment.attrs "device_type,org,location_class,role" --tls.certfiles "$CA_CERT" >/dev/null
}

i=0
for t in "${!CLIENTS_PER_TYPE[@]}"; do
  for ((j=0; j<${CLIENTS_PER_TYPE[$t]}; j++)); do
    enroll_one "client$i" "device_type=${TYPES[$t]}:ecert,org=org$((i%2)):ecert,location_class=loc$((i%3)):ecert,hf.Registrar.Roles=client"
    i=$((i+1))
  done
done
enroll_one "aggregator" "role=aggregator:ecert,hf.Registrar.Roles=client"
echo "enrolled $i clients + aggregator into $OUT   (export CRYPTO_DIR=$OUT)"
