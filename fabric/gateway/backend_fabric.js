'use strict';
/**
 * Real Hyperledger Fabric backend (Fabric Gateway client API).
 * One gateway connection per identity (client0..N, aggregator); all share one gRPC channel to the peer.
 * Identities are the MSP folders created by scripts/enroll_clients.sh: <CRYPTO_DIR>/<name>/msp/{signcerts,keystore}
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

class FabricBackend {
  constructor(env = process.env) {
    this.grpc = require('@grpc/grpc-js');
    this.fg = require('@hyperledger/fabric-gateway');
    this.mspId = env.MSP_ID || 'Org1MSP';
    this.channel = env.CHANNEL || 'abacfl';
    this.chaincode = env.CHAINCODE || 'abacfl';
    this.contractName = env.CONTRACT || 'AbacFl';
    this.cryptoDir = env.CRYPTO_DIR;
    if (!this.cryptoDir) throw new Error('CRYPTO_DIR is required (folder produced by enroll_clients.sh)');
    const tls = fs.readFileSync(env.TLS_CERT_PATH);
    this.client = new this.grpc.Client(env.PEER_ENDPOINT || 'localhost:7051',
      this.grpc.credentials.createSsl(tls), { 'grpc.ssl_target_name_override': env.PEER_HOST_ALIAS || 'peer0.org1.example.com' });
    this.cache = new Map();
  }
  _contract(name) {
    if (this.cache.has(name)) return this.cache.get(name).contract;
    const msp = path.join(this.cryptoDir, name, 'msp');
    const certPem = fs.readFileSync(path.join(msp, 'signcerts', fs.readdirSync(path.join(msp, 'signcerts'))[0]));
    const keyFile = fs.readdirSync(path.join(msp, 'keystore'))[0];
    const key = crypto.createPrivateKey(fs.readFileSync(path.join(msp, 'keystore', keyFile)));
    const gateway = this.fg.connect({
      client: this.client,
      identity: { mspId: this.mspId, credentials: certPem },
      signer: this.fg.signers.newPrivateKeySigner(key),
      evaluateOptions: () => ({ deadline: Date.now() + 10000 }),
      endorseOptions: () => ({ deadline: Date.now() + 30000 }),
      submitOptions: () => ({ deadline: Date.now() + 30000 }),
      commitStatusOptions: () => ({ deadline: Date.now() + 120000 }),
    });
    const contract = gateway.getNetwork(this.channel).getContract(this.chaincode, this.contractName);
    this.cache.set(name, { gateway, contract });
    return contract;
  }
  async evaluate(name, fn, ...args) {
    return Buffer.from(await this._contract(name).evaluateTransaction(fn, ...args.map(String))).toString('utf8');
  }
  async submit(name, fn, ...args) {
    return Buffer.from(await this._contract(name).submitTransaction(fn, ...args.map(String))).toString('utf8');
  }
  async close() { for (const { gateway } of this.cache.values()) gateway.close(); this.client.close(); }
}
module.exports = FabricBackend;
