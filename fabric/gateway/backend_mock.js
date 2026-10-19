'use strict';
/** Runs the REAL chaincode class against an in-memory stub. No Fabric needed. For testing the whole pipeline. */
const path = require('path');
const AbacFl = require(path.join(__dirname, '../chaincode/abacfl-js/lib/contract'));
const { MockStub, buildIdentities } = require('./mock_stub');

class MockBackend {
  constructor(perType) {
    this.cc = new AbacFl();
    this.stub = new MockStub();
    this.ids = buildIdentities(perType);
  }
  _ctx(name) {
    const clientIdentity = this.ids[name];
    if (!clientIdentity) throw new Error(`UNKNOWN_IDENTITY: ${name}`);
    return { stub: this.stub, clientIdentity };
  }
  async evaluate(name, fn, ...args) { return this.cc[fn](this._ctx(name), ...args.map(String)); }
  async submit(name, fn, ...args) { return this.cc[fn](this._ctx(name), ...args.map(String)); }
  async close() {}
}
module.exports = MockBackend;
