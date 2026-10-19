'use strict';
/** In-memory stand-in for the Fabric stub + client identity. Used by unit tests and the gateway's mock mode. */
class MockStub {
  constructor() { this.m = new Map(); }
  async getState(k) { return this.m.has(k) ? Buffer.from(this.m.get(k)) : Buffer.alloc(0); }
  async putState(k, v) { this.m.set(k, Buffer.from(v)); }
  async deleteState(k) { this.m.delete(k); }
  async getStateByRange(start, end) {
    const keys = [...this.m.keys()].filter((k) => k >= start && k < end).sort();
    let i = 0; const m = this.m;
    return {
      async next() {
        if (i < keys.length) { const key = keys[i++]; return { value: { key, value: Buffer.from(m.get(key)) }, done: false }; }
        return { value: undefined, done: true };
      },
      async close() {},
    };
  }
}

class MockIdentity {
  constructor(name, attrs) { this.name = name; this.attrs = Object.assign({ 'hf.EnrollmentID': name }, attrs); }
  getAttributeValue(k) { return this.attrs[k] === undefined ? null : this.attrs[k]; }
  getID() { return `x509::CN=${this.name}::CN=ca.org1.example.com`; }
  getMSPID() { return 'Org1MSP'; }
}

/** Identities mirror scripts/enroll_clients.sh: clients are numbered, device types assigned by cumulative counts. */
function buildIdentities(perType) {
  const ids = {};
  let cid = 0;
  perType.forEach((count, t) => {
    for (let i = 0; i < count; i++, cid++) {
      ids[`client${cid}`] = new MockIdentity(`client${cid}`, { device_type: `type${t}`, org: `org${cid % 2}`, location_class: `loc${cid % 3}` });
    }
  });
  ids.aggregator = new MockIdentity('aggregator', { role: 'aggregator' });
  return ids;
}

module.exports = { MockStub, MockIdentity, buildIdentities };
