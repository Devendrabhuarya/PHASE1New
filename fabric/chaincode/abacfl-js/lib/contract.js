'use strict';
const logic = require('./logic');

// fabric-contract-api is always present when deployed to Fabric. The fallback only exists so the
// unit tests and the gateway's mock mode can run on a machine without Fabric installed.
let Contract;
try { ({ Contract } = require('fabric-contract-api')); } catch (e) { Contract = class { constructor(name) { this.__name = name; } }; }

const pad = (n) => String(n).padStart(8, '0');
const ZERO = '0'.repeat(64);

async function rangeAll(stub, start, end) {
  const out = [];
  const it = await stub.getStateByRange(start, end);
  for (;;) {
    const r = await it.next();
    if (r.value && r.value.value && r.value.value.length > 0) out.push({ key: r.value.key, value: r.value.value.toString('utf8') });
    if (r.done) { await it.close(); break; }
  }
  return out;
}

class AbacFl extends Contract {
  constructor() { super('AbacFl'); }

  // ---------- helpers ----------
  _caller(ctx) {
    const ci = ctx.clientIdentity;
    let id = ci.getAttributeValue('hf.EnrollmentID');
    if (!id) { const m = /CN=([^:/,]+)/.exec(ci.getID()); id = m ? m[1] : ci.getID(); }
    return {
      id,
      attrs: {
        device_type: ci.getAttributeValue('device_type'),
        org: ci.getAttributeValue('org'),
        location_class: ci.getAttributeValue('location_class'),
        role: ci.getAttributeValue('role'),
      },
    };
  }
  _requireAggregator(ctx) {
    const c = this._caller(ctx);
    if (c.attrs.role !== 'aggregator') throw new Error('FORBIDDEN: aggregator role required');
    return c;
  }
  async _get(ctx, key) {
    const b = await ctx.stub.getState(key);
    return (b && b.length > 0) ? JSON.parse(b.toString('utf8')) : null;
  }
  async _put(ctx, key, obj) { await ctx.stub.putState(key, Buffer.from(JSON.stringify(obj))); }
  async _cfg(ctx) {
    const c = await this._get(ctx, 'config');
    if (!c) throw new Error('CONFIG_NOT_SET');
    return c;
  }
  async _meta(ctx) {
    return (await this._get(ctx, 'meta')) || { runId: '', lastRound: -1, lastApplied: -1, lastHash: ZERO };
  }

  // ---------- admin (aggregator only) ----------
  async InitConfig(ctx, cfgJson) {
    this._requireAggregator(ctx);
    const cfg = logic.normalizeConfig(JSON.parse(cfgJson));
    await this._put(ctx, 'config', cfg);
    return 'OK';
  }

  async ResetRun(ctx, runId) {
    this._requireAggregator(ctx);
    const cfg = await this._cfg(ctx);
    let n = 0;
    for (const { key, value } of await rangeAll(ctx.stub, 'client:', 'client;')) {
      const st = JSON.parse(value);
      await this._put(ctx, key, Object.assign({}, st, logic.newState(cfg)));
      n += 1;
    }
    for (const [a, b] of [['upd:', 'upd;'], ['round:', 'round;'], ['ev:', 'ev;']]) {
      for (const { key } of await rangeAll(ctx.stub, a, b)) await ctx.stub.deleteState(key);
    }
    await this._put(ctx, 'meta', { runId: String(runId), lastRound: -1, lastApplied: -1, lastHash: ZERO });
    return JSON.stringify({ reset: n });
  }

  // ---------- client side ----------
  async RegisterClient(ctx) {
    const cfg = await this._cfg(ctx);
    const c = this._caller(ctx);
    if (c.attrs.role === 'aggregator') throw new Error('FORBIDDEN: aggregator cannot register as client');
    if (!c.attrs.device_type) throw new Error('NO_ATTRIBUTES: certificate has no device_type');
    const key = `client:${c.id}`;
    const existing = await this._get(ctx, key);
    if (existing) return JSON.stringify(existing);
    const st = Object.assign({ id: c.id, attrs: { device_type: c.attrs.device_type, org: c.attrs.org, location_class: c.attrs.location_class } }, logic.newState(cfg));
    await this._put(ctx, key, st);
    return JSON.stringify(st);
  }

  /** Step 1 of the design: pre-training ABAC gate. Read-only. */
  async Gate(ctx, clientId) {
    const cfg = await this._cfg(ctx);
    const c = this._caller(ctx);
    if (c.attrs.role !== 'aggregator' && clientId !== c.id) throw new Error('FORBIDDEN: can only gate self');
    const st = await this._get(ctx, `client:${clientId}`);
    const attrs = (c.id === clientId) ? c.attrs : (st ? st.attrs : {});
    return JSON.stringify(logic.gateDecision(st, attrs, cfg));
  }

  /** Step 3: bind an update to the caller's identity. The Fabric proposal signature authenticates the caller. */
  async CommitUpdate(ctx, round, hashHex, nk) {
    const cfg = await this._cfg(ctx);
    const meta = await this._meta(ctx);
    const c = this._caller(ctx);
    const r = parseInt(round, 10), n = parseInt(nk, 10);
    if (!Number.isInteger(r) || r < 0) throw new Error('BAD_ROUND');
    if (r <= meta.lastRound || r <= meta.lastApplied) throw new Error('ROUND_STALE');
    if (!/^[0-9a-f]{64}$/.test(hashHex)) throw new Error('BAD_HASH');
    if (!Number.isInteger(n) || n < 1) throw new Error('BAD_N');
    const st = await this._get(ctx, `client:${c.id}`);
    const g = logic.gateDecision(st, c.attrs, cfg);
    if (!g.allow) throw new Error(`DENIED: ${g.reason}`);
    const key = `upd:${pad(r)}:${c.id}`;
    if (await this._get(ctx, key)) throw new Error('REPLAY');
    await this._put(ctx, key, { hash: hashHex, n });
    return 'OK';
  }

  // ---------- aggregator side ----------
  /** Reputation / revocation transitions. Scores are computed off-chain; the transition rule is on-chain. */
  async ApplyRound(ctx, round, scoresJson) {
    this._requireAggregator(ctx);
    const cfg = await this._cfg(ctx);
    const meta = await this._meta(ctx);
    const r = parseInt(round, 10);
    if (!Number.isInteger(r) || r < 0) throw new Error('BAD_ROUND');
    if (r <= meta.lastApplied || r <= meta.lastRound) throw new Error('ROUND_STALE');
    const scores = JSON.parse(scoresJson);
    const states = {}, events = [];
    for (const { key, value } of await rangeAll(ctx.stub, 'client:', 'client;')) {
      const st = JSON.parse(value);
      let a = null;
      if (Object.prototype.hasOwnProperty.call(scores, st.id) && scores[st.id] !== null) {
        a = scores[st.id];
        if (typeof a !== 'number' || !Number.isFinite(a) || a < 0 || a > 1) throw new Error(`BAD_SCORE: ${st.id}`);
      }
      const { state, event } = logic.transition(st, a, cfg);
      const next = Object.assign({}, st, state);
      await this._put(ctx, key, next);
      states[st.id] = state;
      if (event) events.push([r, st.id, event]);
    }
    await this._put(ctx, `ev:${pad(r)}`, events);
    meta.lastApplied = r;
    await this._put(ctx, 'meta', meta);
    return JSON.stringify({ states, events });
  }

  /** One batched commit per round: Merkle root of the round's records + hash-chain link. */
  async CommitRound(ctx, round, recordsJson, modelHash) {
    this._requireAggregator(ctx);
    const meta = await this._meta(ctx);
    const r = parseInt(round, 10);
    if (r !== meta.lastApplied || r <= meta.lastRound) throw new Error('ROUND_ORDER: apply the round first, once');
    const records = JSON.parse(recordsJson);
    if (!Array.isArray(records) || !records.every((x) => typeof x === 'string')) throw new Error('BAD_RECORDS');
    const root = logic.merkleRoot(records);
    const hash = logic.chainHash(meta.lastHash, root, r);
    await this._put(ctx, `round:${pad(r)}`, { round: r, root, hash, prev: meta.lastHash, n: records.length, modelHash: modelHash || '' });
    meta.lastRound = r; meta.lastHash = hash;
    await this._put(ctx, 'meta', meta);
    return JSON.stringify({ hash, root });
  }

  // ---------- reads ----------
  async GetConfig(ctx) { return JSON.stringify(await this._cfg(ctx)); }
  async GetClientState(ctx, clientId) { return JSON.stringify(await this._get(ctx, `client:${clientId}`)); }
  async GetAllStates(ctx) {
    const out = {};
    for (const { value } of await rangeAll(ctx.stub, 'client:', 'client;')) { const s = JSON.parse(value); out[s.id] = s; }
    return JSON.stringify(out);
  }
  async GetRound(ctx, t) { return JSON.stringify(await this._get(ctx, `round:${pad(parseInt(t, 10))}`)); }
  async GetUpdate(ctx, round, clientId) { return JSON.stringify(await this._get(ctx, `upd:${pad(parseInt(round, 10))}:${clientId}`)); }

  /** Re-derive every hash-chain link from the stored roots. Detects any inconsistency in the round log. */
  async VerifyChain(ctx) {
    let prev = ZERO, n = 0;
    for (const { value } of await rangeAll(ctx.stub, 'round:', 'round;')) {
      const rd = JSON.parse(value);
      if (rd.prev !== prev || rd.hash !== logic.chainHash(prev, rd.root, rd.round)) {
        return JSON.stringify({ ok: false, brokenAtRound: rd.round });
      }
      prev = rd.hash; n += 1;
    }
    return JSON.stringify({ ok: true, rounds: n, head: prev });
  }
}

module.exports = AbacFl;
