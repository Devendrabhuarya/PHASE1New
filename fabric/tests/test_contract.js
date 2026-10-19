'use strict';
// Run: NODE_PATH=tests/stubs node tests/test_contract.js   (works without installing Fabric packages)
const assert = require('assert');
const path = require('path');
const AbacFl = require(path.join(__dirname, '../chaincode/abacfl-js/lib/contract'));
const { MockStub, buildIdentities } = require(path.join(__dirname, '../gateway/mock_stub'));

const cc = new AbacFl();
const stub = new MockStub();
const ids = buildIdentities([3, 2]);               // client0-2 type0, client3-4 type1
const as = (name) => ({ stub, clientIdentity: ids[name] });
const rejects = async (p, msg) => { let e; try { await p; } catch (x) { e = x; } assert(e && e.message.includes(msg), `expected error containing "${msg}", got ${e && e.message}`); };
const hash = (c) => c.repeat(64).slice(0, 64);
let n = 0; const ok = (name) => { n++; console.log('  ok -', name); };

(async () => {
  // --- authorization: only the aggregator role may configure ---
  await rejects(cc.InitConfig(as('client0'), JSON.stringify({})), 'FORBIDDEN'); ok('client cannot InitConfig');
  await rejects(cc.RegisterClient(as('client0')), 'CONFIG_NOT_SET'); ok('register before config fails');
  await cc.InitConfig(as('aggregator'), JSON.stringify({ tau_rep: 0.7, M: 4, N: 6 }));
  await rejects(cc.InitConfig(as('aggregator'), JSON.stringify({ M: 9, N: 3 })), 'CONFIG_INVALID'); ok('invalid config rejected');
  await cc.ResetRun(as('aggregator'), 'test');
  for (let k = 0; k < 5; k++) await cc.RegisterClient(as(`client${k}`));
  await rejects(cc.RegisterClient(as('aggregator')), 'FORBIDDEN'); ok('aggregator cannot register as client');
  const all = JSON.parse(await cc.GetAllStates(as('aggregator')));
  assert.strictEqual(Object.keys(all).length, 5); assert.strictEqual(all.client3.attrs.device_type, 'type1'); ok('5 clients registered with cert attributes');

  // --- gate ---
  let g = JSON.parse(await cc.Gate(as('client0'), 'client0'));
  assert(g.allow && g.cohort === 'type0' && g.cap === 1.0); ok('gate allows active client, cohort from cert');
  await rejects(cc.Gate(as('client0'), 'client1'), 'FORBIDDEN'); ok('client cannot gate others');
  g = JSON.parse(await cc.Gate(as('aggregator'), 'client1')); assert(g.allow); ok('aggregator can gate any client');
  g = JSON.parse(await cc.Gate(as('aggregator'), 'client99')); assert(!g.allow && g.reason === 'NOT_REGISTERED'); ok('unregistered client denied');

  // --- update commitments: replay and stale-round protection ---
  await cc.CommitUpdate(as('client0'), '0', hash('a'), '100');
  await rejects(cc.CommitUpdate(as('client0'), '0', hash('b'), '100'), 'REPLAY'); ok('replayed update rejected');
  await rejects(cc.CommitUpdate(as('client0'), '0', 'xyz', '100'), 'BAD_HASH'); ok('malformed hash rejected');
  for (let k = 1; k < 5; k++) await cc.CommitUpdate(as(`client${k}`), '0', hash('c'), '100');

  // --- round 0: client4 misbehaves (a=1), others fine ---
  const scores = { client0: 0.1, client1: 0.0, client2: 0.2, client3: 0.1, client4: 1.0 };
  await rejects(cc.ApplyRound(as('client0'), '0', JSON.stringify(scores)), 'FORBIDDEN'); ok('client cannot ApplyRound');
  await rejects(cc.ApplyRound(as('aggregator'), '0', JSON.stringify({ client0: 1.5 })), 'BAD_SCORE'); ok('out-of-range score rejected');
  const r0 = JSON.parse(await cc.ApplyRound(as('aggregator'), '0', JSON.stringify(scores)));
  assert(r0.states.client4.R < 0.5 && r0.states.client0.R > 0.5); ok('reputation moves down for anomalous, up for normal');
  await rejects(cc.ApplyRound(as('aggregator'), '0', JSON.stringify(scores)), 'ROUND_STALE'); ok('round cannot be applied twice');
  await rejects(cc.CommitUpdate(as('client1'), '0', hash('d'), '1'), 'ROUND_STALE'); ok('update for closed round rejected');
  const recs = ['{"k":0}', '{"k":1}', '{"k":2}'];
  const c0 = JSON.parse(await cc.CommitRound(as('aggregator'), '0', JSON.stringify(recs), 'abc'));
  await rejects(cc.CommitRound(as('aggregator'), '0', JSON.stringify(recs), 'abc'), 'ROUND_ORDER'); ok('round committed once');

  // --- keep client4 anomalous until it is revoked, then check the gate ---
  let revokedAt = null;
  for (let t = 1; t < 12 && revokedAt === null; t++) {
    const st = JSON.parse(await cc.GetClientState(as('aggregator'), 'client4'));
    const sc = { client0: 0.1, client1: 0.1, client2: 0.1, client3: 0.1 };
    if (st.status !== 'REVOKED') sc.client4 = 1.0;
    const res = JSON.parse(await cc.ApplyRound(as('aggregator'), String(t), JSON.stringify(sc)));
    await cc.CommitRound(as('aggregator'), String(t), JSON.stringify(['{"t":' + t + '}']), '');
    if (res.events.some((e) => e[1] === 'client4' && e[2] === 'REVOKED')) revokedAt = t;
  }
  assert(revokedAt !== null && revokedAt <= 6); ok(`persistent attacker revoked at round ${revokedAt}`);
  g = JSON.parse(await cc.Gate(as('client4'), 'client4')); assert(!g.allow && g.reason === 'REVOKED'); ok('revoked client is denied at the gate');
  await rejects(cc.CommitUpdate(as('client4'), String(revokedAt + 1), hash('e'), '10'), 'DENIED'); ok('revoked client cannot commit updates');
  g = JSON.parse(await cc.Gate(as('client0'), 'client0')); assert(g.allow); ok('honest client still allowed');

  // --- audit log integrity ---
  let v = JSON.parse(await cc.VerifyChain(as('client0'))); assert(v.ok && v.rounds === revokedAt + 1); ok('hash chain verifies');
  const rd = JSON.parse(await cc.GetRound(as('client0'), '0')); assert.strictEqual(rd.hash, c0.hash); ok('round record retrievable');
  const key = 'round:' + String(0).padStart(8, '0'); const tampered = JSON.parse(stub.m.get(key).toString()); tampered.root = 'f'.repeat(64);
  stub.m.set(key, Buffer.from(JSON.stringify(tampered)));
  v = JSON.parse(await cc.VerifyChain(as('client0'))); assert(!v.ok && v.brokenAtRound === 0); ok('tampered round log is detected');

  // --- reset ---
  await cc.ResetRun(as('aggregator'), 'run2'); ok('ResetRun works');
  console.log(`\n${n} checks passed`);
})().catch((e) => { console.error('FAIL', e); process.exit(1); });
