'use strict';
const assert = require('assert');
const logic = require('../chaincode/abacfl-js/lib/logic');
// Known Merkle vectors (also asserted in Python tests): empty -> 32 zero bytes
assert.strictEqual(logic.merkleRoot([]), '0'.repeat(64));
const a = logic.merkleRoot(['x']), b = logic.merkleRoot(['x', 'y']), c = logic.merkleRoot(['x', 'y', 'z']);
assert(a !== b && b !== c && /^[0-9a-f]{64}$/.test(c));
assert.strictEqual(logic.merkleRoot(['x', 'y']), logic.merkleRoot(['x', 'y']));
assert.notStrictEqual(logic.merkleRoot(['x', 'y']), logic.merkleRoot(['y', 'x']));
assert.strictEqual(logic.chainHash('0'.repeat(64), a, 0).length, 64);
// transition basics
const cfg = logic.normalizeConfig({});
let s = logic.newState(cfg);
for (let i = 0; i < 12; i++) { const r = logic.transition(s, 1.0, cfg); s = r.state; }
assert(['REVOKED', 'PROBATION'].includes(s.status));
console.log('logic tests passed');
