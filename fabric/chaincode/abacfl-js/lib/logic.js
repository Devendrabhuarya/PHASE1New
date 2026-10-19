'use strict';
/**
 * Pure, deterministic logic (no Fabric API, no clock, no randomness).
 * Mirrors abacfl/ledger.py (Python) exactly so on-chain and off-chain results can be compared.
 */
const crypto = require('crypto');

const DEFAULT_CFG = {
  R0: 0.5, R_min: 0.2, R_th: 0.3, tau_a: 0.5, tau_rep: null,
  eta_up: 0.05, eta_down: 0.3, N: 5, M: 3, Tc: 5, P: 3, probation_cap: 0.5,
  allowed_device_types: null,   // static ABAC policy P(A_static); null = any
  allowed_orgs: null,
};

function normalizeConfig(c) {
  const cfg = Object.assign({}, DEFAULT_CFG, c || {});
  const num = ['R0', 'R_min', 'R_th', 'tau_a', 'eta_up', 'eta_down', 'probation_cap'];
  for (const k of num) {
    if (typeof cfg[k] !== 'number' || !Number.isFinite(cfg[k])) throw new Error(`CONFIG_INVALID: ${k}`);
  }
  if (cfg.tau_rep !== null && (typeof cfg.tau_rep !== 'number' || !Number.isFinite(cfg.tau_rep))) throw new Error('CONFIG_INVALID: tau_rep');
  for (const k of ['N', 'M', 'Tc', 'P']) {
    if (!Number.isInteger(cfg[k]) || cfg[k] < 1) throw new Error(`CONFIG_INVALID: ${k}`);
  }
  if (cfg.M > cfg.N) throw new Error('CONFIG_INVALID: M must be <= N');
  for (const k of ['allowed_device_types', 'allowed_orgs']) {
    if (cfg[k] !== null && !Array.isArray(cfg[k])) throw new Error(`CONFIG_INVALID: ${k}`);
  }
  return cfg;
}

function newState(cfg) {
  return { R: cfg.R0, status: 'ACTIVE', window: [], cooldown: 0, probation_left: 0 };
}

/** a: anomaly score in [0,1], or null if the client did not participate. Returns {state, event}. */
function transition(s0, a, cfg) {
  const s = Object.assign({}, s0, { window: s0.window.slice() });
  if (s.status === 'REVOKED') {
    s.cooldown -= 1;
    if (s.cooldown <= 0) {
      s.status = 'PROBATION'; s.R = cfg.R0; s.window = []; s.probation_left = cfg.P;
      return { state: s, event: 'PROBATION' };
    }
    return { state: s, event: null };
  }
  if (a !== null) {
    const thr = (cfg.tau_rep === null || cfg.tau_rep === undefined) ? cfg.tau_a : cfg.tau_rep;
    if (a < thr) s.R = Math.min(1.0, s.R + cfg.eta_up * (1 - a));
    else s.R = Math.max(0.0, s.R - cfg.eta_down * a);
  }
  s.window = s.window.concat([s.R < cfg.R_th ? 1 : 0]).slice(-cfg.N);
  const low = s.window.reduce((x, y) => x + y, 0);
  if (low >= cfg.M) {
    s.status = 'REVOKED'; s.cooldown = cfg.Tc; s.window = [];
    return { state: s, event: 'REVOKED' };
  }
  if (s.status === 'PROBATION' && a !== null) {
    s.probation_left -= 1;
    if (s.probation_left <= 0) s.status = 'ACTIVE';
  }
  return { state: s, event: null };
}

/** The ABAC gate predicate. attrs = static attributes from the CA-signed certificate. */
function gateDecision(state, attrs, cfg) {
  if (!state) return { allow: false, reason: 'NOT_REGISTERED' };
  if (cfg.allowed_device_types && !cfg.allowed_device_types.includes(attrs.device_type)) {
    return { allow: false, reason: 'ATTR_POLICY_DEVICE_TYPE' };
  }
  if (cfg.allowed_orgs && !cfg.allowed_orgs.includes(attrs.org)) {
    return { allow: false, reason: 'ATTR_POLICY_ORG' };
  }
  if (state.status === 'REVOKED') return { allow: false, reason: 'REVOKED' };
  if (state.status !== 'ACTIVE' && state.status !== 'PROBATION') return { allow: false, reason: 'STATUS' };
  if (state.R < cfg.R_min - 1e-9) return { allow: false, reason: 'LOW_REPUTATION' };
  return { allow: true, reason: 'OK', cohort: attrs.device_type,
           cap: state.status === 'PROBATION' ? cfg.probation_cap : 1.0, R: state.R, status: state.status };
}

const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest();

/** items are strings (already-canonical record JSON). Same algorithm as abacfl.ledger.merkle_root_str. */
function merkleRoot(items) {
  let layer = items.map((x) => sha256(Buffer.from(x, 'utf8')));
  if (layer.length === 0) layer = [Buffer.alloc(32)];
  while (layer.length > 1) {
    if (layer.length % 2) layer.push(layer[layer.length - 1]);
    const next = [];
    for (let i = 0; i < layer.length; i += 2) next.push(sha256(Buffer.concat([layer[i], layer[i + 1]])));
    layer = next;
  }
  return layer[0].toString('hex');
}

function chainHash(prev, root, t) {
  return sha256(Buffer.from(prev + root + String(t), 'utf8')).toString('hex');
}

module.exports = { DEFAULT_CFG, normalizeConfig, newState, transition, gateDecision, merkleRoot, chainHash };
