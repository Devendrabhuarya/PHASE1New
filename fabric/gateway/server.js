'use strict';
/**
 * REST gateway. The Colab aggregator never holds Fabric credentials; it calls this service.
 * SECURITY: this process holds the private keys of every simulated client. Expose ONLY this port through the
 * tunnel, always set GATEWAY_TOKEN, and never share the tunnel URL. (Real devices would hold their own keys.)
 *
 *   BACKEND=mock   node server.js     # in-memory chaincode, no Fabric (testing)
 *   BACKEND=fabric node server.js     # real network (see PHASE2.md for the environment variables)
 */
const http = require('http');
const crypto = require('crypto');

const PORT = parseInt(process.env.PORT || '8080', 10);
const TOKEN = process.env.GATEWAY_TOKEN;
if (!TOKEN) { console.error('GATEWAY_TOKEN must be set'); process.exit(1); }
const MODE = process.env.BACKEND || 'fabric';
const CONCURRENCY = parseInt(process.env.CONCURRENCY || '16', 10);

let backend;
if (MODE === 'mock') {
  const perType = (process.env.MOCK_CLIENTS_PER_TYPE || '14,10,8,6').split(',').map(Number);
  backend = new (require('./backend_mock'))(perType);
} else {
  backend = new (require('./backend_fabric'))();
}

const cid = (k) => (/^client\d+$/.test(String(k)) ? String(k) : `client${k}`);

async function pool(items, fn) {                       // bounded parallelism
  const out = new Array(items.length); let i = 0;
  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, items.length) }, async () => {
    while (i < items.length) { const j = i++; try { out[j] = { ok: true, value: await fn(items[j]) }; } catch (e) { out[j] = { ok: false, error: String(e.message || e) }; } }
  }));
  return out;
}

const routes = {
  'GET /health': async () => ({ ok: true, mode: MODE }),
  'POST /init': async (b) => ({ result: await backend.submit('aggregator', 'InitConfig', JSON.stringify(b.config)) }),
  'POST /reset': async (b) => JSON.parse(await backend.submit('aggregator', 'ResetRun', b.runId)),
  'POST /register_batch': async (b) => {
    const r = await pool(b.clients.map(cid), (c) => backend.submit(c, 'RegisterClient'));
    return { results: Object.fromEntries(b.clients.map((c, i) => [cid(c), r[i].ok ? 'OK' : r[i].error])) };
  },
  'POST /gate_batch': async (b) => {
    const r = await pool(b.clients.map(cid), async (c) => JSON.parse(await backend.evaluate(c, 'Gate', c)));
    return { results: Object.fromEntries(b.clients.map((c, i) => [cid(c), r[i].ok ? r[i].value : { allow: false, reason: 'ERROR: ' + r[i].error }])) };
  },
  'POST /commit_updates': async (b) => {
    const r = await pool(b.updates, (u) => backend.submit(cid(u.client), 'CommitUpdate', b.round, u.hash, u.n));
    return { results: Object.fromEntries(b.updates.map((u, i) => [cid(u.client), r[i].ok ? 'OK' : r[i].error])) };
  },
  'POST /apply_round': async (b) => JSON.parse(await backend.submit('aggregator', 'ApplyRound', b.round, JSON.stringify(b.scores))),
  'POST /commit_round': async (b) => JSON.parse(await backend.submit('aggregator', 'CommitRound', b.round, JSON.stringify(b.records), b.modelHash || '')),
  'POST /states': async () => JSON.parse(await backend.evaluate('aggregator', 'GetAllStates')),
  'POST /round': async (b) => JSON.parse(await backend.evaluate('aggregator', 'GetRound', b.round)),
  'POST /update': async (b) => JSON.parse(await backend.evaluate('aggregator', 'GetUpdate', b.round, cid(b.client))),
  'POST /verify': async () => JSON.parse(await backend.evaluate('aggregator', 'VerifyChain')),
};

const safeEq = (a, b) => { const x = Buffer.from(a), y = Buffer.from(b); return x.length === y.length && crypto.timingSafeEqual(x, y); };

const server = http.createServer(async (req, res) => {
  const send = (code, obj) => { res.writeHead(code, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(obj)); };
  try {
    const auth = req.headers['authorization'] || '';
    if (!safeEq(auth, `Bearer ${TOKEN}`)) return send(401, { error: 'unauthorized' });
    const route = routes[`${req.method} ${req.url.split('?')[0]}`];
    if (!route) return send(404, { error: 'not found' });
    let body = '';
    for await (const chunk of req) { body += chunk; if (body.length > 50e6) return send(413, { error: 'too large' }); }
    send(200, await route(body ? JSON.parse(body) : {}));
  } catch (e) { send(400, { error: String(e.message || e) }); }
});
server.listen(PORT, '127.0.0.1', () => console.log(`gateway (${MODE}) listening on 127.0.0.1:${PORT}`));
process.on('SIGINT', async () => { await backend.close(); process.exit(0); });
process.on('SIGTERM', async () => { await backend.close(); process.exit(0); });
