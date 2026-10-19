'use strict';
// stdin: {cfg, cases:[{state, scores:[a|null,...]}]}  stdout: trajectories computed by the chaincode logic
const logic = require('../chaincode/abacfl-js/lib/logic');
let buf = '';
process.stdin.on('data', (d) => (buf += d));
process.stdin.on('end', () => {
  const { cfg, cases } = JSON.parse(buf);
  const c = logic.normalizeConfig(cfg);
  const out = cases.map(({ state, scores }) => {
    let s = state; const traj = [];
    scores.forEach((a, i) => { const r = logic.transition(s, a, c); s = r.state; traj.push({ state: s, event: r.event }); });
    return traj;
  });
  process.stdout.write(JSON.stringify(out));
});
