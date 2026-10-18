// Figures for the page, drawn from site/data.
import { svg, h, fmt } from './util.js';

function frame(container, W, H, label) {
  const s = svg('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': label });
  container.replaceChildren(s);
  return s;
}
// Tick values: a round step that gives about 4-5 ticks up to the axis maximum.
function ticks(max) {
  const step = niceMax(max / 5);
  const out = [];
  for (let v = 0; v <= max + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
function niceMax(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

// Share of token-layers whose expert set changed, per layer (census).
export function perLayerBars(container, st, { W = 320, H = 170, label } = {}) {
  const rates = st.per_layer_rates;
  const s = frame(container, W, H, label || 'Share of token-layers with a different expert set, by layer');
  const m = { l: 34, r: 6, t: 10, b: 24 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const ymax = niceMax(Math.max(...rates));
  const y = (v) => m.t + ih - (v / ymax) * ih;
  for (const v of ticks(ymax)) {
    svg('line', { class: 'gridline', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, s);
    svg('text', { class: 'ax', x: m.l - 4, y: y(v) + 3, 'text-anchor': 'end' }, s).textContent = String(+(v * 100).toFixed(2)) + '%';
  }
  const bw = iw / rates.length;
  rates.forEach((r, i) => {
    svg('rect', { class: 'bar', x: m.l + i * bw + 1, y: y(r), width: bw - 2, height: Math.max(1, y(0) - y(r)), rx: 2 }, s);
    if (i % 3 === 0 || i === rates.length - 1) svg('text', { class: 'ax', x: m.l + i * bw + bw / 2, y: H - m.b + 12, 'text-anchor': 'middle' }, s).textContent = i;
  });
  svg('line', { class: 'baseline', x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }, s);
  svg('text', { class: 'ax', x: m.l + iw / 2, y: H - 2, 'text-anchor': 'middle' }, s).textContent = 'layer';
  return s;
}

// One token's 64 router probabilities, top-8 lit, 8th and 9th called out.
export function routerProbs(container, token, { W = 640, H = 230 } = {}) {
  const p = token.probabilities;
  const s = frame(container, W, H, `Router probabilities for layer ${token.layer} token ${token.index}; the 8th and 9th are nearly equal`);
  const m = { l: 42, r: 10, t: 34, b: 28 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const ymax = niceMax(Math.max(...p));
  const y = (v) => m.t + ih - (v / ymax) * ih;
  const order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map(([, i]) => i);
  const e8 = order[7], e9 = order[8];
  for (const v of ticks(ymax)) {
    svg('line', { class: 'gridline', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, s);
    svg('text', { class: 'ax', x: m.l - 5, y: y(v) + 3, 'text-anchor': 'end' }, s).textContent = String(+v.toFixed(4));
  }
  const bw = iw / 64;
  p.forEach((v, i) => {
    const on = token.top8.includes(i);
    svg('rect', {
      class: 'bar', x: m.l + i * bw + 1, y: y(v), width: bw - 2, height: Math.max(1, y(0) - y(v)), rx: 1.5,
      style: `fill: ${i === e9 ? 'var(--orange)' : on ? 'var(--accent)' : 'var(--axis)'}`,
    }, s);
  });
  for (const [e, lab] of [[e8, `#8 · expert ${e8} (chosen)`], [e9, `#9 · expert ${e9} (not chosen)`]]) {
    const x = m.l + e * bw + bw / 2;
    svg('line', { class: 'marker', x1: x, x2: x, y1: y(p[e]) - 4, y2: m.t - (e === e8 ? 18 : 4) }, s);
    svg('text', { class: 'marker-label', x: x + (e === e8 ? -4 : 4), y: m.t - (e === e8 ? 22 : 8), 'text-anchor': e === e8 ? 'end' : 'start' }, s).textContent = lab;
  }
  svg('line', { class: 'baseline', x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }, s);
  [0, 16, 32, 48, 63].forEach((i) => svg('text', { class: 'ax', x: m.l + i * bw + bw / 2, y: H - m.b + 13, 'text-anchor': 'middle' }, s).textContent = i);
  svg('text', { class: 'ax', x: m.l + iw / 2, y: H - 3, 'text-anchor': 'middle' }, s).textContent = 'expert id';
  return { e8, e9 };
}

// Route flips vs accumulator mantissa bits for one accumulator (both layers), with the selected width marked.
export function accumulatorChart(container, rows, acc, bits, { W = 320, H = 180, cursor = null } = {}) {
  const rs = rows.filter((r) => r.accumulator === acc && /^e8m\d+$/.test(r.format)).sort((a, b) => a.mantissa_bits - b.mantissa_bits);
  const s = frame(container, W, H, `Route flips vs mantissa bits of the ${acc} accumulator, layers 0 and 7`);
  const m = { l: 34, r: 8, t: 14, b: 26 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const b0 = 7, b1 = 23;
  const x = (b) => m.l + ((b - b0) / (b1 - b0)) * iw;
  const ymax = niceMax(Math.max(1, ...rs.flatMap((r) => [r.flips_layer0, r.flips_layer7])));
  const y = (v) => m.t + ih - (v / ymax) * ih;
  // Band: widths where calibrated routediff flagged the fault on both layers.
  const flagged = rs.filter((r) => r.calibrated_flagged_L0 && r.calibrated_flagged_L7).map((r) => r.mantissa_bits);
  if (flagged.length) {
    const hi = Math.max(...flagged);
    svg('rect', { class: 'flagband', x: x(b0) - 4, y: m.t, width: x(hi) - x(b0) + 8, height: ih }, s);
    svg('text', { class: 'marker-label', x: x(hi) + 2, y: m.t + 9, 'text-anchor': 'end' }, s).textContent = 'shaded: routediff flags';
  }
  for (const v of ticks(ymax)) {
    svg('line', { class: 'gridline', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, s);
    svg('text', { class: 'ax', x: m.l - 4, y: y(v) + 3, 'text-anchor': 'end' }, s).textContent = Math.round(v);
  }
  for (const [b, lab] of [[7, 'bf16'], [23, 'fp32']]) {
    svg('text', { class: 'ax', x: x(b), y: H - m.b + 12, 'text-anchor': b === b1 ? 'end' : b === b0 ? 'start' : 'middle' }, s).textContent = `${b} ${lab}`;
  }
  [11, 15, 18].forEach((b) => svg('text', { class: 'ax', x: x(b), y: H - m.b + 12, 'text-anchor': 'middle' }, s).textContent = b);
  svg('text', { class: 'ax', x: m.l + iw / 2, y: H - 2, 'text-anchor': 'middle' }, s).textContent = 'mantissa bits in the accumulator';
  svg('line', { class: 'baseline', x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }, s);
  for (const [key, cls, name] of [['flips_layer0', 'line', 'layer 0'], ['flips_layer7', 'line l7', 'layer 7']]) {
    svg('path', { class: cls, d: rs.map((r, i) => `${i ? 'L' : 'M'}${x(r.mantissa_bits)},${y(r[key])}`).join('') }, s);
    rs.forEach((r) => {
      svg('circle', { class: 'dot' + (cls.includes('l7') ? ' l7' : ''), cx: x(r.mantissa_bits), cy: y(r[key]), r: 3 }, s);
    });
  }
  if (bits !== undefined && bits !== null) {
    svg('line', { class: 'marker', x1: x(bits), x2: x(bits), y1: m.t + 28, y2: y(0) }, s);
    svg('text', { class: 'marker-label', x: x(bits) + 4, y: m.t + 40 }, s).textContent = `no flips from ${bits} bits`;
  }
  if (cursor !== null) {
    const r = rs.find((q) => q.mantissa_bits === cursor);
    svg('line', { class: 'cursor', x1: x(cursor), x2: x(cursor), y1: m.t, y2: y(0) }, s);
    if (r) for (const [key, cls] of [['flips_layer0', 'dot'], ['flips_layer7', 'dot l7']]) {
      svg('circle', { class: cls + ' sel', cx: x(cursor), cy: y(r[key]), r: 5.5 }, s);
    }
  }
  // Direct labels for the two series (identity is not colour alone).
  const last = rs[0];
  if (last) {
    const r9 = rs.find((r) => r.mantissa_bits === 9) || last;
    svg('text', { class: 'marker-label', x: x(r9.mantissa_bits) + 6, y: y(r9.flips_layer7) - 6 }, s).textContent = 'layer 7 (dashed)';
    svg('text', { class: 'marker-label', x: x(r9.mantissa_bits) - 4, y: y(r9.flips_layer0) + 16, 'text-anchor': 'end' }, s).textContent = 'layer 0';
  }
  return rs;
}

// One blinded routediff case (Figure 3): the selection-stage ops for one token, then the patch result.
const SHORT = {
  o_proj_out: 'Attention projection', resid_sum: 'Residual add', router_input: 'RMSNorm',
  router_logits: 'Router scores', router_probs: 'Softmax', topk_ids: 'Top 8',
};
export function routediffCase(container, sc) {
  const ops = sc.before.token.ops;
  const first = sc.before.batch.first_flagged_op;
  const list = h('ol', { class: 'rd-ops' });
  for (const id of Object.keys(SHORT)) {
    const o = ops[id];
    let over, val;
    if (id === 'topk_ids') { over = !sc.before.token.route_sets_equal; val = over ? 'expert sets differ' : 'same expert set'; }
    else { over = o.max_scaled_ulp > o.tolerance_scaled_ulp; val = `${fmt(o.max_scaled_ulp)} / ${fmt(o.tolerance_scaled_ulp)}`; }
    list.append(h('li', { class: over ? 'over' : 'ok' },
      h('span', { class: 'rd-mark', 'aria-hidden': 'true' }, over ? '✕' : '✓'),
      h('span', { class: 'rd-name' }, SHORT[id]),
      h('span', { class: 'rd-val' }, val, h('span', { class: 'visually-hidden' }, id === 'topk_ids' ? '' : over ? ' (over tolerance)' : ' (within tolerance)')),
      id === first ? h('span', { class: 'rd-first' }, 'first over tolerance') : null));
  }
  const ref = sc.before.routes.reference, cand = sc.before.routes.candidate;
  const lost = ref.filter((x) => !cand.includes(x)), gained = cand.filter((x) => !ref.includes(x));
  const pc = sc.after?.batch?.patch_check;
  container.replaceChildren(
    h('p', { class: 'rd-legend' }, 'For one token: difference from PyTorch / allowed difference, in scaled ULP (roughly, units of the last bit)'),
    list,
    h('p', { class: 'rd-line' }, `Result: the port chose expert ${gained.join(', ')} instead of ${lost.join(', ')}.`),
    pc ? h('p', { class: 'rd-line good' }, `✓ After patching ${SHORT[first].toLowerCase()} from the reference: ${fmt(pc.route_mismatch_after)} of ${fmt(sc.before.batch.tokens)} tokens with different experts (routes restored).`) : null,
  );
}

// "Which expert wins a tie?": a measured exact tie and the expert each platform kept.
export function tieChooser(container, t) {
  const names = { cpp: 'C++ port', apple_gpu: 'Apple GPU', pytorch_macos: 'PyTorch, Mac', pytorch_linux: 'PyTorch, Linux' };
  const platforms = Object.keys(names).filter((k) => t.choices[k]);
  const [a, b] = t.tied_experts;
  const buttons = h('div', { class: 'tie-buttons', role: 'group', 'aria-label': 'Platform' });
  const out = h('div', { class: 'tie-out', 'aria-live': 'polite' });
  const show = (k) => {
    for (const btn of buttons.children) btn.setAttribute('aria-pressed', String(btn.dataset.k === k));
    const kept = t.choices[k];
    out.replaceChildren(
      h('div', { class: 'tie-pair' }, ...[a, b].map((e) => {
        const won = kept.includes(e);
        return h('div', { class: 'tie-expert' + (won ? ' won' : '') },
          h('span', { class: 'tie-name' }, `Expert ${e}`),
          h('span', { class: 'tie-prob', title: `Full precision: ${t.probabilities[[a, b].indexOf(e)]}` }, `p = ${t.probabilities[[a, b].indexOf(e)].toFixed(8)} (PyTorch CPU)`),
          h('span', { class: 'tie-verdict' }, won ? '✓ runs' : '✕ dropped'));
      })),
      h('p', { class: 'tie-set' }, `${names[k]} runs experts `, ...t.route_sets[k].flatMap((e, i) => [i ? ', ' : '', [a, b].includes(e) ? h('strong', {}, String(e)) : String(e)]), '.'),
    );
  };
  for (const k of platforms) {
    const btn = h('button', { type: 'button', 'data-k': k, 'aria-pressed': 'false' }, names[k]);
    btn.addEventListener('click', () => show(k));
    buttons.append(btn);
  }
  container.replaceChildren(
    h('p', { class: 'tie-lede' }, `A real token (layer ${t.layer}, token ${t.index}, bf16): experts ${a} and ${b} tie exactly for 8th place, so only one can make the top 8. Pick a platform:`),
    buttons, out,
    h('p', { class: 'tie-note', title: t.caveat }, 'Each choice was measured on that platform. Ties have no official winner, and the Apple GPU computes slightly different probabilities than the CPU, so this is a comparison of platforms, not a bug.'),
  );
  show(platforms[0]);
}
