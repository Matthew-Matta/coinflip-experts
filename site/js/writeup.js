// Draws the figures from site/data. The text never depends on them; on a failed load the fallback note stays.
import { loadJSON, h, fmt } from './util.js';
import { routerProbs, perLayerBars, routediffCase, accumulatorChart, tieChooser } from './charts.js';

export function drawFigures(root) {
  const slot = (name) => root.querySelector(`[data-fig="${name}"]`);
  loadJSON('scenarios').then((d) => {
    routerProbs(slot('router'), d.token);
    routediffCase(slot('routediff'), d.scenarios.find((s) => s.id === 'planted_bug'));
    tieChooser(slot('tie'), d.exact_tie);
  }).catch(() => {});
  loadJSON('census').then((d) => perLayerBars(slot('census'), d.structure.bs8, {
    W: 640, H: 220, label: 'Share of token-layers with a different expert set, batch 8 vs batch 1, by layer',
  })).catch(() => {});
  loadJSON('accumulator').then((d) => {
    // Smallest width from which no wider router accumulator flips any route (both layers).
    const rs = d.rows.filter((r) => r.accumulator === 'router' && /^e8m\d+$/.test(r.format));
    const lastFlip = Math.max(...rs.filter((r) => r.flips_total > 0).map((r) => r.mantissa_bits));
    const body = slot('bits');
    const chart = h('div', {});
    const slider = h('input', { type: 'range', min: 7, max: 23, step: 1, value: 12, 'aria-label': 'Mantissa bits in the router accumulator' });
    const out = h('p', { class: 'bits-out', 'aria-live': 'polite' });
    const draw = () => {
      const bits = +slider.value;
      accumulatorChart(chart, d.rows, 'router', lastFlip + 1, { W: 640, H: 240, cursor: bits });
      const r = rs.find((q) => q.mantissa_bits === bits);
      const name = bits === 7 ? ' (bf16)' : bits === 23 ? ' (fp32)' : '';
      const flag = r.calibrated_flagged_L0 && r.calibrated_flagged_L7 ? 'routediff flags it on both layers'
        : r.calibrated_flagged_L0 || r.calibrated_flagged_L7 ? 'routediff flags it on one of the two layers' : 'routediff does not flag it';
      const verb = 'tokens change';
      out.replaceChildren(h('strong', {}, `${bits} bits${name}: ${fmt(r.flips_total)} of ${fmt(2 * r.tokens_per_layer)} ${verb} experts`),
        ` (layer 0: ${fmt(r.flips_layer0)}, layer 7: ${fmt(r.flips_layer7)}); ${flag}.`);
    };
    slider.addEventListener('input', draw);
    body.replaceChildren(chart, h('label', { class: 'bits-slider' }, h('span', {}, '7 bits (bf16)'), slider, h('span', {}, '23 bits (fp32)')), out);
    draw();
  }).catch(() => {});
}
