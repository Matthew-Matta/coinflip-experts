// Small helpers for the figures.

export const SVG_NS = 'http://www.w3.org/2000/svg';

export function svg(tag, attrs = {}, parent) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

export function h(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v);
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) n.append(c);
  return n;
}

// Display formatting only: values come straight from site/data.
export function fmt(v) {
  if (v === null || v === undefined) return 'not measured';
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return v.toLocaleString('en-US');
  if (Math.abs(v) >= 1000) return '≈' + Math.round(v).toLocaleString('en-US');
  if (v !== 0 && Math.abs(v) < 0.01) return v.toExponential(2);
  return String(+v.toFixed(4));
}

export function loadJSON(name) {
  return fetch(`data/${name}.json`, { cache: 'no-cache' }).then((r) => {
    if (!r.ok) throw new Error(`data/${name}.json: HTTP ${r.status}`);
    return r.json();
  });
}
