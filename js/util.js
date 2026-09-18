/* Shared DOM/SVG helpers, small maths, and the controls every section reuses. */

export const SVG_NS = 'http://www.w3.org/2000/svg';

/** Create an HTML element. */
export function h(tag, attrs = {}, kids = []) {
  const n = document.createElement(tag);
  apply(n, attrs);
  add(n, kids);
  return n;
}

/** Create an SVG element. */
export function s(tag, attrs = {}, kids = []) {
  const n = document.createElementNS(SVG_NS, tag);
  apply(n, attrs);
  add(n, kids);
  return n;
}

/**
 * Root <svg> with a viewBox and a definite box on both axes. An <svg> is a
 * replaced element: leave either axis open and its viewBox ratio decides the
 * size, which blows out grid tracks and overflows inset boxes.
 */
export function svgRoot(w, hgt, attrs = {}) {
  return s('svg', {
    viewBox: `0 0 ${w} ${hgt}`,
    width: w,
    height: hgt,
    style: 'width:100%;height:auto;display:block',
    ...attrs,
  });
}

function apply(n, attrs) {
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (k === 'dataset') for (const [dk, dv] of Object.entries(v)) n.dataset[dk] = dv;
    else n.setAttribute(k, v);
  }
}

function add(n, kids) {
  for (const k of [].concat(kids)) {
    if (k == null || k === false) continue;
    n.appendChild(typeof k === 'string' || typeof k === 'number'
      ? document.createTextNode(String(k))
      : k);
  }
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Replace an element's children in one go. */
export function fill(node, kids) {
  node.replaceChildren();
  add(node, kids);
  return node;
}

/* ── maths ────────────────────────────────────────────────────────────────── */

export const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
export const lerp = (a, b, k) => a + (b - a) * k;
export const sum = (xs) => xs.reduce((a, b) => a + b, 0);
export const dot = (a, b) => a.reduce((t, v, i) => t + v * b[i], 0);
export const norm = (a) => Math.sqrt(dot(a, a));

export function cosine(a, b) {
  const d = norm(a) * norm(b);
  return d === 0 ? 0 : dot(a, b) / d;
}

/** Numerically stable softmax with a temperature. */
export function softmax(logits, temperature = 1) {
  const t = Math.max(temperature, 1e-6);
  const scaled = logits.map((v) => v / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const total = sum(exps);
  return exps.map((e) => e / total);
}

/** Matrix-vector product: rows of `m` are the output dimensions. */
export function matVec(m, v) {
  return m.map((row) => dot(row, v));
}

/** Draw one index from a probability vector. */
export function sampleFrom(probs, rng = Math.random) {
  let r = rng();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

/** Deterministic PRNG, so a "random" demo can be replayed exactly. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
export const fixed = (v, d = 2) => (Object.is(v, -0) ? 0 : v).toFixed(d);

/** 1.2K / 3.4M / 1.7B — for parameter counts. */
export function big(n) {
  const units = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
  for (const [k, u] of units) {
    if (n >= k) return `${(n / k).toFixed(n / k >= 100 ? 0 : n / k >= 10 ? 1 : 2)}${u}`;
  }
  return String(Math.round(n));
}

/* ── controls ─────────────────────────────────────────────────────────────── */

/**
 * A labelled range input. `format` renders the live value.
 * Returns the wrapper with `.value`, `.set(v)` and `.sync()` on it.
 */
export function slider({ label, min, max, step, value, format = (v) => v, onInput }) {
  const out = h('output', { class: 'ctl-value', text: format(value) });
  const input = h('input', {
    type: 'range', min, max, step, value,
    class: 'ctl-range',
    'aria-label': label,
  });
  const wrap = h('label', { class: 'ctl' }, [
    h('span', { class: 'ctl-head' }, [h('span', { class: 'ctl-label', text: label }), out]),
    input,
  ]);
  const sync = () => { out.textContent = format(Number(input.value)); };
  input.addEventListener('input', () => {
    // Paint from the handler, not from a frame loop: a hidden tab delivers none.
    sync();
    onInput?.(Number(input.value));
  });
  wrap.input = input;
  Object.defineProperty(wrap, 'value', { get: () => Number(input.value) });
  wrap.set = (v) => { input.value = v; sync(); };
  wrap.sync = sync;
  return wrap;
}

/** A radio-style row of buttons. Returns the wrapper with `.select(value)`. */
export function segmented({ label, options, value, onChange, cls = '' }) {
  const btns = new Map();
  const group = h('div', { class: `seg ${cls}`.trim(), role: 'radiogroup', 'aria-label': label });
  let current = value;

  const select = (v, fire = true) => {
    current = v;
    btns.forEach((b, key) => {
      const on = key === v;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', String(on));
      b.tabIndex = on ? 0 : -1;
    });
    if (fire) onChange?.(v);
  };

  options.forEach((o) => {
    const b = h('button', {
      type: 'button', role: 'radio', class: 'seg-btn',
      title: o.title || '',
      onclick: () => select(o.value),
    }, [o.label]);
    btns.set(o.value, b);
    group.appendChild(b);
  });

  group.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    e.preventDefault();
    const keys = options.map((o) => o.value);
    const i = keys.indexOf(current);
    const next = keys[(i + (e.key === 'ArrowRight' ? 1 : keys.length - 1)) % keys.length];
    select(next);
    btns.get(next).focus();
  });

  select(value, false);
  group.select = select;
  Object.defineProperty(group, 'value', { get: () => current });
  return group;
}

export function button(label, onClick, attrs = {}) {
  return h('button', { type: 'button', class: 'btn', onclick: onClick, ...attrs }, [label]);
}

/* ── animation ────────────────────────────────────────────────────────────── */

/**
 * A wall-clock tween. Never counts frames: a throttled tab stops delivering
 * them and a frame-counted lifetime would stall forever.
 */
export function tween(ms, onFrame, onDone) {
  const t0 = performance.now();
  let stopped = false;
  const step = (now) => {
    if (stopped) return;
    const k = clamp((now - t0) / ms, 0, 1);
    onFrame(k);
    if (k < 1) requestAnimationFrame(step);
    else onDone?.();
  };
  requestAnimationFrame(step);
  return () => { stopped = true; };
}

export const easeOut = (k) => 1 - (1 - k) * (1 - k);

/**
 * setInterval that pauses itself while the page is hidden, and resumes only if
 * it was running when the page went away — a user's pause must survive a tab
 * switch, which a plain visibilitychange restart would quietly undo.
 */
export function ticker(ms, fn) {
  let id = null;
  let wanted = false;
  const run = () => { if (id == null && !document.hidden) id = setInterval(fn, ms); };
  const halt = () => { if (id != null) { clearInterval(id); id = null; } };
  const start = () => { wanted = true; run(); };
  const stop = () => { wanted = false; halt(); };
  const onVis = () => (document.hidden ? halt() : wanted && run());
  document.addEventListener('visibilitychange', onVis);
  return {
    start,
    stop,
    get running() { return wanted; },
    dispose: () => { stop(); document.removeEventListener('visibilitychange', onVis); },
  };
}
