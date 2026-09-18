/**
 * Section 2 — embeddings.
 *
 * A real embedding table has thousands of dimensions and no names for any of
 * them. This one has eight, each hand-labelled, so the geometry is readable.
 * Everything computed from those vectors — the similarities, the analogy
 * arithmetic, the 2D map — is done the way it is done on the real thing.
 */

import { h, s, svgRoot, fill, $, segmented, cosine, dot, clamp, fixed } from './util.js';

const AXES = [
  'alive',
  'human',
  'power',
  'masculine',
  'size',
  'concrete',
  'motion',
  'warmth',
];

const CATS = {
  person:   { label: 'people',    color: 'var(--accent)' },
  animal:   { label: 'animals',   color: 'var(--green)' },
  nature:   { label: 'nature',    color: 'var(--cyan)' },
  thing:    { label: 'things',    color: 'var(--amber)' },
  action:   { label: 'actions',   color: 'var(--rose)' },
  abstract: { label: 'abstract',  color: 'var(--dimmer)' },
};

/* word: [alive, human, power, masculine, size, concrete, motion, warmth] */
const WORDS = [
  ['king',     'person',  [ 0.90,  0.95,  0.95,  0.85,  0.50,  0.80,  0.10,  0.30]],
  ['queen',    'person',  [ 0.90,  0.95,  0.95, -0.85,  0.40,  0.80,  0.10,  0.35]],
  ['prince',   'person',  [ 0.90,  0.95,  0.72,  0.80,  0.15,  0.80,  0.18,  0.30]],
  ['princess', 'person',  [ 0.90,  0.95,  0.72, -0.80,  0.10,  0.80,  0.18,  0.35]],
  ['man',      'person',  [ 0.90,  0.98,  0.05,  0.90,  0.30,  0.85,  0.15,  0.10]],
  ['woman',    'person',  [ 0.90,  0.98,  0.05, -0.90,  0.25,  0.85,  0.15,  0.15]],
  ['boy',      'person',  [ 0.90,  0.98,  0.00,  0.85, -0.30,  0.85,  0.35,  0.20]],
  ['girl',     'person',  [ 0.90,  0.98,  0.00, -0.85, -0.35,  0.85,  0.35,  0.25]],
  ['father',   'person',  [ 0.90,  0.98,  0.35,  0.88,  0.30,  0.80,  0.05,  0.55]],
  ['mother',   'person',  [ 0.90,  0.98,  0.35, -0.88,  0.25,  0.80,  0.05,  0.65]],
  ['soldier',  'person',  [ 0.90,  0.98,  0.40,  0.55,  0.30,  0.85,  0.55, -0.35]],
  ['dog',      'animal',  [ 0.95, -0.70,  0.00,  0.05, -0.10,  0.90,  0.50,  0.50]],
  ['cat',      'animal',  [ 0.95, -0.70,  0.00, -0.05, -0.25,  0.90,  0.45,  0.45]],
  ['horse',    'animal',  [ 0.95, -0.75,  0.10,  0.05,  0.55,  0.90,  0.70,  0.35]],
  ['lion',     'animal',  [ 0.95, -0.75,  0.45,  0.20,  0.50,  0.90,  0.60, -0.10]],
  ['bird',     'animal',  [ 0.95, -0.75,  0.00,  0.00, -0.55,  0.90,  0.85,  0.40]],
  ['tree',     'nature',  [ 0.70, -0.90,  0.05,  0.00,  0.50,  0.95, -0.60,  0.30]],
  ['flower',   'nature',  [ 0.70, -0.90,  0.00, -0.10, -0.60,  0.95, -0.50,  0.60]],
  ['mountain', 'nature',  [-0.80, -0.90,  0.15,  0.00,  0.95,  0.95, -0.70,  0.15]],
  ['river',    'nature',  [-0.60, -0.90,  0.05,  0.00,  0.60,  0.95,  0.85,  0.25]],
  ['sea',      'nature',  [-0.60, -0.90,  0.25,  0.00,  0.95,  0.95,  0.70,  0.10]],
  ['stone',    'thing',   [-0.90, -0.90,  0.00,  0.00, -0.20,  0.98, -0.90,  0.00]],
  ['book',     'thing',   [-0.85,  0.10,  0.10,  0.00, -0.20,  0.90, -0.60,  0.40]],
  ['city',     'thing',   [-0.30,  0.25,  0.25,  0.00,  0.90,  0.85,  0.10,  0.10]],
  ['village',  'thing',   [-0.30,  0.25, -0.20,  0.00, -0.40,  0.85, -0.10,  0.30]],
  ['run',      'action',  [ 0.20,  0.15,  0.05,  0.00,  0.00, -0.60,  0.98,  0.10]],
  ['walk',     'action',  [ 0.20,  0.15,  0.00,  0.00,  0.00, -0.60,  0.70,  0.15]],
  ['sleep',    'action',  [ 0.20,  0.15,  0.00,  0.00,  0.00, -0.60, -0.80,  0.25]],
  ['love',     'abstract',[ 0.00,  0.30,  0.05, -0.10,  0.00, -0.95,  0.10,  0.95]],
  ['fear',     'abstract',[ 0.00,  0.30,  0.05,  0.00,  0.00, -0.95,  0.30, -0.90]],
  ['war',      'abstract',[-0.20,  0.30,  0.60,  0.30,  0.80, -0.70,  0.80, -0.95]],
  ['peace',    'abstract',[-0.20,  0.30,  0.30, -0.15,  0.40, -0.90, -0.60,  0.90]],
  ['idea',     'abstract',[-0.50,  0.20,  0.05,  0.00,  0.00, -0.98,  0.00,  0.30]],
  ['truth',    'abstract',[-0.50,  0.20,  0.25,  0.00,  0.00, -0.98, -0.10,  0.50]],
].map(([word, cat, vec], i) => ({ word, cat, vec, id: 1024 + i * 37 }));

const byWord = new Map(WORDS.map((w) => [w.word, w]));

/* ── a real 2D projection, not hand-placed dots ───────────────────────────── */

/** Top-2 principal components by power iteration, deflating after the first. */
function pca2(vectors) {
  const d = vectors[0].length;
  const mean = Array.from({ length: d }, (_, j) =>
    vectors.reduce((t, v) => t + v[j], 0) / vectors.length);
  const centred = vectors.map((v) => v.map((x, j) => x - mean[j]));

  let cov = Array.from({ length: d }, (_, a) =>
    Array.from({ length: d }, (_, b) =>
      centred.reduce((t, v) => t + v[a] * v[b], 0) / centred.length));

  const axes = [];
  for (let k = 0; k < 2; k++) {
    // A fixed start makes the map identical on every load; a random one would
    // flip the picture between visits for no reason.
    let v = Array.from({ length: d }, (_, j) => Math.cos((j + 1) * (k + 1.7)));
    for (let it = 0; it < 220; it++) {
      const next = cov.map((row) => dot(row, v));
      const len = Math.hypot(...next);
      if (len < 1e-12) break;
      v = next.map((x) => x / len);
    }
    axes.push(v);
    const lambda = dot(v, cov.map((row) => dot(row, v)));
    cov = cov.map((row, a) => row.map((x, b) => x - lambda * v[a] * v[b]));
  }

  return { mean, axes };
}

/* ── views ────────────────────────────────────────────────────────────────── */

function vectorStrip(vec) {
  return h('div', { class: 'vec' }, vec.map((v) => h('div', { class: 'vec-cell' }, [
    v >= 0
      ? h('div', { class: 'vec-pos', style: { height: `${Math.abs(v) * 50}%` } })
      : h('div', { class: 'vec-neg', style: { height: `${Math.abs(v) * 50}%` } }),
  ])));
}

function axisList(vec) {
  return h('div', { class: 'axis-list' }, AXES.map((name, j) => h('div', { class: 'axis-row' }, [
    h('div', { class: 'axis-name', text: name }),
    h('div', {
      class: 'axis-val',
      text: fixed(vec[j], 2),
      style: { color: vec[j] >= 0 ? 'var(--accent-2)' : 'var(--rose)' },
    }),
  ])));
}

function bars(rows, { pickTop = false } = {}) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-6);
  return h('div', { class: 'bars' }, rows.map((r, i) => h('div', {
    class: `bar-row${pickTop && i === 0 ? ' picked' : ''}`,
  }, [
    h('div', { class: 'bar-lab', text: r.label }),
    h('div', { class: 'bar-track' }, [
      h('div', { class: 'bar-fill', style: { width: `${clamp(Math.abs(r.value) / max, 0, 1) * 100}%` } }),
    ]),
    h('div', { class: 'bar-num', text: fixed(r.value, 3) }),
  ])));
}

export function mount() {
  const body = $('#emb-body');
  const foot = $('#emb-foot');
  if (!body) return;

  const state = { mode: 'inspect', word: 'king', a: 'king', b: 'man', c: 'woman' };

  const { mean, axes } = pca2(WORDS.map((w) => w.vec));
  const proj = WORDS.map((w) => {
    const c = w.vec.map((x, j) => x - mean[j]);
    return [dot(c, axes[0]), dot(c, axes[1])];
  });
  const xs = proj.map((p) => p[0]);
  const ys = proj.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];

  const W = 470;
  const H = 400;
  const PAD = 30;
  const sx = (x) => PAD + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * PAD);
  const sy = (y) => H - PAD - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * PAD);

  const svg = svgRoot(W, H, { role: 'img', 'aria-label': 'Two-dimensional projection of the embedding space' });
  const linkLayer = s('g');
  const nodeLayer = s('g');
  // Links are drawn first on purpose: SVG has no z-index, so anything appended
  // later paints over what came before.
  svg.append(linkLayer, nodeLayer);

  // Dots sit where the projection puts them; only the labels are nudged apart,
  // with a leader line wherever one had to move. Moving the dots instead would
  // make the picture easier to read and quietly wrong.
  const labelBoxes = WORDS.map((w, i) => ({
    x: sx(proj[i][0]) + 7,
    y: sy(proj[i][1]),
    w: w.word.length * 5.3 + 3,
    h: 11.5,
  }));
  for (let it = 0; it < 300; it++) {
    let moved = false;
    for (let i = 0; i < labelBoxes.length; i++) {
      for (let j = i + 1; j < labelBoxes.length; j++) {
        const a = labelBoxes[i];
        const b = labelBoxes[j];
        const ox = (a.w + b.w) / 2 - Math.abs((b.x + b.w / 2) - (a.x + a.w / 2));
        const oy = (a.h + b.h) / 2 - Math.abs(b.y - a.y);
        if (ox <= 0 || oy <= 0) continue;
        moved = true;
        if (oy <= ox) {
          const dir = b.y === a.y ? 1 : Math.sign(b.y - a.y);
          const push = (oy / 2 + 0.3) * dir;
          a.y -= push; b.y += push;
        } else {
          const dir = b.x === a.x ? 1 : Math.sign(b.x - a.x);
          const push = (ox / 2 + 0.3) * dir;
          a.x -= push; b.x += push;
        }
      }
    }
    if (!moved) break;
  }
  labelBoxes.forEach((box) => {
    box.x = clamp(box.x, 4, W - box.w - 4);
    box.y = clamp(box.y, 12, H - 8);
  });

  const nodes = WORDS.map((w, i) => {
    const [px, py] = [sx(proj[i][0]), sy(proj[i][1])];
    const box = labelBoxes[i];
    const g = s('g', { class: 'map-node', tabindex: '0', role: 'button', 'aria-label': w.word });
    const disc = s('circle', { cx: px, cy: py, r: 4.2, fill: CATS[w.cat].color, opacity: .85 });
    const label = s('text', { x: box.x, y: box.y + 3.4, text: w.word });
    const far = Math.hypot(box.x - (px + 7), box.y - py) > 6;
    const leader = far ? s('line', {
      x1: px, y1: py, x2: box.x - 2, y2: box.y,
      stroke: CATS[w.cat].color, 'stroke-width': .7, opacity: .35,
    }) : null;
    g.append(disc, ...(leader ? [leader] : []), label);
    const pick = () => { state.word = w.word; render(); };
    g.addEventListener('click', pick);
    g.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); }
    });
    return { ...w, g, disc, label, px, py };
  });
  nodes.forEach((n) => nodeLayer.appendChild(n.g));

  const mapBox = h('div', { class: 'map-wrap' }, [
    svg,
    h('div', { class: 'legend' }, Object.values(CATS).map((c) => h('span', { class: 'legend-item' }, [
      h('span', { class: 'legend-dot', style: { background: c.color } }),
      c.label,
    ]))),
    h('p', { class: 'hint', text: 'The two axes are the first two principal components of the eight-dimensional table, computed in the page. Click any word.' }),
  ]);

  const side = h('div', { class: 'side' });

  const select = (value, onChange, label) => h('select', {
    class: 'select', 'aria-label': label, onchange: (e) => onChange(e.target.value),
  }, WORDS.map((w) => h('option', { value: w.word, selected: w.word === value, text: w.word })));

  function neighbours(vec, exclude = []) {
    return WORDS
      .filter((w) => !exclude.includes(w.word))
      .map((w) => ({ label: w.word, value: cosine(vec, w.vec) }))
      .sort((p, q) => q.value - p.value);
  }

  function renderSide() {
    if (state.mode === 'inspect') {
      const w = byWord.get(state.word);
      fill(side, [
        h('div', { class: 'side-head' }, [
          h('h3', { text: `"${w.word}"` }),
          h('span', { class: 'hint', style: 'margin:0', text: `token id ${w.id}` }),
        ]),
        h('p', { class: 'hint', text: 'The eight numbers the model would store for this token. Bars above the line are positive, below are negative.' }),
        vectorStrip(w.vec),
        h('div', { style: 'margin-top:14px' }, [axisList(w.vec)]),
      ]);
      return;
    }

    if (state.mode === 'similar') {
      const w = byWord.get(state.word);
      const rows = neighbours(w.vec, [w.word]).slice(0, 8);
      fill(side, [
        h('div', { class: 'side-head' }, [
          h('h3', { text: `Closest to "${w.word}"` }),
        ]),
        h('p', { class: 'hint', text: 'Cosine similarity: the angle between two vectors, ignoring their length. 1.00 is the same direction, 0 is unrelated.' }),
        bars(rows, { pickTop: true }),
      ]);
      return;
    }

    const A = byWord.get(state.a).vec;
    const B = byWord.get(state.b).vec;
    const C = byWord.get(state.c).vec;
    const target = A.map((x, j) => x - B[j] + C[j]);
    const rows = neighbours(target, [state.a, state.b, state.c]).slice(0, 6);

    fill(side, [
      h('div', { class: 'side-head' }, [h('h3', { text: 'Vector arithmetic' })]),
      h('div', { class: 'ana-row' }, [
        select(state.a, (v) => { state.a = v; render(); }, 'First word'),
        h('span', { class: 'ana-op', text: '−' }),
        select(state.b, (v) => { state.b = v; render(); }, 'Second word'),
        h('span', { class: 'ana-op', text: '+' }),
        select(state.c, (v) => { state.c = v; render(); }, 'Third word'),
      ]),
      h('p', { class: 'hint', text: 'Subtract one word from another and add a third. The result is a point in space that usually has no word exactly on it — so look for the nearest ones.' }),
      vectorStrip(target.map((v) => clamp(v, -1, 1))),
      h('div', { style: 'margin-top:14px' }, [bars(rows, { pickTop: true })]),
    ]);
  }

  function render() {
    const focus = state.mode === 'analogy' ? state.a : state.word;
    const focusVec = state.mode === 'analogy'
      ? byWord.get(state.a).vec.map((x, j) => x - byWord.get(state.b).vec[j] + byWord.get(state.c).vec[j])
      : byWord.get(state.word).vec;

    const near = new Set(neighbours(focusVec, [focus]).slice(0, 3).map((r) => r.label));

    nodes.forEach((n) => {
      const on = n.word === focus;
      n.g.classList.toggle('on', on);
      n.disc.setAttribute('r', on ? 7 : near.has(n.word) ? 5.6 : 4.2);
      n.disc.setAttribute('opacity', on || near.has(n.word) ? 1 : .55);
    });

    const from = nodes.find((n) => n.word === focus);
    fill(linkLayer, [...near].map((word) => {
      const to = nodes.find((n) => n.word === word);
      return s('line', {
        x1: from.px, y1: from.py, x2: to.px, y2: to.py,
        stroke: 'var(--accent)', 'stroke-width': 1, opacity: .35,
      });
    }));

    renderSide();
  }

  const modes = segmented({
    label: 'View',
    options: [
      { value: 'inspect', label: 'The vector' },
      { value: 'similar', label: 'Neighbours' },
      { value: 'analogy', label: 'Arithmetic' },
    ],
    value: state.mode,
    onChange: (v) => { state.mode = v; render(); },
  });

  fill($('#emb-modes'), [modes]);
  fill(body, [h('div', { class: 'split' }, [mapBox, side])]);
  fill(foot, [h('span', {
    text: 'A frontier model stores a vector of a few thousand numbers for each of ~100,000 tokens — the embedding table alone runs to hundreds of millions of parameters, and it is the smallest part of the model.',
  })]);

  render();
}
