/**
 * Section 7 — gradient descent, actually running.
 *
 * The model is a bigram softmax: one row of logits per character, plus a bias.
 * Roughly 900 weights, all initialised to noise, trained on the corpus by the
 * same objective a language model is trained on — the negative log probability
 * it assigned to the character that really came next.
 */

import { h, fill, $, slider, button, ticker, softmax, sampleFrom, mulberry32, fixed, clamp } from './util.js';
import { CHAR_TEXT } from './corpus.js';

const ALPHABET = [...new Set(CHAR_TEXT)].sort();
const V = ALPHABET.length;
const idOf = new Map(ALPHABET.map((c, i) => [c, i]));
const DATA = [...CHAR_TEXT].map((c) => idOf.get(c));

const BATCH = 128;
const BATCHES_PER_TICK = 25;
const TICK_MS = 55;
const UNIFORM_LOSS = Math.log(V);

const label = (c) => (c === ' ' ? '␣' : c);

/* ── the model ────────────────────────────────────────────────────────────── */

function freshModel(seed) {
  const rng = mulberry32(seed);
  return {
    W: Array.from({ length: V }, () => Array.from({ length: V }, () => (rng() - 0.5) * 0.02)),
    b: new Array(V).fill(0),
    steps: 0,
    seen: 0,
    loss: UNIFORM_LOSS,
    history: [],
    rng: mulberry32(seed + 1),
  };
}

/** One minibatch: forward, cross-entropy, gradient, update. */
function trainBatch(m, lr) {
  const gW = new Map();
  const gB = new Array(V).fill(0);
  let loss = 0;

  for (let n = 0; n < BATCH; n++) {
    const i = Math.floor(m.rng() * (DATA.length - 1));
    const x = DATA[i];
    const y = DATA[i + 1];

    const logits = m.W[x].map((w, j) => w + m.b[j]);
    const p = softmax(logits);
    loss += -Math.log(Math.max(p[y], 1e-12));

    let row = gW.get(x);
    if (!row) gW.set(x, (row = new Array(V).fill(0)));
    for (let j = 0; j < V; j++) {
      const g = (p[j] - (j === y ? 1 : 0)) / BATCH;
      row[j] += g;
      gB[j] += g;
    }
  }

  for (const [x, row] of gW) for (let j = 0; j < V; j++) m.W[x][j] -= lr * row[j];
  for (let j = 0; j < V; j++) m.b[j] -= lr * gB[j];

  m.steps++;
  m.seen += BATCH;
  // An exponential average: a raw per-batch loss is too noisy to read.
  m.loss = m.steps === 1 ? loss / BATCH : m.loss * 0.96 + (loss / BATCH) * 0.04;
  if (m.steps % 4 === 1) m.history.push(m.loss);
}

function sampleText(m, n, seed) {
  const rng = mulberry32(seed);
  let x = idOf.get(' ');
  let out = '';
  for (let i = 0; i < n; i++) {
    const p = softmax(m.W[x].map((w, j) => w + m.b[j]));
    x = sampleFrom(p, rng);
    out += ALPHABET[x];
  }
  return out.trim();
}

/* ── drawing ──────────────────────────────────────────────────────────────── */

const INK = '#7a80a6';
const CURVE = '#8b7cff';
const BASE = 'rgba(255,190,79,.8)';

function sizeCanvas(canvas, cssW, cssH, min = 240) {
  const dpr = clamp(window.devicePixelRatio || 1, 1, 3);
  // Floor the measured width: a hidden or collapsed pane measures 0, and every
  // coordinate derived from a zero box is unrecoverable noise.
  const w = Math.max(min, Math.round(cssW));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h: cssH };
}

function drawLoss(canvas, m, cssW) {
  const { ctx, w, h } = sizeCanvas(canvas, cssW, 190);
  ctx.clearRect(0, 0, w, h);

  const padL = 38;
  const padB = 22;
  const padT = 12;
  const plotW = w - padL - 10;
  const plotH = h - padB - padT;
  const yMax = UNIFORM_LOSS * 1.08;
  const yMin = 1.2;
  const yOf = (v) => padT + (1 - (clamp(v, yMin, yMax) - yMin) / (yMax - yMin)) * plotH;

  ctx.strokeStyle = 'rgba(122,128,166,.22)';
  ctx.fillStyle = INK;
  ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.lineWidth = 1;
  for (let v = Math.ceil(yMin * 2) / 2; v <= yMax; v += 0.5) {
    const y = yOf(v);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - 10, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(1), 6, y + 3.5);
  }

  ctx.strokeStyle = BASE;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padL, yOf(UNIFORM_LOSS));
  ctx.lineTo(w - 10, yOf(UNIFORM_LOSS));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = BASE;
  ctx.fillText('guessing at random', padL + 6, yOf(UNIFORM_LOSS) - 5);

  const pts = m.history;
  if (pts.length > 1) {
    ctx.strokeStyle = CURVE;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((v, i) => {
      const x = padL + (i / (pts.length - 1)) * plotW;
      const y = yOf(v);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  ctx.fillStyle = INK;
  ctx.fillText('training steps →', padL, h - 6);
}

function drawWeights(canvas, m, cssW) {
  const GUT = 13;
  const cell = Math.max(6, Math.floor((Math.max(240, cssW) - GUT) / V));
  const side = GUT + cell * V;
  const { ctx } = sizeCanvas(canvas, side, side, side);
  ctx.clearRect(0, 0, side, side);

  let peak = 1e-6;
  for (const row of m.W) for (const v of row) peak = Math.max(peak, Math.abs(v));

  ctx.font = `${Math.min(9, cell - 1)}px ui-monospace, Menlo, monospace`;
  ctx.fillStyle = INK;
  ctx.textAlign = 'center';
  for (let i = 0; i < V; i++) {
    ctx.fillText(label(ALPHABET[i]), GUT + i * cell + cell / 2, 9);
    ctx.fillText(label(ALPHABET[i]), 6, GUT + i * cell + cell / 2 + 3);
  }
  ctx.textAlign = 'left';

  for (let x = 0; x < V; x++) {
    for (let y = 0; y < V; y++) {
      const v = m.W[x][y] / peak;
      const a = Math.pow(Math.abs(v), 0.6) * 0.95;
      ctx.fillStyle = v >= 0
        ? `rgba(139,124,255,${a.toFixed(3)})`
        : `rgba(255,111,145,${a.toFixed(3)})`;
      // x is the current character (a row), y is the one it predicts (a column).
      ctx.fillRect(GUT + y * cell, GUT + x * cell, cell - 1, cell - 1);
    }
  }
}

/* ── section ──────────────────────────────────────────────────────────────── */

export function mount() {
  const body = $('#train-body');
  const foot = $('#train-foot');
  if (!body) return;

  let model = freshModel(11);
  let lr = 1.2;
  const before = sampleText(model, 150, 3);

  const lossCanvas = h('canvas');
  const wCanvas = h('canvas');
  const lossBox = h('div', { class: 'canvas-box' }, [lossCanvas]);
  const wBox = h('div', { class: 'canvas-box' }, [wCanvas]);
  const stats = h('div', { class: 'grid4', style: 'margin-top:16px' });
  const sampleOut = h('div', { class: 'gen-out', style: 'min-height:96px' });
  const runBtn = button('Train', () => (loop.running ? pause() : start()), { class: 'btn primary' });

  const lrSlider = slider({
    label: 'learning rate', min: 0.1, max: 4, step: 0.1, value: lr,
    format: (v) => fixed(v, 1),
    onInput: (v) => { lr = v; },
  });

  const loop = ticker(TICK_MS, () => {
    for (let i = 0; i < BATCHES_PER_TICK; i++) trainBatch(model, lr);
    if (model.steps > 4000) pause();
    render();
  });

  function start() { loop.start(); render(); }
  function pause() { loop.stop(); render(); }
  function reset() {
    loop.stop();
    model = freshModel(11 + Math.floor(Math.random() * 1000));
    render();
  }

  function measure(box) {
    const r = box.getBoundingClientRect();
    return Math.max(240, Math.round(r.width));
  }

  function draw() {
    drawLoss(lossCanvas, model, measure(lossBox));
    drawWeights(wCanvas, model, measure(wBox));
  }

  function render() {
    runBtn.textContent = loop.running ? 'Pause' : (model.steps ? 'Keep training' : 'Train');

    fill(stats, [
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Loss' }),
        h('div', { class: 'stat-v accent', text: fixed(model.loss, 3) }),
        h('div', { class: 'stat-n', text: `random guessing scores ${fixed(UNIFORM_LOSS, 2)}` }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Perplexity' }),
        h('div', { class: 'stat-v', text: fixed(Math.exp(model.loss), 1) }),
        h('div', { class: 'stat-n', text: 'characters it feels it is choosing between' }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Updates' }),
        h('div', { class: 'stat-v', text: model.steps.toLocaleString() }),
        h('div', { class: 'stat-n', text: `${model.seen.toLocaleString()} examples seen` }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Parameters' }),
        h('div', { class: 'stat-v', text: (V * V + V).toLocaleString() }),
        h('div', { class: 'stat-n', text: `a ${V}×${V} matrix and a bias` }),
      ]),
    ]);

    fill(sampleOut, [
      h('div', { class: 'gen-seed', style: 'font-size:.78rem;letter-spacing:.1em;text-transform:uppercase', text: 'before training' }),
      h('div', { class: 'gen-seed', text: before }),
      h('div', { class: 'gen-seed', style: 'font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;margin-top:12px', text: `after ${model.steps.toLocaleString()} updates` }),
      h('div', { class: 'gen-new', text: sampleText(model, 150, 3) }),
    ]);

    draw();
  }

  fill($('#train-controls'), [h('div', { class: 'btn-row' }, [
    runBtn,
    button('100 steps', () => {
      loop.stop();
      for (let i = 0; i < 100; i++) trainBatch(model, lr);
      render();
    }),
    button('Reset weights', reset),
  ])]);

  fill(body, [
    h('div', { class: 'split train-split' }, [
      h('div', {}, [
        h('div', { class: 'strip-label', text: 'cross-entropy loss, in nats' }),
        lossBox,
        h('p', { class: 'hint', text: 'Loss is how surprised the model was by the character that actually came next. Lower is better; the dashed line is what pure guessing costs.' }),
      ]),
      h('div', {}, [
        h('div', { class: 'strip-label', text: `weights — ${V}×${V}, row is the current character` }),
        wBox,
        h('p', { class: 'hint', text: 'Every cell is one parameter: row is the current character, column is the one it predicts. Purple is positive, pink negative. Structure appears within a few hundred updates.' }),
      ]),
    ]),
    h('div', { class: 'controls', style: 'margin-top:18px' }, [lrSlider]),
    h('div', { style: 'margin-top:20px' }, [
      h('div', { class: 'strip-label', text: 'sampled from the model itself' }),
      sampleOut,
      h('p', { class: 'hint', text: 'A bigram model can only ever learn which letters follow which. That is enough for letter pairs to look English and for whole words to stay out of reach — the ceiling is the architecture, not the training.' }),
    ]),
    stats,
  ]);

  fill(foot, [h('span', {
    text: 'Same objective, same algorithm, different scale: a frontier model runs this loop over hundreds of billions of weights for months on tens of thousands of accelerators. Nothing in the maths changes.',
  })]);

  // Build and paint now rather than on the first animation frame. A section
  // below the fold in a backgrounded tab never gets one, and the canvases would
  // sit empty with every control apparently doing nothing.
  render();

  if (typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => draw());
    ro.observe(lossBox);
    ro.observe(wBox);
  }
}
