/**
 * Section 3 — self-attention, computed rather than illustrated.
 *
 * Each token gets a small feature vector; each head gets real query and key
 * projection matrices. The scores on screen are q·k/√d, the weights are a
 * softmax over them, and the causal mask is applied by setting future scores to
 * -Infinity, which is exactly what the mask does in a transformer.
 */

import { h, fill, $, segmented, softmax, dot, clamp, pct, fixed } from './util.js';

/* Feature dimensions of the toy token vectors. */
const DIMS = ['noun', 'verb', 'pron', 'adj', 'first', 'one', 'rot·sin', 'rot·cos'];
const D = DIMS.length;
const [NOUN, VERB, PRON, ADJ, FIRST, ONE, ROTS, ROTC] = DIMS.map((_, i) => i);

/* The rotation step for the positional pair. Small enough that no two offsets
   within this sentence land on the same angle, which would make a distant token
   score as highly as the adjacent one. */
const THETA = Math.PI / 9;

const SENTENCE = [
  { text: 'The',     tag: 'det' },
  { text: 'cat',     tag: 'noun' },
  { text: 'sat',     tag: 'verb' },
  { text: 'on',      tag: 'prep' },
  { text: 'the',     tag: 'det' },
  { text: 'mat',     tag: 'noun' },
  { text: 'because', tag: 'conj' },
  { text: 'it',      tag: 'pron' },
  { text: 'was',     tag: 'verb' },
  { text: 'warm',    tag: 'adj' },
  { text: '.',       tag: 'punct' },
];

/** One vector per token: its part of speech, plus its position as a rotation. */
const X = SENTENCE.map((t, i) => {
  const v = new Array(D).fill(0);
  if (t.tag === 'noun') v[NOUN] = 1;
  if (t.tag === 'verb') v[VERB] = 1;
  if (t.tag === 'pron') v[PRON] = 1;
  if (t.tag === 'adj') v[ADJ] = 1;
  if (i === 0) v[FIRST] = 1;
  v[ONE] = 1;
  v[ROTS] = Math.sin(i * THETA);
  v[ROTC] = Math.cos(i * THETA);
  return v;
});

const row = (pairs) => {
  const r = new Array(D).fill(0);
  for (const [dim, w] of pairs) r[dim] += w;
  return r;
};

/** A query projection that rotates the position pair backwards by `steps`. */
function rotateBack(steps, scale) {
  const a = steps * THETA;
  return [
    row([[ROTS, scale * Math.cos(a)], [ROTC, -scale * Math.sin(a)]]),
    row([[ROTC, scale * Math.cos(a)], [ROTS, scale * Math.sin(a)]]),
  ];
}
const KEEP_ROT = (scale) => [row([[ROTS, scale]]), row([[ROTC, scale]])];

const HEADS = [
  {
    value: 'prev',
    demo: 7,
    label: 'previous token',
    blurb: 'Reads nothing but position. The query is the token’s own rotation turned back by one step, so it lines up with the key of whatever came immediately before.',
    Wq: rotateBack(1, 54),
    Wk: KEEP_ROT(1),
  },
  {
    value: 'pron',
    demo: 7,
    label: 'pronoun → noun',
    blurb: 'The query fires only for pronouns; the key fires for nouns. Choose "it" and the head goes looking for something it could refer to. For any other query it falls back to a mild preference for nouns.',
    Wq: [row([[PRON, 5]]), row([[ONE, 1.1]])],
    Wk: [row([[NOUN, 1]]), row([[NOUN, 1.1]])],
  },
  {
    value: 'verb',
    demo: 8,
    label: 'verb → arguments',
    blurb: 'Fires when the query is a verb, and looks for the nouns and pronouns around it — the things the verb is about.',
    Wq: [row([[VERB, 5]]), row([[ONE, 0.8]])],
    Wk: [row([[NOUN, 1], [PRON, 1]]), row([[VERB, 0.6]])],
  },
  {
    value: 'content',
    demo: 9,
    label: 'content words',
    blurb: 'No query selectivity at all: every position asks the same question, and the head simply pulls in the words carrying meaning, ignoring "the", "on" and "because".',
    Wq: [row([[ONE, 3]])],
    Wk: [row([[NOUN, 1], [VERB, 0.9], [ADJ, 0.9], [PRON, 0.6]])],
  },
  {
    value: 'sink',
    demo: 9,
    label: 'the sink',
    blurb: 'A head that dumps nearly all of its weight on the first token. Real models grow these, and they are thought to be a way of saying "nothing to retrieve here" — softmax always has to sum to one, so the weight has to go somewhere.',
    Wq: [row([[ONE, 6]])],
    Wk: [row([[FIRST, 1]])],
  },
];

const project = (W, x) => W.map((r) => dot(r, x));

/** Scores, mask, softmax — the whole of attention apart from the value mix. */
function attend(head, causal) {
  const q = X.map((x) => project(head.Wq, x));
  const k = X.map((x) => project(head.Wk, x));
  const scale = Math.sqrt(head.Wq.length);

  const scores = q.map((qi, i) => k.map((kj, j) => (
    causal && j > i ? -Infinity : dot(qi, kj) / scale
  )));
  const weights = scores.map((r) => softmax(r));
  return { scores, weights };
}

function heatColour(w) {
  const a = clamp(Math.pow(w, 0.55), 0, 1);
  return `color-mix(in srgb, var(--accent) ${(a * 100).toFixed(1)}%, var(--surface-3))`;
}

export function mount() {
  const body = $('#att-body');
  const foot = $('#att-foot');
  if (!body) return;

  const state = { head: 'prev', query: 7, causal: true };

  const sentRow = h('div', { class: 'sent' });
  const detail = h('div', { class: 'att-detail' });
  const heat = h('div', { class: 'heat', role: 'grid', 'aria-label': 'Attention weights, queries by keys' });
  const blurb = h('p', { class: 'hint' });

  function render() {
    const head = HEADS.find((x) => x.value === state.head);
    const { scores, weights } = attend(head, state.causal);
    const i = state.query;

    blurb.textContent = head.blurb;

    fill(sentRow, SENTENCE.map((t, j) => h('button', {
      type: 'button',
      class: `sent-tok${j === i ? ' q' : ''}${state.causal && j > i ? ' masked' : ''}`,
      title: `position ${j} · ${t.tag}`,
      onclick: () => { state.query = j; render(); },
    }, [t.text])));

    const rows = SENTENCE
      .map((t, j) => ({ label: t.text, j, w: weights[i][j], sc: scores[i][j] }))
      .filter((r) => Number.isFinite(r.sc));
    const ordered = [...rows].sort((a, b) => b.w - a.w);
    const top = ordered[0];

    // The value mix: the head's output at this position is the weighted average
    // of the other positions' vectors. This is the information actually moving.
    const mixed = new Array(D).fill(0);
    rows.forEach((r) => { for (let d = 0; d < D; d++) mixed[d] += r.w * X[r.j][d]; });

    fill(detail, [
      h('div', { class: 'mathline', html:
        `q<sub>${i}</sub> · k<sub>${top.j}</sub> / √d  =  <b>${fixed(top.sc, 2)}</b>`
        + `  →  softmax over ${rows.length}`
        + `  →  <b>${pct(top.w, 1)}</b> on "${top.label}"` }),
      h('div', { class: 'bars', style: 'margin-top:14px' }, rows.map((r) => h('div', {
        class: `bar-row${r.j === top.j ? ' picked' : ''}${r.w < 0.02 ? ' muted' : ''}`,
      }, [
        h('div', { class: 'bar-lab', text: r.label }),
        h('div', { class: 'bar-track' }, [
          h('div', { class: 'bar-fill', style: { width: `${r.w * 100}%` } }),
        ]),
        h('div', { class: 'bar-num', text: pct(r.w, 1) }),
      ]))),
      h('div', { class: 'mix' }, [
        h('div', { class: 'mix-head' }, [
          h('span', { class: 'panel-title', text: 'what arrives at this position' }),
        ]),
        h('div', { class: 'axis-list' }, DIMS.slice(0, 5).map((name, d) => h('div', { class: 'axis-row' }, [
          h('div', { class: 'axis-name', text: name }),
          h('div', { class: 'bar-track', style: 'height:11px' }, [
            h('div', { class: 'bar-fill', style: { width: `${clamp(mixed[d], 0, 1) * 100}%` } }),
          ]),
          h('div', { class: 'axis-val', text: fixed(mixed[d], 2) }),
        ]))),
      ]),
    ]);

    // role=grid needs real row elements between it and the cells, or browsers
    // prune every cell out of the accessibility tree. .heat-row is display:contents.
    const n = SENTENCE.length;
    heat.style.gridTemplateColumns = `58px repeat(${n}, minmax(15px, 1fr))`;
    fill(heat, [
      h('div', { class: 'heat-row', role: 'row' }, [
        h('div', { class: 'heat-head', role: 'columnheader', text: '' }),
        ...SENTENCE.map((t) => h('div', { class: 'heat-head', role: 'columnheader', text: t.text.slice(0, 4) })),
      ]),
      ...SENTENCE.map((t, r) => h('div', { class: 'heat-row', role: 'row' }, [
        h('div', {
          class: `heat-head${r === i ? ' on' : ''}`,
          role: 'rowheader',
          text: t.text.slice(0, 6),
          onclick: () => { state.query = r; render(); },
        }),
        ...SENTENCE.map((u, c) => h('div', {
          class: 'heat-cell',
          role: 'gridcell',
          'aria-label': `${t.text} attending to ${u.text}: ${pct(weights[r][c], 0)}`,
          title: `${t.text} → ${u.text} · ${pct(weights[r][c], 1)}`,
          style: { background: heatColour(weights[r][c]) },
          onclick: () => { state.query = r; render(); },
        })),
      ])),
    ]);
  }

  const heads = segmented({
    label: 'Attention head',
    options: HEADS.map((x) => ({ value: x.value, label: x.label })),
    value: state.head,
    onChange: (v) => {
      state.head = v;
      // Land on a query that shows what the head is for: several of them do
      // nothing at all unless the query is the right kind of word.
      state.query = HEADS.find((x) => x.value === v).demo;
      render();
    },
  });

  const mask = segmented({
    label: 'Causal mask',
    options: [
      { value: 'on', label: 'masked', title: 'Positions cannot see the future — how a generative model is trained and run' },
      { value: 'off', label: 'unmasked', title: 'Every position sees the whole sentence' },
    ],
    value: 'on',
    onChange: (v) => { state.causal = v === 'on'; render(); },
  });

  fill($('#att-heads'), [heads]);
  fill(body, [
    blurb,
    sentRow,
    h('p', { class: 'hint', text: 'Click a word to make it the query. Dimmed words are in its future and are masked away.' }),
    h('div', { class: 'split att-split' }, [
      detail,
      h('div', {}, [
        h('div', { class: 'strip-label', text: 'every query, every key' }),
        heat,
        h('p', { class: 'hint', text: 'Rows are queries, columns are keys. The empty upper triangle is the causal mask: a token predicting what comes next must not be allowed to read it.' }),
        h('div', { class: 'btn-row', style: 'margin-top:12px' }, [mask]),
      ]),
    ]),
  ]);

  fill(foot, [h('span', {
    text: 'A real layer runs dozens of these heads side by side and concatenates their outputs; a real model has dozens of layers of those. Every head is the same four lines of arithmetic with different learned matrices.',
  })]);

  render();
}
