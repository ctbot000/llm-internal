/** Section 5 — logits, softmax, and the knobs that reshape the distribution. */

import { h, fill, $, slider, segmented, button, clamp, pct, fixed, mulberry32 } from './util.js';
import { VOCAB, distribution, reshape, entropy, backoffLevel, draw } from './ngram.js';

const CONTEXTS = [
  ['the', 'model'],
  ['one', 'token'],
  ['the', 'next'],
  ['a', 'great'],
  ['it', 'is'],
];

const SHOWN = 12;

export function mount() {
  const body = $('#pred-body');
  const foot = $('#pred-foot');
  if (!body) return;

  const state = { ctx: 0, temperature: 1, topK: 0, topP: 1, seed: 1 };
  let lastDraw = null;

  const barBox = h('div', { class: 'bars' });
  const math = h('div', { class: 'mathline' });
  const stats = h('div', { class: 'grid4', style: 'margin-top:16px' });
  const ctxRow = h('div', { class: 'sent' });
  const drawOut = h('div', { class: 'draw-out' });

  const tSlider = slider({
    label: 'temperature', min: 0, max: 2, step: 0.05, value: 1,
    format: (v) => (v < 0.02 ? '0 (greedy)' : fixed(v, 2)),
    onInput: (v) => { state.temperature = v; render(); },
  });
  const kSlider = slider({
    label: 'top-k', min: 0, max: 40, step: 1, value: 0,
    format: (v) => (v === 0 ? 'off' : String(v)),
    onInput: (v) => { state.topK = v; render(); },
  });
  const pSlider = slider({
    label: 'top-p', min: 0.05, max: 1, step: 0.05, value: 1,
    format: (v) => (v >= 0.999 ? 'off' : fixed(v, 2)),
    onInput: (v) => { state.topP = v; render(); },
  });

  function render() {
    const ctx = CONTEXTS[state.ctx];
    const probs = distribution(ctx);
    const { adjusted, final, kept, order } = reshape(probs, state);

    fill(ctxRow, [
      ...ctx.map((w) => h('span', { class: 'sent-tok', text: w })),
      h('span', { class: 'sent-tok q', text: '?' }),
    ]);

    const top = order.slice(0, SHOWN);
    const maxP = adjusted[order[0]] || 1;

    fill(barBox, top.map((i, rank) => h('div', {
      class: `bar-row${rank === 0 ? ' picked' : ''}${kept.has(i) ? '' : ' muted'}`,
      title: kept.has(i) ? '' : 'cut by top-k / top-p before the draw',
    }, [
      h('div', { class: 'bar-lab', text: VOCAB[i] }),
      h('div', { class: 'bar-track' }, [
        h('div', { class: 'bar-fill', style: { width: `${clamp(adjusted[i] / maxP, 0, 1) * 100}%` } }),
      ]),
      h('div', { class: 'bar-num', text: pct(adjusted[i], 1) }),
    ])));

    const best = order[0];
    math.innerHTML = state.temperature < 0.02
      ? `temperature 0 → <b>argmax</b> → always "<b>${VOCAB[best]}</b>"`
      : `softmax(logit / <b>${fixed(state.temperature, 2)}</b>)`
        + `  →  "${VOCAB[best]}" at <b>${pct(adjusted[best], 1)}</b>`
        + `  ·  ${kept.size} of ${VOCAB.length} tokens survive the cut`;

    const H = entropy(final.filter((p) => p > 0));
    fill(stats, [
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Top token' }),
        h('div', { class: 'stat-v accent', text: VOCAB[best] }),
        h('div', { class: 'stat-n', text: `${pct(adjusted[best], 1)} of the mass` }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Candidates kept' }),
        h('div', { class: 'stat-v', text: String(kept.size) }),
        h('div', { class: 'stat-n', text: `out of ${VOCAB.length} in the vocabulary` }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Entropy' }),
        h('div', { class: 'stat-v', text: `${fixed(H, 2)} bits` }),
        h('div', { class: 'stat-n', text: 'how undecided the model is' }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Effective choices' }),
        h('div', { class: 'stat-v', text: fixed(Math.pow(2, H), 1) }),
        h('div', { class: 'stat-n', text: `context matched a ${backoffLevel(ctx)}-gram` }),
      ]),
    ]);

    fill(drawOut, lastDraw == null ? [
      h('span', { class: 'hint', style: 'margin:0', text: 'Nothing drawn yet.' }),
    ] : [
      h('span', { class: 'hint', style: 'margin:0', text: 'drew' }),
      h('span', { class: 'sent-tok q', text: VOCAB[lastDraw] }),
      h('span', { class: 'hint', style: 'margin:0', text: `at ${pct(final[lastDraw], 1)} after reshaping` }),
    ]);
  }

  const drawBtn = button('Draw a token', () => {
    const probs = distribution(CONTEXTS[state.ctx]);
    const { final } = reshape(probs, state);
    lastDraw = draw(final, mulberry32(state.seed++));
    render();
  }, { class: 'btn primary' });

  const resetBtn = button('Reset knobs', () => {
    state.temperature = 1; state.topK = 0; state.topP = 1;
    tSlider.set(1); kSlider.set(0); pSlider.set(1);
    lastDraw = null;
    render();
  });

  fill($('#pred-context'), [segmented({
    label: 'Context',
    cls: 'mono',
    options: CONTEXTS.map((c, i) => ({ value: String(i), label: c.join(' ') })),
    value: '0',
    onChange: (v) => { state.ctx = Number(v); lastDraw = null; render(); },
  })]);

  fill(body, [
    h('p', { class: 'hint', text: 'The distribution below is real: it comes from the model trained on this page’s corpus, over its whole vocabulary. Greyed-out bars were cut before the draw.' }),
    ctxRow,
    h('div', { class: 'controls', style: 'margin:18px 0 20px' }, [tSlider, kSlider, pSlider]),
    math,
    h('div', { style: 'margin-top:16px' }, [barBox]),
    stats,
    h('div', { class: 'btn-row', style: 'margin-top:18px' }, [drawBtn, resetBtn, drawOut]),
  ]);

  fill(foot, [h('span', {
    text: 'Top-k keeps a fixed number of candidates; top-p keeps however many it takes to cover that much probability, which adapts to how confident the model is. Most interfaces expose both and apply them in that order.',
  })]);

  render();
}
