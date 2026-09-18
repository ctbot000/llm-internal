/** Section 6 — the autoregressive loop, one token at a time. */

import { h, fill, $, slider, segmented, button, ticker, clamp, pct, fixed, mulberry32 } from './util.js';
import { VOCAB, distribution, reshape, backoffLevel, draw, stats as modelStats } from './ngram.js';

const PROMPTS = [
  { label: 'The model', seed: ['The', 'model'] },
  { label: 'It is', seed: ['It', 'is'] },
  { label: 'The sea', seed: ['The', 'sea'] },
  { label: 'What the', seed: ['What', 'the'] },
];

const MAX_TOKENS = 72;
const STEP_MS = 110;

const glue = (w) => (w === '.' || w === ',' ? w : ` ${w}`);

export function mount() {
  const body = $('#gen-body');
  const foot = $('#gen-foot');
  if (!body) return;

  const state = {
    prompt: 0,
    temperature: 0.9,
    topK: 8,
    seed: 7,
    tokens: [],
    running: false,
    lastStep: null,
  };
  let rng = mulberry32(state.seed);

  const out = h('div', { class: 'gen-out', 'aria-live': 'polite' });
  const cands = h('div', { class: 'bars' });
  const stats = h('div', { class: 'grid4', style: 'margin-top:16px' });
  const runBtn = button('Run', () => (state.running ? stop() : start()), { class: 'btn primary' });

  const tSlider = slider({
    label: 'temperature', min: 0, max: 1.8, step: 0.05, value: state.temperature,
    format: (v) => (v < 0.02 ? '0 (greedy)' : fixed(v, 2)),
    onInput: (v) => { state.temperature = v; },
  });
  const kSlider = slider({
    label: 'top-k', min: 0, max: 30, step: 1, value: state.topK,
    format: (v) => (v === 0 ? 'off' : String(v)),
    onInput: (v) => { state.topK = v; },
  });

  function reset(newSeed = true) {
    stop();
    if (newSeed) state.seed = (state.seed * 7919 + 13) % 100000;
    rng = mulberry32(state.seed);
    state.tokens = [];
    state.lastStep = null;
    render();
  }

  function step() {
    if (state.tokens.length >= MAX_TOKENS) { stop(); return; }

    const context = [...PROMPTS[state.prompt].seed, ...state.tokens];
    const probs = distribution(context);
    const { adjusted, final, kept, order } = reshape(probs, state);
    const picked = draw(final, rng);

    state.tokens.push(VOCAB[picked]);
    state.lastStep = { adjusted, kept, order, picked, level: backoffLevel(context), size: context.length };
    render();
  }

  const loop = ticker(STEP_MS, () => step());

  function start() {
    if (state.tokens.length >= MAX_TOKENS) reset();
    state.running = true;
    loop.start();
    render();
  }
  function stop() {
    state.running = false;
    loop.stop();
    render();
  }

  function render() {
    const seedWords = PROMPTS[state.prompt].seed;
    fill(out, [
      h('span', { class: 'gen-seed', text: seedWords.join(' ') }),
      ...state.tokens.map((w, i) => h('span', {
        class: `gen-new${i === state.tokens.length - 1 ? ' gen-last' : ''}`,
        text: glue(w),
      })),
      state.running ? h('span', { class: 'caret' }) : null,
    ]);

    runBtn.textContent = state.running ? 'Pause' : (state.tokens.length ? 'Continue' : 'Run');

    const st = state.lastStep;
    fill(cands, st
      ? st.order.slice(0, 6).map((i) => h('div', {
        class: `bar-row${i === st.picked ? ' picked' : ''}${st.kept.has(i) ? '' : ' muted'}`,
      }, [
        h('div', { class: 'bar-lab', text: VOCAB[i] }),
        h('div', { class: 'bar-track' }, [
          h('div', {
            class: 'bar-fill',
            style: { width: `${clamp(st.adjusted[i] / (st.adjusted[st.order[0]] || 1), 0, 1) * 100}%` },
          }),
        ]),
        h('div', { class: 'bar-num', text: pct(st.adjusted[i], 1) }),
      ]))
      : [h('p', { class: 'hint', style: 'margin:0', text: 'Press run, or take a single step, to see the choice the model is making.' })]);

    fill(stats, [
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Tokens written' }),
        h('div', { class: 'stat-v accent', text: `${state.tokens.length} / ${MAX_TOKENS}` }),
        h('div', { class: 'stat-n', text: 'one forward pass each' }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Context length' }),
        h('div', { class: 'stat-v', text: String(st ? st.size + 1 : seedWords.length) }),
        h('div', { class: 'stat-n', text: 'grows by one every step' }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Last token' }),
        h('div', { class: 'stat-v', text: state.tokens.length ? state.tokens[state.tokens.length - 1] : '—' }),
        h('div', { class: 'stat-n', text: st ? `chosen from ${st.kept.size} candidates` : 'nothing yet' }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Seed' }),
        h('div', { class: 'stat-v', text: String(state.seed) }),
        h('div', { class: 'stat-n', text: 'same seed, same text, every time' }),
      ]),
    ]);
  }

  fill($('#gen-controls'), [segmented({
    label: 'Prompt',
    cls: 'mono',
    options: PROMPTS.map((p, i) => ({ value: String(i), label: p.label })),
    value: '0',
    onChange: (v) => { state.prompt = Number(v); reset(false); },
  })]);

  fill(body, [
    out,
    h('p', { class: 'hint', text: 'With four hundred words of training data it often reproduces whole phrases from the corpus. That is memorisation \u2014 what any model does when it has seen something far more often than it has seen an alternative.' }),
    h('div', { class: 'btn-row', style: 'margin-top:14px' }, [
      runBtn,
      button('One step', () => { stop(); step(); }),
      button('New seed', () => reset(true)),
    ]),
    h('div', { class: 'controls', style: 'margin-top:20px' }, [tSlider, kSlider]),
    h('div', { class: 'split gen-split', style: 'margin-top:20px' }, [
      h('div', {}, [
        h('div', { class: 'strip-label', text: 'the choice at this step' }),
        cands,
      ]),
      h('div', {}, [
        h('div', { class: 'strip-label', text: 'the loop' }),
        h('div', { class: 'steps' }, [
          h('div', { class: 'step-item' }, [h('div', { html: '<b>Read</b> the whole context, every token of it.' })]),
          h('div', { class: 'step-item' }, [h('div', { html: '<b>Score</b> every word in the vocabulary.' })]),
          h('div', { class: 'step-item' }, [h('div', { html: '<b>Reshape</b> with temperature and top-k.' })]),
          h('div', { class: 'step-item' }, [h('div', { html: '<b>Draw</b> one token and append it.' })]),
          h('div', { class: 'step-item' }, [h('div', { html: '<b>Go back to step one.</b> Nothing is carried over but the text itself.' })]),
        ]),
      ]),
    ]),
    stats,
  ]);

  fill(foot, [h('span', {
    text: `This model holds ${modelStats.params.toLocaleString()} counts over a ${modelStats.vocab}-word vocabulary, fitted to ${modelStats.tokens.toLocaleString()} words of text. A frontier model holds hundreds of billions of weights fitted to trillions. The loop around them is identical.`,
  })]);

  render();
}
