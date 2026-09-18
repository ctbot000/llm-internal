/** Section 1 — the tokenizer, live over whatever you type. */

import { h, fill, $, segmented, big, fixed } from './util.js';
import { encode, vocabSize, mergeCount } from './tokenizer.js';

const PRESETS = [
  {
    value: 'prose',
    label: 'Prose',
    text: 'The model reads the sentence one token at a time, and guesses what comes next.',
  },
  {
    value: 'spelling',
    label: 'Spelling',
    text: 'How many letter r characters are there in strawberry?',
  },
  {
    value: 'rare',
    label: 'Rare words',
    text: 'Tokenization of antidisestablishmentarianism is unquestionably inelegant.',
  },
  {
    value: 'numbers',
    label: 'Numbers',
    text: 'Add 4821 and 93017, then divide the result by 6.',
  },
  {
    value: 'code',
    label: 'Code',
    text: 'const probs = softmax(logits.map(v => v / temperature));',
  },
  {
    value: 'utf8',
    label: 'Other scripts',
    text: 'Hello, 안녕하세요, здравствуйте 🌊',
  },
];

/** A stable hue per token id, so the same token always gets the same colour. */
const hue = (id) => (id * 47) % 360;

function tokenNode(tok) {
  const isByte = tok.kind === 'byte';
  const raw = tok.text;
  const leading = raw.startsWith(' ');
  const shown = (leading ? raw.slice(1) : raw).replace(/\n/g, '⏎').replace(/\t/g, '⇥');

  const hueValue = isByte ? 348 : hue(tok.id);
  const node = h('span', {
    class: `tok${leading ? ' space' : ''}`,
    title: isByte
      ? `byte 0x${tok.id.toString(16).toUpperCase().padStart(2, '0')} — this character was never seen in training`
      : `${JSON.stringify(raw)} → id ${tok.id}`,
    style: {
      background: `hsl(${hueValue} 72% 55% / .20)`,
      borderColor: `hsl(${hueValue} 72% 58% / .45)`,
    },
  }, [
    h('span', { class: 'tok-txt', text: shown || '·' }),
    h('span', { class: 'tok-id', text: String(tok.id) }),
  ]);
  return node;
}

function stat(k, v, note, accent = false) {
  return h('div', { class: 'stat' }, [
    h('div', { class: 'stat-k', text: k }),
    h('div', { class: `stat-v${accent ? ' accent' : ''}`, text: v }),
    note ? h('div', { class: 'stat-n', text: note }) : null,
  ]);
}

export function mount() {
  const body = $('#tok-body');
  const foot = $('#tok-foot');
  if (!body) return;

  const input = h('textarea', {
    class: 'field',
    rows: 3,
    spellcheck: 'false',
    'aria-label': 'Text to tokenize',
  });
  input.value = PRESETS[0].text;

  const out = h('div', { class: 'tok-out', id: 'tok-out' });
  const stats = h('div', { class: 'grid4', style: 'margin-top:16px' });

  const render = () => {
    const text = input.value;
    const toks = encode(text);

    fill(out, toks.length
      ? toks.map(tokenNode)
      : [h('span', { class: 'hint', text: 'Nothing to tokenize yet.' })]);

    const bytes = toks.filter((t) => t.kind === 'byte').length;
    const unique = new Set(toks.map((t) => t.id)).size;
    fill(stats, [
      stat('Characters', String([...text].length), 'what you typed'),
      stat('Tokens', String(toks.length), 'what the model receives', true),
      stat('Chars / token', toks.length ? fixed([...text].length / toks.length, 2) : '0', 'about 4 in real models'),
      stat('Distinct ids', String(unique), bytes ? `${bytes} byte fallbacks` : 'all in vocabulary'),
    ]);
  };

  const presets = segmented({
    label: 'Example text',
    options: PRESETS.map((p) => ({ value: p.value, label: p.label })),
    value: PRESETS[0].value,
    onChange: (v) => {
      input.value = PRESETS.find((p) => p.value === v).text;
      render();
    },
  });

  // Paint straight from the input event. A frame loop would be wrong here: a
  // backgrounded tab delivers no frames and the output would silently stall.
  input.addEventListener('input', render);

  fill($('#tok-presets'), [presets]);
  fill(body, [
    input,
    h('p', { class: 'hint', text: 'Type anything. Every coloured chip is one token, with its vocabulary id underneath; a dot marks a leading space, which belongs to the token after it.' }),
    out,
    stats,
  ]);

  fill(foot, [
    h('span', {
      text: `Vocabulary: ${big(vocabSize())} entries, of which 256 are raw bytes and ${mergeCount()} were learned by merging frequent pairs. A production tokenizer learns the same way and stops at around 100,000.`,
    }),
  ]);

  render();
}
