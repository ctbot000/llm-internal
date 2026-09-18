/**
 * A small but genuine language model: word-level trigram counts with backoff,
 * fitted to the corpus when the page loads.
 *
 * It is the weakest model that is still recognisably the same object as a
 * transformer — it takes a context, returns a probability for every word it
 * knows, and knows nothing else. Sections 5 and 6 sample from it with the same
 * temperature / top-k / top-p machinery a real sampler uses.
 */

import { CORPUS } from './corpus.js';
import { softmax, sampleFrom, clamp } from './util.js';

const BACKOFF = 0.4;

export const WORD_RE = /[A-Za-z']+|[.,]/g;
export const words = CORPUS.match(WORD_RE) ?? [];

/** vocabulary, sorted so ids are stable across loads */
export const VOCAB = [...new Set(words)].sort();
export const index = new Map(VOCAB.map((w, i) => [w, i]));

function countTable(order) {
  const table = new Map();
  for (let i = order; i < words.length; i++) {
    const key = words.slice(i - order, i).join(' ');
    let row = table.get(key);
    if (!row) table.set(key, (row = { total: 0, next: new Map() }));
    row.total++;
    row.next.set(words[i], (row.next.get(words[i]) ?? 0) + 1);
  }
  return table;
}

const tri = countTable(2);
const bi = countTable(1);
const uni = countTable(0);
const uniRow = uni.get('') ?? { total: 1, next: new Map() };

export const stats = {
  tokens: words.length,
  vocab: VOCAB.length,
  contexts: tri.size,
  params: tri.size + bi.size + VOCAB.length,
};

/**
 * P(next | context) for every word in the vocabulary, by stupid backoff:
 * trust the trigram if it was ever seen, otherwise fall back a level and
 * discount. Returns probabilities that sum to 1.
 */
export function distribution(context) {
  const w1 = context[context.length - 2];
  const w2 = context[context.length - 1];
  const t = w1 !== undefined ? tri.get(`${w1} ${w2}`) : undefined;
  const b = w2 !== undefined ? bi.get(w2) : undefined;

  const scores = VOCAB.map((w) => {
    if (t) {
      const c = t.next.get(w);
      if (c) return c / t.total;
    }
    if (b) {
      const c = b.next.get(w);
      if (c) return BACKOFF * (c / b.total);
    }
    return BACKOFF * BACKOFF * ((uniRow.next.get(w) ?? 0.5) / uniRow.total);
  });

  const total = scores.reduce((a, x) => a + x, 0);
  return scores.map((x) => x / total);
}

/** How far the model had to back off — useful for saying so on screen. */
export function backoffLevel(context) {
  const w1 = context[context.length - 2];
  const w2 = context[context.length - 1];
  if (w1 !== undefined && tri.has(`${w1} ${w2}`)) return 3;
  if (w2 !== undefined && bi.has(w2)) return 2;
  return 1;
}

/**
 * Reshape a distribution the way a sampler does, in the usual order:
 * temperature on the logits, then top-k, then top-p, then renormalise.
 * Returns the adjusted probabilities plus which entries survived.
 */
export function reshape(probs, { temperature = 1, topK = 0, topP = 1 } = {}) {
  const t = clamp(temperature, 0, 5);

  let adjusted;
  if (t < 0.02) {
    // Temperature 0 is not a sample at all: it is argmax.
    let best = 0;
    probs.forEach((p, i) => { if (p > probs[best]) best = i; });
    adjusted = probs.map((_, i) => (i === best ? 1 : 0));
  } else {
    const logits = probs.map((p) => Math.log(Math.max(p, 1e-12)));
    adjusted = softmax(logits, t);
  }

  const order = adjusted.map((p, i) => i).sort((a, b) => adjusted[b] - adjusted[a]);
  const kept = new Set();
  let cumulative = 0;
  for (const i of order) {
    if (topK > 0 && kept.size >= topK) break;
    if (kept.size > 0 && cumulative >= topP) break;
    kept.add(i);
    cumulative += adjusted[i];
  }

  const mass = [...kept].reduce((a, i) => a + adjusted[i], 0) || 1;
  const final = adjusted.map((p, i) => (kept.has(i) ? p / mass : 0));
  return { adjusted, final, kept, order };
}

export function entropy(probs) {
  return -probs.reduce((t, p) => (p > 0 ? t + p * Math.log2(p) : t), 0);
}

export function draw(final, rng) {
  return sampleFrom(final, rng);
}
