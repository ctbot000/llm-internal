/**
 * A real byte-pair-encoding tokenizer, trained in the page.
 *
 * Nothing here is a lookup table of pre-chosen splits: the merge list is learned
 * at load time from the corpus plus a frequency-ordered list of common English
 * words, exactly the way a production tokenizer is built — just several orders
 * of magnitude smaller. That is why familiar words survive whole and unfamiliar
 * ones shatter, which is the point section 1 is making.
 */

import { CORPUS, COMMON_WORDS } from './corpus.js';

/* Ids 0-255 are reserved for raw UTF-8 bytes, so any character at all can be
   encoded even when it never appeared in training. Learned symbols follow. */
const BYTE_IDS = 256;
const MERGES = 440;

/** Joins a symbol pair into a Map key. No symbol can contain it. */
const SEP = String.fromCharCode(31);

/** Split text the way a tokenizer does before BPE ever runs. */
const PRETOKEN = / ?\p{L}+| ?\d| ?[^\s\p{L}\d]+|\s+/gu;

export function pretokenize(text) {
  return text.match(PRETOKEN) ?? [];
}

function trainingWordCounts() {
  const counts = new Map();
  const bump = (w, n) => counts.set(w, (counts.get(w) ?? 0) + n);

  for (const piece of pretokenize(CORPUS)) bump(piece, 3);

  // Weight by rank: the words a model meets constantly are the ones that earn
  // a token of their own. A flat list would merge them in a near-arbitrary order.
  COMMON_WORDS.forEach((w, i) => {
    const weight = Math.max(2, Math.round(600 / (i + 12)));
    const cap = `${w[0].toUpperCase()}${w.slice(1)}`;
    bump(` ${w}`, weight);
    bump(w, Math.max(1, Math.round(weight / 3)));
    bump(` ${cap}`, Math.max(1, Math.round(weight / 6)));
    bump(cap, Math.max(1, Math.round(weight / 8)));
  });

  return counts;
}

function train() {
  const counts = trainingWordCounts();

  /** Each word is a list of symbols; merging rewrites that list in place. */
  const words = [...counts].map(([w, n]) => ({ syms: [...w], n }));

  // Seed every printable ASCII character, the way a byte-level tokenizer seeds
  // all 256 bytes: a character that never appeared in training still needs an
  // entry, or a stray 'z' would drop to the byte fallback.
  const base = new Set();
  for (let c = 32; c < 127; c++) base.add(String.fromCharCode(c));
  for (const { syms } of words) for (const sym of syms) base.add(sym);

  const vocab = [...base].sort();
  const merges = [];

  for (let step = 0; step < MERGES; step++) {
    const pairs = new Map();
    for (const { syms, n } of words) {
      for (let i = 0; i + 1 < syms.length; i++) {
        const key = syms[i] + SEP + syms[i + 1];
        pairs.set(key, (pairs.get(key) ?? 0) + n);
      }
    }
    if (pairs.size === 0) break;

    let best = null;
    let bestN = 0;
    for (const [key, n] of pairs) if (n > bestN) { bestN = n; best = key; }
    if (bestN < 2) break;

    const [a, b] = best.split(SEP);
    const joined = a + b;
    merges.push([a, b]);
    vocab.push(joined);

    for (const word of words) {
      const { syms } = word;
      if (syms.length < 2) continue;
      const out = [];
      for (let i = 0; i < syms.length; i++) {
        if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) { out.push(joined); i++; }
        else out.push(syms[i]);
      }
      word.syms = out;
    }
  }

  const ranks = new Map();
  merges.forEach(([a, b], i) => ranks.set(a + SEP + b, i));

  const ids = new Map();
  vocab.forEach((sym, i) => ids.set(sym, BYTE_IDS + i));

  return { ranks, ids, vocab, merges };
}

let model = null;
/** Trained once, on first use, and only if a demo actually needs it. */
export function tokenizer() {
  if (!model) model = train();
  return model;
}

export const vocabSize = () => BYTE_IDS + tokenizer().vocab.length;
export const mergeCount = () => tokenizer().merges.length;

const encoder = new TextEncoder();

/** Apply the learned merges to one pre-token, best (lowest rank) merge first. */
function bpe(piece, ranks) {
  let syms = [...piece];
  if (syms.length < 2) return syms;

  for (;;) {
    let bestRank = Infinity;
    let at = -1;
    for (let i = 0; i + 1 < syms.length; i++) {
      const r = ranks.get(syms[i] + SEP + syms[i + 1]);
      if (r !== undefined && r < bestRank) { bestRank = r; at = i; }
    }
    if (at < 0) return syms;
    syms = [
      ...syms.slice(0, at),
      syms[at] + syms[at + 1],
      ...syms.slice(at + 2),
    ];
  }
}

/**
 * Encode text to tokens.
 * @returns {{text: string, id: number, kind: 'sym'|'byte'}[]}
 */
export function encode(text) {
  const { ranks, ids } = tokenizer();
  const out = [];

  for (const piece of pretokenize(text)) {
    for (const sym of bpe(piece, ranks)) {
      const id = ids.get(sym);
      if (id !== undefined) { out.push({ text: sym, id, kind: 'sym' }); continue; }

      // Never seen in training — emoji, CJK, anything. Fall back to raw bytes,
      // which is why a tokenizer can encode text it was never trained on at all.
      for (const byte of encoder.encode(sym)) {
        out.push({
          text: `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`,
          id: byte,
          kind: 'byte',
        });
      }
    }
  }
  return out;
}
