/** Page wiring: theme, nav, reveal, and mounting each section. */

import { h, fill, $, $$, clamp } from './util.js';
import { CORPUS, CHAR_TEXT } from './corpus.js';
import { stats as ngramStats } from './ngram.js';

import * as tokensSection from './sec-tokens.js';
import * as vectorsSection from './sec-vectors.js';
import * as attentionSection from './sec-attention.js';
import * as layersSection from './sec-layers.js';
import * as predictSection from './sec-predict.js';
import * as generateSection from './sec-generate.js';
import * as trainingSection from './sec-training.js';

/* ── theme ────────────────────────────────────────────────────────────────── */

function setupTheme() {
  const root = document.documentElement;
  // Storage is write-through only. It throws in private windows and from opaque
  // origins, so the running value lives here and a failed write costs nothing
  // but persistence across reloads.
  let theme = root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';

  const apply = (next) => {
    theme = next;
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('llm-theme', next); } catch (e) { /* not persisted */ }
  };

  $('#theme')?.addEventListener('click', () => apply(theme === 'dark' ? 'light' : 'dark'));
}

/* ── reveal on scroll ─────────────────────────────────────────────────────── */

function setupReveal() {
  const items = $$('.reveal');
  if (!('IntersectionObserver' in window)) {
    items.forEach((el) => el.classList.add('shown'));
    return () => {};
  }

  const show = (el) => { el.classList.add('shown'); io.unobserve(el); };
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) show(e.target);
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.04 });
  items.forEach((el) => io.observe(el));

  // A fast or programmatic scroll can deliver an observer entry after the
  // element has already left the viewport again, so the entry says "not
  // intersecting" and the section stays invisible while sitting on screen.
  // Sweeping on scroll costs nothing and closes that gap.
  return () => {
    for (const el of items) {
      if (el.classList.contains('shown')) continue;
      const r = el.getBoundingClientRect();
      if (r.top < window.innerHeight * 0.94 && r.bottom > 0) show(el);
    }
  };
}

/* ── nav + progress ───────────────────────────────────────────────────────── */

function setupNav(onScroll) {
  const links = $$('#nav a');
  const bar = $('#progress');
  const byId = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
  const sections = [...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean);

  const update = () => {
    const doc = document.documentElement;
    const span = doc.scrollHeight - window.innerHeight;
    if (bar) bar.style.width = `${clamp(span > 0 ? window.scrollY / span : 0, 0, 1) * 100}%`;

    // Whichever section covers the reading line, not whichever is merely visible.
    const line = window.scrollY + window.innerHeight * 0.32;
    let active = null;
    for (const sec of sections) {
      if (sec.offsetTop <= line) active = sec.id;
    }
    links.forEach((a) => a.classList.toggle('active', a.getAttribute('href') === `#${active}`));
    onScroll?.();
  };

  addEventListener('scroll', update, { passive: true });
  addEventListener('resize', update);
  // A hidden page is delivered neither scroll events nor observer entries, so a
  // tab opened in the background can be scrolled into position with nothing
  // reacting. Catch up the moment it is looked at.
  document.addEventListener('visibilitychange', update);
  update();
}

/* ── the corpus panel ─────────────────────────────────────────────────────── */

function mountCorpus() {
  const body = $('#corpus-body');
  if (!body) return;
  fill(body, [
    h('p', { class: 'corpus-text', text: CORPUS }),
    h('div', { class: 'grid4', style: 'margin-top:18px' }, [
      ['Words', ngramStats.tokens.toLocaleString(), 'the whole training set'],
      ['Vocabulary', String(ngramStats.vocab), 'distinct words in it'],
      ['Characters', CHAR_TEXT.length.toLocaleString(), 'after reducing to a small alphabet'],
      ['Contexts', ngramStats.contexts.toLocaleString(), 'distinct word pairs the model has seen'],
    ].map(([k, v, n]) => h('div', { class: 'stat' }, [
      h('div', { class: 'stat-k', text: k }),
      h('div', { class: 'stat-v', text: v }),
      h('div', { class: 'stat-n', text: n }),
    ]))),
    h('p', { class: 'hint', text: 'A frontier model is trained on something like ten trillion words. This one is trained on four hundred, which is the entire difference between the demos above and a model worth talking to.' }),
  ]);
}

/* ── boot ─────────────────────────────────────────────────────────────────── */

/**
 * Sections are mounted eagerly and independently. Building lazily on the first
 * animation frame would leave the page empty in a backgrounded tab, which never
 * gets one; isolating each mount stops one broken section taking the rest down.
 */
function boot() {
  setupTheme();
  setupNav(setupReveal());

  const sections = [
    ['tokens', tokensSection],
    ['vectors', vectorsSection],
    ['attention', attentionSection],
    ['layers', layersSection],
    ['prediction', predictSection],
    ['generation', generateSection],
    ['training', trainingSection],
  ];

  for (const [name, mod] of sections) {
    try {
      mod.mount();
    } catch (err) {
      console.error(`section "${name}" failed to mount`, err);
    }
  }

  try { mountCorpus(); } catch (err) { console.error('corpus panel failed', err); }
}

boot();
