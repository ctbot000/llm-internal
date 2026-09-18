/**
 * Section 4 — the stack.
 *
 * The block diagram is fixed; the arithmetic is not. Every number in the table
 * is computed from the shape controls with the standard parameter formulas for
 * a decoder-only transformer, so the presets land where published models land.
 */

import { h, s, svgRoot, fill, $, slider, segmented, big, pct, clamp } from './util.js';

const PRESETS = {
  toy:   { label: 'this page', d: 8,    L: 1,  heads: 4,  ff: 32,    V: 791,    gated: false, tied: true },
  gpt2:  { label: 'GPT-2 small', d: 768,  L: 12, heads: 12, ff: 3072,  V: 50257,  gated: false, tied: true },
  mid:   { label: '7B-class',  d: 4096, L: 32, heads: 32, ff: 11008, V: 32000,  gated: true,  tied: false },
  large: { label: '70B-class', d: 8192, L: 80, heads: 64, ff: 28672, V: 128256, gated: true,  tied: false },
};

function budget(c) {
  const embed = c.V * c.d * (c.tied ? 1 : 2);
  const attnPer = 4 * c.d * c.d;
  const mlpPer = (c.gated ? 3 : 2) * c.d * c.ff;
  const normPer = 2 * c.d;
  const attn = c.L * attnPer;
  const mlp = c.L * mlpPer;
  const norms = c.L * normPer + c.d;
  const total = embed + attn + mlp + norms;
  return { embed, attn, mlp, norms, total, attnPer, mlpPer, nonEmbed: attn + mlp + norms };
}

function blockDiagram() {
  const W = 300;
  const H = 400;
  const svg = svgRoot(W, H, { role: 'img', 'aria-label': 'One transformer block: attention and feed-forward, each added back into the residual stream' });

  const stream = 52;
  const box = (y, label, cls) => s('g', {}, [
    s('rect', {
      x: 104, y, width: 168, height: 46, rx: 10,
      fill: cls === 'attn' ? 'color-mix(in srgb, var(--accent) 24%, transparent)' : 'color-mix(in srgb, var(--cyan) 20%, transparent)',
      stroke: cls === 'attn' ? 'var(--accent)' : 'var(--cyan)',
    }),
    s('text', {
      x: 188, y: y + 27, 'text-anchor': 'middle',
      fill: 'var(--text)', 'font-size': '12.5', 'font-family': 'var(--sans)', 'font-weight': '650',
      text: label,
    }),
  ]);

  const plus = (cy) => s('g', {}, [
    s('circle', { cx: stream, cy, r: 12, fill: 'var(--surface)', stroke: 'var(--amber)', 'stroke-width': 1.4 }),
    s('text', { x: stream, y: cy + 4.5, 'text-anchor': 'middle', fill: 'var(--amber)', 'font-size': '14', text: '+' }),
  ]);

  const branch = (fromY, toY, boxY) => s('g', { fill: 'none', stroke: 'var(--line)', 'stroke-width': 1.3 }, [
    s('path', { d: `M ${stream} ${fromY} L ${stream} ${boxY + 23} L 104 ${boxY + 23}` }),
    s('path', { d: `M 272 ${boxY + 23} L 288 ${boxY + 23} L 288 ${toY} L ${stream + 13} ${toY}`, 'marker-end': 'url(#ah)' }),
  ]);

  const defs = s('defs', {}, [
    s('marker', { id: 'ah', viewBox: '0 0 10 10', refX: '8', refY: '5', markerWidth: '6', markerHeight: '6', orient: 'auto-start-reverse' }, [
      s('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: 'var(--line)' }),
    ]),
  ]);

  const label = (y, text, colour = 'var(--dimmer)') => s('text', {
    x: stream - 18, y, 'text-anchor': 'end', fill: colour,
    'font-size': '10.5', 'font-family': 'var(--mono)', text,
  });

  svg.append(
    defs,
    s('line', { x1: stream, y1: 18, x2: stream, y2: H - 14, stroke: 'var(--amber)', 'stroke-width': 2.5, opacity: .75 }),
    label(30, 'in'),
    branch(96, 150, 78),
    box(78, 'multi-head attention', 'attn'),
    plus(150),
    label(155, 'add'),
    branch(206, 268, 190),
    box(190, 'feed-forward network', 'mlp'),
    plus(268),
    label(273, 'add'),
    s('text', {
      x: stream - 18, y: 330, 'text-anchor': 'end', fill: 'var(--dimmer)',
      'font-size': '10.5', 'font-family': 'var(--mono)', text: 'out',
    }),
    s('text', {
      x: 150, y: 352, 'text-anchor': 'middle', fill: 'var(--dim)',
      'font-size': '11.5', 'font-family': 'var(--sans)', text: 'the residual stream runs straight through',
    }),
    s('text', {
      x: 150, y: 372, 'text-anchor': 'middle', fill: 'var(--dimmer)',
      'font-size': '11', 'font-family': 'var(--sans)', text: 'each block adds to it, nothing overwrites it',
    }),
  );
  return svg;
}

export function mount() {
  const body = $('#lay-body');
  const foot = $('#lay-foot');
  if (!body) return;

  const cfg = { ...PRESETS.gpt2 };
  let selected = 0;

  const stack = h('div', { class: 'stack-layers' });
  const table = h('div', { class: 'tbl-scroll' });
  const stats = h('div', { class: 'grid3', style: 'margin-top:16px' });
  const shape = h('div', { class: 'controls', style: 'margin-top:4px' });
  const note = h('p', { class: 'hint' });

  const dSlider = slider({
    label: 'model width (d)', min: 128, max: 8192, step: 128, value: cfg.d,
    format: (v) => String(v), onInput: (v) => { cfg.d = v; cfg.ff = v * 4; ffSlider.set(cfg.ff); render(); },
  });
  const lSlider = slider({
    label: 'layers', min: 1, max: 96, step: 1, value: cfg.L,
    format: (v) => String(v), onInput: (v) => { cfg.L = v; selected = Math.min(selected, v - 1); render(); },
  });
  const ffSlider = slider({
    label: 'feed-forward width', min: 128, max: 32768, step: 128, value: cfg.ff,
    format: (v) => String(v), onInput: (v) => { cfg.ff = v; render(); },
  });
  const vSlider = slider({
    label: 'vocabulary', min: 1000, max: 200000, step: 1000, value: cfg.V,
    format: (v) => big(v), onInput: (v) => { cfg.V = v; render(); },
  });

  const mlpKind = segmented({
    label: 'feed-forward shape',
    options: [
      { value: 'plain', label: '2 matrices', title: 'up-projection and down-projection' },
      { value: 'gated', label: '3 (gated)', title: 'gate, up and down — what most recent models use' },
    ],
    value: cfg.gated ? 'gated' : 'plain',
    onChange: (v) => { cfg.gated = v === 'gated'; render(); },
  });

  const presets = segmented({
    label: 'Preset',
    options: Object.entries(PRESETS).map(([k, p]) => ({ value: k, label: p.label })),
    value: 'gpt2',
    onChange: (k) => {
      Object.assign(cfg, PRESETS[k]);
      dSlider.set(cfg.d); lSlider.set(cfg.L); ffSlider.set(cfg.ff); vSlider.set(cfg.V);
      mlpKind.select(cfg.gated ? 'gated' : 'plain', false);
      selected = 0;
      render();
    },
  });

  function render() {
    const b = budget(cfg);

    const shown = Math.min(cfg.L, 28);
    fill(stack, Array.from({ length: shown }, (_, i) => h('div', {
      class: `layer-row${i === selected ? ' on' : ''}`,
      role: 'button',
      tabindex: '0',
      'aria-label': `Layer ${i + 1}`,
      onclick: () => { selected = i; render(); },
      onkeydown: (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selected = i; render(); }
      },
    }, [
      h('div', { class: 'layer-n', text: String(i + 1) }),
      h('div', { class: 'layer-bars' }, [
        h('div', {
          class: 'layer-blk layer-attn',
          style: { flexGrow: String(b.attnPer) },
          text: i === selected ? 'attn' : '',
        }),
        h('div', {
          class: 'layer-blk layer-mlp',
          style: { flexGrow: String(b.mlpPer) },
          text: i === selected ? 'mlp' : '',
        }),
      ]),
    ])));

    note.textContent = cfg.L > shown
      ? `Showing ${shown} of ${cfg.L} layers. Every one of them is the same block with its own weights.`
      : 'Each row is one block. The two bars are sized by where that block’s parameters go.';

    const rows = [
      ['Embeddings', b.embed, cfg.tied ? 'one table, shared with the output' : 'input table plus output head'],
      ['Attention', b.attn, `${cfg.L} × 4 × d² — query, key, value, output`],
      ['Feed-forward', b.mlp, `${cfg.L} × ${cfg.gated ? 3 : 2} × d × ff — where the facts live`],
      ['Norms', b.norms, 'two per block, plus a final one'],
    ];

    fill(table, [h('table', { class: 'tbl' }, [
      h('thead', {}, [h('tr', {}, [
        h('th', { text: 'Where the parameters are' }),
        h('th', { class: 'num', text: 'count' }),
        h('th', { class: 'num', text: 'share' }),
        h('th', { text: '' }),
      ])]),
      h('tbody', {}, rows.map(([k, n, why]) => h('tr', {}, [
        h('td', { text: k }),
        h('td', { class: 'num', text: big(n) }),
        h('td', { class: 'num', text: pct(n / b.total, 1) }),
        h('td', { text: why }),
      ]))),
    ])]);

    const heads = clamp(cfg.heads, 1, 256);
    fill(stats, [
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Parameters' }),
        h('div', { class: 'stat-v accent', text: big(b.total) }),
        h('div', { class: 'stat-n', text: `${big(b.nonEmbed)} of them outside the embedding table` }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Weights in memory' }),
        h('div', { class: 'stat-v', text: `${(b.total * 2 / 1e9).toFixed(2)} GB` }),
        h('div', { class: 'stat-n', text: 'at 2 bytes per weight, before any context is loaded' }),
      ]),
      h('div', { class: 'stat' }, [
        h('div', { class: 'stat-k', text: 'Multiply-adds / token' }),
        h('div', { class: 'stat-v', text: big(2 * b.nonEmbed) }),
        h('div', { class: 'stat-n', text: `roughly 2 per weight \u00b7 ${heads} heads of ${Math.round(cfg.d / heads)} dimensions each` }),
      ]),
    ]);
  }

  fill(shape, [dSlider, lSlider, ffSlider, vSlider]);
  fill($('#lay-controls'), [presets]);
  fill(body, [
    h('div', { class: 'split lay-split' }, [
      h('div', {}, [
        h('div', { class: 'strip-label', text: 'one block' }),
        blockDiagram(),
        h('div', { class: 'btn-row', style: 'margin-top:6px' }, [
          h('span', { class: 'pill', text: 'attention: mixes positions' }),
          h('span', { class: 'pill mlp', text: 'feed-forward: per position' }),
          h('span', { class: 'pill res', text: 'residual stream' }),
        ]),
      ]),
      h('div', {}, [
        h('div', { class: 'strip-label', text: 'the stack' }),
        stack,
        note,
        h('div', { style: 'margin-top:18px' }, [shape, mlpKind]),
      ]),
    ]),
    h('div', { style: 'margin-top:22px' }, [table, stats]),
  ]);

  fill(foot, [h('span', {
    text: 'Real models vary in the details — grouped-query attention shrinks the key and value projections, embeddings are sometimes tied and sometimes not — so published counts land near these numbers rather than exactly on them.',
  })]);

  render();
}
