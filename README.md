# How an LLM Works

An interactive, visual walkthrough of a large language model — from raw text to
a sampled token — with every step running live in the page.

**Live: <https://ctbot000.github.io/llm-internal/>**

## What it covers

| # | Section | What is actually running |
|---|---------|--------------------------|
| 1 | Tokenization | A real byte-pair-encoding tokenizer, trained in the browser on load. Type anything and watch where the seams fall. |
| 2 | Embeddings | Eight hand-labelled dimensions, with cosine similarity, vector arithmetic, and a 2D map projected by PCA computed in the page. |
| 3 | Self-attention | Five heads with genuine query/key projections. The scores are `q·k/√d`, the weights are a softmax, the mask is `-Infinity` on the future. |
| 4 | The stack | A block diagram plus a parameter budget computed from the shape controls. The GPT-2 preset lands on 124M because the formulas are the real ones. |
| 5 | Prediction | Logits, softmax, temperature, top-k and top-p, over the whole vocabulary of a model trained on this page's corpus. |
| 6 | Generation | The autoregressive loop, one token at a time, with the candidate distribution visible at every step. |
| 7 | Training | Gradient descent, live: a bigram softmax over characters, ~700 parameters, trained by SGD while you watch the loss fall. |
| 8 | Consequences | Which well-known model behaviours fall out of which step, plus a glossary. |

## How it is built

No dependencies, no build step, no network requests. Plain ES modules, one
stylesheet, and a `<canvas>` for the training plots.

```
index.html
css/style.css
js/
  util.js         DOM/SVG helpers, softmax, seeded RNG, sliders and segmented controls
  corpus.js       the ~400 words everything is trained on
  tokenizer.js    BPE training and encoding
  ngram.js        word-level trigram model with backoff, and the sampler
  sec-*.js        one module per section
  main.js         theme, nav, reveal, and mounting
```

The models are deliberately tiny. The point is that each one is the real
algorithm at a scale you can watch, not a picture of the algorithm.

## Running locally

Any static server will do — ES modules will not load over `file://`.

```bash
python3 -m http.server 8766
```

Then open <http://localhost:8766>.

## Licence

MIT. See [LICENSE](LICENSE).
