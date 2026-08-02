# Theory — On-Device Inference, ONNX, and Computer Vision Tasks

Every other AI project in this repo sends a request to someone else's GPU. This
one does not. The model runs **inside the browser tab**, on the user's own CPU,
with no API key, no server, and no per-request cost. This file explains how
that is possible and what it costs. The code is `index.js`.

---

## 1. The inversion

```js
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.8.0'
```

That single import is the whole architectural difference. There is no
`.env`, no `server.js`, no key to protect — because there is no request to
authenticate. The model weights are downloaded to the browser and executed
locally.

| | Server-side API (the other projects) | In-browser (this one) |
| --- | --- | --- |
| Where inference runs | Provider's GPU | User's CPU |
| Credential | API key, kept server-side | None |
| Per-request cost | Per token / per image | Zero |
| Data leaves the device | Yes | **No** |
| First-use cost | One HTTP round trip | ~30 MB download |
| Model size ceiling | Effectively unlimited | Tens of MB, realistically |
| Works offline | No | Yes, after first load |
| Scaling cost | Grows with users | Flat — users bring the compute |

The last row is easy to skip past and is often the deciding factor. A server-
side feature gets more expensive with every user. This one does not: each user
contributes the hardware that serves them.

---

## 2. Transformers.js and ONNX

**Transformers.js** is a JavaScript port of Hugging Face's Python
`transformers` library, with a deliberately familiar API — `pipeline(task,
model)` is the same call you would write in Python. It runs models via **ONNX
Runtime Web**.

**ONNX** (Open Neural Network Exchange) is a portable format for trained
models: a computation graph plus weights, independent of the framework that
produced it. Train in PyTorch, export to ONNX, run anywhere there is an ONNX
runtime — including a browser.

This is why the model ID is `Xenova/yolos-tiny` rather than the original
`hustvl/yolos-tiny`. The `Xenova` namespace on the Hugging Face Hub holds
ONNX-converted, browser-ready versions of upstream models. **You cannot point
Transformers.js at an arbitrary Hub model** — it must have been converted
first. That constraint is the first thing to check when a model "doesn't work"
in the browser.

Under ONNX Runtime Web, the graph executes through a **WebAssembly** backend:
the runtime is compiled C++ shipped as `.wasm` and run at near-native speed in
the tab's sandbox. In Transformers.js v2 (used here) that is the practical
path. v3 adds a **WebGPU** backend, which reaches the GPU and is dramatically
faster for larger models — worth knowing about when this project is revisited.

### Quantization makes it fit

Serving a model over the network to a browser puts hard pressure on size, so
browser builds are **quantized** — weights stored at 8 bits instead of 32,
shrinking the download roughly 4× for a small accuracy cost. Combined with
choosing a "tiny" variant in the first place, that is what brings a real object
detector down to a downloadable size.
`../OllamaPracticeMistral/THEORY.md` covers quantization in more detail; it is
the same technique that makes a 7B language model run on a laptop.

---

## 3. The cold-start problem, and how the code handles it

The honest cost of in-browser inference is the first run. Two separate
downloads:

- the model weights (~20 MB for `yolos-tiny`), and
- the ONNX Runtime WASM binary (~10 MB), fetched by the runtime itself.

`index.js` calls out that the progress callback only sees the first:

> Note this covers the Hub files only — the ~10MB onnxruntime WASM is fetched
> by ORT itself and never reaches this callback, which is why the first stretch
> stays quiet.

That is a real UX problem — a progress bar that sits at nothing for several
seconds — and knowing *why* is the difference between fixing it and guessing.

### Lazy loading

```js
let detectorPromise = null

function getDetector() {
    detectorPromise ??= pipeline('object-detection', 'Xenova/yolos-tiny', {...})
    return detectorPromise
}
```

The pipeline is built on the first click, not at import. The comment explains
the alternative:

> Loading it up front blocks this whole module, so the button stays disabled
> and the page looks frozen for the ~20MB the first run has to fetch.

Top-level `await` in a module blocks that module's evaluation. A 20 MB download
in module scope means a page that appears broken. Defer expensive
initialization until the user asks for the feature.

The `??=` also memoizes: subsequent clicks reuse the same promise, so the model
loads exactly once per page.

### The poisoned-promise bug

```js
.catch(error => {
    // Don't let one failed load poison every later click.
    detectorPromise = null
    throw error
})
```

This is a genuinely subtle bug and worth internalizing beyond ML. **A rejected
promise stays rejected forever.** Cache a promise for memoization, and a
one-time network failure is cached too — every later retry re-throws the
original error instantly, without touching the network. The user clicks
"retry", sees the same failure at impossible speed, and concludes the app is
broken.

Clearing the cache on rejection makes the next call actually retry. Any
promise-memoization pattern needs this.

### Caching across sessions

Weights are fetched over HTTP and land in the browser's HTTP cache, so a
returning visitor usually skips the download. It is not guaranteed — cache
eviction is the browser's call — which is why the loading path must stay
correct rather than being treated as one-time setup.

### One configuration gotcha

```js
env.allowLocalModels = false
```

By default the library checks for a local `/models/` copy before hitting the
Hub. Vite's dev server answers *any* unknown path with `index.html` and a 200,
so the check "succeeds" and `JSON.parse` then chokes on `<!DOCTYPE html>`. A
good example of a framework default colliding with a library default to produce
a baffling error — the fix is one line once you know, and unfindable if you
don't.

---

## 4. Object detection as a task

Vision tasks form a hierarchy, and picking the right one shapes everything
downstream:

| Task | Question answered | Output |
| --- | --- | --- |
| Image classification | "What is this a picture of?" | One label + score |
| **Object detection** | "What is in it, and where?" | Many `{label, score, box}` |
| Semantic segmentation | "Which pixels belong to which class?" | Per-pixel class map |
| Instance segmentation | "Which pixels belong to which *object*?" | Per-object pixel mask |

Detection is the middle rung: multiple objects, each localized by a rectangle.
That is why the output is an array and why each entry carries a box.

**YOLOS** ("You Only Look at One Sequence") is a Vision Transformer adapted for
detection. Rather than a purpose-built detection architecture, it treats the
image as a sequence of patches — the same trick that lets transformers handle
text — and predicts a fixed set of boxes. `yolos-tiny` is the smallest variant,
which is exactly why it is viable in a browser.

### Confidence thresholds

```js
const detectedObjects = await detector(image.src, {
    threshold: 0.95,
    percentage: true
})
```

The model does not output "5 objects." It outputs many candidate boxes, each
with a confidence score, and **the threshold is where you draw the line**:

- **High (0.95)** — few boxes, high precision, misses real objects. Good for a
  clean demo.
- **Low (0.3)** — catches more real objects, admits false positives.

This is the precision/recall trade, and it is a product decision, not a
technical default. A safety system wants high recall; a clean visualization
wants high precision. There is no correct value, only a correct value *for the
use case*.

**`percentage: true`** returns coordinates as 0–1 fractions instead of pixels.
The drawing code then uses `100 * xmin + '%'` and CSS positioning, so overlays
stay aligned when the image is resized by responsive layout. Absolute pixel
coordinates would need recomputing on every resize. A small choice that removes
a whole class of bug.

Also note the cleanup:

```js
imageContainer.querySelectorAll('.bounding-box').forEach(box => box.remove())
```

Detection results are rendered as DOM elements; without removing the previous
run's boxes they accumulate on every click. Stateful UI over a stateless
inference call needs explicit teardown.

---

## 5. When to run on-device

**Good fit:**

- **Privacy-sensitive input.** Medical images, documents, camera feeds. The
  strongest guarantee is not a policy — it is that the data never left.
- **High-frequency, low-value inference.** Live camera classification at 30 fps
  is unaffordable per-call and free locally.
- **Offline or unreliable networks.**
- **Latency-critical interaction.** No round trip.
- **Cost-sensitive scale.** Free users bring their own compute.

**Poor fit:**

- **Large models.** Small vision models and embedding models are practical;
  frontier language models are not — they are hundreds of times too large to
  ship to a tab.
- **Cold-start-sensitive first impressions.** A 30 MB download before the first
  result is a real drop-off risk.
- **Guaranteed performance.** You inherit whatever device the user has. A
  five-year-old phone and a workstation run the same code very differently.
- **Model secrecy.** Shipping weights to the client means shipping the weights.
  Anyone can take them.

The practical middle ground most products land on: **on-device for the cheap,
frequent, privacy-relevant step; server-side for the heavy one.** Detect or
embed locally, and send only what needs a frontier model.

---

## 6. Related reading in this repo

- `../HuggingFaceDemo/THEORY.md` — the same models, called as a hosted API
  instead, plus the task taxonomy and the architecture families.
- `../OllamaPracticeMistral/THEORY.md` — the other end of local inference:
  running a full language model on your own machine, and quantization in depth.
- `../EmbeddingsAndVectorDB/THEORY.md` — embedding models are small enough to
  run in the browser too, which makes fully client-side semantic search
  possible.
