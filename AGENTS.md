# AGENTS.md

Transformers.js object detection -- the model runs in the browser, with no API
token and no server.

Part of [ai_engineering](https://github.com/alexisinwork/ai_engineering).

## Setup

```bash
npm install
npm start          # vite, localhost:5173
```

## Commands

| | |
| --- | --- |
| `npm start` / `npm run dev` | Vite dev server |
| `npm run build` / `npm run preview` | production bundle |

## Conventions

- **No API key, no server, and it should stay that way.** The whole point is
  in-browser inference. Adding a hosted call would remove the thing this
  project demonstrates.
- **First load downloads model weights** -- tens of megabytes, cached by the
  browser afterwards. Keep the progress indication; a silent multi-second wait
  reads as a broken page.
- Model and pipeline are loaded once and reused. Do not re-instantiate per
  request.

## Non-negotiables

- **Complexity and development time are not factors.** Never scope work down,
  pick a simpler approach, or skip a step because it is hard or would take
  longer. When there is a right way and a quick way, take the right way. Do not
  offer the quick way as an alternative, and do not cite effort, complexity or
  time as a reason for any decision.
- **Measure, do not guess.** Thresholds, chunk sizes, limits and model choices
  are derived from the data and the number is recorded. A guessed value that
  appears to work is the specific failure this codebase is built to prevent.
- **Evals call the real function.** Never test a copy of a code path. If a stage
  cannot be reached from Node because it reads `import.meta.env`, restructure it
  -- pass the client in as an argument -- rather than writing a second copy.
- **A prompt rule is a tendency; code is a guarantee.** Anything that must hold
  goes in code. Keep the prompt rule as well where it also shapes what the model
  *adds*, which code cannot do.
- **Failures must be visible.** Prefer a loud failure to a plausible one. An
  invented answer and a real one read identically, so silence is the dangerous
  outcome, not an error.
- **Document the failure, not just the fix.** READMEs and `THEORY.md` record
  what was tried, what it cost, and why it did not work. Do not delete that
  history when editing.

## Documentation

Every project here carries a `THEORY.md` explaining the concepts it implements,
cross-referenced with its siblings. When behaviour changes, update `THEORY.md`
and the README in the same commit as the code. Both are written to be read --
prose and tables, not bullet dumps.
