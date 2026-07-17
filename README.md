# Deux — French, for the two of you

A single-file React app that teaches French to exactly two learners sharing one
device: **Carol** (near-total beginner, A0 → A1) and **Stuart** (low-intermediate,
A2/B1, with a spoken-French listening bottleneck). Built on second-language-
acquisition research rather than gamified vanity metrics.

The whole app lives in **`deux.jsx`** (default-exported `DeuxApp` component,
Tailwind core utilities, no required props).

## Run it

The app is designed for a host environment that provides:

- an async KV store at `window.storage.{get,set,delete,list}` (all persistence
  is namespaced per user: `stuart:srs`, `carol:progress`, …),
- a key-less Anthropic API proxy at `https://api.anthropic.com/v1/messages`
  (model `claude-sonnet-4-6`),
- the browser Web Speech API (`speechSynthesis` for fr-FR audio,
  `SpeechRecognition` for speaking — optional, degrades to self-assessment).

For standalone hosting (GitHub Pages, any static host) and local development,
`index.html` loads a prebuilt, fully self-contained build — `app.bundle.js`
(React inlined, minified ESM) and `app.css` (statically generated Tailwind).
No CDNs, no runtime transpilation, instant boot screen. Serve over HTTP:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

After editing `deux.jsx`, regenerate the build artifacts:

```bash
npx esbuild deux.jsx --bundle --minify --loader:.jsx=jsx --format=esm --outfile=app.bundle.js
npx tailwindcss@3.4.5 -i tw.in.css -o app.css --minify   # config content: ["./deux.jsx"]
```

Notes for standalone use: the harness backs `window.storage` with
localStorage (namespaced under `deux::`) so progress survives refreshes —
the app itself only ever talks to the async `window.storage` API. French TTS
quality depends on the browser's installed voices (iPhone Safari and Chrome
both ship fr-FR voices).

## AI content engine (standalone hosting)

Out of the box the hosted app runs on its **built-in offline bank**: 40
listening clips (5 per ladder rung), 30 reading passages, 12 lessons, 40
grammar drill items plus 20 extras, 28 exercise-generator items, 16 placement
questions, and ~70 phonics/connected-speech examples. Every feature works
without any network access.

To unlock **unlimited fresh content** (AI-generated clips, passages, drills,
free conversation with correction reports), open **Settings → AI content
engine** and either:

1. **Paste an Anthropic API key** (console.anthropic.com). It is stored only
   in that device's browser storage and sent only to Anthropic. Simplest —
   fine for a personal device you don't share.
2. **Deploy the included Cloudflare Worker** (`cloudflare-worker.js`, free
   tier, ~5 minutes — instructions in the file header) and paste the worker
   URL instead. The key then lives server-side and never touches the device.
   Set the worker's `ALLOWED_ORIGIN` to your GitHub Pages origin so nobody
   else can spend your credit.

Use "Save & test" to confirm the connection. If the AI is unreachable at any
moment, every feature silently falls back to the built-in bank.

## What's inside

**Shared engine (both profiles)**
- **SRS core** — SM-2 scheduler (ease factor, interval, reps, lapses), a
  production-first review UI (type or speak the French before grading), and a
  per-user due queue. New words flow in automatically from lessons, tapped
  words in reading, conversation corrections and missed exercises.
- **Placement check** — 7 adaptive listening + grammar questions on first open
  of each profile (re-runnable from Settings); sets CEFR level and track.
- **Reading-while-listening library** — AI-generated passages at level, synced
  TTS with speed control, tap-any-word lookup → SRS, dual-subtitle toggle, a
  hide-text mode to test the ear, and a forced-output comprehension question.
- **AI conversation partner** — scenario library + free talk, three strictness
  levels. In normal/strict mode errors (including meaning-changing word-order
  errors) are named, never silently fixed. Every conversation ends with a
  correction report whose fixes land in the SRS deck. Beginner mode offers
  tappable suggested replies; Stuart's does not.
- **On-demand exercise generator** — pick a focus (auto-detected weak spot,
  current unit, a grammar point, or any pasted French text) and a format
  (cloze / translation / dictation / reorder); checked and explained.
- **Habit layer** — flexible streak that only counts real learning thresholds,
  one free streak repair per week, optional gentle reminder, a 3-card quick
  win. No punitive mechanics.
- **Honest dashboard** — CEFR level, five-skill bars (listening / speaking /
  reading / writing / vocabulary) blending placement with measured accuracy,
  mature-word count, dictation-accuracy trend, realistic CEFR hour ranges, and
  an optional non-competitive two-person view.

**Carol's Foundations track** — French phonics lessons (nasal vowels, the R,
silent letters, u/ou, graphemes, linking) with listen-and-repeat; 8 task-based
units (café, numbers, introductions, directions, restaurant, daily routine,
hotel problem), each: pattern-based explanation → blocked practice → output
task with corrective feedback → vocab to SRS. Dual subtitles on by default.

**Stuart's Listening-First track** — the flagship **Listening Comprehension
Trainer**: listen unaided → typed dictation → word-level auto-diff → AI
connected-speech gap analysis (which liaison / elision / enchaînement /
reduction hid each missed boundary) → chunked shadowing with speech-recognition
feedback → French retell with honest correction → narrow-listening nudges. An
8-rung graded ladder (dual subs → French-only → none; slow → natural speed)
advances at ≥80% dictation accuracy, plus six connected-speech mini-lessons,
interleaved grammar drills (blocked → interleaved once ≥75% over 8 items), a
strict-mode conversation partner, and a linked podcast ladder (News in Slow
French → Coffee Break French → InnerFrench → RFI Journal en français facile).

## Design commitments

Sequencing is deterministic (curriculum skeletons, drill banks and difficulty
parameters live in code); the AI only freshens content against them, always
returning JSON that is parsed defensively with static fallbacks. Progress is
framed honestly — no "fluent in 3 weeks", real CEFR hour ranges (~80–100 h to
A1, ~180–200 h to A2, ~350–400 h to B1).
