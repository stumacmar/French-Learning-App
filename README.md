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

For local development, `index.html` is a thin harness (React + Babel via CDN,
in-memory `window.storage` shim). Serve over HTTP:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Notes for local use: without the environment's API proxy, AI calls fail
gracefully — every AI feature has a built-in deterministic fallback, so the
full loop still works offline. French TTS quality depends on the browser's
installed voices (iPhone Safari and Chrome both ship fr-FR voices).

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
