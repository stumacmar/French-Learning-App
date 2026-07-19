# Deux — original build specification

This is the specification the repo's app (`deux.jsx`) was built against. It is
kept here for reference, and so future editions (e.g. a Claude-artifact
rebuild — see `ARTIFACT-SETUP.md`) can be generated from the same source of
truth.

## Goal

A complete, working, single-page web application called **Deux** — a
French-learning app for exactly two users (Stuart and Carol) who share one
device and one household. Production quality; no placeholders; every
described feature functions. Grounded in evidence-based SLA design — not a
generic gamified clone. The differentiators: a listening-comprehension
trainer for connected speech, an SRS core built on active recall, and an
honest (not vanity) progress model.

## Tech constraints

- Single self-contained build: React in one file, Tailwind core utilities,
  default export, no required props.
- No localStorage/sessionStorage in the app itself. State in React during the
  session; persistence via the async `window.storage.get/set/delete/list` KV
  API, wrapped in try/catch, every key namespaced per user (`stuart:srs`).
- AI via `fetch("https://api.anthropic.com/v1/messages")`, model
  `claude-sonnet-4-6`, `max_tokens: 1000`, no API key (environment-provided).
  JSON-only instructions for parseable output; strip code fences; parse
  defensively; deterministic fallbacks everywhere.
- Web Speech API: `speechSynthesis` (fr-FR) for all audio;
  `SpeechRecognition`/`webkitSpeechRecognition` (fr-FR) for speaking, with
  graceful degradation to self-assessment. Never block a lesson on an
  unsupported API.
- iPhone Safari first: touch targets ≥44px, one-handed, responsive.

## The two learners

- **Carol — near-total beginner (A0 → A1).** Knows only hello/goodbye.
  Needs survival vocabulary, the French sound system (phonics), heavy
  scaffolding, generous SRS repetition, dual subtitles by default, low-stakes
  guided speaking before free conversation, blocked practice, a colourful
  encouraging tone, short sessions. Foundations track from A0.
- **Stuart — low-intermediate (A2/B1) with a listening bottleneck.** Can
  handle basic transactional conversation but cannot understand spoken
  French. Skips the beginner grind; Listening-First track. Core weakness is
  connected speech — liaison, elision, enchaînement — so spoken French
  misaligns word boundaries versus the written form he knows. Ear-training
  first, plus interleaved practice of half-known structures.
- A short adaptive placement check (listening + grammar) runs on first open
  of each profile and can be re-run from settings.

## SLA principles baked into every feature

1. Active recall + spaced repetition as the memory backbone: SM-2 scheduler
   (ease factor, interval, repetitions, quality 0–5), production-first cards.
2. Comprehensible input at i+1, AI-graded to the user's level.
3. Input is necessary but not sufficient: every input activity ends in
   forced output with corrective feedback.
4. Block then interleave; drills are flagged with their mode.
5. Reading-while-listening early, then remove the text.
6. Dual coding for concrete vocabulary (emoji/glyph imagery).
7. Desirable difficulty over easy dopamine: no mechanic lets a user "win"
   by tapping the easy option.
8. Honest progress: CEFR-referenced five-skill breakdown, not vanity XP.

## Features

**Shared engine:** SM-2 SRS with production reviews and auto-added words from
every activity; a deterministic CEFR curriculum skeleton (A0→B1+) of
task-based scenarios (explanation → blocked practice → output task → vocab to
SRS) with AI-freshened content; a reading-while-listening library (synced
TTS, tap-any-word lookup → SRS, speed control, hide-text toggle); an AI
conversation partner (scenario library + free talk, lenient/normal/strict —
strict catches every error including meaning-changing word order, never
silently fixes, ends with a correction report pushed to SRS; beginner mode
offers tappable suggested replies); an on-demand exercise generator (cloze,
dictation, translation, reorder, roleplay from any focus or pasted text); a
habit layer subordinate to learning (flexible streak with a real completion
threshold, one free weekly repair, gentle reminders, a daily quick win — no
punitive dark patterns); an honest dashboard (CEFR level, five-skill bars,
mature-word count, listening-accuracy trend, next recommended action, an
optional encouraging two-person view).

**Carol's Foundations track:** French phonics first (nasal vowels, the R,
silent letters, graphemes → sounds) with listen-and-repeat; high-frequency
survival vocabulary in dual-coded SRS; Michel-Thomas-style pattern-building
mini-intros; dual subtitles on by default; blocked practice; guided speaking
before free conversation; warm, short sessions.

**Stuart's Listening-First track (flagship):** a Listening Comprehension
Trainer implementing: clip → listen unaided → typed dictation → automatic
word-level diff → AI gap analysis naming the specific liaison / elision /
enchaînement / reduction that hid each missed boundary (with
English-friendly "sounds like" renderings) → chunked shadowing with
speech-recognition feedback → French retell with honest corrections → narrow
listening (re-listen tracking). Connected-speech mini-lessons; a graded →
authentic ladder (~90–95% comprehensible start, speed control, dual → FR-only
→ no subtitles) gated on dictation accuracy; a linked podcast ladder (News in
Slow French → Coffee Break French → InnerFrench → RFI Journal en français
facile); interleaved grammar drills; strict conversation mode.

## Content rules

All French content generated at runtime against deterministic skeletons and
difficulty parameters defined in code; JSON-only AI output parsed
defensively; TTS audio with per-user/clip speed control; never claim
"fluency in X weeks" — use realistic CEFR hour ranges (~80–100h A1,
~180–200h A2, ~350–400h B1).

## Acceptance flows

- Carol: picker → placement (lands A0/A1) → Foundations lesson with phonics
  and a task → new words in SRS → production review → conversation with
  suggested replies + correction report → dashboard updates honestly.
- Stuart: picker → placement (skips to A2/B1) → full trainer loop (listen →
  dictation → diff → gap analysis → shadow → retell) → interleaved grammar
  drill → strict conversation with correction report → exercise generator →
  honest dashboard.
