# Running Deux inside Claude (covered by your Claude subscription)

Claude artifacts can call Claude's AI **without any API key** — usage counts
against the Claude subscription you already pay for. Artifacts are built by
Claude writing code in a chat, so this route runs a **compact edition** of
Deux ("Deux Pocket") that Claude generates from the prompt below. The full
4,300-line app in this repo is too large to be retyped into an artifact —
that's why the GitHub Pages version and this artifact edition coexist:

| | GitHub Pages version (this repo) | Deux Pocket (Claude artifact) |
|---|---|---|
| AI features | Need an API key (Settings) | **Free with your subscription** |
| Content | Full app, big offline bank | Compact, AI-generated on the fly |
| Where it runs | Any browser | Inside the Claude app / claude.ai |
| Progress saved | Per device | Per Claude account |

## Setup (once, ~2 minutes, works on the phone)

1. Open the **Claude app** (or claude.ai), start a **new chat**.
2. Copy the entire prompt below and send it.
3. Wait for the artifact to build, then tap it to open. Try a feature that
   needs AI (e.g. start a conversation) — if it replies in French, you're
   running on your subscription.
4. Tap **Publish/Share → save it** so it appears in your artifacts list —
   reopen it any time from claude.ai/artifacts without rebuilding.
   Progress persists per Claude account, so using it signed into your
   account keeps Stuart's and Carol's profiles just like the website does.

If the first build has a bug, just tell Claude in the same chat ("the X
button does nothing — fix it") — iterating is the normal artifact workflow.

## The prompt to paste

```
Build a complete single-file React artifact called "Deux Pocket" — a French-learning app for two people sharing one device: Stuart and Carol. Keep it compact enough to fit comfortably in one artifact; prioritise a flawless working core over breadth. No placeholder buttons: everything rendered must work.

Environment: persist ALL state via the async window.storage API (get/set), namespaced per user (e.g. "stuart:srs"); never use localStorage. For AI, call fetch("https://api.anthropic.com/v1/messages") with model "claude-sonnet-4-6", max_tokens 1000, no API key. For AI calls that must be parsed, instruct JSON-only output in the system prompt, strip ```json fences, parse in try/catch, and always have a small hardcoded fallback so nothing breaks offline. Use speechSynthesis with a fr-FR voice for all audio (with a speed slider), and webkitSpeechRecognition (lang fr-FR) for speaking where supported, falling back to self-assessment buttons. Mobile-first: large touch targets, one-handed use.

Users: a profile picker with two tiles. Carol is a near-beginner (A0/A1): warm amber theme, English subtitles on by default, gentle tone, tappable suggested replies in conversation. Stuart is low-intermediate (A2/B1) whose weakness is understanding SPOKEN French: focused indigo theme, no suggested replies, strict error correction.

Features (all AI-generated content, graded to the active user's level):
1. SRS flashcards with a real SM-2 scheduler (ease factor, interval, reps, quality 0-5). Reviews demand PRODUCTION: show English, user types or speaks the French, auto-check (accent-insensitive), then grade Again/Hard/Good/Easy. New words flow in from every other feature. Show due count on home.
2. Listening trainer (Stuart's flagship): AI writes a short spoken-style French clip at his level → he listens via TTS without text → types what he heard → word-level diff highlights exactly which words he missed → AI explains, for each missed segment, WHICH connected-speech phenomenon hid it (liaison, elision, enchaînement, reduction) and how it actually sounds ("les amis" → "lay-za-mee") → he retells the clip in French and gets honest corrections, which go to SRS.
3. Conversation partner: pick a scenario (café, market, directions, hotel, free talk) and chat in French at level. In Stuart's strict mode, name every real error including word-order errors — never silently fix what he said. End button produces a correction report (you said → fix → why) with one tap to add all fixes to SRS.
4. "Make me an exercise": user picks focus (weak spot, a grammar point, or pastes any French text) and format (cloze, translation, dictation, reorder); AI generates a 3-item drill, checks answers, explains mistakes.
5. Mini lesson for Carol: AI generates a short pattern-based micro-lesson (explain simply from English → 3 blocked practice items she must produce → one real-life task with corrective feedback) on a scenario she picks; new vocab goes to her SRS with emoji.
6. Honest dashboard: five skill bars (listening/speaking/reading/writing/vocab) blending level with measured accuracy, mature-word count, streak that only counts when a real session is completed (never one tap), and realistic CEFR hour framing (A1 ≈ 80-100h, A2 ≈ 180-200h, B1 ≈ 350-400h). Never promise fast fluency.

Design: distinctive and clean (serif display headings, stone background, rounded cards), never default-grey. Verify the full loop for both users before finishing.
```

## Honest expectations

- Deux Pocket is a **fresh build by Claude in your chat** — a leaner sibling
  of this repo's app, not a copy. First version may need an iteration or two.
- Artifact AI usage draws on your Claude plan's usage allowance (generous on
  Max; a lesson uses roughly as much as a short conversation with Claude).
- The full-featured Pages site (with the big offline bank, phonics course,
  12 lessons, grammar drills and podcast ladder) remains at
  https://stumacmar.github.io/French-Learning-App/ — the two run happily in
  parallel; the SRS decks are separate per platform.
