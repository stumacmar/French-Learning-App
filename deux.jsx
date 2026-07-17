/*
 * ============================================================================
 *  DEUX — a French-learning app for two: Stuart & Carol
 * ============================================================================
 *  Single-file React app. Design pillars (evidence-based, see README):
 *   - Active recall + SM-2 spaced repetition (production, not recognition)
 *   - Comprehensible input at i+1, generated at runtime against a
 *     deterministic curriculum skeleton defined below
 *   - Input -> forced output + corrective feedback on every activity
 *   - Blocked practice first, then interleaving of confusables
 *   - Reading-while-listening, then remove the text
 *   - Dual coding for concrete vocabulary
 *   - Desirable difficulty; honest CEFR-referenced progress (no vanity XP)
 *   - Flagship: Listening Comprehension Trainer for connected speech
 *     (liaison / elision / enchaînement) — Stuart's track
 *
 *  Environment contract:
 *   - Persistence via async window.storage.{get,set,delete,list} (namespaced
 *     per user, e.g. "stuart:srs"). Falls back to in-memory if absent.
 *   - AI via fetch("https://api.anthropic.com/v1/messages"),
 *     model "claude-sonnet-4-6", no API key (injected by environment).
 *   - Audio via Web Speech API (speechSynthesis fr-FR); speech input via
 *     SpeechRecognition where available, otherwise self-assessment.
 * ============================================================================
 */
import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";

/* ----------------------------- tiny utilities ---------------------------- */

const uid = () => Math.random().toString(36).slice(2, 10) + Math.random().toString(36).slice(2, 6);
const pad2 = (n) => String(n).padStart(2, "0");
const todayKey = (d = new Date()) => `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
const dateKeyOffset = (offsetDays) => { const d = new Date(); d.setDate(d.getDate() + offsetDays); return todayKey(d); };
const weekKey = (d = new Date()) => Math.floor(d.getTime() / (7 * 86400000));
const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
const shuffle = (arr) => { const a = [...arr]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const stripAccents = (s) => (s || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "");
const norm = (s) =>
  stripAccents((s || "").toLowerCase())
    .replace(/[’‘]/g, "'")
    .replace(/[^a-z0-9'\- ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const tokenize = (s) =>
  (s || "")
    .replace(/[’‘]/g, "'")
    .split(/\s+/)
    .map((w) => w.replace(/^[^A-Za-zÀ-ÿ0-9']+|[^A-Za-zÀ-ÿ0-9']+$/g, ""))
    .filter(Boolean);

function levenshtein(a, b) {
  a = norm(a); b = norm(b);
  if (a === b) return 0;
  const m = a.length, n = b.length;
  if (!m) return n; if (!n) return m;
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i++) {
    const cur = [i];
    for (let j = 1; j <= n; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[n];
}

/* Answer checking: exact -> "close" (small edit distance) -> wrong */
function checkAnswer(expected, given) {
  const e = norm(expected), g = norm(given);
  if (!g) return "empty";
  if (e === g) return "exact";
  const dist = levenshtein(e, g);
  if (dist <= Math.max(1, Math.floor(e.length * 0.15))) return "close";
  return "wrong";
}

/* Word-level diff (LCS) between a target transcript and a user attempt.
   Returns target tokens flagged hit/miss, extra tokens the user added,
   and an accuracy percentage. Used by the dictation pass. */
function wordDiff(target, attempt) {
  // Display tokens keep their original punctuation; matching runs on the
  // normalized forms. Punctuation-only tokens render neutral (hit).
  const rawWords = (target || "").replace(/[’‘]/g, "'").split(/\s+/).filter(Boolean);
  const pairs = rawWords.map((raw) => ({ raw, clean: norm(raw) }));
  const tn = pairs.filter((p) => p.clean).map((p) => p.clean);
  const A = tokenize(attempt).map(norm).filter(Boolean);
  const m = tn.length, n = A.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--)
    for (let j = n - 1; j >= 0; j--)
      dp[i][j] = tn[i] === A[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const hits = new Array(m).fill(false);
  const extras = [];
  let i = 0, j = 0;
  while (i < m && j < n) {
    if (tn[i] === A[j]) { hits[i] = true; i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
    else { extras.push(A[j]); j++; }
  }
  while (j < n) { extras.push(A[j]); j++; }
  let k = 0;
  const targetTokens = pairs.map((p) => (p.clean ? { w: p.raw, hit: hits[k++] } : { w: p.raw, hit: true }));
  const hitCount = hits.filter(Boolean).length;
  const acc = m ? Math.round((hitCount / m) * 100) : 0;
  return { targetTokens, extras, acc };
}

/* Group consecutive missed words into segments with one word of context,
   for the connected-speech gap analysis. */
function missedSegments(targetTokens) {
  const segs = [];
  let cur = null;
  targetTokens.forEach((t, idx) => {
    if (!t.hit) {
      if (!cur) cur = { start: Math.max(0, idx - 1), end: idx };
      else cur.end = idx;
    } else if (cur) {
      cur.end = Math.min(targetTokens.length - 1, cur.end + 1);
      segs.push(cur); cur = null;
    }
  });
  if (cur) segs.push(cur);
  return segs.map((s) => targetTokens.slice(s.start, s.end + 1).map((t) => t.w).join(" ")).slice(0, 5);
}

/* Loose similarity 0..100 between two spoken strings (token overlap). */
function similarity(a, b) {
  const A = tokenize(a).map(norm), B = tokenize(b).map(norm);
  if (!A.length || !B.length) return 0;
  const setB = [...B];
  let hits = 0;
  for (const w of A) {
    const k = setB.indexOf(w);
    if (k >= 0) { hits++; setB.splice(k, 1); }
  }
  return Math.round((2 * hits / (A.length + B.length)) * 100);
}

/* Split French text into speakable chunks (no lookbehind — iOS Safari safe) */
function chunkText(text) {
  const parts = (text.match(/[^.!?;,]+[.!?;,]*/g) || [text]).map((s) => s.trim()).filter(Boolean);
  // merge tiny fragments into the previous chunk
  const out = [];
  for (const p of parts) {
    if (out.length && tokenize(p).length <= 2) out[out.length - 1] += " " + p;
    else out.push(p);
  }
  // a tiny leading fragment ("Ce matin,") merges forward instead of standing alone
  if (out.length > 1 && tokenize(out[0]).length <= 2) {
    out[1] = out[0] + " " + out[1];
    out.shift();
  }
  return out;
}

/* ------------------------------ persistence ------------------------------ */
/* Async KV wrapper around window.storage; in-memory fallback so the app
   never breaks. Every key is namespaced by user id (e.g. "carol:srs"). */

const _mem = {};
const store = {
  async get(key) {
    try {
      if (typeof window !== "undefined" && window.storage && window.storage.get) {
        const v = await window.storage.get(key);
        if (v === null || v === undefined) return null;
        const raw = typeof v === "object" && v !== null && "value" in v ? v.value : v;
        if (typeof raw === "string") { try { return JSON.parse(raw); } catch { return raw; } }
        return raw;
      }
    } catch (e) { /* fall through to memory */ }
    return key in _mem ? _mem[key] : null;
  },
  async set(key, value) {
    _mem[key] = value;
    try {
      if (typeof window !== "undefined" && window.storage && window.storage.set) {
        await window.storage.set(key, JSON.stringify(value));
      }
    } catch (e) { /* memory copy still holds */ }
  },
  async del(key) {
    delete _mem[key];
    try {
      if (typeof window !== "undefined" && window.storage && window.storage.delete) {
        await window.storage.delete(key);
      }
    } catch (e) { /* ignore */ }
  },
};

/* --------------------------------- AI layer ------------------------------ */
/* In the managed environment the default endpoint works with no key.
   For standalone hosting (e.g. GitHub Pages) the user can configure, in
   Settings, EITHER their own Anthropic API key (sent directly from the
   browser — stays on this device) OR a personal proxy URL (see
   cloudflare-worker.js) that injects the key server-side. Stored under the
   device-level "deux:ai" key, never in the repo or the page. */

let AI_CFG = { key: "", url: "" };
function setAiConfig(cfg) { AI_CFG = { key: ((cfg && cfg.key) || "").trim(), url: ((cfg && cfg.url) || "").trim() }; }
function aiConfigured() { return !!(AI_CFG.key || AI_CFG.url); }

async function callClaude(system, messages, maxTokens = 1000) {
  const url = AI_CFG.url || "https://api.anthropic.com/v1/messages";
  const headers = { "content-type": "application/json", "anthropic-version": "2023-06-01" };
  if (AI_CFG.key && !AI_CFG.url) {
    headers["x-api-key"] = AI_CFG.key;
    headers["anthropic-dangerous-direct-browser-access"] = "true";
  }
  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({ model: "claude-sonnet-4-6", max_tokens: maxTokens, system, messages }),
  });
  if (!res.ok) throw new Error("API error " + res.status);
  const data = await res.json();
  return (data.content || []).map((b) => b.text || "").join("");
}

function extractJSON(text) {
  let t = (text || "").trim();
  t = t.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  const io = t.indexOf("{"), ia = t.indexOf("[");
  let start = -1;
  if (io >= 0 && ia >= 0) start = Math.min(io, ia);
  else start = Math.max(io, ia);
  if (start < 0) return t;
  const end = Math.max(t.lastIndexOf("}"), t.lastIndexOf("]"));
  return end > start ? t.slice(start, end + 1) : t.slice(start);
}

/* Ask the model for structured data; always returns fallback on any failure. */
async function aiJSON(system, prompt, fallback = null, maxTokens = 1000) {
  try {
    const text = await callClaude(
      system + " Respond with ONLY valid JSON. No preamble, no markdown, no code fences.",
      [{ role: "user", content: prompt }],
      maxTokens
    );
    const parsed = JSON.parse(extractJSON(text));
    return parsed === null || parsed === undefined ? fallback : parsed;
  } catch (e) {
    return fallback;
  }
}

/* ------------------------------ speech (out) ------------------------------ */

let _frVoice = null;
function pickFrenchVoice() {
  try {
    const vs = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
    _frVoice =
      vs.find((v) => v.lang === "fr-FR" && /amelie|thomas|audrey|marie|google fran/i.test(v.name)) ||
      vs.find((v) => v.lang === "fr-FR") ||
      vs.find((v) => (v.lang || "").toLowerCase().startsWith("fr")) ||
      null;
  } catch (e) { _frVoice = null; }
}
if (typeof window !== "undefined" && window.speechSynthesis) {
  pickFrenchVoice();
  try { window.speechSynthesis.onvoiceschanged = pickFrenchVoice; } catch (e) { /* ignore */ }
}

function ttsSupported() {
  return typeof window !== "undefined" && !!window.speechSynthesis;
}

function speak(text, rate = 0.95, onend) {
  try {
    if (!ttsSupported()) { if (onend) onend(); return false; }
    const synth = window.speechSynthesis;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    if (!_frVoice) pickFrenchVoice();
    if (_frVoice) u.voice = _frVoice;
    u.lang = "fr-FR";
    u.rate = clamp(rate, 0.5, 1.4);
    u.pitch = 1;
    if (onend) { u.onend = onend; u.onerror = onend; }
    synth.speak(u);
    return true;
  } catch (e) { if (onend) onend(); return false; }
}

function stopSpeaking() {
  try { if (ttsSupported()) window.speechSynthesis.cancel(); } catch (e) { /* ignore */ }
}

/* ------------------------------ speech (in) ------------------------------- */

function srSupported() {
  return typeof window !== "undefined" && !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

/* Returns a recognizer you can .start()/.stop(), or null when unsupported.
   Callers must degrade to self-assessment when this is null. */
function makeRecognizer(onResult, onEnd) {
  if (!srSupported()) return null;
  try {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const r = new SR();
    r.lang = "fr-FR";
    r.interimResults = false;
    r.maxAlternatives = 1;
    r.continuous = false;
    r.onresult = (e) => {
      try {
        const t = Array.from(e.results).map((res) => res[0].transcript).join(" ");
        onResult(t);
      } catch (err) { /* ignore */ }
    };
    r.onerror = () => { if (onEnd) onEnd(); };
    r.onend = () => { if (onEnd) onEnd(); };
    return r;
  } catch (e) { return null; }
}

/* ------------------------------ SM-2 scheduler ---------------------------- */
/* Classic SuperMemo-2: quality 0-5. q<3 resets repetitions and requeues in
   10 minutes; q>=3 grows the interval by the ease factor. Production-first:
   the review UI asks the learner to *produce* French before grading. */

function newCard({ fr, en, example = "", exampleEn = "", glyph = "", source = "seed" }) {
  return {
    id: uid(), fr, en, example, exampleEn, glyph, source,
    ef: 2.5, reps: 0, interval: 0, lapses: 0,
    due: Date.now(), added: Date.now(),
  };
}

function gradeCard(card, q) {
  const now = Date.now();
  let { ef = 2.5, reps = 0, interval = 0, lapses = 0 } = card;
  if (q < 3) {
    if (reps > 0) lapses += 1;
    return { ...card, ef: Math.max(1.3, ef - 0.2), reps: 0, interval: 0, lapses, due: now + 10 * 60 * 1000 };
  }
  ef = Math.max(1.3, ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)));
  reps += 1;
  if (reps === 1) interval = 1;
  else if (reps === 2) interval = 6;
  else interval = Math.max(1, Math.round(interval * ef));
  return { ...card, ef, reps, interval, lapses, due: now + interval * 86400000 };
}

const dueCards = (deck) => (deck || []).filter((c) => c.due <= Date.now()).sort((a, b) => a.due - b.due);
const matureCount = (deck) => (deck || []).filter((c) => c.interval >= 21).length;
const knownCount = (deck) => (deck || []).filter((c) => c.reps >= 2).length;

function addCardsToDeck(deck, items, source) {
  const have = new Set((deck || []).map((c) => norm(c.fr)));
  const fresh = [];
  for (const it of items) {
    if (!it || !it.fr || have.has(norm(it.fr))) continue;
    have.add(norm(it.fr));
    fresh.push(newCard({ ...it, source }));
  }
  return { deck: [...(deck || []), ...fresh], added: fresh.length };
}

/* ========================================================================== */
/*  DATA — deterministic curriculum skeletons, seeds, banks and fallbacks     */
/* ========================================================================== */

const CEFR_ORDER = ["A0", "A1", "A2", "B1", "B2"];
const CEFR_HOURS = { A1: "≈ 80–100 h", A2: "≈ 180–200 h (cumulative)", B1: "≈ 350–400 h (cumulative)" };
const CEFR_BASE = { A0: 4, A1: 20, A2: 42, B1: 62, B2: 78 };

/* ------------------------------- seed decks ------------------------------- */

const CAROL_SEED_CARDS = [
  { fr: "bonjour", en: "hello / good morning", glyph: "👋", example: "Bonjour, madame !", exampleEn: "Hello, madam!" },
  { fr: "merci", en: "thank you", glyph: "🙏", example: "Merci beaucoup.", exampleEn: "Thank you very much." },
  { fr: "s'il vous plaît", en: "please (polite)", glyph: "🤲", example: "Un café, s'il vous plaît.", exampleEn: "A coffee, please." },
  { fr: "au revoir", en: "goodbye", glyph: "👋", example: "Au revoir, à demain !", exampleEn: "Goodbye, see you tomorrow!" },
  { fr: "oui", en: "yes", glyph: "✅", example: "Oui, bien sûr.", exampleEn: "Yes, of course." },
  { fr: "non", en: "no", glyph: "❌", example: "Non, merci.", exampleEn: "No, thank you." },
  { fr: "l'eau", en: "water", glyph: "💧", example: "Je voudrais de l'eau.", exampleEn: "I would like some water." },
  { fr: "le pain", en: "bread", glyph: "🥖", example: "Le pain est frais.", exampleEn: "The bread is fresh." },
  { fr: "un café", en: "a coffee", glyph: "☕", example: "Un café, s'il vous plaît.", exampleEn: "A coffee, please." },
  { fr: "la gare", en: "the train station", glyph: "🚉", example: "Où est la gare ?", exampleEn: "Where is the station?" },
  { fr: "les toilettes", en: "the toilets", glyph: "🚻", example: "Où sont les toilettes ?", exampleEn: "Where are the toilets?" },
  { fr: "je voudrais", en: "I would like", glyph: "🙋", example: "Je voudrais un thé.", exampleEn: "I would like a tea." },
];

const STUART_SEED_CARDS = [
  { fr: "il y a", en: "there is / there are (sounds like 'ya')", glyph: "👉", example: "Il y a un problème.", exampleEn: "There is a problem." },
  { fr: "je ne sais pas", en: "I don't know (collapses to 'ché pas')", glyph: "🤷", example: "Je ne sais pas où il est.", exampleEn: "I don't know where he is." },
  { fr: "qu'est-ce que", en: "what… ? (sounds like 'kess-kuh')", glyph: "❓", example: "Qu'est-ce que tu fais ?", exampleEn: "What are you doing?" },
  { fr: "du coup", en: "so / as a result (spoken filler)", glyph: "🔗", example: "Du coup, on est partis.", exampleEn: "So, we left." },
  { fr: "quand même", en: "still / all the same", glyph: "🤨", example: "C'est quand même bizarre.", exampleEn: "It's still strange." },
  { fr: "tout à l'heure", en: "earlier / in a bit (liaison: tou-ta-leur)", glyph: "⏰", example: "À tout à l'heure !", exampleEn: "See you in a bit!" },
  { fr: "on y va", en: "let's go (enchaînement: on-ni-va)", glyph: "🏃", example: "Allez, on y va !", exampleEn: "Come on, let's go!" },
  { fr: "il faut que", en: "it's necessary that (+ subjunctive)", glyph: "⚠️", example: "Il faut que je parte.", exampleEn: "I have to leave." },
];

/* --------------------------- phonics (Carol, A0) --------------------------- */

const PHONICS_LESSONS = [
  {
    id: "ph1", title: "Nasal vowels", emoji: "👃",
    tip: "French has vowels that go through the nose. The letter combos on/om, an/en, in/ain and un signal a nasal vowel — the N or M itself is NOT pronounced.",
    items: [
      { text: "bon", note: "on → 'oh' through the nose (good)" },
      { text: "non", note: "same nasal 'on' (no)" },
      { text: "dans", note: "an → open nasal 'ah' (in)" },
      { text: "enfant", note: "en + an — two nasals (child)" },
      { text: "vin", note: "in → nasal 'eh' (wine)" },
      { text: "pain", note: "ain → same nasal as 'vin' (bread)" },
      { text: "un", note: "its own nasal — 'uh' through the nose (one/a)" },
      { text: "lundi", note: "un inside a word (Monday)" },
    ],
  },
  {
    id: "ph2", title: "The French R", emoji: "🗣️",
    tip: "The French R lives at the back of the throat — like a very soft gargle. Don't roll it, don't use the English R. Whisper 'hhh' at the back of your throat, then add voice.",
    items: [
      { text: "rouge", note: "R starts the word (red)" },
      { text: "merci", note: "R in the middle (thank you)" },
      { text: "Paris", note: "soft single R" },
      { text: "très", note: "R after T (very)" },
      { text: "bonjour", note: "R at the end — very light (hello)" },
    ],
  },
  {
    id: "ph3", title: "Silent letters", emoji: "🤫",
    tip: "Most final consonants are silent — this is why written French looks longer than it sounds. Final -s, -t, -d, -x, -p usually vanish. The -ent ending of ils/elles verbs is completely silent.",
    items: [
      { text: "petit", note: "final t silent → 'puh-tee' (small)" },
      { text: "vous", note: "final s silent → 'voo' (you)" },
      { text: "trop", note: "final p silent → 'troh' (too much)" },
      { text: "ils parlent", note: "-ent totally silent → 'eel parl' (they speak)" },
      { text: "l'hôtel", note: "h is always silent (the hotel)" },
    ],
  },
  {
    id: "ph4", title: "u vs ou", emoji: "👄",
    tip: "Two different vowels English merges. 'ou' = English 'oo'. French 'u' doesn't exist in English: say 'ee' and round your lips into a tight circle without moving your tongue.",
    items: [
      { text: "tu", note: "tight French u (you)" },
      { text: "tout", note: "relaxed 'oo' (all)" },
      { text: "rue", note: "French u (street)" },
      { text: "roue", note: "'oo' (wheel)" },
      { text: "su", note: "u (known)" },
      { text: "sous", note: "ou (under)" },
    ],
  },
  {
    id: "ph5", title: "Letter combos → sounds", emoji: "🔡",
    tip: "French spelling is more regular than English once you learn the combos: oi = 'wa', au/eau = 'oh', ai = 'eh', é = 'ay', ch = 'sh', gn = 'ny'.",
    items: [
      { text: "moi", note: "oi → 'mwa' (me)" },
      { text: "trois", note: "oi → 'trwa' (three)" },
      { text: "l'eau", note: "eau → 'loh' (water)" },
      { text: "beau", note: "eau → 'boh' (beautiful)" },
      { text: "café", note: "é → 'ay' (coffee)" },
      { text: "chat", note: "ch → 'sh', silent t → 'sha' (cat)" },
      { text: "montagne", note: "gn → 'ny' like 'canyon' (mountain)" },
    ],
  },
  {
    id: "ph6", title: "Words that link", emoji: "🔗",
    tip: "When a word ends in a (usually silent) consonant and the next starts with a vowel, they link: 'les amis' sounds like 'lay-za-mee'. You'll meet this everywhere — it's normal, not fast talking.",
    items: [
      { text: "les amis", note: "s wakes up as Z → lay-za-mee (the friends)" },
      { text: "vous avez", note: "voo-za-vay (you have)" },
      { text: "un homme", note: "n links → uh-nom (a man)" },
      { text: "c'est un café", note: "t links → say-tuh (it's a coffee)" },
    ],
  },
];

/* ------------------ connected-speech mini-lessons (Stuart) ------------------ */

const CS_LESSONS = [
  {
    id: "cs1", title: "Liaison — silent letters wake up", emoji: "🔗",
    body: "A normally-silent final consonant is pronounced when the next word starts with a vowel — and it attaches to THAT word. 's' and 'x' become /z/, 'd' becomes /t/. This is why 'les amis' has a Z you never see: lay-ZA-mee. Liaison is compulsory after articles (les, des, un), pronouns (vous, nous, ils, on) and one-syllable prepositions (en, dans → dan-z…).",
    pairs: [
      { written: "les amis", sounds: "lay-za-mee" },
      { written: "vous avez", sounds: "voo-za-vay" },
      { written: "un homme", sounds: "uh-nom" },
      { written: "petit ami", sounds: "puh-tee-ta-mee" },
      { written: "c'est un", sounds: "say-tuh" },
      { written: "deux heures", sounds: "deu-zeur" },
    ],
  },
  {
    id: "cs2", title: "Elision — vowels get deleted", emoji: "✂️",
    body: "je, le, la, ne, de, que, se, ce lose their vowel before another vowel and glue on with an apostrophe: je + ai = j'ai, le + homme = l'homme, que + il = qu'il. Your ear must expect ONE syllable where your eye expects two words.",
    pairs: [
      { written: "je ai → j'ai", sounds: "zhay" },
      { written: "le homme → l'homme", sounds: "lom" },
      { written: "ne est pas → n'est pas", sounds: "nay-pa" },
      { written: "que il → qu'il", sounds: "keel" },
      { written: "si il → s'il", sounds: "seel" },
    ],
  },
  {
    id: "cs3", title: "Enchaînement — consonants slide over", emoji: "🌊",
    body: "Even consonants that are ALWAYS pronounced slide onto a following vowel, re-cutting the syllables. 'Elle est' isn't 'elle | est', it's è-LÈ. French re-syllabifies across word boundaries — the boundaries you read simply don't exist in sound.",
    pairs: [
      { written: "elle est", sounds: "è-lè" },
      { written: "une amie", sounds: "u-na-mee" },
      { written: "il arrive", sounds: "ee-la-reev" },
      { written: "avec elle", sounds: "a-vè-kèl" },
      { written: "sept heures", sounds: "sè-teur" },
    ],
  },
  {
    id: "cs4", title: "Everyday reductions", emoji: "🗜️",
    body: "Casual spoken French shrinks. 'ne' is dropped almost always (je sais pas). 'je' before s becomes 'ch' (je sais pas → ché pas / chais pas). 'tu as' → t'as, 'il y a' → ya, 'je suis' → chuis. Nobody is mumbling — these are the standard spoken forms.",
    pairs: [
      { written: "je ne sais pas", sounds: "ché pa / shay pa" },
      { written: "tu as vu", sounds: "t'a vu" },
      { written: "il y a", sounds: "ya" },
      { written: "je suis", sounds: "chuis / shwee" },
      { written: "il ne faut pas", sounds: "eel fo pa" },
    ],
  },
  {
    id: "cs5", title: "Rhythm — no word stress", emoji: "🥁",
    body: "English stresses one syllable per word, which is how your ear finds word boundaries. French has NO word stress: every syllable is even, with a light rise at the end of each phrase. Words merge into one smooth ribbon of equal beats — so train your ear on the phrase, not the word.",
    pairs: [
      { written: "Je vais au marché ce matin", sounds: "zhuh-vay-o-mar-shay-suh-ma-tan (even beats)" },
      { written: "C'est une bonne idée", sounds: "say-tun-bo-nee-day" },
    ],
  },
  {
    id: "cs6", title: "Liaison traps with numbers", emoji: "🔢",
    body: "Numbers change sound depending on what follows. six = 'seess' alone, 'see' before a consonant (six livres), 'seez' before a vowel (six ans). dix behaves the same. neuf heures → neu-Veur. vingt ans → vin-tan.",
    pairs: [
      { written: "six ans", sounds: "see-zan" },
      { written: "dix hommes", sounds: "dee-zom" },
      { written: "neuf heures", sounds: "neu-veur" },
      { written: "vingt ans", sounds: "vin-tan" },
      { written: "deux enfants", sounds: "deu-zan-fan" },
    ],
  },
];

/* ---------------- listening trainer clip ladder (fallbacks) ---------------- */
/* Rungs go from ~95% comprehensible slow speech with dual subtitles down to
   authentic-speed with no subtitle support. AI generates fresh clips at the
   rung's parameters; these built-in clips guarantee the loop always works.  */

const CLIP_LADDER = [
  { rung: 1, level: "A2", rate: 0.8, subs: "dual", topicHint: "simple daily routine" },
  { rung: 2, level: "A2", rate: 0.85, subs: "dual", topicHint: "neighbourhood and habits" },
  { rung: 3, level: "A2+", rate: 0.9, subs: "fr", topicHint: "a recent outing, passé composé" },
  { rung: 4, level: "A2+", rate: 0.95, subs: "fr", topicHint: "plans and weather, casual reductions" },
  { rung: 5, level: "B1", rate: 0.95, subs: "fr", topicHint: "childhood memory, imparfait" },
  { rung: 6, level: "B1", rate: 1.0, subs: "none", topicHint: "explaining a problem, spoken fillers" },
  { rung: 7, level: "B1", rate: 1.0, subs: "none", topicHint: "reported speech, opinions" },
  { rung: 8, level: "B1+", rate: 1.05, subs: "none", topicHint: "abstract opinion, complex clauses" },
];

/* 5 clips per rung; the trainer draws one at random each session. */
const FALLBACK_CLIPS = {
  1: [
    { text: "Ce matin, je vais au marché. J'achète du pain et des pommes. Le marché est près de chez moi.", en: "This morning I'm going to the market. I buy bread and apples. The market is near my place." },
    { text: "Le matin, je me lève à sept heures. Je prends un café et je regarde les nouvelles. Après, je vais au travail à pied.", en: "In the morning I get up at seven. I have a coffee and watch the news. Then I walk to work." },
    { text: "Le dimanche, on fait une grande promenade. On part après le déjeuner et on rentre vers cinq heures.", en: "On Sundays we go for a long walk. We leave after lunch and come home around five." },
    { text: "Ma sœur habite près de chez moi. Elle vient dîner à la maison tous les jeudis soirs.", en: "My sister lives near me. She comes over for dinner every Thursday evening." },
    { text: "Je fais les courses le samedi matin. Il y a moins de monde et les fruits sont plus frais.", en: "I do the shopping on Saturday mornings. There are fewer people and the fruit is fresher." },
  ],
  2: [
    { text: "Il y a un petit café au coin de la rue. J'y vais tous les matins avant le travail. Le serveur me connaît bien.", en: "There's a little café on the corner. I go there every morning before work. The waiter knows me well." },
    { text: "Dans mon quartier, il y a une boulangerie excellente. J'y achète une baguette tous les jours en rentrant.", en: "In my neighbourhood there's an excellent bakery. I buy a baguette there every day on my way home." },
    { text: "Le parc à côté de chez nous est très agréable. On y va souvent le week-end quand il fait beau.", en: "The park next to our place is very pleasant. We often go there at weekends when the weather is nice." },
    { text: "Mes voisins sont très sympas. On s'aide beaucoup, surtout pour les petites choses du quotidien.", en: "My neighbours are really nice. We help each other a lot, especially with everyday little things." },
    { text: "Il y a un nouveau marché le mercredi matin. Les légumes y sont moins chers qu'au supermarché.", en: "There's a new market on Wednesday mornings. The vegetables there are cheaper than at the supermarket." },
  ],
  3: [
    { text: "Hier soir, on est allés au restaurant avec des amis. On a commandé le plat du jour et une carafe d'eau. C'était vraiment bon.", en: "Last night we went to a restaurant with friends. We ordered the daily special and a jug of water. It was really good." },
    { text: "Ce week-end, on a visité un petit village près de la mer. On a marché sur la plage et on a très bien mangé.", en: "This weekend we visited a little village by the sea. We walked on the beach and ate really well." },
    { text: "Hier, j'ai retrouvé un vieil ami en ville. On a pris un verre et on a parlé pendant des heures.", en: "Yesterday I met up with an old friend in town. We had a drink and talked for hours." },
    { text: "Samedi soir, on est allés voir un concert. Il y avait beaucoup de monde, mais c'était génial.", en: "Saturday night we went to see a concert. There were a lot of people, but it was great." },
    { text: "La semaine dernière, j'ai enfin fini ce gros projet. Pour fêter ça, on est sortis dîner tous ensemble.", en: "Last week I finally finished that big project. To celebrate, we all went out for dinner together." },
  ],
  4: [
    { text: "Je ne sais pas si tu as vu la météo, mais il va pleuvoir tout le week-end. On devrait peut-être rester à la maison.", en: "I don't know if you saw the forecast, but it's going to rain all weekend. Maybe we should stay home." },
    { text: "Je sais pas encore ce qu'on fait ce soir. Y a un bon film au cinéma, mais on est un peu fatigués.", en: "I don't know yet what we're doing tonight. There's a good film on at the cinema, but we're a bit tired." },
    { text: "T'as vu qu'il va faire super beau demain ? On devrait en profiter pour aller à la plage.", en: "Did you see it's going to be gorgeous tomorrow? We should make the most of it and go to the beach." },
    { text: "On va peut-être partir quelques jours la semaine prochaine. Ça dépend du travail de ma femme.", en: "We might go away for a few days next week. It depends on my wife's work." },
    { text: "Il paraît qu'il va neiger ce week-end. Du coup, on va rester bien au chaud à la maison.", en: "Apparently it's going to snow this weekend. So we're going to stay nice and warm at home." },
  ],
  5: [
    { text: "Quand j'étais petit, on passait les vacances chez mes grands-parents. Il y avait un grand jardin et on jouait dehors toute la journée.", en: "When I was little we spent the holidays at my grandparents'. There was a big garden and we played outside all day." },
    { text: "Quand j'étais adolescent, j'écoutais de la musique toute la journée. Mes parents trouvaient ça insupportable.", en: "When I was a teenager I listened to music all day long. My parents found it unbearable." },
    { text: "Petits, on passait l'été au bord d'un lac. On se baignait le matin et on pêchait l'après-midi.", en: "As kids we spent the summer by a lake. We swam in the morning and fished in the afternoon." },
    { text: "Ma grand-mère faisait un gâteau au chocolat incroyable. Toute la maison sentait bon quand on arrivait.", en: "My grandmother made an incredible chocolate cake. The whole house smelled wonderful when we arrived." },
    { text: "À l'école, j'avais un copain qui faisait rire toute la classe. Même les profs avaient du mal à rester sérieux.", en: "At school I had a friend who made the whole class laugh. Even the teachers struggled to keep a straight face." },
  ],
  6: [
    { text: "Franchement, je n'ai pas eu le temps de finir. Il y a eu un problème avec le train, du coup je suis arrivé en retard.", en: "Honestly, I didn't have time to finish. There was a problem with the train, so I arrived late." },
    { text: "Bon, en fait, la réunion a été annulée au dernier moment. Personne ne m'avait prévenu, du coup j'ai fait le trajet pour rien.", en: "Well, actually, the meeting was cancelled at the last minute. Nobody had warned me, so I made the trip for nothing." },
    { text: "J'ai un souci avec ma voiture depuis hier. Elle fait un bruit bizarre, il faut que je l'emmène au garage.", en: "I've had a problem with my car since yesterday. It's making a strange noise, I need to take it to the garage." },
    { text: "Franchement, le livreur s'est encore trompé d'adresse. C'est la troisième fois ce mois-ci, ça commence à bien faire.", en: "Honestly, the delivery driver got the address wrong again. It's the third time this month — it's getting a bit much." },
    { text: "On devait partir à huit heures, mais évidemment tout le monde était en retard. Bref, on est arrivés après le début.", en: "We were supposed to leave at eight, but of course everyone was late. Anyway, we arrived after the start." },
  ],
  7: [
    { text: "Elle m'a dit qu'elle arriverait vers huit heures, mais tu sais comment elle est. Si elle n'est pas là à neuf heures, on commence sans elle.", en: "She told me she'd arrive around eight, but you know how she is. If she's not there by nine, we start without her." },
    { text: "Il m'a dit que le restaurant était complet, mais j'ai l'impression qu'il n'a même pas appelé. Ça ne m'étonnerait pas de lui.", en: "He told me the restaurant was full, but I get the feeling he didn't even call. It wouldn't surprise me, coming from him." },
    { text: "Elle pense qu'on devrait vendre la maison, mais moi, je ne suis pas convaincu. On en reparlera calmement ce week-end.", en: "She thinks we should sell the house, but me, I'm not convinced. We'll talk it over calmly this weekend." },
    { text: "D'après la voisine, les travaux dureraient encore deux mois. Si c'est vrai, on ne va pas dormir tranquilles avant l'automne.", en: "According to the neighbour, the building work will last another two months. If that's true, we won't sleep in peace before autumn." },
    { text: "Ils ont annoncé qu'ils allaient se marier l'année prochaine. Tout le monde s'y attendait, mais ça fait quand même plaisir.", en: "They announced they're getting married next year. Everyone was expecting it, but it's still lovely news." },
  ],
  8: [
    { text: "Ce qui m'étonne, c'est qu'on en parle seulement maintenant. Tout le monde était au courant depuis des semaines, mais personne n'a rien dit.", en: "What surprises me is that we're only talking about it now. Everyone had known for weeks, but nobody said anything." },
    { text: "Ce qui me dérange, ce n'est pas la décision elle-même, c'est la façon dont elle a été prise. On aurait pu au moins nous consulter.", en: "What bothers me isn't the decision itself, it's the way it was made. They could at least have consulted us." },
    { text: "On dit souvent que les gens ne lisent plus, mais je n'en suis pas sûr. Ils lisent autrement, voilà tout.", en: "People often say nobody reads any more, but I'm not so sure. They read differently, that's all." },
    { text: "Plus j'y réfléchis, plus je me dis qu'on juge trop vite. La plupart des gens font simplement de leur mieux.", en: "The more I think about it, the more I tell myself we judge too quickly. Most people are simply doing their best." },
    { text: "Si on m'avait dit il y a dix ans que je vivrais ici, je ne l'aurais jamais cru. La vie prend parfois des chemins étonnants.", en: "If someone had told me ten years ago that I'd be living here, I'd never have believed it. Life sometimes takes surprising paths." },
  ],
};

/* Local fallback for the connected-speech gap analysis when the API is
   unreachable — pattern-matches the most common phenomena. */
function localGapAnalysis(segments) {
  return segments.map((seg) => {
    const s = norm(seg);
    let phenomenon = "enchaînement", why =
      "French re-syllabifies across word boundaries — the final consonant of one word attaches to the vowel of the next, so the boundaries you read don't exist in sound.";
    if (/(les|des|vous|nous|ils|elles|on|un|deux|trois|six|dix|est|sont|petit|grand|tout|quand|en) [aeiouyhàâéèêëîïôöùûü]/.test(s + " ")) {
      phenomenon = "liaison";
      why = "A silent final consonant wakes up before the vowel that follows and attaches to it (s/x → /z/, d → /t/), hiding the word boundary.";
    }
    if (/(j'|l'|n'|qu'|d'|s'|c'|m'|t')/.test(seg.toLowerCase())) {
      phenomenon = "elision";
      why = "A one-syllable word lost its vowel and glued onto the next word — two written words, one spoken syllable.";
    }
    if (/je ne .*pas|il y a|je suis|tu as/.test(s)) {
      phenomenon = "reduction";
      why = "Casual speech shrinks this: 'ne' is dropped, 'il y a' → 'ya', 'je suis' → 'chuis'. The full written form is rarely what you hear.";
    }
    return { segment: seg, phenomenon, sounds_like: "", why };
  });
}

/* --------------------- Foundations curriculum (Carol) ---------------------- */
/* Deterministic skeleton; the AI freshens explanations/practice at runtime.
   Each unit: brief explicit grammar -> blocked practice -> output task ->
   vocabulary pushed to SRS.                                                  */

const FOUNDATIONS_UNITS = [
  {
    id: "u1", title: "Bonjour !", scenario: "Greet people and be polite", level: "A0", phonicsId: "ph1",
    grammar: {
      point: "Greetings & politeness",
      brief: "French politeness words do heavy lifting: walk into any shop and say 'Bonjour'; leave with 'Au revoir, merci'. 'Salut' is casual (friends only). 'Ça va ?' means 'how's it going?' — and the answer is also 'Ça va'.",
      examples: [
        { fr: "Bonjour, madame.", en: "Hello, madam." },
        { fr: "Salut ! Ça va ?", en: "Hi! How's it going?" },
        { fr: "Ça va bien, merci.", en: "I'm fine, thanks." },
        { fr: "Au revoir, bonne journée !", en: "Goodbye, have a good day!" },
      ],
    },
    vocab: [
      { fr: "bonjour", en: "hello", glyph: "👋", example: "Bonjour, madame !" },
      { fr: "bonsoir", en: "good evening", glyph: "🌆", example: "Bonsoir, monsieur." },
      { fr: "salut", en: "hi (casual)", glyph: "✌️", example: "Salut, ça va ?" },
      { fr: "ça va", en: "how's it going / I'm fine", glyph: "🙂", example: "Ça va bien, merci." },
      { fr: "bonne journée", en: "have a good day", glyph: "☀️", example: "Au revoir, bonne journée !" },
    ],
    task: { prompt: "A shopkeeper says « Bonjour ! Ça va ? ». Reply: greet them back, say you're fine, and thank them.", model: "Bonjour ! Ça va bien, merci." },
  },
  {
    id: "u2", title: "Un café, s'il vous plaît", scenario: "Order a drink at a café", level: "A0", phonicsId: "ph2",
    grammar: {
      point: "Je voudrais + un/une",
      brief: "'Je voudrais' (I would like) is your magic ordering phrase — polite and works everywhere. Every French noun is masculine (un) or feminine (une): un café, un thé, but une eau. Learn the un/une with the word, like it's part of the spelling.",
      examples: [
        { fr: "Je voudrais un café, s'il vous plaît.", en: "I'd like a coffee, please." },
        { fr: "Je voudrais un croissant.", en: "I'd like a croissant." },
        { fr: "Une eau, s'il vous plaît.", en: "A water, please." },
        { fr: "L'addition, s'il vous plaît.", en: "The bill, please." },
      ],
    },
    vocab: [
      { fr: "un thé", en: "a tea", glyph: "🍵", example: "Je voudrais un thé." },
      { fr: "un croissant", en: "a croissant", glyph: "🥐", example: "Un croissant, s'il vous plaît." },
      { fr: "une eau", en: "a water", glyph: "💧", example: "Une eau, s'il vous plaît." },
      { fr: "l'addition", en: "the bill", glyph: "🧾", example: "L'addition, s'il vous plaît." },
      { fr: "un verre de vin", en: "a glass of wine", glyph: "🍷", example: "Un verre de vin rouge." },
    ],
    task: { prompt: "You're at a café in Paris. Order a coffee and a croissant, politely.", model: "Bonjour, je voudrais un café et un croissant, s'il vous plaît." },
  },
  {
    id: "u3", title: "Les nombres et les prix", scenario: "Numbers 1–20 and paying", level: "A0", phonicsId: "ph3",
    grammar: {
      point: "Numbers & C'est combien ?",
      brief: "Ask a price with 'C'est combien ?' (how much is it?). The answer uses 'ça coûte… / c'est… euros'. Numbers 1–10: un, deux, trois, quatre, cinq, six, sept, huit, neuf, dix. 11–16 are their own words; 17–19 are dix-sept, dix-huit, dix-neuf.",
      examples: [
        { fr: "C'est combien ?", en: "How much is it?" },
        { fr: "C'est trois euros.", en: "It's three euros." },
        { fr: "Ça coûte dix euros.", en: "It costs ten euros." },
        { fr: "Deux cafés, ça fait cinq euros.", en: "Two coffees, that's five euros." },
      ],
    },
    vocab: [
      { fr: "combien", en: "how much / how many", glyph: "❓", example: "C'est combien ?" },
      { fr: "trois", en: "three", glyph: "3️⃣", example: "Trois croissants, s'il vous plaît." },
      { fr: "cinq", en: "five", glyph: "5️⃣", example: "Ça coûte cinq euros." },
      { fr: "dix", en: "ten", glyph: "🔟", example: "C'est dix euros." },
      { fr: "l'argent", en: "money", glyph: "💶", example: "Je n'ai pas d'argent." },
    ],
    task: { prompt: "Ask the baker how much it is, then say 'here are five euros' (Voilà cinq euros).", model: "C'est combien ? Voilà cinq euros." },
  },
  {
    id: "u4", title: "Je me présente", scenario: "Introduce yourself", level: "A1", phonicsId: "ph4",
    grammar: {
      point: "être (je suis / tu es / vous êtes)",
      brief: "'Être' (to be) is your first real verb: je suis (I am), tu es (you are, casual), vous êtes (you are, polite). 'Je m'appelle…' = my name is. 'J'habite à…' = I live in. Nationalities agree: anglais (m) / anglaise (f).",
      examples: [
        { fr: "Je m'appelle Carol.", en: "My name is Carol." },
        { fr: "Je suis anglaise.", en: "I am English (f)." },
        { fr: "J'habite à Londres.", en: "I live in London." },
        { fr: "Enchantée !", en: "Nice to meet you! (f)" },
      ],
    },
    vocab: [
      { fr: "je m'appelle", en: "my name is", glyph: "🏷️", example: "Je m'appelle Carol." },
      { fr: "je suis", en: "I am", glyph: "🧍", example: "Je suis anglaise." },
      { fr: "j'habite à", en: "I live in", glyph: "🏠", example: "J'habite à Londres." },
      { fr: "enchanté", en: "nice to meet you", glyph: "🤝", example: "Enchanté, madame !" },
      { fr: "anglaise", en: "English (feminine)", glyph: "🇬🇧", example: "Elle est anglaise." },
    ],
    task: { prompt: "Introduce yourself: your name, that you're English, and where you live.", model: "Je m'appelle Carol. Je suis anglaise et j'habite à Londres." },
  },
  {
    id: "u5", title: "Où est la gare ?", scenario: "Ask for and follow directions", level: "A1", phonicsId: "ph5",
    grammar: {
      point: "Où est… ? + directions",
      brief: "'Où est… ?' = where is…? Answers use direction words: à gauche (left), à droite (right), tout droit (straight on), près de (near), loin (far). 'Pardon, madame/monsieur' politely opens any question to a stranger.",
      examples: [
        { fr: "Pardon, où est la gare ?", en: "Excuse me, where is the station?" },
        { fr: "C'est à gauche.", en: "It's on the left." },
        { fr: "Allez tout droit.", en: "Go straight on." },
        { fr: "C'est près de la banque.", en: "It's near the bank." },
      ],
    },
    vocab: [
      { fr: "à gauche", en: "on the left", glyph: "⬅️", example: "La gare est à gauche." },
      { fr: "à droite", en: "on the right", glyph: "➡️", example: "Tournez à droite." },
      { fr: "tout droit", en: "straight ahead", glyph: "⬆️", example: "Allez tout droit." },
      { fr: "près de", en: "near", glyph: "📍", example: "C'est près de la gare." },
      { fr: "la rue", en: "the street", glyph: "🛣️", example: "C'est dans cette rue." },
    ],
    task: { prompt: "Stop a passer-by and ask where the station is, politely.", model: "Pardon, madame. Où est la gare, s'il vous plaît ?" },
  },
  {
    id: "u6", title: "Au restaurant", scenario: "Order a meal", level: "A1", phonicsId: "ph6",
    grammar: {
      point: "avoir (j'ai faim) + pour moi",
      brief: "French says 'I HAVE hunger': j'ai faim (I'm hungry), j'ai soif (I'm thirsty). When ordering for people: 'Pour moi, le poulet' (the chicken for me). 'Le plat du jour' is the daily special — usually the best-value thing on the menu.",
      examples: [
        { fr: "J'ai faim !", en: "I'm hungry!" },
        { fr: "Pour moi, le poulet.", en: "The chicken for me." },
        { fr: "Le plat du jour, s'il vous plaît.", en: "The daily special, please." },
        { fr: "C'était délicieux.", en: "It was delicious." },
      ],
    },
    vocab: [
      { fr: "j'ai faim", en: "I'm hungry", glyph: "😋", example: "J'ai faim, on mange ?" },
      { fr: "le poulet", en: "the chicken", glyph: "🍗", example: "Pour moi, le poulet." },
      { fr: "le poisson", en: "the fish", glyph: "🐟", example: "Le poisson est très bon." },
      { fr: "les légumes", en: "the vegetables", glyph: "🥦", example: "Avec des légumes." },
      { fr: "le plat du jour", en: "the daily special", glyph: "🍽️", example: "Le plat du jour, s'il vous plaît." },
    ],
    task: { prompt: "The waiter asks « Vous désirez ? ». Say you're hungry and order the daily special with vegetables.", model: "J'ai faim ! Le plat du jour avec des légumes, s'il vous plaît." },
  },
  {
    id: "u7", title: "Ma journée", scenario: "Describe your day", level: "A1", phonicsId: null,
    grammar: {
      point: "-er verbs in the present",
      brief: "Most French verbs end in -er and follow one pattern: je travaille, tu travailles, il/elle travaille (all sound identical!). Time anchors: le matin (morning), l'après-midi (afternoon), le soir (evening). String them together and you can narrate a whole day.",
      examples: [
        { fr: "Le matin, je travaille.", en: "In the morning, I work." },
        { fr: "À midi, je mange avec Stuart.", en: "At noon, I eat with Stuart." },
        { fr: "Le soir, je regarde la télé.", en: "In the evening, I watch TV." },
        { fr: "J'écoute de la musique.", en: "I listen to music." },
      ],
    },
    vocab: [
      { fr: "je travaille", en: "I work", glyph: "💼", example: "Je travaille le matin." },
      { fr: "je mange", en: "I eat", glyph: "🍽️", example: "Je mange à midi." },
      { fr: "je regarde", en: "I watch", glyph: "📺", example: "Je regarde la télé." },
      { fr: "le matin", en: "the morning", glyph: "🌅", example: "Le matin, je travaille." },
      { fr: "le soir", en: "the evening", glyph: "🌙", example: "Le soir, je lis." },
    ],
    task: { prompt: "Describe your day in two short sentences: what you do in the morning, and in the evening.", model: "Le matin, je travaille. Le soir, je regarde la télé." },
  },
  {
    id: "u8", title: "Un problème à l'hôtel", scenario: "Handle a problem at a hotel", level: "A1", phonicsId: null,
    grammar: {
      point: "il y a / ça ne marche pas / pouvez-vous",
      brief: "Complain politely with three tools: 'Il y a un problème' (there's a problem), 'Ça ne marche pas' (it doesn't work), and 'Pouvez-vous… ?' (can you…?). Name the thing, say it's broken, ask for help — that's the whole formula.",
      examples: [
        { fr: "Excusez-moi, il y a un problème.", en: "Excuse me, there's a problem." },
        { fr: "La douche ne marche pas.", en: "The shower doesn't work." },
        { fr: "Pouvez-vous m'aider ?", en: "Can you help me?" },
        { fr: "Merci beaucoup !", en: "Thank you very much!" },
      ],
    },
    vocab: [
      { fr: "la chambre", en: "the (hotel) room", glyph: "🛏️", example: "La chambre est petite." },
      { fr: "la clé", en: "the key", glyph: "🔑", example: "Voici votre clé." },
      { fr: "la douche", en: "the shower", glyph: "🚿", example: "La douche ne marche pas." },
      { fr: "ça ne marche pas", en: "it doesn't work", glyph: "🛠️", example: "La télé ne marche pas." },
      { fr: "pouvez-vous m'aider", en: "can you help me", glyph: "🆘", example: "Pouvez-vous m'aider, s'il vous plaît ?" },
    ],
    task: { prompt: "At hotel reception: say there's a problem, the shower doesn't work, and ask if they can help.", model: "Excusez-moi, il y a un problème. La douche ne marche pas. Pouvez-vous m'aider ?" },
  },
  {
    id: "u9", title: "Quel temps fait-il ?", scenario: "Talk about the weather (small talk!)", level: "A1", phonicsId: null,
    grammar: {
      point: "il fait + weather",
      brief: "French weather runs on 'il fait' (it makes): il fait beau (it's nice), il fait froid (it's cold), il fait chaud (it's hot). Rain and snow get their own verbs: il pleut, il neige. Weather talk is the universal French small-talk opener — master this and you can chat with any shopkeeper.",
      examples: [
        { fr: "Il fait beau aujourd'hui !", en: "It's lovely today!" },
        { fr: "Il fait très froid ce matin.", en: "It's very cold this morning." },
        { fr: "Il pleut depuis lundi.", en: "It's been raining since Monday." },
        { fr: "Quel temps fait-il à Paris ?", en: "What's the weather like in Paris?" },
      ],
    },
    vocab: [
      { fr: "il fait beau", en: "the weather is nice", glyph: "☀️", example: "Il fait beau aujourd'hui !" },
      { fr: "il fait froid", en: "it's cold", glyph: "🥶", example: "Il fait froid ce matin." },
      { fr: "il pleut", en: "it's raining", glyph: "🌧️", example: "Il pleut beaucoup." },
      { fr: "il neige", en: "it's snowing", glyph: "❄️", example: "Il neige en montagne." },
      { fr: "un parapluie", en: "an umbrella", glyph: "☂️", example: "Prends ton parapluie !" },
    ],
    task: { prompt: "A neighbour says « Bonjour ! Quel temps ! ». Agree, say it's cold and raining, and that you have your umbrella.", model: "Oui ! Il fait froid et il pleut. Mais j'ai mon parapluie !" },
  },
  {
    id: "u10", title: "Ma famille", scenario: "Talk about your family", level: "A1", phonicsId: null,
    grammar: {
      point: "mon / ma / mes (possessives)",
      brief: "'My' changes with the noun: mon mari (my husband, masculine), ma sœur (my sister, feminine), mes parents (my parents, plural). One trap: before a vowel it's always mon, even for feminine words — mon amie. The possessive matches the THING owned, not the owner.",
      examples: [
        { fr: "Mon mari s'appelle Stuart.", en: "My husband is called Stuart." },
        { fr: "Ma sœur habite à Manchester.", en: "My sister lives in Manchester." },
        { fr: "Mes parents sont à la retraite.", en: "My parents are retired." },
        { fr: "Mon amie Claire adore la France.", en: "My friend Claire loves France." },
      ],
    },
    vocab: [
      { fr: "le mari", en: "the husband", glyph: "🤵", example: "Mon mari s'appelle Stuart." },
      { fr: "la femme", en: "the wife / the woman", glyph: "👩", example: "Sa femme est très gentille." },
      { fr: "la sœur", en: "the sister", glyph: "👭", example: "Ma sœur habite à Manchester." },
      { fr: "le frère", en: "the brother", glyph: "👬", example: "Mon frère a deux enfants." },
      { fr: "les parents", en: "the parents", glyph: "👵👴", example: "Mes parents sont à la retraite." },
    ],
    task: { prompt: "Introduce two members of your family: their name and one small fact each.", model: "Mon mari s'appelle Stuart. Il aime le café. Ma sœur s'appelle Anne et elle habite à Londres." },
  },
  {
    id: "u11", title: "Le week-end dernier", scenario: "Say what you did (first past tense!)", level: "A1", phonicsId: null,
    grammar: {
      point: "passé composé with avoir (j'ai + participle)",
      brief: "Your first past tense: take avoir (j'ai, tu as…) and add the past participle — j'ai mangé (I ate), j'ai visité (I visited), j'ai regardé (I watched). For -er verbs the participle just swaps -er for -é. Add 'hier' or 'le week-end dernier' and you're telling stories about your life.",
      examples: [
        { fr: "Hier, j'ai mangé au restaurant.", en: "Yesterday, I ate at a restaurant." },
        { fr: "J'ai visité un musée samedi.", en: "I visited a museum on Saturday." },
        { fr: "Nous avons regardé un film.", en: "We watched a film." },
        { fr: "C'était super !", en: "It was great!" },
      ],
    },
    vocab: [
      { fr: "hier", en: "yesterday", glyph: "📅", example: "Hier, j'ai mangé au restaurant." },
      { fr: "le week-end dernier", en: "last weekend", glyph: "🗓️", example: "Le week-end dernier, j'ai visité Bath." },
      { fr: "j'ai mangé", en: "I ate", glyph: "🍽️", example: "J'ai mangé une crêpe." },
      { fr: "j'ai visité", en: "I visited", glyph: "🏛️", example: "J'ai visité un musée." },
      { fr: "c'était super", en: "it was great", glyph: "🤩", example: "C'était super !" },
    ],
    task: { prompt: "Tell a friend two things you did last weekend, and that it was great.", model: "Le week-end dernier, j'ai visité un marché et j'ai mangé une crêpe. C'était super !" },
  },
  {
    id: "u12", title: "On se retrouve quand ?", scenario: "Make plans with someone", level: "A1", phonicsId: null,
    grammar: {
      point: "on + time expressions",
      brief: "Spoken French loves 'on' — it means 'we' in everyday speech: on va (we go), on se retrouve (we meet up). Add a time — ce soir (tonight), demain (tomorrow), à sept heures — and you can arrange anything. Seal the deal with 'd'accord !' (okay!).",
      examples: [
        { fr: "On se retrouve à sept heures ?", en: "Shall we meet at seven?" },
        { fr: "On va au restaurant ce soir ?", en: "Shall we go to the restaurant tonight?" },
        { fr: "Demain, on visite le château.", en: "Tomorrow we're visiting the castle." },
        { fr: "D'accord, à demain !", en: "Okay, see you tomorrow!" },
      ],
    },
    vocab: [
      { fr: "demain", en: "tomorrow", glyph: "🌅", example: "À demain !" },
      { fr: "ce soir", en: "tonight", glyph: "🌙", example: "On sort ce soir ?" },
      { fr: "à quelle heure", en: "at what time", glyph: "⏰", example: "À quelle heure on se retrouve ?" },
      { fr: "on se retrouve", en: "we meet up", glyph: "🤝", example: "On se retrouve au café." },
      { fr: "d'accord", en: "okay / agreed", glyph: "👌", example: "D'accord, à ce soir !" },
    ],
    task: { prompt: "Suggest meeting a friend at the café tonight at eight, and confirm with 'okay, see you tonight!'.", model: "On se retrouve au café ce soir à huit heures ? D'accord, à ce soir !" },
  },
];

/* ----------------- interleaved grammar topics (Stuart, A2/B1) --------------- */
/* mode starts 'blocked'; flips to 'interleaved' (mixed with confusables) once
   accuracy over enough items crosses the threshold — per SLA principle 4.    */

const GRAMMAR_TOPICS = [
  {
    id: "pc-imp", name: "Passé composé vs imparfait", confusableWith: ["aux-avoir-etre"],
    hint: "Passé composé = completed event ('a photo'); imparfait = background/habit ('a video'). « J'ai mangé » vs « Je mangeais ».",
    fallback: [
      { type: "mcq", prompt: "Quand j'étais petit, je ___ au football tous les samedis.", options: ["jouais", "ai joué", "joue", "jouerai"], answer: "jouais", explanation: "Habitual past action ('every Saturday') → imparfait." },
      { type: "mcq", prompt: "Hier, elle ___ un très bon film.", options: ["a vu", "voyait", "voit", "verra"], answer: "a vu", explanation: "One completed event at a specific time ('hier') → passé composé." },
      { type: "mcq", prompt: "Il ___ quand le téléphone a sonné.", options: ["dormait", "a dormi", "dort", "dormira"], answer: "dormait", explanation: "Ongoing background action interrupted by an event → imparfait." },
      { type: "type", prompt: "Translate: 'We ate at the restaurant yesterday.' (manger, on)", answer: "on a mangé au restaurant hier", explanation: "Completed one-off event → passé composé: on a mangé." },
    ],
  },
  {
    id: "aux-avoir-etre", name: "Avoir vs être in the passé composé", confusableWith: ["pc-imp"],
    hint: "Most verbs take avoir; movement/change verbs (aller, venir, partir, arriver, rester…) and all reflexives take être — and then the participle agrees.",
    fallback: [
      { type: "mcq", prompt: "Elle ___ arrivée à huit heures.", options: ["est", "a", "était", "avait"], answer: "est", explanation: "'Arriver' is a movement verb → être, and 'arrivée' agrees with elle." },
      { type: "mcq", prompt: "Nous ___ mangé ensemble.", options: ["avons", "sommes", "étions", "ont"], answer: "avons", explanation: "'Manger' takes avoir — no agreement needed." },
      { type: "mcq", prompt: "Ils ___ restés à la maison.", options: ["sont", "ont", "avaient", "étaient"], answer: "sont", explanation: "'Rester' is a DR MRS VANDERTRAMP verb → être." },
      { type: "type", prompt: "Translate: 'She left this morning.' (partir)", answer: "elle est partie ce matin", explanation: "'Partir' takes être; participle agrees: partie." },
    ],
  },
  {
    id: "futures", name: "Futur proche vs futur simple", confusableWith: ["pc-imp"],
    hint: "Futur proche (aller + infinitif) = near/planned: « Je vais partir ». Futur simple = predictions, promises, distant plans: « Je partirai ».",
    fallback: [
      { type: "mcq", prompt: "Attends, je ___ t'aider. (right now)", options: ["vais", "irai", "vas", "aiderai"], answer: "vais", explanation: "Immediate intention → futur proche: je vais t'aider." },
      { type: "mcq", prompt: "Un jour, nous ___ en France.", options: ["habiterons", "allons habiter", "habitons", "habitions"], answer: "habiterons", explanation: "Distant, vague future ('un jour') → futur simple." },
      { type: "type", prompt: "Translate: 'It will rain tomorrow.' (pleuvoir → il pleuvra)", answer: "il pleuvra demain", explanation: "Weather prediction → futur simple: il pleuvra." },
      { type: "mcq", prompt: "Ce soir, on ___ un film.", options: ["va regarder", "regardera", "regardait", "a regardé"], answer: "va regarder", explanation: "Planned for tonight → futur proche." },
    ],
  },
  {
    id: "obj-pronouns", name: "Object pronouns (le/la/les/lui/leur)", confusableWith: ["questions"],
    hint: "Pronouns go BEFORE the verb: « Je le vois » (I see him). le/la/les = direct; lui/leur = indirect (to him/her/them). Word order changes meaning — this is where English speakers slip.",
    fallback: [
      { type: "mcq", prompt: "Tu vois Marie ? — Oui, je ___ vois.", options: ["la", "le", "lui", "les"], answer: "la", explanation: "Marie = feminine direct object → la, placed before the verb." },
      { type: "mcq", prompt: "J'ai parlé à mes parents. → Je ___ ai parlé.", options: ["leur", "les", "lui", "la"], answer: "leur", explanation: "'Parler à' → indirect object → leur." },
      { type: "type", prompt: "Translate: 'I love them.' (aimer)", answer: "je les aime", explanation: "Pronoun before the verb: je les aime — NOT 'j'aime les'." },
      { type: "mcq", prompt: "Tu ___ donnes le livre ? (to him)", options: ["lui", "le", "la", "leur"], answer: "lui", explanation: "'Donner à quelqu'un' → indirect → lui, before the verb." },
    ],
  },
  {
    id: "questions", name: "Three ways to ask questions", confusableWith: ["obj-pronouns"],
    hint: "Intonation (Tu viens ?), est-ce que (Est-ce que tu viens ?), inversion (Viens-tu ?). Spoken French overwhelmingly uses the first two.",
    fallback: [
      { type: "mcq", prompt: "Casual spoken French for 'Are you coming tonight?':", options: ["Tu viens ce soir ?", "Viens-tu ce soir ?", "Venez-vous ce soir ?", "Que viens-tu ?"], answer: "Tu viens ce soir ?", explanation: "Rising intonation on a plain statement is the most common spoken question." },
      { type: "type", prompt: "Turn into an est-ce que question: « Il travaille ici. »", answer: "est-ce qu'il travaille ici", explanation: "Est-ce que + il → Est-ce qu'il (elision) travaille ici ?" },
      { type: "mcq", prompt: "« Qu'est-ce que tu fais ? » asks…", options: ["What are you doing?", "Where are you?", "Why are you doing it?", "Who are you?"], answer: "What are you doing?", explanation: "Qu'est-ce que = 'what' (object). Sounds like 'kess-kuh'." },
      { type: "mcq", prompt: "Formal written inversion of « Vous avez l'heure » :", options: ["Avez-vous l'heure ?", "Vous avez l'heure ?", "Est-ce que vous avez l'heure ?", "L'heure avez-vous ?"], answer: "Avez-vous l'heure ?", explanation: "Inversion (verb-pronoun with hyphen) is the formal register." },
    ],
  },
];

/* Extra offline drill items per topic, pooled with each topic's fallback. */
const EXTRA_DRILLS = {
  "pc-imp": [
    { type: "mcq", prompt: "Il ___ toujours du sucre dans son café. (habit)", options: ["mettait", "a mis", "met", "mettra"], answer: "mettait", explanation: "'Toujours' + past habit → imparfait: il mettait." },
    { type: "mcq", prompt: "Soudain, quelqu'un ___ à la porte.", options: ["a frappé", "frappait", "frappe", "frappera"], answer: "a frappé", explanation: "'Soudain' signals a sudden completed event → passé composé." },
    { type: "type", prompt: "Translate: 'The weather was nice and we were happy.' (faire beau / être contents)", answer: "il faisait beau et nous étions contents", explanation: "Descriptions and states in the past → imparfait for both verbs." },
    { type: "mcq", prompt: "Pendant que je ___, le téléphone a sonné.", options: ["cuisinais", "ai cuisiné", "cuisine", "cuisinerai"], answer: "cuisinais", explanation: "'Pendant que' + ongoing background action → imparfait." },
  ],
  "aux-avoir-etre": [
    { type: "mcq", prompt: "Vous ___ venus en voiture ?", options: ["êtes", "avez", "étiez", "aviez"], answer: "êtes", explanation: "'Venir' is a movement verb → être: vous êtes venus." },
    { type: "mcq", prompt: "Elle s'___ levée très tôt ce matin.", options: ["est", "a", "était", "avait"], answer: "est", explanation: "All reflexive verbs take être: elle s'est levée." },
    { type: "type", prompt: "Translate: 'We stayed at home.' (rester, nous)", answer: "nous sommes restés à la maison", explanation: "'Rester' takes être; participle agrees: restés." },
    { type: "mcq", prompt: "J'___ perdu mes clés hier.", options: ["ai", "suis", "étais", "avais"], answer: "ai", explanation: "'Perdre' is an ordinary transitive verb → avoir: j'ai perdu." },
  ],
  futures: [
    { type: "mcq", prompt: "Regarde ces nuages ! Il ___ pleuvoir.", options: ["va", "pleuvra", "allait", "irait"], answer: "va", explanation: "Evidence right in front of you → futur proche: il va pleuvoir." },
    { type: "mcq", prompt: "Quand je serai grand, je ___ pilote.", options: ["serai", "vais être", "suis", "étais"], answer: "serai", explanation: "Distant, hypothetical future → futur simple: je serai." },
    { type: "type", prompt: "Translate: 'We'll see.' (voir, on)", answer: "on verra", explanation: "Classic futur simple set phrase: on verra." },
    { type: "mcq", prompt: "Attends-moi, j'___ dans cinq minutes !", options: ["arrive", "arriverai", "arrivais", "vais arrivé"], answer: "arrive", explanation: "Very near future often uses plain present in French: j'arrive !" },
  ],
  "obj-pronouns": [
    { type: "mcq", prompt: "Tu as téléphoné à ta mère ? — Oui, je ___ ai téléphoné.", options: ["lui", "la", "le", "leur"], answer: "lui", explanation: "'Téléphoner à' → indirect object → lui." },
    { type: "mcq", prompt: "Ces chaussures ? Je ___ achète !", options: ["les", "leur", "lui", "la"], answer: "les", explanation: "Plural direct object → les, before the verb." },
    { type: "type", prompt: "Translate: 'I gave it to him.' (donner, le)", answer: "je le lui ai donné", explanation: "Double pronouns: le (it) before lui (to him), both before the auxiliary." },
    { type: "mcq", prompt: "Tu me ___ prêtes ? (ton vélo)", options: ["le", "lui", "les", "leur"], answer: "le", explanation: "'Ton vélo' → le; order: me le prêtes." },
  ],
  questions: [
    { type: "mcq", prompt: "« Pourquoi est-ce que tu ris ? » asks…", options: ["Why are you laughing?", "What are you laughing at?", "Are you laughing?", "When did you laugh?"], answer: "Why are you laughing?", explanation: "Pourquoi = why; est-ce que keeps normal word order." },
    { type: "type", prompt: "Ask casually (intonation only): 'You saw the match?' (voir, le match)", answer: "tu as vu le match", explanation: "Spoken French: plain statement + rising intonation — Tu as vu le match ?" },
    { type: "mcq", prompt: "« Qui est-ce qui a appelé ? » asks about…", options: ["the person who called", "the thing that was called", "the time of the call", "the reason for the call"], answer: "the person who called", explanation: "Qui est-ce qui = who (subject)." },
    { type: "mcq", prompt: "Formal inversion of « Elle vient demain » :", options: ["Vient-elle demain ?", "Elle vient demain ?", "Est-ce qu'elle vient demain ?", "Demain vient-elle ?"], answer: "Vient-elle demain ?", explanation: "Verb-pronoun inversion with a hyphen is the formal register." },
  ],
};
const topicPool = (topic) => [...topic.fallback, ...(EXTRA_DRILLS[topic.id] || [])];

/* --------------------------- placement question bank ------------------------ */

const PLACEMENT_BANK = {
  A0: [
    { type: "listen", audio: "Bonjour ! Ça va ?", q: "What did you hear?", options: ["Hello! How are you?", "Goodbye, see you soon!", "Thank you very much!", "What time is it?"], answer: "Hello! How are you?" },
    { type: "listen", audio: "Merci beaucoup, au revoir.", q: "What did you hear?", options: ["Thanks a lot, goodbye.", "Please come in.", "I'm very hungry.", "Excuse me, where is it?"], answer: "Thanks a lot, goodbye." },
    { type: "grammar", q: "'Water' in French is…", options: ["l'eau", "le pain", "le vin", "la rue"], answer: "l'eau" },
    { type: "grammar", q: "To order politely you say: « Un café, ___ »", options: ["s'il vous plaît", "au revoir", "bonjour", "très bien"], answer: "s'il vous plaît" },
  ],
  A1: [
    { type: "listen", audio: "Je voudrais un café et un croissant, s'il vous plaît.", q: "What is being ordered?", options: ["A coffee and a croissant", "A tea and some bread", "Two coffees", "The bill"], answer: "A coffee and a croissant" },
    { type: "listen", audio: "La gare est à gauche, près de la banque.", q: "Where is the station?", options: ["On the left, near the bank", "On the right, far away", "Straight ahead", "Behind the bank"], answer: "On the left, near the bank" },
    { type: "grammar", q: "« Je ___ anglaise. »", options: ["suis", "es", "est", "ai"], answer: "suis" },
    { type: "grammar", q: "« Nous ___ deux enfants. »", options: ["avons", "sommes", "ont", "êtes"], answer: "avons" },
  ],
  A2: [
    { type: "listen", audio: "Hier soir, on est allés au restaurant et on a très bien mangé.", q: "What happened?", options: ["They went to a restaurant last night and ate well", "They are going to a restaurant tonight", "They cooked at home yesterday", "They didn't like the restaurant"], answer: "They went to a restaurant last night and ate well" },
    { type: "listen", audio: "Il y a un problème avec le train, du coup je vais arriver en retard.", q: "What's the message?", options: ["A train problem means they'll arrive late", "The train arrived early", "They missed the bus", "The meeting is cancelled"], answer: "A train problem means they'll arrive late" },
    { type: "grammar", q: "« Hier, elle ___ un bon film. »", options: ["a vu", "voyait", "voit", "verra"], answer: "a vu" },
    { type: "grammar", q: "« Quand j'étais petit, je ___ au foot le samedi. »", options: ["jouais", "ai joué", "joue", "jouerai"], answer: "jouais" },
  ],
  B1: [
    { type: "listen", audio: "Elle m'a dit qu'elle arriverait vers huit heures, mais je n'y crois pas trop.", q: "What did you hear?", options: ["She said she'd arrive around eight, but the speaker doubts it", "She arrived at eight exactly", "She asked what time it was", "She refuses to come at all"], answer: "She said she'd arrive around eight, but the speaker doubts it" },
    { type: "listen", audio: "Ce qui m'étonne, c'est que personne n'a rien dit, alors que tout le monde était au courant.", q: "What surprises the speaker?", options: ["Nobody said anything although everyone knew", "Everyone spoke at once", "Nobody knew about it", "Someone finally told the truth"], answer: "Nobody said anything although everyone knew" },
    { type: "grammar", q: "« Il faut que tu ___ plus tôt. »", options: ["viennes", "viens", "venir", "viendras"], answer: "viennes" },
    { type: "grammar", q: "« Tu vois Marie ? — Oui, je ___ vois souvent. »", options: ["la", "lui", "le", "leur"], answer: "la" },
  ],
};

/* ------------------------- conversation scenarios --------------------------- */

const SCENARIOS = [
  { id: "cafe", title: "Au café", emoji: "☕", desc: "Order drinks and chat with the waiter", minLevel: "A0" },
  { id: "marche", title: "Au marché", emoji: "🍎", desc: "Buy fruit and veg, ask prices", minLevel: "A0" },
  { id: "chemin", title: "Demander son chemin", emoji: "🗺️", desc: "You're lost — ask for directions", minLevel: "A1" },
  { id: "hotel", title: "À l'hôtel", emoji: "🏨", desc: "Check in and handle a problem", minLevel: "A1" },
  { id: "weekend", title: "Le week-end", emoji: "🌤️", desc: "Talk about your weekend (past tenses!)", minLevel: "A2" },
  { id: "debat", title: "Petite discussion", emoji: "💬", desc: "Opinions on daily-life topics", minLevel: "B1" },
  { id: "libre", title: "Discussion libre", emoji: "🎲", desc: "Free talk — anything you like", minLevel: "A0" },
];

/* ---------------------- authentic-audio ladder (links) ---------------------- */

const PODCAST_LADDER = [
  { name: "News in Slow French", url: "https://www.newsinslowfrench.com", level: "A2", why: "Real news, deliberately slowed — the gentlest on-ramp." },
  { name: "Coffee Break French", url: "https://coffeebreaklanguages.com/coffeebreakfrench/", level: "A2–B1", why: "Structured lessons with natural dialogue snippets." },
  { name: "InnerFrench", url: "https://innerfrench.com", level: "B1", why: "Slow-but-natural monologues on interesting topics — the classic intermediate bridge." },
  { name: "RFI — Journal en français facile", url: "https://francaisfacile.rfi.fr", level: "B1+", why: "Daily real news in simplified French, with transcripts." },
];

/* ------------------------- fallback reading passages ------------------------ */

/* Several passages per band; the library draws one at random when the AI
   generator is unavailable. */
const FALLBACK_PASSAGES = {
  A0: [
    { title: "Au café", question: "Qu'est-ce que la personne commande ?", question_answer: "Un café et un croissant.",
      sentences: [
        { fr: "Je suis au café.", en: "I am at the café." },
        { fr: "Je voudrais un café et un croissant.", en: "I would like a coffee and a croissant." },
        { fr: "Le serveur est très gentil.", en: "The waiter is very kind." },
        { fr: "Le café coûte trois euros.", en: "The coffee costs three euros." },
        { fr: "Merci, au revoir !", en: "Thank you, goodbye!" },
      ] },
    { title: "Ma famille", question: "Qui aime le café ?", question_answer: "Le mari aime le café.",
      sentences: [
        { fr: "Voici ma famille.", en: "Here is my family." },
        { fr: "Mon mari s'appelle Stuart.", en: "My husband's name is Stuart." },
        { fr: "Il aime le café, moi j'aime le thé.", en: "He likes coffee, I like tea." },
        { fr: "Nous habitons dans une petite maison.", en: "We live in a small house." },
      ] },
    { title: "Au marché", question: "Les tomates coûtent combien ?", question_answer: "Deux euros.",
      sentences: [
        { fr: "Le marché est très joli.", en: "The market is very pretty." },
        { fr: "Il y a des pommes rouges et des tomates.", en: "There are red apples and tomatoes." },
        { fr: "Les tomates coûtent deux euros.", en: "The tomatoes cost two euros." },
        { fr: "J'achète aussi du pain.", en: "I also buy some bread." },
      ] },
    { title: "Dans ma rue", question: "Qu'est-ce qu'il y a à gauche ?", question_answer: "Une boulangerie.",
      sentences: [
        { fr: "J'habite dans une rue calme.", en: "I live on a quiet street." },
        { fr: "À gauche, il y a une boulangerie.", en: "On the left, there is a bakery." },
        { fr: "À droite, il y a un petit parc.", en: "On the right, there is a small park." },
        { fr: "J'aime beaucoup ma rue.", en: "I really like my street." },
      ] },
    { title: "Le petit déjeuner", question: "Qu'est-ce que la personne mange le matin ?", question_answer: "Du pain avec du beurre.",
      sentences: [
        { fr: "Le matin, je mange du pain avec du beurre.", en: "In the morning, I eat bread with butter." },
        { fr: "Je bois un grand thé.", en: "I drink a big tea." },
        { fr: "Mon mari boit deux cafés !", en: "My husband drinks two coffees!" },
        { fr: "Après, nous sommes prêts pour la journée.", en: "After that, we are ready for the day." },
      ] },
    { title: "Au parc", question: "Quel temps fait-il ?", question_answer: "Il fait beau.",
      sentences: [
        { fr: "Aujourd'hui, il fait beau.", en: "Today, the weather is nice." },
        { fr: "Je suis au parc avec une amie.", en: "I am at the park with a friend." },
        { fr: "Il y a des enfants et des chiens.", en: "There are children and dogs." },
        { fr: "Nous mangeons une glace.", en: "We are eating an ice cream." },
      ] },
    { title: "Ma maison", question: "Où est le jardin ?", question_answer: "Derrière la maison.",
      sentences: [
        { fr: "Ma maison est petite mais jolie.", en: "My house is small but pretty." },
        { fr: "Il y a deux chambres et une cuisine.", en: "There are two bedrooms and a kitchen." },
        { fr: "Derrière la maison, il y a un jardin.", en: "Behind the house, there is a garden." },
        { fr: "Le soir, nous mangeons dans le jardin.", en: "In the evening, we eat in the garden." },
      ] },
  ],
  A1: [
    { title: "Ma petite ville", question: "Où est le marché ?", question_answer: "Près de la gare.",
      sentences: [
        { fr: "J'habite dans une petite ville.", en: "I live in a small town." },
        { fr: "Il y a un marché près de la gare.", en: "There is a market near the station." },
        { fr: "Le matin, j'achète du pain et des fruits.", en: "In the morning, I buy bread and fruit." },
        { fr: "L'après-midi, je travaille à la maison.", en: "In the afternoon, I work at home." },
        { fr: "Le soir, je regarde la télé avec mon mari.", en: "In the evening, I watch TV with my husband." },
      ] },
    { title: "Une journée au travail", question: "À quelle heure commence le travail ?", question_answer: "À neuf heures.",
      sentences: [
        { fr: "Je commence le travail à neuf heures.", en: "I start work at nine o'clock." },
        { fr: "À midi, je mange avec mes collègues.", en: "At noon, I eat with my colleagues." },
        { fr: "Nous parlons de tout et de rien.", en: "We talk about everything and nothing." },
        { fr: "Je finis à cinq heures et je rentre à pied.", en: "I finish at five and walk home." },
      ] },
    { title: "Les courses du samedi", question: "Pourquoi la personne aime le fromager ?", question_answer: "Parce qu'il est drôle (il raconte des blagues).",
      sentences: [
        { fr: "Le samedi, nous faisons les courses ensemble.", en: "On Saturdays, we do the shopping together." },
        { fr: "Nous achetons des légumes, du poisson et du fromage.", en: "We buy vegetables, fish and cheese." },
        { fr: "Le fromager raconte toujours des blagues.", en: "The cheesemonger always tells jokes." },
        { fr: "C'est pour ça que j'aime bien aller chez lui.", en: "That's why I like going to his shop." },
      ] },
    { title: "Mon chat", question: "Où dort le chat ?", question_answer: "Sur le canapé.",
      sentences: [
        { fr: "Nous avons un chat qui s'appelle Minou.", en: "We have a cat called Minou." },
        { fr: "Il dort toute la journée sur le canapé.", en: "He sleeps all day on the sofa." },
        { fr: "Le soir, il demande à manger à six heures exactement.", en: "In the evening, he asks for food at exactly six o'clock." },
        { fr: "C'est le vrai chef de la maison.", en: "He is the real boss of the house." },
      ] },
    { title: "Une carte postale", question: "Quel temps fait-il en Bretagne ?", question_answer: "Il fait beau mais il y a du vent.",
      sentences: [
        { fr: "Chère Sophie, nous sommes en Bretagne.", en: "Dear Sophie, we are in Brittany." },
        { fr: "Il fait beau, mais il y a beaucoup de vent.", en: "The weather is nice, but it's very windy." },
        { fr: "Hier, nous avons mangé des crêpes délicieuses.", en: "Yesterday, we ate delicious crêpes." },
        { fr: "À bientôt, grosses bises !", en: "See you soon, big kisses!" },
      ] },
    { title: "Au restaurant", question: "Qu'est-ce que la personne choisit ?", question_answer: "Le poisson avec des légumes.",
      sentences: [
        { fr: "Ce soir, nous dînons au restaurant.", en: "Tonight, we are having dinner at a restaurant." },
        { fr: "Le serveur donne le menu.", en: "The waiter gives us the menu." },
        { fr: "Je choisis le poisson avec des légumes.", en: "I choose the fish with vegetables." },
        { fr: "Mon mari prend le poulet, comme toujours !", en: "My husband has the chicken, as always!" },
      ] },
    { title: "Le dimanche matin", question: "Qu'est-ce qu'ils achètent à la boulangerie ?", question_answer: "Des croissants.",
      sentences: [
        { fr: "Le dimanche matin, nous restons à la maison.", en: "On Sunday mornings, we stay at home." },
        { fr: "Mon mari va à la boulangerie acheter des croissants.", en: "My husband goes to the bakery to buy croissants." },
        { fr: "Nous prenons le petit déjeuner dans le jardin.", en: "We have breakfast in the garden." },
        { fr: "C'est mon moment préféré de la semaine.", en: "It's my favourite moment of the week." },
      ] },
    { title: "Les quatre saisons", question: "Quelle saison la personne préfère-t-elle ?", question_answer: "Le printemps.",
      sentences: [
        { fr: "En été, il fait chaud et les jours sont longs.", en: "In summer, it's hot and the days are long." },
        { fr: "En hiver, il fait froid et il pleut souvent.", en: "In winter, it's cold and it often rains." },
        { fr: "Moi, je préfère le printemps.", en: "Me, I prefer spring." },
        { fr: "Les fleurs sont belles et l'air sent bon.", en: "The flowers are beautiful and the air smells good." },
      ] },
  ],
  A2: [
    { title: "Un week-end à Lyon", question: "Qu'est-ce qu'ils ont fait samedi soir ?", question_answer: "Ils ont mangé dans un bouchon lyonnais.",
      sentences: [
        { fr: "Le mois dernier, nous sommes allés à Lyon.", en: "Last month, we went to Lyon." },
        { fr: "Samedi matin, on a visité la vieille ville.", en: "Saturday morning, we visited the old town." },
        { fr: "Il faisait beau et les rues étaient pleines de monde.", en: "The weather was nice and the streets were full of people." },
        { fr: "Samedi soir, on a mangé dans un bouchon lyonnais.", en: "Saturday evening, we ate in a traditional Lyon restaurant." },
        { fr: "C'était délicieux, mais un peu cher.", en: "It was delicious, but a bit expensive." },
      ] },
    { title: "La nouvelle boulangerie", question: "Pourquoi y a-t-il la queue ?", question_answer: "Parce que le pain est excellent (tout le quartier veut y acheter son pain).",
      sentences: [
        { fr: "Une nouvelle boulangerie a ouvert dans notre rue.", en: "A new bakery has opened on our street." },
        { fr: "Depuis, il y a la queue tous les matins.", en: "Since then, there's been a queue every morning." },
        { fr: "Le boulanger travaillait avant dans un grand restaurant.", en: "The baker used to work in a big restaurant." },
        { fr: "Son pain est tellement bon que tout le quartier en parle.", en: "His bread is so good the whole neighbourhood is talking about it." },
      ] },
    { title: "Un voyage en train", question: "Pourquoi le voyage a-t-il été long ?", question_answer: "Parce que le train est resté bloqué à cause d'un problème technique.",
      sentences: [
        { fr: "La semaine dernière, j'ai pris le train pour Paris.", en: "Last week, I took the train to Paris." },
        { fr: "Au début, tout allait bien.", en: "At first, everything was fine." },
        { fr: "Mais le train est resté bloqué une heure à cause d'un problème technique.", en: "But the train was stuck for an hour because of a technical problem." },
        { fr: "Heureusement, ma voisine de siège était très bavarde et sympathique.", en: "Luckily, the woman next to me was very chatty and friendly." },
      ] },
    { title: "La fête des voisins", question: "Qu'est-ce que chacun a apporté ?", question_answer: "Un plat (quelque chose à manger).",
      sentences: [
        { fr: "Samedi, on a organisé une fête avec les voisins.", en: "On Saturday, we organised a party with the neighbours." },
        { fr: "Chacun a apporté un plat de son pays.", en: "Everyone brought a dish from their country." },
        { fr: "On a goûté des choses qu'on ne connaissait pas du tout.", en: "We tasted things we didn't know at all." },
        { fr: "On a fini la soirée en chantant dans la cour.", en: "We ended the evening singing in the courtyard." },
      ] },
    { title: "Mon premier appartement", question: "Comment était le premier appartement ?", question_answer: "Minuscule (très petit), mais la personne en garde un bon souvenir.",
      sentences: [
        { fr: "Quand j'avais vingt ans, j'ai loué mon premier appartement.", en: "When I was twenty, I rented my first flat." },
        { fr: "Il était minuscule : la cuisine était dans le salon.", en: "It was tiny: the kitchen was in the living room." },
        { fr: "L'hiver, il faisait froid parce que les fenêtres fermaient mal.", en: "In winter it was cold because the windows didn't close properly." },
        { fr: "Pourtant, j'en garde un très bon souvenir.", en: "And yet I have very fond memories of it." },
      ] },
    { title: "Une mauvaise journée", question: "Qu'est-ce qui s'est passé avec le bus ?", question_answer: "La personne a raté le bus (il est parti juste devant elle).",
      sentences: [
        { fr: "Hier, tout est allé de travers.", en: "Yesterday, everything went wrong." },
        { fr: "Je me suis levée en retard et j'ai raté le bus.", en: "I got up late and missed the bus." },
        { fr: "Il est parti juste devant moi, évidemment.", en: "It left right in front of me, of course." },
        { fr: "Le soir, j'ai ri de cette journée avec mon mari.", en: "In the evening, I laughed about the day with my husband." },
      ] },
    { title: "Le marché de Noël", question: "Qu'est-ce qu'ils ont bu au marché ?", question_answer: "Du vin chaud.",
      sentences: [
        { fr: "En décembre, nous sommes allés au marché de Noël.", en: "In December, we went to the Christmas market." },
        { fr: "Il y avait des lumières partout et ça sentait la cannelle.", en: "There were lights everywhere and it smelled of cinnamon." },
        { fr: "On a bu du vin chaud pour se réchauffer.", en: "We drank mulled wine to warm up." },
        { fr: "J'ai acheté des cadeaux pour toute la famille.", en: "I bought presents for the whole family." },
      ] },
    { title: "Un dimanche à la campagne", question: "Qu'est-ce qu'ils ont fait après le déjeuner ?", question_answer: "Ils ont fait une longue promenade dans la forêt.",
      sentences: [
        { fr: "Dimanche dernier, des amis nous ont invités à la campagne.", en: "Last Sunday, some friends invited us to the countryside." },
        { fr: "On a déjeuné dehors, sous un grand arbre.", en: "We had lunch outside, under a big tree." },
        { fr: "Après le déjeuner, on a fait une longue promenade dans la forêt.", en: "After lunch, we went for a long walk in the forest." },
        { fr: "On est rentrés fatigués mais heureux.", en: "We came home tired but happy." },
      ] },
  ],
  B1: [
    { title: "Le télétravail", question: "Quel est l'inconvénient mentionné ?", question_answer: "On se sent parfois isolé.",
      sentences: [
        { fr: "Depuis quelques années, le télétravail s'est beaucoup développé.", en: "In recent years, remote work has grown a lot." },
        { fr: "Beaucoup de gens apprécient de ne plus perdre de temps dans les transports.", en: "Many people appreciate no longer wasting time commuting." },
        { fr: "Pourtant, certains trouvent qu'on se sent parfois isolé.", en: "However, some find that you sometimes feel isolated." },
        { fr: "Ce qui compte, c'est de trouver un équilibre qui convient à chacun.", en: "What matters is finding a balance that suits each person." },
      ] },
    { title: "Les réseaux sociaux", question: "Quel est le paradoxe évoqué ?", question_answer: "Ils nous connectent au monde entier, mais on parle moins aux gens autour de nous.",
      sentences: [
        { fr: "Les réseaux sociaux nous permettent de rester en contact avec le monde entier.", en: "Social media lets us stay in touch with the whole world." },
        { fr: "Paradoxalement, on parle parfois moins aux gens qui nous entourent.", en: "Paradoxically, we sometimes talk less to the people around us." },
        { fr: "Certains décident donc de couper leur téléphone le soir.", en: "So some people decide to switch off their phone in the evening." },
        { fr: "D'après eux, on redécouvre vite le plaisir d'une vraie conversation.", en: "According to them, you quickly rediscover the pleasure of a real conversation." },
      ] },
    { title: "Apprendre une langue", question: "Qu'est-ce qui compte plus que le talent, selon le texte ?", question_answer: "La régularité (un peu tous les jours).",
      sentences: [
        { fr: "On croit souvent qu'il faut un don pour apprendre une langue.", en: "People often think you need a gift to learn a language." },
        { fr: "En réalité, la régularité compte beaucoup plus que le talent.", en: "In reality, consistency matters much more than talent." },
        { fr: "Dix minutes par jour valent mieux que deux heures le dimanche.", en: "Ten minutes a day beats two hours on a Sunday." },
        { fr: "Et surtout, il ne faut pas avoir peur de faire des erreurs.", en: "Above all, you mustn't be afraid of making mistakes." },
      ] },
    { title: "Ville ou campagne ?", question: "Pourquoi certains hésitent-ils à quitter la ville ?", question_answer: "Ils ont peur de s'ennuyer et de perdre leur vie sociale.",
      sentences: [
        { fr: "De plus en plus de citadins rêvent de s'installer à la campagne.", en: "More and more city dwellers dream of moving to the countryside." },
        { fr: "Ils cherchent le calme, l'espace et un air plus pur.", en: "They're looking for calm, space and cleaner air." },
        { fr: "Mais beaucoup hésitent, par peur de s'ennuyer ou de perdre leur vie sociale.", en: "But many hesitate, fearing boredom or losing their social life." },
        { fr: "Au fond, l'idéal serait peut-être d'avoir un pied dans chaque monde.", en: "Deep down, the ideal might be to have a foot in each world." },
      ] },
    { title: "Le sport après cinquante ans", question: "Que conseillent les médecins ?", question_answer: "Une activité douce et régulière, comme la marche ou la natation.",
      sentences: [
        { fr: "On entend souvent qu'il serait trop tard pour se mettre au sport après cinquante ans.", en: "You often hear it's supposedly too late to take up sport after fifty." },
        { fr: "Les médecins affirment exactement le contraire.", en: "Doctors say exactly the opposite." },
        { fr: "Ils conseillent une activité douce et régulière, comme la marche ou la natation.", en: "They recommend gentle, regular activity, like walking or swimming." },
        { fr: "L'essentiel, c'est d'y prendre du plaisir pour continuer longtemps.", en: "The key is to enjoy it so you keep going for a long time." },
      ] },
    { title: "Les repas de famille", question: "Pourquoi les repas durent-ils si longtemps en France ?", question_answer: "Parce qu'on discute de tout à table — le repas est un moment social.",
      sentences: [
        { fr: "En France, les repas de famille peuvent durer des heures.", en: "In France, family meals can last for hours." },
        { fr: "Ce n'est pas seulement une question de nourriture.", en: "It's not only about the food." },
        { fr: "À table, on discute de tout : de politique, des voisins, des souvenirs.", en: "At the table, people discuss everything: politics, the neighbours, memories." },
        { fr: "Le repas est avant tout un moment passé ensemble.", en: "The meal is above all a moment spent together." },
      ] },
    { title: "Voyager autrement", question: "Que propose le texte à la place des grandes villes touristiques ?", question_answer: "Découvrir les petits villages et prendre le temps de rencontrer les habitants.",
      sentences: [
        { fr: "Beaucoup de voyageurs visitent toujours les mêmes grandes villes.", en: "Many travellers always visit the same big cities." },
        { fr: "Pourtant, les petits villages ont souvent plus de charme.", en: "Yet small villages often have more charm." },
        { fr: "On y prend le temps de discuter avec les habitants.", en: "There, you take the time to chat with the locals." },
        { fr: "Ce sont souvent ces rencontres dont on se souvient le plus.", en: "It's often those encounters you remember most." },
      ] },
  ],
};

/* --------------------------- fallback exercises ----------------------------- */

/* Larger per-type pools; each generated fallback drill samples from these. */
const EXERCISE_POOLS = {
  cloze: { instructions: "Fill the gap.", perDrill: 3, items: [
    { prompt: "Hier, nous ___ (aller) au cinéma.", answer: "sommes allés", explanation: "'Aller' takes être in the passé composé; nous → sommes allés." },
    { prompt: "Je ___ (vouloir) un café, s'il vous plaît.", answer: "voudrais", explanation: "Polite requests use the conditional: je voudrais." },
    { prompt: "Quand j'étais petite, je ___ (jouer) dans le jardin.", answer: "jouais", explanation: "Habitual past → imparfait: je jouais." },
    { prompt: "Elle ___ (être) très fatiguée hier soir.", answer: "était", explanation: "A state/description in the past → imparfait: elle était." },
    { prompt: "Demain, il ___ (faire) beau sur toute la France.", answer: "fera", explanation: "Weather prediction → futur simple: il fera." },
    { prompt: "Nous ___ (avoir) deux enfants et un chat.", answer: "avons", explanation: "Present of avoir: nous avons." },
    { prompt: "Tu ___ (pouvoir) m'aider, s'il te plaît ?", answer: "peux", explanation: "Present of pouvoir: tu peux." },
    { prompt: "Ils ne ___ (comprendre) pas la question.", answer: "comprennent", explanation: "Present of comprendre: ils comprennent." },
  ]},
  translation: { instructions: "Translate into French.", perDrill: 3, items: [
    { prompt: "I don't know where the station is.", answer: "je ne sais pas où est la gare", explanation: "Je ne sais pas + où est…" },
    { prompt: "We are going to eat at eight.", answer: "on va manger à huit heures", explanation: "Futur proche: aller + infinitive." },
    { prompt: "She left this morning.", answer: "elle est partie ce matin", explanation: "'Partir' takes être; the participle agrees: partie." },
    { prompt: "I see them every day.", answer: "je les vois tous les jours", explanation: "Object pronoun before the verb: je les vois." },
    { prompt: "It was raining when we arrived.", answer: "il pleuvait quand nous sommes arrivés", explanation: "Background action → imparfait; the arrival → passé composé." },
    { prompt: "Can you help me, please?", answer: "pouvez-vous m'aider, s'il vous plaît", explanation: "Polite request with inversion: pouvez-vous + m'aider." },
    { prompt: "There are too many people here.", answer: "il y a trop de monde ici", explanation: "'Trop de monde' = too many people; 'il y a' introduces it." },
    { prompt: "We used to live in a small village.", answer: "nous habitions dans un petit village", explanation: "'Used to' → imparfait: nous habitions." },
  ]},
  dictation: { instructions: "Type exactly what you hear.", perDrill: 2, items: [
    { prompt: "Écoutez puis écrivez.", answer: "Il y a beaucoup de monde dans les rues ce soir.", explanation: "'Il y a' collapses to 'ya'; 'dans les' links smoothly — listen for the rhythm, not the words." },
    { prompt: "Écoutez puis écrivez.", answer: "Mes amis arrivent à deux heures et demie.", explanation: "Liaisons everywhere: mes‿amis, deux‿heures — the Z sounds are the word boundaries." },
    { prompt: "Écoutez puis écrivez.", answer: "On est allés en Italie avec des amis.", explanation: "'On est allés' chains into o-nay-ta-lay; 'des amis' → day-za-mee." },
    { prompt: "Écoutez puis écrivez.", answer: "Je ne sais pas s'il va venir ce soir.", explanation: "'Je ne sais pas' shrinks toward 'ché pa'; s'il = si + il elided." },
    { prompt: "Écoutez puis écrivez.", answer: "Elle a acheté un petit appartement en ville.", explanation: "petit‿appartement links t to the vowel: puh-tee-ta-par-tuh-mahn." },
    { prompt: "Écoutez puis écrivez.", answer: "Nous avons attendu le bus pendant une heure.", explanation: "nous‿avons‿attendu is one smooth ribbon — three words, no gaps." },
  ]},
  reorder: { instructions: "Rebuild the sentence.", perDrill: 3, items: [
    { prompt: "Put the words in order:", answer: "je les vois souvent le week-end", words: ["je", "les", "vois", "souvent", "le", "week-end"], explanation: "Object pronoun 'les' goes BEFORE the verb: je les vois." },
    { prompt: "Put the words in order:", answer: "elle ne mange pas de viande", words: ["elle", "ne", "mange", "pas", "de", "viande"], explanation: "Negation wraps the verb: ne … pas, and 'de' replaces 'la' after a negative." },
    { prompt: "Put the words in order:", answer: "nous allons partir demain matin", words: ["nous", "allons", "partir", "demain", "matin"], explanation: "Futur proche: aller conjugated + infinitive, time phrase at the end." },
    { prompt: "Put the words in order:", answer: "est-ce que tu as vu mes clés", words: ["est-ce", "que", "tu", "as", "vu", "mes", "clés"], explanation: "Est-ce que + normal word order turns a statement into a question." },
    { prompt: "Put the words in order:", answer: "il lui a donné un cadeau", words: ["il", "lui", "a", "donné", "un", "cadeau"], explanation: "Indirect pronoun 'lui' sits before the auxiliary: il lui a donné." },
    { prompt: "Put the words in order:", answer: "on y va tous les étés", words: ["on", "y", "va", "tous", "les", "étés"], explanation: "'y' (= there) goes before the verb: on y va." },
  ]},
};

function fallbackExercise(type) {
  const pool = EXERCISE_POOLS[type] || EXERCISE_POOLS.cloze;
  return { type, instructions: pool.instructions, items: shuffle(pool.items).slice(0, pool.perDrill) };
}

/* ------------------------------ AI prompt builders -------------------------- */

const LEVEL_DESC = {
  A0: "an absolute beginner (knows only hello/goodbye); use only the most frequent words, 4-8 word sentences",
  A1: "a beginner (CEFR A1); high-frequency vocabulary, simple present tense, short sentences",
  A2: "a low-intermediate learner (CEFR A2); everyday vocabulary, passé composé and imparfait, natural but clear",
  B1: "an intermediate learner (CEFR B1); natural spoken French with common idioms, varied tenses, spoken reductions",
  B2: "an upper-intermediate learner (CEFR B2); fully natural French",
};

const SYS = {
  unit: (unit, level) =>
    `You are a French teacher writing one micro-lesson for ${LEVEL_DESC[level] || LEVEL_DESC.A1}. Topic: "${unit.grammar.point}" in the scenario "${unit.scenario}". Michel-Thomas style: build from patterns the learner can reason about, one concept at a time, warm and encouraging. JSON shape: {"intro": string (max 90 words, English, explains the pattern simply), "patterns": [{"fr": string, "en": string}] (4 items, must use the target structure), "practice": [{"prompt": string (English instruction asking the learner to PRODUCE a French sentence), "answer": string (the French), "hint": string}] (4 items, blocked practice of ONLY this structure), "task_question": string (English: one open task in the scenario forcing the learner to produce 1-2 French sentences)}`,
  taskFeedback: (level) =>
    `You are a kind but honest French teacher. The learner is ${LEVEL_DESC[level] || LEVEL_DESC.A1}. Evaluate their French production for the given task. Do NOT silently fix errors — name them. JSON: {"score": 1-5, "praise": string (one specific thing they got right), "corrections": [{"you_said": string, "fix": string, "why": string}] (empty array if perfect), "model_answer": string (a natural model answer)}`,
  passage: (level, topic) =>
    `Write a short French reading passage for ${LEVEL_DESC[level] || LEVEL_DESC.A1}. Topic: ${topic || "everyday life in France"}. 4-6 sentences. JSON: {"title": string (French), "sentences": [{"fr": string, "en": string}], "question": string (one comprehension question in French), "question_answer": string (short French answer)}`,
  clip: (rung) =>
    `Write a short spoken-style French monologue for a listening exercise. Learner level: ${LEVEL_DESC[rung.level.replace("+", "")] || LEVEL_DESC.A2}. Topic: ${rung.topicHint}. 2-4 sentences, 25-45 words total, natural SPOKEN register (use il y a, du coup, on, elisions where natural for the level). It must be roughly 90-95% comprehensible at this level. JSON: {"text": string (the French), "en": string (English translation)}`,
  gaps: () =>
    `You are a French phonetics coach specialising in connected speech. The learner heard a clip and missed specific segments in dictation. For EACH segment explain WHY it was hard to hear: identify the exact liaison, elision, enchaînement or spoken reduction that hid the word boundary, and spell out how it actually sounds using English-friendly syllables (e.g. "les amis" → "lay-za-mee"). Be concrete and specific to the words given, not generic. JSON: {"gaps": [{"segment": string, "phenomenon": "liaison"|"elision"|"enchaînement"|"reduction"|"other", "sounds_like": string, "why": string (max 45 words)}]}`,
  retell: (level) =>
    `You are a French teacher. The learner listened to a French clip and retold it in their own French words. Compare their retell to the original. Be honest — do not silently fix errors. JSON: {"score": 1-5, "feedback": string (English, max 60 words, what they captured and what they missed), "corrections": [{"you_said": string, "fix": string, "why": string}] (up to 3), "better_version": string (a natural French retell at ${LEVEL_DESC[level] || LEVEL_DESC.A2} level)}`,
  conversation: (user, scenario, strictness) => {
    const strict = strictness === "strict"
      ? "STRICT mode: correct EVERY error including word-order errors that change meaning, gently but explicitly, before continuing the conversation. Never pretend you understood a garbled sentence."
      : strictness === "lenient"
      ? "LENIENT mode: only flag errors that block communication; keep the flow warm and encouraging."
      : "NORMAL mode: let trivial slips pass, but flag real errors — especially word-order and tense errors that change meaning. Do not silently 'fix' what they said.";
    return `You are a friendly native French speaker having a conversation with ${user.name}, ${LEVEL_DESC[user.level] || LEVEL_DESC.A1}. Scenario: ${scenario.title} — ${scenario.desc}. Speak natural French AT THEIR LEVEL (i+1: mostly comprehensible, a little stretch). ${strict} Keep replies to 1-3 short sentences. JSON every turn: {"reply_fr": string, "reply_en": string (translation), "note": string|null (English: if their last message had an error worth flagging, name it briefly; else null)${user.track === "foundations" ? ', "suggestions": [{"fr": string, "en": string}] (3 short things they could plausibly say next)' : ""}}`;
  },
  report: (level) =>
    `You are a French teacher writing an end-of-conversation correction report. Review the learner's messages (level: ${LEVEL_DESC[level] || LEVEL_DESC.A1}). List real errors with fixes and short explanations — word order, gender, tense, vocabulary. Be specific, honest, kind. JSON: {"summary": string (English, max 50 words, overall assessment), "corrections": [{"you_said": string, "fix": string, "why": string (max 25 words)}] (up to 6; empty if truly clean), "wins": [string] (1-3 things they did well)}`,
  exercise: (focus, type, level) =>
    `You are a French exercise generator for ${LEVEL_DESC[level] || LEVEL_DESC.A1}. Focus: ${focus}. Create a "${type}" drill. Types: cloze (sentence with ___ gap), translation (EN→FR), dictation (one French sentence to be read aloud and transcribed), reorder (scrambled words to reorder). JSON: {"type": "${type}", "instructions": string (English, one line), "items": [{"prompt": string, "answer": string (the expected French)${type === "reorder" ? ', "words": [string] (the answer tokens, scrambled)' : ""}, "explanation": string (English, why this is the answer, max 30 words)}] (3 items, 1 for dictation)}`,
  wordLookup: () =>
    `You are a French-English dictionary. Given a French word (possibly inflected) and its sentence context, return JSON: {"word": string (the word as given), "lemma": string (dictionary form with article if noun), "meaning": string (concise English), "example_fr": string (short new example), "example_en": string, "glyph": string (one emoji that depicts it, or "" if abstract)}`,
  placementProbe: () => "",
};

/* ========================================================================== */
/*  PROFILES, STREAK & PROGRESS MODEL                                         */
/* ========================================================================== */

function defaultProfile(id, name) {
  return {
    id, name,
    emoji: "🙂", theme: "indigo",
    level: "A0", track: "foundations", placed: false,
    settings: { strictness: "normal", dualSubs: true, ttsRate: 0.95, reminders: false, reminderTime: "18:00", shareProgress: true },
    deck: [],
    streak: { days: {}, lastRepairWeek: null },
    minutes: 0,
    lessons: {}, phonicsDone: {}, csDone: {},
    trainer: { rung: 1, clipsDone: 0, relistens: 0, history: [] },
    drills: {},
    stats: { dictation: [], writing: [], speaking: [], reading: 0, conversations: 0, reviews: 0 },
  };
}

function seedProfiles() {
  const stuart = {
    ...defaultProfile("stuart", "Stuart"),
    emoji: "🎧", theme: "indigo", level: "A2", track: "listening",
    settings: { strictness: "strict", dualSubs: false, ttsRate: 0.95, reminders: false, reminderTime: "18:00", shareProgress: true },
    deck: STUART_SEED_CARDS.map((c) => newCard({ ...c, source: "seed" })),
  };
  const carol = {
    ...defaultProfile("carol", "Carol"),
    emoji: "🌻", theme: "amber", level: "A0", track: "foundations",
    settings: { strictness: "lenient", dualSubs: true, ttsRate: 0.85, reminders: false, reminderTime: "18:00", shareProgress: true },
    deck: CAROL_SEED_CARDS.map((c) => newCard({ ...c, source: "seed" })),
  };
  return [stuart, carol];
}

/* Flexible streak: a day only counts once a real threshold is crossed
   (finished review session, completed lesson/clip/conversation) — never one
   tap. One free repair per week patches a single missed day. */
function computeStreak(days) {
  let n = 0;
  let offset = days[todayKey()] ? 0 : -1;
  while (days[dateKeyOffset(offset)]) { n++; offset--; }
  return n;
}
function repairAvailable(u) {
  const missedYesterday = !u.streak.days[dateKeyOffset(-1)] && u.streak.days[dateKeyOffset(-2)];
  const unusedThisWeek = u.streak.lastRepairWeek !== weekKey();
  return missedYesterday && unusedThisWeek;
}

/* Honest five-skill estimate: CEFR placement baseline blended with actual
   measured activity. Labelled as an estimate in the UI. */
function skillScores(u) {
  const base = CEFR_BASE[u.level] ?? 10;
  const avg = (arr) => (arr && arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);
  const recent = (arr, n) => avg((arr || []).slice(-n));
  const dict = recent(u.stats.dictation.map((d) => d.acc), 8);
  const writ = recent(u.stats.writing, 10);
  const spk = recent(u.stats.speaking, 10);
  const vocabSignal = clamp((matureCount(u.deck) / 250) * 100, 0, 100);
  const readSignal = clamp(u.stats.reading * 7, 0, 100);
  const blend = (signal) => (signal === null ? base : Math.round(clamp(base * 0.45 + signal * 0.55, 0, 100)));
  return {
    listening: blend(dict),
    speaking: blend(spk),
    reading: blend(u.stats.reading ? Math.max(base, readSignal) : null),
    writing: blend(writ),
    vocabulary: Math.round(clamp(base * 0.35 + vocabSignal * 0.65, 0, 100)),
  };
}

function nextAction(u) {
  const due = dueCards(u.deck).length;
  if (due > 0) return { view: "review", label: `Review ${due} due card${due > 1 ? "s" : ""}`, why: "Spaced repetition works only if reviews happen on schedule." };
  if (u.track === "listening") return { view: "trainer", label: "Listening trainer — next clip", why: "Your bottleneck is the ear. Little and often beats rare marathons." };
  const next = FOUNDATIONS_UNITS.find((un) => !u.lessons[un.id]);
  if (next) return { view: "unit", params: { unitId: next.id }, label: `Next lesson: ${next.title}`, why: "One new structure at a time." };
  return { view: "library", label: "Read & listen to a new passage", why: "Fresh comprehensible input at your level." };
}

/* ========================================================================== */
/*  UI PRIMITIVES                                                             */
/* ========================================================================== */

const THEME = {
  amber: { grad: "from-amber-400 to-rose-400", soft: "bg-amber-50", ring: "ring-amber-300", text: "text-amber-700", btn: "bg-amber-500 hover:bg-amber-600 active:bg-amber-700", chip: "bg-amber-100 text-amber-800", bar: "bg-amber-500", accentL: "border-l-4 border-l-amber-400", edge: "border-amber-200", numChip: "bg-amber-100 text-amber-700" },
  indigo: { grad: "from-indigo-500 to-sky-500", soft: "bg-indigo-50", ring: "ring-indigo-300", text: "text-indigo-700", btn: "bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800", chip: "bg-indigo-100 text-indigo-800", bar: "bg-indigo-500", accentL: "border-l-4 border-l-indigo-400", edge: "border-indigo-200", numChip: "bg-indigo-100 text-indigo-700" },
};
const themeOf = (u) => THEME[u?.theme] || THEME.indigo;

function Btn({ children, onClick, variant = "primary", theme = "indigo", disabled, className = "", small }) {
  const t = THEME[theme] || THEME.indigo;
  const base = `select-none rounded-2xl font-semibold transition-all duration-150 disabled:opacity-40 disabled:pointer-events-none ${small ? "px-4 py-2 text-sm min-h-11" : "px-5 py-3 min-h-12"}`;
  const styles = {
    primary: `${t.btn} text-white shadow-sm`,
    secondary: "bg-white text-stone-800 border border-stone-300 hover:bg-stone-50 active:bg-stone-100",
    ghost: "bg-transparent text-stone-600 hover:bg-stone-100",
    danger: "bg-rose-600 hover:bg-rose-700 text-white",
    success: "bg-emerald-600 hover:bg-emerald-700 text-white",
  };
  return (
    <button onClick={onClick} disabled={disabled} className={`${base} ${styles[variant]} ${className}`}>
      {children}
    </button>
  );
}

function Card({ children, className = "" }) {
  return <div className={`bg-white rounded-3xl border border-stone-200 shadow-sm ${className}`}>{children}</div>;
}

function Chip({ children, tone = "stone" }) {
  const tones = {
    stone: "bg-stone-100 text-stone-700", green: "bg-emerald-100 text-emerald-800", red: "bg-rose-100 text-rose-800",
    amber: "bg-amber-100 text-amber-800", indigo: "bg-indigo-100 text-indigo-800", sky: "bg-sky-100 text-sky-800",
  };
  return <span className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-xs font-semibold ${tones[tone]}`}>{children}</span>;
}

function Bar({ pct, color = "bg-indigo-500", label, right }) {
  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium text-stone-700">{label}</span>
        <span className="text-stone-500">{right}</span>
      </div>
      <div className="h-3 w-full rounded-full bg-stone-200 overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all duration-700`} style={{ width: `${clamp(pct, 2, 100)}%` }} />
      </div>
    </div>
  );
}

function Spinner({ label = "Generating…" }) {
  return (
    <div className="flex items-center gap-3 text-stone-500 py-6 justify-center">
      <div className="w-5 h-5 rounded-full border-2 border-stone-300 border-t-indigo-500 animate-spin" />
      <span className="text-sm">{label}</span>
    </div>
  );
}

function SpeakerBtn({ text, rate = 0.95, theme = "indigo", label = "Listen", big }) {
  const [playing, setPlaying] = useState(false);
  const play = () => {
    if (playing) { stopSpeaking(); setPlaying(false); return; }
    setPlaying(true);
    speak(text, rate, () => setPlaying(false));
  };
  if (!ttsSupported()) return <Chip tone="stone">🔇 audio unavailable</Chip>;
  return (
    <Btn onClick={play} variant={big ? "primary" : "secondary"} theme={theme} small={!big}>
      {playing ? "◼ Stop" : `🔊 ${label}`}
    </Btn>
  );
}

/* Microphone capture button. Falls back to null render when SR unsupported —
   callers must always offer a typed / self-assessed path too. */
function MicBtn({ onText, theme = "indigo", label = "Speak" }) {
  const [listening, setListening] = useState(false);
  const recRef = useRef(null);
  useEffect(() => () => { try { recRef.current && recRef.current.abort && recRef.current.abort(); } catch (e) {} }, []);
  if (!srSupported()) return null;
  const toggle = () => {
    if (listening) { try { recRef.current && recRef.current.stop(); } catch (e) {} setListening(false); return; }
    stopSpeaking();
    const r = makeRecognizer((t) => onText(t), () => setListening(false));
    if (!r) return;
    recRef.current = r;
    setListening(true);
    try { r.start(); } catch (e) { setListening(false); }
  };
  return (
    <Btn onClick={toggle} variant={listening ? "danger" : "secondary"} theme={theme} small>
      {listening ? "◼ Listening…" : `🎙️ ${label}`}
    </Btn>
  );
}

function TopBar({ u, go, view, onSwitch }) {
  const t = themeOf(u);
  const streak = computeStreak(u.streak.days);
  return (
    <div className="sticky top-0 z-40 backdrop-blur bg-stone-50/90 border-b border-stone-200">
      <div className="max-w-2xl mx-auto flex items-center gap-2 px-4 py-2">
        <button onClick={() => go("home")} className="font-serif text-xl font-bold tracking-tight text-stone-900 min-h-11 flex items-center">
          Deux<span className={`${t.text}`}>.</span>
        </button>
        {view !== "home" && (
          <button onClick={() => go("home")} className="text-sm text-stone-500 hover:text-stone-800 min-h-11 px-2">← Home</button>
        )}
        <div className="flex-1" />
        {streak > 0 && <Chip tone="amber">🔥 {streak}</Chip>}
        <button onClick={() => go("dashboard")} className="min-h-11 min-w-11 rounded-full hover:bg-stone-200 text-lg" title="Progress" aria-label="Progress dashboard">📊</button>
        <button onClick={() => go("settings")} className="min-h-11 min-w-11 rounded-full hover:bg-stone-200 text-lg" title="Settings" aria-label="Settings">⚙️</button>
        <button onClick={onSwitch} className={`min-h-11 px-3 rounded-full ${t.chip} font-semibold text-sm max-w-32 truncate`} title="Switch user" aria-label={`Switch user (current: ${u.name})`}>
          {u.emoji} {u.name}
        </button>
      </div>
    </div>
  );
}

function Screen({ children }) {
  return <div className="max-w-2xl mx-auto px-4 py-5 pb-24">{children}</div>;
}

/* Intentional type treatment + view-transition keyframes, shared by every
   top-level render path. No network fonts — platform stacks only. */
function DeuxStyles() {
  return (
    <style>{`
      .font-serif { font-family: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", "New York", Georgia, serif; letter-spacing: -0.01em; }
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
      @keyframes deuxfade { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
      .view-fade { animation: deuxfade 0.18s ease-out both; }
      @media (prefers-reduced-motion: reduce) { .view-fade { animation: none; } }
    `}</style>
  );
}

/* ========================================================================== */
/*  PROFILE PICKER                                                            */
/* ========================================================================== */

function ProfilePicker({ profiles, onPick, onAdd }) {
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  return (
    <div className="min-h-screen bg-stone-100 flex flex-col items-center justify-center px-6 py-10">
      <div className="text-center mb-10">
        <div className="font-serif text-6xl font-bold tracking-tight text-stone-900">Deux<span className="text-rose-500">.</span></div>
        <div className="mt-2 text-stone-500">French, for the two of you.</div>
      </div>
      <div className="grid grid-cols-2 gap-4 w-full max-w-md">
        {profiles.map((p) => {
          const t = THEME[p.theme] || THEME.indigo;
          return (
            <button
              key={p.id}
              onClick={() => onPick(p.id)}
              className={`rounded-3xl bg-gradient-to-br ${t.grad} text-white p-6 min-h-40 flex flex-col items-center justify-center gap-2 shadow-lg active:scale-95 transition-transform`}
            >
              <span className="text-5xl">{p.emoji}</span>
              <span className="text-2xl font-bold">{p.name}</span>
              <span className="text-xs opacity-90">{p.placed ? `${p.level} · ${p.track === "listening" ? "Listening-First" : "Foundations"}` : "tap to start"}</span>
            </button>
          );
        })}
      </div>
      <div className="mt-6 w-full max-w-md">
        {adding ? (
          <div className="flex gap-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Guest name"
              className="flex-1 rounded-2xl border border-stone-300 px-4 py-3 bg-white"
              maxLength={16}
            />
            <Btn onClick={() => { if (name.trim()) { onAdd(name.trim()); setName(""); setAdding(false); } }} small>Add</Btn>
            <Btn onClick={() => setAdding(false)} variant="ghost" small>✕</Btn>
          </div>
        ) : (
          <button onClick={() => setAdding(true)} className="w-full text-center text-stone-500 hover:text-stone-600 text-sm min-h-11">+ add profile</button>
        )}
      </div>
      <div className="mt-10 text-xs text-stone-500 text-center max-w-xs">
        Shared device, separate journeys — each profile keeps its own deck, progress and streak.
      </div>
    </div>
  );
}

/* ========================================================================== */
/*  PLACEMENT CHECK — 7 adaptive questions (listening + grammar)              */
/* ========================================================================== */

function Placement({ u, onDone }) {
  const t = themeOf(u);
  const LEVELS = ["A0", "A1", "A2", "B1"];
  const [levelIdx, setLevelIdx] = useState(u.id === "stuart" ? 2 : 1);
  const [asked, setAsked] = useState([]); // {level, correct}
  const [q, setQ] = useState(null);
  const [picked, setPicked] = useState(null);
  const [phase, setPhase] = useState("intro"); // intro | question | feedback | result
  const usedRef = useRef(new Set());
  const consecRef = useRef(0); // two consecutive corrects required to move up a level
  const TOTAL = 7;

  const drawQuestion = (idx) => {
    const lvl = LEVELS[idx];
    // no French TTS on this browser -> listening questions would be unanswerable
    const bank = PLACEMENT_BANK[lvl].filter((x) => ttsSupported() || x.type !== "listen");
    const avail = bank.filter((x) => !usedRef.current.has(lvl + x.q));
    const pick = (avail.length ? avail : bank)[Math.floor(Math.random() * (avail.length ? avail.length : bank.length))];
    usedRef.current.add(lvl + pick.q);
    setQ({ ...pick, options: shuffle(pick.options), level: lvl });
    setPicked(null);
    setPhase("question");
  };

  const answer = (opt) => {
    if (picked !== null) return;
    setPicked(opt);
    const correct = opt === q.answer;
    setAsked((a) => [...a, { level: q.level, correct }]);
    setPhase("feedback");
  };

  const next = () => {
    const last = asked[asked.length - 1];
    let idx = levelIdx;
    if (last.correct) {
      consecRef.current += 1;
      if (consecRef.current >= 2) { idx = clamp(idx + 1, 0, LEVELS.length - 1); consecRef.current = 0; }
    } else {
      consecRef.current = 0;
      idx = clamp(idx - 1, 0, LEVELS.length - 1);
    }
    setLevelIdx(idx);
    if (asked.length >= TOTAL) { setPhase("result"); return; }
    drawQuestion(idx);
  };

  const computeResult = () => {
    // highest level with a majority of correct answers; fall back down
    const byLevel = {};
    asked.forEach((a) => {
      byLevel[a.level] = byLevel[a.level] || { c: 0, n: 0 };
      byLevel[a.level].n++;
      if (a.correct) byLevel[a.level].c++;
    });
    let result = "A0";
    for (const lvl of LEVELS) {
      const s = byLevel[lvl];
      if (s && s.c >= Math.ceil(s.n / 2)) result = lvl;
    }
    // never place below a level fully aced en route
    return result;
  };

  if (phase === "intro") {
    return (
      <Screen>
        <Card className="p-6 text-center">
          <div className="text-4xl mb-2">{u.emoji}</div>
          <h1 className="font-serif text-2xl font-bold mb-2">Quick placement check</h1>
          <p className="text-stone-600 mb-1">{TOTAL} short questions — some listening, some grammar. It adapts as you go.</p>
          <p className="text-stone-500 text-sm mb-6">No pressure: this just picks your starting point. You can re-run it any time from Settings.</p>
          <Btn theme={u.theme} onClick={() => drawQuestion(levelIdx)}>Start</Btn>
        </Card>
      </Screen>
    );
  }

  if (phase === "result") {
    const level = computeResult();
    const track = CEFR_ORDER.indexOf(level) >= CEFR_ORDER.indexOf("A2") ? "listening" : "foundations";
    return (
      <Screen>
        <Card className="p-6 text-center">
          <div className="text-4xl mb-2">🎯</div>
          <h1 className="font-serif text-2xl font-bold mb-2">Your starting point: {level}</h1>
          <p className="text-stone-600 mb-4">
            {track === "listening"
              ? "You clearly know more than a beginner — no beginner grind for you. We recommend the Listening-First track: your fastest wins are in training your ear for real spoken French."
              : "We'll start from the foundations: the French sound system, survival phrases, and one new pattern at a time. Short sessions, plenty of repetition."}
          </p>
          <div className="text-xs text-stone-500 mb-6">
            Honest yardstick: reaching A1 typically takes {CEFR_HOURS.A1} of study; A2 {CEFR_HOURS.A2}; B1 {CEFR_HOURS.B1}. Deux tracks real hours, not points.
          </div>
          <div className="flex flex-col gap-2 items-center">
            <Btn theme={u.theme} onClick={() => onDone(level, track)}>
              Start the {track === "listening" ? "Listening-First" : "Foundations"} track
            </Btn>
            <Btn variant="ghost" onClick={() => onDone(level, track === "listening" ? "foundations" : "listening")}>
              Use the {track === "listening" ? "Foundations" : "Listening-First"} track instead
            </Btn>
          </div>
        </Card>
      </Screen>
    );
  }

  const num = asked.length + (phase === "feedback" ? 0 : 1);
  return (
    <Screen>
      <div className="flex items-center gap-2 mb-4">
        <div className="flex-1 h-2 bg-stone-200 rounded-full overflow-hidden">
          <div className={`h-full ${t.bar} transition-all duration-500`} style={{ width: `${(asked.length / TOTAL) * 100}%` }} />
        </div>
        <span className="text-xs text-stone-500">{Math.min(num, TOTAL)}/{TOTAL}</span>
      </div>
      <Card className="p-6">
        <Chip tone="stone">{q.type === "listen" ? "🎧 Listening" : "✍️ Grammar"}</Chip>
        <h2 className="font-serif text-xl font-bold mt-3 mb-4">{q.q}</h2>
        {q.type === "listen" && (
          <div className="mb-4">
            <SpeakerBtn text={q.audio} rate={0.9} theme={u.theme} label="Play audio" big />
            <div className="text-xs text-stone-500 mt-2">You can replay it as often as you like.</div>
          </div>
        )}
        <div className="flex flex-col gap-2">
          {q.options.map((opt) => {
            let cls = "bg-stone-50 border-stone-200 hover:bg-stone-100";
            if (picked !== null) {
              if (opt === q.answer) cls = "bg-emerald-50 border-emerald-400 text-emerald-900";
              else if (opt === picked) cls = "bg-rose-50 border-rose-300 text-rose-900";
              else cls = "bg-stone-50 border-stone-200 opacity-60";
            }
            return (
              <button key={opt} onClick={() => answer(opt)} className={`text-left rounded-2xl border px-4 py-3 min-h-12 transition-colors ${cls}`}>
                {opt}
              </button>
            );
          })}
        </div>
        {phase === "feedback" && (
          <div className="mt-4 flex items-center justify-between">
            <span className={`text-sm font-semibold ${picked === q.answer ? "text-emerald-700" : "text-rose-700"}`}>
              {picked === q.answer ? "Correct!" : "Not quite — noted."}
            </span>
            <Btn theme={u.theme} onClick={next} small>{asked.length >= TOTAL ? "See result" : "Next"}</Btn>
          </div>
        )}
      </Card>
    </Screen>
  );
}

/* ========================================================================== */
/*  SRS REVIEW — production-first with SM-2 grading                           */
/* ========================================================================== */

function SRSReview({ u, setU, go, quickWin }) {
  const t = themeOf(u);
  const [queue, setQueue] = useState(() => dueCards(u.deck).slice(0, quickWin ? 3 : 20));
  const [idx, setIdx] = useState(0);
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState(null); // null | {verdict}
  const [doneCount, setDoneCount] = useState(0);
  const inputRef = useRef(null);

  const card = queue[idx];

  const check = () => {
    if (!card) return;
    const verdict = checkAnswer(card.fr, answer);
    setResult({ verdict });
    speak(card.fr, u.settings.ttsRate);
  };

  const grade = (q) => {
    const graded = gradeCard(card, q);
    setU((prev) => {
      const deck = prev.deck.map((c) => (c.id === card.id ? graded : c));
      const stats = { ...prev.stats, reviews: prev.stats.reviews + 1 };
      const writing = result && result.verdict !== "empty"
        ? [...prev.stats.writing, result.verdict === "exact" ? 100 : result.verdict === "close" ? 75 : 20].slice(-40)
        : prev.stats.writing;
      return { ...prev, deck, stats: { ...stats, writing }, minutes: prev.minutes + 0.5 };
    });
    const nextCount = doneCount + 1;
    setDoneCount(nextCount);
    setAnswer(""); setResult(null);
    if (idx + 1 < queue.length) setIdx(idx + 1);
    else finish(nextCount);
  };

  const finish = (count) => {
    setU((prev) => {
      const qualifies = count >= Math.min(5, queue.length) && count > 0;
      if (!qualifies) return prev;
      return { ...prev, streak: { ...prev.streak, days: { ...prev.streak.days, [todayKey()]: true } } };
    });
    setQueue([]); setIdx(0);
  };

  if (!card) {
    const due = dueCards(u.deck).length;
    return (
      <Screen>
        <Card className="p-6 text-center">
          <div className="text-4xl mb-2">{doneCount > 0 ? "🎉" : "🌿"}</div>
          <h1 className="font-serif text-2xl font-bold mb-2">
            {doneCount > 0 ? `${doneCount} card${doneCount > 1 ? "s" : ""} reviewed` : "Nothing due right now"}
          </h1>
          <p className="text-stone-600 mb-4">
            {doneCount > 0
              ? "Each card is now scheduled exactly when you're about to forget it. That's the whole trick."
              : due > 0
              ? `${due} card(s) just became due — start a session.`
              : "Your memory is ahead of the scheduler. New words from lessons and conversations land here automatically."}
          </p>
          <div className="flex gap-2 justify-center">
            {due > 0 && <Btn theme={u.theme} onClick={() => { setQueue(dueCards(u.deck).slice(0, 20)); setDoneCount(0); }}>Review {due} due</Btn>}
            <Btn variant="secondary" onClick={() => go("home")}>Home</Btn>
          </div>
          <div className="mt-4 text-xs text-stone-500">{knownCount(u.deck)} learning · {matureCount(u.deck)} mature · {u.deck.length} total cards</div>
        </Card>
      </Screen>
    );
  }

  return (
    <Screen>
      <div className="flex items-center gap-2 mb-4">
        <div className="flex-1 h-2 bg-stone-200 rounded-full overflow-hidden">
          <div className={`h-full ${t.bar} transition-all`} style={{ width: `${(idx / queue.length) * 100}%` }} />
        </div>
        <span className="text-xs text-stone-500">{idx + 1}/{queue.length}</span>
      </div>
      <Card className="p-6">
        <div className="text-center mb-5">
          {card.glyph && <div className="text-5xl mb-2">{card.glyph}</div>}
          <div className="text-xs uppercase tracking-wide text-stone-500 mb-1">Say it in French</div>
          <div className="font-serif text-2xl font-bold text-stone-900">{card.en}</div>
          {card.exampleEn && !result && <div className="text-sm text-stone-500 mt-2 italic">“{card.exampleEn}”</div>}
        </div>

        {!result ? (
          <div>
            <div className="flex gap-2">
              <input
                ref={inputRef}
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && check()}
                placeholder="Type the French… (or speak)"
                autoCapitalize="none" autoCorrect="off" spellCheck={false}
                className="flex-1 rounded-2xl border border-stone-300 px-4 py-3 bg-white text-lg"
              />
            </div>
            <div className="flex gap-2 mt-3 flex-wrap">
              <Btn theme={u.theme} onClick={check} disabled={!answer.trim()}>Check</Btn>
              <MicBtn theme={u.theme} onText={(txt) => setAnswer(txt)} />
              <Btn variant="ghost" onClick={() => { setResult({ verdict: "reveal" }); speak(card.fr, u.settings.ttsRate); }}>Show answer</Btn>
            </div>
          </div>
        ) : (
          <div>
            <div className={`rounded-2xl p-4 mb-3 ${result.verdict === "exact" ? "bg-emerald-50 border border-emerald-200" : result.verdict === "close" ? "bg-amber-50 border border-amber-200" : "bg-rose-50 border border-rose-200"}`}>
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="font-bold text-lg text-stone-900">{card.fr}</div>
                  {card.example && <div className="text-sm text-stone-500 mt-1 italic">{card.example}</div>}
                </div>
                <SpeakerBtn text={card.example ? `${card.fr}. ${card.example}` : card.fr} rate={u.settings.ttsRate} theme={u.theme} label="" />
              </div>
              <div className="text-sm mt-2 font-medium">
                {result.verdict === "exact" && <span className="text-emerald-700">Perfect recall.</span>}
                {result.verdict === "close" && <span className="text-amber-700">Nearly — check the spelling/accents: you wrote “{answer}”.</span>}
                {result.verdict === "wrong" && <span className="text-rose-700">You wrote “{answer}”. Compare carefully, then grade honestly.</span>}
                {result.verdict === "reveal" && <span className="text-stone-600">Grade yourself honestly — it drives the schedule.</span>}
                {result.verdict === "empty" && <span className="text-stone-600">Try to produce it before revealing next time — the struggle is what strengthens memory.</span>}
              </div>
            </div>
            <div className="grid grid-cols-4 gap-2">
              <Btn variant="danger" small onClick={() => grade(1)}>Again</Btn>
              <Btn variant="secondary" small onClick={() => grade(3)}>Hard</Btn>
              <Btn variant="success" small onClick={() => grade(4)}>Good</Btn>
              <Btn variant="secondary" small className="border-emerald-300 text-emerald-700" onClick={() => grade(5)}>Easy</Btn>
            </div>
            <div className="text-center text-xs text-stone-500 mt-2">
              {result.verdict === "exact" ? "Suggested: Good or Easy" : result.verdict === "close" ? "Suggested: Hard or Good" : "Suggested: Again or Hard"}
            </div>
          </div>
        )}
      </Card>
    </Screen>
  );
}

/* ========================================================================== */
/*  HONEST PROGRESS DASHBOARD                                                 */
/* ========================================================================== */

function Dashboard({ u, partner, go }) {
  const t = themeOf(u);
  const [showPartner, setShowPartner] = useState(false);
  const skills = skillScores(u);
  const streak = computeStreak(u.streak.days);
  const dict = u.stats.dictation.slice(-12);
  const hours = (u.minutes / 60);

  const skillRows = [
    { key: "listening", label: "Listening", icon: "🎧" },
    { key: "speaking", label: "Speaking", icon: "🗣️" },
    { key: "reading", label: "Reading", icon: "📖" },
    { key: "writing", label: "Writing", icon: "✍️" },
    { key: "vocabulary", label: "Vocabulary", icon: "🧠" },
  ];
  const bandOf = (pct) => (pct < 12 ? "A0" : pct < 32 ? "A1" : pct < 52 ? "A2" : pct < 72 ? "B1" : "B2");

  const p = showPartner && partner ? partner : null;
  const pSkills = p ? skillScores(p) : null;

  return (
    <Screen>
      <div className="flex items-center justify-between mb-4">
        <h1 className="font-serif text-2xl font-bold">Progress — honestly</h1>
        {partner && partner.settings.shareProgress && (
          <Btn variant="secondary" small onClick={() => setShowPartner(!showPartner)}>
            {showPartner ? `Just me` : `+ ${partner.name}`}
          </Btn>
        )}
      </div>

      <Card className="p-5 mb-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-xs uppercase tracking-wide text-stone-500">Working level</div>
            <div className="font-serif text-3xl font-bold">{u.level}</div>
          </div>
          <div className="text-right">
            <div className="text-xs uppercase tracking-wide text-stone-500">Study time</div>
            <div className="font-serif text-3xl font-bold">{hours < 1 ? `${Math.round(u.minutes)}m` : `${hours.toFixed(1)}h`}</div>
          </div>
          <div className="text-right">
            <div className="text-xs uppercase tracking-wide text-stone-500">Streak</div>
            <div className="font-serif text-3xl font-bold">🔥 {streak}</div>
          </div>
        </div>
        {skillRows.map((row) => (
          <div key={row.key}>
            <Bar
              label={`${row.icon} ${row.label}${p ? ` — you` : ""}`}
              pct={skills[row.key]}
              color={t.bar}
              right={`~${bandOf(skills[row.key])}`}
            />
            {p && (
              <Bar label={`${row.icon} ${row.label} — ${p.name}`} pct={pSkills[row.key]} color="bg-stone-400" right={`~${bandOf(pSkills[row.key])}`} />
            )}
          </div>
        ))}
        <div className="text-xs text-stone-500 mt-2">
          Estimates blend your placement level with measured accuracy. CEFR reality check: A1 {CEFR_HOURS.A1} · A2 {CEFR_HOURS.A2} · B1 {CEFR_HOURS.B1}. No shortcuts exist; consistency does.
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <Card className="p-4 text-center">
          <div className="text-3xl font-bold font-serif">{matureCount(u.deck)}</div>
          <div className="text-xs text-stone-500">words mature (21d+ interval)</div>
          <div className="text-xs text-stone-500 mt-1">{knownCount(u.deck)} learning · {u.deck.length} in deck</div>
        </Card>
        <Card className="p-4">
          <div className="text-xs text-stone-500 mb-2 text-center">Dictation accuracy (last {dict.length || 0})</div>
          {dict.length ? (
            <div className="flex items-end gap-1 h-16 justify-center">
              {dict.map((d, i) => (
                <div key={i} className={`w-3 rounded-t ${d.acc >= 80 ? "bg-emerald-400" : d.acc >= 60 ? "bg-amber-400" : "bg-rose-400"}`} style={{ height: `${clamp(d.acc, 8, 100)}%` }} title={`${d.acc}%`} />
              ))}
            </div>
          ) : (
            <div className="text-center text-stone-500 text-sm py-4">No dictations yet</div>
          )}
        </Card>
      </div>

      {p && (
        <Card className={`p-4 mb-4 ${THEME[p.theme].soft}`}>
          <div className="font-semibold mb-1">{p.emoji} {p.name} — {p.level}, 🔥 {computeStreak(p.streak.days)} day streak, {matureCount(p.deck)} mature words</div>
          <div className="text-sm text-stone-600">You're on different journeys — cheer, don't compare. A "well done" over dinner beats any leaderboard.</div>
        </Card>
      )}

      <Card className="p-5">
        <div className="text-xs uppercase tracking-wide text-stone-500 mb-1">Recommended next</div>
        {(() => { const na = nextAction(u); return (
          <div className="flex items-center justify-between gap-3">
            <div>
              <div className="font-semibold">{na.label}</div>
              <div className="text-sm text-stone-500">{na.why}</div>
            </div>
            <Btn theme={u.theme} small onClick={() => go(na.view, na.params)}>Go</Btn>
          </div>
        ); })()}
      </Card>
    </Screen>
  );
}

/* ========================================================================== */
/*  SETTINGS                                                                  */
/* ========================================================================== */

/* Device-level AI engine config: an Anthropic API key used directly from
   this browser, or a personal proxy URL (see cloudflare-worker.js). Without
   either, every AI feature falls back to the built-in offline bank. */
function AiEngine({ aiCfg, onSave }) {
  const [key, setKey] = useState(aiCfg.key || "");
  const [url, setUrl] = useState(aiCfg.url || "");
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState(null);
  const dirty = key.trim() !== (aiCfg.key || "") || url.trim() !== (aiCfg.url || "");

  const test = async () => {
    setTesting(true); setResult(null);
    await onSave({ key, url });
    const res = await aiJSON("You are a connection test.", 'Return exactly {"ok": true}', null, 50);
    setResult(res && res.ok === true ? "ok" : "fail");
    setTesting(false);
  };

  return (
    <Card className="p-5 mb-4">
      <div className="flex items-center justify-between mb-1">
        <div className="font-semibold text-stone-800">AI content engine</div>
        <Chip tone={aiConfigured() ? "green" : "stone"}>{aiConfigured() ? "configured" : "built-in content only"}</Chip>
      </div>
      <p className="text-xs text-stone-500 mb-3">
        Unlocks unlimited fresh clips, passages, drills and free conversation. Without it, Deux runs on its built-in offline bank (still fully functional).
      </p>
      <label className="block text-sm font-medium text-stone-700 mb-1">Anthropic API key</label>
      <input
        type="password" value={key} onChange={(e) => { setKey(e.target.value); setResult(null); }}
        placeholder="sk-ant-…" autoCapitalize="none" autoCorrect="off" spellCheck={false}
        className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white mb-1"
      />
      <div className="text-xs text-stone-500 mb-3">
        Stored only on this device and sent only to Anthropic. Get one at console.anthropic.com — usage is pay-as-you-go (typical heavy daily use costs pennies). Never share this device's browser profile with the key saved.
      </div>
      <label className="block text-sm font-medium text-stone-700 mb-1">Or: personal proxy URL (advanced)</label>
      <input
        type="url" value={url} onChange={(e) => { setUrl(e.target.value); setResult(null); }}
        placeholder="https://deux-ai.yourname.workers.dev" autoCapitalize="none" autoCorrect="off" spellCheck={false}
        className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white mb-1"
      />
      <div className="text-xs text-stone-500 mb-3">
        Keeps the key off the device entirely — deploy the ready-made Cloudflare Worker from the repo (cloudflare-worker.js) and paste its URL here. If set, it takes priority over the key field.
      </div>
      <div className="flex gap-2 items-center flex-wrap">
        <Btn small onClick={() => { onSave({ key, url }); setResult(null); }} disabled={!dirty}>Save</Btn>
        <Btn small variant="secondary" onClick={test} disabled={testing || (!key.trim() && !url.trim())}>{testing ? "Testing…" : "Save & test"}</Btn>
        {(key || url) && <Btn small variant="ghost" onClick={() => { setKey(""); setUrl(""); onSave({ key: "", url: "" }); setResult(null); }}>Clear</Btn>}
        {result === "ok" && <Chip tone="green">✓ connected — AI content is live</Chip>}
        {result === "fail" && <Chip tone="red">✗ couldn't reach the AI — check the key/URL</Chip>}
      </div>
    </Card>
  );
}

function Settings({ u, setU, go, onRerunPlacement, onSwitch, aiCfg, onSaveAi }) {
  const set = (patch) => setU((prev) => ({ ...prev, settings: { ...prev.settings, ...patch } }));
  const Row = ({ label, children, hint }) => (
    <div className="py-3 border-b border-stone-100 last:border-0">
      <div className="flex items-center justify-between gap-3">
        <div className="font-medium text-stone-800">{label}</div>
        <div>{children}</div>
      </div>
      {hint && <div className="text-xs text-stone-500 mt-1">{hint}</div>}
    </div>
  );
  const Seg = ({ value, options, onPick }) => (
    <div className="flex rounded-xl bg-stone-100 p-1">
      {options.map((o) => (
        <button key={o} onClick={() => onPick(o)} className={`px-3 py-1.5 rounded-lg text-sm font-medium min-h-9 capitalize ${value === o ? "bg-white shadow text-stone-900" : "text-stone-500"}`}>
          {o}
        </button>
      ))}
    </div>
  );
  return (
    <Screen>
      <h1 className="font-serif text-2xl font-bold mb-4">Settings</h1>
      <Card className="px-5 py-2 mb-4">
        <Row label="Conversation strictness" hint="Strict = every error corrected, including word order. Research says forgiving AI partners create false confidence.">
          <Seg value={u.settings.strictness} options={["lenient", "normal", "strict"]} onPick={(v) => set({ strictness: v })} />
        </Row>
        <Row label="Dual subtitles by default" hint="Show English under French in listening/reading. Wean off as your ear improves.">
          <Seg value={u.settings.dualSubs ? "on" : "off"} options={["on", "off"]} onPick={(v) => set({ dualSubs: v === "on" })} />
        </Row>
        <Row label="Audio speed" hint="Applies everywhere; individual players can still adjust.">
          <div className="flex items-center gap-2">
            <input type="range" min="0.6" max="1.2" step="0.05" value={u.settings.ttsRate} onChange={(e) => set({ ttsRate: parseFloat(e.target.value) })} className="w-28" />
            <span className="text-sm text-stone-500 w-10">{u.settings.ttsRate.toFixed(2)}×</span>
          </div>
        </Row>
        <Row label="Gentle daily reminder" hint="A soft nudge on the home screen — never guilt, never spam.">
          <Seg value={u.settings.reminders ? "on" : "off"} options={["on", "off"]} onPick={(v) => set({ reminders: v === "on" })} />
        </Row>
        {u.settings.reminders && (
          <Row label="Preferred time">
            <input type="time" value={u.settings.reminderTime} onChange={(e) => set({ reminderTime: e.target.value })} className="rounded-xl border border-stone-300 px-3 py-2" />
          </Row>
        )}
        <Row label="Share progress with household" hint="Lets the other profile see your level/streak in their dashboard. Accountability, not competition.">
          <Seg value={u.settings.shareProgress ? "on" : "off"} options={["on", "off"]} onPick={(v) => set({ shareProgress: v === "on" })} />
        </Row>
      </Card>
      <AiEngine aiCfg={aiCfg || { key: "", url: "" }} onSave={onSaveAi} />
      <Card className="px-5 py-2 mb-4">
        <Row label="Re-run placement check" hint="Re-assesses your level and recommended track.">
          <Btn variant="secondary" small onClick={onRerunPlacement}>Re-run</Btn>
        </Row>
        <Row label="Switch track" hint={`Currently: ${u.track === "listening" ? "Listening-First" : "Foundations"}.`}>
          <Btn variant="secondary" small onClick={() => setU((prev) => ({ ...prev, track: prev.track === "listening" ? "foundations" : "listening" }))}>
            Switch
          </Btn>
        </Row>
        <Row label="Switch user">
          <Btn variant="secondary" small onClick={onSwitch}>Profiles</Btn>
        </Row>
      </Card>
      <div className="text-xs text-stone-500 px-2">
        Deux never promises "fluency in weeks". Realistic CEFR study time: A1 {CEFR_HOURS.A1} · A2 {CEFR_HOURS.A2} · B1 {CEFR_HOURS.B1}. The streak protects the habit; the SRS and your ear do the learning.
      </div>
      <div className="mt-4"><Btn variant="ghost" onClick={() => go("home")}>← Back</Btn></div>
    </Screen>
  );
}

/* ========================================================================== */
/*  HOME                                                                      */
/* ========================================================================== */

function Home({ u, setU, go }) {
  const t = themeOf(u);
  const due = dueCards(u.deck).length;
  const streak = computeStreak(u.streak.days);
  const doneToday = !!u.streak.days[todayKey()];
  const na = nextAction(u);
  const canRepair = repairAvailable(u);
  const isCarol = u.track === "foundations";

  const repair = () => {
    setU((prev) => ({
      ...prev,
      streak: { days: { ...prev.streak.days, [dateKeyOffset(-1)]: true }, lastRepairWeek: weekKey() },
    }));
  };

  const featuresBase = isCarol
    ? [
        { view: "units", icon: "🧩", title: "Lessons", sub: "One pattern at a time" },
        { view: "phonics", icon: "👂", title: "Sounds of French", sub: "Phonics & pronunciation" },
        { view: "review", icon: "🗂️", title: "Review", sub: due ? `${due} due` : "All caught up" },
        { view: "conversation", icon: "💬", title: "Talk", sub: "Guided, with suggestions" },
        { view: "library", icon: "📖", title: "Read & listen", sub: "Short stories, dual subtitles" },
        { view: "exercise", icon: "🎯", title: "Make me an exercise", sub: "Any focus, on demand" },
      ]
    : [
        { view: "trainer", icon: "🎧", title: "Listening trainer", sub: `Rung ${u.trainer.rung}/8 — connected speech` },
        { view: "review", icon: "🗂️", title: "Review", sub: due ? `${due} due` : "All caught up" },
        { view: "cslessons", icon: "🔗", title: "Why French links", sub: "Liaison · elision · enchaînement" },
        { view: "drills", icon: "⚔️", title: "Grammar drills", sub: "Blocked → interleaved" },
        { view: "conversation", icon: "💬", title: "Talk (strict)", sub: "Errors get caught" },
        { view: "library", icon: "📖", title: "Read & listen", sub: "Then hide the text" },
        { view: "exercise", icon: "🎯", title: "Make me an exercise", sub: "Any focus, on demand" },
      ];

  /* when the "Up next" hero already shows reviews, don't repeat the tile */
  const features =
    na.view === "review"
      ? featuresBase.filter((f) => f.view !== "review").concat([{ view: "dashboard", icon: "📊", title: "Progress", sub: "Honest numbers" }])
      : featuresBase;

  const now = new Date();
  const reminderDue = u.settings.reminders && !doneToday && `${pad2(now.getHours())}:${pad2(now.getMinutes())}` >= u.settings.reminderTime;

  return (
    <Screen>
      <div className={`rounded-3xl bg-gradient-to-br ${t.grad} text-white p-6 mb-4 shadow-lg`}>
        <div className="flex items-start justify-between">
          <div>
            <div className="text-sm opacity-90">{isCarol ? "Bonjour" : "Salut"}, {u.name} 👋</div>
            <div className="font-serif text-2xl font-bold mt-1">
              {doneToday ? "Today is banked. Anything extra is a bonus." : isCarol ? "Ten good minutes. That's the whole job today." : "Train the ear. The rest follows."}
            </div>
          </div>
          <div className="text-4xl">{u.emoji}</div>
        </div>
        <div className="flex gap-2 mt-4 flex-wrap">
          {streak > 0 ? (
            <span className="bg-white/20 rounded-full px-3 py-1 text-xs font-semibold">🔥 {streak}-day streak{doneToday ? " ✓" : ""}</span>
          ) : (
            <span className="bg-white/20 rounded-full px-3 py-1 text-xs font-semibold">🌱 Start your streak today</span>
          )}
          <span className="bg-white/20 rounded-full px-3 py-1 text-xs font-semibold">{u.level} · {isCarol ? "Foundations" : "Listening-First"}</span>
          <span className="bg-white/20 rounded-full px-3 py-1 text-xs font-semibold">🧠 {matureCount(u.deck)} words solid</span>
        </div>
      </div>

      {reminderDue && (
        <Card className="p-4 mb-4 border-amber-200 bg-amber-50">
          <div className="text-sm text-amber-900">🌤️ Gentle nudge: your {u.settings.reminderTime} practice window is open. One quick win keeps the streak honest — no guilt either way.</div>
        </Card>
      )}

      {canRepair && (
        <Card className="p-4 mb-4 border-rose-200 bg-rose-50">
          <div className="flex items-center justify-between gap-3">
            <div className="text-sm text-rose-900">Yesterday slipped by. Life happens — use this week's free repair?</div>
            <Btn variant="secondary" small onClick={repair}>🩹 Repair</Btn>
          </div>
        </Card>
      )}

      <Card className={`p-5 mb-4 ${t.accentL}`}>
        <div className="text-xs uppercase tracking-wide text-stone-500 mb-1">Up next</div>
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="font-semibold text-lg">{na.label}</div>
            <div className="text-sm text-stone-500">{na.why}</div>
          </div>
          <Btn theme={u.theme} onClick={() => go(na.view, na.params)}>Start</Btn>
        </div>
        {!doneToday && due >= 3 && (
          <button onClick={() => go("review", { quickWin: true })} className="mt-3 text-sm text-stone-500 underline underline-offset-2 min-h-11">
            ⚡ Only got 2 minutes? Do a 3-card quick win.
          </button>
        )}
      </Card>

      <div className="grid grid-cols-2 gap-3">
        {features.map((f) => (
          <button key={f.view + f.title} onClick={() => go(f.view)} className={`bg-white rounded-3xl border shadow-sm p-4 text-left hover:shadow-md active:scale-95 transition-all min-h-24 ${t.edge}`}>
            <div className={`w-10 h-10 rounded-2xl ${t.chip} flex items-center justify-center text-xl mb-2`}>{f.icon}</div>
            <div className="font-semibold text-stone-900">{f.title}</div>
            <div className="text-xs text-stone-500">{f.sub}</div>
          </button>
        ))}
      </div>

      {!isCarol && (
        <Card className="p-4 mt-4">
          <div className="text-xs uppercase tracking-wide text-stone-500 mb-2">When you're ready for real audio</div>
          {PODCAST_LADDER.map((pod) => (
            <a key={pod.name} href={pod.url} target="_blank" rel="noreferrer" className="flex items-center justify-between py-2 border-b border-stone-100 last:border-0 min-h-11">
              <div>
                <div className="font-medium text-stone-800 text-sm">{pod.name} <Chip tone="sky">{pod.level}</Chip></div>
                <div className="text-xs text-stone-500">{pod.why}</div>
              </div>
              <span className="text-stone-300">↗</span>
            </a>
          ))}
        </Card>
      )}
    </Screen>
  );
}

/* ========================================================================== */
/*  FOUNDATIONS UNITS (Carol) — explain → blocked practice → task → SRS       */
/* ========================================================================== */

function UnitList({ u, go }) {
  const t = themeOf(u);
  return (
    <Screen>
      <h1 className="font-serif text-2xl font-bold mb-1">Lessons</h1>
      <p className="text-sm text-stone-500 mb-4">Task-based scenarios, one new pattern each. Explain → practise → do it for real → words go to your review deck.</p>
      {FOUNDATIONS_UNITS.map((unit, i) => {
        const done = !!u.lessons[unit.id];
        const prevDone = i === 0 || !!u.lessons[FOUNDATIONS_UNITS[i - 1].id];
        const locked = !done && !prevDone;
        return (
          <button
            key={unit.id}
            onClick={() => !locked && go("unit", { unitId: unit.id })}
            className={`w-full text-left mb-3 rounded-3xl border p-4 flex items-center gap-4 transition-all min-h-20 ${done ? "bg-emerald-50 border-emerald-200" : locked ? "bg-stone-50 border-stone-200 opacity-50" : "bg-white border-stone-200 shadow-sm hover:shadow-md"}`}
          >
            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center text-lg font-bold ${done ? "bg-emerald-500 text-white" : locked ? "bg-stone-100 text-stone-500" : t.numChip}`}>
              {done ? "✓" : locked ? "🔒" : i + 1}
            </div>
            <div className="flex-1">
              <div className="font-semibold text-stone-900">{unit.title}</div>
              <div className="text-xs text-stone-500">
                {locked ? `Finish “${FOUNDATIONS_UNITS[i - 1].title}” first` : `${unit.scenario} · ${unit.grammar.point}`}
              </div>
            </div>
            <Chip tone={done ? "green" : "stone"}>{unit.level}</Chip>
          </button>
        );
      })}
    </Screen>
  );
}

function UnitView({ u, setU, go, unitId }) {
  const unit = FOUNDATIONS_UNITS.find((x) => x.id === unitId) || FOUNDATIONS_UNITS[0];
  const t = themeOf(u);
  const [content, setContent] = useState(null);
  const [step, setStep] = useState(0); // 0 intro, 1 patterns, 2 practice, 3 task, 4 words
  const [pIdx, setPIdx] = useState(0);
  const [pAnswer, setPAnswer] = useState("");
  const [pResult, setPResult] = useState(null);
  const [pScore, setPScore] = useState(0);
  const [taskText, setTaskText] = useState("");
  const [taskFb, setTaskFb] = useState(null);
  const [taskBusy, setTaskBusy] = useState(false);
  const phonics = unit.phonicsId ? PHONICS_LESSONS.find((p) => p.id === unit.phonicsId) : null;

  useEffect(() => {
    let alive = true;
    const fallback = {
      intro: unit.grammar.brief,
      patterns: unit.grammar.examples,
      practice: unit.vocab.slice(0, 4).map((v) => ({ prompt: `How do you say “${v.en}” in French?`, answer: v.fr, hint: v.example })),
      task_question: unit.task.prompt,
    };
    aiJSON(SYS.unit(unit, u.level === "A0" ? "A0" : "A1"), `Generate the micro-lesson now. Vocabulary to weave in where natural: ${unit.vocab.map((v) => v.fr).join(", ")}.`, null)
      .then((res) => {
        if (!alive) return;
        const ok = res && res.intro && Array.isArray(res.patterns) && Array.isArray(res.practice) && res.practice.length >= 3 && res.practice.every((x) => x && x.prompt && x.answer);
        setContent(ok ? { ...res, task_question: res.task_question || unit.task.prompt } : fallback);
      })
      .catch(() => alive && setContent(fallback));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [unitId]);

  if (!content) return <Screen><Spinner label="Writing today's lesson for you…" /></Screen>;

  const checkPractice = () => {
    const item = content.practice[pIdx];
    const verdict = checkAnswer(item.answer, pAnswer);
    setPResult(verdict);
    if (verdict === "exact" || verdict === "close") setPScore((s) => s + 1);
    speak(item.answer, u.settings.ttsRate);
    setU((prev) => ({ ...prev, stats: { ...prev.stats, writing: [...prev.stats.writing, verdict === "exact" ? 100 : verdict === "close" ? 75 : 20].slice(-40) } }));
  };

  const nextPractice = () => {
    setPAnswer(""); setPResult(null);
    if (pIdx + 1 < content.practice.length) setPIdx(pIdx + 1);
    else setStep(3);
  };

  const submitTask = async () => {
    if (!taskText.trim()) return;
    setTaskBusy(true);
    const fb = await aiJSON(
      SYS.taskFeedback(u.level === "A0" ? "A0" : "A1"),
      `Task: ${content.task_question}\nLearner's answer: ${taskText}`,
      { score: 3, praise: "You had a real go at producing French — that's exactly the muscle we're training.", corrections: [], model_answer: unit.task.model }
    );
    setTaskFb(fb);
    setTaskBusy(false);
    setU((prev) => ({ ...prev, stats: { ...prev.stats, speaking: [...prev.stats.speaking, (fb.score || 3) * 20].slice(-40) } }));
  };

  const finishUnit = () => {
    setU((prev) => {
      const { deck } = addCardsToDeck(prev.deck, unit.vocab, `unit:${unit.id}`);
      return {
        ...prev,
        deck,
        lessons: { ...prev.lessons, [unit.id]: { done: true, at: Date.now() } },
        minutes: prev.minutes + 8,
        streak: { ...prev.streak, days: { ...prev.streak.days, [todayKey()]: true } },
      };
    });
    go("units");
  };

  const StepDots = () => (
    <div className="flex gap-1.5 justify-center mb-4">
      {[0, 1, 2, 3, 4].map((s) => (
        <div key={s} className={`h-1.5 rounded-full transition-all ${s === step ? `w-6 ${t.bar}` : s < step ? "w-3 bg-stone-400" : "w-3 bg-stone-200"}`} />
      ))}
    </div>
  );

  return (
    <Screen>
      <div className="flex items-center justify-between mb-2">
        <h1 className="font-serif text-xl font-bold">{unit.title}</h1>
        <Chip tone="amber">{unit.grammar.point}</Chip>
      </div>
      <StepDots />

      {step === 0 && (
        <Card className="p-6">
          <div className="text-xs uppercase tracking-wide text-stone-500 mb-2">The idea (from English, one step at a time)</div>
          <p className="text-stone-800 leading-relaxed mb-4">{content.intro}</p>
          {phonics && (
            <div className={`rounded-2xl ${t.soft} p-4 mb-4`}>
              <div className="font-semibold text-sm mb-1">{phonics.emoji} Sound focus: {phonics.title}</div>
              <div className="text-sm text-stone-600 mb-2">{phonics.tip}</div>
              <div className="flex gap-2 flex-wrap">
                {phonics.items.slice(0, 3).map((it) => (
                  <button key={it.text} onClick={() => speak(it.text, 0.8)} className="bg-white rounded-xl border border-stone-200 px-3 py-2 text-sm font-medium min-h-11">
                    🔊 {it.text}
                  </button>
                ))}
                <button onClick={() => go("phonics", { lessonId: phonics.id })} className="text-sm text-stone-500 underline min-h-11">full sound lesson →</button>
              </div>
            </div>
          )}
          <Btn theme={u.theme} onClick={() => setStep(1)}>Hear the pattern →</Btn>
        </Card>
      )}

      {step === 1 && (
        <Card className="p-6">
          <div className="text-xs uppercase tracking-wide text-stone-500 mb-3">Listen and repeat — tap each line</div>
          {content.patterns.map((ex, i) => (
            <button key={i} onClick={() => speak(ex.fr, u.settings.ttsRate)} className="w-full text-left rounded-2xl border border-stone-200 p-3 mb-2 hover:bg-stone-50 min-h-14">
              <div className="font-semibold text-stone-900">🔊 {ex.fr}</div>
              {u.settings.dualSubs && <div className="text-sm text-stone-500">{ex.en}</div>}
            </button>
          ))}
          <div className="text-xs text-stone-500 mb-3">Say each one out loud after the audio — mouth muscles count.</div>
          <Btn theme={u.theme} onClick={() => setStep(2)}>Practise →</Btn>
        </Card>
      )}

      {step === 2 && (() => { const item = content.practice[pIdx]; return (
        <Card className="p-6">
          <div className="flex justify-between items-center mb-3">
            <Chip tone="indigo">Blocked practice · {pIdx + 1}/{content.practice.length}</Chip>
            <span className="text-xs text-stone-500">one structure only</span>
          </div>
          <div className="font-serif text-lg font-bold mb-3">{item.prompt}</div>
          {!pResult ? (
            <div>
              <input
                value={pAnswer} onChange={(e) => setPAnswer(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && pAnswer.trim() && checkPractice()}
                placeholder="Type the French…" autoCapitalize="none" autoCorrect="off" spellCheck={false}
                className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white text-lg mb-3"
              />
              <div className="flex gap-2 flex-wrap">
                <Btn theme={u.theme} onClick={checkPractice} disabled={!pAnswer.trim()}>Check</Btn>
                <MicBtn theme={u.theme} onText={setPAnswer} />
                {item.hint && <Btn variant="ghost" small onClick={() => speak(item.hint, u.settings.ttsRate)}>💡 hint (audio)</Btn>}
              </div>
            </div>
          ) : (
            <div>
              <div className={`rounded-2xl p-4 mb-3 ${pResult === "exact" ? "bg-emerald-50 border border-emerald-200" : pResult === "close" ? "bg-amber-50 border border-amber-200" : "bg-rose-50 border border-rose-200"}`}>
                <div className="font-bold">{item.answer} <button onClick={() => speak(item.answer, u.settings.ttsRate)}>🔊</button></div>
                <div className="text-sm mt-1">
                  {pResult === "exact" && <span className="text-emerald-700">Exactly right !</span>}
                  {pResult === "close" && <span className="text-amber-700">So close — compare with what you wrote: “{pAnswer}”.</span>}
                  {pResult === "wrong" && <span className="text-rose-700">You wrote “{pAnswer}”. Say the correct one out loud once before moving on.</span>}
                </div>
              </div>
              <Btn theme={u.theme} onClick={nextPractice}>{pIdx + 1 < content.practice.length ? "Next" : "On to the real task →"}</Btn>
            </div>
          )}
        </Card>
      ); })()}

      {step === 3 && (
        <Card className="p-6">
          <Chip tone="amber">The task — say it for real</Chip>
          <div className="font-serif text-lg font-bold mt-3 mb-3">{content.task_question}</div>
          {!taskFb ? (
            <div>
              <div className="mb-3">
                <div className="text-xs text-stone-500 mb-1.5">Words you might need — tap to hear (then build your own sentence):</div>
                <div className="flex flex-wrap gap-2">
                  {unit.vocab.map((v) => (
                    <button key={v.fr} onClick={() => speak(v.fr, u.settings.ttsRate)} className={`rounded-xl ${t.soft} border ${t.edge} px-3 py-2 text-sm min-h-11`} aria-label={`Listen: ${v.fr}`}>
                      {v.glyph} {v.fr}
                    </button>
                  ))}
                </div>
              </div>
              <textarea
                value={taskText} onChange={(e) => setTaskText(e.target.value)}
                placeholder="Write (or speak) your French here… imperfect is perfect."
                autoCapitalize="none" autoCorrect="off" spellCheck={false}
                className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white min-h-24 mb-3"
              />
              <div className="flex gap-2 flex-wrap">
                <Btn theme={u.theme} onClick={submitTask} disabled={!taskText.trim() || taskBusy}>{taskBusy ? "Checking…" : "Submit"}</Btn>
                <MicBtn theme={u.theme} onText={(txt) => setTaskText((prev) => (prev ? prev + " " : "") + txt)} />
              </div>
            </div>
          ) : (
            <div>
              <div className="rounded-2xl bg-emerald-50 border border-emerald-200 p-4 mb-3">
                <div className="text-sm font-semibold text-emerald-800">💚 {taskFb.praise}</div>
              </div>
              {(taskFb.corrections || []).length > 0 && (
                <div className="rounded-2xl bg-rose-50 border border-rose-200 p-4 mb-3">
                  <div className="text-xs uppercase tracking-wide text-rose-500 mb-2">Worth fixing</div>
                  {taskFb.corrections.map((c, i) => (
                    <div key={i} className="mb-2 text-sm">
                      <span className="line-through text-rose-600">{c.you_said}</span> → <span className="font-semibold text-stone-900">{c.fix}</span>
                      <div className="text-stone-500 text-xs">{c.why}</div>
                    </div>
                  ))}
                </div>
              )}
              <div className="rounded-2xl bg-stone-50 border border-stone-200 p-4 mb-3">
                <div className="text-xs uppercase tracking-wide text-stone-500 mb-1">A natural model answer</div>
                <div className="font-medium">{taskFb.model_answer} <button onClick={() => speak(taskFb.model_answer, u.settings.ttsRate)}>🔊</button></div>
              </div>
              <Btn theme={u.theme} onClick={() => setStep(4)}>Collect your new words →</Btn>
            </div>
          )}
        </Card>
      )}

      {step === 4 && (
        <Card className="p-6">
          <div className="text-center mb-4">
            <div className="text-4xl mb-1">🧺</div>
            <div className="font-serif text-xl font-bold">New words → your review deck</div>
            <div className="text-sm text-stone-500">They'll come back exactly when you're about to forget them.</div>
          </div>
          {unit.vocab.map((v) => (
            <div key={v.fr} className="flex items-center gap-3 rounded-2xl border border-stone-200 p-3 mb-2">
              <span className="text-2xl">{v.glyph}</span>
              <div className="flex-1">
                <div className="font-semibold">{v.fr}</div>
                <div className="text-xs text-stone-500">{v.en}</div>
              </div>
              <button onClick={() => speak(v.fr, u.settings.ttsRate)} className="min-h-11 min-w-11">🔊</button>
            </div>
          ))}
          <div className="text-center text-sm text-stone-500 my-3">Practice got {pScore}/{content.practice.length} on first try — honest score, honest schedule.</div>
          <Btn theme={u.theme} onClick={finishUnit} className="w-full">Finish lesson ✓</Btn>
        </Card>
      )}
    </Screen>
  );
}

/* ========================================================================== */
/*  PHONICS (Carol)                                                           */
/* ========================================================================== */

function Phonics({ u, setU, go, lessonId }) {
  const t = themeOf(u);
  const [openId, setOpenId] = useState(lessonId || null);
  const [echo, setEcho] = useState({}); // itemText -> similarity result

  const tryEcho = (text) => {
    const r = makeRecognizer(
      (heard) => setEcho((prev) => ({ ...prev, [text]: similarity(text, heard) })),
      () => {}
    );
    if (r) { stopSpeaking(); try { r.start(); } catch (e) {} }
  };

  const markDone = (id) => {
    setU((prev) => ({
      ...prev,
      phonicsDone: { ...prev.phonicsDone, [id]: true },
      minutes: prev.minutes + 3,
      stats: { ...prev.stats, speaking: [...prev.stats.speaking, 60].slice(-40) },
    }));
    setOpenId(null);
  };

  if (openId) {
    const lesson = PHONICS_LESSONS.find((p) => p.id === openId);
    return (
      <Screen>
        <button onClick={() => setOpenId(null)} className="text-sm text-stone-500 mb-3 min-h-11">← All sounds</button>
        <Card className="p-6">
          <h1 className="font-serif text-2xl font-bold mb-2">{lesson.emoji} {lesson.title}</h1>
          <p className="text-stone-600 text-sm mb-4">{lesson.tip}</p>
          <div className="text-xs uppercase tracking-wide text-stone-500 mb-2">Listen, then repeat out loud</div>
          {lesson.items.map((it) => (
            <div key={it.text} className="flex items-center gap-2 rounded-2xl border border-stone-200 p-3 mb-2">
              <button onClick={() => speak(it.text, 0.75)} className="text-xl min-h-11 min-w-11">🔊</button>
              <div className="flex-1">
                <div className="font-bold text-lg">{it.text}</div>
                <div className="text-xs text-stone-500">{it.note}</div>
              </div>
              {srSupported() ? (
                <button onClick={() => tryEcho(it.text)} className={`min-h-11 px-3 rounded-xl text-sm font-semibold ${echo[it.text] === undefined ? "bg-stone-100 text-stone-600" : echo[it.text] >= 60 ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                  {echo[it.text] === undefined ? "🎙️ try" : echo[it.text] >= 60 ? "✓ heard you!" : "again?"}
                </button>
              ) : null}
            </div>
          ))}
          {!srSupported() && <div className="text-xs text-stone-500 mb-3">No speech recognition on this browser — repeat out loud anyway; your mouth is the judge today.</div>}
          <Btn theme={u.theme} onClick={() => markDone(lesson.id)} className="w-full mt-2">Done — my ears are warmer ✓</Btn>
        </Card>
      </Screen>
    );
  }

  return (
    <Screen>
      <h1 className="font-serif text-2xl font-bold mb-1">The sounds of French</h1>
      <p className="text-sm text-stone-500 mb-4">Map the sounds correctly from day one and every later word arrives pre-pronounced.</p>
      {PHONICS_LESSONS.map((p) => (
        <button key={p.id} onClick={() => setOpenId(p.id)} className={`w-full text-left mb-3 rounded-3xl border p-4 flex items-center gap-4 min-h-20 ${u.phonicsDone[p.id] ? "bg-emerald-50 border-emerald-200" : "bg-white border-stone-200 shadow-sm hover:shadow-md"}`}>
          <span className="text-3xl">{p.emoji}</span>
          <div className="flex-1">
            <div className="font-semibold">{p.title}</div>
            <div className="text-xs text-stone-500 line-clamp-1">{p.tip}</div>
          </div>
          {u.phonicsDone[p.id] && <span className="text-emerald-600 font-bold">✓</span>}
        </button>
      ))}
    </Screen>
  );
}

/* ========================================================================== */
/*  CONNECTED-SPEECH MINI-LESSONS (Stuart)                                    */
/* ========================================================================== */

function CSLessons({ u, setU }) {
  const [openId, setOpenId] = useState(null);

  if (openId) {
    const lesson = CS_LESSONS.find((l) => l.id === openId);
    return (
      <Screen>
        <button onClick={() => setOpenId(null)} className="text-sm text-stone-500 mb-3 min-h-11">← All lessons</button>
        <Card className="p-6">
          <h1 className="font-serif text-2xl font-bold mb-3">{lesson.emoji} {lesson.title}</h1>
          <p className="text-stone-700 leading-relaxed text-sm mb-5">{lesson.body}</p>
          <div className="text-xs uppercase tracking-wide text-stone-500 mb-2">Hear it — slow, then natural</div>
          {lesson.pairs.map((p, i) => (
            <div key={i} className="rounded-2xl border border-stone-200 p-3 mb-2">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="font-semibold">{p.written}</div>
                  <div className="text-xs text-indigo-600 font-mono">{p.sounds}</div>
                </div>
                <div className="flex gap-1">
                  <button onClick={() => speak(p.written.replace(/.*→\s*/, ""), 0.7)} className="min-h-11 px-3 rounded-xl bg-stone-100 text-sm" aria-label={`Listen slowly: ${p.written}`}>🐢</button>
                  <button onClick={() => speak(p.written.replace(/.*→\s*/, ""), 1.0)} className="min-h-11 px-3 rounded-xl bg-stone-100 text-sm" aria-label={`Listen at natural speed: ${p.written}`}>🐇</button>
                </div>
              </div>
            </div>
          ))}
          <Btn
            theme={u.theme}
            className="w-full mt-3"
            onClick={() => { setU((prev) => ({ ...prev, csDone: { ...prev.csDone, [lesson.id]: true }, minutes: prev.minutes + 3 })); setOpenId(null); }}
          >
            Got it ✓
          </Btn>
        </Card>
      </Screen>
    );
  }

  return (
    <Screen>
      <h1 className="font-serif text-2xl font-bold mb-1">Why you can read it but not hear it</h1>
      <p className="text-sm text-stone-500 mb-4">Spoken French moves the word boundaries. These six short lessons are the decoder ring.</p>
      {CS_LESSONS.map((l) => (
        <button key={l.id} onClick={() => setOpenId(l.id)} className={`w-full text-left mb-3 rounded-3xl border p-4 flex items-center gap-4 min-h-20 ${u.csDone[l.id] ? "bg-emerald-50 border-emerald-200" : "bg-white border-stone-200 shadow-sm hover:shadow-md"}`}>
          <span className="text-3xl">{l.emoji}</span>
          <div className="flex-1">
            <div className="font-semibold">{l.title}</div>
            <div className="text-xs text-stone-500 line-clamp-1">{l.body}</div>
          </div>
          {u.csDone[l.id] && <span className="text-emerald-600 font-bold">✓</span>}
        </button>
      ))}
    </Screen>
  );
}

/* ========================================================================== */
/*  READING-WHILE-LISTENING LIBRARY                                           */
/* ========================================================================== */

function Library({ u, setU, go }) {
  const t = themeOf(u);
  const [topic, setTopic] = useState("");
  const [passage, setPassage] = useState(null);
  const [busy, setBusy] = useState(false);
  const [rate, setRate] = useState(u.settings.ttsRate);
  const [showText, setShowText] = useState(true);
  const [dual, setDual] = useState(u.settings.dualSubs);
  const [playingIdx, setPlayingIdx] = useState(-1);
  const [lookup, setLookup] = useState(null); // {word, busy, data}
  const [qAnswer, setQAnswer] = useState("");
  const [qDone, setQDone] = useState(false);
  const playingRef = useRef(false);

  const levelKey = ["A0", "A1"].includes(u.level) ? u.level : CEFR_ORDER.indexOf(u.level) >= 3 ? "B1" : "A2";

  const generate = async () => {
    setBusy(true); setPassage(null); setQDone(false); setQAnswer("");
    const fb = pick(FALLBACK_PASSAGES[levelKey] || FALLBACK_PASSAGES.A1);
    const res = await aiJSON(SYS.passage(u.level, topic), "Write the passage now.", null);
    const ok = res && Array.isArray(res.sentences) && res.sentences.length >= 3 && res.sentences.every((s) => s && s.fr);
    setPassage(ok ? res : fb);
    setBusy(false);
  };

  const playAll = () => {
    if (playingRef.current) { playingRef.current = false; stopSpeaking(); setPlayingIdx(-1); return; }
    playingRef.current = true;
    const sentences = passage.sentences;
    const playFrom = (i) => {
      if (!playingRef.current || i >= sentences.length) { playingRef.current = false; setPlayingIdx(-1); return; }
      setPlayingIdx(i);
      speak(sentences[i].fr, rate, () => playFrom(i + 1));
    };
    playFrom(0);
  };
  useEffect(() => () => { playingRef.current = false; stopSpeaking(); }, []);

  const tapWord = async (word, sentence) => {
    const clean = word.replace(/[^A-Za-zÀ-ÿ'’-]/g, "");
    if (!clean) return;
    setLookup({ word: clean, busy: true, data: null });
    const data = await aiJSON(
      SYS.wordLookup(),
      `Word: "${clean}"\nContext: "${sentence}"`,
      { word: clean, lemma: clean, meaning: "(saved — meaning to fill in at review)", example_fr: sentence, example_en: "", glyph: "" }
    );
    setLookup({ word: clean, busy: false, data });
  };

  const addLookup = () => {
    const d = lookup.data;
    setU((prev) => {
      const { deck } = addCardsToDeck(prev.deck, [{ fr: d.lemma || d.word, en: d.meaning, example: d.example_fr, exampleEn: d.example_en, glyph: d.glyph }], "library");
      return { ...prev, deck };
    });
    setLookup(null);
  };

  const finishPassage = () => {
    setQDone(true);
    setU((prev) => ({
      ...prev,
      stats: { ...prev.stats, reading: prev.stats.reading + 1 },
      minutes: prev.minutes + 5,
      streak: { ...prev.streak, days: { ...prev.streak.days, [todayKey()]: true } },
    }));
  };

  if (!passage) {
    return (
      <Screen>
        <h1 className="font-serif text-2xl font-bold mb-1">Read & listen</h1>
        <p className="text-sm text-stone-500 mb-4">A fresh passage at your level ({u.level}). Read while listening, tap any word to save it, then hide the text and test your ear.</p>
        <Card className="p-5">
          <input
            value={topic} onChange={(e) => setTopic(e.target.value)}
            placeholder="Topic you fancy (optional) — e.g. cooking, Paris, cats…"
            className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white mb-3"
          />
          {busy ? <Spinner label="Writing your passage…" /> : <Btn theme={u.theme} onClick={generate} className="w-full">Generate passage</Btn>}
        </Card>
      </Screen>
    );
  }

  return (
    <Screen>
      <div className="flex items-center justify-between mb-3">
        <h1 className="font-serif text-xl font-bold">{passage.title}</h1>
        <Btn variant="ghost" small onClick={() => { stopSpeaking(); playingRef.current = false; setPassage(null); }}>new ↻</Btn>
      </div>
      <Card className="p-4 mb-3">
        <div className="flex items-center gap-3 flex-wrap">
          <Btn theme={u.theme} small onClick={playAll}>{playingIdx >= 0 ? "◼ Stop" : "🔊 Play all"}</Btn>
          <label className="flex items-center gap-2 text-sm text-stone-600">
            <span>🐢</span>
            <input type="range" min="0.6" max="1.2" step="0.05" value={rate} onChange={(e) => setRate(parseFloat(e.target.value))} className="w-24" />
            <span>🐇 {rate.toFixed(2)}×</span>
          </label>
          <button onClick={() => setShowText(!showText)} className={`min-h-11 px-3 rounded-xl text-sm font-semibold ${showText ? "bg-stone-100 text-stone-700" : "bg-stone-800 text-white"}`}>
            {showText ? "🙈 Hide text (test your ear)" : "👀 Show text"}
          </button>
          {showText && (
            <button onClick={() => setDual(!dual)} className={`min-h-11 px-3 rounded-xl text-sm font-semibold ${dual ? "bg-sky-100 text-sky-800" : "bg-stone-100 text-stone-500"}`}>
              EN {dual ? "on" : "off"}
            </button>
          )}
        </div>
      </Card>

      {showText ? (
        <Card className="p-5 mb-3">
          {passage.sentences.map((s, i) => (
            <div key={i} className={`mb-3 rounded-xl p-2 -mx-2 transition-colors ${playingIdx === i ? "bg-amber-50" : ""}`}>
              <div className="text-lg leading-relaxed">
                {s.fr.split(/\s+/).map((w, wi) => (
                  <span key={wi}>
                    <button onClick={() => tapWord(w, s.fr)} className="hover:bg-amber-100 rounded px-0.5 active:bg-amber-200">{w}</button>{" "}
                  </span>
                ))}
                <button onClick={() => speak(s.fr, rate)} className="text-stone-300 hover:text-stone-500 text-sm">🔊</button>
              </div>
              {dual && s.en && <div className="text-sm text-stone-500">{s.en}</div>}
            </div>
          ))}
          <div className="text-xs text-stone-500">Tap any word to look it up and add it to your deck.</div>
        </Card>
      ) : (
        <Card className="p-8 mb-3 text-center">
          <div className="text-4xl mb-2">👂</div>
          <div className="text-stone-600 text-sm">Text hidden. Play it and see how much lands without your eyes helping.</div>
        </Card>
      )}

      {passage.question && !qDone && (
        <Card className="p-5">
          <div className="text-xs uppercase tracking-wide text-stone-500 mb-1">Prove you got it (forced output!)</div>
          <div className="font-semibold mb-2">{passage.question} <button onClick={() => speak(passage.question, rate)}>🔊</button></div>
          <textarea
            value={qAnswer} onChange={(e) => setQAnswer(e.target.value)}
            placeholder="Answer in French, even roughly…"
            autoCapitalize="none" autoCorrect="off" spellCheck={false}
            className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white min-h-16 mb-2"
          />
          <div className="flex gap-2 items-center flex-wrap">
            <Btn theme={u.theme} small onClick={finishPassage} disabled={!qAnswer.trim()}>Submit & finish</Btn>
            <MicBtn theme={u.theme} onText={(txt) => setQAnswer((prev) => (prev ? prev + " " : "") + txt)} />
          </div>
        </Card>
      )}
      {qDone && (
        <Card className="p-5 bg-emerald-50 border-emerald-200">
          <div className="font-semibold text-emerald-800 mb-1">Passage finished ✓</div>
          {passage.question_answer && <div className="text-sm text-stone-600">A natural answer: <span className="font-medium">{passage.question_answer}</span> <button onClick={() => speak(passage.question_answer, rate)}>🔊</button></div>}
          <div className="flex gap-2 mt-3">
            <Btn variant="secondary" small onClick={() => { setPassage(null); }}>Another passage</Btn>
            <Btn theme={u.theme} small onClick={() => go("home")}>Home</Btn>
          </div>
        </Card>
      )}

      {lookup && (
        <div className="fixed inset-0 z-50 bg-black/40 flex items-end sm:items-center justify-center p-4" onClick={() => setLookup(null)}>
          <div className="bg-white rounded-3xl p-5 w-full max-w-sm" onClick={(e) => e.stopPropagation()}>
            {lookup.busy ? (
              <Spinner label={`Looking up “${lookup.word}”…`} />
            ) : (
              <div>
                <div className="flex items-center gap-3 mb-2">
                  {lookup.data.glyph && <span className="text-3xl">{lookup.data.glyph}</span>}
                  <div>
                    <div className="font-serif text-xl font-bold">{lookup.data.lemma || lookup.word}</div>
                    <div className="text-sm text-stone-500">{lookup.data.meaning}</div>
                  </div>
                  <button onClick={() => speak(lookup.data.lemma || lookup.word, u.settings.ttsRate)} className="ml-auto min-h-11 min-w-11">🔊</button>
                </div>
                {lookup.data.example_fr && <div className="text-sm text-stone-600 italic mb-3">“{lookup.data.example_fr}”</div>}
                <div className="flex gap-2">
                  <Btn theme={u.theme} small onClick={addLookup} className="flex-1">+ Add to my deck</Btn>
                  <Btn variant="ghost" small onClick={() => setLookup(null)}>Close</Btn>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </Screen>
  );
}

/* ========================================================================== */
/*  AI CONVERSATION PARTNER — strictness, correction report → SRS             */
/* ========================================================================== */

function Conversation({ u, setU, go }) {
  const t = themeOf(u);
  const isBeginner = u.track === "foundations";
  const [scenario, setScenario] = useState(null);
  const [messages, setMessages] = useState([]); // {role:'user'|'ai', fr, en, note}
  const [suggestions, setSuggestions] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [report, setReport] = useState(null);
  const [reportBusy, setReportBusy] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => { bottomRef.current && bottomRef.current.scrollIntoView({ behavior: "smooth" }); }, [messages, busy]);

  const availScenarios = SCENARIOS.filter((s) => CEFR_ORDER.indexOf(u.level) >= CEFR_ORDER.indexOf(s.minLevel) || s.id === "libre");

  const startScenario = async (sc) => {
    setScenario(sc); setMessages([]); setReport(null); setSuggestions([]);
    setBusy(true);
    const first = await aiJSON(
      SYS.conversation(u, sc, u.settings.strictness),
      `Open the conversation naturally in this scenario. The learner hasn't spoken yet.`,
      { reply_fr: sc.id === "cafe" ? "Bonjour ! Bienvenue. Qu'est-ce que je vous sers ?" : "Bonjour ! Ça va ?", reply_en: sc.id === "cafe" ? "Hello! Welcome. What can I get you?" : "Hello! How are you?", note: null, suggestions: isBeginner ? [{ fr: "Bonjour ! Un café, s'il vous plaît.", en: "Hello! A coffee, please." }, { fr: "Bonjour, ça va bien !", en: "Hello, I'm fine!" }, { fr: "Je voudrais un thé.", en: "I'd like a tea." }] : [] }
    );
    pushAi(first);
    setBusy(false);
  };

  const pushAi = (res) => {
    if (!res || !res.reply_fr) {
      setMessages((m) => [...m, { role: "ai", fr: "Désolé, je n'ai pas pu répondre (connexion). On réessaie ?", en: "Sorry, I couldn't answer (connection). Try again?", note: null }]);
      return;
    }
    setMessages((m) => [...m, { role: "ai", fr: res.reply_fr, en: res.reply_en || "", note: res.note || null }]);
    setSuggestions(isBeginner && Array.isArray(res.suggestions) ? res.suggestions.slice(0, 3) : []);
    speak(res.reply_fr, u.settings.ttsRate);
  };

  const send = async (text) => {
    const msg = (text || input).trim();
    if (!msg || busy) return;
    setInput(""); setSuggestions([]);
    setMessages((m) => [...m, { role: "user", fr: msg }]);
    setBusy(true);
    const history = [...messages, { role: "user", fr: msg }].map((m) => ({ role: m.role === "ai" ? "assistant" : "user", content: m.fr }));
    let res = null;
    try {
      const raw = await callClaude(
        SYS.conversation(u, scenario, u.settings.strictness) + " Respond with ONLY valid JSON. No preamble, no markdown.",
        history
      );
      res = JSON.parse(extractJSON(raw));
    } catch (e) { res = null; }
    pushAi(res);
    setBusy(false);
  };

  const endConversation = async () => {
    setReportBusy(true);
    const userLines = messages.filter((m) => m.role === "user").map((m) => m.fr);
    const transcript = messages.map((m) => `${m.role === "ai" ? "Partner" : u.name}: ${m.fr}`).join("\n");
    const rep = await aiJSON(
      SYS.report(u.level),
      `Transcript:\n${transcript}\n\nLearner messages to review: ${JSON.stringify(userLines)}`,
      { summary: "Report unavailable offline — but you produced real French out loud today, and that is the work.", corrections: [], wins: ["You kept the conversation going in French."] }
    );
    setReport(rep);
    setReportBusy(false);
    setU((prev) => {
      const qualifies = userLines.length >= 4;
      return {
        ...prev,
        stats: { ...prev.stats, conversations: prev.stats.conversations + 1, speaking: [...prev.stats.speaking, clamp(50 + userLines.length * 5, 40, 95)].slice(-40) },
        minutes: prev.minutes + Math.max(3, userLines.length),
        streak: qualifies ? { ...prev.streak, days: { ...prev.streak.days, [todayKey()]: true } } : prev.streak,
      };
    });
  };

  const addCorrections = () => {
    const items = (report.corrections || []).map((c) => ({
      fr: c.fix, en: `not: “${c.you_said}” — ${c.why}`, example: c.fix, glyph: "🛠️",
    }));
    setU((prev) => {
      const { deck } = addCardsToDeck(prev.deck, items, "conversation");
      return { ...prev, deck };
    });
    go("home");
  };

  if (!scenario) {
    return (
      <Screen>
        <h1 className="font-serif text-2xl font-bold mb-1">Conversation</h1>
        <p className="text-sm text-stone-500 mb-2">
          Mode: <Chip tone={u.settings.strictness === "strict" ? "red" : u.settings.strictness === "lenient" ? "green" : "amber"}>{u.settings.strictness}</Chip>
          {u.settings.strictness !== "lenient" ? " — errors get named, not silently forgiven. That's a feature." : " — gentle mode: only blocking errors get flagged."}
        </p>
        <p className="text-xs text-stone-500 mb-4">Every conversation ends with an honest correction report, and the fixes go to your review deck.</p>
        {availScenarios.map((s) => (
          <button key={s.id} onClick={() => startScenario(s)} className="w-full text-left mb-3 rounded-3xl border border-stone-200 bg-white shadow-sm p-4 flex items-center gap-4 hover:shadow-md min-h-20">
            <span className="text-3xl">{s.emoji}</span>
            <div className="flex-1">
              <div className="font-semibold">{s.title}</div>
              <div className="text-xs text-stone-500">{s.desc}</div>
            </div>
            <Chip tone="stone">{s.minLevel}+</Chip>
          </button>
        ))}
      </Screen>
    );
  }

  if (report) {
    return (
      <Screen>
        <Card className="p-6">
          <h1 className="font-serif text-2xl font-bold mb-2">📋 Correction report</h1>
          <p className="text-stone-600 text-sm mb-4">{report.summary}</p>
          {(report.wins || []).length > 0 && (
            <div className="rounded-2xl bg-emerald-50 border border-emerald-200 p-4 mb-3">
              <div className="text-xs uppercase tracking-wide text-emerald-600 mb-1">What worked</div>
              {report.wins.map((w, i) => <div key={i} className="text-sm text-emerald-900">💚 {w}</div>)}
            </div>
          )}
          {(report.corrections || []).length > 0 ? (
            <div className="rounded-2xl bg-rose-50 border border-rose-200 p-4 mb-4">
              <div className="text-xs uppercase tracking-wide text-rose-500 mb-2">What to fix (the useful part)</div>
              {report.corrections.map((c, i) => (
                <div key={i} className="mb-3 text-sm">
                  <div><span className="line-through text-rose-600">{c.you_said}</span></div>
                  <div className="font-semibold text-stone-900">→ {c.fix} <button onClick={() => speak(c.fix, u.settings.ttsRate)}>🔊</button></div>
                  <div className="text-stone-500 text-xs">{c.why}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-stone-500 mb-4">No corrections worth flagging this time — raise the difficulty next round.</div>
          )}
          <div className="flex gap-2">
            {(report.corrections || []).length > 0 && <Btn theme={u.theme} onClick={addCorrections} className="flex-1">Add fixes to my deck</Btn>}
            <Btn variant="secondary" onClick={() => go("home")}>Done</Btn>
          </div>
        </Card>
      </Screen>
    );
  }

  return (
    <Screen>
      <div className="flex items-center justify-between mb-3">
        <div className="font-semibold">{scenario.emoji} {scenario.title}</div>
        <Btn variant="secondary" small onClick={endConversation} disabled={reportBusy || messages.filter((m) => m.role === "user").length === 0}>
          {reportBusy ? "Writing report…" : "End + report"}
        </Btn>
      </div>
      <div className="mb-3">
        {messages.map((m, i) => (
          <div key={i} className={`mb-2 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-[85%] rounded-3xl px-4 py-3 ${m.role === "user" ? `${t.btn.split(" ")[0]} text-white` : "bg-white border border-stone-200"}`}>
              <div className={m.role === "user" ? "" : "text-stone-900"}>
                {m.fr} {m.role === "ai" && <button onClick={() => speak(m.fr, u.settings.ttsRate)} className="opacity-60">🔊</button>}
              </div>
              {m.role === "ai" && u.settings.dualSubs && m.en && <div className="text-xs text-stone-500 mt-1">{m.en}</div>}
              {m.note && <div className="text-xs mt-2 rounded-xl bg-amber-50 border border-amber-200 text-amber-800 px-2 py-1">✏️ {m.note}</div>}
            </div>
          </div>
        ))}
        {busy && <div className="text-stone-500 text-sm px-2">…</div>}
        <div ref={bottomRef} />
      </div>
      {suggestions.length > 0 && (
        <div className="mb-3 flex flex-col gap-2">
          <div className="text-xs text-stone-500">Tap a reply, or write your own:</div>
          {suggestions.map((s, i) => (
            <button key={i} onClick={() => send(s.fr)} className="text-left rounded-2xl border border-dashed border-stone-300 bg-stone-50 px-4 py-2.5 text-sm hover:bg-white min-h-11">
              <span className="font-medium">{s.fr}</span>
              {u.settings.dualSubs && <span className="text-stone-500"> — {s.en}</span>}
            </button>
          ))}
        </div>
      )}
      <div className="flex gap-2 items-end">
        <textarea
          value={input} onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          placeholder="Réponds en français…" rows={1}
          autoCapitalize="none" autoCorrect="off" spellCheck={false}
          className="flex-1 rounded-2xl border border-stone-300 px-4 py-3 bg-white resize-none"
        />
        <MicBtn theme={u.theme} onText={(txt) => setInput((prev) => (prev ? prev + " " : "") + txt)} />
        <Btn theme={u.theme} onClick={() => send()} disabled={busy || !input.trim()} small>➤</Btn>
      </div>
    </Screen>
  );
}

/* ========================================================================== */
/*  ON-DEMAND EXERCISE GENERATOR                                              */
/* ========================================================================== */

function ExerciseGen({ u, setU, go }) {
  const t = themeOf(u);
  const [focusKind, setFocusKind] = useState("weak");
  const [grammarPick, setGrammarPick] = useState(GRAMMAR_TOPICS[0].id);
  const [customText, setCustomText] = useState("");
  const [type, setType] = useState("cloze");
  const [ex, setEx] = useState(null);
  const [busy, setBusy] = useState(false);
  const [idx, setIdx] = useState(0);
  const [answer, setAnswer] = useState("");
  const [verdict, setVerdict] = useState(null);
  const [order, setOrder] = useState([]); // for reorder
  const [pool, setPool] = useState([]);
  const [score, setScore] = useState(0);

  const skills = skillScores(u);
  const weakest = Object.entries(skills).sort((a, b) => a[1] - b[1])[0][0];
  const weakFocusDesc = {
    listening: "listening comprehension — short dictation-style sentences with connected speech",
    speaking: "producing spoken-style French sentences",
    reading: "reading comprehension at the learner's level",
    writing: "writing accurate French — gender, agreement, word order",
    vocabulary: "high-frequency vocabulary the learner should know",
  }[weakest];

  const currentUnit = FOUNDATIONS_UNITS.find((un) => !u.lessons[un.id]) || FOUNDATIONS_UNITS[FOUNDATIONS_UNITS.length - 1];

  const focusDesc = () =>
    focusKind === "weak" ? `the learner's detected weak spot: ${weakFocusDesc}`
    : focusKind === "unit" ? `the current lesson topic: ${currentUnit.grammar.point} (scenario: ${currentUnit.scenario})`
    : focusKind === "grammar" ? `the grammar point: ${GRAMMAR_TOPICS.find((g) => g.id === grammarPick).name}`
    : `this text supplied by the learner (build the drill from its vocabulary and structures): """${customText.slice(0, 600)}"""`;

  const generate = async () => {
    setBusy(true); setEx(null); setIdx(0); setScore(0); setVerdict(null); setAnswer("");
    const fb = fallbackExercise(type);
    const res = await aiJSON(SYS.exercise(focusDesc(), type, u.level), "Generate the drill now.", null);
    const ok = res && Array.isArray(res.items) && res.items.length > 0 && res.items.every((x) => x && x.prompt !== undefined && x.answer);
    const chosen = ok ? res : fb;
    setEx(chosen);
    initItem(chosen, 0);
    setBusy(false);
    if (type === "dictation") setTimeout(() => speak(chosen.items[0].answer, u.settings.ttsRate), 400);
  };

  const initItem = (exercise, i) => {
    const item = exercise.items[i];
    if (exercise.type === "reorder") {
      const words = item.words && item.words.length ? [...item.words] : tokenize(item.answer);
      setPool([...words].sort(() => Math.random() - 0.5));
      setOrder([]);
    }
  };

  const check = () => {
    const item = ex.items[idx];
    const given = ex.type === "reorder" ? order.join(" ") : answer;
    const v = checkAnswer(item.answer, given);
    setVerdict(v);
    if (v === "exact" || v === "close") setScore((s) => s + 1);
    if (ex.type !== "dictation") speak(item.answer, u.settings.ttsRate);
    setU((prev) => ({ ...prev, stats: { ...prev.stats, writing: [...prev.stats.writing, v === "exact" ? 100 : v === "close" ? 75 : 20].slice(-40) } }));
  };

  const next = () => {
    setVerdict(null); setAnswer("");
    if (idx + 1 < ex.items.length) {
      const ni = idx + 1;
      setIdx(ni); initItem(ex, ni);
      if (ex.type === "dictation") setTimeout(() => speak(ex.items[ni].answer, u.settings.ttsRate), 300);
    } else {
      setU((prev) => ({ ...prev, minutes: prev.minutes + 3 }));
      setEx({ ...ex, finished: true });
    }
  };

  const addMissesToDeck = () => {
    // add answers the learner missed as SRS cards
    const missed = ex.items.filter((_, i) => i <= idx).map((item) => ({ fr: item.answer, en: item.prompt.replace(/_+/g, "…"), glyph: "🎯" }));
    setU((prev) => {
      const { deck } = addCardsToDeck(prev.deck, missed, "exercise");
      return { ...prev, deck };
    });
    go("home");
  };

  if (!ex) {
    const Option = ({ kind, label, sub }) => (
      <button onClick={() => setFocusKind(kind)} className={`w-full text-left rounded-2xl border p-3 mb-2 min-h-14 ${focusKind === kind ? `${t.soft} border-stone-400` : "bg-white border-stone-200"}`}>
        <div className="font-semibold text-sm">{label}</div>
        <div className="text-xs text-stone-500">{sub}</div>
      </button>
    );
    return (
      <Screen>
        <h1 className="font-serif text-2xl font-bold mb-1">Make me an exercise</h1>
        <p className="text-sm text-stone-500 mb-4">Pick a focus and a format — a fresh targeted drill, checked and explained.</p>
        <Card className="p-5 mb-3">
          <div className="text-xs uppercase tracking-wide text-stone-500 mb-2">Focus</div>
          <Option kind="weak" label={`My weak spot: ${weakest}`} sub="Auto-detected from your recent accuracy" />
          <Option kind="unit" label={`Current lesson: ${currentUnit.grammar.point}`} sub={currentUnit.title} />
          <Option kind="grammar" label="A grammar point" sub="Choose from the drill topics" />
          {focusKind === "grammar" && (
            <select value={grammarPick} onChange={(e) => setGrammarPick(e.target.value)} className="w-full rounded-xl border border-stone-300 px-3 py-2.5 mb-2 bg-white">
              {GRAMMAR_TOPICS.map((g) => <option key={g.id} value={g.id}>{g.name}</option>)}
            </select>
          )}
          <Option kind="custom" label="Any French text I paste" sub="Menu, email, song lyric — anything" />
          {focusKind === "custom" && (
            <textarea value={customText} onChange={(e) => setCustomText(e.target.value)} placeholder="Paste French text here…" className="w-full rounded-xl border border-stone-300 px-3 py-2.5 min-h-20 bg-white" />
          )}
        </Card>
        <Card className="p-5 mb-4">
          <div className="text-xs uppercase tracking-wide text-stone-500 mb-2">Format</div>
          <div className="grid grid-cols-2 gap-2">
            {[["cloze", "✂️ Cloze (fill gap)"], ["translation", "🔁 Translation"], ["dictation", "🎧 Dictation"], ["reorder", "🧩 Reorder"]].map(([k, label]) => (
              <button key={k} onClick={() => setType(k)} className={`rounded-2xl border p-3 text-sm font-semibold min-h-12 ${type === k ? `${t.soft} border-stone-400` : "bg-white border-stone-200"}`}>
                {label}
              </button>
            ))}
          </div>
        </Card>
        {busy ? <Spinner label="Building your drill…" /> : <Btn theme={u.theme} onClick={generate} className="w-full" disabled={focusKind === "custom" && !customText.trim()}>Generate ⚡</Btn>}
      </Screen>
    );
  }

  if (ex.finished) {
    return (
      <Screen>
        <Card className="p-6 text-center">
          <div className="text-4xl mb-2">{score === ex.items.length ? "🏆" : "💪"}</div>
          <h1 className="font-serif text-2xl font-bold mb-2">{score}/{ex.items.length} first-time right</h1>
          <p className="text-stone-600 text-sm mb-4">{score === ex.items.length ? "Clean sweep. Next time we make it harder — easy reps don't build memory." : "The misses are the valuable part — put them into rotation."}</p>
          <div className="flex gap-2 justify-center flex-wrap">
            {score < ex.items.length && <Btn theme={u.theme} onClick={addMissesToDeck}>Add to my deck</Btn>}
            <Btn variant="secondary" onClick={() => setEx(null)}>Another drill</Btn>
            <Btn variant="ghost" onClick={() => go("home")}>Home</Btn>
          </div>
        </Card>
      </Screen>
    );
  }

  const item = ex.items[idx];
  return (
    <Screen>
      <div className="flex items-center justify-between mb-3">
        <Chip tone="indigo">{ex.type} · {idx + 1}/{ex.items.length}</Chip>
        <Btn variant="ghost" small onClick={() => setEx(null)}>✕ quit</Btn>
      </div>
      <Card className="p-6">
        <div className="text-sm text-stone-500 mb-2">{ex.instructions}</div>
        {ex.type === "dictation" ? (
          <div className="mb-4">
            <SpeakerBtn text={item.answer} rate={u.settings.ttsRate} theme={u.theme} label="Play sentence" big />
            <div className="text-xs text-stone-500 mt-1">Replay as often as you need. Type exactly what you hear.</div>
          </div>
        ) : (
          <div className="font-serif text-lg font-bold mb-4">{item.prompt}</div>
        )}

        {ex.type === "reorder" ? (
          <div className="mb-3">
            <div className="min-h-14 rounded-2xl border-2 border-dashed border-stone-300 p-2 mb-2 flex flex-wrap gap-2">
              {order.map((w, i) => (
                <button key={i} onClick={() => { setOrder(order.filter((_, k) => k !== i)); setPool([...pool, w]); }} className="rounded-xl bg-stone-800 text-white px-3 py-2 text-sm min-h-11">
                  {w}
                </button>
              ))}
              {order.length === 0 && <span className="text-stone-300 text-sm self-center px-2">tap words below in order…</span>}
            </div>
            <div className="flex flex-wrap gap-2">
              {pool.map((w, i) => (
                <button key={i} onClick={() => { setPool(pool.filter((_, k) => k !== i)); setOrder([...order, w]); }} className="rounded-xl bg-stone-100 border border-stone-300 px-3 py-2 text-sm min-h-11">
                  {w}
                </button>
              ))}
            </div>
          </div>
        ) : !verdict ? (
          <input
            value={answer} onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && answer.trim() && check()}
            placeholder="Your French…" autoCapitalize="none" autoCorrect="off" spellCheck={false}
            className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white text-lg mb-3"
          />
        ) : null}

        {!verdict ? (
          <div className="flex gap-2 flex-wrap">
            <Btn theme={u.theme} onClick={check} disabled={ex.type === "reorder" ? order.length === 0 : !answer.trim()}>Check</Btn>
            {ex.type !== "reorder" && <MicBtn theme={u.theme} onText={setAnswer} />}
          </div>
        ) : (
          <div>
            <div className={`rounded-2xl p-4 mb-3 ${verdict === "exact" ? "bg-emerald-50 border border-emerald-200" : verdict === "close" ? "bg-amber-50 border border-amber-200" : "bg-rose-50 border border-rose-200"}`}>
              <div className="font-bold">{item.answer} <button onClick={() => speak(item.answer, u.settings.ttsRate)}>🔊</button></div>
              <div className="text-sm mt-1 text-stone-600">
                {verdict === "exact" ? "Spot on." : verdict === "close" ? "Nearly — check the details." : "Compare carefully with yours."}
              </div>
              {item.explanation && <div className="text-sm mt-2 text-stone-700 border-t border-stone-200 pt-2">💡 {item.explanation}</div>}
            </div>
            <Btn theme={u.theme} onClick={next}>{idx + 1 < ex.items.length ? "Next" : "Finish"}</Btn>
          </div>
        )}
      </Card>
    </Screen>
  );
}

/* ========================================================================== */
/*  LISTENING COMPREHENSION TRAINER (Stuart's flagship)                       */
/*  listen → dictate → auto-diff → connected-speech gap analysis →            */
/*  shadow → retell → narrow-listening + ladder                               */
/* ========================================================================== */

function ListeningTrainer({ u, setU, go }) {
  const t = themeOf(u);
  const rung = CLIP_LADDER[clamp(u.trainer.rung, 1, 8) - 1];
  const [clip, setClip] = useState(null);
  const [phase, setPhase] = useState("loading"); // loading|listen|dictate|diff|gaps|shadow|retell|done
  const [listens, setListens] = useState(0);
  const [dictation, setDictation] = useState("");
  const [diff, setDiff] = useState(null);
  const [gaps, setGaps] = useState(null);
  const [gapsBusy, setGapsBusy] = useState(false);
  const [chunks, setChunks] = useState([]);
  const [chunkIdx, setChunkIdx] = useState(0);
  const [shadowResults, setShadowResults] = useState([]);
  const [retellText, setRetellText] = useState("");
  const [retellFb, setRetellFb] = useState(null);
  const [retellBusy, setRetellBusy] = useState(false);
  const [rate, setRate] = useState(rung.rate);
  const [showSubs, setShowSubs] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      const fb = pick(FALLBACK_CLIPS[rung.rung]);
      const res = await aiJSON(SYS.clip(rung), "Write the clip now.", null);
      const ok = res && res.text && tokenize(res.text).length >= 12;
      if (!alive) return;
      const chosen = ok ? res : fb;
      try { window.__deuxClip = chosen.text; } catch (e) { /* test/debug hook only */ }
      setClip(chosen);
      setPhase("listen");
    })();
    return () => { alive = false; stopSpeaking(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const play = () => { speak(clip.text, rate); setListens((n) => n + 1); };

  const runDiff = () => {
    const d = wordDiff(clip.text, dictation);
    setDiff(d);
    setU((prev) => ({
      ...prev,
      stats: { ...prev.stats, dictation: [...prev.stats.dictation, { date: todayKey(), acc: d.acc, rung: rung.rung }].slice(-60) },
    }));
    setPhase("diff");
  };

  const runGaps = async () => {
    setPhase("gaps");
    const segs = missedSegments(diff.targetTokens);
    if (segs.length === 0) { setGaps([]); return; }
    setGapsBusy(true);
    const res = await aiJSON(
      SYS.gaps(),
      `Full clip: "${clip.text}"\nMissed segments: ${JSON.stringify(segs)}`,
      { gaps: localGapAnalysis(segs) }
    );
    setGaps(Array.isArray(res && res.gaps) && res.gaps.length ? res.gaps : localGapAnalysis(segs));
    setGapsBusy(false);
  };

  const startShadow = () => {
    setChunks(chunkText(clip.text));
    setChunkIdx(0); setShadowResults([]);
    setPhase("shadow");
  };

  const shadowChunk = (text) => { speak(text, Math.min(rate, 0.95)); };

  const recordShadow = (text) => {
    const r = makeRecognizer(
      (heard) => {
        const sim = similarity(text, heard);
        setShadowResults((prev) => { const n = [...prev]; n[chunkIdx] = { sim, heard }; return n; });
        setU((prev) => ({ ...prev, stats: { ...prev.stats, speaking: [...prev.stats.speaking, sim].slice(-40) } }));
      },
      () => {}
    );
    if (r) { stopSpeaking(); try { r.start(); } catch (e) {} }
  };

  const selfAssessShadow = (ok) => {
    setShadowResults((prev) => { const n = [...prev]; n[chunkIdx] = { sim: ok ? 80 : 40, heard: null, self: true }; return n; });
    setU((prev) => ({ ...prev, stats: { ...prev.stats, speaking: [...prev.stats.speaking, ok ? 75 : 45].slice(-40) } }));
  };

  const submitRetell = async () => {
    if (!retellText.trim()) return;
    setRetellBusy(true);
    const fb = await aiJSON(
      SYS.retell(rung.level.replace("+", "")),
      `Original clip: "${clip.text}"\nLearner's retell: "${retellText}"`,
      { score: 3, feedback: "Offline — but you turned listening into speaking, which is the entire point of this step.", corrections: [], better_version: clip.text }
    );
    setRetellFb(fb);
    setRetellBusy(false);
  };

  const finishClip = (advance) => {
    setU((prev) => {
      const newRung = advance ? clamp(prev.trainer.rung + 1, 1, 8) : prev.trainer.rung;
      const { deck } = addCardsToDeck(
        prev.deck,
        (retellFb && retellFb.corrections ? retellFb.corrections : []).map((c) => ({ fr: c.fix, en: `not: “${c.you_said}” — ${c.why}`, glyph: "🛠️" })),
        "trainer"
      );
      return {
        ...prev,
        deck,
        trainer: { ...prev.trainer, rung: newRung, clipsDone: prev.trainer.clipsDone + 1, relistens: prev.trainer.relistens + Math.max(0, listens - 1), history: [...prev.trainer.history, { date: todayKey(), rung: rung.rung, acc: diff ? diff.acc : null, listens }].slice(-60) },
        minutes: prev.minutes + 10,
        stats: { ...prev.stats, speaking: prev.stats.speaking },
        streak: { ...prev.streak, days: { ...prev.streak.days, [todayKey()]: true } },
      };
    });
    go("home");
  };

  if (phase === "loading" || !clip) return <Screen><Spinner label={`Preparing a rung-${rung.rung} clip…`} /></Screen>;

  const Steps = () => {
    // dots + a label for only the current step: always fits a phone width
    const order = ["listen", "dictate", "diff", "gaps", "shadow", "retell", "done"];
    const labels = { listen: "Step 1 · Listen", dictate: "Step 2 · Dictation", diff: "Step 3 · The diff", gaps: "Step 4 · Why you missed it", shadow: "Step 5 · Shadow", retell: "Step 6 · Retell", done: "Done" };
    const cur = order.indexOf(phase);
    return (
      <div className="flex items-center gap-2.5 mb-4">
        <div className="flex gap-1.5 shrink-0">
          {order.map((s, i) => (
            <div key={s} className={`h-2 rounded-full transition-all ${i === cur ? `w-5 ${t.bar}` : i < cur ? "w-2 bg-emerald-400" : "w-2 bg-stone-300"}`} />
          ))}
        </div>
        <span className="text-xs font-semibold text-stone-600 truncate">{labels[order[cur]]}</span>
      </div>
    );
  };

  return (
    <Screen>
      <div className="flex items-center justify-between mb-2">
        <h1 className="font-serif text-xl font-bold">🎧 Listening trainer</h1>
        <Chip tone="indigo">Rung {rung.rung}/8 · {rung.level}</Chip>
      </div>
      <Steps />

      {phase === "listen" && (
        <Card className="p-6 text-center">
          <div className="text-4xl mb-3">👂</div>
          <h2 className="font-serif text-xl font-bold mb-1">First: just listen. No text.</h2>
          <p className="text-sm text-stone-500 mb-4">Your eyes don't get to help yet. Listen 1–2 times and hold whatever you catch.</p>
          <div className="flex items-center justify-center gap-3 mb-4 flex-wrap">
            <Btn theme={u.theme} onClick={play}>🔊 Play clip ({listens}× so far)</Btn>
          </div>
          <label className="flex items-center gap-2 text-sm text-stone-600 justify-center mb-5">
            <span>🐢</span>
            <input type="range" min="0.6" max="1.1" step="0.05" value={rate} onChange={(e) => setRate(parseFloat(e.target.value))} className="w-28" />
            <span>🐇 {rate.toFixed(2)}×</span>
          </label>
          <Btn theme={u.theme} onClick={() => setPhase("dictate")} disabled={listens === 0}>
            I've listened — dictation →
          </Btn>
          <div className="text-xs text-stone-500 mt-2">
            {listens === 0 ? "Play the clip at least once to continue." : "Tip: two listens beats one. Re-listening to the same clip is the method, not cheating."}
          </div>
        </Card>
      )}

      {phase === "dictate" && (
        <Card className="p-6">
          <h2 className="font-serif text-xl font-bold mb-1">Type what you hear</h2>
          <p className="text-sm text-stone-500 mb-3">Every word you can catch — guesses welcome. The diff will show exactly where spoken French hid the boundaries.</p>
          <div className="flex gap-2 mb-3 flex-wrap">
            <SpeakerBtn text={clip.text} rate={rate} theme={u.theme} label="Replay" />
            <Btn variant="ghost" small onClick={() => { setListens((n) => n + 1); speak(clip.text, Math.max(0.6, rate - 0.2)); }}>🐢 slower</Btn>
          </div>
          <textarea
            value={dictation} onChange={(e) => setDictation(e.target.value)}
            placeholder="J'écris ce que j'entends…" autoCapitalize="none" autoCorrect="off" spellCheck={false}
            className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white min-h-28 mb-3 text-lg"
          />
          <Btn theme={u.theme} onClick={runDiff} disabled={!dictation.trim()}>Check my dictation</Btn>
        </Card>
      )}

      {phase === "diff" && (
        <Card className="p-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-serif text-xl font-bold">The diff</h2>
            <Chip tone={diff.acc >= 80 ? "green" : diff.acc >= 35 ? "amber" : "red"}>{diff.acc}% caught</Chip>
          </div>
          <div className="rounded-2xl bg-stone-50 border border-stone-200 p-4 mb-3 text-lg leading-relaxed">
            {diff.targetTokens.map((tok, i) => (
              <span key={i} className={tok.hit ? "text-stone-900" : "bg-rose-100 text-rose-800 rounded px-1 font-semibold"}>
                {tok.w}{" "}
              </span>
            ))}
            <button onClick={() => speak(clip.text, rate)} className="text-stone-500" aria-label="Listen to the clip again">🔊</button>
          </div>
          {diff.extras.length > 0 && (
            <div className="text-xs text-stone-500 mb-3">You also wrote: {diff.extras.map((w, i) => <span key={i} className="line-through mr-1">{w}</span>)}</div>
          )}
          {u.settings.dualSubs || showSubs ? (
            <div className="text-sm text-stone-500 mb-3 border-l-2 border-stone-200 pl-3">{clip.en}</div>
          ) : (
            <button onClick={() => setShowSubs(true)} className="text-xs text-stone-500 underline mb-3 min-h-11">show English translation</button>
          )}
          <p className="text-sm text-stone-600 mb-4">
            {diff.acc >= 80 ? "Strong. Now find out why the few misses hid from you." : "The highlighted words aren't ones you don't know — they're ones the sound stream disguised. Next: exactly why."}
          </p>
          <Btn theme={u.theme} onClick={runGaps}>Why did I miss those? →</Btn>
        </Card>
      )}

      {phase === "gaps" && (
        <Card className="p-6">
          <h2 className="font-serif text-xl font-bold mb-1">The teaching moment</h2>
          <p className="text-sm text-stone-500 mb-4">Each missed segment, decoded: what the connected speech did to the word boundaries.</p>
          {gapsBusy && <Spinner label="Analysing your gaps…" />}
          {gaps && gaps.length === 0 && (
            <div className="rounded-2xl bg-emerald-50 border border-emerald-200 p-4 mb-4 text-sm text-emerald-800">
              Perfect dictation — nothing hid from you. Time to move up a rung.
            </div>
          )}
          {gaps && gaps.map((g, i) => (
            <div key={i} className="rounded-2xl border border-stone-200 p-4 mb-3">
              <div className="flex items-center gap-2 mb-1 flex-wrap">
                <span className="font-bold">{g.segment}</span>
                <button onClick={() => speak(g.segment, 0.75)} className="min-h-11" aria-label={`Listen slowly: ${g.segment}`}>🐢</button>
                <button onClick={() => speak(g.segment, 1.0)} className="min-h-11" aria-label={`Listen at natural speed: ${g.segment}`}>🐇</button>
                <Chip tone={g.phenomenon === "liaison" ? "sky" : g.phenomenon === "elision" ? "amber" : g.phenomenon === "reduction" ? "red" : "indigo"}>{g.phenomenon}</Chip>
              </div>
              {g.sounds_like && <div className="text-sm font-mono text-indigo-600 mb-1">sounds like: {g.sounds_like}</div>}
              <div className="text-sm text-stone-600">{g.why}</div>
            </div>
          ))}
          {gaps && <Btn theme={u.theme} onClick={startShadow}>Shadow it →</Btn>}
        </Card>
      )}

      {phase === "shadow" && (() => {
        const ck = chunks[chunkIdx];
        const res = shadowResults[chunkIdx];
        return (
          <Card className="p-6">
            <div className="flex items-center justify-between mb-2">
              <h2 className="font-serif text-xl font-bold">Shadow the rhythm</h2>
              <span className="text-xs text-stone-500">{chunkIdx + 1}/{chunks.length}</span>
            </div>
            <p className="text-sm text-stone-500 mb-4">Play the chunk, then say it straight back — match the melody and the linking, not just the words.</p>
            <div className="rounded-2xl bg-stone-50 border border-stone-200 p-4 mb-3 text-lg font-medium text-center">{ck}</div>
            <div className="flex gap-2 justify-center mb-3 flex-wrap">
              <Btn variant="secondary" small onClick={() => shadowChunk(ck)}>🔊 Play chunk</Btn>
              {srSupported() ? (
                <Btn theme={u.theme} small onClick={() => recordShadow(ck)}>🎙️ Now me</Btn>
              ) : (
                <>
                  <Btn variant="success" small onClick={() => selfAssessShadow(true)}>I matched it 👍</Btn>
                  <Btn variant="secondary" small onClick={() => selfAssessShadow(false)}>Rough 😅</Btn>
                </>
              )}
            </div>
            {res && (
              <div className={`rounded-2xl p-3 mb-3 text-sm text-center ${res.sim >= 70 ? "bg-emerald-50 text-emerald-800" : res.sim >= 45 ? "bg-amber-50 text-amber-800" : "bg-rose-50 text-rose-800"}`}>
                {res.self ? (res.sim >= 70 ? "Marked as matched — trust your ear." : "Marked rough — run it once more.") : `Match: ${res.sim}%${res.heard ? ` — I heard: “${res.heard}”` : ""}`}
              </div>
            )}
            <div className="flex gap-2">
              {chunkIdx + 1 < chunks.length ? (
                <Btn theme={u.theme} onClick={() => setChunkIdx(chunkIdx + 1)} className="flex-1">Next chunk →</Btn>
              ) : (
                <Btn theme={u.theme} onClick={() => setPhase("retell")} className="flex-1">Retell it →</Btn>
              )}
              {chunkIdx > 0 && <Btn variant="ghost" small onClick={() => setChunkIdx(chunkIdx - 1)}>←</Btn>}
            </div>
          </Card>
        );
      })()}

      {phase === "retell" && (
        <Card className="p-6">
          <h2 className="font-serif text-xl font-bold mb-1">Retell it in your French</h2>
          <p className="text-sm text-stone-500 mb-3">Summarise the clip out loud (or typed) — comprehension becomes production here.</p>
          {!retellFb ? (
            <div>
              <textarea
                value={retellText} onChange={(e) => setRetellText(e.target.value)}
                placeholder="En gros, la personne dit que…" autoCapitalize="none" autoCorrect="off" spellCheck={false}
                className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white min-h-24 mb-3"
              />
              <div className="flex gap-2 flex-wrap">
                <Btn theme={u.theme} onClick={submitRetell} disabled={!retellText.trim() || retellBusy}>{retellBusy ? "Checking…" : "Get feedback"}</Btn>
                <MicBtn theme={u.theme} onText={(txt) => setRetellText((prev) => (prev ? prev + " " : "") + txt)} />
                <Btn variant="ghost" small onClick={() => speak(clip.text, rate)}>🔊 hear clip again</Btn>
              </div>
            </div>
          ) : (
            <div>
              <div className="rounded-2xl bg-stone-50 border border-stone-200 p-4 mb-3">
                <div className="flex items-center gap-2 mb-1">
                  <Chip tone={retellFb.score >= 4 ? "green" : retellFb.score >= 3 ? "amber" : "red"}>Score {retellFb.score}/5</Chip>
                </div>
                <div className="text-sm text-stone-700">{retellFb.feedback}</div>
              </div>
              {(retellFb.corrections || []).map((c, i) => (
                <div key={i} className="text-sm mb-2 rounded-xl bg-rose-50 border border-rose-100 p-3">
                  <span className="line-through text-rose-600">{c.you_said}</span> → <span className="font-semibold">{c.fix}</span>
                  <div className="text-xs text-stone-500">{c.why}</div>
                </div>
              ))}
              {retellFb.better_version && (
                <div className="rounded-2xl bg-indigo-50 border border-indigo-100 p-4 mb-3 text-sm">
                  <div className="text-xs uppercase tracking-wide text-indigo-400 mb-1">A natural retell</div>
                  {retellFb.better_version} <button onClick={() => speak(retellFb.better_version, rate)}>🔊</button>
                </div>
              )}
              <Btn theme={u.theme} onClick={() => setPhase("done")}>Wrap up →</Btn>
            </div>
          )}
        </Card>
      )}

      {phase === "done" && (
        <Card className="p-6 text-center">
          <div className="text-4xl mb-2">{diff && diff.acc >= 80 ? "🪜" : "🔁"}</div>
          <h2 className="font-serif text-xl font-bold mb-2">Clip complete</h2>
          <div className="grid grid-cols-3 gap-2 mb-3">
            <div className="rounded-2xl bg-stone-50 border border-stone-200 p-3">
              <div className="font-serif text-xl font-bold">{diff ? `${diff.acc}%` : "–"}</div>
              <div className="text-xs text-stone-500">dictation</div>
            </div>
            <div className="rounded-2xl bg-stone-50 border border-stone-200 p-3">
              <div className="font-serif text-xl font-bold">{listens}×</div>
              <div className="text-xs text-stone-500">listens</div>
            </div>
            <div className="rounded-2xl bg-stone-50 border border-stone-200 p-3">
              <div className="font-serif text-xl font-bold">{(retellFb && (retellFb.corrections || []).length) || 0}</div>
              <div className="text-xs text-stone-500">fixes banked</div>
            </div>
          </div>
          <p className="text-sm text-stone-500 mb-4">
            Narrow listening works: hearing the SAME clip 2–3× teaches more than three new clips. {listens < 3 ? "Give it one more spin before you leave —" : "You re-listened like a pro."}
          </p>
          <div className="flex gap-2 justify-center mb-4 flex-wrap">
            <Btn variant="secondary" small onClick={() => { setListens((n) => n + 1); speak(clip.text, rate); }}>🔁 Listen once more ({listens}×)</Btn>
          </div>
          {diff && diff.acc >= 80 && u.trainer.rung < 8 ? (
            <div className="mb-3 text-sm text-emerald-700 font-semibold">≥80% — you've earned rung {u.trainer.rung + 1} ({CLIP_LADDER[u.trainer.rung].subs === "none" ? "no subtitles" : CLIP_LADDER[u.trainer.rung].subs === "fr" ? "French-only subtitles" : "dual subtitles"}).</div>
          ) : diff && diff.acc < 80 ? (
            <div className="mb-3 text-sm text-stone-500">Under 80% — same rung next time. That's the system working, not a failure.</div>
          ) : null}
          <div className="flex gap-2 justify-center">
            {diff && diff.acc >= 80 && u.trainer.rung < 8 && <Btn theme={u.theme} onClick={() => finishClip(true)}>Advance & finish ✓</Btn>}
            <Btn variant={diff && diff.acc >= 80 && u.trainer.rung < 8 ? "secondary" : "primary"} theme={u.theme} onClick={() => finishClip(false)}>
              {diff && diff.acc >= 80 && u.trainer.rung < 8 ? "Stay on this rung" : "Finish ✓"}
            </Btn>
          </div>
        </Card>
      )}
    </Screen>
  );
}

/* ========================================================================== */
/*  GRAMMAR DRILLS — blocked → interleaved (Stuart)                           */
/* ========================================================================== */

function GrammarDrills({ u, setU, go }) {
  const t = themeOf(u);
  const [topicId, setTopicId] = useState(null);
  const [items, setItems] = useState(null);
  const [mode, setMode] = useState("blocked"); // flagged mode of the running drill
  const [busy, setBusy] = useState(false);
  const [idx, setIdx] = useState(0);
  const [picked, setPicked] = useState(null);
  const [typed, setTyped] = useState("");
  const [verdict, setVerdict] = useState(null);
  const [score, setScore] = useState(0);

  const topicStats = (id) => u.drills[id] || { seen: 0, correct: 0 };
  const topicMode = (id) => {
    const s = topicStats(id);
    return s.seen >= 8 && s.correct / s.seen >= 0.75 ? "interleaved" : "blocked";
  };

  const start = async (topic) => {
    setTopicId(topic.id); setBusy(true); setItems(null); setIdx(0); setScore(0); setVerdict(null); setPicked(null); setTyped("");
    const m = topicMode(topic.id);
    setMode(m);
    let drillItems = null;
    if (m === "interleaved") {
      // mix this topic with its confusable partners — the point of interleaving
      const partners = topic.confusableWith.map((id) => GRAMMAR_TOPICS.find((g) => g.id === id)).filter(Boolean);
      const focus = `INTERLEAVED drill mixing: ${[topic, ...partners].map((x) => x.name).join(" AND ")}. Shuffle the structures so the learner must decide which applies — do not group by structure.`;
      const res = await aiJSON(SYS.exercise(focus, "cloze", u.level), "Generate 6 items. For each, options field: 4 plausible choices as {\"options\": [..]} within the item, answer among them.", null, 1000);
      if (res && Array.isArray(res.items) && res.items.length >= 4) {
        drillItems = res.items.map((it) => ({ type: it.options && it.options.length ? "mcq" : "type", ...it }));
      } else {
        const poolAll = [topic, ...partners].flatMap((x) => topicPool(x));
        drillItems = shuffle(poolAll).slice(0, 6);
      }
    } else {
      const res = await aiJSON(SYS.exercise(`BLOCKED practice of one structure only: ${topic.name}. ${topic.hint}`, "cloze", u.level), "Generate 4 items with {\"options\": [..]} per item where multiple-choice fits.", null, 1000);
      if (res && Array.isArray(res.items) && res.items.length >= 3) {
        drillItems = res.items.map((it) => ({ type: it.options && it.options.length ? "mcq" : "type", ...it }));
      } else {
        drillItems = shuffle(topicPool(topic)).slice(0, 6);
      }
    }
    setItems(drillItems.map((it) => (it.options && it.options.length ? { ...it, options: shuffle(it.options) } : it)));
    setBusy(false);
  };

  const record = (correct) => {
    setU((prev) => {
      const s = prev.drills[topicId] || { seen: 0, correct: 0 };
      return { ...prev, drills: { ...prev.drills, [topicId]: { seen: s.seen + 1, correct: s.correct + (correct ? 1 : 0) } } };
    });
    if (correct) setScore((x) => x + 1);
  };

  const answerMcq = (opt) => {
    if (picked !== null) return;
    setPicked(opt);
    const item = items[idx];
    const correct = norm(opt) === norm(item.answer);
    setVerdict(correct ? "exact" : "wrong");
    record(correct);
  };

  const answerTyped = () => {
    const item = items[idx];
    const v = checkAnswer(item.answer, typed);
    setVerdict(v);
    record(v === "exact" || v === "close");
    speak(item.answer, u.settings.ttsRate);
  };

  const next = () => {
    setPicked(null); setTyped(""); setVerdict(null);
    if (idx + 1 < items.length) setIdx(idx + 1);
    else {
      setU((prev) => ({ ...prev, minutes: prev.minutes + 4 }));
      setItems({ finished: true, total: items.length });
    }
  };

  if (!topicId) {
    return (
      <Screen>
        <h1 className="font-serif text-2xl font-bold mb-1">Grammar drills</h1>
        <p className="text-sm text-stone-500 mb-4">New structures start <b>blocked</b> (one at a time). Hit 75% over 8+ items and they switch to <b>interleaved</b> — mixed with their confusables, the way real French arrives.</p>
        {GRAMMAR_TOPICS.map((g) => {
          const s = topicStats(g.id);
          const m = topicMode(g.id);
          const acc = s.seen ? Math.round((s.correct / s.seen) * 100) : null;
          return (
            <button key={g.id} onClick={() => start(g)} className="w-full text-left mb-3 rounded-3xl border border-stone-200 bg-white shadow-sm p-4 hover:shadow-md min-h-20">
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="font-semibold">{g.name}</span>
                <Chip tone={m === "interleaved" ? "indigo" : "amber"}>{m}</Chip>
              </div>
              <div className="text-xs text-stone-500">{g.hint}</div>
              {m === "blocked" ? (
                <div className="mt-2">
                  <div className="h-1.5 rounded-full bg-stone-100 overflow-hidden">
                    <div className="h-full rounded-full bg-amber-400 transition-all" style={{ width: `${Math.min(100, (s.seen / 8) * 100)}%` }} />
                  </div>
                  <div className="text-xs text-stone-500 mt-1">
                    {s.seen}/8 items{s.seen ? ` · ${acc}% correct` : ""} — 75% over 8+ unlocks interleaving
                  </div>
                </div>
              ) : (
                acc !== null && <div className="text-xs text-stone-500 mt-1">{acc}% over {s.seen} items — mixing with confusables</div>
              )}
            </button>
          );
        })}
      </Screen>
    );
  }

  if (busy || !items) return <Screen><Spinner label="Setting the drill…" /></Screen>;

  if (items.finished) {
    const topic = GRAMMAR_TOPICS.find((g) => g.id === topicId);
    const nowMode = topicMode(topicId);
    return (
      <Screen>
        <Card className="p-6 text-center">
          <div className="text-4xl mb-2">⚔️</div>
          <h1 className="font-serif text-2xl font-bold mb-1">{score}/{items.total}</h1>
          <p className="text-sm text-stone-600 mb-2">{topic.name} — {mode} mode</p>
          {mode === "blocked" && nowMode === "interleaved" && (
            <div className="rounded-2xl bg-indigo-50 border border-indigo-200 p-3 mb-3 text-sm text-indigo-800">🔓 Unlocked: this structure now drills <b>interleaved</b> with {topic.confusableWith.map((id) => GRAMMAR_TOPICS.find((g) => g.id === id).name).join(", ")}.</div>
          )}
          <div className="flex gap-2 justify-center">
            <Btn theme={u.theme} onClick={() => start(topic)}>Again</Btn>
            <Btn variant="secondary" onClick={() => setTopicId(null)}>Topics</Btn>
            <Btn variant="ghost" onClick={() => go("home")}>Home</Btn>
          </div>
        </Card>
      </Screen>
    );
  }

  const item = items[idx];
  const isMcq = item.type === "mcq" || (item.options && item.options.length);
  return (
    <Screen>
      <div className="flex items-center justify-between mb-3">
        <Chip tone={mode === "interleaved" ? "indigo" : "amber"}>{mode} · {idx + 1}/{items.length}</Chip>
        <Btn variant="ghost" small onClick={() => setTopicId(null)}>✕</Btn>
      </div>
      <Card className="p-6">
        <div className="font-serif text-lg font-bold mb-4">{item.prompt}</div>
        {isMcq ? (
          <div className="flex flex-col gap-2 mb-3">
            {item.options.map((opt) => {
              let cls = "bg-stone-50 border-stone-200 hover:bg-stone-100";
              if (picked !== null) {
                if (norm(opt) === norm(item.answer)) cls = "bg-emerald-50 border-emerald-400";
                else if (opt === picked) cls = "bg-rose-50 border-rose-300";
                else cls = "bg-stone-50 border-stone-200 opacity-50";
              }
              return (
                <button key={opt} onClick={() => answerMcq(opt)} className={`text-left rounded-2xl border px-4 py-3 min-h-12 ${cls}`}>{opt}</button>
              );
            })}
          </div>
        ) : !verdict ? (
          <div className="mb-3">
            <input
              value={typed} onChange={(e) => setTyped(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && typed.trim() && answerTyped()}
              placeholder="Type the French…" autoCapitalize="none" autoCorrect="off" spellCheck={false}
              className="w-full rounded-2xl border border-stone-300 px-4 py-3 bg-white text-lg"
            />
            <div className="flex gap-2 mt-2">
              <Btn theme={u.theme} small onClick={answerTyped} disabled={!typed.trim()}>Check</Btn>
              <MicBtn theme={u.theme} onText={setTyped} />
            </div>
          </div>
        ) : null}
        {verdict && (
          <div>
            <div className={`rounded-2xl p-3 mb-3 text-sm ${verdict === "exact" || verdict === "close" ? "bg-emerald-50 border border-emerald-200 text-emerald-900" : "bg-rose-50 border border-rose-200 text-rose-900"}`}>
              <div className="font-semibold">{item.answer} <button onClick={() => speak(item.answer, u.settings.ttsRate)}>🔊</button></div>
              {item.explanation && <div className="mt-1 text-stone-600">💡 {item.explanation}</div>}
            </div>
            <Btn theme={u.theme} onClick={next}>{idx + 1 < items.length ? "Next" : "Finish"}</Btn>
          </div>
        )}
      </Card>
    </Screen>
  );
}

/* ========================================================================== */
/*  APP ROOT — profiles, persistence, routing                                 */
/* ========================================================================== */

function hydrateProfile(id, meta, srs, progress) {
  const base = defaultProfile(id, (meta && meta.name) || id);
  return {
    ...base,
    ...(meta || {}),
    settings: { ...base.settings, ...((meta && meta.settings) || {}) },
    deck: Array.isArray(srs) ? srs : base.deck,
    ...(progress
      ? {
          streak: { ...base.streak, ...(progress.streak || {}) },
          minutes: progress.minutes ?? 0,
          lessons: progress.lessons || {},
          phonicsDone: progress.phonicsDone || {},
          csDone: progress.csDone || {},
          trainer: { ...base.trainer, ...(progress.trainer || {}) },
          drills: progress.drills || {},
          stats: { ...base.stats, ...(progress.stats || {}) },
        }
      : {}),
  };
}

async function loadProfile(id) {
  const [meta, srs, progress] = await Promise.all([
    store.get(`${id}:profile`),
    store.get(`${id}:srs`),
    store.get(`${id}:progress`),
  ]);
  return hydrateProfile(id, meta, srs, progress);
}

async function saveProfile(u) {
  const { deck, streak, minutes, lessons, phonicsDone, csDone, trainer, drills, stats, ...meta } = u;
  await Promise.all([
    store.set(`${u.id}:profile`, meta),
    store.set(`${u.id}:srs`, deck),
    store.set(`${u.id}:progress`, { streak, minutes, lessons, phonicsDone, csDone, trainer, drills, stats }),
  ]);
}

export default function DeuxApp() {
  const [users, setUsers] = useState({});
  const [order, setOrder] = useState([]);
  const [currentId, setCurrentId] = useState(null);
  const [view, setView] = useState("picker");
  const [params, setParams] = useState({});
  const [ready, setReady] = useState(false);
  const [aiCfg, setAiCfg] = useState({ key: "", url: "" });
  const saveTimer = useRef(null);

  const onSaveAi = useCallback(async (cfg) => {
    const clean = { key: (cfg.key || "").trim(), url: (cfg.url || "").trim() };
    setAiCfg(clean);
    setAiConfig(clean);
    try { await store.set("deux:ai", clean); } catch (e) { /* memory copy holds */ }
  }, []);

  /* boot: load or seed the two household profiles */
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        let index = await store.get("deux:profiles");
        if (!Array.isArray(index) || index.length === 0) {
          const seeded = seedProfiles();
          index = seeded.map((p) => p.id);
          await store.set("deux:profiles", index);
          for (const p of seeded) await saveProfile(p);
        }
        const loaded = {};
        for (const id of index) loaded[id] = await loadProfile(id);
        const ai = await store.get("deux:ai");
        if (!alive) return;
        if (ai && (ai.key || ai.url)) { setAiConfig(ai); setAiCfg({ key: ai.key || "", url: ai.url || "" }); }
        setUsers(loaded);
        setOrder(index);
      } catch (e) {
        /* worst case: seed in memory so the app still runs */
        const seeded = seedProfiles();
        const loaded = {};
        seeded.forEach((p) => { loaded[p.id] = p; });
        if (!alive) return;
        setUsers(loaded);
        setOrder(seeded.map((p) => p.id));
      }
      setReady(true);
    })();
    return () => { alive = false; };
  }, []);

  const u = currentId ? users[currentId] : null;

  /* debounced autosave of the active profile */
  useEffect(() => {
    if (!u) return;
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => { saveProfile(u); }, 500);
    return () => saveTimer.current && clearTimeout(saveTimer.current);
  }, [u]);

  const setU = useCallback(
    (fn) => setUsers((prev) => (currentId ? { ...prev, [currentId]: fn(prev[currentId]) } : prev)),
    [currentId]
  );

  const go = useCallback((v, p) => { stopSpeaking(); setView(v); setParams(p || {}); }, []);

  const pick = (id) => {
    stopSpeaking();
    setCurrentId(id);
    setView(users[id] && users[id].placed ? "home" : "placement");
    setParams({});
  };

  const addProfile = async (name) => {
    const id = norm(name).replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || uid();
    if (users[id]) { pick(id); return; }
    const p = { ...defaultProfile(id, name), emoji: "🌱", theme: Math.random() < 0.5 ? "amber" : "indigo" };
    const nextOrder = [...order, id];
    setUsers((prev) => ({ ...prev, [id]: p }));
    setOrder(nextOrder);
    try { await store.set("deux:profiles", nextOrder); await saveProfile(p); } catch (e) { /* in-memory fallback holds */ }
  };

  const onPlacementDone = (level, track) => {
    setU((prev) => ({ ...prev, level, track, placed: true }));
    go("home");
  };

  const onSwitch = () => { stopSpeaking(); setCurrentId(null); setView("picker"); };

  const partner = useMemo(() => {
    if (!u) return null;
    const otherId = order.find((id) => id !== u.id && users[id] && users[id].placed);
    const p = otherId ? users[otherId] : null;
    return p && p.settings.shareProgress ? p : null;
  }, [u, users, order]);

  if (!ready) {
    return (
      <div className="min-h-screen bg-stone-100 flex items-center justify-center">
        <DeuxStyles />
        <div className="text-center">
          <div className="font-serif text-5xl font-bold text-stone-900 mb-3">Deux<span className="text-rose-500">.</span></div>
          <div className="w-6 h-6 mx-auto rounded-full border-2 border-stone-300 border-t-rose-500 animate-spin" />
        </div>
      </div>
    );
  }

  if (!u || view === "picker") {
    return (
      <>
        <DeuxStyles />
        <ProfilePicker profiles={order.map((id) => users[id]).filter(Boolean)} onPick={pick} onAdd={addProfile} />
      </>
    );
  }

  const body = (() => {
    switch (view) {
      case "placement": return <Placement u={u} onDone={onPlacementDone} />;
      case "home": return <Home u={u} setU={setU} go={go} />;
      case "review": return <SRSReview key={params.quickWin ? "qw" : "full"} u={u} setU={setU} go={go} quickWin={!!params.quickWin} />;
      case "dashboard": return <Dashboard u={u} partner={partner} go={go} />;
      case "settings": return <Settings u={u} setU={setU} go={go} onRerunPlacement={() => go("placement")} onSwitch={onSwitch} aiCfg={aiCfg} onSaveAi={onSaveAi} />;
      case "units": return <UnitList u={u} go={go} />;
      case "unit": return <UnitView key={params.unitId} u={u} setU={setU} go={go} unitId={params.unitId} />;
      case "phonics": return <Phonics u={u} setU={setU} go={go} lessonId={params.lessonId} />;
      case "cslessons": return <CSLessons u={u} setU={setU} />;
      case "library": return <Library u={u} setU={setU} go={go} />;
      case "conversation": return <Conversation u={u} setU={setU} go={go} />;
      case "exercise": return <ExerciseGen u={u} setU={setU} go={go} />;
      case "trainer": return <ListeningTrainer key={u.trainer.clipsDone} u={u} setU={setU} go={go} />;
      case "drills": return <GrammarDrills u={u} setU={setU} go={go} />;
      default: return <Home u={u} setU={setU} go={go} />;
    }
  })();

  return (
    <div className="min-h-screen bg-stone-100 text-stone-900" style={{ WebkitTapHighlightColor: "transparent" }}>
      <DeuxStyles />
      {view !== "placement" && <TopBar u={u} go={go} view={view} onSwitch={onSwitch} />}
      <div key={view + (params.unitId || "")} className="view-fade">{body}</div>
    </div>
  );
}

/* Self-mount when opened via the bundled index.html dev harness. In the
   production environment the default export is rendered by the host. */
if (typeof document !== "undefined") {
  const el = document.getElementById("deux-root");
  if (el) {
    import("react-dom/client")
      .then(({ createRoot }) => createRoot(el).render(<DeuxApp />))
      .catch(() => {});
  }
}
