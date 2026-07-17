/*
 * Deux — personal AI proxy (Cloudflare Worker)
 * ---------------------------------------------------------------------------
 * Keeps your Anthropic API key OFF your devices and OUT of the website.
 * The Deux app POSTs Anthropic Messages-API payloads here; this worker adds
 * your key server-side and forwards the request.
 *
 * Deploy (free tier is plenty — takes ~5 minutes):
 *   1. Sign up / log in at https://dash.cloudflare.com
 *   2. Workers & Pages -> Create -> Worker -> paste this whole file -> Deploy
 *   3. Worker -> Settings -> Variables and Secrets ->
 *        Add secret:  ANTHROPIC_API_KEY = sk-ant-...   (from console.anthropic.com)
 *        Optional:    ALLOWED_ORIGIN    = https://<your-github-username>.github.io
 *   4. Copy the worker URL (https://<name>.<account>.workers.dev) and paste it
 *      into Deux -> Settings -> "AI content engine" -> proxy URL.
 *
 * Security notes:
 *   - Set ALLOWED_ORIGIN so only your own site can use your key. Without it,
 *     anyone who discovers the URL could spend your credit.
 *   - The worker forwards ONLY to the Anthropic Messages endpoint and only
 *     accepts the fields Deux sends (model, max_tokens, system, messages).
 */

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const allowed = env.ALLOWED_ORIGIN || "*";
    const originOk = allowed === "*" || origin === allowed;

    const corsHeaders = {
      "Access-Control-Allow-Origin": originOk ? (allowed === "*" ? "*" : origin) : "null",
      "Access-Control-Allow-Methods": "POST, OPTIONS",
      "Access-Control-Allow-Headers": "content-type, anthropic-version",
      "Access-Control-Max-Age": "86400",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders });
    }
    if (request.method !== "POST") {
      return new Response(JSON.stringify({ error: "POST only" }), { status: 405, headers: { ...corsHeaders, "content-type": "application/json" } });
    }
    if (!originOk) {
      return new Response(JSON.stringify({ error: "origin not allowed" }), { status: 403, headers: { ...corsHeaders, "content-type": "application/json" } });
    }
    if (!env.ANTHROPIC_API_KEY) {
      return new Response(JSON.stringify({ error: "ANTHROPIC_API_KEY secret not set on the worker" }), { status: 500, headers: { ...corsHeaders, "content-type": "application/json" } });
    }

    let body;
    try { body = await request.json(); } catch (e) {
      return new Response(JSON.stringify({ error: "invalid JSON" }), { status: 400, headers: { ...corsHeaders, "content-type": "application/json" } });
    }

    // Pass through only the fields Deux uses; cap max_tokens defensively.
    const payload = {
      model: typeof body.model === "string" ? body.model : "claude-sonnet-4-6",
      max_tokens: Math.min(Number(body.max_tokens) || 1000, 2000),
      system: typeof body.system === "string" ? body.system : undefined,
      messages: Array.isArray(body.messages) ? body.messages : [],
    };

    const upstream = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": env.ANTHROPIC_API_KEY,
      },
      body: JSON.stringify(payload),
    });

    const text = await upstream.text();
    return new Response(text, { status: upstream.status, headers: { ...corsHeaders, "content-type": "application/json" } });
  },
};
