# RUNBOOK.md — Deploying and checking the site

The site is static: no database, no secrets, no environment variables. A
deploy can't break a schema, but it can still break in ways CI doesn't
see, so every deploy follows the same three steps. They are adapted from
transformer-explainer's RUNBOOK §7.

## 1. Deploy from a clean export

The Vercel project (`llm-inference-explained`) is not on Vercel's Git
integration. Deploy with the logged-in Vercel CLI from a clean export of
`HEAD`, so nothing untracked (caches, `node_modules`, local files) is
uploaded:

```bash
rm -rf /tmp/lie-deploy && mkdir /tmp/lie-deploy
git archive HEAD | tar -x -C /tmp/lie-deploy
cp -r .vercel /tmp/lie-deploy/
(cd /tmp/lie-deploy && vercel deploy --yes)          # preview
(cd /tmp/lie-deploy && vercel deploy --prod --yes)   # production
vercel ls llm-inference-explained | head             # newest must be ● Ready
```

Test a preview first. Previews are protected by Vercel Authentication; to
smoke-check one, create a protection-bypass token in the project settings
and pass it as `VERCEL_BYPASS` (never commit or print it).

## 2. Smoke-check

```bash
pnpm smoke https://llm-inference-explained.vercel.app
# a protected preview:
VERCEL_BYPASS=… pnpm smoke https://<preview-url>
```

It fetches every page and fails on any non-200 (redirects included), on a
chapter whose server-rendered HTML lacks its title or its interactive (or,
for the client-only simulator chapters 11, 13 and 14, its marked
placeholder), and on any KaTeX error. It also fetches the nine recorded
workloads the live simulator runs on and checks each has 800 rows and the
vendored engine's commit: 26 checks in all. Then open two or three chapters
in a browser and use their interactives, including a preset on
`/learn/13-live-simulator` (every cell should show ✓): the widgets run
client-side, which the smoke check can't see.

## 3. Read the logs

```bash
vercel logs --environment production --since 15m --no-branch --expand
```

A static site should log almost nothing. On the Hobby plan the CLI only
reaches back about an hour; the dashboard's Logs view keeps more.

## Why e2e runs under `--no-experimental-require-module`

Plain Node 20.19+ / 22.12+ can `require()` an ES module; Vercel's function
loader can't. transformer-explainer shipped a comment renderer that passed
every local and CI test and returned 500 on Vercel for that reason
(2026-10-04). This site has no server functions today, but CI runs the e2e
server under the flag anyway, so the class of bug can't arrive unnoticed if
one is added.
