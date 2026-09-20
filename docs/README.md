# Scene stylization user study

A static site (`index.html` + `app.js` + `style.css` + `manifest.json` + `images/`)
that runs a 2AFC preference study: 30 trials, each showing the original scene,
the style (and color, for second-run trials) reference, and two stylization
results side by side in randomized left/right order. Visitors pick left,
right, or "about the same".

## 1. Create the Google Form (collects responses)

This is the only manual step. Create a new Google Form with these exact
settings:

- Turn **off** "Collect email addresses" (Settings tab) — keep it anonymous.
- Add **11 short-answer questions**, none marked required, with these exact
  titles (case-sensitive, used only so you can read the resulting sheet —
  the app doesn't parse the titles):

  1. `participant_id`
  2. `trial_id`
  3. `dataset`
  4. `scene`
  5. `kind`
  6. `style`
  7. `color`
  8. `choice`
  9. `preferred`
  10. `left_is_ours`
  11. `trial_order`

- Link the form to a Google Sheet (Responses tab → the green sheet icon) so
  submissions land as rows automatically.
- Get the form's **live/shareable link** (Send → link icon), and send it back
  so the app can be wired up to submit to it — the exact submission endpoint
  and per-question IDs get filled into `app.js` (`FORM_ACTION` and `ENTRY` at
  the top of the file) once that link is available.

Each trial submits one row immediately when answered, so even a visitor who
closes the tab partway through leaves their earlier answers recorded.

`preferred` is already computed client-side as `ours` / `baseline` / `tie`,
so you don't need to cross-reference `left_is_ours` for basic analysis —
it's kept only as an audit trail.

## 2. Host it

Any static host works. Simplest with the existing GitHub repo:

1. Push this `docs/` folder to the `main` branch.
2. GitHub repo → Settings → Pages → Source: "Deploy from a branch" → Branch:
   `main`, folder: `/docs` → Save.
3. The study goes live at `https://maiolsabater.github.io/TFM/`.

## 3. Before sharing the link

- Open the page yourself once, answer a couple of trials, and check the
  linked Google Sheet gets rows.
- If `FORM_ACTION` in `app.js` is still empty, answers are collected in the
  browser and downloadable at the end, but nothing is sent anywhere — don't
  circulate the link until that's filled in.

## After any future edit to app.js, style.css, manifest.json or images/

GitHub Pages caches these for 10 minutes, and browsers can hold on to them
longer than that. Bump the version so returning visitors can't get served a
stale mix of old/new files:

1. Bump `SITE_VERSION` in `app.js`.
2. Update the matching `?v=` on the `style.css` and `app.js` `<script>`/
   `<link>` tags in `index.html` to the same number.

Then test in a private/incognito window (not just a reload) before
circulating.

## Notes on the data

- 30 trials sampled across 6 scenes (`flower`, `horns`, `garden`, `family`,
  `m60`, `truck`) from `Ours/StylizedGS` vs. the baseline `StylizedGS`
  outputs, 22 single-style trials + 8 style+color trials.
- Left/right placement of "ours" vs. baseline is randomized per trial
  (baked into `manifest.json`'s `left_is_ours`), and trial order is
  reshuffled per visitor in `app.js`.
- Images are re-encoded JPEGs (max width 500px for reference thumbnails,
  1000px for results) to keep the site light (~7MB total for 80 images).
