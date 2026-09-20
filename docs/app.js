"use strict";

/**
 * Google Form submission target. Fill these in once the form exists (see
 * README.md in this folder for the exact fields to create), then this file
 * needs no other changes. Until FORM_ACTION is set, answers are still
 * collected and downloadable, just not sent anywhere.
 */
const FORM_ACTION = "https://docs.google.com/forms/d/e/1FAIpQLSfSJAo7jmyoko4UVvAcWNJfQLy1YgLfiCvwEfHD3Y0lq8etTQ/formResponse";
const ENTRY = {
  participant_id: "entry.1441168395",
  trial_id: "entry.2026560637",
  dataset: "entry.1387060064",
  scene: "entry.1563710009",
  kind: "entry.1598734123",
  style: "entry.773870065",
  color: "entry.382881730",
  choice: "entry.578103266",
  preferred: "entry.1969638757",
  left_is_ours: "entry.2094191848",
  trial_order: "entry.1399333869",
};

const IMG_DIR = "images/";

// Bump this on every push that changes manifest.json or images/, so a
// visitor's browser (and GitHub Pages' CDN) can't silently keep serving a
// stale manifest that points at filenames which no longer exist.
const SITE_VERSION = "3";
function withVersion(path) {
  return path + (path.includes("?") ? "&" : "?") + "v=" + SITE_VERSION;
}

function getParticipantId() {
  try {
    let id = localStorage.getItem("study_participant_id");
    if (!id) {
      id = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random());
      localStorage.setItem("study_participant_id", id);
    }
    return id;
  } catch (e) {
    return "anon-" + Math.random().toString(36).slice(2);
  }
}

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function submitToForm(row) {
  if (!FORM_ACTION) {
    console.warn("[study] FORM_ACTION not configured yet; answer kept locally only.", row);
    return;
  }
  const body = new URLSearchParams();
  for (const key of Object.keys(ENTRY)) {
    const entryId = ENTRY[key];
    if (entryId) body.append(entryId, String(row[key] ?? ""));
  }
  fetch(FORM_ACTION, {
    method: "POST",
    mode: "no-cors",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  }).catch(() => {
    // no-cors gives us no visibility into success anyway; local backup covers this.
  });
}

function downloadJSON(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function main() {
  const participantId = getParticipantId();
  const answers = [];

  const screenIntro = document.getElementById("screen-intro");
  const screenTrial = document.getElementById("screen-trial");
  const screenDone = document.getElementById("screen-done");
  const btnStart = document.getElementById("btn-start");
  const btnDownload = document.getElementById("btn-download");
  const doneCount = document.getElementById("done-count");

  const trialIndexEl = document.getElementById("trial-index");
  const trialTotalEl = document.getElementById("trial-total");
  const progressFill = document.getElementById("progress-fill");

  const imgContent = document.getElementById("img-content");
  const imgStyle = document.getElementById("img-style");
  const imgColor = document.getElementById("img-color");
  const refColorCard = document.getElementById("ref-color-card");
  const imgLeft = document.getElementById("img-left");
  const imgRight = document.getElementById("img-right");

  const cardLeft = document.getElementById("card-left");
  const cardRight = document.getElementById("card-right");
  const btnPickLeft = document.getElementById("btn-pick-left");
  const btnPickRight = document.getElementById("btn-pick-right");

  let manifest = [];
  try {
    const res = await fetch(withVersion("manifest.json"), { cache: "no-store" });
    manifest = await res.json();
  } catch (e) {
    document.body.innerHTML = "<p style='padding:40px;font-family:sans-serif'>Could not load the study data (manifest.json). If you're opening this file directly, serve it over http:// instead of file://.</p>";
    return;
  }

  const trials = shuffle(manifest);
  trialTotalEl.textContent = String(trials.length);

  let current = 0;

  function renderTrial() {
    const t = trials[current];
    trialIndexEl.textContent = String(current + 1);
    progressFill.style.width = ((current) / trials.length * 100) + "%";

    imgContent.src = withVersion(IMG_DIR + t.content);
    imgStyle.src = withVersion(IMG_DIR + t.style);
    if (t.color) {
      imgColor.src = withVersion(IMG_DIR + t.color);
      refColorCard.hidden = false;
    } else {
      refColorCard.hidden = true;
    }
    imgLeft.src = withVersion(IMG_DIR + t.left);
    imgRight.src = withVersion(IMG_DIR + t.right);

    setChoiceEnabled(true);
  }

  function setChoiceEnabled(enabled) {
    [cardLeft, cardRight, btnPickLeft, btnPickRight].forEach((el) => {
      el.disabled = !enabled;
      el.style.opacity = enabled ? "1" : "0.5";
      el.style.pointerEvents = enabled ? "auto" : "none";
    });
  }

  function recordChoice(choice) {
    const t = trials[current];
    const pickedIsOurs = (choice === "left") === t.left_is_ours;
    const preferred = pickedIsOurs ? "ours" : "baseline";

    const row = {
      participant_id: participantId,
      trial_id: t.id,
      dataset: t.dataset,
      scene: t.scene,
      kind: t.kind,
      style: t.style.replace(/^style_|\.jpg$/g, ""),
      color: t.color ? t.color.replace(/^style_|\.jpg$/g, "") : "",
      choice,
      preferred,
      left_is_ours: t.left_is_ours,
      trial_order: current,
    };
    answers.push(row);
    submitToForm(row);

    setChoiceEnabled(false);
    current += 1;
    if (current >= trials.length) {
      setTimeout(finish, 150);
    } else {
      setTimeout(renderTrial, 150);
    }
  }

  function finish() {
    progressFill.style.width = "100%";
    screenTrial.hidden = true;
    screenDone.hidden = false;
    doneCount.textContent = String(answers.length);
  }

  cardLeft.addEventListener("click", () => recordChoice("left"));
  cardRight.addEventListener("click", () => recordChoice("right"));
  btnPickLeft.addEventListener("click", () => recordChoice("left"));
  btnPickRight.addEventListener("click", () => recordChoice("right"));

  btnStart.addEventListener("click", () => {
    screenIntro.hidden = true;
    screenTrial.hidden = false;
    renderTrial();
  });

  btnDownload.addEventListener("click", () => {
    downloadJSON(`study-answers-${participantId.slice(0, 8)}.json`, answers);
  });
}

main();
