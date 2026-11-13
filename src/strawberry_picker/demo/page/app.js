// StrawberryPicker inspection demo: renders one tray photo as a printed market report.
// Talks to src/strawberry_picker/demo/server.py: /api/status, /api/samples, /api/jobs/<id>,
// /api/image/<id>, POST /api/inspect (raw image bytes).
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const svgNS = "http://www.w3.org/2000/svg";

  const el = {
    sheet: $("sheet"), stage: $("stage"), photo: $("photo"), marks: $("marks"), contact: $("contact"),
    intake: $("intake"), file: $("file"), choose: $("choose"), warmHint: $("warm-hint"),
    plateCaption: $("plate-caption"), toggleWrap: $("model-toggle-wrap"), toggle: $("model-toggle"),
    wire: $("wire"), wireText: $("wire-text"), dateline: $("dateline"),
    form: $("form"), samples: $("samples"), samplesEmpty: $("samples-empty"),
    filed: $("filed"), verdict: $("verdict"), stamp: $("stamp"), condition: $("condition"),
    tally: $("tally"), mixbar: $("mixbar"), lots: $("lots"), lotRows: $("lot-rows"),
    newFile: $("new-file"), printing: $("printing"), printingText: $("printing-text"),
    errata: $("errata"), report: $("report"), fileName: $("file-name"), trays: $("trays"),
    mixes: $("mixes"), gradebar: $("gradebar"), mixCount: $("mix-count"), gradeCount: $("grade-count"),
    slip: $("slip"), slipClose: $("slip-close"), slipCrop: $("slip-crop"), slipCode: $("slip-code"),
    slipRipeness: $("slip-ripeness"), slipDefects: $("slip-defects"), slipGrade: $("slip-grade"),
    slipConfidence: $("slip-confidence"), slipProbs: $("slip-probs"),
  };

  const MATURITY = ["unripe", "ripe", "overripe"];
  const GRADE_ORDER = { reject: 0, standard: 1, premium: 2 };
  const FRAME = 18; // photo padding + border, px
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const narrow = window.matchMedia("(max-width: 960px)");

  const state = {
    model: "warming",
    reportNo: 0,
    sampleCount: 0,
    samples: [],
    activeSample: null,
    job: null,        // the job payload on screen
    berries: [],      // berries with lot codes, in table order
    img: null,        // the <img> inside the plate
    on: null,         // berry id under the pointer or keyboard focus
    pinned: null,     // berry id whose slip stays open
    polling: null,
    timer: null,
    printDone: null,
  };

  // ------------------------------------------------------------- helpers

  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(path, options);
    } catch {
      const error = new Error("The demo server is not answering. Is scripts/demo.py still running?");
      error.offline = true;
      throw error;
    }
    let body = null;
    try { body = await response.json(); } catch { body = null; }
    if (!response.ok) {
      const error = new Error((body && body.error) || `Request failed (${response.status})`);
      error.status = response.status;
      throw error;
    }
    return body;
  }

  const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;
  const pad = (n, width = 2) => String(n).padStart(width, "0");
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const defectLabel = (d) => d.replace(/_/g, " ");
  const isRejectDefect = (d) => d === "mold" || d === "rot";

  function stamped(date = new Date()) {
    const day = date.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
    return `${day} · ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  function node(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const child of children) if (child !== null && child !== undefined) n.append(child);
    return n;
  }

  function svg(tag, attrs = {}) {
    const n = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    return n;
  }

  function showErrata(title, message) {
    el.errata.replaceChildren(node("b", { text: title }), document.createTextNode(message));
    el.errata.hidden = false;
  }
  const clearErrata = () => { el.errata.hidden = true; };

  // ------------------------------------------------------------- model status

  function setModel(next, detail) {
    state.model = next;
    el.wire.dataset.state = next;
    el.wireText.textContent = {
      warming: "Model warming up",
      ready: "Model ready",
      error: "Model unavailable",
      offline: "Server offline",
    }[next] || "Model unavailable";
    el.wire.title = detail || "";
    const live = next === "ready";
    el.choose.disabled = !live;
    el.file.disabled = !live;
    el.warmHint.textContent = live
      ? "A live photo takes about half a minute. Sample trays open right away."
      : next === "warming"
        ? "Live photos open once the model has warmed up. Sample trays open right away."
        : "Live photos are unavailable right now. Sample trays still open.";
    el.intake.querySelector(".intake-sub").textContent = live
      ? "or click to choose a file · JPEG, PNG or iPhone HEIC"
      : "Live photos open once the model has warmed up";
  }

  async function pollStatus() {
    try {
      const status = await api("/api/status");
      const detail = [
        status.run && `maturity run ${status.run}`,
        status.quality_run && `defect run ${status.quality_run}`,
      ].filter(Boolean).join(" · ");
      setModel(status.state, status.error || detail);
      if (status.state === "warming") setTimeout(pollStatus, 2000);
    } catch (error) {
      setModel(error.offline ? "offline" : "error", error.message);
      setTimeout(pollStatus, 4000);
    }
  }

  // ------------------------------------------------------------- samples

  const VERDICT_WORD = { premium: "premium", standard: "sort", reject: "reject", empty: "no berries" };

  async function loadSamples() {
    let list = [];
    try {
      list = (await api("/api/samples")).samples || [];
    } catch {
      list = [];
    }
    state.sampleCount = list.length;
    state.samples = list;
    el.trays.replaceChildren(
      node("span", { class: "trays-label", text: "Trays" }),
      ...list.map((sample, i) => node("button", {
        class: "tray", type: "button", "data-id": sample.id,
        "aria-label": `Open sample tray ${pad(i + 1)}, ${sample.title}`,
        onclick: () => openSample(sample),
      }, document.createTextNode(pad(i + 1)))));
    el.trays.hidden = list.length === 0;
    el.samplesEmpty.hidden = list.length > 0;
    el.samples.replaceChildren();
    list.forEach((sample, i) => {
      const meta = node("span", { class: "sample-meta", text: "·" });
      const button = node("button", {
        class: "sample", type: "button", onclick: () => openSample(sample),
        "aria-label": `Open sample tray ${sample.title}`,
      },
      node("span", { class: "sample-photo" },
        node("img", { src: `/api/image/${encodeURIComponent(sample.image)}`, alt: "", loading: "lazy" })),
      node("span", { class: "sample-line" },
        node("span", { class: "sample-no", text: `Tray ${pad(i + 1)}` }),
        node("span", { class: "sample-title", text: sample.title })),
      meta);
      el.samples.append(node("li", {}, button));
      api(`/api/jobs/${encodeURIComponent(sample.id)}`).then((job) => {
        const summary = job.summary || { count: 0, verdict: "empty" };
        meta.replaceChildren(
          document.createTextNode(`${plural(summary.count, "berry", "berries")} · `),
          node("span", { class: `is-${summary.verdict}`, text: VERDICT_WORD[summary.verdict] }),
        );
      }).catch(() => { meta.textContent = ""; });
    });
    if (!state.job && el.printing.hidden) showEmptyPlate();
  }

  function showEmptyPlate() {
    const hasSamples = state.sampleCount > 0;
    el.photo.hidden = true;
    el.contact.hidden = !hasSamples;
    el.intake.hidden = hasSamples;
  }

  async function openSample(sample) {
    clearErrata();
    state.activeSample = sample.id;
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(sample.id)}`);
      await showReport(job, sample.title);
    } catch (error) {
      showErrata("Could not open", ` ${error.message}`);
    }
  }

  // ------------------------------------------------------------- live photo

  async function fileLivePhoto(file) {
    if (!file) return;
    clearErrata();
    if (state.model !== "ready") {
      showErrata("Not yet", " The model is still warming up. Open a sample tray, or try again in a minute.");
      return;
    }
    let started;
    try {
      started = await api("/api/inspect", {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file,
      });
    } catch (error) {
      const title = error.status === 400 ? "Can't read this file"
        : error.status === 413 ? "Photo too large"
          : error.status === 503 ? "Not yet" : "Not filed";
      const advice = error.status === 400 ? " Try a JPEG, PNG or HEIC photo." : "";
      showErrata(title, ` ${cap(error.message.replace(/\.$/, ""))}.${advice}`);
      return;
    }
    const title = file.name.replace(/\.[^.]+$/, "") || "Live photo";
    showPrinting(started.image, title);
    pollJob(started.job, title);
  }

  function showPrinting(imageId, title) {
    stopPolling();
    hideSlip();
    state.job = null;
    state.berries = [];
    el.form.hidden = true;
    el.filed.hidden = true;
    el.printing.hidden = false;
    el.toggleWrap.hidden = true;
    el.marks.replaceChildren();
    removeTags();
    el.photo.classList.add("is-reading");
    placePhoto(`/api/image/${encodeURIComponent(imageId)}`, null);
    el.plateCaption.textContent = `Plate · ${title} · reading`;
    el.fileName.textContent = title;
    el.dateline.textContent = `Report ${pad(state.reportNo + 1, 3)} · ${stamped()} · reading`;
    const started = Date.now();
    const tick = () => {
      const s = Math.round((Date.now() - started) / 1000);
      el.printingText.textContent =
        `Finding each berry, then grading it. This takes about half a minute on this laptop · ${s}s so far.`;
    };
    tick();
    state.timer = setInterval(tick, 1000);
  }

  function pollJob(jobId, title) {
    const step = async () => {
      try {
        const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
        if (job.state === "done") {
          stopPolling();
          await showReport(job, title);
          return;
        }
        if (job.state === "error") {
          stopPolling();
          backToForm();
          showErrata("Not filed", ` ${job.error || "The model could not read this photo."}`);
          return;
        }
      } catch (error) {
        stopPolling();
        backToForm();
        showErrata("Lost the report", ` ${error.message}`);
        return;
      }
      state.polling = setTimeout(step, 900);
    };
    step();
  }

  function stopPolling() {
    clearTimeout(state.polling);
    clearInterval(state.timer);
    state.polling = null;
    state.timer = null;
  }

  // ------------------------------------------------------------- photo plate

  function placePhoto(src, size) {
    return new Promise((resolve) => {
      const img = new Image();
      img.alt = "Tray photo";
      img.decoding = "async";
      img.onload = () => {
        el.photo.querySelector("img")?.remove();
        el.photo.prepend(img);
        state.img = img;
        el.photo.dataset.w = size ? size[0] : img.naturalWidth;
        el.photo.dataset.h = size ? size[1] : img.naturalHeight;
        el.photo.hidden = false;
        el.intake.hidden = true;
        el.contact.hidden = true;
        fitPhoto();
        resolve(img);
      };
      img.onerror = () => resolve(null);
      img.src = src;
    });
  }

  function fitPhoto() {
    if (el.photo.hidden) return;
    const w = Number(el.photo.dataset.w) || 1;
    const h = Number(el.photo.dataset.h) || 1;
    // narrow screens: the plate is as wide as the sheet and as tall as the photo needs
    const plate = el.stage.parentElement;
    const plateW = plate.clientWidth - parseFloat(getComputedStyle(plate).paddingLeft) * 2;
    const availW = Math.max(120, (narrow.matches ? plateW : el.stage.clientWidth) - FRAME);
    const availH = narrow.matches ? Infinity : Math.max(120, el.stage.clientHeight - FRAME);
    const scale = Math.min(availW / w, availH / h);
    el.photo.style.width = `${Math.round(w * scale) + FRAME}px`;
    el.photo.style.height = `${Math.round(h * scale) + FRAME}px`;
    if (state.job) drawMarks();
    const shown = state.on ?? state.pinned;
    if (shown !== null && shown !== undefined) positionSlip(shown);
  }

  function removeTags() { el.photo.querySelector(".tags")?.remove(); }

  function drawMarks() {
    const { width: W, height: H } = state.job;
    const shownW = el.photo.clientWidth - FRAME || W;
    const px = W / shownW; // viewBox units per screen pixel
    el.marks.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.marks.setAttribute("preserveAspectRatio", "none");
    el.marks.replaceChildren();
    removeTags();
    const tags = node("div", { class: "tags" });

    state.berries.forEach((berry, i) => {
      const [x0, y0, x1, y1] = berry.bbox;
      const g = svg("g", { class: `mark m-${berry.maturity}${berry.grade === "reject" ? " g-reject" : ""}` });
      g.dataset.id = berry.id;
      g.style.setProperty("--i", i);
      const box = { x: x0, y: y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0), rx: 6 * px };
      g.append(svg("rect", { ...box, class: "halo" }), svg("rect", { ...box, class: "line" }));
      if (berry.maturity === "overripe") {
        const inset = 5 * px;
        g.append(svg("rect", {
          x: x0 + inset, y: y0 + inset,
          width: Math.max(1, box.width - inset * 2), height: Math.max(1, box.height - inset * 2),
          rx: 4 * px, class: "inner",
        }));
      }
      el.marks.append(g);

      const flipped = y0 / H < 0.07;
      const said = [berry.maturity, ...berry.defects.map(defectLabel), berry.grade].filter(Boolean).join(", ");
      const tag = node("button", {
        class: `tag m-${berry.maturity}${flipped ? " is-flipped" : ""}`,
        type: "button",
        "aria-label": `${berry.code}: ${said}`,
        style: `left:${(x0 / W) * 100}%;top:${(y0 / H) * 100}%;--i:${i}`,
      }, document.createTextNode(berry.code));
      if (berry.grade === "reject") tag.append(node("span", { class: "tag-reject", text: "Reject" }));
      tag.dataset.id = berry.id;
      bindBerry(tag, berry.id);
      tags.append(tag);
    });
    el.photo.append(tags);
    const innerW = el.photo.clientWidth - FRAME;
    for (const tag of tags.children) {
      const leftPx = tag.offsetLeft;
      if (leftPx + tag.offsetWidth > innerW) tag.style.left = `${Math.max(0, innerW - tag.offsetWidth)}px`;
    }
    paintOn();
  }

  // ------------------------------------------------------------- report

  function codeBerries(berries) {
    // Lot codes follow reading order on the photo: rows top to bottom, then left to right.
    const heights = berries.map((b) => b.bbox[3] - b.bbox[1]).sort((a, b) => a - b);
    const rowH = (heights[Math.floor(heights.length / 2)] || 1) * 0.75;
    const spatial = [...berries].sort((a, b) => {
      const ra = Math.round((a.bbox[1] + a.bbox[3]) / 2 / rowH);
      const rb = Math.round((b.bbox[1] + b.bbox[3]) / 2 / rowH);
      return ra - rb || (a.bbox[0] + a.bbox[2]) - (b.bbox[0] + b.bbox[2]);
    });
    spatial.forEach((b, i) => { b.code = `B-${pad(i + 1)}`; b.order = i; });
  }

  function headlineFor(summary) {
    const rejects = summary.grade ? summary.grade.reject : 0;
    switch (summary.verdict) {
      case "premium": return "Ship as premium.";
      case "standard": return "Sort before shipping.";
      case "reject":
        return rejects >= summary.count
          ? "Reject this tray."
          : `Pull ${plural(rejects, "berry", "berries")} before shipping.`;
      default: return "No strawberries found.";
    }
  }

  function renderTally(summary) {
    const totals = [["Berries found", summary.count], ["Ready to ship", summary.ready]];
    if (summary.packout !== null && summary.packout !== undefined) {
      totals.push(["Premium pack-out", `${Math.round(summary.packout * 100)}%`]);
    }
    const left = totals.map(([label, value]) =>
      node("div", {}, node("dt", { text: label }), node("dd", { text: String(value) })));
    const right = MATURITY.map((m) => node("div", {},
      node("dt", {}, node("span", { class: `form-swatch m-${m}`, "aria-hidden": "true" }), document.createTextNode(cap(m))),
      node("dd", { text: String(summary.maturity[m] || 0) })));
    // totals fill the first printed column, ripeness the second
    while (left.length < right.length) left.push(node("div", { "aria-hidden": "true" }));
    el.tally.replaceChildren(...left, ...right);
    el.mixbar.replaceChildren(...MATURITY
      .filter((m) => summary.maturity[m])
      .map((m) => node("span", { class: `m-${m}`, style: `flex:${summary.maturity[m]}` })));
    el.mixbar.setAttribute("aria-label", MATURITY.map((m) => `${summary.maturity[m] || 0} ${m}`).join(", "));
    const grades = summary.grade || {};
    const gradeKeys = ["premium", "standard", "reject"].filter((g) => grades[g]);
    el.gradebar.replaceChildren(...gradeKeys.map((g) => node("span", { class: `g-${g}`, style: `flex:${grades[g]}` })));
    el.gradebar.setAttribute("aria-label", gradeKeys.map((g) => `${grades[g]} ${g}`).join(", "));
    el.mixCount.textContent = MATURITY.filter((m) => summary.maturity[m]).map((m) => `${summary.maturity[m]} ${m}`).join(" · ");
    el.gradeCount.textContent = gradeKeys.map((g) => `${grades[g]} ${g}`).join(" · ");
    for (const n of [el.gradebar, el.gradebar.previousElementSibling, el.gradeCount]) n.hidden = gradeKeys.length === 0;
  }

  function renderLots() {
    el.lotRows.replaceChildren();
    state.berries.forEach((berry, i) => {
      const row = node("tr", { style: `--i:${i + 5}` },
        node("td", {}, node("button", { class: "lot-code", type: "button", "aria-label": `Show ${berry.code}` },
          node("span", { class: `form-swatch m-${berry.maturity}`, "aria-hidden": "true" }),
          document.createTextNode(berry.code))),
        node("td", { class: `ripeness m-${berry.maturity}`, text: berry.maturity }),
        node("td", {
          class: `defects${berry.defects.some(isRejectDefect) ? " has-reject" : ""}`,
          text: berry.defects.length ? berry.defects.map(defectLabel).join(", ") : "—",
        }),
        node("td", { class: `grade num g-${berry.grade || "none"}` },
          berry.grade === "reject"
            ? node("span", { class: "mini-stamp", text: "Reject" })
            : document.createTextNode(berry.grade || "—")),
      );
      row.dataset.id = berry.id;
      bindBerry(row, berry.id);
      el.lotRows.append(row);
    });
  }

  function replay(target, cls) {
    target.classList.remove(cls);
    void target.offsetWidth; // restart the CSS animation
    target.classList.add(cls);
  }

  async function showReport(job, title) {
    stopPolling();
    hideSlip();
    state.reportNo += 1;
    const berries = (job.berries || []).map((b) => ({ ...b, defects: b.defects || [], grade: b.grade || null }));
    codeBerries(berries);
    state.berries = [...berries].sort((a, b) =>
      (GRADE_ORDER[a.grade] ?? 3) - (GRADE_ORDER[b.grade] ?? 3) || a.order - b.order);
    state.job = job;
    state.on = null;
    state.pinned = null;

    await placePhoto(`/api/image/${encodeURIComponent(job.image)}`, [job.width, job.height]);
    el.photo.classList.remove("is-reading");

    el.fileName.textContent = title;
    for (const tray of el.trays.querySelectorAll(".tray")) {
      const current = tray.dataset.id === job.image;
      tray.classList.toggle("is-current", current);
      if (current) tray.setAttribute("aria-current", "true");
      else tray.removeAttribute("aria-current");
    }
    const summary = job.summary || { count: 0, verdict: "empty", reasons: [], maturity: {}, grade: null };
    const empty = summary.count === 0;
    const no = pad(state.reportNo, 3);
    el.dateline.textContent = `Report ${no} · ${stamped()}`;
    el.plateCaption.textContent = `Plate ${no} · ${title} · ${plural(summary.count, "berry", "berries")} outlined`;
    el.toggleWrap.hidden = empty;

    el.verdict.textContent = headlineFor(summary);
    const stampWord = { premium: "Premium", standard: "Standard", reject: "Reject" }[summary.verdict];
    el.stamp.hidden = !stampWord;
    el.stamp.textContent = stampWord || "";
    el.stamp.className = `stamp g-${summary.verdict}`;
    el.condition.replaceChildren(
      node("b", { text: "Condition" }),
      document.createTextNode(`${summary.reasons.join("; ")}.`),
    );

    renderTally(summary);
    renderLots();
    el.tally.hidden = empty;
    el.mixes.hidden = empty;
    el.lots.hidden = empty;

    el.form.hidden = true;
    el.printing.hidden = true;
    el.filed.hidden = false;
    el.report.scrollTop = 0;

    if (!reduceMotion.matches) {
      el.filed.querySelector(".verdict-row").style.setProperty("--i", 0);
      el.condition.style.setProperty("--i", 1);
      el.tally.style.setProperty("--i", 2);
      el.mixes.style.setProperty("--i", 3);
      replay(el.filed, "is-printing");
      replay(el.photo, "is-printing");
      replay(el.stamp, "is-pressed");
      clearTimeout(state.printDone);
      state.printDone = setTimeout(() => {
        el.filed.classList.remove("is-printing");
        el.photo.classList.remove("is-printing");
      }, 340 + 55 * (state.berries.length + 6) + 200);
    }
    el.verdict.focus({ preventScroll: true });
  }

  function backToForm() {
    stopPolling();
    hideSlip();
    state.job = null;
    state.berries = [];
    state.on = null;
    state.pinned = null;
    el.filed.hidden = true;
    el.printing.hidden = true;
    el.form.hidden = false;
    el.photo.classList.remove("is-reading");
    showEmptyPlate();
    el.toggleWrap.hidden = true;
    el.plateCaption.textContent = "Plate · awaiting a tray photo";
    el.dateline.textContent = `Report ${pad(state.reportNo + 1, 3)} · ${stamped()} · unfiled`;
    el.fileName.textContent = "unfiled";
    for (const tray of el.trays.querySelectorAll(".tray")) {
      tray.classList.remove("is-current");
      tray.removeAttribute("aria-current");
    }
    el.marks.replaceChildren();
    removeTags();
  }

  // ------------------------------------------------------------- linking + slip

  function bindBerry(target, id) {
    target.addEventListener("mouseenter", () => setOn(id));
    target.addEventListener("mouseleave", () => setOn(null));
    target.addEventListener("focusin", () => setOn(id));
    target.addEventListener("focusout", () => setOn(null));
    target.addEventListener("click", (event) => {
      event.stopPropagation();
      state.pinned = state.pinned === id ? null : id;
      setOn(id);
    });
  }

  function setOn(id) {
    state.on = id;
    paintOn();
    const shown = id ?? state.pinned;
    if (shown === null || shown === undefined) hideSlip();
    else showSlip(shown);
  }

  function paintOn() {
    const active = state.on ?? state.pinned;
    const has = active !== null && active !== undefined;
    el.marks.classList.toggle("has-focus", has);
    el.photo.querySelector(".tags")?.classList.toggle("has-focus", has);
    for (const n of document.querySelectorAll(".mark, .tag, #lot-rows tr")) {
      n.classList.toggle("is-on", has && String(n.dataset.id) === String(active));
    }
  }

  const berryById = (id) => state.berries.find((b) => String(b.id) === String(id));

  function showSlip(id) {
    const berry = berryById(id);
    if (!berry || !state.img) return;
    el.slipCode.textContent = `Lot ${berry.code}`;
    el.slipRipeness.textContent = berry.maturity;
    el.slipRipeness.className = `slip-ripeness m-${berry.maturity}`;
    el.slipDefects.textContent = berry.defects.length
      ? berry.defects.map((d) => cap(defectLabel(d))).join(", ")
      : "None seen";
    el.slipDefects.className = berry.defects.some(isRejectDefect) ? "is-reject" : "";
    el.slipGrade.textContent = berry.grade ? cap(berry.grade) : "Not graded";
    el.slipGrade.className = berry.grade === "reject" ? "is-reject" : berry.grade === "premium" ? "is-premium" : "";
    const probs = Object.entries(berry.probabilities || {}).sort((a, b) => b[1] - a[1]);
    el.slipProbs.replaceChildren(...probs.map(([label, p]) => node("li", {},
      node("span", { text: label }),
      node("span", { class: "bar", style: `--p:${(p * 100).toFixed(1)}` }),
      node("span", { class: "pct", text: `${Math.round(p * 100)}%` }))));
    el.slipConfidence.hidden = probs.length === 0;
    el.slipCrop.setAttribute("aria-label", `Close-up of ${berry.code}`);
    dockSlip(id);
    el.slip.hidden = false;
    paintCrop(berry);
    positionSlip(id);
  }

  // Desktop: the slip is a leaf laid on the plate. Narrow screens: it opens in flow under its row.
  const slipRow = node("tr", { class: "slip-row" }, node("td", { colspan: "4" }));
  function dockSlip(id) {
    if (narrow.matches) {
      const row = el.lotRows.querySelector(`tr[data-id="${id}"]`);
      if (row) {
        row.after(slipRow);
        slipRow.firstChild.append(el.slip);
        return;
      }
    }
    if (el.slip.parentElement !== el.stage) el.stage.append(el.slip);
    slipRow.remove();
  }

  function paintCrop(berry) {
    const [x0, y0, x1, y1] = berry.bbox;
    const cw = el.slipCrop.clientWidth;
    const ch = el.slipCrop.clientHeight;
    const scale = Math.min(cw / (Math.max(1, x1 - x0) * 1.35), ch / (Math.max(1, y1 - y0) * 1.35));
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    el.slipCrop.style.backgroundImage = `url("${state.img.src}")`;
    el.slipCrop.style.backgroundSize = `${state.job.width * scale}px ${state.job.height * scale}px`;
    el.slipCrop.style.backgroundPosition = `${cw / 2 - cx * scale}px ${ch / 2 - cy * scale}px`;
  }

  function positionSlip(id) {
    const berry = berryById(id);
    if (!berry || el.slip.hidden || narrow.matches) return;
    const stage = el.stage.getBoundingClientRect();
    const photo = el.photo.getBoundingClientRect();
    const k = (photo.width - FRAME) / state.job.width;
    const [x0, y0, x1] = berry.bbox;
    const originX = photo.left - stage.left + FRAME / 2;
    const originY = photo.top - stage.top + FRAME / 2;
    const y1 = berry.bbox[3];
    const box = {
      left: originX + x0 * k, right: originX + x1 * k,
      top: originY + y0 * k, bottom: originY + y1 * k,
    };
    const gap = 14;
    // right, left, below, above; then the same without the close-up; then across the fold
    const place = () => {
      const slipW = el.slip.offsetWidth;
      const slipH = el.slip.offsetHeight;
      const clampY = (y) => Math.min(Math.max(0, y), Math.max(0, stage.height - slipH));
      const clampX = (x) => Math.min(Math.max(0, x), Math.max(0, stage.width - slipW));
      return [
        { x: box.right + gap, y: clampY(box.top - 8), ok: box.right + gap + slipW <= stage.width },
        { x: box.left - gap - slipW, y: clampY(box.top - 8), ok: box.left - gap - slipW >= 0 },
        { x: clampX(box.left), y: box.bottom + gap, ok: box.bottom + gap + slipH <= stage.height },
        { x: clampX(box.left), y: box.top - gap - slipH, ok: box.top - gap - slipH >= 0 },
      ].find((c) => c.ok);
    };
    el.slip.classList.remove("is-compact");
    let pick = place();
    if (!pick) {
      el.slip.classList.add("is-compact");
      pick = place() || { x: box.right + gap, y: Math.max(0, box.top - 8) };
    }
    const x = pick.x;
    const y = pick.y;
    el.slip.style.left = `${x}px`;
    el.slip.style.top = `${y}px`;
  }

  function hideSlip() {
    el.slip.hidden = true;
    slipRow.remove();
    if (el.slip.parentElement !== el.stage) el.stage.append(el.slip);
    el.slipConfidence.open = false;
  }

  // ------------------------------------------------------------- input

  el.choose.addEventListener("click", () => el.file.click());
  el.file.addEventListener("change", () => {
    fileLivePhoto(el.file.files[0]);
    el.file.value = "";
  });
  el.newFile.addEventListener("click", () => { backToForm(); clearErrata(); });
  el.slipClose.addEventListener("click", () => { state.pinned = null; setOn(null); });
  el.toggle.addEventListener("change", () => el.photo.classList.toggle("model-off", !el.toggle.checked));
  el.stage.addEventListener("click", (event) => {
    if (event.target === el.stage || event.target === el.photo || event.target === state.img) {
      state.pinned = null;
      setOn(null);
    }
  });

  let dragDepth = 0;
  const hasFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  window.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
    el.sheet.classList.add("is-dragging");
  });
  window.addEventListener("dragover", (event) => { if (hasFiles(event)) event.preventDefault(); });
  window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) el.sheet.classList.remove("is-dragging");
  });
  window.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    el.sheet.classList.remove("is-dragging");
    fileLivePhoto(event.dataTransfer.files[0]);
  });

  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) && target.type !== "checkbox") return;
    const key = event.key;
    if (key === "Escape") {
      state.pinned = null;
      setOn(null);
      return;
    }
    if (!state.job) return;
    if (key === "m" || key === "M") {
      el.toggle.checked = !el.toggle.checked;
      el.toggle.dispatchEvent(new Event("change"));
    } else if (key === "n" || key === "N") {
      backToForm();
      clearErrata();
    } else if (["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft"].includes(key) && state.berries.length) {
      event.preventDefault();
      const current = state.berries.findIndex((b) => String(b.id) === String(state.pinned ?? state.on));
      const delta = key === "ArrowDown" || key === "ArrowRight" ? 1 : -1;
      const count = state.berries.length;
      const next = current < 0 ? (delta > 0 ? 0 : count - 1) : (current + delta + count) % count;
      const berry = state.berries[next];
      state.pinned = berry.id;
      setOn(berry.id);
      el.lotRows.querySelector(`tr[data-id="${berry.id}"]`)?.scrollIntoView({ block: "nearest" });
    }
  });

  new ResizeObserver(() => fitPhoto()).observe(el.stage.parentElement);

  el.dateline.textContent = `Report 001 · ${stamped()} · unfiled`;
  setModel("warming");
  pollStatus();
  loadSamples();
})();
