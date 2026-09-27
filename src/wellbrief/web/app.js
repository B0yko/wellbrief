// wellbrief local UI: Ask, Brief and NPT views over this same server's own JSON API.
//
// Rules this file keeps everywhere, because the server's Content-Security-Policy
// (`default-src 'self'`, no `unsafe-inline`) depends on them and because the archive holds
// text nobody has reviewed for markup: every document, question and quote reaches the page
// through `textContent` or `document.createTextNode`, never through `innerHTML`, `outerHTML`,
// `insertAdjacentHTML` or `document.write`. Every element is built with `createElement` and
// assembled with `appendChild`. The only styles this script sets on an element are numeric bar
// widths and a highlighted quote's `<mark>`, both through the `style`/DOM APIs, never by writing
// a `style` attribute as a string. Every request this page makes is a relative path on this same
// origin (`/api/...`); it never names another scheme or host.
//
// No build step: this is the file the browser runs, in the ECMAScript 5 subset every current
// browser understands without transpilation.

(function () {
  "use strict";

  // ---------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------

  function $(id) {
    return document.getElementById(id);
  }

  function clear(el) {
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  function setText(el, value) {
    el.textContent = value === null || value === undefined ? "" : String(value);
  }

  function setStatus(el, message, isError) {
    el.textContent = message || "";
    if (isError) {
      el.setAttribute("data-tone", "error");
    } else {
      el.removeAttribute("data-tone");
    }
  }

  function formatHours(n) {
    return (typeof n === "number" ? n : 0).toFixed(1) + " h";
  }

  function formatUsd(n) {
    var rounded = Math.round(typeof n === "number" ? n : 0);
    return "$" + rounded.toLocaleString("en-US");
  }

  function formatPercent(fraction) {
    return Math.round((typeof fraction === "number" ? fraction : 0) * 100) + " %";
  }

  function listOrDash(values) {
    return values && values.length ? values.join(", ") : "-";
  }

  function humanizeKey(key) {
    return key.replace(/_/g, " ");
  }

  function shortHash(hash) {
    return hash ? hash.slice(0, 16) : "n/a";
  }

  // A JSON fetch against this same origin. Rejects with an Error carrying the API's own
  // `error` message when the response is not ok, so every caller can show one line to the user.
  function api(path, options) {
    options = options || {};
    var init = { method: options.method || "GET", headers: {} };
    if (options.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    return fetch(path, init).then(function (response) {
      return response
        .json()
        .catch(function () {
          return {};
        })
        .then(function (data) {
          if (!response.ok) {
            var message = (data && data.error) || "request failed with status " + response.status;
            throw new Error(message);
          }
          return data;
        });
    });
  }

  function fillFieldSelect(select, fields, opts) {
    clear(select);
    if (opts && opts.allOption) {
      var allOpt = document.createElement("option");
      allOpt.value = "";
      allOpt.textContent = opts.allOption;
      select.appendChild(allOpt);
    } else if (opts && opts.placeholder) {
      var placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = opts.placeholder;
      placeholder.disabled = true;
      placeholder.selected = true;
      select.appendChild(placeholder);
    }
    fields.forEach(function (name) {
      var option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    });
  }

  function appendRow(dl, label, value, alignText) {
    var dt = document.createElement("dt");
    dt.textContent = label;
    var dd = document.createElement("dd");
    if (alignText) {
      dd.className = "text";
    }
    dd.textContent = value === null || value === undefined || value === "" ? "-" : String(value);
    dl.appendChild(dt);
    dl.appendChild(dd);
  }

  function appendSubHeading(dl, label) {
    var dt = document.createElement("dt");
    dt.className = "kv-sub-heading";
    dt.textContent = label;
    dl.appendChild(dt);
  }

  // ---------------------------------------------------------------------
  // Finding a verbatim quote inside a document's raw text, for highlighting.
  //
  // A quote is copied out of its source with internal whitespace runs collapsed to one space
  // (see the citation quoting on the server side), so an exact substring search is tried first
  // and a whitespace-insensitive one second, before giving up and showing the document with no
  // highlight -- never a reason to fall back to markup.
  // ---------------------------------------------------------------------

  function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function findQuoteSpan(text, quote) {
    if (!quote) {
      return null;
    }
    var direct = text.indexOf(quote);
    if (direct !== -1) {
      return [direct, direct + quote.length];
    }
    var words = quote
      .trim()
      .split(/\s+/)
      .filter(function (w) {
        return w.length > 0;
      })
      .map(escapeRegExp);
    if (!words.length) {
      return null;
    }
    var pattern;
    try {
      pattern = new RegExp(words.join("\\s+"));
    } catch (err) {
      return null;
    }
    var match = pattern.exec(text);
    return match ? [match.index, match.index + match[0].length] : null;
  }

  function pageForOffset(pages, offset) {
    for (var i = 0; i < pages.length; i++) {
      if (offset < pages[i]) {
        return i + 1;
      }
    }
    return pages.length || null;
  }

  // ---------------------------------------------------------------------
  // Tabs: a standard tablist (Left/Right/Home/End move focus and selection; a tab is a
  // <button role="tab">, its panel a <section role="tabpanel"> hidden through the `hidden`
  // attribute, so a print of the active panel never shows the others).
  // ---------------------------------------------------------------------

  var TAB_ORDER = ["ask", "brief", "npt"];

  function tabButton(name) {
    return $("tab-" + name);
  }

  function panelSection(name) {
    return $("panel-" + name);
  }

  function selectTab(name) {
    TAB_ORDER.forEach(function (candidate) {
      var selected = candidate === name;
      var button = tabButton(candidate);
      button.setAttribute("aria-selected", selected ? "true" : "false");
      button.tabIndex = selected ? 0 : -1;
      panelSection(candidate).hidden = !selected;
    });
  }

  function onTabKeydown(event, name) {
    var index = TAB_ORDER.indexOf(name);
    var nextIndex = null;
    if (event.key === "ArrowRight") {
      nextIndex = (index + 1) % TAB_ORDER.length;
    } else if (event.key === "ArrowLeft") {
      nextIndex = (index - 1 + TAB_ORDER.length) % TAB_ORDER.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = TAB_ORDER.length - 1;
    }
    if (nextIndex === null) {
      return;
    }
    event.preventDefault();
    var nextName = TAB_ORDER[nextIndex];
    selectTab(nextName);
    tabButton(nextName).focus();
  }

  function initTabs() {
    TAB_ORDER.forEach(function (name) {
      var button = tabButton(name);
      button.addEventListener("click", function () {
        selectTab(name);
      });
      button.addEventListener("keydown", function (event) {
        onTabKeydown(event, name);
      });
    });
  }

  // ---------------------------------------------------------------------
  // Source document side panel: shared by every citation chip, in the Ask answer, the
  // mitigations under it, and a brief's risk evidence and mitigations alike.
  // ---------------------------------------------------------------------

  var docPanelReturnFocus = null;

  function onDocPanelKeydown(event) {
    if (event.key === "Escape") {
      closeDocPanel();
    }
  }

  function openDocPanel(docId, quote, triggerEl) {
    docPanelReturnFocus = triggerEl || null;
    var panel = $("doc-panel");
    var backdrop = $("doc-panel-backdrop");
    panel.hidden = false;
    panel.setAttribute("aria-hidden", "false");
    backdrop.hidden = false;
    setText($("doc-panel-title"), docId);
    clear($("doc-panel-meta"));
    $("doc-panel-page-note").hidden = true;
    setText($("doc-panel-page-note"), "");
    var body = $("doc-panel-body");
    clear(body);
    var loading = document.createElement("p");
    loading.textContent = "Loading source document...";
    body.appendChild(loading);
    document.addEventListener("keydown", onDocPanelKeydown);
    $("doc-panel-close").focus();

    api("/api/doc/" + encodeURIComponent(docId))
      .then(function (doc) {
        renderDocPanel(doc, quote);
      })
      .catch(function (err) {
        clear(body);
        var p = document.createElement("p");
        p.textContent = "Could not load this document: " + err.message;
        body.appendChild(p);
      });
  }

  function closeDocPanel() {
    var panel = $("doc-panel");
    panel.hidden = true;
    panel.setAttribute("aria-hidden", "true");
    $("doc-panel-backdrop").hidden = true;
    document.removeEventListener("keydown", onDocPanelKeydown);
    if (docPanelReturnFocus && typeof docPanelReturnFocus.focus === "function") {
      docPanelReturnFocus.focus();
    }
    docPanelReturnFocus = null;
  }

  function renderDocPanel(doc, quote) {
    var meta = $("doc-panel-meta");
    appendRow(meta, "Type", doc.doc_type, true);
    appendRow(meta, "Well", doc.well, true);
    appendRow(meta, "Field", doc.field, true);
    appendRow(meta, "Date", doc.date, true);

    var body = $("doc-panel-body");
    clear(body);
    var span = findQuoteSpan(doc.text, quote);
    var pageNote = $("doc-panel-page-note");
    if (span && doc.pages) {
      var page = pageForOffset(doc.pages, span[0]);
      if (page) {
        setText(pageNote, "Page " + page + " of " + doc.pages.length);
        pageNote.hidden = false;
      }
    }

    if (span) {
      body.appendChild(document.createTextNode(doc.text.slice(0, span[0])));
      var mark = document.createElement("mark");
      mark.textContent = doc.text.slice(span[0], span[1]);
      body.appendChild(mark);
      body.appendChild(document.createTextNode(doc.text.slice(span[1])));
      window.requestAnimationFrame(function () {
        mark.scrollIntoView({ block: "center" });
      });
    } else {
      body.appendChild(document.createTextNode(doc.text));
    }
  }

  // A row of chips, one per citation ({doc_id, well, quote}); clicking one opens the source
  // document with that quote highlighted. `emptyText`, when given, is shown instead of an
  // empty row.
  function renderCitationChips(container, citations, emptyText) {
    clear(container);
    if (!citations || !citations.length) {
      if (emptyText) {
        var note = document.createElement("p");
        note.className = "status-line";
        note.textContent = emptyText;
        container.appendChild(note);
      }
      return;
    }
    citations.forEach(function (citation) {
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = citation.well ? citation.doc_id + " · " + citation.well : citation.doc_id;
      chip.addEventListener("click", function () {
        openDocPanel(citation.doc_id, citation.quote, chip);
      });
      container.appendChild(chip);
    });
  }

  // ---------------------------------------------------------------------
  // Ask
  // ---------------------------------------------------------------------

  var EXAMPLE_TEMPLATES = [
    function (field) {
      return field + " field. What non-productive time has this field recorded?";
    },
    function (field) {
      return field + " field. What lessons were learned on these wells?";
    },
    function (field) {
      return field + " field. What is the largest driver of non-productive time?";
    },
  ];

  function renderExamples(fields) {
    var container = $("ask-examples");
    clear(container);
    fields.forEach(function (field, index) {
      var template = EXAMPLE_TEMPLATES[index % EXAMPLE_TEMPLATES.length];
      var question = template(field);
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = question;
      chip.addEventListener("click", function () {
        $("ask-field").value = "";
        $("ask-question").value = question;
        runAsk();
      });
      container.appendChild(chip);
    });
  }

  function runAsk() {
    var question = $("ask-question").value.trim();
    if (!question) {
      return;
    }
    var field = $("ask-field").value;
    var topKRaw = $("ask-top-k").value;
    var body = { question: question };
    if (field) {
      body.field = field;
    }
    var topK = topKRaw ? parseInt(topKRaw, 10) : NaN;
    if (isFinite(topK) && topK > 0) {
      body.top_k = topK;
    }
    setStatus($("ask-status"), "Asking...", false);
    $("ask-result").hidden = true;
    api("/api/ask", { method: "POST", body: body })
      .then(function (answer) {
        setStatus($("ask-status"), "", false);
        renderAnswer(answer);
      })
      .catch(function (err) {
        setStatus($("ask-status"), err.message, true);
      });
  }

  function renderAnswer(answer) {
    var badge = $("ask-verify-badge");
    if (answer.abstained) {
      setText(badge, "Abstained");
      badge.setAttribute("data-state", "abstained");
    } else if (answer.citation_warnings && answer.citation_warnings.length) {
      setText(badge, "NOT VERIFIED");
      badge.setAttribute("data-state", "not-verified");
    } else {
      setText(badge, "Verified");
      badge.setAttribute("data-state", "verified");
    }

    var banner = $("ask-narrator-banner");
    if (answer.narrator_rejected) {
      var reasons = (answer.narrator_rejected.reasons || []).join("; ");
      setText(
        banner,
        "The " +
          answer.narrator +
          " narrator's answer failed verification" +
          (reasons ? " (" + reasons + ")" : "") +
          "; showing the deterministic offline answer instead."
      );
      banner.hidden = false;
    } else {
      banner.hidden = true;
      setText(banner, "");
    }

    setText($("ask-answer-text"), answer.text);
    renderCitationChips($("ask-citations"), answer.citations, "No citations.");

    var mitigationsSection = $("ask-mitigations-section");
    var mitigationsList = $("ask-mitigations-list");
    clear(mitigationsList);
    if (answer.mitigations && answer.mitigations.length) {
      answer.mitigations.forEach(function (mitigation) {
        var li = document.createElement("li");
        var quote = document.createElement("p");
        quote.className = "mitigation-quote";
        quote.textContent = mitigation.text;
        li.appendChild(quote);
        var chips = document.createElement("div");
        chips.className = "chip-row";
        renderCitationChips(chips, [
          { doc_id: mitigation.doc_id, well: mitigation.well, quote: mitigation.text },
        ]);
        li.appendChild(chips);
        mitigationsList.appendChild(li);
      });
      mitigationsSection.hidden = false;
    } else {
      mitigationsSection.hidden = true;
    }

    var plan = answer.query_plan || {};
    setText($("ask-plan-summary"), plan.summary || "");
    renderQueryPlan($("ask-plan-list"), plan);

    var figuresBlock = $("ask-figures-block");
    if (answer.figures) {
      renderFigures($("ask-figures-list"), answer.figures);
      figuresBlock.hidden = false;
    } else {
      figuresBlock.hidden = true;
    }

    $("ask-result").hidden = false;
  }

  function renderQueryPlan(dl, plan) {
    clear(dl);
    appendRow(dl, "Fields", listOrDash(plan.fields), true);
    appendRow(dl, "Wells", listOrDash(plan.wells), true);
    appendRow(dl, "Hole sections", listOrDash(plan.hole_sections), true);
    appendRow(dl, "Formations", listOrDash(plan.formations), true);
    appendRow(dl, "NPT codes", listOrDash(plan.codes), true);
    appendRow(dl, "Document types", listOrDash(plan.doc_types), true);
    if (plan.depth_m !== null && plan.depth_m !== undefined) {
      appendRow(dl, "Depth", plan.depth_m.toLocaleString("en-US") + " m (" + (plan.depth_mode || "-") + ")", true);
    }
    appendRow(dl, "General NPT question", plan.npt ? "yes" : "no", true);
  }

  var BREAKDOWN_LABELS = {
    by_code: "By code",
    by_field: "By field",
    by_well: "By well",
    by_section: "By section",
  };

  function breakdownName(row) {
    return row.label || row.field || row.well || row.section || row.code || row.key || "-";
  }

  function renderFigures(dl, figures) {
    clear(dl);
    appendRow(dl, "Total hours", formatHours(figures.total_hours));
    appendRow(dl, "Total cost", formatUsd(figures.total_cost_usd));
    appendRow(dl, "Avoidable hours", formatHours(figures.avoidable_hours));
    appendRow(dl, "Avoidable share", formatPercent(figures.avoidable_share));
    appendRow(dl, "Events", figures.event_count);
    appendRow(dl, "Wells", figures.well_count);
    appendRow(dl, "Reports", figures.report_count);

    ["by_code", "by_field", "by_well", "by_section"].forEach(function (key) {
      var rows = figures[key];
      if (!rows || !rows.length) {
        return;
      }
      appendSubHeading(dl, BREAKDOWN_LABELS[key] || key);
      rows.forEach(function (row) {
        var value = formatHours(row.hours);
        if (row.cost_usd !== undefined) {
          value += " / " + formatUsd(row.cost_usd);
        }
        appendRow(dl, breakdownName(row), value);
      });
    });
  }

  function initAsk() {
    $("ask-form").addEventListener("submit", function (event) {
      event.preventDefault();
      runAsk();
    });
  }

  // ---------------------------------------------------------------------
  // Brief
  // ---------------------------------------------------------------------

  var lastBriefParams = null;
  var APPLIES_LABELS = {
    applies: "Applies",
    does_not_apply: "Does not apply",
    not_specified: "Not specified",
  };

  function briefQueryString(params) {
    var qs = new URLSearchParams();
    qs.set("field", params.field);
    qs.set("well", params.well);
    qs.set("td", String(params.td));
    if (params.rig) {
      qs.set("rig", params.rig);
    }
    if (params.mwd) {
      qs.set("mwd", params.mwd);
    }
    if (params.spread_rate !== null && params.spread_rate !== undefined) {
      qs.set("spread_rate", String(params.spread_rate));
    }
    return qs.toString();
  }

  function setDownloadLink(anchor, fmt, enabled) {
    if (enabled && lastBriefParams) {
      anchor.href = "/api/brief." + fmt + "?" + briefQueryString(lastBriefParams);
      anchor.removeAttribute("aria-disabled");
      anchor.removeAttribute("tabindex");
    } else {
      anchor.removeAttribute("href");
      anchor.setAttribute("aria-disabled", "true");
      anchor.setAttribute("tabindex", "-1");
    }
  }

  function guardDisabledLink(event) {
    if (event.currentTarget.getAttribute("aria-disabled") === "true") {
      event.preventDefault();
    }
  }

  function runBrief() {
    var field = $("brief-field").value;
    var well = $("brief-well").value.trim();
    var td = parseFloat($("brief-td").value);
    var rig = $("brief-rig").value.trim();
    var mwd = $("brief-mwd").value.trim();
    var spreadRateRaw = $("brief-spread-rate").value;
    if (!field || !well || !isFinite(td)) {
      setStatus($("brief-status"), "A field, a well name and a planned TD are required.", true);
      return;
    }
    var body = { field: field, well: well, td: td };
    if (rig) {
      body.rig = rig;
    }
    if (mwd) {
      body.mwd = mwd;
    }
    var spreadRate = null;
    if (spreadRateRaw) {
      var parsedRate = parseFloat(spreadRateRaw);
      if (isFinite(parsedRate)) {
        spreadRate = parsedRate;
        body.spread_rate = parsedRate;
      }
    }
    lastBriefParams = {
      field: field,
      well: well,
      td: td,
      rig: rig || null,
      mwd: mwd || null,
      spread_rate: spreadRate,
    };
    setStatus($("brief-status"), "Building brief...", false);
    $("brief-result").hidden = true;
    api("/api/brief", { method: "POST", body: body })
      .then(function (payload) {
        setStatus($("brief-status"), "", false);
        renderBrief(payload);
      })
      .catch(function (err) {
        setStatus($("brief-status"), err.message, true);
      });
  }

  function renderBrief(payload) {
    var badge = $("brief-verify-badge");
    if (payload.not_verified) {
      setText(badge, "NOT VERIFIED");
      badge.setAttribute("data-state", "not-verified");
    } else {
      setText(badge, "Verified");
      badge.setAttribute("data-state", "verified");
    }

    setText($("brief-heading"), payload.well_name + " (" + payload.field_name + ")");
    setText(
      $("brief-totals"),
      "Total expected NPT: " +
        formatHours(payload.total_expected_npt_hours) +
        " / " +
        formatUsd(payload.total_exposure_usd) +
        " at " +
        formatUsd(payload.spread_rate_usd_per_day) +
        "/day (risks that do not apply to this plan excluded)."
    );

    var backgroundEl = $("brief-background");
    var unavoidable = payload.unavoidable_hours || {};
    var codes = Object.keys(unavoidable);
    if (codes.length) {
      var total = codes.reduce(function (sum, code) {
        return sum + unavoidable[code];
      }, 0);
      setText(backgroundEl, "Unavoidable background NPT: " + formatHours(total) + " (" + codes.join(", ") + ")");
      backgroundEl.hidden = false;
    } else {
      backgroundEl.hidden = true;
    }

    renderRiskTable(payload.risks || []);
    renderProvenance($("brief-provenance"), payload.provenance || {}, payload.verification || {});
    setText($("brief-disclaimer"), payload.disclaimer);

    var verified = !payload.not_verified;
    setDownloadLink($("brief-download-md"), "md", verified);
    setDownloadLink($("brief-download-json"), "json", verified);

    $("brief-result").hidden = false;
  }

  function appendNumCell(row, text) {
    var td = document.createElement("td");
    td.className = "num";
    td.textContent = text;
    row.appendChild(td);
  }

  function buildRiskDetail(risk) {
    var wrap = document.createElement("div");
    var mitigations = risk.mitigations || [];
    var citations = risk.citations || [];
    var evidenceCount = Math.max(0, citations.length - mitigations.length);
    var evidence = citations.slice(0, evidenceCount);
    var mitigationCitations = citations.slice(evidenceCount);

    var mitigationsHeading = document.createElement("h4");
    mitigationsHeading.textContent = "Mitigations";
    wrap.appendChild(mitigationsHeading);
    if (mitigations.length) {
      var list = document.createElement("ul");
      list.className = "mitigation-list";
      mitigations.forEach(function (text, index) {
        var li = document.createElement("li");
        var quote = document.createElement("p");
        quote.className = "mitigation-quote";
        quote.textContent = text;
        li.appendChild(quote);
        var cite = mitigationCitations[index];
        if (cite) {
          var chips = document.createElement("div");
          chips.className = "chip-row";
          renderCitationChips(chips, [cite]);
          li.appendChild(chips);
        }
        list.appendChild(li);
      });
      wrap.appendChild(list);
    } else {
      var none = document.createElement("p");
      none.className = "status-line";
      none.textContent = "No related mitigations recorded.";
      wrap.appendChild(none);
    }

    var evidenceHeading = document.createElement("h4");
    evidenceHeading.textContent = "Evidence";
    wrap.appendChild(evidenceHeading);
    var evidenceChips = document.createElement("div");
    evidenceChips.className = "chip-row evidence-chips";
    renderCitationChips(evidenceChips, evidence, "No supporting NPT reports recorded.");
    wrap.appendChild(evidenceChips);

    if (risk.counter_examples && risk.counter_examples.length) {
      var wellsNote = document.createElement("p");
      wellsNote.className = "status-line";
      wellsNote.textContent = "Wells that avoided this: " + risk.counter_examples.join(", ") + ".";
      wrap.appendChild(wellsNote);
    }

    return wrap;
  }

  function renderRiskTable(risks) {
    var tbody = $("risk-table-body");
    clear(tbody);
    if (!risks.length) {
      var row = document.createElement("tr");
      var cell = document.createElement("td");
      cell.colSpan = 9;
      cell.textContent = "No risks exceeded the configured thresholds for this field and depth.";
      row.appendChild(cell);
      tbody.appendChild(row);
      return;
    }

    risks.forEach(function (risk, index) {
      var detailId = "risk-detail-" + index;

      var summaryRow = document.createElement("tr");
      summaryRow.className = "risk-row";

      var titleCell = document.createElement("td");
      var toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "risk-toggle";
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-controls", detailId);
      var chevron = document.createElement("span");
      chevron.className = "chevron";
      chevron.setAttribute("aria-hidden", "true");
      chevron.textContent = "▸";
      var titleText = document.createElement("span");
      titleText.textContent = risk.title;
      toggle.appendChild(chevron);
      toggle.appendChild(titleText);
      titleCell.appendChild(toggle);
      summaryRow.appendChild(titleCell);

      var appliesCell = document.createElement("td");
      var appliesBadge = document.createElement("span");
      appliesBadge.className = "badge applies-badge";
      appliesBadge.setAttribute("data-applies", risk.applies);
      appliesBadge.textContent = APPLIES_LABELS[risk.applies] || risk.applies;
      appliesCell.appendChild(appliesBadge);
      summaryRow.appendChild(appliesCell);

      appendNumCell(summaryRow, risk.wells_affected + "/" + risk.wells_total);
      appendNumCell(summaryRow, formatPercent(risk.probability));
      appendNumCell(summaryRow, risk.mean_npt_hours.toFixed(1));
      appendNumCell(summaryRow, risk.p90_npt_hours.toFixed(1));
      appendNumCell(summaryRow, risk.expected_npt_hours.toFixed(1));
      appendNumCell(summaryRow, formatUsd(risk.expected_cost_usd));

      var driverCell = document.createElement("td");
      driverCell.textContent = risk.driver || "-";
      summaryRow.appendChild(driverCell);

      var detailRow = document.createElement("tr");
      detailRow.className = "risk-detail";
      detailRow.id = detailId;
      detailRow.hidden = true;
      var detailCell = document.createElement("td");
      detailCell.colSpan = 9;
      detailCell.appendChild(buildRiskDetail(risk));
      detailRow.appendChild(detailCell);

      toggle.addEventListener("click", function () {
        var expanded = toggle.getAttribute("aria-expanded") === "true";
        toggle.setAttribute("aria-expanded", expanded ? "false" : "true");
        detailRow.hidden = expanded;
      });

      tbody.appendChild(summaryRow);
      tbody.appendChild(detailRow);
    });
  }

  function renderProvenance(dl, provenance, verification) {
    clear(dl);
    appendRow(dl, "Version", provenance.wellbrief_version, true);
    appendRow(dl, "Generated", provenance.generated_at, true);
    appendRow(dl, "Corpus hash", shortHash(provenance.corpus_hash), true);
    appendRow(dl, "Index hash", shortHash(provenance.index_manifest_hash), true);
    appendRow(dl, "Narrator", provenance.narrator, true);
    appendRow(dl, "Spread rate", formatUsd(provenance.spread_rate_usd_per_day) + "/day");
    appendRow(dl, "Citations checked", verification.citations_checked || 0);

    var thresholds = provenance.thresholds || {};
    var keys = Object.keys(thresholds);
    if (keys.length) {
      appendSubHeading(dl, "Thresholds");
      keys.forEach(function (key) {
        appendRow(dl, humanizeKey(key), thresholds[key]);
      });
    }
  }

  function initBrief() {
    $("brief-form").addEventListener("submit", function (event) {
      event.preventDefault();
      runBrief();
    });
    $("brief-download-md").addEventListener("click", guardDisabledLink);
    $("brief-download-json").addEventListener("click", guardDisabledLink);
  }

  // ---------------------------------------------------------------------
  // NPT
  // ---------------------------------------------------------------------

  function renderBarList(container, rows, labelFn, avoidableFn) {
    clear(container);
    if (!rows.length) {
      var note = document.createElement("p");
      note.className = "status-line";
      note.textContent = "No NPT recorded for these filters.";
      container.appendChild(note);
      return;
    }
    var maxHours = rows.reduce(function (m, r) {
      return Math.max(m, r.hours);
    }, 0) || 1;
    rows.forEach(function (row) {
      var wrap = document.createElement("div");
      wrap.className = "bar-row";

      var label = document.createElement("span");
      label.className = "bar-label";
      label.textContent = labelFn(row);
      wrap.appendChild(label);

      var track = document.createElement("div");
      track.className = "bar-track";
      var fill = document.createElement("div");
      fill.className = "bar-fill";
      if (avoidableFn) {
        fill.setAttribute("data-avoidable", avoidableFn(row) ? "true" : "false");
      }
      var pct = Math.max(2, Math.round((row.hours / maxHours) * 100));
      fill.style.width = pct + "%";
      track.appendChild(fill);
      wrap.appendChild(track);

      var value = document.createElement("span");
      value.className = "bar-value";
      value.textContent = row.hours.toFixed(1) + " h · " + formatUsd(row.cost_usd);
      wrap.appendChild(value);

      container.appendChild(wrap);
    });
  }

  function renderNpt(payload) {
    var line =
      "Total: " +
      formatHours(payload.total_hours) +
      " / " +
      formatUsd(payload.total_cost_usd) +
      " at " +
      formatUsd(payload.spread_rate_usd_per_day) +
      "/day (" +
      formatPercent(payload.avoidable_share) +
      " avoidable)";
    if (payload.since) {
      line += ", since " + payload.since;
    }
    setText($("npt-totals"), line);

    renderBarList(
      $("npt-by-code"),
      payload.by_code || [],
      function (row) {
        return row.label || row.code;
      },
      function (row) {
        return row.avoidable;
      }
    );
    renderBarList(
      $("npt-by-well"),
      payload.worst_wells || [],
      function (row) {
        return row.well;
      },
      null
    );

    $("npt-result").hidden = false;
  }

  function runNpt() {
    var field = $("npt-field").value;
    var since = $("npt-since").value;
    var qs = new URLSearchParams();
    if (field) {
      qs.set("field", field);
    }
    if (since) {
      qs.set("since", since);
    }
    var path = "/api/npt" + (qs.toString() ? "?" + qs.toString() : "");
    setStatus($("npt-status"), "Loading...", false);
    $("npt-result").hidden = true;
    api(path)
      .then(function (payload) {
        setStatus($("npt-status"), "", false);
        renderNpt(payload);
      })
      .catch(function (err) {
        setStatus($("npt-status"), err.message, true);
      });
  }

  function initNpt() {
    $("npt-form").addEventListener("submit", function (event) {
      event.preventDefault();
      runNpt();
    });
  }

  // ---------------------------------------------------------------------
  // Header: workspace name, network-mode badge, and the field lists every form draws on.
  // ---------------------------------------------------------------------

  function loadStatus() {
    return api("/api/status")
      .then(function (status) {
        setText($("workspace-name"), status.workspace);
        var badge = $("network-badge");
        setText(badge, "network: " + status.network_mode);
        badge.setAttribute("data-mode", status.network_mode);

        var fields = (status.fields || []).map(function (f) {
          return f.field;
        });
        fillFieldSelect($("ask-field"), fields, { allOption: "All fields" });
        fillFieldSelect($("brief-field"), fields, { placeholder: "Select a field" });
        fillFieldSelect($("npt-field"), fields, { allOption: "All fields" });
        renderExamples(fields);
        return status;
      })
      .catch(function (err) {
        setText($("workspace-name"), "unavailable");
        setStatus($("ask-status"), "Could not reach the local API: " + err.message, true);
      });
  }

  function init() {
    initTabs();
    initAsk();
    initBrief();
    initNpt();
    $("doc-panel-close").addEventListener("click", closeDocPanel);
    $("doc-panel-backdrop").addEventListener("click", closeDocPanel);
    loadStatus();
  }

  init();
})();
