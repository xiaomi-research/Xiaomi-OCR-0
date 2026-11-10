/* Xiaomi-OCR-0 · split compare + typewriter */

const DEFAULT_CFG = {
  max_tokens: 4096,
  temperature: 0,
};
const LS_KEY = "xiaomi_ocr_demo_cfg";
const THEME_KEY = "xiaomi_ocr_demo_theme";

const state = {
  items: [],
  activeId: null,
  typingTimer: null,
  typedDone: false,
  custom: null,
  busy: false,
  mode: "page", // page | region
  task: "document", // document | kie | vqa
  pdfUrl: null,
};

const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* theme */
function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem(THEME_KEY, t);
  $("btnTheme").textContent = t === "dark" ? "☀" : "◐";
}
(function () {
  let t = localStorage.getItem(THEME_KEY);
  if (!t) t = "light";
  applyTheme(t);
})();
$("btnTheme").onclick = function () {
  applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
};

/* config */
function loadCfg() {
  try {
    const s = JSON.parse(localStorage.getItem(LS_KEY) || "{}");
    const o = {};
    Object.keys(DEFAULT_CFG).forEach(function (k) {
      o[k] = s[k] != null && s[k] !== "" ? s[k] : DEFAULT_CFG[k];
    });
    return o;
  } catch (e) { return Object.assign({}, DEFAULT_CFG); }
}
function saveCfg(c) { localStorage.setItem(LS_KEY, JSON.stringify(c)); }
function cfg() { return loadCfg(); }

/* ── optional result views: RAW / JSON stay hidden unless enabled in settings ── */
const VIEW_KEY = "xiaomi-ocr-show-views";
function showViews() { return localStorage.getItem(VIEW_KEY) !== "0"; }
function recordResult(text, meta) {
  state.lastText = String(text || "");
  state.lastMeta = meta || null;
  state.lastTask = state.task;
  $("btnCopy").disabled = !state.lastText;
  $("btnDownload").disabled = !state.lastText;
}
function paintResult() {
  const el = $("resText");
  if (!el || !state.lastText) return;
  const v = state.view || "pretty";
  if (v === "raw") {
    el.classList.remove("markdown");
    el.classList.remove("typing");
    el.classList.add("source-view");
    el.textContent = state.lastText;
  } else if (v === "json") {
    el.classList.remove("markdown");
    el.classList.remove("typing");
    el.classList.add("source-view");
    el.textContent = JSON.stringify({
      model: (state.lastMeta && state.lastMeta.model) || "SeerRay-Lab/Xiaomi-OCR-0",
      latency_ms: state.lastMeta ? state.lastMeta.latency_ms : null,
      usage: (state.lastMeta && state.lastMeta.usage) || null,
      content: state.lastText,
    }, null, 2);
  } else {
    el.classList.remove("typing", "source-view");
    el.classList.add("markdown");
    el.innerHTML = state.lastTask === "kie" ? "<pre><code>" + escapeHtml(state.lastText) + "</code></pre>" : renderMarkdown(normalizeOcrOutput(state.lastText));
  }
  const pane = $("resPane");
  if (pane) pane.scrollTop = 0;
}
function setView(v) {
  state.view = v;
  document.querySelectorAll("#viewSeg button").forEach(function (b) {
    b.classList.toggle("active", b.dataset.view === v);
    b.setAttribute("aria-pressed", String(b.dataset.view === v));
  });
  paintResult();
}
function applyViewVisibility() {
  const seg = $("viewSeg");
  if (!seg) return;
  const on = showViews();
  seg.hidden = !on;
  if (!on) {
    state.view = "pretty";
    document.querySelectorAll("#viewSeg button").forEach(function (b) {
      b.classList.toggle("active", b.dataset.view === "pretty");
    });
  }
}
(function initViewSeg() {
  const seg = $("viewSeg");
  if (seg) {
    seg.addEventListener("click", function (e) {
      const b = e.target.closest("button[data-view]");
      if (b) setView(b.dataset.view);
    });
  }
  const box = $("cfgShowViews");
  if (box) box.checked = showViews();
  applyViewVisibility();
})();

function openDrawer() {
  const c = cfg();
  $("cfgMaxTokens").value = c.max_tokens;
  $("drawer").hidden = false;
  $("backdrop").hidden = false;
  document.querySelector(".shell").inert = true;
  $("cfgMaxTokens").focus();
  $("drawer").classList.add("open");
  $("backdrop").classList.add("open");
}
function closeDrawer() {
  $("drawer").hidden = true;
  $("backdrop").hidden = true;
  document.querySelector(".shell").inert = false;
  $("btnSettings").focus();
  $("drawer").classList.remove("open");
  $("backdrop").classList.remove("open");
}
$("btnSettings").onclick = openDrawer;
$("btnCloseSettings").onclick = closeDrawer;
$("btnCancelCfg").onclick = closeDrawer;
$("backdrop").onclick = closeDrawer;
$("btnSaveCfg").onclick = function () {
  if (!$("cfgMaxTokens").reportValidity()) return;
  saveCfg({
    max_tokens: Number($("cfgMaxTokens").value || 4096),
    temperature: 0,
  });
  localStorage.setItem(VIEW_KEY, $("cfgShowViews") && $("cfgShowViews").checked ? "1" : "0");
  applyViewVisibility();
  paintResult();
  closeDrawer();
};


function setLed(mode) {
  const led = $("ledLive");
  if (!led) return;
  led.classList.remove("on", "err");
  if (mode === "on") led.classList.add("on");
  else if (mode === "err") led.classList.add("err");
}

/* parse mode: page (whole-image stream) | region (layout + parallel) */
(function initModeSeg() {
  const seg = $("modeSeg");
  if (!seg) return;
  seg.addEventListener("click", function (e) {
    const b = e.target.closest("button");
    if (!b || state.busy || state.task !== "document" || b.disabled) return;
    state.mode = b.dataset.mode;
    seg.querySelectorAll("button").forEach(function (x) {
      x.classList.toggle("active", x.dataset.mode === state.mode);
    });
    uiText("stageHint", () => t(state.mode === "region"
      ? "REGION · layout + parallel"
      : "CLICK TO PARSE"));
    updateTaskInputs();
    resetResult();
    clearError();
  });
})();

function updateTaskInputs() {
  const isDoc = state.task === "document";
  const isPdf = !!(state.custom && state.custom.isPdf);
  const region = isDoc && state.mode === "region";
  $("docPromptWrap").hidden = !isDoc || region;
  $("kieFieldsWrap").hidden = state.task !== "kie";
  $("vqaQuestionWrap").hidden = state.task !== "vqa";
  $("regionPromptNote").hidden = !region;
  document.querySelectorAll("#modeSeg button").forEach(function (b) {
    b.disabled = !isDoc || state.busy;
    b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode));
    b.classList.toggle("active", b.dataset.mode === state.mode);
  });
  if (isPdf && !isDoc) {
    state.task = "document";
    document.querySelectorAll("#taskSeg button").forEach(function (b) { b.classList.toggle("active", b.dataset.task === "document"); });
    updateTaskInputs();
    showError("PDF 目前用于文档解析；KIE 和 VQA 请上传或选择图像。");
    return;
  }
  document.querySelectorAll("#taskSeg button").forEach(b => { b.classList.toggle("active", b.dataset.task === state.task); b.setAttribute("aria-pressed", String(b.dataset.task === state.task)); });
  uiText("stageHint", () => t(isPdf ? (region ? "PDF · REGION PARSING" : "PDF · PAGE PARSING") : (region ? "REGION · layout + parallel" : "CLICK TO PARSE")));
}

(function initTaskSeg() {
  const seg = $("taskSeg");
  if (!seg) return;
  seg.addEventListener("click", function (e) {
    const b = e.target.closest("button[data-task]");
    if (!b || state.busy) return;
    if (state.custom && state.custom.isPdf && b.dataset.task !== "document") { showError("PDF 支持文档解析；字段提取和问答请选择图像。"); return; }
    state.task = b.dataset.task;
    if (state.task !== "document") state.mode = "page";
    seg.querySelectorAll("button").forEach(function (x) { x.classList.toggle("active", x === b); });
    document.querySelectorAll("#modeSeg button").forEach(function (x) { x.classList.toggle("active", x.dataset.mode === state.mode); });
    updateTaskInputs();
    seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x.dataset.task === state.task)));
    resetResult();
    clearError();
  });
  $("docPrompt").addEventListener("input", function () { if (state.task === "document") resetResult(); });
  $("kieFields").addEventListener("input", resetResult);
  $("vqaQuestion").addEventListener("input", resetResult);
})();

function firePulse() {
  const p = $("pulse");
  p.classList.remove("go");
  void p.offsetWidth;
  p.classList.add("go");
}

/* ── markdown + katex ── */
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * OTSL (Optical Structure Recognition Language) → HTML table
 *
 * Tags (as emitted by dots.ocr / PaddleOCR-VL / qwen OCR pipelines):
 *   <fcel>  filled cell — content follows until next tag
 *   <ecel>  empty cell
 *   <lcel>  null-left cell — this grid slot is covered by the cell to its left (colspan++)
 *   <ucel>  null-up cell   — this grid slot is covered by the cell above (rowspan++)
 *   <nl>    row break
 *
 * Some models also emit tab-separated rows with tags glued to content
 * (e.g. `本集团<lcel>` or `<ucel><ucel>\t2024-12-31`). Both shapes are accepted.
 */
function parseOtsl(text) {
  // Normalize tags to a canonical token stream.
  // Also honor real newlines as <nl> when the model used tabs+newlines.
  let src = String(text || "");
  src = src
    .replace(/<nl\s*\/?>/gi, "\n")
    .replace(/\r\n?/g, "\n");

  // Tokenize: tags first, then runs of plain text / tabs / newlines
  const TAG = /<(fcel|ecel|lcel|ucel)>/i;
  const tokens = [];
  let i = 0;
  while (i < src.length) {
    if (src[i] === "\n") {
      tokens.push({ t: "nl" });
      i++;
      continue;
    }
    if (src[i] === "\t") {
      // Tab is a cell boundary in hybrid mode (text\ttext, or text<tag>\ttext).
      tokens.push({ t: "softbreak" });
      i++;
      continue;
    }
    const rest = src.slice(i);
    const m = rest.match(/^<(fcel|ecel|lcel|ucel)>/i);
    if (m) {
      tokens.push({ t: m[1].toLowerCase() });
      i += m[0].length;
      continue;
    }
    // plain text run until next tag / newline / tab
    let j = i;
    while (j < src.length && src[j] !== "\n" && src[j] !== "\t" && !TAG.test(src.slice(j, j + 6))) {
      j++;
    }
    if (j === i) { i++; continue; } // safety
    tokens.push({ t: "text", v: src.slice(i, j) });
    i = j;
  }

  // Rebuild grid with rowspan/colspan
  // cell: {text, r, c, rowspan, colspan}
  const cells = [];
  let row = 0, col = 0;
  let pendingText = "";
  // occupancy map to find the master cell covering (r,c)
  const occ = new Map(); // "r,c" -> cell ref

  function key(r, c) { return r + "," + c; }

  function placeFilled(text, isEmpty) {
    const cell = {
      text: isEmpty ? "" : text.trim(),
      r: row, c: col,
      rowspan: 1, colspan: 1,
    };
    cells.push(cell);
    occ.set(key(row, col), cell);
    col++;
    pendingText = "";
  }

  function markLeft() {
    // covered by cell to the left — find nearest master on this row
    let c = col - 1;
    while (c >= 0 && !occ.has(key(row, c))) c--;
    const master = c >= 0 ? occ.get(key(row, c)) : null;
    if (master) master.colspan++;
    occ.set(key(row, col), master || { text: "", r: row, c: col, rowspan: 1, colspan: 1, phantom: true });
    col++;
  }

  function markUp() {
    // covered by cell above — walk up in this column
    let r = row - 1;
    while (r >= 0 && !occ.has(key(r, col))) r--;
    const master = r >= 0 ? occ.get(key(r, col)) : null;
    if (master) master.rowspan++;
    occ.set(key(row, col), master || { text: "", r: row, c: col, rowspan: 1, colspan: 1, phantom: true });
    col++;
  }

  function flushPendingAsCell() {
    if (pendingText.trim()) placeFilled(pendingText, false);
  }

  for (const tok of tokens) {
    if (tok.t === "nl") {
      flushPendingAsCell();
      // if the row had no cells at all, still advance
      row++;
      col = 0;
      continue;
    }
    if (tok.t === "softbreak") {
      // Tab boundary: pending text becomes its own cell; consecutive tabs → empty cells
      if (pendingText.trim()) {
        placeFilled(pendingText, false);
      } else if (col > 0 || cells.length) {
        // empty cell between tabs, but not a leading empty at row start
        // only emit if we already have content on this row or previous cell just placed
        // Heuristic: if last placed cell was on this row, emit empty
        const last = cells[cells.length - 1];
        if (last && last.r === row) {
          placeFilled("", true);
        }
      }
      continue;
    }
    if (tok.t === "text") {
      pendingText += tok.v;
      continue;
    }
    if (tok.t === "fcel") {
      // <fcel> means "a filled cell starts; content follows until next tag".
      // If there is pending text (tab-mode: text BEFORE the tag), that text
      // is the cell. If pending is empty, the cell content comes AFTER this
      // tag — do not emit a spurious empty cell.
      if (pendingText.trim()) {
        placeFilled(pendingText, false);
      } else {
        pendingText = ""; // content will accumulate from following text tokens
      }
      continue;
    }
    if (tok.t === "ecel") {
      // empty cell; if there was pending text, treat as filled (tab-mode quirk)
      if (pendingText.trim()) placeFilled(pendingText, false);
      else placeFilled("", true);
      continue;
    }
    if (tok.t === "lcel") {
      // If pending text exists, this is a filled cell that ALSO extends right
      // (e.g. `本集团<lcel>` → header spanning two date columns).
      if (pendingText.trim()) {
        placeFilled(pendingText, false);
        // placeFilled already advanced col by 1; extend one more for the lcel
        markLeft();
      } else {
        markLeft();
      }
      continue;
    }
    if (tok.t === "ucel") {
      flushPendingAsCell();
      markUp();
      continue;
    }
  }
  flushPendingAsCell();

  // Drop phantoms
  const real = cells.filter((c) => !c.phantom);
  if (!real.length) return null;

  // Normalize: max row/col
  const maxR = Math.max(...real.map((c) => c.r + c.rowspan));
  const maxC = Math.max(...real.map((c) => c.c + c.colspan));

  // Emit HTML
  const byRC = new Map();
  real.forEach((c) => byRC.set(key(c.r, c.c), c));

  let html = "<table>";
  for (let r = 0; r < maxR; r++) {
    html += "<tr>";
    let c = 0;
    while (c < maxC) {
      const cell = byRC.get(key(r, c));
      if (cell && cell.r === r && cell.c === c) {
        const rs = cell.rowspan > 1 ? ` rowspan="${cell.rowspan}"` : "";
        const cs = cell.colspan > 1 ? ` colspan="${cell.colspan}"` : "";
        const txt = escapeHtml(cell.text || "");
        html += `<td${rs}${cs}>${txt}</td>`;
        c += cell.colspan;
      } else if (cell) {
        // covered by a master elsewhere — skip
        c += 1;
      } else {
        // hole: emit empty cell to keep grid rectangular
        html += "<td></td>";
        c += 1;
      }
    }
    html += "</tr>";
  }
  html += "</table>";
  return html;
}

function otslToMarkdownTable(t) {
  let src = String(t || "");

  // 1) Keep any well-formed HTML tables the model already emitted
  const htmlTables = [];
  src = src.replace(/<table[\s\S]*?<\/table>/gi, function (m) {
    htmlTables.push(m);
    return "\u0000HT" + (htmlTables.length - 1) + "\u0000";
  });

  // 2) Only convert contiguous OTSL regions, leave surrounding prose alone.
  //    A region = one or more consecutive lines that contain a cell tag.
  const cellTag = /<(?:fcel|ecel|lcel|ucel)>/i;
  const nlTag = /<nl\s*\/?>/i;
  const isOtslLine = function (line) {
    return cellTag.test(line) || nlTag.test(line);
  };

  const lines = src.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    if (!isOtslLine(lines[i])) {
      out.push(lines[i]);
      i++;
      continue;
    }
    // gather the whole OTSL block
    const block = [];
    while (i < lines.length && isOtslLine(lines[i])) {
      block.push(lines[i]);
      i++;
    }
    const blockText = block.join("\n");
    const tbl = parseOtsl(blockText);
    if (tbl) {
      const id = htmlTables.length;
      htmlTables.push(tbl);
      out.push("\u0000HT" + id + "\u0000");
    } else {
      // fallback: keep raw
      out.push(blockText);
    }
  }

  return out.join("\n").replace(/\u0000HT(\d+)\u0000/g, function (_, i) {
    return htmlTables[+i] || "";
  });
}

function unescapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'").replace(/&amp;/g, "&");
}

function inlineTex(s) {
  // safety: if entities leaked in, restore before KaTeX
  const tex = unescapeHtml(s);
  if (typeof katex === "undefined") return '<span class="math">' + escapeHtml(tex) + "</span>";
  try {
    return katex.renderToString(tex, { displayMode: false, throwOnError: false, output: "html" });
  } catch (e) {
    return '<span class="math">' + escapeHtml(tex) + "</span>";
  }
}

/**
 * 修复 OCR 常见 LaTeX 污染，让 KaTeX 可解析：
 *   \] 单独成行 + (5)  → 合并进 math 块为 \tag{5}
 *   合法的 \{ \}、\left\{ \right\} 和分组花括号保持原样
 */
function sanitizeLatex(t) {
  let s = String(t || "");
  // Escaped braces (including \{top\}) can be valid literal notation.
  // Preserve them rather than guessing that the OCR intended grouping/commands.
  // Merge equation number that OCR put on the next line after \]
  //    "...\]\n\n(5)\n"  →  "...\\tag{5}\n\]\n"
  s = s.replace(/\\]\s*\n+\s*\((\d{1,3})\)\s*\n/g, "\\tag{$1}\n\\]\n");
  return s;
}

// Rebuild only table markup. Model output must never become executable HTML.
function sanitizeTable(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  const allowed = new Set(["TABLE", "THEAD", "TBODY", "TFOOT", "TR", "TH", "TD", "CAPTION", "BR", "B", "STRONG", "EM", "I", "SUB", "SUP"]);
  function clean(node) {
    if (node.nodeType === 3) return escapeHtml(node.textContent);
    if (node.nodeType !== 1) return "";
    if (!allowed.has(node.tagName)) return escapeHtml(node.textContent);
    const tag = node.tagName.toLowerCase();
    let attrs = "";
    if (tag === "td" || tag === "th") ["colspan", "rowspan"].forEach(key => {
      const value = node.getAttribute(key);
      if (/^[1-9][0-9]{0,2}$/.test(value || "")) attrs += " " + key + '=\"' + value + '\"';
    });
    if (tag === "br") return "<br>";
    return "<" + tag + attrs + ">" + Array.from(node.childNodes).map(clean).join("") + "</" + tag + ">";
  }
  return Array.from(template.content.childNodes).map(clean).join("");
}

function renderMarkdown(md) {
  md = sanitizeLatex(md);
  let src = String(md || "");
  if (/<(?:fcel|ecel|lcel|ucel|nl)\b/i.test(src)) src = otslToMarkdownTable(src);

  const fenced = [];
  src = src.replace(/```([\w-]*)\n([\s\S]*?)```/g, function (_, lang, code) {
    fenced.push(code);
    return "\u0000FENCE" + (fenced.length - 1) + "\u0000";
  });

  const htmls = [];
  src = src.replace(/(<table[\s\S]*?<\/table>)/gi, function (m) {
    htmls.push(sanitizeTable(m));
    return "\u0000HTML" + (htmls.length - 1) + "\u0000";
  });

  // display math
  const mathBlocks = [];
  src = src.replace(/\\\[([\s\S]*?)\\\]/g, function (_, b) {
    mathBlocks.push(b.trim());
    return "\n\u0000MATH" + (mathBlocks.length - 1) + "\u0000\n";
  });
  src = src.replace(/\$\$([\s\S]*?)\$\$/g, function (_, b) {
    mathBlocks.push(b.trim());
    return "\n\u0000MATH" + (mathBlocks.length - 1) + "\u0000\n";
  });

  function inline(s) {
    // Extract math FIRST so < > & are not HTML-escaped before KaTeX.
    const mathBits = [];
    let t = String(s == null ? "" : s);
    t = t.replace(/\\\(([\s\S]*?)\\\)/g, function (_, tex) {
      mathBits.push(inlineTex(tex));
      return "\u0000M" + (mathBits.length - 1) + "\u0000";
    });
    t = t.replace(/\$([^$\n]+)\$/g, function (_, tex) {
      mathBits.push(inlineTex(tex));
      return "\u0000M" + (mathBits.length - 1) + "\u0000";
    });
    t = escapeHtml(t);
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // restore rendered math (already HTML-safe from KaTeX)
    t = t.replace(/\u0000M(\d+)\u0000/g, function (_, i) { return mathBits[+i] || ""; });
    return t;
  }

  const lines = src.split(/\r?\n/);
  const out = [];
  let i = 0, inTable = false, trows = [], list = null;

  function flushList() { if (list) { out.push(list === "ul" ? "</ul>" : "</ol>"); list = null; } }
  function flushTable() {
    if (!inTable) return;
    if (trows.length) {
      const head = trows[0];
      const body = trows.slice(1).filter(function (r) {
        return !r.every(function (c) { return /^:?-{2,}:?$/.test(c.trim()); });
      });
      let html = "<table><thead><tr>";
      head.forEach(function (c) { html += "<th>" + inline(c.trim()) + "</th>"; });
      html += "</tr></thead><tbody>";
      body.forEach(function (r) {
        html += "<tr>";
        r.forEach(function (c) { html += "<td>" + inline(c.trim()) + "</td>"; });
        html += "</tr>";
      });
      out.push(html + "</tbody></table>");
    }
    trows = []; inTable = false;
  }

  while (i < lines.length) {
    const line = lines[i];

    const mathM = line.match(/^\s*\u0000MATH(\d+)\u0000\s*$/);
    if (mathM) {
      flushList(); flushTable();
      const tex = unescapeHtml(mathBlocks[+mathM[1]] || "");
      let rendered;
      if (typeof katex !== "undefined") {
        try {
          rendered = katex.renderToString(tex, { displayMode: true, throwOnError: false, output: "html" });
        } catch (e) { rendered = "<code>" + escapeHtml(tex) + "</code>"; }
      } else {
        rendered = "<code>" + escapeHtml(tex) + "</code>";
      }
      out.push('<div class="math-block">' + rendered + "</div>");
      i++; continue;
    }

    const fenceM = line.match(/^\u0000FENCE(\d+)\u0000$/);
    if (fenceM) {
      flushList(); flushTable();
      out.push("<pre><code>" + escapeHtml(fenced[+fenceM[1]].replace(/\n$/, "")) + "</code></pre>");
      i++; continue;
    }
    const htmlM = line.match(/^\u0000HTML(\d+)\u0000$/);
    if (htmlM) {
      flushList(); flushTable();
      out.push(htmls[+htmlM[1]]);
      i++; continue;
    }

    if (/^\s*\|.*\|\s*$/.test(line)) {
      flushList(); inTable = true;
      trows.push(line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|"));
      i++; continue;
    } else if (inTable) flushTable();

    if (!line.trim()) { flushList(); i++; continue; }

    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flushList();
      const lv = Math.min(h[1].length, 4);
      out.push("<h" + lv + ">" + inline(h[2]) + "</h" + lv + ">");
      i++; continue;
    }
    if (/^>\s?/.test(line)) {
      flushList();
      out.push("<blockquote>" + inline(line.replace(/^>\s?/, "")) + "</blockquote>");
      i++; continue;
    }
    if (/^[-*+]\s+/.test(line)) {
      if (list !== "ul") { flushList(); list = "ul"; out.push("<ul>"); }
      out.push("<li>" + inline(line.replace(/^[-*+]\s+/, "")) + "</li>");
      i++; continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      if (list !== "ol") { flushList(); list = "ol"; out.push("<ol>"); }
      out.push("<li>" + inline(line.replace(/^\d+\.\s+/, "")) + "</li>");
      i++; continue;
    }
    flushList();
    out.push("<p>" + inline(line) + "</p>");
    i++;
  }
  flushList(); flushTable();
  return out.join("\n");
}

function showError(msg, values = {}) {
  const b = $("errorBanner");
  uiText(b, () => t(msg, values));
  b.classList.add("show");

}
function clearError() {
  $("errorBanner").classList.remove("show");
}

function stopTyping() {
  if (state.typingTimer) { clearTimeout(state.typingTimer); state.typingTimer = null; }
  $("resText").classList.remove("typing");
}

/**
 * Typewriter over markdown source, then swap to rendered HTML.
 * Looks more "live" for demo; final state is fully rendered (KaTeX etc).
 */
function typeResult(rawText) {
  stopTyping();
  const el = $("resText");
  const full = String(rawText || "");
  recordResult(full, null);
  el.style.display = "block";
  $("resEmpty").style.display = "none";
  uiText("badgeRight", () => t("TYPING"));
  $("badgeRight").className = "badge on";

  if (reduceMotion) {
    el.classList.remove("typing");
    el.classList.add("markdown");
    el.innerHTML = renderMarkdown(full);
    state.typedDone = true;
    uiText("badgeRight", () => t("DONE"));
    return;
  }

  el.classList.add("typing");
  el.textContent = "";
  state.typedDone = false;

  // chunk size: finish in ~2.5–4s
  const total = full.length;
  const steps = Math.min(140, Math.max(30, Math.ceil(total / 18)));
  const chunk = Math.max(1, Math.ceil(total / steps));
  let i = 0;

  function tick() {
    i = Math.min(total, i + chunk);
    el.textContent = full.slice(0, i);
    const pane = $("resPane");
    pane.scrollTop = pane.scrollHeight;
    if (i < total) {
      state.typingTimer = setTimeout(tick, 22);
    } else {
      // swap to rendered HTML
      el.classList.remove("typing");
      el.classList.add("markdown");
      el.innerHTML = renderMarkdown(full);
      state.typedDone = true;
      uiText("badgeRight", () => t("DONE"));
      pane.scrollTop = 0;
    }
  }
  tick();
}

function resetResult() {
  paintPills();
  stopTyping();
  state.typedDone = false;
  state.lastText = "";
  state.lastMeta = null;
  $("btnCopy").disabled = true;
  $("btnDownload").disabled = true;
  $("resText").classList.remove("source-view", "typing");
  $("resText").style.display = "none";
  $("resText").innerHTML = "";
  $("resEmpty").style.display = "flex";
  uiText("badgeRight", () => t("待解析"));
  $("badgeRight").className = "badge";
  uiText("sideHint", () => t("点击画面开始解析"));
  const hint = $("clickHint");
  if (hint) { uiText(hint, () => t("点击图片 · 开始解析")); hint.style.opacity = ""; }
}

function paintPills() {
  uiText("pillLat", () => t("耗时 —"));
  uiText("pillTok", () => t("TOKENS —"));
  uiText("pillModeText", () => t("待解析"));
  setLed("off");
}

function renderThumbs() {
  const host = $("thumbs");
  host.innerHTML = "";

  // upload tile first
  const up = document.createElement("button");
  up.type = "button";
  up.className = "film-upload" + (state.activeId === "custom" ? " active" : "");
  up.innerHTML =
    '<span class="plus">+</span>' +
    "<span>" + t("上传图像 / PDF") + "</span>" +
    '<span class="sub">PNG · JPG · WEBP · PDF</span>';
  up.onclick = function () { $("imgUpload").click(); };
  host.appendChild(up);

  state.items.forEach(function (item) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "film" + (state.activeId === item.id ? " active" : "");
    b.dataset.id = item.id;
    b.innerHTML =
      '<img src="' + item.image + '" alt="' + escapeHtml(t(item.title)) + '" loading="lazy" />' +
      '<span class="cap">' + escapeHtml(t(item.title)) + '<small>' + escapeHtml(t(item.desc || '')) + '</small></span>';
    b.onclick = function () { selectItem(item.id); };
    host.appendChild(b);
  });
}

$("imgUpload").onchange = function () {
  const f = this.files && this.files[0];
  if (!f || state.busy) return;
  if (f.size > 55 * 1024 * 1024) { showError("文件超过 55 MiB，请压缩或拆分后重试。"); this.value = ""; return; }
  const isPdf = f.type === "application/pdf" || /\.pdf$/i.test(f.name);
  if (!isPdf && (!f.type || f.type.indexOf("image/") !== 0)) {
    showError("请选择图像或 PDF 文件");
    return;
  }
  clearError();
  if (state.pdfUrl) URL.revokeObjectURL(state.pdfUrl);
  if (isPdf) {
    state.pdfUrl = URL.createObjectURL(f);
    state.custom = { name: f.name, file: f, isPdf: true, size: f.size };
    state.activeId = "custom";
    document.querySelectorAll(".film").forEach(function (el) { el.classList.remove("active"); });
    document.querySelectorAll(".film-upload").forEach(function (el) { el.classList.add("active"); });
    uiText("stageTitle", () => f.name);
    $("stageImg").hidden = true;
    $("stagePdf").hidden = false;
    $("stagePdf").src = state.pdfUrl;
    uiText("badgeLeft", () => t("PDF"));
    uiText("pillLat", () => t("—")); uiText("pillTok", () => t("—"));
    uiText("pillModeText", () => t("PDF · LOCAL")); setLed("on");
    state.task = "document";
    document.querySelectorAll("#taskSeg button").forEach(function (b) { b.classList.toggle("active", b.dataset.task === "document"); });
    updateTaskInputs(); resetResult();
    uiText("sideHint", () => t("点击画面解析 PDF 全部页面"));
    this.value = "";
    return;
  }
  const r = new FileReader();
  r.onload = function () {
    if (state.busy) return;
    state.custom = { name: f.name, dataUrl: r.result, size: f.size, isPdf: false };
    state.activeId = "custom";
    document.querySelectorAll(".film").forEach(function (el) { el.classList.remove("active"); });
    document.querySelectorAll(".film-upload").forEach(function (el) { el.classList.add("active"); });
    uiText("stageTitle", () => f.name);
    $("stageImg").hidden = false;
    $("stagePdf").hidden = true; $("stagePdf").removeAttribute("src");
    $("stageImg").src = r.result;
    $("stageImg").alt = f.name;
    uiText("pillLat", () => t("—"));
    uiText("pillTok", () => t("—"));
    uiText("pillModeText", () => t("LIVE")); setLed("on");
    uiText("badgeLeft", () => t("IMAGE"));
    updateTaskInputs();
    resetResult();
    uiText("sideHint", () => t("点击画面开始实时解析"));
  };
  r.readAsDataURL(f);
  this.value = "";
};

function itemById(id) {
  return state.items.filter(function (x) { return x.id === id; })[0];
}

function selectItem(id) {
  const item = itemById(id);
  if (!item || state.busy) return;
  uiText("stageTitle", () => t(item.title));
  clearError();
  stopTyping();
  state.activeId = id;
  state.custom = null;
  if (state.pdfUrl) { URL.revokeObjectURL(state.pdfUrl); state.pdfUrl = null; }
  document.querySelectorAll(".film-upload").forEach(function (el) { el.classList.remove("active"); });
  document.querySelectorAll(".film").forEach(function (el) {
    el.classList.toggle("active", el.dataset.id === id);
  });
  $("stageImg").src = item.image;
  $("stageImg").hidden = false;
  $("stagePdf").hidden = true; $("stagePdf").removeAttribute("src");
  uiText("badgeLeft", () => t("IMAGE"));
  $("stageImg").alt = t(item.title);
  paintPills(item);
  updateTaskInputs();
  resetResult();

}

/* Live OCR for custom upload */
const LIVE_PROMPT =
  "Extract all information from the main body of the document image and represent it in markdown format, " +
  "ignoring headers and footers. Tables should be expressed in OTSL format, formulas in the document should " +
  "be represented using LATEX format, and the parsing should be organized according to the reading order.";

/** The browser receives shared monitoring snapshots and one final result. */
async function runImageOcr(dataUrl, prompt, task, mode) {
  const c = cfg();
  const body = {
    image_base64: dataUrl, prompt: prompt || LIVE_PROMPT, task: task || "document",
    fields: $("kieFields").value || "", question: $("vqaQuestion").value || "",
    max_tokens: task === "vqa" ? 1024 : Math.max(1, Math.min(16384, c.max_tokens || 4096)),
  };
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), 300000);
  let reader;
  try {
    const path = mode === "region" ? "/api/parse/regions/stream" : "/api/ocr/stream";
    const res = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body), signal: ac.signal});
    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.error || ("HTTP " + res.status));
    }
    reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", result = null;
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      buffer += decoder.decode(part.value, {stream: true});
      const events = buffer.split("\n\n");
      buffer = events.pop();
      for (const block of events) {
        let event = "", data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const value = JSON.parse(data);
        if (event === "progress") showRuntime(value);
        if (event === "error") throw new Error(value.error || "OCR failed");
        if (event === "done") result = value;
      }
    }
    if (!result) throw new Error("推理流中断，未收到完成事件");
    return result;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("解析超时（300s）");
    throw error;
  } finally {
    clearTimeout(timer);
    if (reader) await reader.cancel().catch(() => {});
  }
}

/** Resolve a data URL for whatever is on stage (custom or showcase). */
async function stageImageDataUrl() {
  if (state.custom && state.custom.dataUrl) return state.custom.dataUrl;
  if (state.custom && state.custom.isPdf) throw new Error("PDF 不能用于图像任务");
  const item = itemById(state.activeId);
  if (!item || !item.image) throw new Error("无可用图像");
  const res = await fetch(item.image);
  if (!res.ok) throw new Error("图像加载失败");
  return await blobToDataUrl(await res.blob());
}

function fileToDataUrl(file) { return blobToDataUrl(file); }

async function runPdfParse(file) {
  if (file.size > 55 * 1024 * 1024) throw new Error("PDF 超过 55 MiB；请拆分文件后重试。");
  $("taskProgress").hidden = false;
  $("taskProgressFill").style.width = "0%";
  const c = cfg();
  const pdfData = await fileToDataUrl(file);
  const start = await fetch("/api/book", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pdf_base64: pdfData, pdf_name: file.name, mode: state.mode,
      prompt: $("docPrompt").value || LIVE_PROMPT, max_tokens: c.max_tokens || 4096,
      concurrency: state.mode === "region" ? 2 : 4, dpi: 150 }),
  });
  const started = await start.json();
  if (!start.ok || started.error) throw new Error(started.error || ("HTTP " + start.status));
  let status;
  while (true) {
    await new Promise(function (resolve) { setTimeout(resolve, 1000); });
    const res = await fetch("/api/book/" + encodeURIComponent(started.id));
    status = await res.json();
    if (status.monitor) showRuntime(status.monitor);
    if (!res.ok) throw new Error(status.error || "读取 PDF 任务状态失败");
    const total = status.total || 0;
    const done = status.done || 0;
    const percent = total ? Math.round(done * 100 / total) : 0;
    $("taskProgressFill").style.width = percent + "%";
    uiText("stageHint", () => t(status.phase === "render" ? "渲染 PDF 页面" : status.phase === "ocr" ? "识别 PDF 页面" : status.status === "done" ? "已完成" : "PDF 处理中") + (total ? " · " + percent + "%" : ""));
    uiText("sideHint", () => t("PDF {done}/{total} 页 · {mode}", {done, total, mode: t(state.mode === "region" ? "区域模式" : "整页模式")}));
    uiText("pillModeText", () => t("PDF · " + done + "/" + (total || "…")));
    if (["done", "error", "cancelled"].includes(status.status)) break;
  }
  if (status.status !== "done" && !status.has_markdown) throw new Error(status.error || "PDF 解析未完成");
  $("taskProgressFill").style.width = "100%";
  const rawRes = await fetch("/api/book/" + encodeURIComponent(started.id) + "/raw");
  const raw = await rawRes.json();
  if (!rawRes.ok || raw.error) throw new Error(raw.error || "读取解析结果失败");
  return { content: raw.markdown || "", latency_ms: raw.elapsed_ms || status.elapsed_ms || 0,
    usage: status.usage || {}, model: status.model || "SeerRay-Lab/Xiaomi-OCR-0", pages: status.total || 0, warning: status.status !== "done" ? (status.error || "部分页面未完成") : "" };
}

$("srcPane").onclick = async function () {
  if (state.busy) return;
  const item = itemById(state.activeId);

  const isPdf = !!(state.custom && state.custom.isPdf);
  const task = state.task;
  const useRegion = task === "document" && state.mode === "region";

  if (!item && !state.custom) { showError("请先选择样例或上传文件。"); return; }
  if (task === "vqa" && !$("vqaQuestion").value.trim()) { showError("请先输入你的问题。"); $("vqaQuestion").focus(); return; }
  clearError();
  resetResult();
  setBusy(true);
  resetRuntimeProgress();
  firePulse();
  $("imgWrap").classList.add("thinking");
  const hint = $("clickHint");
  if (hint) { uiText(hint, () => t(isPdf ? "PDF 解析中…" : (useRegion ? "区域解析中…" : "思考中…"))); hint.style.opacity = "1"; }
  uiText("sideHint", () => t(isPdf ? "PDF 页面渲染与解析中" : (useRegion ? "版面检测与区域识别" : "识别中…")));
  uiText("stageHint", () => t(isPdf ? "PDF…" : (useRegion ? "REGION…" : "OCR…")));
  uiText("badgeRight", () => t("LIVE"));
  $("badgeRight").className = "badge on";

  stopTyping();
  const el = $("resText");
  el.classList.remove("markdown", "source-view");
  el.classList.add("typing");
  el.style.display = "block";
  el.textContent = "";
  $("resEmpty").style.display = "none";

  try {
    let data;
    if (isPdf) {
      data = await runPdfParse(state.custom.file);
      el.classList.remove("typing");
      el.classList.add("markdown");
      recordResult(data.content, data);
      if (data.warning) showError("部分 PDF 页面失败，已保留可下载的结果：{error}", {error: data.warning});
      el.innerHTML = renderMarkdown(normalizeOcrOutput(data.content));
      uiText("pillLat", () => "LATENCY " + ((data.latency_ms || 0) / 1000).toFixed(2) + "s");
      uiText("pillTok", () => "TOKENS " + ((data.usage && data.usage.completion_tokens) || "—"));
      uiText("pillModeText", () => t("PDF · {count} 页", {count: data.pages})); setLed("on");
      uiText("sideHint", () => t("点击画面可重新解析 PDF"));
      uiText("stageHint", () => t("PDF · DONE"));
      if (hint) { uiText(hint, () => t("点击解析")); hint.style.opacity = ""; }
    } else {
      const dataUrl = await stageImageDataUrl();
      if (useRegion) {
      data = await runImageOcr(dataUrl, "", "document", "region");
      const regions = data.regions || [];
      const count = regions.length;
      const okN = regions.filter(region => region.ok).length;
      el.classList.remove("typing");
      el.classList.add("markdown");
      recordResult(data.content, data);
      el.innerHTML = renderMarkdown(data.content || "");
      uiText("pillLat", () => "LATENCY " + ((data.latency_ms || 0) / 1000).toFixed(2) + "s");
      uiText("pillTok", () => "TOKENS " + ((data.usage && data.usage.completion_tokens) || "—"));
      uiText("pillModeText", () => t("REGION · " + okN + "/" + count));
      setLed("on");
      uiText("sideHint", () => t("版面检测 {ms}ms · 并发 {count}", {ms: data.layout_ms || 0, count: data.concurrency || 4}));
      uiText("stageHint", () => t("REGION · " + count + " · " + okN + " ok"));
      if (hint) { uiText(hint, () => t("点击解析")); hint.style.opacity = ""; }
      } else {
      const chosenPrompt = task === "document" ? ($("docPrompt").value || LIVE_PROMPT) : "";
      data = await runImageOcr(dataUrl, chosenPrompt, task, "page");
      let finalContent = (data.content || "").trim();
      el.classList.remove("typing");
      el.classList.add("markdown");
      recordResult(finalContent, data);
      el.innerHTML = task === "kie"
        ? "<pre><code>" + escapeHtml(finalContent) + "</code></pre>"
        : renderMarkdown(normalizeOcrOutput(finalContent));
      const lat = (data.latency_ms / 1000).toFixed(2) + "s";
      const u = data.usage || {};
      uiText("pillLat", () => "LATENCY " + lat);
      uiText("pillTok", () => "TOKENS " + (u.completion_tokens != null ? u.completion_tokens : "—"));
      uiText("pillModeText", () => t(task === "document" ? "LIVE" : (task === "kie" ? "KIE · LIVE" : "VQA · LIVE"))); setLed("on");
      uiText("sideHint", () => t("点击画面可重新解析"));
      uiText("stageHint", () => t((task === "document" ? "PAGE" : task.toUpperCase()) + " · LIVE"));
      if (hint) { uiText(hint, () => t("点击解析")); hint.style.opacity = ""; }
      }
    }
  } catch (e) {
      el.classList.remove("typing");
      el.textContent = "";
      $("resEmpty").style.display = "flex";
      showError(e.message || String(e));
      markRuntimeError();
      uiText("badgeRight", () => t("ERROR"));
      $("badgeRight").className = "badge";
      setLed("err");
      uiText("sideHint", () => t(isPdf ? "确认 PDF 依赖与本机模型已就绪" : "确认本机模型和区域识别依赖已启动"));
      uiText("stageHint", () => t("ERROR"));
      if (hint) { uiText(hint, () => t("点击重试")); hint.style.opacity = ""; }
  } finally {
    setBusy(false);
    if (state.lastText) { uiText("badgeRight", () => t(state.lastMeta && state.lastMeta.warning ? "部分完成" : "已完成")); paintResult(); }
    $("imgWrap").classList.remove("thinking");
    if (isPdf) setTimeout(function () { $("taskProgress").hidden = true; }, 1100);
  }
};

// Runtime snapshots come from the shared monitor. Idle pages do not poll devices.
let runtimeState = null, runtimeReceivedAt = 0;
const STAGE_LABELS = {
  request: "任务", inspect_runtime: "读取设备信息", prepare_image: "准备图片", prepare_pdf: "读取 PDF",
  render_pdf: "渲染 PDF 页面", import: "导入版面组件", load: "加载版面模型", queue: "等待可用资源",
  inference: "推理", assemble: "整理结果", assemble_pdf: "合并 PDF", heartbeat: "处理中",
};
function deviceLabel(info) {
  if (!info) return t("等待设备信息");
  if (info.display_device === "not_loaded") return t("尚未加载");
  if (!info.display_device || info.display_device === "unknown") return t("未知设备");
  return info.display_device;
}
function showRuntime(state) {
  if (!state) return;
  if (state !== runtimeState) {
    runtimeState = state;
    runtimeReceivedAt = performance.now();
  }
  const components = state.components || {};
  for (const name of ["layout", "ocr"]) {
    const info = components[name];
    if (!info) continue;
    const label = deviceLabel(info);
    uiText("runtime" + (name === "layout" ? "Layout" : "Ocr"), () => deviceLabel(info));
    const detail = [info.model, info.source];
    if (info.requested_device) detail.push("requested: " + info.requested_device);
    if (info.cache_age_ms != null) detail.push("cache: " + (info.cache_age_ms / 1000).toFixed(1) + "s");
    if (info.gpu_processes) info.gpu_processes.forEach(p => detail.push("PID " + p.pid + " · GPU " + p.gpu_index + " · " + p.used_memory_mib + " MiB"));
    if (info.note) detail.push(info.note);
    if (info.errors) detail.push(JSON.stringify(info.errors));
    $("runtime" + (name === "layout" ? "Layout" : "Ocr")).title = detail.filter(Boolean).join("\n");
    uiText("runtime" + (name === "layout" ? "LayoutDetail" : "OcrDetail"), () => {
      if (name === "layout" && info.loaded && info.requested_device && info.requested_device !== "auto" && info.requested_device !== info.device)
        return t("请求 {requested}，实际 {actual}", {requested: info.requested_device, actual: label});
      if (info.device === "unknown") return info.note || t("尚无实际设备依据");
      return name === "layout" ? "PP-DocLayoutV3" : (info.backend || "Xiaomi-OCR-0");
    });
  }
  const counts = Object.entries(state.progress_counts || {});
  const regions = counts.filter(([key]) => key.startsWith("ocr/regions"));
  const rows = counts.filter(([key]) => !key.startsWith("ocr/regions")).concat(regions.slice(-8));
  const fragment = document.createDocumentFragment();
  for (const [key, count] of rows) {
    const page = key.match(/page=(\d+)/);
    const title = key.includes("rendered_pages") ? t("已渲染页面") : key.includes("recognized_pages") ? t("已处理页面")
      : key.includes("failed_pages") ? t("失败页面") : key.includes("blank_pages") ? t("空白页面")
      : page ? t("第 {page} 页区域", {page: page[1]}) : t("区域");
    const row = document.createElement("div"), name = document.createElement("span"), value = document.createElement("b");
    name.textContent = title;
    value.textContent = count.completed + " / " + count.total;
    row.append(name, value);
    fragment.appendChild(row);
  }
  if (regions.length > 8) {
    const more = document.createElement("span");
    more.className = "runtime-more";
    more.textContent = t("另 {count} 页区域已记录", {count: regions.length - 8});
    fragment.appendChild(more);
  }
  $("runtimeCounts").replaceChildren(fragment);
  $("runtimeTaskId").textContent = state.request_id || "";
  $("runtimeEvidence").textContent = ["layout", "ocr"].map(name => {
    const info = components[name];
    if (!info) return "";
    return (name === "layout" ? t("版面检测") : "OCR") + ": " + (info.source || "—")
      + (info.cache_age_ms != null ? " · " + t("缓存 {seconds}s", {seconds: (info.cache_age_ms / 1000).toFixed(0)}) : "");
  }).filter(Boolean).join("\n");
  paintRuntimeStage();
}
function paintRuntimeStage() {
  const state = runtimeState;
  if (!state) return;
  const summary = state.summary || {};
  const running = state.status === "running";
  const delta = running ? performance.now() - runtimeReceivedAt : 0;
  const seconds = (((state.elapsed_ms || 0) + delta) / 1000).toFixed(1);
  const phase = summary.phase;
  const terminal = {completed: "已完成", partial: "部分完成", failed: "失败", cancelled: "已取消"};
  const phaseLabel = phase === "inference" ? (summary.component === "layout" ? "版面检测中" : "文字识别中") : STAGE_LABELS[phase] || "处理中";
  const label = terminal[state.status] || (!state.tool ? (running ? "任务启动中…" : "准备就绪") : phaseLabel);
  $("runtimePanel").dataset.status = state.status || "idle";
  uiText("runtimeStageLabel", () => t(label));
  $("runtimeElapsed").textContent = state.tool ? seconds + "s" : "";
  const pages = summary.pages, regions = summary.regions;
  const count = pages || regions;
  const countLabel = count ? t(pages ? "页面 {done}/{total}" : "区域 {done}/{total}", {done: count.completed, total: count.total}) : "";
  $("runtimeCountLabel").textContent = countLabel;
  const bar = $("runtimeProgress");
  bar.hidden = !count || count.total <= 0;
  bar.max = count ? count.total || 1 : 1;
  bar.value = count ? count.completed : 0;
  bar.setAttribute("aria-valuetext", countLabel);
  const activePages = summary.active_pages || [];
  const context = [];
  if (running && activePages.length) context.push(t("正在处理第 {pages} 页", {pages: activePages.join(", ")}));
  if (pages && regions) context.push(t("已完成 {count} 个区域", {count: regions.completed}));
  if (pages && pages.blank) context.push(t("{count} 页空白", {count: pages.blank}));
  $("runtimeContext").textContent = context.join(" · ");
  $("runtimeContext").title = context.join(" · ");
  const issue = state.error || state.warning || "";
  $("runtimeIssue").textContent = issue;
  $("runtimeIssue").hidden = !issue;
  const fragment = document.createDocumentFragment();
  (running ? state.active_stages || [] : []).slice(0, 4).forEach(s => {
    const line = document.createElement("span");
    line.textContent = [s.component === "layout" ? t("版面检测") : s.component === "ocr" ? "OCR" : "",
      t(STAGE_LABELS[s.stage] || s.stage), s.page ? t("第 {page} 页", {page: s.page}) : "",
      s.region ? t("区域 {region}", {region: s.region}) : "", ((s.elapsed_ms + delta) / 1000).toFixed(1) + "s"].filter(Boolean).join(" · ");
    fragment.appendChild(line);
  });
  $("runtimeStages").replaceChildren(fragment);
}
function resetRuntimeProgress() {
  showRuntime({components: runtimeState ? runtimeState.components : {}, status: "running"});
}
function markRuntimeError() {
  if (runtimeState) {
    if (runtimeState.status !== "cancelled") runtimeState.status = "failed";
    paintRuntimeStage();
  } else {
    showRuntime({status: "failed"});
  }
}
setInterval(() => { if (!document.hidden && runtimeState && runtimeState.status === "running") paintRuntimeStage(); }, 1000);
document.addEventListener("click", event => {
  if (!$("runtimeDetails").contains(event.target)) $("runtimeDetails").open = false;
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") $("runtimeDetails").open = false;
});

/* load */
function loadShowcase() {
  return fetch("/showcase.json")
    .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then(function (data) {
      state.items = (data && data.items) || [];
      renderThumbs();
      uiText("sampleCount", () => t("{count} 个样例", {count: state.items.length}));
      const first = state.items[0];
      if (first) selectItem(first.id);
    })
    .catch(function (e) {
      uiText("stageTitle", () => t("Showcase 加载失败"));
      showError(e.message);
    });
}

(function init() {
  if (!localStorage.getItem(LS_KEY)) localStorage.setItem(LS_KEY, JSON.stringify(DEFAULT_CFG));
  loadShowcase();
  checkHealth();
  fetch("/api/runtime").then(r => r.ok ? r.json() : Promise.reject()).then(info => {
    if (!runtimeState) showRuntime(info);
  }).catch(() => {});
  setInterval(() => { if (!document.hidden) checkHealth(); }, 30000);
})();

function blobToDataUrl(blob) {
  return new Promise(function (resolve, reject) {
    const r = new FileReader();
    r.onload = function () { resolve(r.result); };
    r.onerror = function () { reject(r.error || new Error("FileReader failed")); };
    r.readAsDataURL(blob);
  });
}

/** Convert OTSL markers into markdown tables when present. */
function normalizeOcrOutput(text) {
  const t = String(text || "");
  if (/<(?:fcel|ecel|lcel|ucel)\b/i.test(t) || /<nl\s*\/?>/i.test(t)) {
    return otslToMarkdownTable(t);
  }
  return t;
}


function setBusy(busy) {
  state.busy = busy;
  document.querySelectorAll("#thumbs button, #taskSeg button, #modeSeg button, #btnUpload, #btnRun, #btnSettings, #docPrompt, #kieFields, #vqaQuestion, #imgUpload").forEach(el => { el.disabled = busy; });
  uiText("btnRun", () => t(busy ? "解析中…" : "开始解析 ↗"));
  $("srcPane").setAttribute("aria-busy", String(busy));
  if (!busy) document.querySelectorAll("#modeSeg button").forEach(el => { el.disabled = state.task !== "document"; });
}
$("btnRun").onclick = () => $("srcPane").onclick();
$("btnUpload").onclick = () => $("imgUpload").click();
$("btnCopy").onclick = async function () {
  try {
    const field = document.createElement("textarea");
    field.value = state.lastText; field.className = "sr-only";
    document.body.appendChild(field); field.select();
    let copied = false;
    try { copied = document.execCommand("copy"); } finally { field.remove(); }
    if (!copied) {
      await Promise.race([
        navigator.clipboard.writeText(state.lastText),
        new Promise((_, reject) => setTimeout(() => reject(new Error("Clipboard timeout")), 2500))
      ]);
    }
    uiText(this, () => t("已复制"));
    setTimeout(() => { uiText(this, () => t("复制")); }, 1500);
  } catch (_) { showError("无法访问剪贴板，请切换原文视图复制，或下载结果。"); }
};
$("btnDownload").onclick = () => {
  if (!state.lastText) return;
  const form = document.createElement("form");
  form.method = "POST"; form.action = "/api/export"; form.hidden = true;
  for (const [name, value] of Object.entries({content: state.lastText, format: state.lastTask === "kie" ? "json" : "md"})) {
    const input = document.createElement("input"); input.type = "hidden";
    input.name = name; input.value = value; form.appendChild(input);
  }
  document.body.appendChild(form); form.submit(); form.remove();
};
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("drawer").hidden) closeDrawer();
  if (e.key === "Tab" && !$("drawer").hidden) {
    const nodes = [...$("drawer").querySelectorAll("button,input")];
    const first = nodes[0], last = nodes[nodes.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
});
function updateRegionDependencyStatus(region) {
  const available = region && typeof region.available === "boolean";
  $("regionInstallHint").hidden = !available || region.available;
  uiText("regionDependencyStatus", () => available
    ? (region.available ? t("区域解析依赖已安装。")
      : t("缺少区域解析依赖：{missing}。", {missing: (region.missing || []).join(", ")}))
    : t("无法确认区域解析依赖，请检查服务连接。"));
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("服务检查失败");
    const health = await res.json();
    updateRegionDependencyStatus(health.region);
    $("connDot").classList.toggle("off", !health.ready);
    uiText("statusText", () => t(health.ready ? "模型已连接" : "模型未连接"));
    $("statusText").title = health.ready ? health.model : health.error;
  } catch (_) {
    updateRegionDependencyStatus(null);
    $("connDot").classList.add("off");
    uiText("statusText", () => t("服务未连接"));
  }
}

// Refresh sample labels without resetting user input, task progress or OCR output.
document.addEventListener("languagechange", () => {
  renderThumbs();
  if (runtimeState) showRuntime(runtimeState);
  if (state.busy) document.querySelectorAll("#thumbs button").forEach(button => { button.disabled = true; });
  const item = itemById(state.activeId);
  if (item && !state.custom) $("stageImg").alt = t(item.title);
  else if (state.custom) $("stageImg").alt = state.custom.name;
});
