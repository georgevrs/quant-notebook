#!/usr/bin/env node
/**
 * render_math.mjs — author LaTeX, ship MathML (no JavaScript in the published pages).
 *
 *   node tools/render_math.mjs site                 render every \( … \) and \[ … \] in place
 *   node tools/render_math.mjs --check site         exit 1 if raw TeX or un-annotated <math> remains
 *   node tools/render_math.mjs --refresh site       re-render every <math> from its TeX annotation
 *   node tools/render_math.mjs --unrender page.html turn <math> back into \( … \) / \[ … \] for editing
 *
 * Paths may be files or directories (every *.html below them).
 *
 * How it works: a raw-text scanner (not a DOM parser — `\(a<b\)` would be read as a <b> tag).
 * It skips <pre>, <code>, <svg>, <math>, <script>, <style>, <title>, <textarea> and comments,
 * so Python regexes in code blocks and SVG labels are never touched. Inside TeX, `\\` is
 * consumed as a pair, so `\\[2pt]` in an aligned block is a line break, not a new display.
 * HTML entities in the TeX are decoded before Temml sees them; Temml escapes the TeX it keeps
 * in <annotation encoding="application/x-tex">, which is what --refresh / --unrender read back.
 * Display math is wrapped in <div class="eq"> so wide equations scroll on phones.
 * Authors write \lt and \gt (or &lt; &gt;) for < and >. Shared macros live in tools/macros.tex.
 */
import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import temml from "temml";

const HERE = dirname(fileURLToPath(import.meta.url));
const MACROS = temml.definePreamble(readFileSync(join(HERE, "macros.tex"), "utf8"));

const SKIP_TAGS = ["pre", "code", "svg", "math", "script", "style", "title", "textarea"];

export function decodeEntities(s) {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&");
}

export function renderTex(tex, display) {
  const opts = { annotate: true, throwOnError: true, displayMode: display, macros: { ...MACROS } };
  const mathml = temml.renderToString(decodeEntities(tex).trim(), opts);
  return display ? `<div class="eq">${mathml}</div>` : mathml;
}

/** Index just past the end of a skipped region starting at `i`, or -1 if none starts here. */
function skipRegion(html, i) {
  if (html.startsWith("<!--", i)) {
    const end = html.indexOf("-->", i + 4);
    return end === -1 ? html.length : end + 3;
  }
  if (html[i] !== "<") return -1;
  for (const tag of SKIP_TAGS) {
    const open = html.slice(i + 1, i + 1 + tag.length).toLowerCase();
    const after = html[i + 1 + tag.length];
    if (open === tag && (after === ">" || after === " " || after === "\n" || after === "\t" || after === "/")) {
      const close = html.toLowerCase().indexOf(`</${tag}>`, i);
      return close === -1 ? html.length : close + tag.length + 3;
    }
  }
  return -1;
}

/**
 * Find TeX spans. Returns [{start, end, tex, display}] with start/end covering the delimiters.
 * Throws on an unterminated delimiter so a typo cannot silently swallow a page.
 */
export function findTex(html) {
  const spans = [];
  let i = 0;
  while (i < html.length) {
    const skipTo = skipRegion(html, i);
    if (skipTo !== -1) { i = skipTo; continue; }
    if (html[i] === "\\") {
      const nx = html[i + 1];
      if (nx === "(" || nx === "[") {
        const display = nx === "[";
        const closer = display ? "]" : ")";
        let j = i + 2;
        let found = -1;
        while (j < html.length) {
          if (html[j] === "\\") {
            if (html[j + 1] === closer) { found = j; break; }
            j += 2; // consume escaped pair, e.g. \\ or \{ or \%
            continue;
          }
          if (html[j] === "<" && /^<\/?(p|div|section|li|td|th|h[1-6]|table|ul|ol)\b/i.test(html.slice(j, j + 10))) {
            break; // ran into block markup: unterminated
          }
          j += 1;
        }
        if (found === -1) {
          const line = html.slice(0, i).split("\n").length;
          throw new Error(`unterminated \\${nx} at line ${line}: ${html.slice(i, i + 60).replace(/\n/g, " ")}`);
        }
        spans.push({ start: i, end: found + 2, tex: html.slice(i + 2, found), display });
        i = found + 2;
        continue;
      }
      i += 2;
      continue;
    }
    i += 1;
  }
  return spans;
}

export function renderHtml(html) {
  const spans = findTex(html);
  if (!spans.length) return html;
  let out = "";
  let last = 0;
  for (const s of spans) {
    out += html.slice(last, s.start);
    try {
      out += renderTex(s.tex, s.display);
    } catch (e) {
      const line = html.slice(0, s.start).split("\n").length;
      throw new Error(`TeX error at line ${line}: ${s.tex.slice(0, 80)}\n  ${e.message}`);
    }
    last = s.end;
  }
  return out + html.slice(last);
}

const MATH_RE = /(<div class="eq">\s*)?<math\b[^>]*>[\s\S]*?<\/math>(\s*<\/div>)?/g;
const ANNOT_RE = /<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/;

/** Replace every rendered <math> by its TeX source with the original delimiters. */
export function unrenderHtml(html) {
  return html.replace(MATH_RE, (whole, openDiv, closeDiv) => {
    const m = whole.match(ANNOT_RE);
    if (!m) throw new Error(`<math> without TeX annotation cannot be unrendered: ${whole.slice(0, 80)}`);
    const tex = m[1]; // stays HTML-escaped: safe inside page text, decoded again on render
    const display = Boolean(openDiv) || /\bdisplay="block"/.test(whole);
    if (display) return `\\[${tex}\\]`;
    return `\\(${tex}\\)`;
  });
}

export function checkHtml(html) {
  const problems = [];
  let spans = [];
  try {
    spans = findTex(html);
  } catch (e) {
    problems.push(e.message);
  }
  for (const s of spans) {
    const line = html.slice(0, s.start).split("\n").length;
    problems.push(`raw TeX at line ${line}: ${html.slice(s.start, Math.min(s.end, s.start + 60))}`);
  }
  for (const m of html.matchAll(/<math\b[\s\S]*?<\/math>/g)) {
    if (!ANNOT_RE.test(m[0])) problems.push(`<math> without TeX annotation: ${m[0].slice(0, 60)}`);
  }
  return problems;
}

function listHtml(paths) {
  const files = [];
  for (const p of paths) {
    const st = statSync(p);
    if (st.isDirectory()) {
      for (const name of readdirSync(p)) {
        if (name === "node_modules" || name.startsWith(".")) continue;
        files.push(...listHtml([join(p, name)]));
      }
    } else if (p.endsWith(".html")) {
      files.push(p);
    }
  }
  return files.sort();
}

function write(path, text) {
  writeFileSync(path, text.replace(/\r\n/g, "\n"), { encoding: "utf8" });
}

function main(argv) {
  const flags = new Set(argv.filter((a) => a.startsWith("--")));
  const paths = argv.filter((a) => !a.startsWith("--"));
  if (!paths.length) {
    console.error("usage: render_math.mjs [--check|--refresh|--unrender] <file-or-dir> …");
    process.exit(2);
  }
  let failed = 0;
  let changed = 0;
  for (const file of listHtml(paths)) {
    const html = readFileSync(file, "utf8");
    try {
      if (flags.has("--check")) {
        const problems = checkHtml(html);
        for (const p of problems) console.log(`FAIL ${file}: ${p}`);
        if (problems.length) failed++;
        continue;
      }
      let next = html;
      if (flags.has("--unrender")) next = unrenderHtml(html);
      else if (flags.has("--refresh")) next = renderHtml(unrenderHtml(html));
      else next = renderHtml(html);
      if (next !== html) {
        write(file, next);
        changed++;
        console.log(`math: ${file}`);
      }
    } catch (e) {
      failed++;
      console.log(`FAIL ${file}: ${e.message}`);
    }
  }
  if (!flags.has("--check")) console.log(`math: ${changed} file(s) updated`);
  process.exit(failed ? 1 : 0);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main(process.argv.slice(2));
}
