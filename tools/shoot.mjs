#!/usr/bin/env node
/**
 * shoot.mjs — visual QA for pages: element screenshots at phone and desktop width, plus
 * automatic layout checks. Uses the system Microsoft Edge through playwright-core (no browser
 * download) with a throwaway profile per run, so parallel runs never collide.
 *
 *   node tools/shoot.mjs site/unit02-…/session03-….html [more.html …]
 *   node tools/shoot.mjs --widths 390,1000 --out .shots page.html
 *   node tools/shoot.mjs --no-shots page.html          checks only
 *
 * Output: .shots/<page>/<width>-<nn>-<kind>.png and a printed report. Exit 1 if any check fails:
 *   * SVG <text> that overflows the smallest <rect> it starts in, or the SVG viewBox
 *   * horizontal page overflow (document wider than the viewport) at any width
 *   * an element (table/pre/.eq excepted — they scroll by design) wider than its container
 *   * a web font that failed to load
 * Set EDGE_PATH to override the browser executable; CHANNEL=chrome to use Chrome instead.
 */
import { mkdirSync, rmSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright-core";

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  if (i === -1) return dflt;
  const v = args[i + 1];
  args.splice(i, 2);
  return v;
};
const flag = (name) => {
  const i = args.indexOf(name);
  if (i === -1) return false;
  args.splice(i, 1);
  return true;
};
const widths = opt("--widths", "390,1000").split(",").map(Number);
const outRoot = opt("--out", ".shots");
const noShots = flag("--no-shots");
const pages = args;
if (!pages.length) {
  console.error("usage: shoot.mjs [--widths 390,1000] [--out .shots] [--no-shots] page.html …");
  process.exit(2);
}

const launchOpts = process.env.EDGE_PATH
  ? { executablePath: process.env.EDGE_PATH }
  : { channel: process.env.CHANNEL || "msedge" };

/** Runs inside the page: returns a list of layout problems. */
function pageChecks() {
  const problems = [];
  const vw = document.documentElement.clientWidth;
  if (document.documentElement.scrollWidth > vw + 1) {
    problems.push(`page scrolls sideways: ${document.documentElement.scrollWidth}px > ${vw}px`);
  }
  // SVG text overflow against the smallest rect containing its anchor point
  document.querySelectorAll("figure.diagram svg").forEach((svg, si) => {
    const vb = svg.viewBox.baseVal;
    const rects = [...svg.querySelectorAll("rect")].map((r) => r.getBBox());
    svg.querySelectorAll("text").forEach((t) => {
      const b = t.getBBox();
      const label = (t.textContent || "").trim().slice(0, 40);
      if (b.x < vb.x - 1 || b.y < vb.y - 1 || b.x + b.width > vb.x + vb.width + 1 || b.y + b.height > vb.y + vb.height + 1) {
        problems.push(`fig #${si + 1}: text outside the viewBox: "${label}"`);
        return;
      }
      const ax = parseFloat(t.getAttribute("x") || "0");
      const ay = parseFloat(t.getAttribute("y") || "0");
      const inside = rects
        .filter((r) => ax >= r.x && ax <= r.x + r.width && ay >= r.y && ay <= r.y + r.height && r.width < vb.width * 0.98)
        .sort((p, q) => p.width * p.height - q.width * q.height);
      const box = inside[0];
      if (box && (b.x < box.x - 1.5 || b.x + b.width > box.x + box.width + 1.5)) {
        problems.push(`fig #${si + 1}: text overflows its box by ${Math.round(Math.max(box.x - b.x, b.x + b.width - box.x - box.width))}px: "${label}"`);
      }
    });
  });
  // blocks wider than their container (tables, pre and .eq scroll on purpose)
  document.querySelectorAll("main > section > *:not(table):not(pre):not(.eq), header.cover > *").forEach((el) => {
    if (el.scrollWidth > el.clientWidth + 2 && getComputedStyle(el).overflowX === "visible") {
      problems.push(`<${el.tagName.toLowerCase()} class="${el.className}"> overflows its width by ${el.scrollWidth - el.clientWidth}px`);
    }
  });
  return problems;
}

const browser = await chromium.launch({ ...launchOpts, headless: true });
let failed = 0;
try {
  for (const page of pages) {
    const url = page.startsWith("http://") || page.startsWith("https://") ? page : pathToFileURL(resolve(page)).href;
    const stem = basename(page.replace(/[?#].*$/, ""), ".html") || "index";
    const outDir = join(outRoot, stem);
    if (!noShots) {
      rmSync(outDir, { recursive: true, force: true });
      mkdirSync(outDir, { recursive: true });
    }
    console.log(`\n${page}`);
    for (const w of widths) {
      const ctx = await browser.newContext({ viewport: { width: w, height: 900 }, deviceScaleFactor: w < 600 ? 2 : 1 });
      const p = await ctx.newPage();
      await p.goto(url, { waitUntil: "load" });
      const fontsOk = await p.evaluate(async () => {
        await document.fonts.ready;
        return [...document.fonts].filter((f) => f.status === "error").map((f) => f.family);
      });
      if (fontsOk.length) {
        console.log(`  FAIL ${w}px: fonts failed to load: ${fontsOk.join(", ")}`);
        failed++;
      }
      const problems = await p.evaluate(pageChecks);
      for (const pr of problems) console.log(`  FAIL ${w}px: ${pr}`);
      failed += problems.length;
      if (!noShots) {
        let n = 0;
        const cover = await p.$("header.cover");
        if (cover) await cover.screenshot({ path: join(outDir, `${w}-${String(n++).padStart(2, "0")}-cover.png`) });
        for (const [sel, kind] of [["figure.diagram", "fig"], [".eq", "eq"], ["main table", "table"], ["main pre", "code"]]) {
          const els = await p.$$(sel);
          for (const el of els) {
            await el.screenshot({ path: join(outDir, `${w}-${String(n++).padStart(2, "0")}-${kind}.png`) });
          }
        }
        console.log(`  ${w}px: ${n} screenshots → ${outDir}`);
      }
      await ctx.close();
    }
  }
} finally {
  await browser.close();
}
console.log(failed ? `\n${failed} problem(s)` : "\nno layout problems");
process.exit(failed ? 1 : 0);
