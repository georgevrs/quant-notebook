import { test } from "node:test";
import assert from "node:assert/strict";
import { findTex, renderHtml, unrenderHtml, checkHtml } from "../render_math.mjs";

test("inline and display math render to MathML with annotations", () => {
  const out = renderHtml("<p>Return \\(r_t = \\ln P_t\\) and</p>\n\\[\\SR = \\frac{\\mu}{\\sigma}\\]");
  assert.match(out, /<math>/);
  assert.match(out, /<div class="eq"><math[^>]*display="block"/);
  assert.equal((out.match(/application\/x-tex/g) || []).length, 2);
  assert.doesNotMatch(out, /\\\(/);
});

test("rendering is idempotent", () => {
  const once = renderHtml("<p>\\(a + b\\)</p>");
  assert.equal(renderHtml(once), once);
});

test("less-than and ampersand survive via entities and \\lt", () => {
  const out = renderHtml("<p>\\(a &lt; b\\) and \\(c \\lt d\\) and \\[\\begin{aligned} x &amp;= 1 \\\\ y &= 2 \\end{aligned}\\]</p>");
  assert.match(out, /<mo>&lt;<\/mo>/);
  assert.equal(checkHtml(out).length, 0);
});

test("\\\\[2pt] inside aligned is a line break, not display math", () => {
  const html = "\\[\\begin{aligned} a &= b \\\\[2pt] c &= d \\end{aligned}\\]";
  const spans = findTex(html);
  assert.equal(spans.length, 1);
  assert.equal(spans[0].display, true);
  assert.match(renderHtml(html), /<mtable/);
});

test("code, pre, svg and comments are never touched", () => {
  const html = '<pre>re.sub(r"\\(x\\)", s)</pre><code>\\(x\\)</code><svg><text>\\(x\\)</text></svg><!-- \\(x\\) -->';
  assert.equal(renderHtml(html), html);
  assert.equal(checkHtml(html).length, 0);
});

test("round trip: unrender then render gives the same MathML", () => {
  const src = "<p>\\(\\E[r] = \\mu\\) then</p>\n\\[w^\\star = \\frac{1}{\\gamma}\\Sigma^{-1}\\mu\\]";
  const rendered = renderHtml(src);
  const back = unrenderHtml(rendered);
  assert.match(back, /\\\(/);
  assert.match(back, /\\\[/);
  assert.equal(renderHtml(back), rendered);
});

test("check flags raw TeX and bad TeX fails loudly", () => {
  assert.ok(checkHtml("<p>\\(x\\)</p>").length > 0);
  assert.throws(() => renderHtml("<p>\\(\\frac{a\\)</p>"), /TeX error/);
  assert.throws(() => findTex("<p>\\(x </p><p>more</p>"), /unterminated/);
});
