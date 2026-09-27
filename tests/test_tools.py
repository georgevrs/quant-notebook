import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import build  # noqa: E402
import sync_nav  # noqa: E402
import verify_labs  # noqa: E402
from course import Course, rel_link, sub_outside_comments, term_anchor  # noqa: E402


@pytest.fixture(scope="module")
def course():
    return Course()


def test_course_integrity(course):
    assert len(course.units) == 16
    assert len(course.sessions) == 100
    ids = [s.id for s in course.sessions]
    assert len(set(ids)) == len(ids)
    for s in course.sessions:
        for p in s.prereqs:
            assert course.by_id[p].index < s.index, f"{s.id} depends on later {p}"
    owners = {}
    for s in course.sessions:
        for t in s.owns:
            assert t.lower() not in owners, f"{t} owned twice"
            owners[t.lower()] = s.id


def test_paths(course):
    s = course.by_id["2.3"]
    assert s.rel_path == "unit02-markets-instruments/session03-returns-pnl-leverage.html"
    assert s.repo_rel(s.lab_py) == "labs/u02/s03_returns_pnl_leverage.py"
    assert s.results_json.name == "s03.results.json"


def test_rel_link():
    a = "unit02-markets-instruments/session03-x.html"
    assert rel_link(a, "index.html") == "../index.html"
    assert rel_link(a, "unit02-markets-instruments/session04-y.html") == "session04-y.html"
    assert rel_link(a, "unit06-b/session01-z.html") == "../unit06-b/session01-z.html"
    assert rel_link("index.html", "unit06-b/session01-z.html") == "unit06-b/session01-z.html"


def test_term_anchor():
    assert term_anchor("Sharpe ratio") == "g-sharpe-ratio"
    assert term_anchor("P&L") == "g-p-l"
    assert term_anchor("Bayes' rule") == "g-bayes-rule"


def test_fill_requires_exactly_one_marker():
    text = "<!-- nav:tabs --><!-- /nav:tabs -->"
    assert "X" in sync_nav.fill(text, "tabs", "X")
    with pytest.raises(sync_nav.NavError):
        sync_nav.fill(text + text, "tabs", "X")
    with pytest.raises(sync_nav.NavError):
        sync_nav.fill("", "tabs", "X")


def test_fill_is_idempotent():
    text = "a <!-- nav:x --><!-- /nav:x --> b"
    once = sync_nav.fill(text, "x", "<p>hi</p>")
    assert sync_nav.fill(once, "x", "<p>hi</p>") == once


def test_xref_published_and_unpublished(course):
    page = course.by_id["1.3"].rel_path
    problems = []
    # 2.3 published in this test's view, 6.7 not
    course.by_id["2.3"].status = "published"
    course.by_id["6.7"].status = "planned"
    out = sync_nav.resolve_xrefs(course, page, '<a data-xref="2.3#s4"></a> and <a data-xref="6.7"></a>', problems)
    assert 'href="../unit02-markets-instruments/session03-returns-pnl-leverage.html#s4"' in out
    assert "Session 2.3 — Returns, P&amp;L, Compounding &amp; Leverage" in out
    assert '<span class="xref soon" data-xref="6.7"' in out
    assert not problems
    # idempotent: resolving again changes nothing
    assert sync_nav.resolve_xrefs(course, page, out, problems) == out
    # an unknown id is reported, not silently dropped
    sync_nav.resolve_xrefs(course, page, '<a data-xref="99.9"></a>', problems)
    assert problems


def test_terms_link_to_owner(course):
    course.by_id["2.3"].status = "published"
    problems = []
    out = sync_nav.resolve_terms(course, course.by_id["1.3"].rel_path,
                                 '<a data-term="Sharpe ratio">Sharpe ratios</a>', problems)
    assert "session03-returns-pnl-leverage.html#g-sharpe-ratio" in out
    assert ">Sharpe ratios<" in out
    sync_nav.resolve_terms(course, "index.html", '<a data-term="no such term">x</a>', problems)
    assert problems


def test_nothing_rewritten_inside_comments(course):
    problems = []
    text = '<!-- example: <a data-xref="6.3"></a> -->'
    assert sync_nav.resolve_xrefs(course, "index.html", text, problems) == text
    pat = re.compile(r"x")
    assert sub_outside_comments(pat, "y", "x<!-- x -->x") == "y<!-- x -->y"


def test_gloss_ids():
    html = '<div class="gloss" id="glossary">\n<div><b>Sharpe ratio</b> — d.</div>\n</div>'
    out = sync_nav.normalise_gloss_ids(html)
    assert '<div id="g-sharpe-ratio"><b>Sharpe ratio</b>' in out
    assert sync_nav.normalise_gloss_ids(out) == out


def test_fmt_value_no_negative_zero():
    assert build.fmt_value(-1e-17, ".1%") == "0.0%"
    assert build.fmt_value(-0.196, ".1%") == "−19.6%"
    assert build.fmt_value(2000.0, ",.0f") == "2,000"
    assert build.fmt_value("text", None) == "text"
    assert build.fmt_value(-1.0, "$,.2f") == "−$1.00"
    assert build.fmt_value(1234.5, "$,.2f") == "$1,234.50"
    assert build.fmt_value(-0.001, "$.2f") == "$0.00"


def test_verify_labs_skips_nondeterministic_scs_keys():
    # SCS's own near-zero residual varies run to run even on identical CI hardware (observed
    # swinging from -6e-5 to +1e-3 across two back-to-back runs), so its value is never compared —
    # only that the key is present on both sides.
    assert verify_labs.compare(
        {"tol_daily_scs_dvol_bp": -9.79e-08, "tol_daily_scs_dw": 9.51e-09, "other": 1.0},
        {"tol_daily_scs_dvol_bp": 1.07e-03, "tol_daily_scs_dw": 2.32e-05, "other": 1.0},
    ) == []
    # an ordinary key with the same-sized drift is still a real, reported difference
    assert verify_labs.compare({"other": -9.79e-08}, {"other": -6.04e-05}) != []


def test_verify_labs_skips_wall_clock_timings():
    # Session 4.5's storage-stack timings vary with disk cache / machine load by design.
    assert verify_labs.compare(
        {"t_duckdb_ms": 28.9, "speedup_duckdb_vs_csv": 68.5, "t_write_csv": 7.25, "other": 1.0},
        {"t_duckdb_ms": 33.4, "speedup_duckdb_vs_csv": 62.8, "t_write_csv": 6.81, "other": 1.0},
    ) == []
    # a bare "t_" prefix used for something else (Student-t params, a period count) is NOT
    # exempted — only the deliberate "_ms" suffix / "speedup_"/"slowdown_" prefix are.
    assert verify_labs.compare({"t_nu_sim": 6.0}, {"t_nu_sim": 5.0}) != []
    assert verify_labs.compare({"t_oos_years": 8.0}, {"t_oos_years": 7.0}) != []
    # the key must still be present on both sides
    assert verify_labs.compare({"tol_daily_scs_dw": 1.0}, {}) == ["tol_daily_scs_dw: missing after re-run"]
