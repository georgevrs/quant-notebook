# Data sources

**No third-party market data is committed to this repository.** Most financial data cannot be
redistributed, and free sources change without notice. Labs therefore:

1. **Default to seeded synthetic data** (`quantnb.synth`) with the statistical properties the lesson
   needs — legal to share, identical on every machine, and the truth is known.
2. **Download public data at runtime** into `data/cache/` (git-ignored) when a lesson needs the
   real thing, always guarded by `quantnb.offline()` with a synthetic fallback so CI runs offline.

| Source | Used for | Terms to respect |
| --- | --- | --- |
| FRED / ALFRED (St. Louis Fed) | rates, macro, point-in-time vintages | Public-domain series only may be redistributed; series marked "Copyrighted" (e.g. SP500, DJIA, NASDAQCOM, VIXCLS, ICE BofA indices) may not. Cite FRED. |
| Kenneth French Data Library | factor returns, portfolio sorts | Copyright Fama & French — fetch at runtime, never commit; cite the library. |
| Open Source Asset Pricing (Chen & Zimmermann) | factor zoo, replication | Cite the authors and data release. |
| SEC EDGAR / XBRL APIs | fundamentals, filings text | Max 10 requests/second; a descriptive User-Agent is required (set `SEC_USER_AGENT`). |
| Binance public data (data.binance.vision) | crypto bars, trades, funding | Exchange terms; unavailable from some jurisdictions (e.g. US IP addresses). |
| LOBSTER sample files | limit-order-book reconstruction | Free samples for education; licensed beyond that. |

If you add a committed sample file (public-domain only), list it here with its source, retrieval
date, licence and citation.
