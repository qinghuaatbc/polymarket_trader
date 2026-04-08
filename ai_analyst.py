"""
AI Analyst — Phase 4
Input: market question + news context
Output: probability estimate + confidence + reasoning
"""
from __future__ import annotations
import os
import re
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).parent / ".env")

MODEL = "claude-sonnet-4-6"

ANALYST_SYSTEM = """You are a professional prediction market analyst specializing in Polymarket.

Your task:
1. Analyze the given prediction market question
2. Incorporate any provided news and background information
3. Estimate the probability of a YES outcome
4. Explain your reasoning
5. Assess your confidence level

Output format (strict JSON):
{
  "yes_probability": 0.65,      // probability of YES (0-1)
  "confidence": 0.75,           // your confidence in this estimate (0-1)
  "edge": 0.15,                 // edge vs market price (positive = underpriced)
  "recommendation": "BUY_YES",  // BUY_YES | BUY_NO | SKIP
  "reasoning": "brief analysis...",
  "key_factors": ["factor1", "factor2"],
  "risks": ["risk1", "risk2"],
  "time_sensitivity": "low"     // low | medium | high
}

Principles:
- Stay objective, base analysis on facts
- Acknowledge uncertainty
- If information is insufficient, state what additional info is needed
- Recommend SKIP if confidence < 0.55
- Recommend SKIP if edge < 2% (fees eat profit)
- Recommend BUY_YES or BUY_NO whenever you have a genuine view with confidence >= 0.55
"""


def _fetch_news(query: str, max_items: int = 5) -> str:
    """Fetch recent headlines from Google News RSS for a given query."""
    try:
        q = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            xml = resp.read()
        root = ET.fromstring(xml)
        items = root.findall(".//item")[:max_items]
        headlines = []
        for item in items:
            title = item.findtext("title", "").strip()
            pub = item.findtext("pubDate", "").strip()
            if title:
                headlines.append(f"- {title} ({pub[:16]})")
        return "\n".join(headlines) if headlines else "No recent news found."
    except Exception as e:
        return f"News fetch failed: {e}"


class AIAnalyst:
    def __init__(self):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            self._client = None
        else:
            self._client = anthropic.Anthropic(api_key=key)

    def analyse(self, question: str, market_price: float, news: str = "", context: str = "") -> dict:
        """Analyze a market and return a probability estimate."""
        if not self._client:
            return self._fallback(market_price)

        # Auto-fetch news if none provided
        if not news:
            news = _fetch_news(question)

        prompt = f"""Analyze this prediction market:

**Question:** {question}
**Current market price (YES probability):** {market_price*100:.1f}%

**Recent news:**
{news}

**Other context:**
{context or "None"}

Please provide your probability estimate and analysis."""

        try:
            msg = self._client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=ANALYST_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            # Extract JSON object from response
            m = re.search(r'\{[\s\S]*\}', text)
            if not m:
                raise ValueError(f"No JSON found in response: {text[:100]}")
            result = json.loads(m.group())
            result["market_price"] = market_price
            result["edge"] = round(result.get("yes_probability", market_price) - market_price, 4)
            return result
        except Exception as e:
            r = self._fallback(market_price)
            r["error"] = str(e)
            return r

    def daily_summary(self, predictions: list, paper_stats: dict) -> str:
        """Daily performance review."""
        if not self._client:
            return "Configure ANTHROPIC_API_KEY to enable AI analysis."

        prompt = f"""Help me analyze today's trading performance:

Number of predictions: {len(predictions)}
Paper trading stats: {json.dumps(paper_stats, ensure_ascii=False)}
Recent predictions: {json.dumps(predictions[-5:], ensure_ascii=False, default=str)}

Please provide:
1. Today's performance summary
2. Strengths and weaknesses in judgment
3. Key areas to watch tomorrow
4. Improvement suggestions"""

        try:
            msg = self._client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            return f"Analysis failed: {e}"

    def find_opportunities(self, markets: list[dict]) -> list[dict]:
        """Identify the most valuable opportunities from a list of markets."""
        if not self._client:
            return []

        # Pre-filter: exclude near-certain markets (price < 5% or > 95%) — no edge possible
        tradeable = [
            m for m in markets
            if 0.05 <= m.get('yes_price', 0.5) <= 0.95
        ]

        market_list = "\n".join([
            f"- slug={m['slug']} | {m['question']} | price={m['yes_price']*100:.1f}% | vol=${m['volume']:,.0f} | liq=${m.get('liquidity',0):,.0f}"
            for m in tradeable
        ])

        prompt = f"""You are a prediction market analyst. Review these Polymarket markets and find mispriced opportunities.

{market_list}

Rules:
- ONLY pick markets where you believe the price is SIGNIFICANTLY WRONG (at least 8% off fair value)
- Prefer markets with liquidity > $10,000 (easier to enter/exit)
- Avoid markets priced 45-55% (too uncertain to have an edge)
- Avoid markets expiring within 24 hours (no time to be right)
- Pick exactly 3 best opportunities

Output ONLY a JSON array, no other text:
[{{"slug": "...", "reason": "brief reason why price is wrong", "suggested_side": "YES or NO"}}]"""

        try:
            msg = self._client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            m = re.search(r'\[[\s\S]*\]', text)
            if not m:
                return []
            return json.loads(m.group())
        except Exception:
            return []

    def _fallback(self, market_price: float) -> dict:
        return {
            "yes_probability": market_price,
            "confidence": 0.3,
            "edge": 0.0,
            "recommendation": "SKIP",
            "reasoning": "AI analysis unavailable. Please configure ANTHROPIC_API_KEY.",
            "key_factors": [],
            "risks": ["No AI support"],
            "time_sensitivity": "low",
            "market_price": market_price,
        }
