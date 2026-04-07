"""
AI Analyst — Phase 4
Input: market question + news context
Output: probability estimate + confidence + reasoning
"""
from __future__ import annotations
import os
import json
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
- Recommend SKIP if confidence < 0.6
- Recommend SKIP if edge < 3% (fees eat profit)
"""


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

        prompt = f"""Analyze this prediction market:

**Question:** {question}
**Current market price (YES probability):** {market_price*100:.1f}%

**Relevant news/background:**
{news or "No additional information"}

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
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
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

        market_list = "\n".join([
            f"- {m['question']} (market price: {m['yes_price']*100:.1f}%, volume: ${m['volume']:,.0f})"
            for m in markets[:20]
        ])

        prompt = f"""From these Polymarket markets, identify the most noteworthy opportunities:

{market_list}

Select the top 3 markets you believe may be mispriced and explain why.
Output as a JSON array: [{{"slug": "...", "reason": "...", "suggested_side": "YES/NO"}}]"""

        try:
            msg = self._client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
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
