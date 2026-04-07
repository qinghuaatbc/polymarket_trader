"""
AI 分析模型 — 第四阶段
输入：市场问题 + 新闻背景
输出：概率判断 + 置信度 + 推理
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).parent / ".env")

MODEL = "claude-sonnet-4-6"

ANALYST_SYSTEM = """你是一个专业的预测市场分析师，专注于 Polymarket 平台。

你的任务：
1. 分析给定的预测市场问题
2. 结合提供的新闻和背景信息
3. 给出你对 YES 结果的概率估计
4. 解释你的推理过程
5. 评估置信度

输出格式（严格JSON）：
{
  "yes_probability": 0.65,      // YES 发生的概率 0-1
  "confidence": 0.75,           // 你对这个判断的置信度 0-1
  "edge": 0.15,                 // 相对于市场价格的优势（正=被低估）
  "recommendation": "BUY_YES",  // BUY_YES | BUY_NO | SKIP
  "reasoning": "简要分析...",
  "key_factors": ["因素1", "因素2"],
  "risks": ["风险1", "风险2"],
  "time_sensitivity": "low"     // low | medium | high
}

原则：
- 保持客观，基于事实
- 承认不确定性
- 如果信息不足，说明需要哪些信息
- 置信度 < 0.6 建议 SKIP
- 优势 < 3% 建议 SKIP（手续费吃掉利润）
"""


class AIAnalyst:
    def __init__(self):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            self._client = None
        else:
            self._client = anthropic.Anthropic(api_key=key)

    def analyse(self, question: str, market_price: float, news: str = "", context: str = "") -> dict:
        """分析市场，返回概率判断"""
        if not self._client:
            return self._fallback(market_price)

        prompt = f"""分析这个预测市场：

**问题：** {question}
**当前市场价格（YES概率）：** {market_price*100:.1f}%

**相关新闻/背景：**
{news or "无额外信息"}

**其他背景：**
{context or "无"}

请给出你的概率判断和分析。"""

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
        """每日复盘分析"""
        if not self._client:
            return "需要配置 ANTHROPIC_API_KEY 才能使用 AI 分析"

        prompt = f"""帮我分析今天的交易表现：

预测记录数量：{len(predictions)}
纸上交易统计：{json.dumps(paper_stats, ensure_ascii=False)}
今日预测：{json.dumps(predictions[-5:], ensure_ascii=False, default=str)}

请提供：
1. 今日表现总结
2. 判断的优缺点
3. 明日关注重点
4. 改进建议"""

        try:
            msg = self._client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            return f"分析失败: {e}"

    def find_opportunities(self, markets: list[dict]) -> list[dict]:
        """从市场列表中找出最有价值的机会"""
        if not self._client:
            return []

        market_list = "\n".join([
            f"- {m['question']} (市场价: {m['yes_price']*100:.1f}%, 成交量: ${m['volume']:,.0f})"
            for m in markets[:20]
        ])

        prompt = f"""从以下 Polymarket 市场中，找出最值得关注的机会：

{market_list}

选出前3个你认为市场定价可能有误的市场，说明原因。
输出JSON数组：[{{"slug": "...", "reason": "...", "suggested_side": "YES/NO"}}]"""

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
            "reasoning": "AI 分析不可用，请配置 ANTHROPIC_API_KEY",
            "key_factors": [],
            "risks": ["无 AI 支持"],
            "time_sensitivity": "low",
            "market_price": market_price,
        }
