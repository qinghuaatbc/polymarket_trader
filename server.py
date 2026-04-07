"""
Polymarket 预测交易系统 — Web 服务器
运行: python3 server.py
访问: http://localhost:8002
"""
from __future__ import annotations
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from polymarket import PolymarketClient
from journal import (
    add_prediction, resolve_prediction, get_predictions, prediction_stats,
    paper_trade, resolve_paper_trade, get_paper_trades, paper_trade_stats,
    get_settings, reset_paper_account,
)
from ai_analyst import AIAnalyst

_poly    = PolymarketClient()
_analyst = AIAnalyst()
_market_cache: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 预加载市场数据
    try:
        markets = _poly.fetch_markets(limit=50)
        _market_cache.extend([m.__dict__ for m in markets])
    except Exception as e:
        print(f"Market preload error: {e}")
    yield


app = FastAPI(title="Polymarket Trader", lifespan=lifespan)


# ── 市场数据 ──────────────────────────────────────────────────────────────────

@app.get("/api/markets")
def api_markets(limit: int = 50, category: str = "", closed: bool = False):
    try:
        markets = _poly.fetch_markets(limit=limit, category=category, closed=closed)
        return {"markets": [m.__dict__ for m in markets]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/markets/resolved")
def api_resolved(limit: int = 20):
    try:
        markets = _poly.fetch_resolved(limit=limit)
        return {"markets": [m.__dict__ for m in markets]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/market/{slug}")
def api_market(slug: str):
    try:
        raw = _poly.get_market(slug)
        if not raw:
            raise HTTPException(404, "Market not found")
        return _poly.parse_market(raw).__dict__
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Phase 1: 预测日志 ─────────────────────────────────────────────────────────

class PredictionIn(BaseModel):
    market_slug: str
    question: str
    market_price: float
    my_probability: float
    reasoning: str = ""
    category: str = ""

class ResolveIn(BaseModel):
    outcome: str   # YES | NO

@app.post("/api/predictions")
def api_add_prediction(body: PredictionIn):
    return add_prediction(**body.dict())

@app.get("/api/predictions")
def api_get_predictions(resolved: bool = False, unresolved: bool = False):
    return {"predictions": get_predictions(resolved_only=resolved, unresolved_only=unresolved)}

@app.post("/api/predictions/{pred_id}/resolve")
def api_resolve_prediction(pred_id: str, body: ResolveIn):
    return resolve_prediction(pred_id, body.outcome)

@app.get("/api/predictions/stats")
def api_pred_stats():
    return prediction_stats()


# ── Phase 2: 纸上交易 ─────────────────────────────────────────────────────────

class TradeIn(BaseModel):
    market_slug: str
    question: str
    side: str
    price: float
    size_usd: float
    my_probability: float
    reasoning: str = ""

@app.post("/api/paper/trade")
def api_paper_trade(body: TradeIn):
    return paper_trade(**body.dict())

@app.get("/api/paper/trades")
def api_get_trades(resolved: bool = False):
    return {"trades": get_paper_trades(resolved_only=resolved)}

@app.post("/api/paper/trades/{trade_id}/resolve")
def api_resolve_trade(trade_id: str, body: ResolveIn):
    return resolve_paper_trade(trade_id, body.outcome)

@app.get("/api/paper/stats")
def api_paper_stats():
    return paper_trade_stats()

@app.post("/api/paper/reset")
def api_reset(balance: float = 1000.0):
    return reset_paper_account(balance)


# ── Phase 4: AI 分析 ──────────────────────────────────────────────────────────

class AnalyseIn(BaseModel):
    question: str
    market_price: float
    news: str = ""
    context: str = ""

@app.post("/api/ai/analyse")
def api_analyse(body: AnalyseIn):
    return _analyst.analyse(body.question, body.market_price, body.news, body.context)

@app.get("/api/ai/opportunities")
def api_opportunities():
    try:
        markets = _poly.fetch_markets(limit=30)
        market_list = [m.__dict__ for m in markets]
        opps = _analyst.find_opportunities(market_list)
        return {"opportunities": opps}
    except Exception as e:
        return {"opportunities": [], "error": str(e)}

@app.get("/api/ai/daily-summary")
def api_daily_summary():
    preds = get_predictions()
    stats = paper_trade_stats()
    summary = _analyst.daily_summary(preds, stats)
    return {"summary": summary}


# ── 设置 ──────────────────────────────────────────────────────────────────────

@app.get("/api/settings")
def api_settings():
    return get_settings()


# ── 主页 ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=False)
