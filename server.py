"""
Polymarket Prediction Trading System — Web Server
Run: python3 server.py
Visit: http://localhost:8002
"""
from __future__ import annotations
import json
import datetime
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

# AI auto-trade activity log (in-memory, last 100 entries)
_auto_log: list[dict] = []
MAX_LOG = 100


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        markets = _poly.fetch_markets(limit=50)
        _market_cache.extend([m.__dict__ for m in markets])
    except Exception as e:
        print(f"Market preload error: {e}")
    yield


app = FastAPI(title="Polymarket Trader", lifespan=lifespan)


# ── Market data ───────────────────────────────────────────────────────────────

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


# ── Phase 1: Prediction journal ───────────────────────────────────────────────

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


# ── Phase 2: Paper trading ────────────────────────────────────────────────────

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


@app.post("/api/paper/auto-settle")
def api_auto_settle():
    """
    Check all open paper trades against Polymarket API.
    If the market has resolved, automatically settle the trade.
    """
    open_trades = [t for t in get_paper_trades() if not t["resolved"]]
    results = {"checked": len(open_trades), "settled": [], "still_open": [], "errors": []}

    for trade in open_trades:
        slug = trade["market_slug"]
        try:
            raw = _poly.get_market(slug)
            if not raw:
                results["errors"].append(f"{slug}: market not found")
                continue

            market = _poly.parse_market(raw)

            if not market.resolved or not market.resolution:
                results["still_open"].append(slug)
                continue

            # Market has resolved — settle the trade
            outcome = market.resolution.upper()  # "YES" or "NO"
            settled = resolve_paper_trade(trade["id"], outcome)
            settled["market_slug"] = slug
            settled["auto_settled"] = True
            results["settled"].append(settled)
            _log(
                f"Auto-settled {slug[:30]} → {outcome} | PnL: {'+'if settled.get('pnl',0)>=0 else ''}${settled.get('pnl',0):.2f}",
                "trade" if settled.get("pnl", 0) >= 0 else "error",
                settled,
            )

        except Exception as e:
            results["errors"].append(f"{slug}: {e}")
            _log(f"Auto-settle error {slug}: {e}", "error")

    return results


# ── Phase 4: AI analysis ──────────────────────────────────────────────────────

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


# ── AI Auto Paper Trading ─────────────────────────────────────────────────────

class AutoTradeConfig(BaseModel):
    min_confidence: float = 0.65   # minimum AI confidence to trade
    min_edge: float = 0.05         # minimum edge vs market price
    trade_size: float = 20.0       # USD per trade
    scan_limit: int = 20           # how many markets to scan

def _log(msg: str, status: str = "info", trade: dict = None):
    entry = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "msg": msg,
        "status": status,  # info | trade | skip | error
        "trade": trade,
    }
    _auto_log.insert(0, entry)
    if len(_auto_log) > MAX_LOG:
        _auto_log.pop()

@app.post("/api/ai/auto-trade")
def api_auto_trade(cfg: AutoTradeConfig):
    """
    One scan cycle: AI scans markets → analyses top candidates → places paper trades.
    Call this periodically from the frontend.
    """
    results = {"scanned": 0, "analysed": 0, "traded": 0, "skipped": 0, "errors": []}

    # Check balance
    stats = paper_trade_stats()
    balance = stats.get("balance", 0)
    open_trades = stats.get("open_trades", 0)

    if balance < cfg.trade_size:
        _log(f"Insufficient balance (${balance:.2f})", "error")
        return {**results, "error": f"Insufficient balance: ${balance:.2f}"}

    if open_trades >= 10:
        _log(f"Too many open trades ({open_trades}), skipping scan", "skip")
        return {**results, "skipped": open_trades}

    # Step 1: Fetch markets
    try:
        markets = _poly.fetch_markets(limit=cfg.scan_limit)
        market_list = [m.__dict__ for m in markets]
        results["scanned"] = len(market_list)
        _log(f"Scanned {len(market_list)} markets")
    except Exception as e:
        _log(f"Failed to fetch markets: {e}", "error")
        return {**results, "error": str(e)}

    # Step 2: AI picks top candidates
    try:
        candidates = _analyst.find_opportunities(market_list)
        _log(f"AI identified {len(candidates)} candidates")
    except Exception as e:
        _log(f"Opportunity scan failed: {e}", "error")
        candidates = []

    if not candidates:
        _log("No candidates found this cycle", "skip")
        return results

    # Step 3: Deep-analyse each candidate and trade if criteria met
    for opp in candidates[:3]:  # max 3 trades per cycle
        slug = opp.get("slug", "")
        suggested_side = opp.get("suggested_side", "YES").upper()

        # Find market data
        mkt = next((m for m in market_list if m["slug"] == slug), None)
        if not mkt:
            continue

        # Deep analysis
        try:
            analysis = _analyst.analyse(
                question=mkt["question"],
                market_price=mkt["yes_price"],
            )
            results["analysed"] += 1
        except Exception as e:
            _log(f"Analysis failed for {slug}: {e}", "error")
            results["errors"].append(str(e))
            continue

        confidence = analysis.get("confidence", 0)
        edge = abs(analysis.get("edge", 0))
        recommendation = analysis.get("recommendation", "SKIP")

        if recommendation == "SKIP" or confidence < cfg.min_confidence or edge < cfg.min_edge:
            _log(
                f"SKIP {slug[:30]} — confidence={confidence:.0%} edge={edge:.1%} rec={recommendation}",
                "skip"
            )
            results["skipped"] += 1
            continue

        # Decide side
        side = "YES" if recommendation == "BUY_YES" else "NO"
        price = mkt["yes_price"] if side == "YES" else mkt["no_price"]

        # Check won't exceed 10% balance rule
        size = min(cfg.trade_size, balance * 0.1)

        # Place paper trade
        trade_result = paper_trade(
            market_slug=slug,
            question=mkt["question"],
            side=side,
            price=price,
            size_usd=size,
            my_probability=analysis.get("yes_probability", mkt["yes_price"]),
            reasoning=f"[AI Auto] {analysis.get('reasoning','')[:200]}",
        )

        if "error" in trade_result:
            _log(f"Trade failed: {trade_result['error']}", "error")
            results["errors"].append(trade_result["error"])
        else:
            msg = f"TRADE {side} {slug[:25]} @ {price:.1%} | conf={confidence:.0%} edge={edge:.1%} size=${size:.0f}"
            _log(msg, "trade", trade_result)
            results["traded"] += 1
            balance -= size  # update local balance estimate

            if balance < cfg.trade_size:
                _log("Balance too low for more trades", "info")
                break

    return results


@app.get("/api/ai/auto-log")
def api_auto_log():
    return {"log": _auto_log}


# ── Settings ──────────────────────────────────────────────────────────────────

@app.get("/api/settings")
def api_settings():
    return get_settings()


# ── Homepage ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=False)
