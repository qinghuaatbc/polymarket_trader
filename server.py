"""
Polymarket Prediction Trading System — Web Server
Run: python3 server.py
Visit: http://localhost:8002
"""
from __future__ import annotations
import json
import datetime
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from polymarket import PolymarketClient
from journal import (
    add_prediction, resolve_prediction, get_predictions, prediction_stats,
    paper_trade, resolve_paper_trade, get_paper_trades, paper_trade_stats,
    get_settings, reset_paper_account, ai_feedback_stats,
)
from ai_analyst import AIAnalyst

_poly    = PolymarketClient()
_analyst = AIAnalyst()
_market_cache: list[dict] = []

# AI auto-trade activity log (in-memory, last 100 entries)
_auto_log: list[dict] = []
MAX_LOG = 100

# Background scheduler state
_auto_state: dict = {
    "running": False,
    "interval_min": 30,
    "next_run_ts": None,  # unix timestamp of next scan
    "cfg": {"min_confidence": 0.55, "min_edge": 0.03, "trade_size": 30.0, "scan_limit": 300},
}
_auto_thread: threading.Thread | None = None
_auto_stop_event = threading.Event()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        markets = _poly.fetch_markets_multi(per_strategy=100)
        _market_cache.extend([m.__dict__ for m in markets])
    except Exception as e:
        print(f"Market preload error: {e}")
    # Auto-start scheduler on boot with default config
    _auto_state["running"] = True
    _auto_state["next_run_ts"] = time.time()
    t = threading.Thread(target=_auto_worker, daemon=True)
    t.start()
    global _auto_thread
    _auto_thread = t
    yield


app = FastAPI(title="Polymarket Trader", lifespan=lifespan)


# ── Market data ───────────────────────────────────────────────────────────────

@app.get("/api/markets")
def api_markets(limit: int = 200, category: str = "", closed: bool = False):
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


@app.get("/api/paper/expired")
def api_paper_expired():
    """
    Return open trades where the market end_date has passed but
    Polymarket hasn't resolved the market yet.
    """
    open_trades = [t for t in get_paper_trades() if not t["resolved"]]
    now = datetime.datetime.utcnow()
    expired = []
    for trade in open_trades:
        slug = trade["market_slug"]
        try:
            raw = _poly.get_market(slug)
            if not raw:
                continue
            market = _poly.parse_market(raw)
            if market.resolved:
                continue  # will be caught by auto-settle
            if not market.end_date:
                continue
            # Parse end_date — Polymarket returns ISO format
            end_str = market.end_date.replace("Z", "+00:00")
            try:
                end_dt = datetime.datetime.fromisoformat(end_str).replace(tzinfo=None)
            except Exception:
                continue
            if end_dt < now:
                expired.append({
                    "id": trade["id"],
                    "market_slug": slug,
                    "question": trade["question"],
                    "side": trade["side"],
                    "end_date": market.end_date,
                    "hours_overdue": round((now - end_dt).total_seconds() / 3600, 1),
                })
        except Exception:
            continue
    return {"expired": expired}


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
    min_confidence: float = 0.55   # minimum AI confidence to trade
    min_edge: float = 0.03         # minimum edge vs market price
    trade_size: float = 30.0       # USD per trade
    scan_limit: int = 300          # how many markets to scan per strategy

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

def _run_auto_cycle(cfg: AutoTradeConfig) -> dict:
    """Core scan cycle — shared by the manual endpoint and the background scheduler."""
    results = {"scanned": 0, "analysed": 0, "traded": 0, "skipped": 0, "errors": []}

    stats = paper_trade_stats()
    balance = stats.get("balance", 0)
    open_trades = stats.get("open_trades", 0)

    if balance < cfg.trade_size:
        _log(f"Insufficient balance (${balance:.2f})", "error")
        return {**results, "error": f"Insufficient balance: ${balance:.2f}"}

    if open_trades >= 30:
        _log(f"Too many open trades ({open_trades}), skipping scan", "skip")
        return {**results, "skipped": open_trades}

    try:
        markets = _poly.fetch_markets_multi(per_strategy=cfg.scan_limit)
        market_list = [m.__dict__ for m in markets]
        results["scanned"] = len(market_list)
        _log(f"Scanned {len(market_list)} markets (multi-strategy)")
    except Exception as e:
        _log(f"Failed to fetch markets: {e}", "error")
        return {**results, "error": str(e)}

    try:
        candidates = _analyst.find_opportunities(market_list)
        _log(f"AI identified {len(candidates)} candidates")
    except Exception as e:
        _log(f"Opportunity scan failed: {e}", "error")
        candidates = []

    if not candidates:
        _log("No candidates found this cycle", "skip")
        return results

    # Track already-open slugs to avoid doubling up
    open_slugs = {t["market_slug"] for t in get_paper_trades() if not t.get("resolved")}

    for opp in candidates[:3]:
        slug = opp.get("slug", "")
        if slug in open_slugs:
            _log(f"SKIP {slug[:30]} — already have open position", "skip")
            results["skipped"] += 1
            continue
        mkt = next((m for m in market_list if m["slug"] == slug), None)
        if not mkt:
            continue

        try:
            analysis = _analyst.analyse(question=mkt["question"], market_price=mkt["yes_price"])
            results["analysed"] += 1
        except Exception as e:
            _log(f"Analysis failed for {slug}: {e}", "error")
            results["errors"].append(str(e))
            continue

        confidence = analysis.get("confidence", 0)
        edge = abs(analysis.get("edge", 0))
        recommendation = analysis.get("recommendation", "SKIP")

        if recommendation == "SKIP" or confidence < cfg.min_confidence or edge < cfg.min_edge:
            _log(f"SKIP {slug[:30]} — confidence={confidence:.0%} edge={edge:.1%} rec={recommendation}", "skip")
            results["skipped"] += 1
            continue

        side = "YES" if recommendation == "BUY_YES" else "NO"
        price = mkt["yes_price"] if side == "YES" else mkt["no_price"]
        size = min(cfg.trade_size, balance * 0.1)

        trade_result = paper_trade(
            market_slug=slug,
            question=mkt["question"],
            side=side,
            price=price,
            size_usd=size,
            my_probability=analysis.get("yes_probability", mkt["yes_price"]),
            reasoning=f"[AI Auto] {analysis.get('reasoning','')[:200]}",
            ai_confidence=confidence,
            ai_edge=edge,
        )

        if "error" in trade_result:
            _log(f"Trade failed: {trade_result['error']}", "error")
            results["errors"].append(trade_result["error"])
        else:
            _log(f"TRADE {side} {slug[:25]} @ {price:.1%} | conf={confidence:.0%} edge={edge:.1%} size=${size:.0f}", "trade", trade_result)
            results["traded"] += 1
            balance -= size
            if balance < cfg.trade_size:
                _log("Balance too low for more trades", "info")
                break

    return results


def _auto_worker():
    """Background thread: runs scan cycles on the configured interval until stopped."""
    cfg = AutoTradeConfig(**_auto_state["cfg"])
    interval_sec = _auto_state["interval_min"] * 60

    while not _auto_stop_event.is_set():
        _log("[Scheduler] Starting scan cycle")
        try:
            _run_auto_cycle(cfg)
        except Exception as e:
            _log(f"[Scheduler] Cycle error: {e}", "error")

        if _auto_stop_event.is_set():
            break

        _auto_state["next_run_ts"] = time.time() + interval_sec
        _auto_stop_event.wait(timeout=interval_sec)

    _auto_state["running"] = False
    _auto_state["next_run_ts"] = None


@app.post("/api/ai/auto-trade")
def api_auto_trade(cfg: AutoTradeConfig):
    """Single manual scan cycle (Scan Once button)."""
    return _run_auto_cycle(cfg)


class AutoStartConfig(BaseModel):
    min_confidence: float = 0.55
    min_edge: float = 0.03
    trade_size: float = 30.0
    scan_limit: int = 300
    interval_min: int = 30


@app.post("/api/ai/auto-start")
def api_auto_start(body: AutoStartConfig):
    global _auto_thread
    if _auto_state["running"]:
        return {"status": "already_running"}
    _auto_state["cfg"] = {
        "min_confidence": body.min_confidence,
        "min_edge": body.min_edge,
        "trade_size": body.trade_size,
        "scan_limit": body.scan_limit,
    }
    _auto_state["interval_min"] = body.interval_min
    _auto_state["running"] = True
    _auto_state["next_run_ts"] = time.time()
    _auto_stop_event.clear()
    _auto_thread = threading.Thread(target=_auto_worker, daemon=True)
    _auto_thread.start()
    return {"status": "started"}


@app.post("/api/ai/auto-stop")
def api_auto_stop():
    _auto_stop_event.set()
    _auto_state["running"] = False
    _auto_state["next_run_ts"] = None
    return {"status": "stopped"}


@app.get("/api/ai/auto-status")
def api_auto_status():
    return {
        "running": _auto_state["running"],
        "interval_min": _auto_state["interval_min"],
        "next_run_ts": _auto_state["next_run_ts"],
        "cfg": _auto_state["cfg"],
    }


@app.get("/api/ai/feedback")
def api_ai_feedback():
    return ai_feedback_stats()


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
