"""
预测日志 + 纸上交易跟踪系统
Phase 1: 每日预测记录
Phase 2: 纸上交易 $1000 虚拟资金
"""
from __future__ import annotations
import json
import uuid
import datetime
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

PREDICTIONS_FILE  = DATA / "predictions.json"
PAPER_TRADES_FILE = DATA / "paper_trades.json"
SETTINGS_FILE     = DATA / "settings.json"


def _load(path: Path) -> list:
    return json.loads(path.read_text()) if path.exists() else []

def _save(path: Path, data):
    path.write_text(json.dumps(data, indent=2, default=str))

def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        return json.loads(SETTINGS_FILE.read_text())
    default = {"paper_balance": 1000.0, "initial_balance": 1000.0, "phase": 1}
    _save(SETTINGS_FILE, default)
    return default

def _save_settings(s: dict):
    _save(SETTINGS_FILE, s)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 1: 预测日志
# ─────────────────────────────────────────────────────────────────────────────

def add_prediction(
    market_slug: str,
    question: str,
    market_price: float,     # 市场当前价格 0-1
    my_probability: float,   # 我认为的真实概率 0-1
    reasoning: str = "",
    category: str = "",
) -> dict:
    predictions = _load(PREDICTIONS_FILE)
    entry = {
        "id": str(uuid.uuid4())[:8],
        "date": datetime.date.today().isoformat(),
        "market_slug": market_slug,
        "question": question,
        "market_price": market_price,
        "my_probability": my_probability,
        "edge": round(my_probability - market_price, 4),  # 正=我认为被低估
        "reasoning": reasoning,
        "category": category,
        "outcome": None,    # 结算后填写: "YES" | "NO"
        "resolved": False,
        "correct": None,    # 结算后: True/False
        "created_at": datetime.datetime.now().isoformat(),
    }
    predictions.append(entry)
    _save(PREDICTIONS_FILE, predictions)
    return entry


def resolve_prediction(pred_id: str, outcome: str) -> dict:
    """记录市场结果，计算预测准确率"""
    predictions = _load(PREDICTIONS_FILE)
    for p in predictions:
        if p["id"] == pred_id:
            p["outcome"] = outcome.upper()
            p["resolved"] = True
            # 如果我认为YES概率>0.5且结果是YES，或YES<0.5且结果是NO，则判断正确
            if outcome.upper() == "YES":
                p["correct"] = p["my_probability"] > 0.5
            else:
                p["correct"] = p["my_probability"] <= 0.5
            p["resolved_at"] = datetime.datetime.now().isoformat()
            _save(PREDICTIONS_FILE, predictions)
            return p
    return {"error": "Not found"}


def get_predictions(resolved_only=False, unresolved_only=False) -> list:
    preds = _load(PREDICTIONS_FILE)
    if resolved_only:
        return [p for p in preds if p["resolved"]]
    if unresolved_only:
        return [p for p in preds if not p["resolved"]]
    return preds


def prediction_stats() -> dict:
    preds = _load(PREDICTIONS_FILE)
    resolved = [p for p in preds if p["resolved"]]
    if not resolved:
        return {"total": len(preds), "resolved": 0, "win_rate": None, "avg_edge": None, "calibration": None}

    correct = [p for p in resolved if p["correct"]]
    win_rate = len(correct) / len(resolved)

    # 校准度：我的概率预测准确性
    avg_edge = sum(p["edge"] for p in preds) / len(preds) if preds else 0

    # 按概率分组统计校准度
    buckets = {}
    for p in resolved:
        bucket = round(p["my_probability"] * 10) / 10  # 0.1 steps
        if bucket not in buckets:
            buckets[bucket] = {"predicted": [], "actual": []}
        buckets[bucket]["predicted"].append(p["my_probability"])
        buckets[bucket]["actual"].append(1 if p["outcome"] == "YES" else 0)

    calibration = []
    for prob, data in sorted(buckets.items()):
        calibration.append({
            "predicted_prob": prob,
            "actual_rate": round(sum(data["actual"]) / len(data["actual"]), 3),
            "count": len(data["actual"]),
        })

    return {
        "total": len(preds),
        "resolved": len(resolved),
        "correct": len(correct),
        "win_rate": round(win_rate, 4),
        "avg_edge": round(avg_edge, 4),
        "calibration": calibration,
        "ready_for_phase2": win_rate >= 0.55 and len(resolved) >= 10,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Phase 2: 纸上交易
# ─────────────────────────────────────────────────────────────────────────────

def paper_trade(
    market_slug: str,
    question: str,
    side: str,              # "YES" | "NO"
    price: float,           # 买入价格
    size_usd: float,        # 投入金额
    my_probability: float,
    reasoning: str = "",
) -> dict:
    settings = _load_settings()
    if size_usd > settings["paper_balance"]:
        return {"error": f"余额不足: ${settings['paper_balance']:.2f}"}
    if size_usd > settings["paper_balance"] * 0.1:
        return {"error": f"单笔上限10%: ${settings['paper_balance']*0.1:.2f}"}

    trades = _load(PAPER_TRADES_FILE)
    shares = size_usd / price
    trade = {
        "id": str(uuid.uuid4())[:8],
        "date": datetime.date.today().isoformat(),
        "market_slug": market_slug,
        "question": question,
        "side": side.upper(),
        "price": price,
        "size_usd": size_usd,
        "shares": round(shares, 4),
        "my_probability": my_probability,
        "reasoning": reasoning,
        "outcome": None,
        "pnl": None,
        "resolved": False,
        "created_at": datetime.datetime.now().isoformat(),
    }
    trades.append(trade)
    settings["paper_balance"] = round(settings["paper_balance"] - size_usd, 4)
    _save(PAPER_TRADES_FILE, trades)
    _save_settings(settings)
    return trade


def resolve_paper_trade(trade_id: str, outcome: str) -> dict:
    trades = _load(PAPER_TRADES_FILE)
    settings = _load_settings()

    for t in trades:
        if t["id"] == trade_id:
            t["outcome"] = outcome.upper()
            t["resolved"] = True
            t["resolved_at"] = datetime.datetime.now().isoformat()

            won = (t["side"] == outcome.upper())
            if won:
                # 赢：每股收回 $1，减去买入成本
                proceeds = t["shares"] * 1.0
                t["pnl"] = round(proceeds - t["size_usd"], 4)
            else:
                t["pnl"] = -t["size_usd"]

            settings["paper_balance"] = round(
                settings["paper_balance"] + t["size_usd"] + t["pnl"], 4
            )
            _save(PAPER_TRADES_FILE, trades)
            _save_settings(settings)
            return t
    return {"error": "Not found"}


def get_paper_trades(resolved_only=False) -> list:
    trades = _load(PAPER_TRADES_FILE)
    if resolved_only:
        return [t for t in trades if t["resolved"]]
    return trades


def paper_trade_stats() -> dict:
    trades = _load(PAPER_TRADES_FILE)
    settings = _load_settings()
    resolved = [t for t in trades if t["resolved"]]

    if not resolved:
        return {
            "balance": settings["paper_balance"],
            "initial": settings["initial_balance"],
            "total_trades": len(trades),
            "resolved": 0,
            "win_rate": None,
            "total_pnl": 0,
            "roi": 0,
            "ready_for_phase3": False,
        }

    wins    = [t for t in resolved if t["pnl"] and t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in resolved if t["pnl"] is not None)
    win_rate  = len(wins) / len(resolved)
    roi = total_pnl / settings["initial_balance"]

    return {
        "balance": round(settings["paper_balance"], 2),
        "initial": settings["initial_balance"],
        "total_trades": len(trades),
        "open_trades": len([t for t in trades if not t["resolved"]]),
        "resolved": len(resolved),
        "wins": len(wins),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 4),
        "roi": round(roi, 4),
        "avg_pnl": round(total_pnl / len(resolved), 4),
        "ready_for_phase3": win_rate >= 0.55 and len(resolved) >= 20 and total_pnl > 0,
    }


def get_settings() -> dict:
    return _load_settings()


def reset_paper_account(balance: float = 1000.0):
    settings = _load_settings()
    settings["paper_balance"] = balance
    settings["initial_balance"] = balance
    _save_settings(settings)
    _save(PAPER_TRADES_FILE, [])
    return settings
