"""
Polymarket API client — market data, prices, history.
"""
from __future__ import annotations
import json
import httpx
from dataclasses import dataclass

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"


@dataclass
class Market:
    id: str
    slug: str
    question: str
    category: str
    volume: float
    liquidity: float
    yes_price: float   # current YES probability (0-1)
    no_price: float
    end_date: str
    active: bool
    closed: bool
    resolved: bool
    resolution: str    # "YES" | "NO" | ""

    @property
    def market_price(self) -> float:
        return self.yes_price

    @property
    def implied_prob(self) -> str:
        return f"{self.yes_price*100:.1f}%"


class PolymarketClient:
    def __init__(self):
        self._http = httpx.Client(timeout=15.0)

    def get_markets(self, limit=50, offset=0, category="", closed=False) -> list[dict]:
        params = {
            "limit": limit,
            "offset": offset,
            "active": "true",
            "closed": str(closed).lower(),
            "order": "volume24hr",
            "ascending": "false",
        }
        if category:
            params["tag"] = category
        r = self._http.get(f"{GAMMA_API}/markets", params=params)
        r.raise_for_status()
        return r.json()

    def get_market(self, slug: str) -> dict | None:
        r = self._http.get(f"{GAMMA_API}/markets", params={"slug": slug})
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None

    def get_tags(self) -> list[dict]:
        r = self._http.get(f"{GAMMA_API}/tags")
        r.raise_for_status()
        return r.json()

    def parse_market(self, raw: dict) -> Market:
        prices_raw = raw.get("outcomePrices", "[0.5,0.5]")
        if isinstance(prices_raw, str):
            prices = [float(p) for p in json.loads(prices_raw)]
        else:
            prices = [float(p) for p in prices_raw]
        yes_price = prices[0] if prices else 0.5
        no_price  = prices[1] if len(prices) > 1 else 1 - yes_price

        return Market(
            id=raw.get("id", ""),
            slug=raw.get("slug", ""),
            question=raw.get("question", ""),
            category=raw.get("groupItemTitle", raw.get("category", "")),
            volume=float(raw.get("volumeNum", 0) or 0),
            liquidity=float(raw.get("liquidityNum", 0) or 0),
            yes_price=yes_price,
            no_price=no_price,
            end_date=raw.get("endDate", "")[:10],
            active=raw.get("active", True),
            closed=raw.get("closed", False),
            resolved=bool(raw.get("resolved")),
            resolution=raw.get("resolution", ""),
        )

    def fetch_markets(self, limit=50, category="", closed=False) -> list[Market]:
        raws = self.get_markets(limit=limit, category=category, closed=closed)
        return [self.parse_market(r) for r in raws]

    def fetch_resolved(self, limit=30) -> list[Market]:
        """Fetch recently resolved markets for learning."""
        raws = self.get_markets(limit=limit, closed=True)
        return [self.parse_market(r) for r in raws]
