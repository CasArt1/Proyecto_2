from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Dict

@dataclass
class PortfolioConfig:
    initial_cash: float = 10_000.0
    fee_rate: float = 0.00125       # 0.125% per side
    max_long_pct: float = 0.95      # use up to 95% of cash for longs
    max_short_pct: float = 0.50     # use up to 50% of cash for shorts

@dataclass
class Trade:
    timestamp: Any
    side: str
    reason: str
    price: float
    qty: float
    notional: float
    fee: float
    cash_after: float
    position_after: float
    equity_after: float

class Portfolio:
    def __init__(self, cfg: PortfolioConfig):
        self.cfg = cfg
        self.cash = float(cfg.initial_cash)
        self.qty = 0.0                           # positive = long BTC, negative = short BTC
        self.entry_price: Optional[float] = None
        self.position_type = "none"              # {"none","long","short"}
        self.trade_log: List[Trade] = []
        self.history: List[Dict[str, float]] = []

    # ------- helpers -------
    def _equity(self, price: float) -> float:
        return self.cash + self.qty * price

    def get_portfolio_value(self, price: float) -> float:
        return self._equity(price)

    def _log(self, ts, side, reason, price, qty):
        notional = abs(qty) * float(price)
        fee = notional * self.cfg.fee_rate
        self.trade_log.append(
            Trade(ts, side, reason, float(price), float(qty), float(notional),
                  float(fee), float(self.cash), float(self.qty), float(self._equity(price)))
        )

    def mark_to_market(self, price: float, ts=None):
        self.history.append({
            "timestamp": ts, "price": float(price), "cash": float(self.cash),
            "qty": float(self.qty), "equity": float(self._equity(price)),
            "position_type": self.position_type,
            "entry_price": None if self.entry_price is None else float(self.entry_price)
        })

    # ------- dynamic sizing -------
    def _long_qty(self, price: float) -> float:
        budget = self.cash * self.cfg.max_long_pct
        return max(0.0, budget / (price * (1.0 + self.cfg.fee_rate))) if price > 0 else 0.0

    def _short_qty(self, price: float) -> float:
        budget = self.cash * self.cfg.max_short_pct
        return max(0.0, budget / (price * (1.0 - self.cfg.fee_rate))) if price > 0 else 0.0

    # ------- order actions -------
    def open_long(self, price: float, ts=None) -> bool:
        if self.position_type != "none": return False
        qty = self._long_qty(price)
        if qty <= 0: return False
        notional = qty * price
        fee = notional * self.cfg.fee_rate
        cost = notional + fee
        if cost > self.cash: return False
        self.cash -= cost
        self.qty += qty
        self.entry_price = float(price)
        self.position_type = "long"
        self._log(ts, "BUY", "open_long", price, qty)
        return True

    def open_short(self, price: float, ts=None) -> bool:
        if self.position_type != "none": return False
        qty = self._short_qty(price)
        if qty <= 0: return False
        notional = qty * price
        fee = notional * self.cfg.fee_rate
        proceeds = notional - fee
        self.cash += proceeds
        self.qty -= qty
        self.entry_price = float(price)
        self.position_type = "short"
        self._log(ts, "SELL", "open_short", price, -qty)
        return True

    def close_position(self, price: float, ts=None, reason="close") -> bool:
        # close long
        if self.position_type == "long" and self.qty > 0:
            qty = self.qty
            notional = qty * price
            fee = notional * self.cfg.fee_rate
            self.cash += (notional - fee)
            self.qty = 0.0
            self.position_type = "none"
            self.entry_price = None
            self._log(ts, "SELL", reason, price, -qty)
            return True
        # close short
        if self.position_type == "short" and self.qty < 0:
            qty = abs(self.qty)
            notional = qty * price
            fee = notional * self.cfg.fee_rate
            cost = notional + fee
            if cost > self.cash:  # safety
                cost = self.cash
            self.cash -= cost
            self.qty = 0.0
            self.position_type = "none"
            self.entry_price = None
            self._log(ts, "BUY", reason, price, qty)
            return True
        return False

# --- self-test: python -m src.portfolio ---
if __name__ == "__main__":
    p = Portfolio(PortfolioConfig())
    ts = "2024-01-01 00:00"
    print("Open long @ 50,000:", p.open_long(50_000, ts))
    p.mark_to_market(51_000, ts)
    print("Equity after +2% move:", round(p.get_portfolio_value(51_000), 2))
    print("Close:", p.close_position(51_000, ts, "test_close"))
    print("Trades:", len(p.trade_log))
