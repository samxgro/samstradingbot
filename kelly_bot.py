"""
Turbine Kelly Bot

Ports the Kelly criterion + probability framework from polymarket_bot to
Turbine's 15-minute binary prediction markets.

Algorithm:
  1. Fetch live price from Pyth Network (same oracle Turbine uses)
  2. Estimate P(price > strike at expiry) using log-normal model:
       d = ln(S/K) / (σ * √T_remaining)
       P(YES) = Φ(d)
  3. Get the market's implied probability from the YES orderbook mid
  4. Compute edge and Kelly fraction:
       abs_edge = model_prob - market_prob
       kelly_f  = abs_edge / (1 - market_prob)        [YES side]
       pct_edge = abs_edge / market_prob               [ranking metric]
  5. Trade if pct_edge >= min_pct_edge and abs_edge >= min_abs_edge
  6. Size position = bankroll * fractional_kelly (capped at max_position)

Key insight (from polymarket_bot/kelly.py): ranking by % edge, not absolute
edge, maximizes geometric growth. A 233% edge at 3c is better than 22% at 45c.

Volatility: estimated live from rolling Pyth price samples. Falls back to
DEFAULT_VOL_15MIN if not enough history yet.

Usage:
    python kelly_bot.py
    python kelly_bot.py --bankroll 50 --kelly-scalar 0.25 --min-pct-edge 0.15
    python kelly_bot.py --assets BTC,ETH
"""

import argparse
import asyncio
import math
import os
import re
import statistics
import time
from pathlib import Path
from dotenv import load_dotenv
import httpx

from turbine_client import TurbineClient, Outcome, Side, QuickMarket
from turbine_client.exceptions import TurbineApiError

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
CLAIM_ONLY_MODE = os.environ.get("CLAIM_ONLY_MODE", "false").lower() == "true"
CHAIN_ID = int(os.environ.get("CHAIN_ID", "84532"))
TURBINE_HOST = os.environ.get("TURBINE_HOST", "http://localhost:8080")

# Kelly parameters (from polymarket_bot/kelly.py)
DEFAULT_BANKROLL_USDC = 50.0        # Total capital Kelly sizes from
DEFAULT_KELLY_SCALAR = 0.25         # Fractional Kelly — 25% of full Kelly
DEFAULT_MIN_PCT_EDGE = 0.15         # Min 15% expected return on capital (pct_edge)
DEFAULT_MIN_ABS_EDGE = 0.03         # Min 3% absolute edge
DEFAULT_MAX_POSITION_USDC = 10.0    # Hard cap per asset per market
MIN_ORDER_USDC = 1.0                # Turbine API minimum for taker orders

# Volatility model
DEFAULT_VOL_15MIN = 0.004           # 0.4% per 15 min fallback (≈ 75% annualized)
VOL_LOOKBACK_N = 30                 # Number of price samples to compute live vol

# Timing
PRICE_POLL_SECONDS = 5              # How often to fetch prices and evaluate

# Pyth Network Hermes API
PYTH_HERMES_URL = "https://hermes.pyth.network/v2/updates/price/latest"
PYTH_FEED_IDS = {
    "BTC": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
}
SUPPORTED_ASSETS = list(PYTH_FEED_IDS.keys())


# ============================================================
# PROBABILITY + KELLY MATH (no scipy — uses math.erfc)
# ============================================================

def norm_cdf(x: float) -> float:
    """Standard normal CDF. Exact equivalent of scipy.stats.norm.cdf."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def estimate_prob_above_strike(
    current_price: float,
    strike: float,
    vol_15min: float,
    time_fraction: float,   # fraction of the 15-min window remaining [0, 1]
) -> float:
    """
    P(price > strike at expiry) under log-normal model, zero drift.

    For sub-15-minute horizons drift is negligible. Uses:
        d = ln(S/K) / (σ * √T_fraction)
        P(S_T > K) = Φ(d)

    This is the same form used in polymarket_bot/models/probability.py
    (Brownian first-passage), adapted for price-above-strike rather than
    price-below-depeg-threshold.
    """
    if vol_15min <= 0 or time_fraction <= 0:
        return 1.0 if current_price > strike else 0.0
    effective_vol = vol_15min * math.sqrt(time_fraction)
    if effective_vol < 1e-9:
        return 1.0 if current_price > strike else 0.0
    d = math.log(current_price / strike) / effective_vol
    return float(norm_cdf(d))


def kelly_f_yes(model_prob: float, market_price: float) -> float:
    """
    Full Kelly fraction for YES side.
    f* = (p - market_price) / (1 - market_price)
    Positive only when model_prob > market_price.
    """
    if market_price >= 1.0 or market_price <= 0.0:
        return 0.0
    return (model_prob - market_price) / (1.0 - market_price)


def kelly_f_no(model_prob: float, market_price_no: float) -> float:
    """
    Full Kelly fraction for NO side.
    f* = ((1 - model_prob) - market_price_no) / (1 - market_price_no)
    """
    no_prob = 1.0 - model_prob
    if market_price_no >= 1.0 or market_price_no <= 0.0:
        return 0.0
    return (no_prob - market_price_no) / (1.0 - market_price_no)


def pct_edge_for(true_prob: float, market_price: float) -> float:
    """
    Expected % return on capital at risk.
    Primary ranking metric — maximizes geometric growth (from kelly.py).
    """
    if market_price <= 0:
        return 0.0
    return (true_prob - market_price) / market_price


# ============================================================
# CREDENTIAL HELPERS (unchanged from price_action_bot.py)
# ============================================================

def get_or_create_api_credentials(env_path: Path = None):
    """Get existing credentials or register new ones and save to .env."""
    if env_path is None:
        env_path = Path(__file__).parent / ".env"

    api_key_id = os.environ.get("TURBINE_API_KEY_ID")
    api_private_key = os.environ.get("TURBINE_API_PRIVATE_KEY")

    if api_key_id and api_private_key:
        print("Using existing API credentials")
        return api_key_id, api_private_key

    private_key = os.environ.get("TURBINE_PRIVATE_KEY")
    if not private_key:
        raise ValueError("Set TURBINE_PRIVATE_KEY in your .env file")

    print("Registering new API credentials...")
    credentials = TurbineClient.request_api_credentials(
        host=TURBINE_HOST,
        private_key=private_key,
    )

    api_key_id = credentials["api_key_id"]
    api_private_key = credentials["api_private_key"]

    _save_credentials_to_env(env_path, api_key_id, api_private_key)
    os.environ["TURBINE_API_KEY_ID"] = api_key_id
    os.environ["TURBINE_API_PRIVATE_KEY"] = api_private_key

    print(f"API credentials saved to {env_path}")
    return api_key_id, api_private_key


def _save_credentials_to_env(env_path: Path, api_key_id: str, api_private_key: str):
    """Save API credentials to .env file."""
    env_path = Path(env_path)
    if env_path.exists():
        content = env_path.read_text()
        if "TURBINE_API_KEY_ID=" in content:
            content = re.sub(r'^TURBINE_API_KEY_ID=.*$', f'TURBINE_API_KEY_ID={api_key_id}', content, flags=re.MULTILINE)
        else:
            content = content.rstrip() + f"\nTURBINE_API_KEY_ID={api_key_id}"
        if "TURBINE_API_PRIVATE_KEY=" in content:
            content = re.sub(r'^TURBINE_API_PRIVATE_KEY=.*$', f'TURBINE_API_PRIVATE_KEY={api_private_key}', content, flags=re.MULTILINE)
        else:
            content = content.rstrip() + f"\nTURBINE_API_PRIVATE_KEY={api_private_key}"
        env_path.write_text(content + "\n")
    else:
        content = (
            f"# Turbine Bot Config\n"
            f"TURBINE_PRIVATE_KEY={os.environ.get('TURBINE_PRIVATE_KEY', '')}\n"
            f"TURBINE_API_KEY_ID={api_key_id}\n"
            f"TURBINE_API_PRIVATE_KEY={api_private_key}\n"
        )
        env_path.write_text(content)


# ============================================================
# STATE
# ============================================================

class AssetState:
    """Per-asset trading state."""

    def __init__(self, asset: str):
        self.asset = asset
        self.market_id: str | None = None
        self.settlement_address: str | None = None
        self.contract_address: str | None = None
        self.strike_price: int = 0          # 6-decimal (divide by 1e6 for USD)
        self.end_time: int = 0              # Unix timestamp — used for time remaining
        self.position_usdc: dict[str, float] = {}
        self.active_orders: dict[str, str] = {}
        self.processed_trade_ids: set[int] = set()
        self.pending_order_txs: set[str] = set()
        self.traded_markets: dict[str, str] = {}   # market_id -> contract_address

        # Live vol estimation (updated each price poll)
        self.price_history: list[float] = []
        self.vol_15min: float = DEFAULT_VOL_15MIN


# ============================================================
# BOT
# ============================================================

class KellyBot:
    """
    Kelly criterion bot for Turbine 15-minute binary markets.

    Replaces fixed order sizing with probability-weighted Kelly sizing.
    Infrastructure (USDC approval, market transitions, claiming) is
    identical to price_action_bot.py.
    """

    def __init__(
        self,
        client: TurbineClient,
        assets: list[str],
        bankroll_usdc: float = DEFAULT_BANKROLL_USDC,
        kelly_scalar: float = DEFAULT_KELLY_SCALAR,
        min_pct_edge: float = DEFAULT_MIN_PCT_EDGE,
        min_abs_edge: float = DEFAULT_MIN_ABS_EDGE,
        max_position_usdc: float = DEFAULT_MAX_POSITION_USDC,
        dry_run: bool = False,
    ):
        self.client = client
        self.assets = assets
        self.bankroll_usdc = bankroll_usdc
        self.kelly_scalar = kelly_scalar
        self.min_pct_edge = min_pct_edge
        self.min_abs_edge = min_abs_edge
        self.max_position_usdc = max_position_usdc
        self.dry_run = dry_run
        self.running = True

        # Dry-run simulation state
        self.sim_balance: float = bankroll_usdc
        self.sim_pnl: float = 0.0
        # {market_id: {"outcome": Outcome, "price": float, "usdc": float, "shares": float}}
        self.sim_positions: dict[str, dict] = {}

        self.asset_states: dict[str, AssetState] = {
            asset: AssetState(asset) for asset in assets
        }
        self.approved_settlements: dict[str, int] = {}
        self._http_client: httpx.AsyncClient | None = None

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    MAX_APPROVAL_THRESHOLD = (2**256 - 1) // 2

    def get_position_usdc(self, state: AssetState, market_id: str) -> float:
        return state.position_usdc.get(market_id, 0.0)

    def can_trade(self, state: AssetState, usdc_amount: float) -> bool:
        current = self.get_position_usdc(state, state.market_id)
        return (current + usdc_amount) <= self.max_position_usdc

    def calculate_shares_from_usdc(self, usdc_amount: float, price: int) -> int:
        """Shares in 6 decimals from USDC amount at given price."""
        if price <= 0:
            return 0
        return math.ceil((usdc_amount * 1_000_000 * 1_000_000) / price)

    # ----------------------------------------------------------
    # Probability + Kelly signal
    # ----------------------------------------------------------

    def _update_vol_estimate(self, state: AssetState, current_price: float) -> None:
        """Update rolling vol estimate from price history."""
        state.price_history.append(current_price)
        if len(state.price_history) > VOL_LOOKBACK_N:
            state.price_history.pop(0)

        if len(state.price_history) >= 4:
            log_returns = [
                math.log(state.price_history[i] / state.price_history[i - 1])
                for i in range(1, len(state.price_history))
            ]
            per_sample_vol = statistics.stdev(log_returns)
            # Scale per-sample vol to 15-min vol
            # Each sample = PRICE_POLL_SECONDS; 15 min = 900 sec
            scale = math.sqrt(900.0 / PRICE_POLL_SECONDS)
            state.vol_15min = max(per_sample_vol * scale, 1e-5)

    def _get_market_implied_prob(self, state: AssetState) -> float:
        """
        Fetch YES mid-price from orderbook as market's implied P(YES).
        Falls back to 0.5 if orderbook is unavailable or empty.
        """
        try:
            orderbook = self.client.get_orderbook(state.market_id)

            # Orderbook returns lists of orders; we need YES bids and asks.
            # Prices are in 6-decimal format (500000 = 50%).
            yes_bids: list[int] = []
            yes_asks: list[int] = []

            orders = orderbook if isinstance(orderbook, list) else getattr(orderbook, "orders", [])
            for o in orders:
                # Support both dict and object formats
                if isinstance(o, dict):
                    outcome_val = o.get("outcome", -1)
                    side_val = o.get("side", -1)
                    price_val = o.get("price", 0)
                else:
                    outcome_val = getattr(o, "outcome", -1)
                    side_val = getattr(o, "side", -1)
                    price_val = getattr(o, "price", 0)

                is_yes = (outcome_val == 0 or outcome_val == Outcome.YES)
                is_buy = (side_val == 0 or side_val == Side.BUY)

                if is_yes and is_buy:
                    yes_bids.append(int(price_val))
                elif is_yes and not is_buy:
                    yes_asks.append(int(price_val))

            if yes_bids and yes_asks:
                mid = (max(yes_bids) + min(yes_asks)) / 2
                return mid / 1_000_000
            elif yes_asks:
                return min(yes_asks) / 1_000_000
            elif yes_bids:
                return max(yes_bids) / 1_000_000

        except Exception:
            pass

        return 0.5  # fallback: assume 50/50 market

    async def calculate_signal(
        self, state: AssetState, current_price: float
    ) -> tuple[str, float, int]:
        """
        Compute Kelly signal for one asset.

        Returns (action, kelly_fraction, limit_price_6dec).
        action is one of: "BUY_YES", "BUY_NO", "HOLD"
        kelly_fraction drives position sizing: position = bankroll * kelly_fraction
        limit_price is the maker limit order price in 6-decimal format.
        """
        if current_price <= 0 or state.strike_price <= 0:
            return "HOLD", 0.0, 0

        strike = state.strike_price / 1e6

        # Update rolling vol from latest price sample
        self._update_vol_estimate(state, current_price)

        # Time remaining in this market (fraction of 15-min window)
        now = time.time()
        if state.end_time > 0:
            remaining_sec = max(30.0, state.end_time - now)
        else:
            remaining_sec = 450.0  # assume ~7.5 min if unknown
        time_fraction = min(1.0, remaining_sec / 900.0)

        # Model probability: P(price > strike at expiry)
        model_prob_yes = estimate_prob_above_strike(
            current_price=current_price,
            strike=strike,
            vol_15min=state.vol_15min,
            time_fraction=time_fraction,
        )

        # Market's implied probability from orderbook
        market_prob_yes = self._get_market_implied_prob(state)
        market_prob_no = 1.0 - market_prob_yes

        # ---- Evaluate YES and NO edges ----
        yes_true = model_prob_yes
        no_true = 1.0 - model_prob_yes

        yes_kelly = kelly_f_yes(yes_true, market_prob_yes)
        no_kelly = kelly_f_no(yes_true, market_prob_no)

        yes_pct = pct_edge_for(yes_true, market_prob_yes)
        no_pct = pct_edge_for(no_true, market_prob_no)

        yes_abs = yes_true - market_prob_yes
        no_abs = no_true - market_prob_no

        yes_tradeable = (
            yes_kelly > 0
            and yes_abs >= self.min_abs_edge
            and yes_pct >= self.min_pct_edge
        )
        no_tradeable = (
            no_kelly > 0
            and no_abs >= self.min_abs_edge
            and no_pct >= self.min_pct_edge
        )

        # Pick the higher % edge (geometric growth maximization)
        if yes_tradeable and (not no_tradeable or yes_pct >= no_pct):
            frac = yes_kelly * self.kelly_scalar
            limit_price = int((market_prob_yes - 0.015) * 1_000_000)
            limit_price = max(10_000, min(limit_price, 990_000))
            print(
                f"[{state.asset}] ${current_price:,.2f} | model={model_prob_yes:.3f} "
                f"market={market_prob_yes:.3f} | BUY_YES "
                f"abs={yes_abs:+.3f} pct={yes_pct:+.1%} kelly={frac:.3f} "
                f"vol={state.vol_15min:.4f} T={time_fraction:.2f}"
            )
            return "BUY_YES", frac, limit_price

        elif no_tradeable:
            frac = no_kelly * self.kelly_scalar
            limit_price = int((market_prob_no - 0.015) * 1_000_000)
            limit_price = max(10_000, min(limit_price, 990_000))
            print(
                f"[{state.asset}] ${current_price:,.2f} | model={model_prob_yes:.3f} "
                f"market={market_prob_yes:.3f} | BUY_NO "
                f"abs={no_abs:+.3f} pct={no_pct:+.1%} kelly={frac:.3f} "
                f"vol={state.vol_15min:.4f} T={time_fraction:.2f}"
            )
            return "BUY_NO", frac, limit_price

        else:
            print(
                f"[{state.asset}] ${current_price:,.2f} | model={model_prob_yes:.3f} "
                f"market={market_prob_yes:.3f} | HOLD "
                f"(yes_pct={yes_pct:+.1%} no_pct={no_pct:+.1%}) "
                f"vol={state.vol_15min:.4f}"
            )
            return "HOLD", 0.0, 0

    # ----------------------------------------------------------
    # Order execution
    # ----------------------------------------------------------

    async def execute_signal(
        self, state: AssetState, action: str, kelly_fraction: float, limit_price: int
    ) -> None:
        """
        Post a Kelly-sized resting limit order.

        Position size = bankroll * kelly_fraction, capped at max_position_usdc.
        Minimum enforced at MIN_ORDER_USDC (Turbine API requirement).
        Order verification runs as a background task (non-blocking).
        """
        if action == "HOLD" or kelly_fraction <= 0 or limit_price <= 0:
            return

        # Kelly-sized position
        position_usdc = min(
            kelly_fraction * self.bankroll_usdc,
            self.max_position_usdc,
        )
        position_usdc = max(position_usdc, MIN_ORDER_USDC)

        if self.dry_run:
            if position_usdc > self.sim_balance:
                print(f"[{state.asset}] [SIM] Insufficient sim balance: ${self.sim_balance:.3f}")
                return
            outcome = Outcome.YES if action == "BUY_YES" else Outcome.NO
            price = limit_price / 1_000_000
            shares = position_usdc / price
            self.sim_balance -= position_usdc
            self.sim_positions[state.market_id] = {
                "outcome": outcome, "price": price,
                "usdc": position_usdc, "shares": shares, "asset": state.asset,
            }
            print(
                f"[{state.asset}] [SIM] {action} @ {price:.1%} — "
                f"${position_usdc:.3f} → {shares:.4f} shares | "
                f"sim balance: ${self.sim_balance:.3f} | pnl: ${self.sim_pnl:+.3f}"
            )
            return

        if not self.can_trade(state, position_usdc):
            current = self.get_position_usdc(state, state.market_id)
            print(f"[{state.asset}] Position cap: ${current:.2f} / ${self.max_position_usdc:.2f}")
            return

        outcome = Outcome.YES if action == "BUY_YES" else Outcome.NO

        shares = self.calculate_shares_from_usdc(position_usdc, limit_price)
        if shares <= 0:
            return

        # Balance check
        try:
            usdc_balance = self.client.get_usdc_balance()
            balance_usdc = usdc_balance / 1_000_000
            if balance_usdc < position_usdc:
                print(
                    f"[{state.asset}] Low USDC: ${balance_usdc:.2f} < ${position_usdc:.2f} needed. "
                    f"Fund: {self.client.address}"
                )
                return
        except Exception:
            pass

        try:
            order = self.client.create_limit_buy(
                market_id=state.market_id,
                outcome=outcome,
                price=limit_price,
                size=shares,
                expiration=int(time.time()) + 600,
                settlement_address=state.settlement_address,
            )

            result = self.client.post_order(order)
            outcome_str = "YES" if outcome == Outcome.YES else "NO"

            if result and isinstance(result, dict):
                status = result.get("status", "unknown")
                order_hash = result.get("orderHash", order.order_hash)
                print(
                    f"[{state.asset}] -> {outcome_str} @ {limit_price / 10000:.1f}% "
                    f"${position_usdc:.2f} = {shares / 1_000_000:.4f} shares "
                    f"(status: {status})"
                )
                asyncio.create_task(self._verify_order(state, order_hash, action, shares))
            else:
                print(f"[{state.asset}] Unexpected order response: {result}")

        except TurbineApiError as e:
            print(f"[{state.asset}] Order failed: {e}")
        except Exception as e:
            print(f"[{state.asset}] Unexpected error: {e}")

    async def _verify_order(self, state: AssetState, order_hash: str, action: str, shares: int) -> None:
        """Background task: check order status 2s after submission."""
        try:
            await asyncio.sleep(2)

            try:
                failed_trades = self.client.get_failed_trades()
                my_failed = [
                    t for t in failed_trades
                    if t.market_id == state.market_id
                    and t.buyer_address.lower() == self.client.address.lower()
                    and t.fill_size == shares
                ]
                if my_failed:
                    reason = my_failed[0].reason
                    if "simulation" in reason.lower():
                        try:
                            bal = self.client.get_usdc_balance() / 1_000_000
                            reason += f" (USDC: ${bal:.2f})"
                        except Exception:
                            pass
                    print(f"[{state.asset}] Order FAILED: {reason}")
                    return
            except Exception as e:
                print(f"  [{state.asset}] Warning: failed trade check: {e}")

            try:
                pending_trades = self.client.get_pending_trades()
                my_pending = [
                    t for t in pending_trades
                    if t.market_id == state.market_id
                    and t.buyer_address.lower() == self.client.address.lower()
                    and t.fill_size == shares
                ]
                if my_pending:
                    print(f"[{state.asset}] Order PENDING (TX: {my_pending[0].tx_hash[:16]}...)")
                    state.pending_order_txs.add(my_pending[0].tx_hash)
                    return
            except Exception as e:
                print(f"  [{state.asset}] Warning: pending trade check: {e}")

            try:
                trades = self.client.get_trades(market_id=state.market_id, limit=20)
                recent_threshold = time.time() - 10
                my_trades = [
                    t for t in trades
                    if t.buyer.lower() == self.client.address.lower()
                    and t.timestamp > recent_threshold
                    and t.id not in state.processed_trade_ids
                ]
                if my_trades:
                    trade = my_trades[0]
                    state.processed_trade_ids.add(trade.id)
                    usdc_spent = (trade.size * trade.price) / (1_000_000 * 1_000_000)
                    state.position_usdc[state.market_id] = self.get_position_usdc(state, state.market_id) + usdc_spent
                    print(f"[{state.asset}] FILLED: ${usdc_spent:.2f} -> {trade.size / 1_000_000:.4f} shares")
                    return
            except Exception as e:
                print(f"  [{state.asset}] Warning: trade check: {e}")

            try:
                my_orders = self.client.get_orders(trader=self.client.address, market_id=state.market_id)
                matching = [o for o in my_orders if o.order_hash == order_hash]
                if matching:
                    print(f"[{state.asset}] Order OPEN on orderbook")
                    state.active_orders[order_hash] = action
                else:
                    print(f"[{state.asset}] Order not found — may have been rejected")
            except Exception as e:
                print(f"  [{state.asset}] Warning: order check: {e}")

        except Exception as e:
            print(f"  [{state.asset}] Verification error: {e}")

    # ----------------------------------------------------------
    # Async HTTP client
    # ----------------------------------------------------------

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()

    # ----------------------------------------------------------
    # USDC approval (gasless, one-time per settlement)
    # ----------------------------------------------------------

    def ensure_settlement_approved(self, settlement_address: str) -> None:
        if self.dry_run or settlement_address in self.approved_settlements:
            return

        current_allowance = self.client.get_usdc_allowance(spender=settlement_address)
        if current_allowance >= self.MAX_APPROVAL_THRESHOLD:
            print(f"  Existing USDC max approval found")
            self.approved_settlements[settlement_address] = current_allowance
            return

        print(f"\n{'='*50}")
        print(f"GASLESS USDC APPROVAL (one-time max permit)")
        print(f"{'='*50}")
        print(f"Settlement: {settlement_address}")

        try:
            result = self.client.approve_usdc_for_settlement(settlement_address)
            tx_hash = result.get("tx_hash", "unknown")
            print(f"Relayer TX: {tx_hash}")
            print("Waiting for confirmation...")

            for _ in range(30):
                try:
                    allowance = self.client.get_usdc_allowance(spender=settlement_address)
                    if allowance >= self.MAX_APPROVAL_THRESHOLD:
                        print(f"Max USDC approval confirmed (gasless)")
                        self.approved_settlements[settlement_address] = allowance
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                print(f"Approval pending (may still confirm)")
                self.approved_settlements[settlement_address] = 2**256 - 1

        except Exception as e:
            print(f"Gasless approval failed: {e}")
            raise

        print(f"{'='*50}\n")

    # ----------------------------------------------------------
    # Pending order cleanup
    # ----------------------------------------------------------

    async def cleanup_pending_orders(self, state: AssetState) -> None:
        try:
            pending_trades = self.client.get_pending_trades()
            pending_txs = {
                t.tx_hash for t in pending_trades
                if t.market_id == state.market_id
                and t.buyer_address.lower() == self.client.address.lower()
            }
            resolved_txs = state.pending_order_txs - pending_txs
            if resolved_txs:
                print(f"  [{state.asset}] {len(resolved_txs)} order(s) settled")
                state.pending_order_txs -= resolved_txs
                trades = self.client.get_trades(market_id=state.market_id, limit=20)
                my_recent = [
                    t for t in trades
                    if t.buyer.lower() == self.client.address.lower()
                    and t.id not in state.processed_trade_ids
                ]
                for trade in my_recent:
                    state.processed_trade_ids.add(trade.id)
                    usdc_spent = (trade.size * trade.price) / (1_000_000 * 1_000_000)
                    state.position_usdc[state.market_id] = self.get_position_usdc(state, state.market_id) + usdc_spent
                    outcome_str = "YES" if trade.outcome == 0 else "NO"
                    print(f"  [{state.asset}] Filled: ${usdc_spent:.2f} -> {trade.size / 1_000_000:.2f} {outcome_str}")
        except Exception as e:
            print(f"  [{state.asset}] Warning: pending order cleanup: {e}")

    async def sync_position(self, state: AssetState) -> None:
        if not state.market_id:
            return
        try:
            positions = self.client.get_user_positions(
                address=self.client.address, chain_id=self.client.chain_id
            )
            for position in positions:
                if position.market_id == state.market_id:
                    total_shares = position.yes_shares + position.no_shares
                    estimated_usdc = (total_shares * 0.5) / 1_000_000
                    if total_shares > 0:
                        print(f"[{state.asset}] Position synced: ~${estimated_usdc:.2f} USDC in shares")
                        state.position_usdc[state.market_id] = estimated_usdc
                    else:
                        state.position_usdc[state.market_id] = 0.0
                    return
            state.position_usdc[state.market_id] = 0.0
        except Exception as e:
            print(f"[{state.asset}] Failed to sync position: {e}")
            state.position_usdc[state.market_id] = 0.0

    # ----------------------------------------------------------
    # Price fetching (Pyth)
    # ----------------------------------------------------------

    async def get_current_prices(self) -> dict[str, float]:
        try:
            http_client = await self._get_http_client()
            feed_ids = [PYTH_FEED_IDS[asset] for asset in self.assets]
            response = await http_client.get(
                PYTH_HERMES_URL,
                params=[("ids[]", fid) for fid in feed_ids],
            )
            response.raise_for_status()
            data = response.json()

            prices: dict[str, float] = {}
            if not data.get("parsed"):
                return prices

            feed_to_asset = {PYTH_FEED_IDS[asset]: asset for asset in self.assets}
            for parsed in data["parsed"]:
                feed_id = "0x" + parsed["id"]
                asset = feed_to_asset.get(feed_id)
                if asset:
                    price_data = parsed["price"]
                    price_int = int(price_data["price"])
                    expo = price_data["expo"]
                    prices[asset] = price_int * (10 ** expo)

            return prices
        except Exception as e:
            print(f"Failed to fetch prices from Pyth: {e}")
            return {}

    # ----------------------------------------------------------
    # Main trading loop
    # ----------------------------------------------------------

    async def trading_loop(self) -> None:
        if CLAIM_ONLY_MODE:
            print("CLAIM ONLY MODE — trading disabled")
            while self.running:
                await asyncio.sleep(60)
            return

        while self.running:
            try:
                prices = await self.get_current_prices()

                for asset in self.assets:
                    state = self.asset_states[asset]
                    if not state.market_id:
                        continue

                    if state.pending_order_txs:
                        await self.cleanup_pending_orders(state)

                    current_price = prices.get(asset, 0.0)
                    if current_price <= 0:
                        continue

                    action, kelly_frac, limit_price = await self.calculate_signal(state, current_price)

                    if action != "HOLD":
                        await self.execute_signal(state, action, kelly_frac, limit_price)

                await asyncio.sleep(PRICE_POLL_SECONDS)

            except Exception as e:
                print(f"Trading loop error: {e}")
                await asyncio.sleep(PRICE_POLL_SECONDS)

    # ----------------------------------------------------------
    # Market management
    # ----------------------------------------------------------

    async def get_active_market(self, asset: str) -> tuple[str, int, int] | None:
        response = self.client._http.get(f"/api/v1/quick-markets/{asset}")
        quick_market_data = response.get("quickMarket")
        if not quick_market_data:
            return None
        quick_market = QuickMarket.from_dict(quick_market_data)
        return quick_market.market_id, quick_market.end_time, quick_market.start_price

    async def switch_to_new_market(
        self, state: AssetState, new_market_id: str, end_time: int = 0, start_price: int = 0
    ) -> None:
        old_market_id = state.market_id

        if old_market_id and state.contract_address:
            state.traded_markets[old_market_id] = state.contract_address

        if old_market_id:
            print(f"\n{'='*50}")
            print(f"[{state.asset}] MARKET TRANSITION")
            print(f"Old: {old_market_id[:8]}... | New: {new_market_id[:8]}...")
            print(f"{'='*50}\n")
            await self.cancel_asset_orders(state)

        state.market_id = new_market_id
        state.strike_price = start_price
        state.end_time = end_time
        state.active_orders = {}
        state.processed_trade_ids.clear()
        state.pending_order_txs.clear()
        state.price_history.clear()  # reset vol estimation for new market

        try:
            markets = self.client.get_markets()
            for market in markets:
                if market.id == new_market_id:
                    state.settlement_address = market.settlement_address
                    try:
                        stats = self.client.get_market(new_market_id)
                        state.contract_address = stats.contract_address
                    except Exception:
                        pass
                    break
        except Exception as e:
            print(f"[{state.asset}] Warning: could not fetch market addresses: {e}")

        if state.settlement_address:
            self.ensure_settlement_approved(state.settlement_address)

        strike_usd = start_price / 1e6 if start_price else 0
        print(f"[{state.asset}] Market: {new_market_id[:8]}... | Strike: ${strike_usd:,.2f}")

        await self.sync_position(state)

    async def monitor_market_transitions(self) -> None:
        while self.running:
            try:
                for asset in self.assets:
                    state = self.asset_states[asset]
                    try:
                        market_info = await self.get_active_market(asset)
                    except Exception as e:
                        print(f"[{asset}] Market monitor error: {e}")
                        continue

                    if not market_info:
                        continue

                    new_market_id, end_time, start_price = market_info
                    if new_market_id != state.market_id:
                        await self.switch_to_new_market(state, new_market_id, end_time, start_price)

            except Exception as e:
                print(f"Market monitor error: {e}")

            await asyncio.sleep(5)

    async def cancel_all_orders(self) -> None:
        try:
            open_orders = self.client.get_orders(trader=self.client.address, status="open")
        except Exception as e:
            print(f"Failed to fetch open orders: {e}")
            return

        if not open_orders:
            return

        print(f"Cancelling {len(open_orders)} orders...")
        for order in open_orders:
            try:
                self.client.cancel_order(
                    order.order_hash,
                    market_id=order.market_id,
                    side=Side(order.side),
                )
            except TurbineApiError as e:
                if "404" not in str(e):
                    print(f"Failed to cancel order: {e}")

        for state in self.asset_states.values():
            state.active_orders.clear()

    async def cancel_asset_orders(self, state: AssetState) -> None:
        if not state.market_id:
            return
        try:
            open_orders = self.client.get_orders(
                trader=self.client.address, market_id=state.market_id, status="open"
            )
        except Exception as e:
            print(f"[{state.asset}] Failed to fetch open orders: {e}")
            return

        if not open_orders:
            return

        print(f"[{state.asset}] Cancelling {len(open_orders)} orders...")
        for order in open_orders:
            try:
                self.client.cancel_order(
                    order.order_hash,
                    market_id=order.market_id,
                    side=Side(order.side),
                )
            except TurbineApiError as e:
                if "404" not in str(e):
                    print(f"[{state.asset}] Failed to cancel: {e}")
        state.active_orders.clear()

    # ----------------------------------------------------------
    # Claiming
    # ----------------------------------------------------------

    async def claim_resolved_markets(self) -> None:
        while self.running:
            try:
                if self.dry_run:
                    for market_id, pos in list(self.sim_positions.items()):
                        try:
                            resolution = self.client.get_resolution(market_id)
                            if resolution and resolution.resolved:
                                won = (pos["outcome"] == Outcome.YES and resolution.result == "YES") or \
                                      (pos["outcome"] == Outcome.NO and resolution.result == "NO")
                                payout = pos["shares"] if won else 0.0
                                trade_pnl = payout - pos["usdc"]
                                self.sim_pnl += trade_pnl
                                self.sim_balance += payout
                                result_str = "WON" if won else "LOST"
                                print(
                                    f"[{pos['asset']}] [SIM] Market resolved {resolution.result} — "
                                    f"{result_str} ${trade_pnl:+.3f} | "
                                    f"sim balance: ${self.sim_balance:.3f} | total pnl: ${self.sim_pnl:+.3f}"
                                )
                                del self.sim_positions[market_id]
                        except Exception:
                            continue
                    await asyncio.sleep(120)
                    continue

                all_traded: list[tuple[str, str, AssetState]] = []
                for state in self.asset_states.values():
                    for market_id, contract_address in list(state.traded_markets.items()):
                        all_traded.append((market_id, contract_address, state))

                if not all_traded:
                    await asyncio.sleep(120)
                    continue

                resolved: list[tuple[str, str, AssetState]] = []
                for market_id, contract_address, state in all_traded:
                    try:
                        resolution = self.client.get_resolution(market_id)
                        if resolution and resolution.resolved:
                            resolved.append((market_id, contract_address, state))
                    except Exception:
                        continue

                if not resolved:
                    await asyncio.sleep(120)
                    continue

                market_addresses = [addr for _, addr, _ in resolved]
                try:
                    result = self.client.batch_claim_winnings(market_addresses)
                    tx_hash = result.get("txHash", result.get("tx_hash", "unknown"))
                    print(f"Claimed {len(resolved)} markets TX: {tx_hash}")
                    for market_id, _, state in resolved:
                        del state.traded_markets[market_id]
                except ValueError as e:
                    if "no winning tokens" in str(e).lower():
                        for market_id, _, state in resolved:
                            del state.traded_markets[market_id]
                except Exception as e:
                    print(f"Batch claim error: {e}")

            except Exception as e:
                print(f"Claim monitor error: {e}")

            await asyncio.sleep(120)

    # ----------------------------------------------------------
    # Entry point
    # ----------------------------------------------------------

    async def run(self) -> None:
        monitor_task = asyncio.create_task(self.monitor_market_transitions())
        claim_task = asyncio.create_task(self.claim_resolved_markets())
        trade_task = asyncio.create_task(self.trading_loop())

        try:
            for asset in self.assets:
                try:
                    market_info = await self.get_active_market(asset)
                    if market_info:
                        market_id, end_time, start_price = market_info
                        await self.switch_to_new_market(self.asset_states[asset], market_id, end_time, start_price)
                    else:
                        print(f"[{asset}] Waiting for market...")
                except Exception as e:
                    print(f"[{asset}] Failed to get initial market: {e}")

            while self.running:
                await asyncio.sleep(1)

        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            self.running = False
            monitor_task.cancel()
            claim_task.cancel()
            trade_task.cancel()
            await asyncio.gather(monitor_task, claim_task, trade_task, return_exceptions=True)
            await self.cancel_all_orders()
            await self.close()


# ============================================================
# CLI
# ============================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Turbine Kelly Bot — fractional Kelly criterion position sizing"
    )
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL_USDC,
                        help=f"Total capital for Kelly sizing (default: ${DEFAULT_BANKROLL_USDC})")
    parser.add_argument("--kelly-scalar", type=float, default=DEFAULT_KELLY_SCALAR,
                        help=f"Fractional Kelly scalar (default: {DEFAULT_KELLY_SCALAR}, i.e. 25%% of full Kelly)")
    parser.add_argument("--min-pct-edge", type=float, default=DEFAULT_MIN_PCT_EDGE,
                        help=f"Min %% edge to trade (default: {DEFAULT_MIN_PCT_EDGE})")
    parser.add_argument("--min-abs-edge", type=float, default=DEFAULT_MIN_ABS_EDGE,
                        help=f"Min absolute edge to trade (default: {DEFAULT_MIN_ABS_EDGE})")
    parser.add_argument("--max-position", type=float, default=DEFAULT_MAX_POSITION_USDC,
                        help=f"Max USDC per asset per market (default: ${DEFAULT_MAX_POSITION_USDC})")
    parser.add_argument("-a", "--assets", type=str,
                        default=os.environ.get("ASSETS", ",".join(SUPPORTED_ASSETS)),
                        help=f"Assets to trade (default: {','.join(SUPPORTED_ASSETS)})")
    parser.add_argument("--dry-run", action="store_true",
                        default=os.environ.get("DRY_RUN", "false").lower() == "true",
                        help="Simulate trading without placing real orders")
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets.split(",")]
    for asset in assets:
        if asset not in PYTH_FEED_IDS:
            print(f"Error: Unsupported asset '{asset}'. Supported: {', '.join(SUPPORTED_ASSETS)}")
            return

    private_key = os.environ.get("TURBINE_PRIVATE_KEY")
    if not private_key:
        print("Error: Set TURBINE_PRIVATE_KEY in your .env file")
        return

    api_key_id, api_private_key = get_or_create_api_credentials()

    client = TurbineClient(
        host=TURBINE_HOST,
        chain_id=CHAIN_ID,
        private_key=private_key,
        api_key_id=api_key_id,
        api_private_key=api_private_key,
    )

    print(f"\n{'='*60}")
    print(f"TURBINE KELLY BOT")
    print(f"{'='*60}")
    print(f"Wallet:      {client.address}")
    print(f"Chain:       {CHAIN_ID}")
    print(f"Assets:      {', '.join(assets)}")
    print(f"Bankroll:    ${args.bankroll:.2f} USDC")
    print(f"Kelly:       {args.kelly_scalar:.0%} of full Kelly")
    print(f"Min edge:    {args.min_pct_edge:.0%} pct / {args.min_abs_edge:.0%} abs")
    print(f"Max pos:     ${args.max_position:.2f} per asset")
    if args.dry_run:
        print(f"Mode:        DRY RUN (no real orders — sim balance: ${args.bankroll:.2f})")
    else:
        try:
            balance = client.get_usdc_balance() / 1_000_000
            print(f"USDC bal:    ${balance:.2f}")
            if balance < args.min_abs_edge * len(assets):
                print(f"  Warning: low balance. Fund wallet: {client.address}")
        except Exception as e:
            print(f"USDC bal:    unknown ({e})")
    print(f"{'='*60}\n")

    bot = KellyBot(
        client,
        assets=assets,
        bankroll_usdc=args.bankroll,
        kelly_scalar=args.kelly_scalar,
        min_pct_edge=args.min_pct_edge,
        min_abs_edge=args.min_abs_edge,
        max_position_usdc=args.max_position,
        dry_run=args.dry_run,
    )

    try:
        await bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        bot.running = False
        await bot.cancel_all_orders()
        client.close()
        print("Bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
