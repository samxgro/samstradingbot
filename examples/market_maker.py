"""
Turbine Market Maker Bot

Smart probability-based market maker for BTC, ETH, and SOL 15-minute prediction
markets. Uses a statistical model (normal CDF) to compute YES/NO probabilities
from real-time Pyth Network prices, with advanced features:

Algorithm:
  - P(YES) = Φ(deviation / (volatility × √timeRemaining))
    where Φ is the normal CDF. This correctly handles time decay:
    +0.45% with 1 min left → ~99% YES, +0.45% with 7 min left → ~75% YES.
  - Momentum tracking shifts probability in the direction of price movement
  - Inventory skew adjusts quotes to reduce net exposure
  - Spread widens on high volatility and strong momentum
  - One-sided quoting when probability deviates significantly from 50%
  - Circuit breaker trips when adverse selection is detected
  - Orders pulled in last 30 seconds, spread widened in last 90 seconds
  - Filled orders replaced at CURRENT fair value, not the old fill price
  - Graceful rebalance: new orders placed BEFORE old ones cancelled

Features:
  - Multi-asset: trades BTC, ETH, and SOL simultaneously (configurable via --assets)
  - Statistical probability model from Pyth prices (normal CDF, not linear)
  - Price tracker with velocity, volatility, and momentum signals
  - Inventory tracking with adverse selection detection and circuit breaker
  - One-sided quoting: kills losing side when market is trending
  - Multi-level quoting with geometric size distribution
  - Allocation skew: more capital on the likely-winning side
  - Auto-approves USDC gaslessly when entering a new market
  - Automatic market transition when 15-minute markets rotate
  - Automatic claiming of winnings via Multicall3 on-chain discovery (every 5 min)

Usage:
    TURBINE_PRIVATE_KEY=0x... python examples/market_maker.py

    # Custom parameters
    TURBINE_PRIVATE_KEY=0x... python examples/market_maker.py \\
        --allocation 50 \\
        --spread 0.012 \\
        --levels 6 \\
        --base-vol 0.03

    # Trade only BTC and ETH
    TURBINE_PRIVATE_KEY=0x... python examples/market_maker.py \\
        --assets BTC,ETH

    # Per-asset volatility overrides
    TURBINE_PRIVATE_KEY=0x... python examples/market_maker.py \\
        --asset-vol BTC=0.025 ETH=0.035 SOL=0.05
"""

import argparse
import asyncio
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import stdev

from dotenv import load_dotenv
import httpx

from turbine_client import TurbineClient, TurbineWSClient, Outcome, Side, QuickMarket
from turbine_client.exceptions import TurbineApiError, WebSocketError

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
CLAIM_ONLY_MODE = os.environ.get("CLAIM_ONLY_MODE", "false").lower() == "true"
CHAIN_ID = int(os.environ.get("CHAIN_ID", "84532"))
TURBINE_HOST = os.environ.get("TURBINE_HOST", "http://localhost:8080")

# Default trading parameters
DEFAULT_ALLOCATION_USDC = 60.0   # $60 total allocation per asset
DEFAULT_SPREAD = 0.012           # 1.2% spread around target probability
DEFAULT_NUM_LEVELS = 6           # Orders per side
DEFAULT_GEOMETRIC_LAMBDA = 1.5   # Geometric distribution parameter
DEFAULT_BASE_PROBABILITY = 0.50  # Starting YES probability
DEFAULT_MAX_PROBABILITY = 0.80   # Cap for extreme moves

# Statistical model parameters
DEFAULT_BASE_VOLATILITY = 0.03   # 3% daily vol (typical BTC)
SECONDS_PER_DAY = 86400.0

# Rebalance thresholds
REBALANCE_THRESHOLD = 0.02       # 2% probability change triggers requote
MIN_REBALANCE_INTERVAL = 2       # Minimum seconds between rebalances
FAST_POLL_INTERVAL = 2           # Seconds between price fetches

# Momentum and volatility parameters
MOMENTUM_FACTOR = 0.5            # How much momentum shifts probability
MOMENTUM_SPREAD_FACTOR = 2.0     # Spread multiplier per unit momentum
VOLATILITY_SPREAD_FACTOR = 10.0  # Spread multiplier per unit volatility
VOLATILITY_ALERT_THRESHOLD = 0.005  # Vol level forcing rebalance

# Inventory management
INVENTORY_SKEW_FACTOR = 0.05     # Quote skew per unit net exposure
ALLOCATION_SKEW_FACTOR = 0.3     # Allocation skew based on probability

# Adverse selection / circuit breaker
ADVERSE_SELECTION_THRESHOLD = 0.80  # Fill ratio to trigger circuit breaker
CIRCUIT_BREAKER_COOLDOWN = 10       # Seconds to stay dark after trigger

# One-sided quoting threshold
ONE_SIDE_THRESHOLD = 0.15        # Skip losing side when YES > 0.65 or < 0.35

# End-of-market behavior
END_OF_MARKET_PULL_SECONDS = 30  # Pull all orders in last N seconds
END_OF_MARKET_WIDEN_SECONDS = 90 # Start widening spread in last N seconds

# Pyth Network Hermes API
PYTH_HERMES_URL = "https://hermes.pyth.network/v2/updates/price/latest"
PYTH_FEED_IDS = {
    "BTC": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
}
SUPPORTED_ASSETS = list(PYTH_FEED_IDS.keys())


# ============================================================
# UTILITY: Normal CDF
# ============================================================

def normal_cdf(x: float) -> float:
    """Cumulative distribution function of the standard normal distribution."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


# ============================================================
# PRICE TRACKER
# ============================================================

@dataclass
class PriceSignals:
    """Computed market microstructure signals."""
    current_price: float = 0.0
    velocity: float = 0.0       # price change per second
    volatility: float = 0.0     # stddev of returns over window
    momentum: float = 0.0       # EMA of velocity (trend direction)
    is_stale: bool = True
    observation_age: float = 0.0


class PriceTracker:
    """Rolling window of price observations with velocity, volatility, and momentum."""

    def __init__(self, window_size: int = 60, max_age: float = 120.0, ema_alpha: float = 0.3):
        self.window_size = window_size
        self.max_age = max_age  # seconds
        self.ema_alpha = ema_alpha
        self.observations: deque[tuple[float, float]] = deque(maxlen=window_size)  # (price, timestamp)
        self._last_ema: float = 0.0

    def add_observation(self, price: float) -> None:
        """Record a new price observation."""
        now = time.time()
        self.observations.append((price, now))
        # Prune old observations
        cutoff = now - self.max_age
        while self.observations and self.observations[0][1] < cutoff:
            self.observations.popleft()

    def get_signals(self) -> PriceSignals:
        """Compute velocity, volatility, and momentum from price history."""
        n = len(self.observations)
        if n == 0:
            return PriceSignals(is_stale=True)

        latest_price, latest_time = self.observations[-1]
        age = time.time() - latest_time
        signals = PriceSignals(
            current_price=latest_price,
            observation_age=age,
            is_stale=age > self.max_age,
        )

        if n < 2:
            return signals

        # Velocity: last 5 observations
        window = min(5, n)
        first_price, first_time = self.observations[-window]
        dt = latest_time - first_time
        if dt > 0:
            signals.velocity = (latest_price - first_price) / dt

        # Volatility: stddev of returns over all observations
        returns = []
        obs_list = list(self.observations)
        for i in range(1, len(obs_list)):
            prev_price = obs_list[i - 1][0]
            if prev_price > 0:
                ret = (obs_list[i][0] - prev_price) / prev_price
                returns.append(ret)
        if len(returns) > 1:
            signals.volatility = stdev(returns)

        # Momentum: EMA of velocity
        self._last_ema = self.ema_alpha * signals.velocity + (1.0 - self.ema_alpha) * self._last_ema
        signals.momentum = self._last_ema

        return signals

    def reset(self) -> None:
        """Clear all observations."""
        self.observations.clear()
        self._last_ema = 0.0


# ============================================================
# INVENTORY TRACKER
# ============================================================

@dataclass
class FillRecord:
    """A single fill event."""
    side: str       # "BUY" or "SELL"
    outcome: str    # "YES" or "NO"
    price: int
    size: int
    timestamp: float


class InventoryTracker:
    """Tracks net position and fill history for adverse selection detection."""

    def __init__(self, fill_max_age: float = 180.0):
        self.fill_max_age = fill_max_age  # seconds
        self.yes_position: int = 0  # net shares (positive = long)
        self.no_position: int = 0
        self.recent_fills: list[FillRecord] = []

    def record_fill(self, side: str, outcome: str, price: int, size: int) -> None:
        """Record a fill and update net position."""
        signed_size = size
        if side == "BUY":
            if outcome == "YES":
                self.yes_position += signed_size
            else:
                self.no_position += signed_size
        else:
            if outcome == "YES":
                self.yes_position -= signed_size
            else:
                self.no_position -= signed_size

        self.recent_fills.append(FillRecord(
            side=side, outcome=outcome, price=price, size=size, timestamp=time.time()
        ))
        self._prune_old_fills()

    def get_net_exposure(self) -> float:
        """Normalized skew from -1.0 to +1.0. Positive = long YES / short NO."""
        total = abs(self.yes_position) + abs(self.no_position)
        if total == 0:
            return 0.0
        return (self.yes_position - self.no_position) / total

    def is_adversely_selected(self, threshold: float = ADVERSE_SELECTION_THRESHOLD) -> bool:
        """True if one side is getting filled disproportionately in last 30 seconds."""
        cutoff = time.time() - 30.0
        buy_count = sum(1 for f in self.recent_fills if f.timestamp >= cutoff and f.side == "BUY")
        sell_count = sum(1 for f in self.recent_fills if f.timestamp >= cutoff and f.side == "SELL")
        total = buy_count + sell_count
        if total < 3:
            return False
        ratio = max(buy_count, sell_count) / total
        return ratio > threshold

    def reset(self) -> None:
        """Clear all state."""
        self.yes_position = 0
        self.no_position = 0
        self.recent_fills.clear()

    def _prune_old_fills(self) -> None:
        cutoff = time.time() - self.fill_max_age
        self.recent_fills = [f for f in self.recent_fills if f.timestamp >= cutoff]


# ============================================================
# CREDENTIAL MANAGEMENT
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
        content = f"# Turbine Bot Config\nTURBINE_PRIVATE_KEY={os.environ.get('TURBINE_PRIVATE_KEY', '')}\nTURBINE_API_KEY_ID={api_key_id}\nTURBINE_API_PRIVATE_KEY={api_private_key}\n"
        env_path.write_text(content)


# ============================================================
# ASSET STATE
# ============================================================

class AssetState:
    """Per-asset market making state."""

    def __init__(self, asset: str):
        self.asset = asset
        self.market_id: str | None = None
        self.settlement_address: str | None = None
        self.contract_address: str | None = None
        self.strike_price: int = 0  # In 1e6 units
        self.market_start_time: int = 0
        self.market_end_time: int = 0

        # Dynamic pricing state
        self.yes_target: float = DEFAULT_BASE_PROBABILITY
        self.no_target: float = 1.0 - DEFAULT_BASE_PROBABILITY
        self.current_spread: float = DEFAULT_SPREAD
        self.yes_target_at_rebalance: float = DEFAULT_BASE_PROBABILITY
        self.last_rebalance_time: int = 0

        # Order tracking: order_hash -> {side, outcome, price, size}
        self.active_orders: dict[str, dict] = {}

        # Smart components
        self.price_tracker: PriceTracker = PriceTracker(window_size=60, max_age=120.0, ema_alpha=0.3)
        self.inventory: InventoryTracker = InventoryTracker(fill_max_age=180.0)

        # Circuit breaker
        self.circuit_breaker_tripped: bool = False
        self.circuit_breaker_until: float = 0.0

        # End-of-market
        self.orders_pulled: bool = False

        # Track markets we've traded in for claiming winnings
        self.traded_markets: dict[str, str] = {}  # market_id -> contract_address


# ============================================================
# MARKET MAKER
# ============================================================

class MarketMaker:
    """Smart probability-based market maker for quick prediction markets.

    Uses a statistical model (normal CDF) to compute fair probability from
    live prices, with momentum tracking, inventory management, adverse
    selection detection, and one-sided quoting.
    """

    def __init__(
        self,
        client: TurbineClient,
        assets: list[str],
        allocation_usdc: float = DEFAULT_ALLOCATION_USDC,
        spread: float = DEFAULT_SPREAD,
        num_levels: int = DEFAULT_NUM_LEVELS,
        base_volatility: float = DEFAULT_BASE_VOLATILITY,
        asset_volatilities: dict[str, float] | None = None,
    ):
        self.client = client
        self.assets = assets
        self.allocation_usdc = allocation_usdc
        self.base_spread = spread
        self.num_levels = num_levels
        self.base_volatility = base_volatility
        self.asset_volatilities = asset_volatilities or {}
        self.running = True

        # Per-asset state
        self.asset_states: dict[str, AssetState] = {
            asset: AssetState(asset) for asset in assets
        }

        # Track approved settlement contracts
        self.approved_settlements: dict[str, int] = {}

        # Async HTTP client
        self._http_client: httpx.AsyncClient | None = None

    def get_base_volatility_for_asset(self, asset: str) -> float:
        """Get per-asset volatility, falling back to global default."""
        if asset in self.asset_volatilities:
            return self.asset_volatilities[asset]
        return self.base_volatility

    # ------------------------------------------------------------------
    # Pyth price fetching
    # ------------------------------------------------------------------

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        return self._http_client

    async def close(self) -> None:
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def get_current_prices(self) -> dict[str, float]:
        """Fetch current prices for all active assets from Pyth Network."""
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

    # ------------------------------------------------------------------
    # Statistical probability model (normal CDF)
    # ------------------------------------------------------------------

    def calculate_smart_prices(
        self, state: AssetState, current_price: float
    ) -> tuple[float, float, float]:
        """Compute YES target, NO target, and spread using statistical model.

        P(YES) = Φ(deviation / (volatility × √timeRemaining))

        This correctly handles:
        - BTC +0.45% with 1 min left → YES ≈ 99%
        - BTC +0.45% with 7 min left → YES ≈ 75%
        - BTC ±0% at any time → YES = 50%

        Returns:
            (yes_target, no_target, spread)
        """
        strike_usd = state.strike_price / 1e6
        if strike_usd <= 0 or current_price <= 0:
            return DEFAULT_BASE_PROBABILITY, 1.0 - DEFAULT_BASE_PROBABILITY, self.base_spread

        now = int(time.time())
        seconds_remaining = max(1, state.market_end_time - now)

        # === STATISTICAL MODEL ===
        return_deviation = (current_price - strike_usd) / strike_usd

        # Estimate volatility per sqrt-second from price tracker
        signals = state.price_tracker.get_signals()
        vol_per_sqrt_second = 0.0
        if signals.volatility > 0:
            poll_interval = float(FAST_POLL_INTERVAL)
            vol_per_sqrt_second = signals.volatility / math.sqrt(poll_interval)

        # Floor at daily vol / sqrt(86400)
        min_daily_vol = self.get_base_volatility_for_asset(state.asset)
        min_vol_per_sqrt_second = min_daily_vol / math.sqrt(SECONDS_PER_DAY)
        vol_per_sqrt_second = max(vol_per_sqrt_second, min_vol_per_sqrt_second)

        # Expected remaining volatility
        expected_vol_to_expiry = vol_per_sqrt_second * math.sqrt(seconds_remaining)

        # Z-score
        z_score = return_deviation / expected_vol_to_expiry if expected_vol_to_expiry > 0 else 0.0

        # P(YES) = normal CDF of z-score
        yes_target = normal_cdf(z_score)

        # === MOMENTUM ADJUSTMENT (lead the price) ===
        momentum_shift = signals.momentum * MOMENTUM_FACTOR / 1e6
        momentum_shift = max(-0.10, min(0.10, momentum_shift))
        yes_target += momentum_shift

        # === INVENTORY SKEW ===
        net_exposure = state.inventory.get_net_exposure()
        inventory_skew = net_exposure * INVENTORY_SKEW_FACTOR
        yes_target -= inventory_skew

        # Clamp
        yes_target = max(1.0 - DEFAULT_MAX_PROBABILITY, min(DEFAULT_MAX_PROBABILITY, yes_target))
        no_target = 1.0 - yes_target

        # === SPREAD CALCULATION ===
        spread = self.base_spread

        # Volatility-adjusted
        vol_multiplier = 1.0 + signals.volatility * VOLATILITY_SPREAD_FACTOR
        spread *= vol_multiplier

        # Momentum-adjusted
        mom_mag = abs(signals.momentum) * MOMENTUM_SPREAD_FACTOR / 1e6
        spread *= (1.0 + mom_mag)

        # End-of-market: widen in last 90 seconds
        if END_OF_MARKET_PULL_SECONDS < seconds_remaining < END_OF_MARKET_WIDEN_SECONDS:
            end_factor = 1.0 + 2.0 * (1.0 - seconds_remaining / END_OF_MARKET_WIDEN_SECONDS)
            spread *= end_factor

        # Clamp spread
        spread = max(0.01, min(0.20, spread))

        return yes_target, no_target, spread

    # ------------------------------------------------------------------
    # Multi-level geometric distribution
    # ------------------------------------------------------------------

    def calculate_geometric_weights(self, n: int, side: str) -> list[float]:
        """Geometric size distribution weights."""
        lam = DEFAULT_GEOMETRIC_LAMBDA
        weights = []
        for i in range(n):
            if side == "BUY":
                w = lam ** i
            else:
                w = lam ** (n - 1 - i)
            weights.append(w)
        total = sum(weights)
        if total <= 0:
            return [1.0 / n] * n
        return [w / total for w in weights]

    def generate_level_prices(self, min_price: float, max_price: float, n: int) -> list[int]:
        """Generate evenly spaced prices clamped to [10000, 990000]."""
        if n <= 1:
            mid = (min_price + max_price) / 2
            return [max(10000, min(990000, int(mid * 1_000_000)))]
        step = (max_price - min_price) / (n - 1)
        return [max(10000, min(990000, int((min_price + i * step) * 1_000_000))) for i in range(n)]

    def calculate_shares_from_usdc(self, usdc_amount: float, price: int) -> int:
        """Calculate shares from USDC amount at given price."""
        if price <= 0:
            return 0
        return int((usdc_amount * 1_000_000 * 1_000_000) / price)

    # ------------------------------------------------------------------
    # USDC approval
    # ------------------------------------------------------------------

    MAX_APPROVAL_THRESHOLD = (2**256 - 1) // 2

    def ensure_settlement_approved(self, settlement_address: str) -> None:
        """Ensure USDC is approved via gasless max permit."""
        if settlement_address in self.approved_settlements:
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

            # Wait for confirmation by polling allowance via API
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

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            open_orders = self.client.get_orders(trader=self.client.address, status="open")
        except Exception as e:
            print(f"Failed to fetch open orders: {e}")
            return

        if not open_orders:
            return

        print(f"Cancelling {len(open_orders)} open orders...")
        cancelled = 0
        for order in open_orders:
            try:
                self.client.cancel_order(order.order_hash, market_id=order.market_id, side=Side(order.side))
                cancelled += 1
            except TurbineApiError as e:
                if "404" not in str(e):
                    print(f"  Failed to cancel {order.order_hash[:10]}...: {e}")
        print(f"  Cancelled {cancelled}/{len(open_orders)} orders")
        for state in self.asset_states.values():
            state.active_orders.clear()

    async def cancel_asset_orders(self, state: AssetState) -> None:
        """Cancel all open orders for a specific asset's market."""
        if not state.market_id:
            return
        try:
            open_orders = self.client.get_orders(
                trader=self.client.address, market_id=state.market_id, status="open"
            )
        except Exception:
            return

        for order in open_orders:
            try:
                self.client.cancel_order(order.order_hash, market_id=order.market_id, side=Side(order.side))
            except TurbineApiError:
                pass
        state.active_orders.clear()

    async def place_smart_quotes(self, state: AssetState) -> dict[str, dict]:
        """Place multi-level quotes with one-sided quoting and allocation skew.

        When the market is trending (YES target far from 50%), this will:
        - Skip the losing side entirely (one-sided quoting)
        - Allocate more capital to the likely-winning outcome
        - Skew within each outcome toward sells when probability is high

        Returns:
            Dict of order_hash -> {side, outcome, price, size}
        """
        new_orders: dict[str, dict] = {}

        n = self.num_levels
        spread = state.current_spread
        half_spread = spread / 2
        expiration = int(time.time()) + 300  # 5 min expiration

        # --- ONE-SIDED QUOTING ---
        # Determine which sides to quote based on target deviation from 0.50
        yes_deviation = state.yes_target - 0.5
        quote_yes_buy = True
        quote_yes_sell = True
        quote_no_buy = True
        quote_no_sell = True

        if yes_deviation > ONE_SIDE_THRESHOLD:
            # YES is likely — don't provide NO liquidity
            quote_no_buy = False
            quote_no_sell = False
            print(f"  [{state.asset}] ONE-SIDED: YES={state.yes_target:.2f} — skipping NO orders (trending UP)")
        elif yes_deviation < -ONE_SIDE_THRESHOLD:
            # NO is likely — don't provide YES liquidity
            quote_yes_buy = False
            quote_yes_sell = False
            print(f"  [{state.asset}] ONE-SIDED: YES={state.yes_target:.2f} — skipping YES orders (trending DOWN)")

        # Also check momentum as a leading indicator
        signals = state.price_tracker.get_signals()
        momentum_strength = signals.momentum / 1e6
        if momentum_strength > 5.0 and yes_deviation > 0:
            quote_no_buy = False
            quote_no_sell = False
        elif momentum_strength < -5.0 and yes_deviation < 0:
            quote_yes_buy = False
            quote_yes_sell = False

        # Count active sides
        active_sides = sum([quote_yes_buy, quote_yes_sell, quote_no_buy, quote_no_sell])
        if active_sides == 0:
            print(f"  [{state.asset}] SKIP: No sides to quote (extreme momentum)")
            return new_orders

        # --- ALLOCATION SKEW ---
        total_allocation = self.allocation_usdc
        yes_fraction = 0.5 + (state.yes_target - 0.5) * ALLOCATION_SKEW_FACTOR
        yes_fraction = max(0.2, min(0.8, yes_fraction))

        yes_allocation = total_allocation * yes_fraction
        no_allocation = total_allocation - yes_allocation

        # Within each outcome, skew toward SELL when probability is high
        yes_sell_fraction = 0.5 + (state.yes_target - 0.5) * 0.3
        yes_sell_fraction = max(0.3, min(0.7, yes_sell_fraction))

        no_sell_fraction = 0.5 + (state.no_target - 0.5) * 0.3
        no_sell_fraction = max(0.3, min(0.7, no_sell_fraction))

        # Calculate per-bucket allocations
        if quote_yes_buy and quote_yes_sell:
            yes_buy_alloc = yes_allocation * (1 - yes_sell_fraction)
            yes_sell_alloc = yes_allocation - yes_buy_alloc
        elif quote_yes_buy:
            yes_buy_alloc = yes_allocation
            yes_sell_alloc = 0.0
        elif quote_yes_sell:
            yes_buy_alloc = 0.0
            yes_sell_alloc = yes_allocation
        else:
            yes_buy_alloc = 0.0
            yes_sell_alloc = 0.0

        if quote_no_buy and quote_no_sell:
            no_buy_alloc = no_allocation * (1 - no_sell_fraction)
            no_sell_alloc = no_allocation - no_buy_alloc
        elif quote_no_buy:
            no_buy_alloc = no_allocation
            no_sell_alloc = 0.0
        elif quote_no_sell:
            no_buy_alloc = 0.0
            no_sell_alloc = no_allocation
        else:
            no_buy_alloc = 0.0
            no_sell_alloc = 0.0

        buy_weights = self.calculate_geometric_weights(n, "BUY")
        sell_weights = self.calculate_geometric_weights(n, "SELL")

        print(
            f"[{state.asset}] Quoting: YES {state.yes_target:.1%} / NO {state.no_target:.1%} | "
            f"Spread {state.current_spread:.1%} | Sides {active_sides}/4 | "
            f"Alloc YES[B=${yes_buy_alloc:.1f} S=${yes_sell_alloc:.1f}] NO[B=${no_buy_alloc:.1f} S=${no_sell_alloc:.1f}]"
        )

        # --- PLACE ORDERS ---
        for outcome, target, quote_buy, quote_sell, buy_alloc, sell_alloc in [
            (Outcome.YES, state.yes_target, quote_yes_buy, quote_yes_sell, yes_buy_alloc, yes_sell_alloc),
            (Outcome.NO, state.no_target, quote_no_buy, quote_no_sell, no_buy_alloc, no_sell_alloc),
        ]:
            outcome_name = "YES" if outcome == Outcome.YES else "NO"

            # Bid range
            bid_max = target - half_spread / 2
            bid_min = max(0.01, bid_max - spread)
            # Ask range
            ask_min = target + half_spread / 2
            ask_max = min(0.99, ask_min + spread)

            # Place bids
            if quote_buy and buy_alloc >= 1.0:
                bid_prices = self.generate_level_prices(bid_min, bid_max, n)
                for i in range(n):
                    usdc_for_level = buy_alloc * buy_weights[i]
                    if usdc_for_level < 1.0:
                        continue
                    price = bid_prices[i]
                    shares = self.calculate_shares_from_usdc(usdc_for_level, price)
                    if shares <= 0:
                        continue
                    try:
                        order = self.client.create_limit_buy(
                            market_id=state.market_id,
                            outcome=outcome,
                            price=price,
                            size=shares,
                            expiration=expiration,
                            settlement_address=state.settlement_address,
                        )
                        self.client.post_order(order)
                        new_orders[order.order_hash] = {
                            "side": "BUY", "outcome": outcome_name,
                            "price": price, "size": shares
                        }
                    except TurbineApiError as e:
                        print(f"  [{state.asset}] Failed {outcome_name} bid L{i}: {e}")

            # Place asks
            if quote_sell and sell_alloc >= 1.0:
                ask_prices = self.generate_level_prices(ask_min, ask_max, n)
                for i in range(n):
                    usdc_for_level = sell_alloc * sell_weights[i]
                    if usdc_for_level < 1.0:
                        continue
                    price = ask_prices[i]
                    shares = self.calculate_shares_from_usdc(usdc_for_level, price)
                    if shares <= 0:
                        continue
                    try:
                        order = self.client.create_limit_sell(
                            market_id=state.market_id,
                            outcome=outcome,
                            price=price,
                            size=shares,
                            expiration=expiration,
                            settlement_address=state.settlement_address,
                        )
                        self.client.post_order(order)
                        new_orders[order.order_hash] = {
                            "side": "SELL", "outcome": outcome_name,
                            "price": price, "size": shares
                        }
                    except TurbineApiError as e:
                        print(f"  [{state.asset}] Failed {outcome_name} ask L{i}: {e}")

        if new_orders:
            buy_count = sum(1 for o in new_orders.values() if o["side"] == "BUY")
            sell_count = sum(1 for o in new_orders.values() if o["side"] == "SELL")
            print(f"  [{state.asset}] Placed {buy_count} BUY + {sell_count} SELL ({len(new_orders)} total)")

        return new_orders

    async def graceful_rebalance(self, state: AssetState) -> None:
        """Place new orders FIRST, then cancel old ones (no gap in liquidity)."""
        old_orders = dict(state.active_orders)
        state.active_orders.clear()

        # Place new orders at current fair value
        new_orders = await self.place_smart_quotes(state)
        state.active_orders.update(new_orders)

        # Brief pause for in-flight trades
        await asyncio.sleep(0.2)

        # Cancel old orders
        for order_hash, info in old_orders.items():
            try:
                side = Side.BUY if info["side"] == "BUY" else Side.SELL
                self.client.cancel_order(order_hash, market_id=state.market_id, side=side)
            except TurbineApiError:
                pass  # Order may already be filled or expired

    async def check_and_refresh_fills(self, state: AssetState) -> None:
        """Detect filled orders, record in inventory, and replace at CURRENT fair value."""
        if not state.market_id or not state.active_orders:
            return

        try:
            user_orders = self.client.get_orders(
                trader=self.client.address, market_id=state.market_id, status="open"
            )
        except Exception:
            return

        # Build set of active order hashes from API
        api_active = {o.order_hash for o in user_orders}

        # Find filled orders
        filled = []
        for order_hash, info in list(state.active_orders.items()):
            if order_hash not in api_active:
                filled.append((order_hash, info))
                del state.active_orders[order_hash]

                # Record fill in inventory
                state.inventory.record_fill(
                    side=info["side"],
                    outcome=info["outcome"],
                    price=info["price"],
                    size=info["size"],
                )
                print(f"  [{state.asset}] FILL: {info['side']} {info['outcome']} @ "
                      f"{info['price'] / 1e6:.4f} (size: {info['size'] / 1e6:.2f})")

        # Replace filled orders at CURRENT fair value
        if filled:
            print(f"  [{state.asset}] Replacing {len(filled)} filled orders at current fair value")
            spread = state.current_spread
            half_spread = spread / 2
            expiration = int(time.time()) + 300

            for _, info in filled:
                # Determine target for this outcome
                if info["outcome"] == "YES":
                    target = state.yes_target
                    outcome = Outcome.YES
                else:
                    target = state.no_target
                    outcome = Outcome.NO

                if info["side"] == "BUY":
                    new_price_float = max(0.01, min(0.99, target - half_spread))
                    new_price = int(new_price_float * 1_000_000)
                    try:
                        order = self.client.create_limit_buy(
                            market_id=state.market_id,
                            outcome=outcome,
                            price=new_price,
                            size=info["size"],
                            expiration=expiration,
                            settlement_address=state.settlement_address,
                        )
                        self.client.post_order(order)
                        state.active_orders[order.order_hash] = {
                            "side": "BUY", "outcome": info["outcome"],
                            "price": new_price, "size": info["size"]
                        }
                    except TurbineApiError as e:
                        print(f"  [{state.asset}] Failed to replace BUY: {e}")
                else:
                    new_price_float = max(0.01, min(0.99, target + half_spread))
                    new_price = int(new_price_float * 1_000_000)
                    try:
                        order = self.client.create_limit_sell(
                            market_id=state.market_id,
                            outcome=outcome,
                            price=new_price,
                            size=info["size"],
                            expiration=expiration,
                            settlement_address=state.settlement_address,
                        )
                        self.client.post_order(order)
                        state.active_orders[order.order_hash] = {
                            "side": "SELL", "outcome": info["outcome"],
                            "price": new_price, "size": info["size"]
                        }
                    except TurbineApiError as e:
                        print(f"  [{state.asset}] Failed to replace SELL: {e}")

    # ------------------------------------------------------------------
    # Market lifecycle
    # ------------------------------------------------------------------

    async def get_active_market(self, asset: str) -> tuple[str, int, int, int] | None:
        """Get the currently active quick market for an asset."""
        response = self.client._http.get(f"/api/v1/quick-markets/{asset}")
        quick_market_data = response.get("quickMarket")
        if not quick_market_data:
            return None
        quick_market = QuickMarket.from_dict(quick_market_data)
        return (
            quick_market.market_id,
            quick_market.start_time,
            quick_market.end_time,
            quick_market.start_price,
        )

    async def switch_to_new_market(
        self, state: AssetState, new_market_id: str,
        start_time: int = 0, end_time: int = 0, start_price: int = 0
    ) -> None:
        """Switch an asset to a new market."""
        old_market_id = state.market_id

        if old_market_id and state.contract_address:
            state.traded_markets[old_market_id] = state.contract_address

        if old_market_id:
            print(f"\n{'='*50}")
            print(f"[{state.asset}] MARKET TRANSITION")
            print(f"Old: {old_market_id[:8]}... | New: {new_market_id[:8]}...")
            print(f"{'='*50}\n")
            state.active_orders.clear()

        # Update market state
        state.market_id = new_market_id
        state.strike_price = start_price
        state.market_start_time = start_time
        state.market_end_time = end_time
        state.active_orders = {}

        # Reset smart components
        state.yes_target = DEFAULT_BASE_PROBABILITY
        state.no_target = 1.0 - DEFAULT_BASE_PROBABILITY
        state.current_spread = self.base_spread
        state.yes_target_at_rebalance = DEFAULT_BASE_PROBABILITY
        state.last_rebalance_time = 0
        state.price_tracker.reset()
        state.inventory.reset()
        state.circuit_breaker_tripped = False
        state.circuit_breaker_until = 0.0
        state.orders_pulled = False

        # Fetch settlement and contract addresses
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
            print(f"[{state.asset}] Warning: Could not fetch market addresses: {e}")

        if state.settlement_address:
            self.ensure_settlement_approved(state.settlement_address)

        strike_usd = start_price / 1e6 if start_price else 0
        print(f"[{state.asset}] Trading: {new_market_id[:8]}... | Strike: ${strike_usd:,.2f} | ${self.allocation_usdc:.2f} allocation")

    async def monitor_market_transitions(self) -> None:
        """Background task polling for new markets."""
        while self.running:
            try:
                for asset in self.assets:
                    state = self.asset_states[asset]
                    try:
                        market_info = await self.get_active_market(asset)
                    except Exception:
                        continue

                    if not market_info:
                        continue

                    new_market_id, start_time, end_time, start_price = market_info
                    if new_market_id != state.market_id:
                        await self.switch_to_new_market(state, new_market_id, start_time, end_time, start_price)
                    else:
                        state.market_end_time = end_time

            except Exception as e:
                print(f"Market monitor error: {e}")

            await asyncio.sleep(5)

    async def claim_resolved_markets(self) -> None:
        """Background task to periodically discover and claim all winnings.

        Uses Multicall3-based on-chain discovery (claim_all_winnings) to find
        and claim ALL resolved positions — not just markets traded in this session.
        This ensures the MM recycles liquidity from any prior session or wallet activity.
        """
        # Wait for initial market setup before first claim attempt
        await asyncio.sleep(60)

        while self.running:
            try:
                print("[CLAIM] Scanning for claimable positions (Multicall3 discovery)...")
                result = self.client.claim_all_winnings()
                tx_hash = result.get("txHash", result.get("tx_hash", "unknown"))
                print(f"[CLAIM] 💰 Claimed winnings! TX: {tx_hash}")

                # Clear tracked markets that were just claimed
                for state in self.asset_states.values():
                    state.traded_markets.clear()

            except ValueError as e:
                # "No claimable positions found" — normal, nothing to claim
                if "no claimable" in str(e).lower():
                    print(f"[CLAIM] No claimable positions found — all clear")
                else:
                    print(f"[CLAIM] Error: {e}")
            except Exception as e:
                print(f"[CLAIM] Discovery/claim error: {e}")

            # Run every 5 minutes — frequent enough to recycle liquidity promptly
            # after 15-min markets resolve, without hammering the RPC
            await asyncio.sleep(300)

    # ------------------------------------------------------------------
    # Main trading loop (fast poll)
    # ------------------------------------------------------------------

    async def smart_trading_loop(self) -> None:
        """Fast-polling trading loop with smart pricing.

        Runs every FAST_POLL_INTERVAL seconds:
        1. Fetch prices, update price tracker
        2. Check circuit breaker
        3. Check adverse selection
        4. Compute new targets via statistical model
        5. Rebalance if threshold exceeded
        6. Check for fills and replace at current fair value
        """
        if CLAIM_ONLY_MODE:
            print("CLAIM ONLY MODE — Trading disabled")
            while self.running:
                await asyncio.sleep(60)
            return

        while self.running:
            active_assets = [a for a in self.assets if self.asset_states[a].market_id]
            if not active_assets:
                await asyncio.sleep(1)
                continue

            # Initial quote placement
            prices = await self.get_current_prices()
            for asset in active_assets:
                state = self.asset_states[asset]
                current_price = prices.get(asset, 0.0)
                if current_price > 0:
                    state.price_tracker.add_observation(current_price)
                    yes, no, spread = self.calculate_smart_prices(state, current_price)
                    state.yes_target = yes
                    state.no_target = no
                    state.current_spread = spread
                    state.yes_target_at_rebalance = yes
                    state.last_rebalance_time = int(time.time())

                new_orders = await self.place_smart_quotes(state)
                state.active_orders.update(new_orders)

            # Fast polling loop
            while self.running:
                await asyncio.sleep(FAST_POLL_INTERVAL)

                prices = await self.get_current_prices()
                now = int(time.time())

                for asset in list(active_assets):
                    state = self.asset_states[asset]
                    if not state.market_id:
                        continue

                    seconds_remaining = state.market_end_time - now

                    # === END-OF-MARKET: Pull all orders ===
                    if 0 < seconds_remaining <= END_OF_MARKET_PULL_SECONDS:
                        if not state.orders_pulled:
                            print(f"[{state.asset}] PULLING all orders ({seconds_remaining}s remaining — too risky)")
                            await self.cancel_asset_orders(state)
                            state.orders_pulled = True
                        continue

                    # Market expired
                    if seconds_remaining <= 0:
                        continue

                    # === FETCH PRICE ===
                    current_price = prices.get(asset, 0.0)
                    if current_price <= 0:
                        continue
                    state.price_tracker.add_observation(current_price)
                    signals = state.price_tracker.get_signals()

                    # === CIRCUIT BREAKER ===
                    if state.circuit_breaker_tripped:
                        if time.time() < state.circuit_breaker_until:
                            continue
                        state.circuit_breaker_tripped = False
                        print(f"[{state.asset}] Circuit breaker RESET — resuming quoting")

                    # === ADVERSE SELECTION CHECK ===
                    if state.inventory.is_adversely_selected():
                        print(f"[{state.asset}] ADVERSE SELECTION detected — circuit breaker for {CIRCUIT_BREAKER_COOLDOWN}s")
                        await self.cancel_asset_orders(state)
                        state.circuit_breaker_tripped = True
                        state.circuit_breaker_until = time.time() + CIRCUIT_BREAKER_COOLDOWN
                        continue

                    # === COMPUTE NEW TARGETS ===
                    new_yes, new_no, new_spread = self.calculate_smart_prices(state, current_price)

                    # === REBALANCE DECISION ===
                    target_diff = abs(new_yes - state.yes_target_at_rebalance)
                    time_since_rebalance = now - state.last_rebalance_time

                    should_rebalance = (
                        target_diff > REBALANCE_THRESHOLD
                        and time_since_rebalance >= MIN_REBALANCE_INTERVAL
                    )

                    # Force rebalance on volatility spike
                    if signals.volatility > VOLATILITY_ALERT_THRESHOLD and time_since_rebalance >= 5:
                        should_rebalance = True

                    # Update state
                    state.yes_target = new_yes
                    state.no_target = new_no
                    state.current_spread = new_spread

                    if should_rebalance:
                        strike_usd = state.strike_price / 1e6
                        dev_pct = ((current_price - strike_usd) / strike_usd) * 100 if strike_usd > 0 else 0
                        print(
                            f"[{state.asset}] REBALANCE: ${current_price:,.2f} ({dev_pct:+.2f}%) | "
                            f"YES {state.yes_target_at_rebalance:.1%} → {new_yes:.1%} | "
                            f"Spread {new_spread:.1%} | Inv {state.inventory.get_net_exposure():.2f} | "
                            f"{seconds_remaining}s left"
                        )
                        state.last_rebalance_time = now
                        state.yes_target_at_rebalance = new_yes
                        await self.graceful_rebalance(state)

                    # Check for fills
                    await self.check_and_refresh_fills(state)

                # Check if active markets changed (new market started)
                new_active = [a for a in self.assets if self.asset_states[a].market_id]
                if set(new_active) != set(active_assets):
                    break  # Re-enter outer loop to initialize new markets

    async def run(self) -> None:
        """Main entry point."""
        monitor_task = asyncio.create_task(self.monitor_market_transitions())
        claim_task = asyncio.create_task(self.claim_resolved_markets())
        trading_task = asyncio.create_task(self.smart_trading_loop())

        try:
            for asset in self.assets:
                try:
                    market_info = await self.get_active_market(asset)
                    if market_info:
                        market_id, start_time, end_time, start_price = market_info
                        await self.switch_to_new_market(
                            self.asset_states[asset], market_id, start_time, end_time, start_price
                        )
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
            trading_task.cancel()
            await asyncio.gather(monitor_task, claim_task, trading_task, return_exceptions=True)
            await self.close()


# ============================================================
# CLI
# ============================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Smart market maker for BTC, ETH, and SOL 15-minute prediction markets"
    )
    parser.add_argument(
        "-a", "--allocation", type=float, default=DEFAULT_ALLOCATION_USDC,
        help=f"Total USDC allocation per asset (default: ${DEFAULT_ALLOCATION_USDC})"
    )
    parser.add_argument(
        "--spread", type=float, default=DEFAULT_SPREAD,
        help=f"Base spread (default: {DEFAULT_SPREAD} = {DEFAULT_SPREAD * 100:.1f}%%)"
    )
    parser.add_argument(
        "--levels", type=int, default=DEFAULT_NUM_LEVELS,
        help=f"Price levels per side (default: {DEFAULT_NUM_LEVELS})"
    )
    parser.add_argument(
        "--base-vol", type=float, default=DEFAULT_BASE_VOLATILITY,
        help=f"Base daily volatility for probability model (default: {DEFAULT_BASE_VOLATILITY} = {DEFAULT_BASE_VOLATILITY * 100:.0f}%%)"
    )
    parser.add_argument(
        "--asset-vol", nargs="*", metavar="ASSET=VOL",
        help="Per-asset volatility overrides (e.g., BTC=0.025 ETH=0.035 SOL=0.05)"
    )
    parser.add_argument(
        "--assets", type=str, default=",".join(SUPPORTED_ASSETS),
        help=f"Comma-separated assets to trade (default: {','.join(SUPPORTED_ASSETS)})"
    )
    args = parser.parse_args()

    # Parse assets
    assets = [a.strip().upper() for a in args.assets.split(",")]
    for asset in assets:
        if asset not in PYTH_FEED_IDS:
            print(f"Error: Unsupported asset '{asset}'. Supported: {', '.join(SUPPORTED_ASSETS)}")
            return

    # Parse per-asset volatilities
    asset_vols: dict[str, float] = {}
    if args.asset_vol:
        for entry in args.asset_vol:
            if "=" not in entry:
                print(f"Error: Invalid --asset-vol format '{entry}'. Use ASSET=VOL (e.g., BTC=0.025)")
                return
            asset_name, vol_str = entry.split("=", 1)
            asset_vols[asset_name.upper()] = float(vol_str)

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
    print(f"TURBINE SMART MARKET MAKER")
    print(f"{'='*60}")
    print(f"Wallet:      {client.address}")
    print(f"Chain:       {CHAIN_ID}")
    print(f"Assets:      {', '.join(assets)}")
    print(f"Allocation:  ${args.allocation:.2f} per asset")
    print(f"Spread:      {args.spread * 100:.1f}% base (dynamic: widens on vol/momentum)")
    print(f"Levels:      {args.levels} per side")
    print(f"Model:       Statistical (normal CDF)")
    print(f"Base vol:    {args.base_vol * 100:.1f}% daily")
    if asset_vols:
        for a, v in asset_vols.items():
            print(f"  {a} vol:   {v * 100:.1f}% daily")
    print(f"One-sided:   Skip losing side when YES deviates >{ONE_SIDE_THRESHOLD * 100:.0f}% from 50%")
    print(f"Circuit brk: {CIRCUIT_BREAKER_COOLDOWN}s cooldown on adverse selection")
    print(f"End-of-mkt:  Pull at {END_OF_MARKET_PULL_SECONDS}s, widen at {END_OF_MARKET_WIDEN_SECONDS}s")
    print(f"Poll:        {FAST_POLL_INTERVAL}s (rebalance min {MIN_REBALANCE_INTERVAL}s)")
    print(f"USDC:        Gasless approval (one-time max permit)")
    try:
        usdc_balance = client.get_usdc_balance()
        balance_display = usdc_balance / 1_000_000
        print(f"Balance:     ${balance_display:.2f} USDC")
        min_needed = args.allocation * len(assets)
        if balance_display < min_needed:
            print(f"⚠️  Warning: Balance (${balance_display:.2f}) may be low for {len(assets)} assets x ${args.allocation:.2f}")
            print(f"   Fund your wallet: {client.address}")
    except Exception as e:
        print(f"Balance:     unknown ({e})")
    print(f"{'='*60}\n")

    # Ensure USDC and CTF token approvals are in place before trading.
    # USDC: checked against threshold, approved via gasless permit if low.
    # CTF: setApprovalForAll is idempotent, so we call it unconditionally.
    USDC_APPROVAL_THRESHOLD = 1_000_000_000  # 1000 USDC (6 decimals)
    try:
        allowance = client.get_usdc_allowance()
        if allowance < USDC_APPROVAL_THRESHOLD:
            print("USDC allowance low — approving via gasless permit...")
            client.approve_usdc_for_settlement()
            print("USDC approved ✓")
        else:
            print("USDC allowance sufficient ✓")
        client.approve_ctf_for_settlement()
        print("CTF (ERC1155) approved ✓")
    except Exception as e:
        print(f"Warning: Could not check/approve token allowances: {e}")

    bot = MarketMaker(
        client,
        assets=assets,
        allocation_usdc=args.allocation,
        spread=args.spread,
        num_levels=args.levels,
        base_volatility=args.base_vol,
        asset_volatilities=asset_vols,
    )

    try:
        await bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        bot.running = False
        await bot.cancel_all_orders()
        await bot.close()
        client.close()
        print("Bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
