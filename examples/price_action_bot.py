"""
Turbine Price Action Bot

Trades BTC, ETH, and SOL 15-minute prediction markets using real-time price
data from Pyth Network. USDC is approved gaslessly via a one-time max permit
per settlement contract — no native gas required, no per-order permit overhead.

Algorithm: Fetches real-time prices from Pyth Network (same oracle Turbine uses)
           and compares them to each market's strike price to make trading decisions.
           - If price is above strike → buy YES (bet it stays above)
           - If price is below strike → buy NO (bet it stays below)
           - Confidence scales with distance from strike price

Features:
- Multi-asset: trades BTC, ETH, and SOL simultaneously (configurable via --assets)
- Order size and max position configured in USDC terms (per asset)
- Auto-approves USDC gaslessly when entering a new market
- Automatic market transition when 15-minute markets rotate
- Automatic claiming of winnings from resolved markets

Usage:
    TURBINE_PRIVATE_KEY=0x... python examples/price_action_bot.py

    # Custom USDC amounts
    TURBINE_PRIVATE_KEY=0x... python examples/price_action_bot.py \\
        --order-size 5 \\
        --max-position 50

    # Trade only BTC and ETH
    TURBINE_PRIVATE_KEY=0x... python examples/price_action_bot.py \\
        --assets BTC,ETH
"""

import argparse
import asyncio
import math
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import httpx

from turbine_client import TurbineClient, Outcome, Side, QuickMarket
from turbine_client.exceptions import TurbineApiError

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION - Adjust these parameters for your strategy
# ============================================================
# Claim-only mode: Set CLAIM_ONLY_MODE=true in .env to disable trading
CLAIM_ONLY_MODE = os.environ.get("CLAIM_ONLY_MODE", "false").lower() == "true"
# Chain ID: Set CHAIN_ID in .env (default: 84532 for Base Sepolia)
CHAIN_ID = int(os.environ.get("CHAIN_ID", "84532"))
# API Host: Set TURBINE_HOST in .env (default: localhost for testing)
TURBINE_HOST = os.environ.get("TURBINE_HOST", "http://localhost:8080")

# Default trading parameters (in USDC terms)
DEFAULT_ORDER_SIZE_USDC = 1.0  # $1 USDC per order
DEFAULT_MAX_POSITION_USDC = 10.0  # $10 max position per asset per market
PRICE_POLL_SECONDS = 10  # How often to check prices

# Price Action parameters
PRICE_THRESHOLD_BPS = 10  # 0.1% threshold before taking action
MIN_CONFIDENCE = 0.6  # Minimum confidence to place a trade
MAX_CONFIDENCE = 0.9  # Cap confidence at 90%

# Pyth Network Hermes API - same price source Turbine uses
PYTH_HERMES_URL = "https://hermes.pyth.network/v2/updates/price/latest"

# Pyth feed IDs per asset
PYTH_FEED_IDS = {
    "BTC": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
    "ETH": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
    "SOL": "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
}

# Supported assets
SUPPORTED_ASSETS = list(PYTH_FEED_IDS.keys())


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

    # Auto-save to .env
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
        # Update or append each credential
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


class AssetState:
    """Per-asset trading state."""

    def __init__(self, asset: str):
        self.asset = asset
        self.market_id: str | None = None
        self.settlement_address: str | None = None
        self.contract_address: str | None = None
        self.strike_price: int = 0  # Price when market created (6 decimals)
        self.position_usdc: dict[str, float] = {}  # market_id -> usdc spent
        self.active_orders: dict[str, str] = {}  # order_hash -> side
        self.processed_trade_ids: set[int] = set()
        self.pending_order_txs: set[str] = set()
        self.traded_markets: dict[str, str] = {}  # market_id -> contract_address
        self.market_expiring = False


class PriceActionBot:
    """Price action trader for BTC, ETH, and SOL 15-minute prediction markets.

    Tracks positions in USDC terms per asset. USDC is approved gaslessly for new
    markets via one-time max EIP-2612 permit submitted through the relayer.
    """

    def __init__(
        self,
        client: TurbineClient,
        assets: list[str],
        order_size_usdc: float = DEFAULT_ORDER_SIZE_USDC,
        max_position_usdc: float = DEFAULT_MAX_POSITION_USDC,
    ):
        self.client = client
        self.assets = assets
        self.order_size_usdc = order_size_usdc
        self.max_position_usdc = max_position_usdc
        self.running = True

        # Per-asset state
        self.asset_states: dict[str, AssetState] = {
            asset: AssetState(asset) for asset in assets
        }

        # Track approved settlement contracts (shared across assets)
        self.approved_settlements: dict[str, int] = {}  # settlement_address -> approved_amount

        # Async HTTP client for non-blocking price fetches
        self._http_client: httpx.AsyncClient | None = None

    def calculate_shares_from_usdc(self, usdc_amount: float, price: int) -> int:
        """Calculate shares from USDC amount at given price.

        Args:
            usdc_amount: Amount of USDC to spend (e.g., 10.0 for $10)
            price: Price per share in 6 decimals (e.g., 500000 = 50%)

        Returns:
            Number of shares in 6 decimals
        """
        if price <= 0:
            return 0
        # shares = usdc / (price / 1_000_000) = usdc * 1_000_000 / price
        # Result in 6 decimals
        return math.ceil((usdc_amount * 1_000_000 * 1_000_000) / price)

    def get_position_usdc(self, state: AssetState, market_id: str) -> float:
        """Get current position in USDC for a market."""
        return state.position_usdc.get(market_id, 0.0)

    def can_trade(self, state: AssetState, usdc_amount: float) -> bool:
        """Check if trade would exceed max position for this asset."""
        current = self.get_position_usdc(state, state.market_id)
        return (current + usdc_amount) <= self.max_position_usdc

    # Half of max uint256 — threshold for "already has max approval"
    MAX_APPROVAL_THRESHOLD = (2**256 - 1) // 2

    def ensure_settlement_approved(self, settlement_address: str) -> None:
        """Ensure USDC is approved for the settlement contract.

        Uses a gasless max permit via the relayer. No native gas required.
        """
        # Check if already approved in this session
        if settlement_address in self.approved_settlements:
            return

        # Check on-chain allowance
        current_allowance = self.client.get_usdc_allowance(spender=settlement_address)

        if current_allowance >= self.MAX_APPROVAL_THRESHOLD:
            print(f"  Existing USDC max approval found")
            self.approved_settlements[settlement_address] = current_allowance
            return

        # Need to approve via gasless max permit
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
                        print(f"✓ Max USDC approval confirmed (gasless)")
                        self.approved_settlements[settlement_address] = allowance
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                print(f"⚠ Approval pending (may still confirm)")
                self.approved_settlements[settlement_address] = 2**256 - 1

        except Exception as e:
            print(f"✗ Gasless approval failed: {e}")
            raise

        print(f"{'='*50}\n")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=5.0)
        return self._http_client

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def cleanup_pending_orders(self, state: AssetState) -> None:
        """Check pending orders and remove any that have settled or failed."""
        try:
            pending_trades = self.client.get_pending_trades()
            pending_txs = {t.tx_hash for t in pending_trades
                          if t.market_id == state.market_id
                          and t.buyer_address.lower() == self.client.address.lower()}

            # Remove any TXs that are no longer pending
            resolved_txs = state.pending_order_txs - pending_txs
            if resolved_txs:
                print(f"  [{state.asset}] {len(resolved_txs)} order(s) settled")
                state.pending_order_txs -= resolved_txs

                # Check if they filled by looking at recent trades
                trades = self.client.get_trades(market_id=state.market_id, limit=20)
                my_recent_trades = [t for t in trades
                                   if t.buyer.lower() == self.client.address.lower()
                                   and t.id not in state.processed_trade_ids]

                for trade in my_recent_trades:
                    state.processed_trade_ids.add(trade.id)
                    # Calculate USDC spent for this trade
                    usdc_spent = (trade.size * trade.price) / (1_000_000 * 1_000_000)
                    state.position_usdc[state.market_id] = self.get_position_usdc(state, state.market_id) + usdc_spent

                    outcome_str = "YES" if trade.outcome == 0 else "NO"
                    print(f"  [{state.asset}] Filled: ${usdc_spent:.2f} USDC → {trade.size / 1_000_000:.2f} {outcome_str} shares")

        except Exception as e:
            print(f"  [{state.asset}] Warning: Could not cleanup pending orders: {e}")

    async def sync_position(self, state: AssetState) -> None:
        """Sync position by checking user positions for current market."""
        if not state.market_id:
            return

        try:
            positions = self.client.get_user_positions(
                address=self.client.address,
                chain_id=self.client.chain_id
            )

            for position in positions:
                if position.market_id == state.market_id:
                    # Estimate USDC value from shares (rough estimate at 50% price)
                    total_shares = position.yes_shares + position.no_shares
                    estimated_usdc = (total_shares * 0.5) / 1_000_000

                    if total_shares > 0:
                        print(f"[{state.asset}] Position synced: ~${estimated_usdc:.2f} USDC in shares")
                        state.position_usdc[state.market_id] = estimated_usdc
                    else:
                        print(f"[{state.asset}] Position synced: No existing positions")
                        state.position_usdc[state.market_id] = 0.0
                    return

            state.position_usdc[state.market_id] = 0.0
            print(f"[{state.asset}] Position synced: No existing positions")

        except Exception as e:
            print(f"[{state.asset}] Failed to sync position: {e}")
            state.position_usdc[state.market_id] = 0.0

    async def get_current_prices(self) -> dict[str, float]:
        """Fetch current prices for all active assets from Pyth Network in one request."""
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

            # Map feed IDs back to assets
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

    async def calculate_signal(self, state: AssetState, current_price: float) -> tuple[str, float]:
        """Calculate trading signal based on current price vs strike price for an asset."""
        if current_price <= 0:
            return "HOLD", 0.0

        strike_usd = state.strike_price / 1e6
        if strike_usd <= 0:
            return "HOLD", 0.0

        price_diff_pct = ((current_price - strike_usd) / strike_usd) * 100
        threshold_pct = PRICE_THRESHOLD_BPS / 100

        if abs(price_diff_pct) < threshold_pct:
            return "HOLD", 0.0

        raw_confidence = min(abs(price_diff_pct) / 2, MAX_CONFIDENCE)
        confidence = max(raw_confidence, MIN_CONFIDENCE) if abs(price_diff_pct) >= threshold_pct else 0.0

        if price_diff_pct > 0:
            print(f"[{state.asset}] ${current_price:,.2f} is {price_diff_pct:+.2f}% above strike ${strike_usd:,.2f}")
            return "BUY_YES", confidence
        else:
            print(f"[{state.asset}] ${current_price:,.2f} is {price_diff_pct:+.2f}% below strike ${strike_usd:,.2f}")
            return "BUY_NO", confidence

    async def execute_signal(self, state: AssetState, action: str, confidence: float) -> None:
        """Execute the trading signal using gasless max permit (no per-order permits)."""
        if action == "HOLD" or confidence < MIN_CONFIDENCE:
            return

        if state.pending_order_txs:
            print(f"[{state.asset}] ⏳ Waiting for {len(state.pending_order_txs)} pending order(s) to settle...")
            return

        if state.market_expiring:
            print(f"[{state.asset}] ⏰ Market expiring soon - not placing new orders")
            return

        # Check position limits (in USDC)
        if not self.can_trade(state, self.order_size_usdc):
            current = self.get_position_usdc(state, state.market_id)
            print(f"[{state.asset}] Position limit reached: ${current:.2f} / ${self.max_position_usdc:.2f}")
            return

        outcome = Outcome.YES if action == "BUY_YES" else Outcome.NO

        try:
            orderbook = self.client.get_orderbook(state.market_id, outcome=outcome)
        except Exception as e:
            print(f"[{state.asset}] Failed to get orderbook: {e}")
            return

        if not orderbook.asks:
            print(f"[{state.asset}] No asks available for {outcome.name}")
            return

        # Pay slightly above best ask to ensure fill
        price = min(orderbook.asks[0].price + 5000, 999000)

        # Calculate shares from USDC amount
        shares = self.calculate_shares_from_usdc(self.order_size_usdc, price)
        if shares <= 0:
            print(f"[{state.asset}] Order too small: ${self.order_size_usdc:.2f} at {price/10000:.1f}%")
            return

        # Check USDC balance before trading
        try:
            usdc_balance = self.client.get_usdc_balance()
            balance_usdc = usdc_balance / 1_000_000
            if balance_usdc < self.order_size_usdc:
                print(f"[{state.asset}] ⚠️  Insufficient USDC balance: ${balance_usdc:.2f} < ${self.order_size_usdc:.2f} order size")
                print(f"   Fund wallet: {self.client.address}")
                return
        except Exception:
            pass  # Don't block trading if balance check fails

        try:
            # Create order without per-trade permit (using max permit allowance)
            order = self.client.create_limit_buy(
                market_id=state.market_id,
                outcome=outcome,
                price=price,
                size=shares,
                expiration=int(time.time()) + 300,
                settlement_address=state.settlement_address,
            )

            # No per-trade permit - relying on one-time max permit allowance

            result = self.client.post_order(order)
            outcome_str = "YES" if outcome == Outcome.YES else "NO"

            if result and isinstance(result, dict):
                status = result.get("status", "unknown")
                order_hash = result.get("orderHash", order.order_hash)

                print(f"[{state.asset}] → Order submitted: {outcome_str} @ {price / 10000:.1f}% | ${self.order_size_usdc:.2f} = {shares/1_000_000:.4f} shares (status: {status})")

                await asyncio.sleep(2)

                # Check for failed trades
                try:
                    failed_trades = self.client.get_failed_trades()
                    my_failed = [t for t in failed_trades
                                 if t.market_id == state.market_id
                                 and t.buyer_address.lower() == self.client.address.lower()
                                 and t.fill_size == shares]

                    if my_failed:
                        failed = my_failed[0]
                        reason = failed.reason
                        if "simulation" in reason.lower():
                            try:
                                usdc_balance = self.client.get_usdc_balance()
                                balance_usdc = usdc_balance / 1_000_000
                                reason += f" (USDC balance: ${balance_usdc:.2f})"
                            except Exception:
                                pass
                        print(f"[{state.asset}] ✗ Order FAILED: {reason}")
                        return
                except Exception as e:
                    print(f"  [{state.asset}] Warning: Could not check failed trades: {e}")

                # Check for pending trades
                try:
                    pending_trades = self.client.get_pending_trades()
                    my_pending = [t for t in pending_trades
                                  if t.market_id == state.market_id
                                  and t.buyer_address.lower() == self.client.address.lower()
                                  and t.fill_size == shares]

                    if my_pending:
                        pending = my_pending[0]
                        print(f"[{state.asset}] ⏳ Order PENDING on-chain (TX: {pending.tx_hash[:16]}...)")
                        state.pending_order_txs.add(pending.tx_hash)
                        return
                except Exception as e:
                    print(f"  [{state.asset}] Warning: Could not check pending trades: {e}")

                # Check if immediately filled
                try:
                    trades = self.client.get_trades(market_id=state.market_id, limit=20)
                    recent_threshold = time.time() - 10
                    my_trades = [t for t in trades
                                 if t.buyer.lower() == self.client.address.lower()
                                 and t.timestamp > recent_threshold
                                 and t.id not in state.processed_trade_ids]

                    if my_trades:
                        trade = my_trades[0]
                        state.processed_trade_ids.add(trade.id)

                        # Track position in USDC
                        usdc_spent = (trade.size * trade.price) / (1_000_000 * 1_000_000)
                        state.position_usdc[state.market_id] = self.get_position_usdc(state, state.market_id) + usdc_spent

                        print(f"[{state.asset}] ✓ FILLED: ${usdc_spent:.2f} USDC → {trade.size / 1_000_000:.4f} shares")
                        return
                except Exception as e:
                    print(f"  [{state.asset}] Warning: Could not check trades: {e}")

                # Check if still open on orderbook
                try:
                    my_orders = self.client.get_orders(
                        trader=self.client.address,
                        market_id=state.market_id,
                    )
                    matching = [o for o in my_orders if o.order_hash == order_hash]

                    if matching:
                        print(f"[{state.asset}] ✓ Order OPEN on orderbook")
                        state.active_orders[order_hash] = action
                    else:
                        print(f"[{state.asset}] ⚠ Order not found - may have been rejected")
                except Exception as e:
                    print(f"  [{state.asset}] Warning: Could not check open orders: {e}")

            else:
                print(f"[{state.asset}] ⚠ Unexpected order response: {result}")

        except TurbineApiError as e:
            print(f"[{state.asset}] ✗ Order failed: {e}")
        except Exception as e:
            print(f"[{state.asset}] ✗ Unexpected error: {e}")

    async def price_action_loop(self) -> None:
        """Main loop that monitors prices and executes trades for all assets."""
        if CLAIM_ONLY_MODE:
            print("CLAIM ONLY MODE - Trading disabled")
            while self.running:
                await asyncio.sleep(60)
            return

        while self.running:
            try:
                # Fetch all prices in one request
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

                    action, confidence = await self.calculate_signal(state, current_price)
                    if action != "HOLD":
                        await self.execute_signal(state, action, confidence)
                    else:
                        strike_usd = state.strike_price / 1e6
                        if strike_usd > 0:
                            diff_pct = ((current_price - strike_usd) / strike_usd) * 100
                            pos = self.get_position_usdc(state, state.market_id)
                            print(f"[{asset}] ${current_price:,.2f} ({diff_pct:+.2f}% from ${strike_usd:,.2f}) | Pos: ${pos:.2f}/${self.max_position_usdc:.2f} - HOLD")

                await asyncio.sleep(PRICE_POLL_SECONDS)
            except Exception as e:
                print(f"Price action error: {e}")
                await asyncio.sleep(PRICE_POLL_SECONDS)

    async def get_active_market(self, asset: str) -> tuple[str, int, int] | None:
        """Get the currently active quick market for an asset."""
        response = self.client._http.get(f"/api/v1/quick-markets/{asset}")
        quick_market_data = response.get("quickMarket")
        if not quick_market_data:
            return None
        quick_market = QuickMarket.from_dict(quick_market_data)
        return quick_market.market_id, quick_market.end_time, quick_market.start_price

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders by querying the API."""
        try:
            open_orders = self.client.get_orders(
                trader=self.client.address, status="open"
            )
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
        """Cancel all open orders for a specific asset's market."""
        if not state.market_id:
            return
        try:
            open_orders = self.client.get_orders(
                trader=self.client.address,
                market_id=state.market_id,
                status="open",
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
                    print(f"[{state.asset}] Failed to cancel order: {e}")
        state.active_orders.clear()

    async def switch_to_new_market(self, state: AssetState, new_market_id: str, start_price: int = 0) -> None:
        """Switch an asset to a new market and ensure gasless USDC approval."""
        old_market_id = state.market_id

        # Track old market for claiming winnings
        if old_market_id and state.contract_address:
            state.traded_markets[old_market_id] = state.contract_address

        if old_market_id:
            print(f"\n{'='*50}")
            print(f"[{state.asset}] MARKET TRANSITION")
            print(f"Old: {old_market_id[:8]}... | New: {new_market_id[:8]}...")
            print(f"{'='*50}\n")
            await self.cancel_asset_orders(state)

        # Update market state
        state.market_id = new_market_id
        state.strike_price = start_price
        state.active_orders = {}
        state.processed_trade_ids.clear()
        state.pending_order_txs.clear()
        state.market_expiring = False

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

        # Ensure gasless USDC approval for this settlement contract
        if state.settlement_address:
            self.ensure_settlement_approved(state.settlement_address)

        strike_usd = start_price / 1e6 if start_price else 0
        print(f"[{state.asset}] Trading market: {new_market_id[:8]}... | Strike: ${strike_usd:,.2f}")

        await self.sync_position(state)

    async def monitor_market_transitions(self) -> None:
        """Background task that polls for new markets across all assets."""
        POLL_INTERVAL = 5

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
                        await self.switch_to_new_market(state, new_market_id, start_price)

                    time_remaining = end_time - int(time.time())
                    if time_remaining <= 60 and time_remaining > 0:
                        if not state.market_expiring:
                            print(f"[{asset}] ⏰ Market expires in {time_remaining}s - stopping trades")
                            state.market_expiring = True
                    elif time_remaining > 60:
                        state.market_expiring = False

            except Exception as e:
                print(f"Market monitor error: {e}")

            await asyncio.sleep(POLL_INTERVAL)

    async def claim_resolved_markets(self) -> None:
        """Background task to claim winnings from resolved markets across all assets."""
        retry_delay = 120

        while self.running:
            try:
                # Collect all traded markets across all assets
                all_traded: list[tuple[str, str, AssetState]] = []
                for state in self.asset_states.values():
                    for market_id, contract_address in list(state.traded_markets.items()):
                        all_traded.append((market_id, contract_address, state))

                if not all_traded:
                    await asyncio.sleep(retry_delay)
                    continue

                # Collect all resolved markets first
                resolved: list[tuple[str, str, AssetState]] = []
                for market_id, contract_address, state in all_traded:
                    try:
                        resolution = self.client.get_resolution(market_id)
                        if resolution and resolution.resolved:
                            resolved.append((market_id, contract_address, state))
                    except Exception:
                        continue

                if not resolved:
                    await asyncio.sleep(retry_delay)
                    continue

                # Batch claim in one transaction
                market_addresses = [addr for _, addr, _ in resolved]
                try:
                    result = self.client.batch_claim_winnings(market_addresses)
                    tx_hash = result.get("txHash", result.get("tx_hash", "unknown"))
                    print(f"💰 Batch claimed {len(resolved)} markets TX: {tx_hash}")
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

            await asyncio.sleep(retry_delay)

    async def run(self) -> None:
        """Main trading loop."""
        monitor_task = asyncio.create_task(self.monitor_market_transitions())
        claim_task = asyncio.create_task(self.claim_resolved_markets())
        price_task = asyncio.create_task(self.price_action_loop())

        try:
            # Initialize all asset markets
            for asset in self.assets:
                try:
                    market_info = await self.get_active_market(asset)
                    if market_info:
                        market_id, _, start_price = market_info
                        await self.switch_to_new_market(self.asset_states[asset], market_id, start_price)
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
            price_task.cancel()
            await asyncio.gather(monitor_task, claim_task, price_task, return_exceptions=True)
            await self.cancel_all_orders()
            await self.close()


async def main():
    parser = argparse.ArgumentParser(
        description="Turbine price action bot for BTC, ETH, and SOL 15-minute prediction markets"
    )
    parser.add_argument(
        "-s", "--order-size",
        type=float,
        default=DEFAULT_ORDER_SIZE_USDC,
        help=f"Order size in USDC (default: ${DEFAULT_ORDER_SIZE_USDC})"
    )
    parser.add_argument(
        "-m", "--max-position",
        type=float,
        default=DEFAULT_MAX_POSITION_USDC,
        help=f"Max position per asset in USDC (default: ${DEFAULT_MAX_POSITION_USDC})"
    )
    parser.add_argument(
        "-a", "--assets",
        type=str,
        default=",".join(SUPPORTED_ASSETS),
        help=f"Comma-separated list of assets to trade (default: {','.join(SUPPORTED_ASSETS)})"
    )
    args = parser.parse_args()

    # Parse and validate assets
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
    print(f"TURBINE PRICE ACTION BOT")
    print(f"{'='*60}")
    print(f"Wallet: {client.address}")
    print(f"Chain: {CHAIN_ID}")
    print(f"Assets: {', '.join(assets)}")
    print(f"Order size: ${args.order_size:.2f} USDC")
    print(f"Max position: ${args.max_position:.2f} USDC per asset")
    print(f"USDC approval: gasless (one-time max permit per settlement)")
    try:
        usdc_balance = client.get_usdc_balance()
        balance_display = usdc_balance / 1_000_000
        print(f"USDC balance: ${balance_display:.2f}")
        min_needed = args.order_size * len(assets)
        if balance_display < min_needed:
            print(f"⚠️  Warning: Balance (${balance_display:.2f}) may be low for {len(assets)} assets")
            print(f"   Fund your wallet: {client.address}")
    except Exception as e:
        print(f"USDC balance: unknown ({e})")
    print(f"{'='*60}\n")

    bot = PriceActionBot(
        client,
        assets=assets,
        order_size_usdc=args.order_size,
        max_position_usdc=args.max_position,
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
