import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from numba import jit
import time
from datetime import datetime
import pytz
import os
import sys

# Core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HURST_WINDOW, SMA_PERIOD, HURST_TAUS, RISK_PCT, MNQ_POINT_VALUE
from core.mt5_base import BaseMT5Engine

# ═════════════════════════════════════════════════════════════════════════════
# 1. PRODUCTION CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
SYMBOL = "MNQH26"          # Front-month Micro NQ
MAGIC_NUMBER = 999111      # Engine 3 Unique ID

TIMEFRAME = mt5.TIMEFRAME_M15
NY_TZ = pytz.timezone("America/New_York")

# ═════════════════════════════════════════════════════════════════════════════
# 2. MATHEMATICAL KERNELS (NUMBA ACCELERATED)
# ═════════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def calculate_latest_hurst(prices):
    """Calculates the Hurst Exponent for the most recent window of data."""
    # FIX 4.5: Remove noisy short tau, add longer tau for better regression fit
    taus = np.array(HURST_TAUS)
    log_taus = np.log(taus)
    log_vars = np.zeros(len(taus))
    
    for i in range(len(taus)):
        tau = taus[i]
        diffs = prices[tau:] - prices[:-tau]
        log_vars[i] = np.log(np.var(diffs) + 1e-8)
        
    # Simple linear regression log(var) vs log(tau)
    cov = np.sum((log_taus - np.mean(log_taus)) * (log_vars - np.mean(log_vars)))
    var = np.sum((log_taus - np.mean(log_taus))**2)
    return (cov / var) / 2.0

# ═════════════════════════════════════════════════════════════════════════════
# 3. ENGINE CLASS
# ═════════════════════════════════════════════════════════════════════════════

class HurstEngine(BaseMT5Engine):
    def __init__(self, symbol=SYMBOL, magic_number=MAGIC_NUMBER, account_size=50000.0):
        super().__init__(symbols=[symbol], magic_numbers=[magic_number], account_size=account_size, log_file="logs/hurst_engine.log")
        self.symbol = symbol
        self.magic_number = magic_number
        self.last_processed_bar = None
        self.cached_sma = None

    def get_market_data(self):
        """Fetches sufficient M15 bars to calculate Hurst and SMA."""
        rates = mt5.copy_rates_from_pos(self.symbol, TIMEFRAME, 0, 400)
        if rates is None or len(rates) < 250:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def close_position(self, comment="Exit Triggered"):
        # Leverage the base flattening function for simplicity
        self.flatten_and_cancel(self.symbol, self.magic_numbers, comment)

    def calculate_position_size(self, entry_price, band_level):
        """FIX 2.4: Risk-based position sizing matching quant engine pattern."""
        risk_pts = abs(entry_price - band_level)
        if risk_pts == 0:
            return 0.0
        risk_amount = self.account_size * RISK_PCT
        qty = risk_amount / (risk_pts * MNQ_POINT_VALUE)
        sym_info = mt5.symbol_info(self.symbol)
        if sym_info is None:
            return 0.0
        qty = max(sym_info.volume_min, np.floor(qty / sym_info.volume_step) * sym_info.volume_step)
        return round(qty, 2)

    def execute_entry(self, direction, lot):
        """Places a mean reversion entry. Direction: 'LONG' or 'SHORT'."""
        # FIX 2.1: Support both long and short entries
        if lot <= 0: return
        tick = mt5.symbol_info_tick(self.symbol)
        if direction == 'LONG':
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(lot),
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": self.magic_number,
            "comment": f"Hurst MR {direction}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        self.logger.info(f"ENTRY: Hurst MR {direction} | Result: {res.comment}")

    def engine_loop(self):
        # FAST POLL: Watchdog checks (position P&L, hard stop)
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic_number)
        if positions:
            unrealized_pnl = sum([p.profit for p in positions])
            # FIX 4.4: Dynamic circuit breaker based on position risk
            MAX_RISK_MULTIPLE = 2.0  # Kill trade if loss exceeds 2x the intended risk
            intended_risk = self.account_size * RISK_PCT
            circuit_breaker = -(intended_risk * MAX_RISK_MULTIPLE)
            
            if unrealized_pnl <= circuit_breaker:
                self.close_position(f"CIRCUIT BREAKER: {unrealized_pnl:.0f} < {circuit_breaker:.0f}")
                time.sleep(0.5)
                return

            # FIX 2.3: SMA exit check using cached value (updated on new bar only)
            if self.cached_sma is not None:
                tick = mt5.symbol_info_tick(self.symbol)
                live_price = (tick.bid + tick.ask) / 2.0
                pos_type = positions[0].type
                # FIX 2.1: Direction-aware SMA exit
                if pos_type == mt5.POSITION_TYPE_BUY and live_price >= self.cached_sma:
                    self.close_position("Target Met (SMA) - Long")
                    time.sleep(0.5)
                    return
                elif pos_type == mt5.POSITION_TYPE_SELL and live_price <= self.cached_sma:
                    self.close_position("Target Met (SMA) - Short")
                    time.sleep(0.5)
                    return

        # SLOW POLL: Only fetch full data and recalculate on new bar
        df = self.get_market_data()
        if df is not None:
            current_bar_time = df.iloc[-1]['time']
            current_price = df.iloc[-1]['close']
            
            if self.last_processed_bar != current_bar_time:
                self.last_processed_bar = current_bar_time
                
                closes = df['close'].values
                window_prices = closes[-(HURST_WINDOW + 1):-1] 
                
                h_val = calculate_latest_hurst(window_prices)
                sma = np.mean(closes[-SMA_PERIOD-1:-1])
                std = np.std(closes[-SMA_PERIOD-1:-1])
                self.cached_sma = sma
                lower_band = sma - (1.5 * std)
                upper_band = sma + (1.5 * std)
                
                self.logger.info(f"SCAN: H={h_val:.4f} | SMA={sma:.2f} | Price={current_price:.2f} | Band={lower_band:.2f}/{upper_band:.2f}")
                
                if not positions:
                    # FIX 2.1: Add short entries for upper band breach
                    if h_val < 0.48 and current_price < lower_band:
                        lot = self.calculate_position_size(current_price, sma)
                        if lot > 0:
                            self.execute_entry('LONG', lot)
                    elif h_val < 0.48 and current_price > upper_band:
                        lot = self.calculate_position_size(current_price, sma)
                        if lot > 0:
                            self.execute_entry('SHORT', lot)
                else:
                    if h_val > 0.55:
                        self.close_position("Regime Change (H > 0.55)")

    def on_shutdown(self):
        self.close_position("Manual Shutdown Flatten")

if __name__ == "__main__":
    engine = HurstEngine()
    engine.safe_run()
    mt5.shutdown()
