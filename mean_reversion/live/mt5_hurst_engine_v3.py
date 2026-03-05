import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from numba import jit
import time
from datetime import datetime
import pytz

# ═════════════════════════════════════════════════════════════════════════════
# 1. PRODUCTION CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
SYMBOL = "MNQH26"          # Front-month Micro NQ
MAGIC_NUMBER = 999111      # Engine 3 Unique ID
LOT_SIZE = 2.0             # Fixed at 2 Micro Contracts for $50k Risk Parity
HARD_STOP_USD = -800.0     # Strategy-level circuit breaker

TIMEFRAME = mt5.TIMEFRAME_M15
HURST_WINDOW = 200
SMA_PERIOD = 20
NY_TZ = pytz.timezone("America/New_York")

# ═════════════════════════════════════════════════════════════════════════════
# 2. MATHEMATICAL KERNELS (NUMBA ACCELERATED)
# ═════════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def calculate_latest_hurst(prices):
    """Calculates the Hurst Exponent for the most recent window of data."""
    taus = np.array([2, 4, 8, 16, 32])
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
# 3. MT5 INTERFACE & EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def connect_mt5():
    if not mt5.initialize():
        print(f"[{datetime.now()}] MT5 Init Failed: {mt5.last_error()}")
        return False
    if not mt5.symbol_select(SYMBOL, True):
        print(f"[{datetime.now()}] Symbol {SYMBOL} not found.")
        return False
    return True

def get_market_data():
    """Fetches sufficient M15 bars to calculate Hurst and SMA."""
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 400)
    if rates is None or len(rates) < 250:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def close_position(comment="Exit Triggered"):
    positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
    if not positions:
        return
    
    for pos in positions:
        tick = mt5.symbol_info_tick(SYMBOL)
        type_dict = {mt5.POSITION_TYPE_BUY: mt5.ORDER_TYPE_SELL, mt5.POSITION_TYPE_SELL: mt5.ORDER_TYPE_BUY}
        price_dict = {mt5.POSITION_TYPE_BUY: tick.bid, mt5.POSITION_TYPE_SELL: tick.ask}
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": type_dict[pos.type],
            "position": pos.ticket,
            "price": price_dict[pos.type],
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        print(f"[{datetime.now()}] FLATTEN: {comment} | Result: {res.comment}")

def execute_entry():
    tick = mt5.symbol_info_tick(SYMBOL)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 10,
        "magic": MAGIC_NUMBER,
        "comment": "Hurst MR Entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(request)
    print(f"[{datetime.now()}] ENTRY: Hurst MR Long | Result: {res.comment}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. PRODUCTION WATCHDOG LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_v3():
    print(f"[{datetime.now()}] Engine 3 (Structural Hurst) starting...")
    if not connect_mt5(): return
    
    last_processed_bar = None
    
    while True:
        try:
            # 1. Fetch current data and state
            df = get_market_data()
            if df is None:
                time.sleep(1)
                continue
                
            current_bar_time = df.iloc[-1]['time']
            current_price = df.iloc[-1]['close']
            positions = mt5.positions_get(symbol=SYMBOL, magic=MAGIC_NUMBER)
            
            # 2. MID-CANDLE WATCHDOG (Hard Stop and SMA Exit)
            if positions:
                unrealized_pnl = sum([p.profit for p in positions])
                
                # Circuit Breaker Check
                if unrealized_pnl <= HARD_STOP_USD:
                    close_position("CIRCUIT BREAKER HIT")
                    continue
                
                # Instant SMA Exit Check
                closes = df['close'].values
                sma = np.mean(closes[-SMA_PERIOD:])
                if current_price >= sma:
                    close_position("Target Met (SMA)")
                    continue

            # 3. NEW CANDLE LOGIC (Hurst Recalculation)
            if last_processed_bar != current_bar_time:
                # We only process on the first tick of a new bar, or first run
                last_processed_bar = current_bar_time
                
                closes = df['close'].values
                # We calculate stats using bars PRIOR to the currently forming bar
                window_prices = closes[-(HURST_WINDOW + 1):-1] 
                
                h_val = calculate_latest_hurst(window_prices)
                sma = np.mean(closes[-SMA_PERIOD-1:-1])
                std = np.std(closes[-SMA_PERIOD-1:-1])
                lower_band = sma - (1.5 * std)
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] SCAN: H={h_val:.4f} | SMA={sma:.2f} | Price={current_price:.2f} | Band={lower_band:.2f}")
                
                # Check Entry Condition
                if not positions:
                    if h_val < 0.48 and current_price < lower_band:
                        execute_entry()
                
                # Check Regime Change Exit
                else:
                    if h_val > 0.55:
                        close_position("Regime Change (H > 0.55)")

            time.sleep(0.5) # High-frequency polling
            
        except KeyboardInterrupt:
            print("\nShutting down engine...")
            break
        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_engine_v3()
    mt5.shutdown()
