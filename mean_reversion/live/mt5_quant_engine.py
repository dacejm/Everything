import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import os
import traceback

# ═════════════════════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION & DUAL-SYMBOL SETUP
# ═════════════════════════════════════════════════════════════════════════════

DATA_SYMBOL = "@ENQ"       # Continuous reference contract for robust historical data
EXEC_SYMBOL = "MNQH26"     # Tradable front-month micro contract for order routing

# We use two distinct Magic Numbers to easily separate the Scale-Out and the Runner
MAGIC_SCALE_OUT = 8884441
MAGIC_RUNNER = 8884442

ACCOUNT_SIZE = 50000.0
RISK_AMOUNT = ACCOUNT_SIZE * 0.01  # $500 Max Risk
MNQ_POINT_VALUE = 2.0

NY_TIMEZONE = pytz.timezone("America/New_York")

# Quant Model Parameters
WARMUP_DAYS = 45
VOL_PERIOD = 20
SKEW_PERIOD = 5

# ═════════════════════════════════════════════════════════════════════════════
# 2. GLOBAL STATE INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════
daily_levels_generated = False
current_bias = None
p75_level = 0.0
p50_level = 0.0
p95_level = 0.0

# ═════════════════════════════════════════════════════════════════════════════
# 3. STATE MANAGEMENT & INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def connect_mt5():
    """Initializes connection and ensures both symbols are visible."""
    if not mt5.initialize():
        print(f"[{datetime.now()}] CRITICAL: MT5 initialization failed. Code: {mt5.last_error()}")
        return False
    
    for sym in [DATA_SYMBOL, EXEC_SYMBOL]:
        if not mt5.symbol_select(sym, True):
            print(f"[{datetime.now()}] ERROR: Failed to select {sym} in Market Watch.")
            return False
            
    print(f"[{datetime.now()}] MT5 Connected. Data: {DATA_SYMBOL} | Exec: {EXEC_SYMBOL}")
    return True

def has_active_trades_today():
    """Checks for open positions or resting orders on the execution symbol."""
    # Check resting orders
    orders = mt5.orders_get(symbol=EXEC_SYMBOL)
    if orders is not None:
        for order in orders:
            if order.magic in [MAGIC_SCALE_OUT, MAGIC_RUNNER]:
                print(f"[{datetime.now(NY_TIMEZONE).strftime('%H:%M:%S')}] Resting order already exists. Skipping entry.")
                return True
                
    # Check open positions
    positions = mt5.positions_get(symbol=EXEC_SYMBOL)
    if positions is not None:
        for pos in positions:
            if pos.magic in [MAGIC_SCALE_OUT, MAGIC_RUNNER]:
                print(f"[{datetime.now(NY_TIMEZONE).strftime('%H:%M:%S')}] Open position already exists. Skipping entry.")
                return True
                
    return False

def flatten_and_cancel():
    """Kill Switch: Cancels all resting orders and closes open positions."""
    print(f"[{datetime.now(NY_TIMEZONE).strftime('%H:%M:%S')}] Initiating End-of-Day Flatten & Cancel Sequence...")
    
    # 1. Cancel Resting Orders
    orders = mt5.orders_get(symbol=EXEC_SYMBOL)
    if orders:
        for order in orders:
            if order.magic in [MAGIC_SCALE_OUT, MAGIC_RUNNER]:
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                }
                res = mt5.order_send(request)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"[{datetime.now(NY_TIMEZONE).strftime('%H:%M:%S')}] Canceled Order: {order.ticket}")

    # 2. Close Open Positions
    positions = mt5.positions_get(symbol=EXEC_SYMBOL)
    if positions:
        for pos in positions:
            if pos.magic in [MAGIC_SCALE_OUT, MAGIC_RUNNER]:
                tick = mt5.symbol_info_tick(EXEC_SYMBOL)
                order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": EXEC_SYMBOL,
                    "volume": pos.volume,
                    "type": order_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 20,
                    "magic": pos.magic,
                    "comment": "Time Stop Flatten",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res = mt5.order_send(request)
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"[{datetime.now(NY_TIMEZONE).strftime('%H:%M:%S')}] Closed Position: {pos.ticket}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. DATA PIPELINE & MATH CORE (ENGINE 2)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_data():
    """Fetches historical 1-minute data from the continuous reference contract."""
    rates = mt5.copy_rates_from_pos(DATA_SYMBOL, mt5.TIMEFRAME_M1, 0, 100000)
    if rates is None or len(rates) == 0:
        print(f"[{datetime.now()}] Data Fetch Failed for {DATA_SYMBOL}. Code: {mt5.last_error()}")
        return None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.index = df.index.tz_convert(NY_TIMEZONE)
    return df

def calculate_quant_signals(df):
    """Calculates Engine 2 logic on the continuous reference data."""
    global current_bias, p75_level, p50_level, p95_level
    
    df['time_str'] = df.index.strftime('%H:%M')
    rth_mask = (df['time_str'] >= '09:30') & (df['time_str'] <= '15:59')
    df_rth = df[rth_mask].copy()
    df_rth['date'] = df_rth.index.date
    
    df_rth['min_in_sess'] = (df_rth.index.hour - 9) * 60 + df_rth.index.minute - 30
    
    df_daily = df_rth.groupby('date').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    })
    
    df_daily['up_move'] = df_daily['high'] - df_daily['open']
    df_daily['dn_move'] = df_daily['open'] - df_daily['low']
    up_var = df_daily['up_move'].rolling(SKEW_PERIOD).var()
    dn_var = df_daily['dn_move'].rolling(SKEW_PERIOD).var()
    df_daily['bias'] = 0 
    df_daily.loc[dn_var > up_var * 1.2, 'bias'] = -1
    df_daily.loc[up_var > dn_var * 1.2, 'bias'] = 1 
    
    df_daily['vol'] = np.log(df_daily['close'] / df_daily['close'].shift(1)).rolling(20).std(ddof=1)
    
    ib_mask = df_rth['min_in_sess'] < 60
    ib_data = df_rth[ib_mask].groupby('date').agg({'high': 'max', 'low': 'min'}).rename(columns={'high': 'IBH', 'low': 'IBL'})
    df_daily = df_daily.merge(ib_data, left_index=True, right_index=True)
    
    df_daily['ext_up'] = (df_daily['high'] - df_daily['IBH']).clip(lower=0)
    df_daily['ext_dn'] = (df_daily['IBL'] - df_daily['low']).clip(lower=0)
    df_daily.dropna(inplace=True)
    
    if len(df_daily) < WARMUP_DAYS + 1:
        return None
        
    today = df_daily.iloc[-1]
    
    cur_vol = float(today['vol'])
    cur_bias = int(today['bias'])
    ib_high = float(today['IBH'])
    ib_low = float(today['IBL'])
    
    print(f"DEBUG: IB High is type {type(ib_high)} value {ib_high}")
    print(f"DEBUG: Bias is {cur_bias} (type {type(cur_bias)})")
    
    if cur_vol == 0: 
        return None
        
    history = df_daily.iloc[-WARMUP_DAYS-1:-1]
    
    scaled_up = history['ext_up'].values * (cur_vol / history['vol'].values)
    scaled_dn = history['ext_dn'].values * (cur_vol / history['vol'].values)
            
    p50_up, p75_up, p95_up = np.percentile(scaled_up, [50, 75, 95])
    p50_dn, p75_dn, p95_dn = np.percentile(scaled_dn, [50, 75, 95])
    
    p50_up, p75_up, p95_up = float(p50_up), float(p75_up), float(p95_up)
    p50_dn, p75_dn, p95_dn = float(p50_dn), float(p75_dn), float(p95_dn)
    
    tick_size = mt5.symbol_info(EXEC_SYMBOL).trade_tick_size
    def round_tick(val): return round(val / tick_size) * tick_size
        
    signal = None
    if cur_bias == -1:
        signal = {
            'dir': 'SHORT',
            'type': mt5.ORDER_TYPE_SELL_LIMIT, 
            'entry': round_tick(ib_high + p75_up), 
            'tp': round_tick(ib_high + p50_up),
            'sl': round_tick(ib_high + p95_up)
        }
    elif cur_bias == 1:
        signal = {
            'dir': 'LONG',
            'type': mt5.ORDER_TYPE_BUY_LIMIT, 
            'entry': round_tick(ib_low - p75_dn), 
            'tp': round_tick(ib_low - p50_dn),
            'sl': round_tick(ib_low - p95_dn)
        }
    
    if signal:
        current_bias = signal['dir']
        p75_level = signal['entry']
        p50_level = signal['tp']
        p95_level = signal['sl']
        
    return signal

# ═════════════════════════════════════════════════════════════════════════════
# 5. ORDER ROUTING (SPLIT SIZING)
# ═════════════════════════════════════════════════════════════════════════════

def route_split_orders(signal):
    """Calculates risk, splits position into halves, and places two limit orders with volume normalization."""
    risk_pts = abs(signal['entry'] - signal['sl'])
    if risk_pts == 0: return False
    
    # 1. Fetch Symbol Info for Volume Constraints
    sym_info = mt5.symbol_info(EXEC_SYMBOL)
    if sym_info is None:
        print(f"[{datetime.now()}] ERROR: Could not fetch symbol info for {EXEC_SYMBOL}")
        return False
        
    vol_step = sym_info.volume_step
    min_vol = sym_info.volume_min
    
    # 2. Calculate Total Qty
    total_qty_raw = RISK_AMOUNT / (risk_pts * MNQ_POINT_VALUE)
    # Ensure total_qty is at least min_vol and a multiple of vol_step
    total_qty = max(min_vol, np.floor(total_qty_raw / vol_step) * vol_step)
    total_qty = round(total_qty, 2) # Clean up float precision
    
    # 3. Dynamic Runner Logic
    if total_qty <= min_vol:
        # If total qty is at minimum, we cannot split. Full position is Scale-Out (hits TP at P50).
        qty_h1 = total_qty
        qty_h2 = 0.0
        print(f"[DEBUG] Total Qty {total_qty} <= Min Vol {min_vol}. No runner logic possible.")
    else:
        # Split into two halves
        qty_h2_raw = (total_qty / 2.0)
        # Normalize H2 to vol_step
        qty_h2 = np.floor(qty_h2_raw / vol_step) * vol_step
        qty_h1 = total_qty - qty_h2
        
        # Ensure H2 is still at least min_vol if it's not 0
        if qty_h2 < min_vol and qty_h2 > 0:
            qty_h1 = total_qty
            qty_h2 = 0.0
            print(f"[DEBUG] Split Runner {qty_h2} < Min Vol {min_vol}. Reverting to full Scale-Out.")

    qty_h1 = round(qty_h1, 2)
    qty_h2 = round(qty_h2, 2)
    
    print(f"[DEBUG] Adjusted Scale-Out Qty: {qty_h1}")
    print(f"[DEBUG] Adjusted Runner Qty: {qty_h2}")
    print(f"[{datetime.now(NY_TIMEZONE).strftime('%H:%M:%S')}] Signal: {signal['dir']} @ {signal['entry']} | Risk: {risk_pts:.2f} pts | Total Qty: {total_qty}")
    
    now_ny = datetime.now(NY_TIMEZONE)
    eod_utc = now_ny.replace(hour=15, minute=59, second=0).astimezone(pytz.utc)
    
    def send_order(qty, tp, magic, comment):
        if qty <= 0: return
        
        print(f"[DEBUG] Sending {comment} with type_time: ORDER_TIME_GTC")
        
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": EXEC_SYMBOL,
            "volume": float(qty),
            "type": signal['type'],
            "price": float(signal['entry']),
            "sl": float(signal['sl']),
            "tp": float(tp), 
            "deviation": 10,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        res = mt5.order_send(request)
        
        # Fallback Logic: If GTC fails with 10022, try ORDER_TIME_DAY
        if res.retcode == 10022 and "expiration" in res.comment.lower():
            print(f" -> [DEBUG] GTC Unsupported. Retrying with ORDER_TIME_DAY...")
            request["type_time"] = mt5.ORDER_TIME_DAY
            res = mt5.order_send(request)

        if res.retcode != mt5.TRADE_RETCODE_DONE:
            print(f" -> ERROR: {comment} Rejected. Code: {res.retcode} | {res.comment}")
        else:
            print(f" -> SUCCESS: {comment} Placed. Qty: {qty} | Ticket: {res.order}")

    # Order 1: Scale-Out (Has P50 Target)
    send_order(qty_h1, signal['tp'], MAGIC_SCALE_OUT, "H1_ScaleOut")
    
    # Order 2: Runner
    if qty_h2 > 0:
        runner_tp = signal['entry'] - 500 if signal['dir'] == 'SHORT' else signal['entry'] + 500
        send_order(qty_h2, runner_tp, MAGIC_RUNNER, "H2_Runner")
    
    return True

# ═════════════════════════════════════════════════════════════════════════════
# 6. WATCHDOG: LIVE TRADE MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def watchdog_loop(entry_price):
    """
    Monitors active trades.
    1. If Scale-Out hits target, shifts Runner Stop Loss to Breakeven.
    2. Flattens all positions at 15:45 EST.
    """
    print(f"[{datetime.now(NY_TIMEZONE).strftime('%H:%M:%S')}] Watchdog active. Polling broker at 10Hz (100ms).")
    breakeven_shifted = False
    
    while True:
        now_ny = datetime.now(NY_TIMEZONE)
        
        # 1. TIME STOP
        if now_ny.hour == 15 and now_ny.minute >= 45:
            flatten_and_cancel()
            print(f"[{now_ny.strftime('%H:%M:%S')}] Session closed. Watchdog disengaging.")
            break
            
        # 2. BREAKEVEN SHIFT LOGIC
        if not breakeven_shifted:
            positions = mt5.positions_get(symbol=EXEC_SYMBOL)
            runner_pos = None
            scale_out_active = False
            
            if positions is not None:
                for p in positions:
                    if p.magic == MAGIC_RUNNER:
                        runner_pos = p
                    elif p.magic == MAGIC_SCALE_OUT:
                        scale_out_active = True
            
            if runner_pos is not None:
                if not scale_out_active:
                    if entry_price > 0:
                        if abs(runner_pos.sl - entry_price) > 0.01:
                            print(f"[{now_ny.strftime('%H:%M:%S')}] SCALE-OUT TARGET HIT! Shifting Runner to Breakeven...")
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": runner_pos.ticket,
                                "symbol": EXEC_SYMBOL,
                                "sl": float(entry_price),
                                "tp": float(runner_pos.tp)
                            }
                            res = mt5.order_send(request)
                            if res.retcode == mt5.TRADE_RETCODE_DONE:
                                print(f"[{now_ny.strftime('%H:%M:%S')}] SUCCESS: Runner Stop Loss updated to {entry_price}")
                                breakeven_shifted = True
                            else:
                                print(f"[{now_ny.strftime('%H:%M:%S')}] RETRYING: Runner SL modification failed (Code: {res.retcode})")
                            
        time.sleep(0.1) # 10Hz polling

# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN SERVER LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_server():
    global daily_levels_generated
    
    print(f"[{datetime.now()}] Quant Engine v2.0 (Runner Logic) Starting...")
    if not connect_mt5(): return
    
    last_date = None

    while True:
        try:
            now_ny = datetime.now(NY_TIMEZONE)
            
            # Robust New Day Detection
            if last_date != now_ny.date():
                daily_levels_generated = False
                last_date = now_ny.date()
                print(f"\n[{now_ny.strftime('%H:%M:%S')}] --- New Trading Day: {last_date} ---")

            if now_ny.weekday() < 5:
                # ═════════════════════════════════════════════════════════════════════════
                # TRIGGER: Signal Generation (10:30 AM EST to 16:00 PM EST Catch-Up)
                # ═════════════════════════════════════════════════════════════════════════
                current_time_val = now_ny.hour * 100 + now_ny.minute
                
                if current_time_val >= 1030 and current_time_val < 1600 and not daily_levels_generated:
                    if current_time_val > 1031:
                        print(f"[{now_ny.strftime('%H:%M:%S')}] [DEBUG] Catch-Up Logic Triggered: Fetching IB...")
                    else:
                        print(f"[{now_ny.strftime('%H:%M:%S')}] 10:30 AM EST. IB period complete. Generating signals...")
                    
                    if not has_active_trades_today():
                        df = fetch_data()
                        if df is not None:
                            sig = calculate_quant_signals(df)
                            if sig:
                                print(f"[{now_ny.strftime('%H:%M:%S')}] SIGNAL GENERATED:")
                                print(f" -> Bias: {sig['dir']}")
                                print(f" -> Entry: {sig['entry']}")
                                print(f" -> Target (P50): {sig['tp']}")
                                print(f" -> Stop (P95): {sig['sl']}")
                                
                                if route_split_orders(sig):
                                    watchdog_loop(sig['entry'])
                            else:
                                print(f"[{now_ny.strftime('%H:%M:%S')}] Neutral Skew. Standing aside.")
                    
                    daily_levels_generated = True
                    time.sleep(60)
                
                # TRIGGER 2: Kill Switch (15:45 PM EST)
                elif current_time_val >= 1545 and now_ny.hour < 16:
                    flatten_and_cancel()
                    time.sleep(60)
                    
            time.sleep(1)
            
        except KeyboardInterrupt:
            print(f"\n[{datetime.now(NY_TIMEZONE).strftime('%H:%M:%S')}] Manual Shutdown Initiated.")
            flatten_and_cancel()
            break
        except Exception as e:
            print(f"[{datetime.now(NY_TIMEZONE).strftime('%H:%M:%S')}] CRITICAL ERROR: {e}")
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    run_server()
    mt5.shutdown()
