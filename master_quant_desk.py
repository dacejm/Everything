import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from numba import jit
import time
from datetime import datetime
import pytz
import os
import traceback

# ═════════════════════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Trading Symbols
DATA_SYMBOL_E2 = "@ENQ"    # Continuous Reference for E2
EXEC_SYMBOL = "MNQH26"     # Tradable Micro for E2 & E3

# Magic Numbers
MAGIC_E2_SCALE_OUT = 8884441
MAGIC_E2_RUNNER = 8884442
MAGIC_E3 = 999111

# Risk Settings
ACCOUNT_SIZE = 50000.0
MNQ_POINT_VALUE = 2.0
RISK_BUCKET_USD = 25000.0 * 0.01  # $250 risk per trade (1% of $25k)
HARD_STOP_E3_USD = -800.0         # Prop firm safety circuit breaker

# Timezones
NY_TZ = pytz.timezone("America/New_York")

# Strategy Parameters
WARMUP_DAYS = 45
VOL_PERIOD = 20
SKEW_PERIOD = 5
HURST_WINDOW = 200
SMA_PERIOD = 20

# ═════════════════════════════════════════════════════════════════════════════
# 2. GLOBAL STATE
# ═════════════════════════════════════════════════════════════════════════════
daily_levels_generated_e2 = False
last_hurst_value_e3 = 0.0
last_bias_e2 = "Neutral"
last_date = None

# ═════════════════════════════════════════════════════════════════════════════
# 3. NOTIFICATION & UTILS
# ═════════════════════════════════════════════════════════════════════════════

def send_trade_alert(message):
    """Webhook placeholder - Currently prints to console."""
    ts = datetime.now(NY_TZ).strftime('%H:%M:%S')
    print(f"\n[ALERT] {ts} | {message}\n")

def connect_mt5():
    if not mt5.initialize():
        print(f"[{datetime.now()}] CRITICAL: MT5 initialization failed.")
        return False
    for sym in [DATA_SYMBOL_E2, EXEC_SYMBOL]:
        if not mt5.symbol_select(sym, True):
            print(f"[{datetime.now()}] ERROR: Failed to select {sym}")
            return False
    return True

def is_desk_flat():
    """Ensures no strategy-specific positions are active (The No-Intercept Rule)."""
    positions = mt5.positions_get(symbol=EXEC_SYMBOL)
    if positions:
        for pos in positions:
            if pos.magic in [MAGIC_E2_SCALE_OUT, MAGIC_E2_RUNNER, MAGIC_E3]:
                return False
    return True

def get_daily_pnl():
    """Calculates total PnL from closed trades today."""
    now = datetime.now(NY_TZ)
    start_of_day = int(now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    deals = mt5.history_deals_get(start_of_day, int(now.timestamp()))
    if not deals: return 0.0
    return sum([d.profit for d in deals if d.magic in [MAGIC_E2_SCALE_OUT, MAGIC_E2_RUNNER, MAGIC_E3]])

# ═════════════════════════════════════════════════════════════════════════════
# 4. MATHEMATICAL KERNELS
# ═════════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def calculate_hurst(prices):
    taus = np.array([2, 4, 8, 16, 32])
    log_taus = np.log(taus)
    log_vars = np.zeros(len(taus))
    for i in range(len(taus)):
        tau = taus[i]
        diffs = prices[tau:] - prices[:-tau]
        log_vars[i] = np.log(np.var(diffs) + 1e-8)
    cov = np.sum((log_taus - np.mean(log_taus)) * (log_vars - np.mean(log_vars)))
    var = np.sum((log_taus - np.mean(log_taus))**2)
    return (cov / var) / 2.0

# ═════════════════════════════════════════════════════════════════════════════
# 5. ENGINE 2: IB MEAN REVERSION
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_2_logic():
    global daily_levels_generated_e2, last_bias_e2
    
    # 1. Fetch Data
    rates = mt5.copy_rates_from_pos(DATA_SYMBOL_E2, mt5.TIMEFRAME_M1, 0, 100000)
    if rates is None or len(rates) == 0: return
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df = df.tz_convert(NY_TZ)
    
    # 2. Daily Features
    df['time_str'] = df.index.strftime('%H:%M')
    df_rth = df[(df['time_str'] >= '09:30') & (df['time_str'] <= '15:59')].copy()
    df_rth['date'] = df_rth.index.date
    df_rth['min_in_sess'] = (df_rth.index.hour * 60 + df_rth.index.minute) - 570 # 09:30 = 570 mins
    
    df_daily = df_rth.groupby('date').agg({'open':'first','high':'max','low':'min','close':'last'})
    df_daily['up_move'] = df_daily['high'] - df_daily['open']
    df_daily['dn_move'] = df_daily['open'] - df_daily['low']
    df_daily['vol'] = np.log(df_daily['close'] / df_daily['close'].shift(1)).rolling(VOL_PERIOD).std(ddof=1)
    
    ib_data = df_rth[df_rth['min_in_sess'] < 60].groupby('date').agg({'high':'max','low':'min'}).rename(columns={'high':'IBH','low':'IBL'})
    df_daily = df_daily.merge(ib_data, left_index=True, right_index=True)
    df_daily['ext_up'] = (df_daily['high'] - df_daily['IBH']).clip(lower=0)
    df_daily['ext_dn'] = (df_daily['IBL'] - df_daily['low']).clip(lower=0)
    df_daily.dropna(inplace=True)
    
    if len(df_daily) < WARMUP_DAYS + 1: return
    
    # 3. Signals
    today = df_daily.iloc[-1]
    hist = df_daily.iloc[-WARMUP_DAYS-1:-1]
    
    # Corrected Bias calculation
    u_v = df_daily['up_move'].rolling(5).var().iloc[-1]
    d_v = df_daily['dn_move'].rolling(5).var().iloc[-1]
    cur_bias_val = -1 if d_v > u_v * 1.2 else (1 if u_v > d_v * 1.2 else 0)
    cv = float(today['vol'])
    
    if cur_bias_val == 0 or cv == 0: return
    
    scaled_up = hist['ext_up'].values * (cv / hist['vol'].values)
    scaled_dn = hist['ext_dn'].values * (cv / hist['vol'].values)
    p50u, p75u, p95u = np.percentile(scaled_up, [50, 75, 95])
    p50d, p75d, p95d = np.percentile(scaled_dn, [50, 75, 95])
    
    sym_info = mt5.symbol_info(EXEC_SYMBOL)
    tick_size = sym_info.trade_tick_size
    def rt(v): return round(v/tick_size)*tick_size
    
    ibh, ibl = float(today['IBH']), float(today['IBL'])
    if cur_bias_val == -1:
        signal = {'type': mt5.ORDER_TYPE_SELL_LIMIT, 'ent': rt(ibh+p75u), 'tp': rt(ibh+p50u), 'sl': rt(ibh+p95u), 'dir': 'Short'}
    else:
        signal = {'type': mt5.ORDER_TYPE_BUY_LIMIT, 'ent': rt(ibl-p75d), 'tp': rt(ibl-p50d), 'sl': rt(ibl-p95d), 'dir': 'Long'}
    
    last_bias_e2 = signal['dir']
    
    # 4. Execution
    if is_desk_flat():
        risk_pts = abs(signal['ent'] - signal['sl'])
        total_qty = max(sym_info.volume_min, np.floor(RISK_BUCKET_USD / (risk_pts * MNQ_POINT_VALUE) / sym_info.volume_step) * sym_info.volume_step)
        
        # Split Order
        qty_h2 = np.floor((total_qty/2) / sym_info.volume_step) * sym_info.volume_step
        qty_h1 = round(total_qty - qty_h2, 2)
        
        for q, tp, m, c in [(qty_h1, signal['tp'], MAGIC_E2_SCALE_OUT, "E2_ScaleOut"), (qty_h2, signal['ent']+(500 if signal['dir']=='Long' else -500), MAGIC_E2_RUNNER, "E2_Runner")]:
            if q <= 0: continue
            req = {
                "action": mt5.TRADE_ACTION_PENDING, "symbol": EXEC_SYMBOL, "volume": float(q), "type": signal['type'],
                "price": float(signal['ent']), "sl": float(signal['sl']), "tp": float(tp), "magic": m, "comment": c,
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_RETURN
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                send_trade_alert(f"Engine 2 Entry: {signal['dir']} {q} contracts at {signal['ent']}")
    
    daily_levels_generated_e2 = True

# ═════════════════════════════════════════════════════════════════════════════
# 6. ENGINE 3: HURST STRUCTURAL
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_3_logic():
    global last_hurst_value_e3
    
    rates = mt5.copy_rates_from_pos(EXEC_SYMBOL, mt5.TIMEFRAME_M15, 0, 400)
    if rates is None or len(rates) < 250: return
    
    df = pd.DataFrame(rates)
    closes = df['close'].values
    h_val = calculate_hurst(closes[-HURST_WINDOW-1:-1])
    last_hurst_value_e3 = h_val
    
    sma = np.mean(closes[-SMA_PERIOD-1:-1])
    std = np.std(closes[-SMA_PERIOD-1:-1])
    lower_band = sma - (1.5 * std)
    cur_p = closes[-1]
    
    if is_desk_flat():
        if h_val < 0.48 and cur_p < lower_band:
            req = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": EXEC_SYMBOL, "volume": 2.0, "type": mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(EXEC_SYMBOL).ask, "magic": MAGIC_E3, "comment": "E3_Hurst_Long",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC
            }
            res = mt5.order_send(req)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                send_trade_alert(f"Engine 3 Entry: Long 2 MNQ at {cur_p} (H={h_val:.4f})")

# ═════════════════════════════════════════════════════════════════════════════
# 7. MONITOR & EXITS
# ═════════════════════════════════════════════════════════════════════════════

def monitor_active_trades():
    positions = mt5.positions_get(symbol=EXEC_SYMBOL)
    if not positions: return
    
    now = datetime.now(NY_TZ)
    cur_p = mt5.symbol_info_tick(EXEC_SYMBOL).last
    
    for pos in positions:
        # Time Stop (15:45 EST)
        if now.hour == 15 and now.minute >= 45:
            flatten_all("Time Stop")
            break
            
        # Engine 3 Exits
        if pos.magic == MAGIC_E3:
            if pos.profit <= HARD_STOP_E3_USD:
                close_pos(pos, "E3 Circuit Breaker")
            # SMA Target check
            if cur_p >= pos.tp: # Fallback if TP is set, but logic uses SMA
                pass 
            # We recalculate SMA for exit target monitoring
            rates = mt5.copy_rates_from_pos(EXEC_SYMBOL, mt5.TIMEFRAME_M15, 0, SMA_PERIOD + 1)
            if rates is not None:
                sma = np.mean([r[4] for r in rates[:-1]]) # Mean of closed bars
                if cur_p >= sma: close_pos(pos, "E3 SMA Target")
                if last_hurst_value_e3 > 0.55: close_pos(pos, "E3 Regime Change")

        # Engine 2 Runner Logic (Breakeven Shift)
        if pos.magic == MAGIC_E2_RUNNER:
            scale_out_alive = any([p.magic == MAGIC_E2_SCALE_OUT for p in positions])
            if not scale_out_alive:
                # If Runner is alive but ScaleOut is gone, it hit TP (or SL, but then Runner is gone too)
                if abs(pos.sl - pos.price_open) > 0.5:
                    req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "symbol": EXEC_SYMBOL, "sl": pos.price_open, "tp": pos.tp}
                    mt5.order_send(req)
                    send_trade_alert(f"Engine 2: Runner shifted to Breakeven at {pos.price_open}")

def close_pos(pos, reason):
    tick = mt5.symbol_info_tick(EXEC_SYMBOL)
    t = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    p = tick.bid if t == mt5.ORDER_TYPE_SELL else tick.ask
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": EXEC_SYMBOL, "volume": pos.volume, "type": t, "position": pos.ticket, "price": p, "magic": pos.magic, "comment": reason, "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    res = mt5.order_send(req)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        send_trade_alert(f"Exit: {reason} | Ticket {pos.ticket}")

def flatten_all(reason):
    positions = mt5.positions_get(symbol=EXEC_SYMBOL)
    if positions:
        for p in positions:
            if p.magic in [MAGIC_E2_SCALE_OUT, MAGIC_E2_RUNNER, MAGIC_E3]:
                close_pos(p, reason)
    orders = mt5.orders_get(symbol=EXEC_SYMBOL)
    if orders:
        for o in orders:
            if o.magic in [MAGIC_E2_SCALE_OUT, MAGIC_E2_RUNNER, MAGIC_E3]:
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket})

# ═════════════════════════════════════════════════════════════════════════════
# 8. MASTER LOOP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global last_date, daily_levels_generated_e2
    print("Master Quant Desk starting...")
    if not connect_mt5(): return
    
    last_min = -1
    last_15min = -1
    
    while True:
        try:
            now = datetime.now(NY_TZ)
            
            # Date Reset
            if last_date != now.date():
                last_date = now.date()
                daily_levels_generated_e2 = False
                print(f"\n--- SESSION START: {last_date} ---")
            
            # Dashboard (15 min)
            if now.minute % 15 == 0 and now.minute != last_15min:
                # os.system('cls' if os.name == 'nt' else 'clear') # Disabling for debugging
                print("="*80)
                print(f"QUANT DESK DASHBOARD | {now.strftime('%Y-%m-%d %H:%M:%S')} EST")
                print("="*80)
                print(f"Positions:  {len(mt5.positions_get(symbol=EXEC_SYMBOL) or [])}")
                print(f"E2 Bias:    {last_bias_e2}")
                print(f"E3 Hurst:   {last_hurst_value_e3:.4f}")
                print(f"Daily PnL:  ${get_daily_pnl():.2f}")
                print("="*80)
                last_15min = now.minute

            # Weekday Only
            if now.weekday() < 5:
                # Engine 2 Check (1 min)
                if now.minute != last_min:
                    curr_time = now.hour * 100 + now.minute
                    if curr_time >= 1030 and curr_time < 1545 and not daily_levels_generated_e2:
                        try:
                            run_engine_2_logic()
                        except Exception as e:
                            print(f"Engine 2 Logic Error: {e}")
                    last_min = now.minute
                
                # Engine 3 Check (15 min)
                if now.minute % 15 == 0 and now.second < 2:
                    try:
                        run_engine_3_logic()
                    except Exception as e:
                        print(f"Engine 3 Logic Error: {e}")
                
                # Global Watchdog (1Hz)
                monitor_active_trades()
                
            time.sleep(1)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"System Error: {e}")
            traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    main()
    mt5.shutdown()
