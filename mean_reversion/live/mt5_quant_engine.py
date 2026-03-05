import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pytz
import os
import sys
import traceback

# Core imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BIAS_THRESHOLD, VOL_PERIOD, SKEW_PERIOD, WARMUP_DAYS, MNQ_POINT_VALUE
from core.mt5_base import BaseMT5Engine

# ═════════════════════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION & DUAL-SYMBOL SETUP
# ═════════════════════════════════════════════════════════════════════════════

DATA_SYMBOL = "@ENQ"       
EXEC_SYMBOL = "MNQH26"     

MAGIC_SCALE_OUT = 8884441
MAGIC_RUNNER = 8884442

NY_TIMEZONE = pytz.timezone("America/New_York")

# ═════════════════════════════════════════════════════════════════════════════
# ENGINE CLASS
# ═════════════════════════════════════════════════════════════════════════════

class QuantEngine(BaseMT5Engine):
    def __init__(self, data_symbol=DATA_SYMBOL, exec_symbol=EXEC_SYMBOL, account_size=50000.0, risk_pct=0.01):
        super().__init__(symbols=[data_symbol, exec_symbol], magic_numbers=[MAGIC_SCALE_OUT, MAGIC_RUNNER], account_size=account_size, log_file="logs/quant_engine.log")
        self.data_symbol = data_symbol
        self.exec_symbol = exec_symbol
        self.risk_amount = account_size * risk_pct
        self.daily_levels_generated = False
        self.current_bias = None
        self.p75_level = 0.0
        self.p50_level = 0.0
        self.p95_level = 0.0
        self.last_date = None

    def has_active_trades_today(self):
        """Checks for open positions or resting orders on the execution symbol."""
        orders = mt5.orders_get(symbol=self.exec_symbol)
        if orders is not None:
            for order in orders:
                if order.magic in self.magic_numbers:
                    self.logger.info(f"Resting order already exists. Skipping entry.")
                    return True
                    
        positions = mt5.positions_get(symbol=self.exec_symbol)
        if positions is not None:
            for pos in positions:
                if pos.magic in self.magic_numbers:
                    self.logger.info(f"Open position already exists. Skipping entry.")
                    return True
                    
        return False

    def fetch_data(self):
        """Fetches historical 1-minute data from the continuous reference contract."""
        rates = mt5.copy_rates_from_pos(self.data_symbol, mt5.TIMEFRAME_M1, 0, 100000)
        if rates is None or len(rates) == 0:
            self.logger.error(f"Data Fetch Failed for {self.data_symbol}. Code: {mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df.index = df.index.tz_convert(NY_TIMEZONE)
        return df

    def calculate_quant_signals(self, df):
        """Calculates Engine 2 logic on the continuous reference data."""
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
        
        # FIX 3.1: Widen bias dead zone from 1.2x to BIAS_THRESHOLD
        df_daily.loc[dn_var > up_var * BIAS_THRESHOLD, 'bias'] = -1
        df_daily.loc[up_var > dn_var * BIAS_THRESHOLD, 'bias'] = 1 
        
        df_daily['vol'] = np.log(df_daily['close'] / df_daily['close'].shift(1)).rolling(VOL_PERIOD).std(ddof=1)
        
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
        
        self.logger.debug(f"DEBUG: IB High is type {type(ib_high)} value {ib_high}")
        self.logger.debug(f"DEBUG: Bias is {cur_bias} (type {type(cur_bias)})")
        
        if cur_vol == 0: 
            return None
            
        history = df_daily.iloc[-WARMUP_DAYS-1:-1]
        
        # FIX 2.2: Protected division matching backtest kernel
        hist_vol = history['vol'].values
        safe_vol = np.where(hist_vol > 0, hist_vol, cur_vol)
        scaled_up = history['ext_up'].values * (cur_vol / safe_vol)
        scaled_dn = history['ext_dn'].values * (cur_vol / safe_vol)
                
        p50_up, p75_up, p95_up = np.percentile(scaled_up, [50, 75, 95])
        p50_dn, p75_dn, p95_dn = np.percentile(scaled_dn, [50, 75, 95])
        
        p50_up, p75_up, p95_up = float(p50_up), float(p75_up), float(p95_up)
        p50_dn, p75_dn, p95_dn = float(p50_dn), float(p75_dn), float(p95_dn)
        
        tick_size = mt5.symbol_info(self.exec_symbol).trade_tick_size
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
            self.current_bias = signal['dir']
            self.p75_level = signal['entry']
            self.p50_level = signal['tp']
            self.p95_level = signal['sl']
            
        return signal

    def route_split_orders(self, signal):
        """Calculates risk, splits position into halves, and places two limit orders with volume normalization."""
        risk_pts = abs(signal['entry'] - signal['sl'])
        if risk_pts == 0: return False
        
        sym_info = mt5.symbol_info(self.exec_symbol)
        if sym_info is None:
            self.logger.error(f"Could not fetch symbol info for {self.exec_symbol}")
            return False
            
        vol_step = sym_info.volume_step
        min_vol = sym_info.volume_min
        
        total_qty_raw = self.risk_amount / (risk_pts * MNQ_POINT_VALUE)
        total_qty = max(min_vol, np.floor(total_qty_raw / vol_step) * vol_step)
        total_qty = round(total_qty, 2)
        
        if total_qty <= min_vol:
            qty_h1 = total_qty
            qty_h2 = 0.0
            self.logger.debug(f"[DEBUG] Total Qty {total_qty} <= Min Vol {min_vol}. No runner logic possible.")
        else:
            qty_h2_raw = (total_qty / 2.0)
            qty_h2 = np.floor(qty_h2_raw / vol_step) * vol_step
            qty_h1 = total_qty - qty_h2
            
            if qty_h2 < min_vol and qty_h2 > 0:
                qty_h1 = total_qty
                qty_h2 = 0.0
                self.logger.debug(f"[DEBUG] Split Runner {qty_h2} < Min Vol {min_vol}. Reverting to full Scale-Out.")

        qty_h1 = round(qty_h1, 2)
        qty_h2 = round(qty_h2, 2)
        
        self.logger.debug(f"[DEBUG] Adjusted Scale-Out Qty: {qty_h1}")
        self.logger.debug(f"[DEBUG] Adjusted Runner Qty: {qty_h2}")
        self.logger.info(f"Signal: {signal['dir']} @ {signal['entry']} | Risk: {risk_pts:.2f} pts | Total Qty: {total_qty}")
        
        def send_order(qty, tp, magic, comment):
            if qty <= 0: return
            
            self.logger.debug(f"[DEBUG] Sending {comment} with type_time: ORDER_TIME_GTC")
            
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": self.exec_symbol,
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
            
            if res.retcode == 10022 and "expiration" in res.comment.lower():
                self.logger.debug(f" -> [DEBUG] GTC Unsupported. Retrying with ORDER_TIME_DAY...")
                request["type_time"] = mt5.ORDER_TIME_DAY
                res = mt5.order_send(request)

            if res.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f" -> ERROR: {comment} Rejected. Code: {res.retcode} | {res.comment}")
            else:
                self.logger.info(f" -> SUCCESS: {comment} Placed. Qty: {qty} | Ticket: {res.order}")

        # Order 1: Scale-Out (Has P50 Target)
        send_order(qty_h1, signal['tp'], MAGIC_SCALE_OUT, "H1_ScaleOut")
        
        # FIX 4.2: Set runner TP to 3x the risk distance, or use no TP (0.0)
        # Order 2: Runner
        if qty_h2 > 0:
            risk_distance = abs(signal['entry'] - signal['sl'])
            if signal['dir'] == 'SHORT':
                runner_tp = signal['entry'] - (risk_distance * 3.0)
            else:
                runner_tp = signal['entry'] + (risk_distance * 3.0)
            
            tick_size = mt5.symbol_info(self.exec_symbol).trade_tick_size
            def round_tick(val): return round(val / tick_size) * tick_size
            runner_tp = round_tick(runner_tp)
            
            send_order(qty_h2, runner_tp, MAGIC_RUNNER, "H2_Runner")
        
        return True

    def watchdog_loop(self, entry_price):
        """
        Monitors active trades.
        1. If Scale-Out hits target, shifts Runner Stop Loss to Breakeven.
        2. Flattens all positions at 15:45 EST.
        """
        self.logger.info("Watchdog active. Polling broker at 10Hz (100ms).")
        breakeven_shifted = False
        
        # FIX 4.3: Watchdog heartbeat
        last_heartbeat = time.time()
        
        while True:
            now_ny = datetime.now(NY_TIMEZONE)
            
            if time.time() - last_heartbeat > 60:
                positions = mt5.positions_get(symbol=self.exec_symbol)
                pos_count = len([p for p in (positions or []) if p.magic in self.magic_numbers])
                self.logger.info(f"HEARTBEAT: {pos_count} positions active | BE shifted: {breakeven_shifted}")
                last_heartbeat = time.time()
            
            # 1. TIME STOP
            if now_ny.hour == 15 and now_ny.minute >= 45:
                self.flatten_and_cancel(self.exec_symbol, self.magic_numbers, "Time Stop Flatten")
                self.logger.info("Session closed. Watchdog disengaging.")
                break
                
            # 2. BREAKEVEN SHIFT LOGIC
            if not breakeven_shifted:
                positions = mt5.positions_get(symbol=self.exec_symbol)
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
                                self.logger.info("SCALE-OUT TARGET HIT! Shifting Runner to Breakeven...")
                                request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "position": runner_pos.ticket,
                                    "symbol": self.exec_symbol,
                                    "sl": float(entry_price),
                                    "tp": float(runner_pos.tp)
                                }
                                res = mt5.order_send(request)
                                if res.retcode == mt5.TRADE_RETCODE_DONE:
                                    self.logger.info(f"SUCCESS: Runner Stop Loss updated to {entry_price}")
                                    breakeven_shifted = True
                                else:
                                    self.logger.warning(f"RETRYING: Runner SL modification failed (Code: {res.retcode})")
                                
            time.sleep(0.1) # 10Hz polling
            if not self.check_connection():
                time.sleep(1)

    def engine_loop(self):
        """Main daily loop."""
        now_ny = datetime.now(NY_TIMEZONE)
        
        # Robust New Day Detection
        if self.last_date != now_ny.date():
            self.daily_levels_generated = False
            self.last_date = now_ny.date()
            self.logger.info(f"--- New Trading Day: {self.last_date} ---")

        if now_ny.weekday() < 5:
            # TRIGGER: Signal Generation (10:30 AM EST to 16:00 PM EST Catch-Up)
            current_time_val = now_ny.hour * 100 + now_ny.minute
            
            if current_time_val >= 1030 and current_time_val < 1600 and not self.daily_levels_generated:
                if current_time_val > 1031:
                    self.logger.info("Catch-Up Logic Triggered: Fetching IB...")
                else:
                    self.logger.info("10:30 AM EST. IB period complete. Generating signals...")
                
                if not self.has_active_trades_today():
                    df = self.fetch_data()
                    if df is not None:
                        sig = self.calculate_quant_signals(df)
                        if sig:
                            self.logger.info(f"SIGNAL GENERATED:")
                            self.logger.info(f" -> Bias: {sig['dir']}")
                            self.logger.info(f" -> Entry: {sig['entry']}")
                            self.logger.info(f" -> Target (P50): {sig['tp']}")
                            self.logger.info(f" -> Stop (P95): {sig['sl']}")
                            
                            if self.route_split_orders(sig):
                                self.watchdog_loop(sig['entry'])
                        else:
                            self.logger.info("Neutral Skew. Standing aside.")
                
                self.daily_levels_generated = True
                time.sleep(60)
            
            # TRIGGER 2: Kill Switch (15:45 PM EST)
            elif current_time_val >= 1545 and now_ny.hour < 16:
                self.flatten_and_cancel(self.exec_symbol, self.magic_numbers, "End of Day Flatten")
                time.sleep(60)

    def on_shutdown(self):
        self.flatten_and_cancel(self.exec_symbol, self.magic_numbers, "Manual Shutdown Flatten")

if __name__ == "__main__":
    engine = QuantEngine()
    engine.safe_run()
    mt5.shutdown()
