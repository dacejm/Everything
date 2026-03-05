import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import os

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & DATA FETCHING
# ═════════════════════════════════════════════════════════════════════════════
DATA_SYMBOL = "@ENQ"
NUM_BARS = 500000 
NY_TZ = pytz.timezone("America/New_York")
WARMUP_DAYS = 45 # Reduced from 90 to increase valid sample size

def fetch_and_normalize_data():
    if not mt5.initialize():
        print("MT5 Initialization failed.")
        return None
    
    if not mt5.symbol_select(DATA_SYMBOL, True):
        print(f"Failed to select {DATA_SYMBOL}")
        mt5.shutdown()
        return None

    print(f"Fetching {NUM_BARS} bars for {DATA_SYMBOL}...")
    rates = mt5.copy_rates_from_pos(DATA_SYMBOL, mt5.TIMEFRAME_M1, 0, NUM_BARS)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        print("No data retrieved.")
        return None
        
    df = pd.DataFrame(rates)
    # MT5 time is seconds since epoch (UTC)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # CRITICAL FIX: Timezone Normalization
    # Localize to UTC then convert to NY (pandas handles DST automatically)
    df.index = df.index.tz_localize('UTC').tz_convert(NY_TZ)
    
    print(f"Data Normalized. Range: {df.index[0]} to {df.index[-1]}")
    return df

# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (STRICT NY SESSION SLICING)
# ═════════════════════════════════════════════════════════════════════════════
def prepare_daily_features(df_m1):
    print("Pre-calculating daily session features...")
    df_m1['date'] = df_m1.index.date
    unique_dates = np.unique(df_m1['date'])
    
    daily_records = []
    debug_count = 0
    
    for d in unique_dates:
        # 1. Slice for the full session (09:30 - 16:00 NY)
        day_data = df_m1[df_m1['date'] == d]
        session_data = day_data.between_time('09:30', '15:59')
        
        if session_data.empty: continue
        
        # 2. Extract IB (09:30 - 10:29:59)
        ib_data = day_data.between_time('09:30', '10:29')
        if ib_data.empty: continue
        
        ibh = ib_data['high'].max()
        ibl = ib_data['low'].min()
        
        if debug_count < 5:
            print(f"DEBUG: {d} | NY Open: {session_data.iloc[0].name} | IBH: {ibh:.2f} | IBL: {ibl:.2f}")
            debug_count += 1
            
        daily_records.append({
            'date': d,
            'open': session_data['open'].iloc[0],
            'high': session_data['high'].max(),
            'low': session_data['low'].min(),
            'close': session_data['close'].iloc[-1],
            'IBH': ibh,
            'IBL': ibl
        })
        
    df_daily = pd.DataFrame(daily_records).set_index('date')
    
    # 3. Calculate Model Math
    df_daily['up_move'] = df_daily['high'] - df_daily['open']
    df_daily['dn_move'] = df_daily['open'] - df_daily['low']
    df_daily['ext_up'] = (df_daily['high'] - df_daily['IBH']).clip(lower=0)
    df_daily['ext_dn'] = (df_daily['IBL'] - df_daily['low']).clip(lower=0)
    
    # Log-Return Vol (20-day)
    df_daily['vol'] = np.log(df_daily['close'] / df_daily['close'].shift(1)).rolling(20).std(ddof=1)
    
    # Skewness Filter (5-day)
    u_var = df_daily['up_move'].rolling(5).var()
    d_var = df_daily['dn_move'].rolling(5).var()
    df_daily['bias'] = 0
    df_daily.loc[d_var > u_var * 1.2, 'bias'] = -1 # Neg Skew (Short Only)
    df_daily.loc[u_var > d_var * 1.2, 'bias'] = 1  # Pos Skew (Long Only)
    
    return df_daily.dropna()

# ═════════════════════════════════════════════════════════════════════════════
# 3. THE SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
def run_simulation(df_daily, df_m1):
    print("Starting forward-looking simulation...")
    trades = []
    dates = df_daily.index.tolist()
    
    for i in range(WARMUP_DAYS + 1, len(dates)):
        today_date = dates[i]
        hist = df_daily.iloc[i-WARMUP_DAYS:i] # Previous days for scaling
        today = df_daily.iloc[i]
        
        cur_vol = today['vol']
        cur_bias = today['bias']
        if cur_bias == 0 or cur_vol == 0: continue
        
        # Scaling historical extensions to today's volatility
        scaled_up = hist['ext_up'].values * (cur_vol / hist['vol'].values)
        scaled_dn = hist['ext_dn'].values * (cur_vol / hist['vol'].values)
        
        # Project Levels
        p50_up, p75_up, p95_up = np.percentile(scaled_up, [50, 75, 95])
        p50_dn, p75_dn, p95_dn = np.percentile(scaled_dn, [50, 75, 95])
        
        # Slice intraday data for simulation (10:30 - 15:45 NY)
        day_bars = df_m1[df_m1['date'] == today_date].between_time('10:30', '15:45')
        
        in_trade = False
        entry_t = None
        
        if cur_bias == -1: # Short Bias (Fade Upside)
            entry_p = today['IBH'] + p75_up
            target_p = today['IBH'] + p50_up
            stop_p = today['IBH'] + p95_up
            
            for t, bar in day_bars.iterrows():
                if not in_trade:
                    if bar['high'] >= entry_p:
                        in_trade = True
                        entry_t = t
                else:
                    if bar['high'] >= stop_p:
                        trades.append({'Date': today_date, 'Dir': 'Short', 'Entry_T': entry_t, 'Entry': entry_p, 'Exit_T': t, 'Exit': stop_p, 'Reason': 'Stop', 'PnL': entry_p - stop_p})
                        break
                    if bar['low'] <= target_p:
                        trades.append({'Date': today_date, 'Dir': 'Short', 'Entry_T': entry_t, 'Entry': entry_p, 'Exit_T': t, 'Exit': target_p, 'Reason': 'Target', 'PnL': entry_p - target_p})
                        break
                    if t.hour == 15 and t.minute == 45:
                        trades.append({'Date': today_date, 'Dir': 'Short', 'Entry_T': entry_t, 'Entry': entry_p, 'Exit_T': t, 'Exit': bar['close'], 'Reason': 'Time', 'PnL': entry_p - bar['close']})
                        break

        elif cur_bias == 1: # Long Bias (Fade Downside)
            entry_p = today['IBL'] - p75_dn
            target_p = today['IBL'] - p50_dn
            stop_p = today['IBL'] - p95_dn
            
            for t, bar in day_bars.iterrows():
                if not in_trade:
                    if bar['low'] <= entry_p:
                        in_trade = True
                        entry_t = t
                else:
                    if bar['low'] <= stop_p:
                        trades.append({'Date': today_date, 'Dir': 'Long', 'Entry_T': entry_t, 'Entry': entry_p, 'Exit_T': t, 'Exit': stop_p, 'Reason': 'Stop', 'PnL': stop_p - entry_p})
                        break
                    if bar['high'] >= target_p:
                        trades.append({'Date': today_date, 'Dir': 'Long', 'Entry_T': entry_t, 'Entry': entry_p, 'Exit_T': t, 'Exit': target_p, 'Reason': 'Target', 'PnL': target_p - entry_p})
                        break
                    if t.hour == 15 and t.minute == 45:
                        trades.append({'Date': today_date, 'Dir': 'Long', 'Entry_T': entry_t, 'Entry': entry_p, 'Exit_T': t, 'Exit': bar['close'], 'Reason': 'Time', 'PnL': bar['close'] - entry_p})
                        break
                        
    return pd.DataFrame(trades)

# ═════════════════════════════════════════════════════════════════════════════
# 4. EXECUTION & REPORTING
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df_m1 = fetch_and_normalize_data()
    if df_m1 is not None:
        df_daily = prepare_daily_features(df_m1)
        df_trades = run_simulation(df_daily, df_m1)
        
        if not df_trades.empty:
            df_trades.to_csv("quant_model_historical_trades_fixed.csv", index=False)
            pnl = df_trades['PnL']
            win_rate = len(df_trades[pnl > 0]) / len(df_trades) * 100
            
            print("\n" + "="*50)
            print("FIXED QUANT MODEL HISTORICAL REPORT")
            print("="*50)
            print(f"Total Trades:   {len(df_trades)}")
            print(f"Win Rate:       {win_rate:.1f}%")
            print(f"Gross PnL:      {pnl.sum():.2f} pts")
            print(f"Avg PnL/Trade:  {pnl.mean():.2f} pts")
            print(f"Max DD (pts):   {np.max(np.maximum.accumulate(pnl.cumsum()) - pnl.cumsum()):.2f}")
            print(f"CSV saved:      quant_model_historical_trades_fixed.csv")
            print("="*50)
        else:
            print("Simulation complete. No trades triggered with current parameters.")
