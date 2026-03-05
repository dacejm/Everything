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
NUM_BARS = 500000 # ~2 years of 1m data
NY_TZ = pytz.timezone("America/New_York")

def fetch_historical_m1():
    if not mt5.initialize():
        print("MT5 Initialization failed.")
        return None
    
    # Ensure symbol is selected
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
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    # Align to NY Session
    df.index = df.index.tz_convert(NY_TZ)
    return df

# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (DAILY SESSION LEVEL)
# ═════════════════════════════════════════════════════════════════════════════
def prepare_daily_features(df_m1):
    print("Pre-calculating daily session features...")
    # Filter RTH (09:30 - 16:00)
    df_m1['time_str'] = df_m1.index.strftime('%H:%M')
    df_rth = df_m1[(df_m1['time_str'] >= '09:30') & (df_m1['time_str'] <= '15:59')].copy()
    df_rth['date'] = df_rth.index.date
    
    # Calculate IB (First 60 mins)
    # 09:30 is 570 mins from midnight.
    df_rth['min_in_sess'] = (df_rth.index.hour * 60 + df_rth.index.minute) - 570
    ib_data = df_rth[df_rth['min_in_sess'] < 60].groupby('date').agg({'high': 'max', 'low': 'min'}).rename(columns={'high':'IBH', 'low':'IBL'})
    
    # Daily Aggregates
    df_daily = df_rth.groupby('date').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    df_daily = df_daily.merge(ib_data, left_index=True, right_index=True)
    
    # Moves & Volatility
    df_daily['up_move'] = df_daily['high'] - df_daily['open']
    df_daily['dn_move'] = df_daily['open'] - df_daily['low']
    df_daily['ext_up'] = (df_daily['high'] - df_daily['IBH']).clip(lower=0)
    df_daily['ext_dn'] = (df_daily['IBL'] - df_daily['low']).clip(lower=0)
    
    df_daily['vol'] = np.log(df_daily['close'] / df_daily['close'].shift(1)).rolling(20).std(ddof=1)
    
    # Skewness Filter (5-day)
    u_var = df_daily['up_move'].rolling(5).var()
    d_var = df_daily['dn_move'].rolling(5).var()
    df_daily['bias'] = 0
    df_daily.loc[d_var > u_var * 1.2, 'bias'] = -1 # Neg Skew (Short Only)
    df_daily.loc[u_var > d_var * 1.2, 'bias'] = 1  # Pos Skew (Long Only)
    
    return df_daily.dropna(), df_rth

# ═════════════════════════════════════════════════════════════════════════════
# 3. THE SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
def run_simulation(df_daily, df_rth):
    print("Starting bar-by-bar simulation...")
    trades = []
    dates = df_daily.index.tolist()
    
    # Warmup lookback: start after 90 days of valid history
    for i in range(91, len(dates)):
        today_date = dates[i]
        hist = df_daily.iloc[i-90:i] # Previous 90 days
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
        
        # Fetch Intraday bars from 10:30 to 15:45
        day_bars = df_rth[(df_rth['date'] == today_date) & (df_rth['min_in_sess'] >= 60) & (df_rth['min_in_sess'] <= 375)]
        
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
                    # Check Stop Loss First (Conservative)
                    if bar['high'] >= stop_p:
                        trades.append({'Date': today_date, 'Dir': 'Short', 'Entry_T': entry_t, 'Entry': entry_p, 'Exit_T': t, 'Exit': stop_p, 'Reason': 'Stop', 'PnL': entry_p - stop_p})
                        break
                    # Check Target
                    if bar['low'] <= target_p:
                        trades.append({'Date': today_date, 'Dir': 'Short', 'Entry_T': entry_t, 'Entry': entry_p, 'Exit_T': t, 'Exit': target_p, 'Reason': 'Target', 'PnL': entry_p - target_p})
                        break
                    # Time Stop (15:45)
                    if bar['min_in_sess'] == 375:
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
                    if bar['min_in_sess'] == 375:
                        trades.append({'Date': today_date, 'Dir': 'Long', 'Entry_T': entry_t, 'Entry': entry_p, 'Exit_T': t, 'Exit': bar['close'], 'Reason': 'Time', 'PnL': bar['close'] - entry_p})
                        break
                        
    return pd.DataFrame(trades)

# ═════════════════════════════════════════════════════════════════════════════
# 4. EXECUTION & REPORTING
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df_m1 = fetch_historical_m1()
    if df_m1 is not None:
        df_daily, df_rth = prepare_daily_features(df_m1)
        df_trades = run_simulation(df_daily, df_rth)
        
        if not df_trades.empty:
            # Export
            df_trades.to_csv("quant_model_historical_trades.csv", index=False)
            
            # Summary
            pnl = df_trades['PnL']
            win_rate = len(df_trades[pnl > 0]) / len(df_trades) * 100
            cum_pnl = pnl.cumsum()
            max_dd = np.max(np.maximum.accumulate(cum_pnl) - cum_pnl)
            
            print("\n" + "="*50)
            print("QUANT MODEL HISTORICAL SIMULATION REPORT")
            print("="*50)
            print(f"Total Trades:   {len(df_trades)}")
            print(f"Win Rate:       {win_rate:.1f}%")
            print(f"Gross PnL:      {pnl.sum():.2f} pts")
            print(f"Avg PnL/Trade:  {pnl.mean():.2f} pts")
            print(f"Max Drawdown:   {max_dd:.2f} pts")
            print(f"Log exported to: quant_model_historical_trades.csv")
            print("="*50)
        else:
            print("No trades triggered in the historical period.")
