import pandas as pd
import numpy as np
from numba import jit
from core.data_manager import DataManager
from config import DATA_FILES

from core.math_utils import _pine_percentile, _calc_retail_projections, _calc_quant_projections

import time
import os

# ═════════════════════════════════════════════════════════════════════════════
# 1. CORE NUMBA COMPUTATION ENGINES
# ═════════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def execute_fade_strategy(highs, lows, closes, minutes, day_indices, 
                          entry_lvls, tp_lvls, sl_lvls, 
                          is_short, start_min, end_min, stop_min):
    n_days = len(day_indices) - 1
    pnl_results = []
    
    for d in range(n_days):
        start_idx = day_indices[d]
        end_idx = day_indices[d+1]
        
        entry_p = entry_lvls[start_idx]
        tp_p = tp_lvls[start_idx]
        sl_p = sl_lvls[start_idx]
        
        if np.isnan(entry_p): continue
        
        in_trade = False
        
        for i in range(start_idx, end_idx):
            m = minutes[i]
            
            # --- ENTRY ---
            if not in_trade and m >= start_min and m <= end_min:
                triggered = (highs[i] >= entry_p) if is_short else (lows[i] <= entry_p)
                if triggered:
                    in_trade = True
                    continue 
            
            # --- EXIT ---
            if in_trade:
                # Stop Loss First
                sl_hit = (highs[i] >= sl_p) if is_short else (lows[i] <= sl_p)
                if sl_hit:
                    pnl_results.append(entry_p - sl_p if is_short else sl_p - entry_p)
                    in_trade = False
                    break 
                
                # Take Profit
                tp_hit = (lows[i] <= tp_p) if is_short else (highs[i] >= tp_p)
                if tp_hit:
                    pnl_results.append(entry_p - tp_p if is_short else tp_p - entry_p)
                    in_trade = False
                    break 
                
                # Time Stop
                if m >= stop_min:
                    pnl_results.append(entry_p - closes[i] if is_short else closes[i] - entry_p)
                    in_trade = False
                    break

    return np.array(pnl_results)

# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA PROCESSING AND BACKTEST EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def calculate_metrics(pnl_array):
    if len(pnl_array) == 0: 
        return {k: 0 for k in ["Total Trades", "Win Rate %", "Total PnL (pts)", "EV (pts)", "Profit Factor", "Sharpe Ratio", "Max DD (pts)"]}
    
    wins = pnl_array[pnl_array > 0]
    losses = pnl_array[pnl_array < 0]
    
    total_pnl = np.sum(pnl_array)
    win_rate = len(wins) / len(pnl_array) * 100
    
    std = np.std(pnl_array)
    sharpe = (np.mean(pnl_array) / std) * np.sqrt(252 * 2) if std != 0 else 0
    
    pf = np.sum(wins) / np.abs(np.sum(losses)) if np.sum(losses) != 0 else np.inf
    
    cum_pnl = np.cumsum(pnl_array)
    max_dd = np.max(np.maximum.accumulate(cum_pnl) - cum_pnl)
    
    return {
        "Total Trades": len(pnl_array),
        "Win Rate %": round(win_rate, 2),
        "Total PnL (pts)": round(total_pnl, 1),
        "EV (pts)": round(np.mean(pnl_array), 2),
        "Profit Factor": round(pf, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max DD (pts)": round(max_dd, 1)
    }

def run_backtest(file_path):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, header=None, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
    df.set_index('DateTime', inplace=True)
    
    print("Generating Quant Model projections...")
    df_daily = df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    df_daily['up_move'] = df_daily['High'] - df_daily['Open']
    df_daily['dn_move'] = df_daily['Open'] - df_daily['Low']
    df_daily['daily_range'] = df_daily['High'] - df_daily['Low']
    
    avg_vol = df_daily['Volume'].rolling(20).mean()
    is_valid = (df_daily['daily_range'] >= 50.0) & (df_daily['Volume'] / avg_vol >= 0.5)
    
    log_ret = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
    vols = log_ret.rolling(20).std(ddof=1).fillna(0.0).values
    
    q_proj = _calc_quant_projections(
        df_daily['up_move'].values, df_daily['dn_move'].values, 
        vols, is_valid.values, 90, 30
    )
    
    cols = ['upP50', 'upP75', 'upP90', 'upP95', 'dnP50', 'dnP75', 'dnP90', 'dnP95']
    df_proj = pd.DataFrame(q_proj, index=df_daily.index, columns=cols)
    df_proj['Session_Open'] = df_daily['Open']
    
    # Map back to 1m
    df['date'] = df.index.date
    df_proj['date'] = df_proj.index.date
    df = df.reset_index().merge(df_proj, on='date', how='left').set_index('DateTime')
    df = df.dropna(subset=['upP50'])
    
    df['minutes'] = df.index.hour * 60 + df.index.minute
    df['day_change'] = (df.index.date != pd.Series(df.index.date).shift(1).values)
    day_indices = np.where(df['day_change'])[0]
    day_indices = np.append(day_indices, len(df))
    
    h, l, c, m = df['High'].values, df['Low'].values, df['Close'].values, df['minutes'].values
    
    # Strategy levels
    s_entry = (df['Session_Open'] + df['upP75']).values
    s_tp = (df['Session_Open'] + df['upP50']).values
    s_sl = (df['Session_Open'] + df['upP95']).values
    
    l_entry = (df['Session_Open'] - df['dnP75']).values
    l_tp = (df['Session_Open'] - df['dnP50']).values
    l_sl = (df['Session_Open'] - df['dnP95']).values
    
    final_results = {}
    
    # Time stops: Market closes at 16:00, so we exit at 15:59 (959 mins)
    for name, start_m, end_m in [("Run 1: Unfiltered", 0, 1440), ("Run 2: TOD Filtered", 600, 915)]:
        print(f"Testing {name}...")
        pnl_s = execute_fade_strategy(h, l, c, m, day_indices, s_entry, s_tp, s_sl, True, start_m, end_m, 959)
        pnl_l = execute_fade_strategy(h, l, c, m, day_indices, l_entry, l_tp, l_sl, False, start_m, end_m, 959)
        all_pnl = np.concatenate([pnl_s, pnl_l])
        final_results[name] = calculate_metrics(all_pnl)
        
    return pd.DataFrame(final_results).T

if __name__ == "__main__":
    file_path = DATA_FILES["NQ"]
    if os.path.exists(file_path):
        results = run_backtest(file_path)
        print("\n" + "="*80)
        print("STRATEGY PERFORMANCE COMPARISON: RAW EDGE VS. TOD FILTER")
        print("="*80)
        print(results.to_string())
    else:
        print("Data file not found.")
