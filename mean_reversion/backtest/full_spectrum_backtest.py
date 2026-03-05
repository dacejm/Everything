import pandas as pd
import numpy as np
from numba import jit
from core.data_manager import DataManager
from config import DATA_FILES

from core.time_utils import assign_trading_sessions
from core.math_utils import _pine_percentile, _calc_retail_projections, _calc_quant_projections

import time
import os
import sys

# FIX 5.2: Single source of truth for all parameters
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BIAS_THRESHOLD

# ═════════════════════════════════════════════════════════════════════════════
# 1. CORE NUMBA COMPUTATION ENGINES
# ═════════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def execute_engine(highs, lows, closes, minutes, day_indices, 
                   entry_up, tp_up, sl_up, entry_dn, tp_dn, sl_dn, 
                   bias_vals, start_min, end_min, stop_min):
    n_days = len(day_indices) - 1
    pnl = []
    for d in range(n_days):
        idx_s, idx_e = day_indices[d], day_indices[d+1]
        cur_bias = bias_vals[idx_s]
        short_done, long_done = False, False
        for i in range(idx_s, idx_e):
            m = minutes[i]
            if not short_done and cur_bias == -1 and m >= start_min and m <= end_min:
                e, tp, sl = entry_up[i], tp_up[i], sl_up[i]
                if not np.isnan(e) and highs[i] >= e:
                    short_done = True
                    for j in range(i+1, idx_e):
                        if highs[j] >= sl: pnl.append(e - sl); break
                        if lows[j] <= tp: pnl.append(e - tp); break
                        if minutes[j] >= stop_min: pnl.append(e - closes[j]); break
            if not long_done and cur_bias == 1 and m >= start_min and m <= end_min:
                e, tp, sl = entry_dn[i], tp_dn[i], sl_dn[i]
                if not np.isnan(e) and lows[i] <= e:
                    long_done = True
                    for j in range(i+1, idx_e):
                        if lows[j] <= sl: pnl.append(sl - e); break
                        if highs[j] >= tp: pnl.append(tp - e); break
                        if minutes[j] >= stop_min: pnl.append(closes[j] - e); break
    return np.array(pnl)

# ═════════════════════════════════════════════════════════════════════════════
# 2. EVALUATION & METRIC KERNELS
# ═════════════════════════════════════════════════════════════════════════════

def calculate_full_metrics(p, engine_name, ticker):
    if len(p) == 0: return {"Engine": engine_name, "Ticker": ticker, "Trades": 0, "Win%": 0, "PnL": 0, "EV": 0, "PF": 0, "Sharpe": 0, "MaxDD": 0}
    cum_pnl = np.cumsum(p)
    max_dd = np.max(np.maximum.accumulate(cum_pnl) - cum_pnl)
    w = p[p > 0]; std = np.std(p)
    return {
        "Engine": engine_name, "Ticker": ticker, "Trades": len(p),
        "Win%": round(len(w)/len(p)*100, 1), "PnL": round(np.sum(p), 1),
        "EV": round(np.mean(p), 2), 
        "PF": round(np.sum(w)/np.abs(np.sum(p[p<0])), 2) if np.sum(p<0) != 0 else np.inf,
        "Sharpe": round((np.mean(p)/std) * np.sqrt(252*2), 2) if std != 0 else 0,
        "MaxDD": round(max_dd, 1)
    }

def run_mae_mfe_profiler(df):
    df['date'] = df.index.date
    df['upP75_lvl'] = df['Session_Open'] + df['upP75']
    df['dnP75_lvl'] = df['Session_Open'] - df['dnP75']
    df['up_breach'] = df['High'] >= df['upP75_lvl']
    df['dn_breach'] = df['Low'] <= df['dnP75_lvl']
    
    first_up = df[df['up_breach']].groupby('date').head(1).index
    first_dn = df[df['dn_breach']].groupby('date').head(1).index
    
    df['EOD_High'] = df.groupby('date')['High'].transform('max')
    df['EOD_Low']  = df.groupby('date')['Low'].transform('min')
    
    shorts = df.loc[first_up].copy()
    shorts['MAE'] = (shorts['EOD_High'] - shorts['upP75_lvl']).clip(lower=0)
    shorts['MFE'] = (shorts['upP75_lvl'] - shorts['EOD_Low']).clip(lower=0)
    
    longs = df.loc[first_dn].copy()
    longs['MAE'] = (longs['dnP75_lvl'] - longs['EOD_Low']).clip(lower=0)
    longs['MFE'] = (longs['EOD_High'] - longs['dnP75_lvl']).clip(lower=0)
    
    all_mae = pd.concat([shorts['MAE'], longs['MAE']]).dropna()
    all_mfe = pd.concat([shorts['MFE'], longs['MFE']]).dropna()
    
    return {
        "MAE_P50": round(np.percentile(all_mae, 50), 2) if len(all_mae) > 0 else 0,
        "MAE_P75": round(np.percentile(all_mae, 75), 2) if len(all_mae) > 0 else 0,
        "MFE_P50": round(np.percentile(all_mfe, 50), 2) if len(all_mfe) > 0 else 0,
        "MFE_P75": round(np.percentile(all_mfe, 75), 2) if len(all_mfe) > 0 else 0
    }

# ═════════════════════════════════════════════════════════════════════════════
# 3. ENGINE RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

def run_engines_on_ticker(df, ticker):
    print(f"Analyzing {ticker}...")
    df = assign_trading_sessions(df)
    df_daily = df.groupby('session_date').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_daily['date'] = df_daily.index
    df_daily['up_move'] = df_daily['High'] - df_daily['Open']
    df_daily['dn_move'] = df_daily['Open'] - df_daily['Low']
    
    uv, dv = df_daily['up_move'].rolling(5).var(), df_daily['dn_move'].rolling(5).var()
    df_daily['bias'] = 0
    # FIX 3.1: Widen bias dead zone
    df_daily.loc[dv > uv * BIAS_THRESHOLD, 'bias'] = -1; df_daily.loc[uv > dv * BIAS_THRESHOLD, 'bias'] = 1
    df_daily['vol'] = np.log(df_daily['Close'] / df_daily['Close'].shift(1)).rolling(20).std(ddof=1).fillna(0)
    
    p_list = np.array([50.0, 75.0, 90.0, 95.0])
    q_proj = _calc_quant_projections(df_daily['up_move'].values, df_daily['dn_move'].values, df_daily['vol'].values, np.ones(len(df_daily), dtype=np.bool_), 90, 30, p_list)
    cols = ['upP50', 'upP75', 'upP90', 'upP95', 'dnP50', 'dnP75', 'dnP90', 'dnP95']
    df_proj = pd.DataFrame(q_proj, index=df_daily.index, columns=cols)
    df_proj[['bias', 'vol', 'Session_Open', 'date']] = df_daily[['bias', 'vol', 'Open', 'date']]
    
    df['date'] = df.index.date
    df_m = df.reset_index().merge(df_proj, on='date', how='left').set_index('DateTime')
    df_m = df_m.dropna(subset=['upP50'])
    df_m['minutes'] = df_m.index.hour * 60 + df_m.index.minute
    d_idx = np.where(df_m.index.date != pd.Series(df_m.index.date).shift(1).values)[0]
    d_idx = np.append(d_idx, len(df_m))
    
    excursion_results = run_mae_mfe_profiler(df_m)
    h, l, c, m = df_m['High'].values, df_m['Low'].values, df_m['Close'].values, df_m['minutes'].values
    b = df_m['bias'].values
    
    # Engine 1: Daily Fade
    e_up, tp_up, sl_up = (df_m['Session_Open'] + df_m['upP75']).values, (df_m['Session_Open'] + df_m['upP50']).values, (df_m['Session_Open'] + df_m['upP95']).values
    e_dn, tp_dn, sl_dn = (df_m['Session_Open'] - df_m['dnP75']).values, (df_m['Session_Open'] - df_m['dnP50']).values, (df_m['Session_Open'] - df_m['dnP95']).values
    pnl_e1 = execute_engine(h, l, c, m, d_idx, e_up, tp_up, sl_up, e_dn, tp_dn, sl_dn, b, 0, 1440, 945)
    
    res = []
    res.append(calculate_full_metrics(pnl_e1, "Daily Fade", ticker))
    for k, v in excursion_results.items(): res[0][k] = v
    return res

if __name__ == "__main__":
    ticker_files = DATA_FILES
    all_res = []
    for ticker, file in ticker_files.items():
        if os.path.exists(file):
            df = pd.read_csv(file, header=None, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
            df.set_index('DateTime', inplace=True)
            all_res.extend(run_engines_on_ticker(df, ticker))
            
    master_df = pd.DataFrame(all_res)
    print("\n" + "="*125 + "\nFULL QUANTITATIVE PORTFOLIO PROFILE (EXCURSIONS + PERFORMANCE)\n" + "="*125)
    print(master_df.to_string(index=False))
