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
def execute_engine(highs, lows, closes, minutes, day_indices, 
                   entry_up, tp_up, sl_up, entry_dn, tp_dn, sl_dn, 
                   bias_vals, start_min, end_min, stop_min):
    n_days = len(day_indices) - 1
    pnl = []
    
    for d in range(n_days):
        idx_s, idx_e = day_indices[d], day_indices[d+1]
        cur_bias = bias_vals[idx_s]
        
        short_done = False
        long_done = False
        
        for i in range(idx_s, idx_e):
            m = minutes[i]
            
            # --- SHORT ENTRY ---
            if not short_done and cur_bias == -1 and m >= start_min and m <= end_min:
                e, tp, sl = entry_up[i], tp_up[i], sl_up[i]
                if not np.isnan(e) and highs[i] >= e:
                    short_done = True
                    for j in range(i+1, idx_e):
                        if highs[j] >= sl:
                            pnl.append(e - sl); break
                        if lows[j] <= tp:
                            pnl.append(e - tp); break
                        if minutes[j] >= stop_min:
                            pnl.append(e - closes[j]); break

            # --- LONG ENTRY ---
            if not long_done and cur_bias == 1 and m >= start_min and m <= end_min:
                e, tp, sl = entry_dn[i], tp_dn[i], sl_dn[i]
                if not np.isnan(e) and lows[i] <= e:
                    long_done = True
                    for j in range(i+1, idx_e):
                        if lows[j] <= sl:
                            pnl.append(sl - e); break
                        if highs[j] >= tp:
                            pnl.append(tp - e); break
                        if minutes[j] >= stop_min:
                            pnl.append(closes[j] - e); break
                            
    return np.array(pnl)

# ═════════════════════════════════════════════════════════════════════════════
# 2. STRATEGY ENGINE IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

def run_engine_1(df, df_daily, day_indices):
    h, l, c, m = df['High'].values, df['Low'].values, df['Close'].values, df['minutes'].values
    b = df['bias'].values
    e_up, tp_up, sl_up = (df['Session_Open'] + df['upP75']).values, (df['Session_Open'] + df['upP50']).values, (df['Session_Open'] + df['upP95']).values
    e_dn, tp_dn, sl_dn = (df['Session_Open'] - df['dnP75']).values, (df['Session_Open'] - df['dnP50']).values, (df['Session_Open'] - df['dnP95']).values
    return execute_engine(h, l, c, m, day_indices, e_up, tp_up, sl_up, e_dn, tp_dn, sl_dn, b, 0, 1440, 945)

def run_engine_2(df, df_daily, day_indices, cols):
    df_copy = df.copy()
    df_copy['is_ib'] = (df_copy['minutes'] >= 0) & (df_copy['minutes'] < 60)
    ib_data = df_copy[df_copy['is_ib']].groupby('date').agg({'High': 'max', 'Low': 'min'}).rename(columns={'High': 'IBH', 'Low': 'IBL'})
    
    df_d = df_daily.merge(ib_data, left_on='date', right_index=True)
    df_d['ext_up'] = (df_d['High'] - df_d['IBH']).clip(lower=0)
    df_d['ext_dn'] = (df_d['IBL'] - df_d['Low']).clip(lower=0)
    
    p_list = np.array([50.0, 75.0, 90.0, 95.0])
    vols = df_d['vol'].values
    is_valid = (df_d['daily_range'] > 0).values 
    proj = _calc_quant_projections(df_d['ext_up'].values, df_d['ext_dn'].values, vols, is_valid, 90, 30, p_list)
    
    df_proj = pd.DataFrame(proj, index=df_d.index, columns=cols)
    df_proj['date'] = df_d['date'].values
    df_proj = df_proj.merge(ib_data, left_index=True, right_index=True)
    
    df_clean = df.drop(columns=cols, errors='ignore')
    df_clean['join_date'] = df_clean.index.date
    df_proj['join_date'] = df_proj['date']
    
    df_m = df_clean.reset_index().merge(df_proj.drop(columns=['date']), on='join_date', how='left').set_index('DateTime')
    df_m = df_m.dropna(subset=['upP75'])
    
    h, l, c, m_v = df_m['High'].values, df_m['Low'].values, df_m['Close'].values, df_m['minutes'].values
    b = df_m['bias'].values
    e_up, tp_up, sl_up = (df_m['IBH'] + df_m['upP75']).values, df_m['IBH'].values, (df_m['IBH'] + df_m['upP95']).values
    e_dn, tp_dn, sl_dn = (df_m['IBL'] - df_m['dnP75']).values, df_m['IBL'].values, (df_m['IBL'] - df_m['dnP95']).values
    
    d_idx = np.where(df_m.index.date != pd.Series(df_m.index.date).shift(1).values)[0]
    d_idx = np.append(d_idx, len(df_m))
    
    return execute_engine(h, l, c, m_v, d_idx, e_up, tp_up, sl_up, e_dn, tp_dn, sl_dn, b, 60, 1440, 945)

def run_engine_3(df, df_daily, day_indices):
    df_copy = df.copy()
    df_copy['cv'] = df_copy['Close'] * df_copy['Volume']
    df_copy['vwap'] = df_copy.groupby('date')['cv'].cumsum() / df_copy.groupby('date')['Volume'].cumsum()
    df_copy['log_ret'] = np.log(df_copy['Close'] / df_copy['Close'].shift(1))
    df_copy['intra_vol'] = df_copy['log_ret'].rolling(60).std()
    df_copy['vol_ratio'] = df_copy['intra_vol'] / df_copy['vol']
    
    e_up = (df_copy['vwap'] + df_copy['upP75'] * df_copy['vol_ratio']).values
    tp_up = df_copy['vwap'].values
    sl_up = (df_copy['vwap'] + df_copy['upP95'] * df_copy['vol_ratio']).values
    e_dn = (df_copy['vwap'] - df_copy['dnP75'] * df_copy['vol_ratio']).values
    tp_dn = df_copy['vwap'].values
    sl_dn = (df_copy['vwap'] - df_copy['dnP95'] * df_copy['vol_ratio']).values
    
    h, l, c, m = df_copy['High'].values, df_copy['Low'].values, df_copy['Close'].values, df_copy['minutes'].values
    b = df_copy['bias'].values
    return execute_engine(h, l, c, m, day_indices, e_up, tp_up, sl_up, e_dn, tp_dn, sl_dn, b, 60, 1440, 945)

# ═════════════════════════════════════════════════════════════════════════════
# 3. COORDINATOR
# ═════════════════════════════════════════════════════════════════════════════

def calculate_metrics(p, engine_name, ticker):
    if len(p) == 0: return {"Engine": engine_name, "Ticker": ticker, "Trades": 0, "Win%": 0, "PnL": 0, "EV": 0, "PF": 0, "Sharpe": 0}
    w, l = p[p > 0], p[p < 0]
    std = np.std(p)
    return {
        "Engine": engine_name, "Ticker": ticker, "Trades": len(p),
        "Win%": round(len(w)/len(p)*100, 1), "PnL": round(np.sum(p), 1),
        "EV": round(np.mean(p), 2), 
        "PF": round(np.sum(w)/np.abs(np.sum(l)), 2) if np.sum(l) != 0 else np.inf,
        "Sharpe": round((np.mean(p)/std) * np.sqrt(252*2), 2) if std != 0 else 0
    }

def process_ticker(df, ticker):
    print(f"Processing {ticker}...")
    df_daily = df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_daily['date'] = df_daily.index.date
    df_daily['up_move'] = df_daily['High'] - df_daily['Open']
    df_daily['dn_move'] = df_daily['Open'] - df_daily['Low']
    df_daily['daily_range'] = df_daily['High'] - df_daily['Low']
    
    uv, dv = df_daily['up_move'].rolling(5).var(), df_daily['dn_move'].rolling(5).var()
    df_daily['bias'] = 0
    df_daily.loc[dv > uv * 1.2, 'bias'] = -1
    df_daily.loc[uv > dv * 1.2, 'bias'] = 1
    df_daily['vol'] = np.log(df_daily['Close'] / df_daily['Close'].shift(1)).rolling(20).std(ddof=1).fillna(0)
    
    is_v = (df_daily['daily_range'] > 0).values
    p_list = np.array([50.0, 75.0, 90.0, 95.0])
    q_proj = _calc_quant_projections(df_daily['up_move'].values, df_daily['dn_move'].values, df_daily['vol'].values, is_v, 90, 30, p_list)
    
    cols = ['upP50', 'upP75', 'upP90', 'upP95', 'dnP50', 'dnP75', 'dnP90', 'dnP95']
    df_proj = pd.DataFrame(q_proj, index=df_daily.index, columns=cols)
    df_proj[['bias', 'vol', 'Session_Open', 'date']] = df_daily[['bias', 'vol', 'Open', 'date']]
    
    df['date'] = df.index.date
    df_m = df.reset_index().merge(df_proj, on='date', how='left').set_index('DateTime')
    df_m = df_m.dropna(subset=['upP50'])
    df_m['minutes'] = df_m.index.hour * 60 + df_m.index.minute
    
    d_idx = np.where(df_m.index.date != pd.Series(df_m.index.date).shift(1).values)[0]
    d_idx = np.append(d_idx, len(df_m))
    
    res = []
    res.append(calculate_metrics(run_engine_1(df_m, df_daily, d_idx), "1. Daily", ticker))
    res.append(calculate_metrics(run_engine_2(df_m, df_daily, d_idx, cols), "2. IB Anchor", ticker))
    res.append(calculate_metrics(run_engine_3(df_m, df_daily, d_idx), "3. VWAP Roll", ticker))
    return res

if __name__ == "__main__":
    t_start = time.time()
    ticker_files = DATA_FILES
    all_res = []
    for ticker, file in ticker_files.items():
        if os.path.exists(file):
            df = pd.read_csv(file, header=None, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
            df.set_index('DateTime', inplace=True)
            all_res.extend(process_ticker(df, ticker))
        else:
            print(f"File {file} not found for {ticker}.")
            
    master_df = pd.DataFrame(all_res)
    print("\n" + "="*95 + "\nPORTFOLIO PERFORMANCE MATRIX (REAL DATA)\n" + "="*95)
    print(master_df.to_string(index=False))
    
    print("\n" + "="*95 + "\nPORTFOLIO AGGREGATES BY ENGINE\n" + "="*95)
    agg = master_df.groupby('Engine').agg({'Trades': 'sum', 'PnL': 'sum', 'EV': 'mean', 'PF': 'mean', 'Sharpe': 'mean'})
    print(agg.to_string())
    print(f"\nExecution Time: {time.time() - t_start:.2f}s")
