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
    # Path Data: [PnL, MAE, MFE]
    path_data = [] 
    n_days = len(day_indices) - 1
    for d in range(n_days):
        idx_s, idx_e = day_indices[d], day_indices[d+1]
        cur_bias = bias_vals[idx_s]
        short_done, long_done = False, False
        for i in range(idx_s, idx_e):
            m = minutes[i]
            # SHORT
            if not short_done and cur_bias == -1 and m >= start_min and m <= end_min:
                e, tp, sl = entry_up[i], tp_up[i], sl_up[i]
                if not np.isnan(e) and highs[i] >= e:
                    short_done = True
                    mae, mfe = 0.0, 0.0
                    for j in range(i+1, idx_e):
                        mae = max(mae, highs[j] - e)
                        mfe = max(mfe, e - lows[j])
                        if highs[j] >= sl: path_data.append([e-sl, mae, mfe]); break
                        if lows[j] <= tp: path_data.append([e-tp, mae, mfe]); break
                        if minutes[j] >= stop_min: path_data.append([e-closes[j], mae, mfe]); break
            # LONG
            if not long_done and cur_bias == 1 and m >= start_min and m <= end_min:
                e, tp, sl = entry_dn[i], tp_dn[i], sl_dn[i]
                if not np.isnan(e) and lows[i] <= e:
                    long_done = True
                    mae, mfe = 0.0, 0.0
                    for j in range(i+1, idx_e):
                        mae = max(mae, e - lows[j])
                        mfe = max(mfe, highs[j] - e)
                        if lows[j] <= sl: path_data.append([sl-e, mae, mfe]); break
                        if highs[j] >= tp: path_data.append([tp-e, mae, mfe]); break
                        if minutes[j] >= stop_min: path_data.append([closes[j]-e, mae, mfe]); break
    return np.array(path_data)

# ═════════════════════════════════════════════════════════════════════════════
# 2. EVALUATION & METRIC KERNELS
# ═════════════════════════════════════════════════════════════════════════════

def get_stats(path_data, engine, ticker):
    if len(path_data) == 0: return {"Engine": engine, "Ticker": ticker, "Trades": 0}
    pnl, mae, mfe = path_data[:, 0], path_data[:, 1], path_data[:, 2]
    w = pnl[pnl > 0]
    cum_pnl = np.cumsum(pnl)
    max_dd = np.max(np.maximum.accumulate(cum_pnl) - cum_pnl) if len(pnl) > 0 else 0
    std = np.std(pnl)
    return {
        "Engine": engine, "Ticker": ticker, "Trades": len(pnl),
        "Win%": round(len(w)/len(pnl)*100, 1), "PnL": round(np.sum(pnl), 1),
        "EV": round(np.mean(pnl), 2), "PF": round(np.sum(w)/np.abs(np.sum(pnl[pnl<0])), 2) if np.any(pnl<0) else np.inf,
        "Sharpe": round((np.mean(pnl)/std) * np.sqrt(252*2), 2) if std != 0 else 0,
        "MaxDD": round(max_dd, 1), "MAE50": round(np.percentile(mae, 50), 2), "MFE50": round(np.percentile(mfe, 50), 2)
    }

# ═════════════════════════════════════════════════════════════════════════════
# 3. MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_analysis(df, ticker):
    print(f"Analyzing {ticker}...")
    df_d = df.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_d['date'] = df_d.index.date
    df_d['up_m'] = df_d['High'] - df_d['Open']
    df_d['dn_m'] = df_d['Open'] - df_d['Low']
    uv, dv = df_d['up_m'].rolling(5).var(), df_d['dn_m'].rolling(5).var()
    df_d['bias'] = 0
    df_d.loc[dv > uv * 1.2, 'bias'] = -1; df_d.loc[uv > dv * 1.2, 'bias'] = 1
    df_d['vol'] = np.log(df_d['Close'] / df_d['Close'].shift(1)).rolling(20).std(ddof=1).fillna(0)
    
    p_list = np.array([50.0, 75.0, 90.0, 95.0])
    q_proj = _calc_quant_projections(df_d['up_m'].values, df_d['dn_m'].values, df_d['vol'].values, np.ones(len(df_d), dtype=np.bool_), 90, 30, p_list)
    cols = ['upP50', 'upP75', 'upP90', 'upP95', 'dnP50', 'dnP75', 'dnP90', 'dnP95']
    df_p = pd.DataFrame(q_proj, index=df_d.index, columns=cols)
    df_p[['bias', 'vol', 'S_O', 'date']] = df_d[['bias', 'vol', 'Open', 'date']]
    
    df['date'] = df.index.date
    df_m = df.reset_index().merge(df_p, on='date', how='left').set_index('DateTime')
    df_m = df_m.dropna(subset=['upP50'])
    df_m['min'] = df_m.index.hour * 60 + df_m.index.minute
    d_idx = np.where(df_m.index.date != pd.Series(df_m.index.date).shift(1).values)[0]
    d_idx = np.append(d_idx, len(df_m))
    
    h, l, c, m, b = df_m['High'].values, df_m['Low'].values, df_m['Close'].values, df_m['min'].values, df_m['bias'].values
    
    # Engine 1: Daily
    e_u, tp_u, sl_u = (df_m['S_O']+df_m['upP75']).values, (df_m['S_O']+df_m['upP50']).values, (df_m['S_O']+df_m['upP95']).values
    e_d, tp_d, sl_d = (df_m['S_O']-df_m['dnP75']).values, (df_m['S_O']-df_m['dnP50']).values, (df_m['S_O']-df_m['dnP95']).values
    res1 = execute_engine(h, l, c, m, d_idx, e_u, tp_u, sl_u, e_d, tp_d, sl_d, b, 0, 1440, 945)
    
    # Engine 2: IB
    df_m['IBH'] = df_m.groupby('date')['High'].transform(lambda x: x[:60].max())
    df_m['IBL'] = df_m.groupby('date')['Low'].transform(lambda x: x[:60].min())
    e_u, tp_u, sl_u = (df_m['IBH']+df_m['upP75']).values, df_m['IBH'].values, (df_m['IBH']+df_m['upP95']).values
    e_d, tp_d, sl_d = (df_m['IBL']-df_m['dnP75']).values, df_m['IBL'].values, (df_m['IBL']-df_m['dnP95']).values
    res2 = execute_engine(h, l, c, m, d_idx, e_u, tp_u, sl_u, e_d, tp_d, sl_d, b, 60, 1440, 945)
    
    # Engine 3: VWAP
    df_m['cv'] = df_m['Close'] * df_m['Volume']
    df_m['vwap'] = df_m.groupby('date')['cv'].cumsum() / df_m.groupby('date')['Volume'].cumsum()
    df_m['vr'] = (df_m['Close'].pct_change().rolling(60).std() / df_m['vol']).fillna(0)
    e_u, tp_u, sl_u = (df_m['vwap']+df_m['upP75']*df_m['vr']).values, df_m['vwap'].values, (df_m['vwap']+df_m['upP95']*df_m['vr']).values
    e_d, tp_d, sl_d = (df_m['vwap']-df_m['dnP75']*df_m['vr']).values, df_m['vwap'].values, (df_m['vwap']-df_m['dnP95']*df_m['vr']).values
    res3 = execute_engine(h, l, c, m, d_idx, e_u, tp_u, sl_u, e_d, tp_d, sl_d, b, 60, 1440, 945)
    
    return [get_stats(res1, "1.Daily", ticker), get_stats(res2, "2.IBAnc", ticker), get_stats(res3, "3.VWAPR", ticker)]

if __name__ == "__main__":
    ticker_files = DATA_FILES
    results = []
    for t, f in ticker_files.items():
        if os.path.exists(f):
            df = pd.read_csv(f, header=None, names=['DateTime','Open','High','Low','Close','Volume'])
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
            df.set_index('DateTime', inplace=True)
            results.extend(run_analysis(df, t))
    
    master = pd.DataFrame(results)
    print("\n" + "="*125 + "\nFULL TRIPLE-ENGINE PORTFOLIO AUDIT (ALL SYMBOLS)\n" + "="*125)
    print(master.to_string(index=False))
