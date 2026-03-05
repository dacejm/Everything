import pandas as pd
import numpy as np
from numba import jit
from core.data_manager import DataManager
from config import DATA_FILES

from core.time_utils import assign_trading_sessions
from core.math_utils import _pine_percentile, _calc_retail_projections, _calc_quant_projections

import os
import sys

# FIX 5.2: Single source of truth for all parameters
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BIAS_THRESHOLD

# FIX 3.3: Correct per-contract commission
META = {
    'NQ':  {'tick': 0.25, 'pt_val': 20.0, 'rt_comm': 4.50},   # Full-size NQ
    'ES':  {'tick': 0.25, 'pt_val': 50.0, 'rt_comm': 4.50},
    'YM':  {'tick': 1.0,  'pt_val': 5.0,  'rt_comm': 4.50},
    'RTY': {'tick': 0.1,  'pt_val': 50.0, 'rt_comm': 4.50},
    'CL':  {'tick': 0.01, 'pt_val': 1000.0, 'rt_comm': 4.50},
    'GC':  {'tick': 0.1,  'pt_val': 100.0, 'rt_comm': 4.50}
}

@jit(nopython=True)
def execute_engine_friction(highs, lows, closes, minutes, day_indices, 
                            entry_up, tp_up, sl_up, entry_dn, tp_dn, sl_dn, 
                            bias_vals, start_min, end_min, stop_min):
    path_data = [] 
    for d in range(len(day_indices)-1):
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
                        if highs[j] >= sl: path_data.append([e-sl, 0.0]); break
                        if lows[j] <= tp: path_data.append([e-tp, 1.0]); break
                        if minutes[j] >= stop_min: path_data.append([e-closes[j], 2.0]); break
            if not long_done and cur_bias == 1 and m >= start_min and m <= end_min:
                e, tp, sl = entry_dn[i], tp_dn[i], sl_dn[i]
                if not np.isnan(e) and lows[i] <= e:
                    long_done = True
                    for j in range(i+1, idx_e):
                        if lows[j] <= sl: path_data.append([sl-e, 0.0]); break
                        if highs[j] >= tp: path_data.append([tp-e, 1.0]); break
                        if minutes[j] >= stop_min: path_data.append([closes[j]-e, 2.0]); break
    return np.array(path_data)

def apply_friction(results, ticker):
    if len(results) == 0: return None
    m = META[ticker]
    gross_pnl = results[:, 0]
    exit_type = results[:, 1]
    comm_pts = m['rt_comm'] / m['pt_val']
    net_pnls = []
    np.random.seed(42)
    for i in range(len(gross_pnl)):
        p = gross_pnl[i]; ext = exit_type[i]
        # FIX 3.4: Universal minimum slippage
        # Base slippage: 1 tick on every trade (entry fill)
        slip = m['tick']
        # Additional adverse slippage on stops and time exits
        if ext == 0 or ext == 2 or p < 0:
            slip += np.random.randint(0, 2) * m['tick']
        net_pnls.append(p - comm_pts - slip)
    net_pnls = np.array(net_pnls)
    std = np.std(net_pnls)
    sharpe = (np.mean(net_pnls) / std) * np.sqrt(252*2) if std != 0 else 0
    w = net_pnls[net_pnls > 0]; l = net_pnls[net_pnls < 0]
    return {
        'Ticker': ticker, 'Gross EV': round(np.mean(gross_pnl), 2), 
        'Net EV': round(np.mean(net_pnls), 2), 
        'Net EV USD': round(np.mean(net_pnls) * m['pt_val'], 2), 
        'Net PF': round(np.sum(w) / np.abs(np.sum(l)), 2) if len(l) > 0 else np.inf, 
        'Net Sharpe': round(sharpe, 2), 'Trades': len(net_pnls)
    }

if __name__ == "__main__":
    ticker_files = DATA_FILES
    final_rows = []
    for t, f in ticker_files.items():
        if os.path.exists(f):
            print(f"Loading {t}...")
            df = pd.read_csv(f, header=None, names=['DT','O','H','L','C','V'])
            df['DT'] = pd.to_datetime(df['DT'], format='%Y%m%d %H%M%S')
            df.set_index('DT', inplace=True); df['date'] = df.index.date
            
            df = assign_trading_sessions(df)
            df_d = df.groupby('session_date').agg({'O':'first','H':'max','L':'min','C':'last','V':'sum'}).dropna()
            df_d['date'] = df_d.index; df_d['up_m'] = df_d['H']-df_d['O']; df_d['dn_m'] = df_d['O']-df_d['L']
            uv, dv = df_d['up_m'].rolling(5).var(), df_d['dn_m'].rolling(5).var()
            
            # FIX 3.1: Widen bias dead zone
            df_d['bias'] = 0; df_d.loc[dv>uv*BIAS_THRESHOLD,'bias']=-1; df_d.loc[uv>dv*BIAS_THRESHOLD,'bias']=1
            df_d['vol'] = np.log(df_d['C']/df_d['C'].shift(1)).rolling(20).std(ddof=1).fillna(0)
            p_list = np.array([50.0, 75.0, 90.0, 95.0])
            q_proj = _calc_quant_projections(df_d['up_m'].values, df_d['dn_m'].values, df_d['vol'].values, np.ones(len(df_d), dtype=np.bool_), 90, 30, p_list)
            cols = ['upP50','upP75','upP90','upP95','dnP50','dnP75','dnP90','dnP95']
            df_p = pd.DataFrame(q_proj, index=df_d.index, columns=cols)
            df_p[['bias','vol','S_O','date']] = df_d[['bias','vol','O','date']]
            df_m = df.reset_index().merge(df_p, on='date', how='left').set_index('DT')
            df_m = df_m.dropna(subset=['upP50']); df_m['min'] = df_m.index.hour*60+df_m.index.minute
            d_idx = np.where(df_m.index.date != pd.Series(df_m.index.date).shift(1).values)[0]
            d_idx = np.append(d_idx, len(df_m))
            h, l, c, m, b = df_m['H'].values, df_m['L'].values, df_m['C'].values, df_m['min'].values, df_m['bias'].values
            
            # E1: Daily
            e_u, tp_u, sl_u = (df_m['S_O']+df_m['upP75']).values, (df_m['S_O']+df_m['upP50']).values, (df_m['S_O']+df_m['upP95']).values
            e_d, tp_d, sl_d = (df_m['S_O']-df_m['dnP75']).values, (df_m['S_O']-df_m['dnP50']).values, (df_m['S_O']-df_m['dnP95']).values
            r1 = execute_engine_friction(h, l, c, m, d_idx, e_u, tp_u, sl_u, e_d, tp_d, sl_d, b, 0, 1440, 945)
            metrics1 = apply_friction(r1, t); metrics1['Engine'] = '1.Daily'; final_rows.append(metrics1)
            
            # E2: IB
            df_m['IBH'] = df_m.groupby('date')['H'].transform(lambda x: x[:60].max())
            df_m['IBL'] = df_m.groupby('date')['L'].transform(lambda x: x[:60].min())
            e_u, tp_u, sl_u = (df_m['IBH']+df_m['upP75']).values, df_m['IBH'].values, (df_m['IBH']+df_m['upP95']).values
            e_d, tp_d, sl_d = (df_m['IBL']-df_m['dnP75']).values, df_m['IBL'].values, (df_m['IBL']-df_m['dnP95']).values
            r2 = execute_engine_friction(h, l, c, m, d_idx, e_u, tp_u, sl_u, e_d, tp_d, sl_d, b, 60, 1440, 945)
            metrics2 = apply_friction(r2, t); metrics2['Engine'] = '2.IBAnc'; final_rows.append(metrics2)

    res_df = pd.DataFrame(final_rows)
    print("\n" + "="*115 + "\nFRICTION-ADJUSTED PERFORMANCE MATRIX\n" + "="*115)
    print(res_df[['Engine','Ticker','Gross EV','Net EV','Net EV USD','Net PF','Net Sharpe','Trades']].to_string(index=False))
    print('\nPortfolio Net Aggregate Sharpe:', round(res_df['Net Sharpe'].mean(), 2))