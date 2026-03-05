import pandas as pd
import numpy as np
from numba import jit
import os
from datetime import datetime

# ═════════════════════════════════════════════════════════════════════════════
# 1. THE ULTIMATE NUMBA KERNEL (FULL DATASET PROCESSOR)
# ═════════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def execute_runner_backtest_numba(highs, lows, closes, minutes, starts, ends, 
                                 biases, entries, t1s, stops):
    results = np.zeros(len(starts))
    
    for i in range(len(starts)):
        s_idx, e_idx = starts[i], ends[i]
        bias, ent, t1, sl = biases[i], entries[i], t1s[i], stops[i]
        
        if bias == 0: continue
        
        entry_idx = -1
        for j in range(s_idx, e_idx):
            if (bias == -1 and highs[j] >= ent) or (bias == 1 and lows[j] <= ent):
                entry_idx = j
                break
        
        if entry_idx == -1: continue

        t1_hit = False
        cur_sl = sl
        pnl_h1, pnl_h2 = 0.0, 0.0
        finished = False
        
        for k in range(entry_idx + 1, e_idx):
            m = minutes[k]
            if not t1_hit:
                if (bias == -1 and highs[k] >= cur_sl) or (bias == 1 and lows[k] <= cur_sl):
                    pnl_h1 = pnl_h2 = (ent - cur_sl) if bias == -1 else (cur_sl - ent)
                    finished = True; break
                if (bias == -1 and lows[k] <= t1) or (bias == 1 and highs[k] >= t1):
                    t1_hit = True
                    pnl_h1 = (ent - t1) if bias == -1 else (t1 - ent)
                    cur_sl = ent
            else:
                if (bias == -1 and highs[k] >= cur_sl) or (bias == 1 and lows[k] <= cur_sl):
                    pnl_h2 = 0.0
                    finished = True; break
            
            if m >= 945:
                if not t1_hit: pnl_h1 = (ent - closes[k]) if bias == -1 else (closes[k] - ent)
                pnl_h2 = (ent - closes[k]) if bias == -1 else (closes[k] - ent)
                finished = True; break
        
        if not finished:
            if not t1_hit: pnl_h1 = (ent - closes[e_idx-1]) if bias == -1 else (closes[e_idx-1] - ent)
            pnl_h2 = (ent - closes[e_idx-1]) if bias == -1 else (closes[e_idx-1] - ent)

        results[i] = (pnl_h1 + pnl_h2) / 2
        
    return results

# ═════════════════════════════════════════════════════════════════════════════
# 2. OPTIMIZED DATA PREP
# ═════════════════════════════════════════════════════════════════════════════

def run_vectorized_backtest():
    files = [f for f in os.listdir('.') if f.endswith(('.csv', '.txt')) and 'NQ' in f.upper()]
    target = max(files, key=os.path.getsize)
    print(f"Vectorizing Backtest on: {target}")

    df = pd.read_csv(target, header=None, names=['T','O','H','L','C','V'], engine='c')
    df['T'] = pd.to_datetime(df['T'], format='%Y%m%d %H%M%S')
    df.set_index('T', inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    
    df['date'] = df.index.date
    df_rth = df.between_time('09:30', '15:59')
    df_daily = df_rth.groupby('date').agg({'O':'first','H':'max','L':'min','C':'last'})
    
    ib = df.between_time('09:30', '10:29').groupby('date').agg({'H':'max','L':'min'}).rename(columns={'H':'IBH','L':'IBL'})
    df_daily = df_daily.join(ib, how='inner')
    
    df_daily['vol'] = np.log(df_daily['C'] / df_daily['C'].shift(1)).rolling(20).std(ddof=1)
    df_daily['u_var'] = (df_daily['H'] - df_daily['O']).rolling(5).var()
    df_daily['d_var'] = (df_daily['O'] - df_daily['L']).rolling(5).var()
    
    print("Calculating session boundaries...")
    df['idx'] = np.arange(len(df))
    # Filter bounds based on 10:30-15:45 window
    df_sim_window = df.between_time('10:30', '15:45')
    bounds = df_sim_window.groupby('date')['idx'].agg(['first', 'last'])
    
    df_daily = df_daily.join(bounds, how='inner').dropna()
    WARMUP = 20
    
    print("Generating projections...")
    entries, t1s, stops, biases = [], [], [], []
    for i in range(WARMUP, len(df_daily)):
        hist = df_daily.iloc[i-WARMUP:i]
        today = df_daily.iloc[i]
        mult = today['vol'] / hist['vol'].values
        bias = -1 if today['d_var'] > today['u_var']*1.2 else (1 if today['u_var'] > today['d_var']*1.2 else 0)
        p50u, p75u, p95u = np.percentile((hist['H']-hist['IBH']).clip(lower=0).values * mult, [50, 75, 95])
        p50d, p75d, p95d = np.percentile((hist['IBL']-hist['L']).clip(lower=0).values * mult, [50, 75, 95])
        biases.append(bias); entries.append(today['IBH']+p75u if bias == -1 else today['IBL']-p75d)
        t1s.append(today['IBH']+p50u if bias == -1 else today['IBL']-p50d)
        stops.append(today['IBH']+p95u if bias == -1 else today['IBL']-p95d)

    df_test = df_daily.iloc[WARMUP:].copy()
    print("Executing Numba kernel...")
    pnl_results = execute_runner_backtest_numba(
        df['H'].values, df['L'].values, df['C'].values, 
        (df.index.hour * 60 + df.index.minute).values,
        df_test['first'].values, df_test['last'].values + 1,
        np.array(biases), np.array(entries), np.array(t1s), np.array(stops)
    )

    df_test['PnL'] = pnl_results
    final = df_test[df_test['PnL'] != 0]
    if not final.empty:
        p = final['PnL']
        print("\n" + "="*50 + f"\nNQ RUNNER RESULTS ({len(final)} Trades)\n" + "="*50)
        print(f"Win Rate:      {len(p[p>0])/len(p)*100:.1f}%")
        print(f"Profit Factor: {p[p>0].sum()/abs(p[p<0].sum()) if not p[p<0].empty else np.inf:.2f}")
        print(f"Total Points:  {p.sum():.2f}")
        final.to_csv("quant_model_runner_backtest.csv")
    else:
        print("No trades triggered.")

if __name__ == "__main__":
    run_vectorized_backtest()
