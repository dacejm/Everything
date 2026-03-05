import pandas as pd
import numpy as np
from numba import jit
from sklearn.metrics import brier_score_loss, log_loss
import time
import os

# ═════════════════════════════════════════════════════════════════════════════
# 1. CORE NUMBA COMPUTATION ENGINES
# ═════════════════════════════════════════════════════════════════════════════

@jit(nopython=True)
def _pine_percentile(arr, q):
    """Exact replication of the Pine Script mathematical percentile function."""
    if len(arr) < 5:
        return np.nan
    s = np.sort(arr)
    idx = min(int(np.floor(len(arr) * q / 100.0)), len(arr) - 1)
    return s[idx]

@jit(nopython=True)
def _calc_retail_projections(up_moves, dn_moves, regimes, is_valid, lookback, min_samples):
    """Simulates the Retail Pine Script logic with separated regime arrays."""
    n = len(up_moves)
    proj = np.full((n, 6), np.nan)
    
    # Global tracking arrays (max size = lookback)
    hist_up = np.zeros(lookback)
    hist_dn = np.zeros(lookback)
    count = 0
    
    # Regime-specific tracking arrays
    hist_up_high = np.zeros(lookback); hist_dn_high = np.zeros(lookback); count_high = 0
    hist_up_norm = np.zeros(lookback); hist_dn_norm = np.zeros(lookback); count_norm = 0
    hist_up_low = np.zeros(lookback);  hist_dn_low = np.zeros(lookback);  count_low = 0
    
    for i in range(n):
        # Use yesterday's regime as the starting bias for today
        cur_regime = regimes[i-1] if i > 0 else 0
        
        arr_up = hist_up[:count]
        arr_dn = hist_dn[:count]
        
        # Route to specific regime arrays if minimum sample size is met
        if cur_regime == 1 and count_high >= min_samples // 3:
            arr_up = hist_up_high[:count_high]
            arr_dn = hist_dn_high[:count_high]
        elif cur_regime == 2 and count_low >= min_samples // 3:
            arr_up = hist_up_low[:count_low]
            arr_dn = hist_dn_low[:count_low]
        elif count_norm >= min_samples // 3:
            arr_up = hist_up_norm[:count_norm]
            arr_dn = hist_dn_norm[:count_norm]
            
        multiplier = 1.0
        if cur_regime == 1:
            multiplier = 1.25
        elif cur_regime == 2:
            multiplier = 0.85
            
        # Lock in projected percentiles for day `i`
        if count >= min_samples:
            proj[i, 0] = _pine_percentile(arr_up, 50) * multiplier
            proj[i, 1] = _pine_percentile(arr_up, 75) * multiplier
            proj[i, 2] = _pine_percentile(arr_up, 90) * multiplier
            proj[i, 3] = _pine_percentile(arr_dn, 50) * multiplier
            proj[i, 4] = _pine_percentile(arr_dn, 75) * multiplier
            proj[i, 5] = _pine_percentile(arr_dn, 90) * multiplier
            
        # Record day `i` data into the historical tracking buffer if valid
        if is_valid[i]:
            for j in range(lookback - 1, 0, -1):
                hist_up[j] = hist_up[j-1]; hist_dn[j] = hist_dn[j-1]
            hist_up[0] = up_moves[i]; hist_dn[0] = dn_moves[i]
            if count < lookback: count += 1
            
            # Map into the respective regime tracking arrays
            r = regimes[i]
            if r == 1:
                for j in range(lookback - 1, 0, -1):
                    hist_up_high[j] = hist_up_high[j-1]; hist_dn_high[j] = hist_dn_high[j-1]
                hist_up_high[0] = up_moves[i]; hist_dn_high[0] = dn_moves[i]
                if count_high < lookback: count_high += 1
            elif r == 2:
                for j in range(lookback - 1, 0, -1):
                    hist_up_low[j] = hist_up_low[j-1]; hist_dn_low[j] = hist_dn_low[j-1]
                hist_up_low[0] = up_moves[i]; hist_dn_low[0] = dn_moves[i]
                if count_low < lookback: count_low += 1
            else:
                for j in range(lookback - 1, 0, -1):
                    hist_up_norm[j] = hist_up_norm[j-1]; hist_dn_norm[j] = hist_dn_norm[j-1]
                hist_up_norm[0] = up_moves[i]; hist_dn_norm[0] = dn_moves[i]
                if count_norm < lookback: count_norm += 1

    return proj

@jit(nopython=True)
def _calc_quant_projections(up_moves, dn_moves, vols, is_valid, lookback, min_samples):
    """Simulates the Quant Pine Script with dynamic continuous scaling."""
    n = len(up_moves)
    proj = np.full((n, 6), np.nan)
    
    hist_up = np.zeros(lookback)
    hist_dn = np.zeros(lookback)
    hist_vol = np.zeros(lookback)
    count = 0
    
    for i in range(n):
        # We project for day `i` based on the volatility tracked at the end of `i-1`
        cur_vol = vols[i-1] if i > 0 else 0.0
        
        # Lock in projected scaled percentiles for day `i`
        if count >= min_samples and cur_vol > 0:
            scaled_up = np.zeros(count)
            scaled_dn = np.zeros(count)
            
            for j in range(count):
                hv = hist_vol[j]
                multiplier = (cur_vol / hv) if hv > 0 else 1.0
                scaled_up[j] = hist_up[j] * multiplier
                scaled_dn[j] = hist_dn[j] * multiplier
                
            proj[i, 0] = _pine_percentile(scaled_up, 50)
            proj[i, 1] = _pine_percentile(scaled_up, 75)
            proj[i, 2] = _pine_percentile(scaled_up, 90)
            proj[i, 3] = _pine_percentile(scaled_dn, 50)
            proj[i, 4] = _pine_percentile(scaled_dn, 75)
            proj[i, 5] = _pine_percentile(scaled_dn, 90)
            
        # Record day `i` outcome to history to be used in future scaling
        if is_valid[i]:
            for j in range(lookback - 1, 0, -1):
                hist_up[j] = hist_up[j-1]
                hist_dn[j] = hist_dn[j-1]
                hist_vol[j] = hist_vol[j-1]
                
            hist_up[0] = up_moves[i]
            hist_dn[0] = dn_moves[i]
            hist_vol[0] = vols[i]  # Record the volatility as it closed on this day
            if count < lookback: count += 1
            
    return proj

# ═════════════════════════════════════════════════════════════════════════════
# 2. PANDAS DATAFRAME WRAPPERS
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest_models(df_1m, lookback=90, min_samples=30):
    t0 = time.time()
    print("Preparing and resampling data...")
    
    # Identify unique sessions. NQ trades Sunday evening to Friday afternoon.
    # Simple resample('D') works if the data is aligned.
    # Using 'B' (business day) resample might be better, or just 'D' and dropping NaNs.
    df_daily = df_1m.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    df_daily['up_move'] = df_daily['High'] - df_daily['Open']
    df_daily['dn_move'] = df_daily['Open'] - df_daily['Low']
    df_daily['daily_range'] = df_daily['High'] - df_daily['Low']
    
    avg_vol = df_daily['Volume'].rolling(20).mean()
    vol_ratio_filter = np.where(avg_vol > 0, df_daily['Volume'] / avg_vol, 1.0)
    is_valid = (df_daily['daily_range'] >= 50.0) & (vol_ratio_filter >= 0.5)
    
    # ---------------------------------------------------------
    # RUN RETAIL MODEL
    # ---------------------------------------------------------
    print("Executing Retail Numba kernel...")
    high_low = df_daily['High'] - df_daily['Low']
    high_close = np.abs(df_daily['High'] - df_daily['Close'].shift(1))
    low_close = np.abs(df_daily['Low'] - df_daily['Close'].shift(1))
    atr14 = np.maximum(high_low, np.maximum(high_close, low_close)).rolling(14).mean()
    avg_atr = atr14.rolling(28).mean()
    
    atr_ratio = np.where(avg_atr > 0, atr14 / avg_atr, 1.0)
    regimes = np.zeros(len(df_daily), dtype=np.int32)
    regimes[atr_ratio > 1.3] = 1 # High
    regimes[atr_ratio < 0.7] = 2 # Low
    
    r_proj = _calc_retail_projections(
        df_daily['up_move'].values, df_daily['dn_move'].values, 
        regimes, is_valid.values, lookback, min_samples
    )
    df_retail = pd.DataFrame(r_proj, index=df_daily.index, columns=[
        'r_up50', 'r_up75', 'r_up90', 'r_dn50', 'r_dn75', 'r_dn90'
    ])

    # ---------------------------------------------------------
    # RUN QUANT MODEL
    # ---------------------------------------------------------
    print("Executing Quant Numba kernel...")
    log_ret = np.log(df_daily['Close'] / df_daily['Close'].shift(1))
    stdev_20 = log_ret.rolling(20).std(ddof=1).fillna(0.0).values
    
    q_proj = _calc_quant_projections(
        df_daily['up_move'].values, df_daily['dn_move'].values, 
        stdev_20, is_valid.values, lookback, min_samples
    )
    df_quant = pd.DataFrame(q_proj, index=df_daily.index, columns=[
        'q_up50', 'q_up75', 'q_up90', 'q_dn50', 'q_dn75', 'q_dn90'
    ])
    
    print(f"Kernel Execution Time: {time.time() - t0:.3f}s")
    return df_daily, df_retail, df_quant


def evaluate_calibration(df_1m, df_daily, df_retail, df_quant):
    print("Mapping arrays to 1-minute dataframe and executing evaluation...")
    df_eval = pd.concat([df_daily[['Open']].rename(columns={'Open': 'Session_Open'}), df_retail, df_quant], axis=1)
    
    # We use .copy() to avoid SettingWithCopyWarnings
    df_1m_copy = df_1m.copy()
    df_1m_copy['date'] = df_1m_copy.index.date
    df_eval['date'] = df_eval.index.date
    
    # Use reset_index() and set_index() to maintain original index
    original_index_name = df_1m_copy.index.name
    df_merged = df_1m_copy.reset_index().merge(df_eval, on='date', how='left').set_index(original_index_name if original_index_name else 'index')
    df_merged = df_merged.dropna(subset=['r_up50', 'q_up50']) 
    
    for m_prefix in ['r', 'q']:
        for dr, p in [('up', 50), ('up', 75), ('up', 90), ('dn', 50), ('dn', 75), ('dn', 90)]:
            col = f'{m_prefix}_{dr}{p}'
            sign = 1 if dr == 'up' else -1
            df_merged[f'{col.upper()}_TARGET'] = df_merged['Session_Open'] + (df_merged[col] * sign)

    breaches_1m = pd.DataFrame(index=df_merged.index)
    for m_prefix in ['R', 'Q']:
        for p in [50, 75, 90]:
            breaches_1m[f'{m_prefix}_UP{p}_HIT'] = df_merged['High'] >= df_merged[f'{m_prefix}_UP{p}_TARGET']
            breaches_1m[f'{m_prefix}_DN{p}_HIT'] = df_merged['Low'] <= df_merged[f'{m_prefix}_DN{p}_TARGET']
    
    df_merged['date'] = df_merged.index.date
    daily_actuals = breaches_1m.groupby(df_merged['date']).max().astype(int)

    exp_probs = {50: 0.50, 75: 0.25, 90: 0.10} 
    results = []
    
    print("\n" + "="*60)
    print("📉 STATISTICAL CALIBRATION COMPARISON (Brier/LogLoss)")
    print("="*60)
    
    for prefix, model_name in [('R', 'Retail'), ('Q', 'Quant')]:
        print(f"\n[{model_name.upper()} MODEL]")
        
        for p, exp_rate in exp_probs.items():
            combined_actuals = np.concatenate([
                daily_actuals[f'{prefix}_UP{p}_HIT'].values, 
                daily_actuals[f'{prefix}_DN{p}_HIT'].values
            ])
            forecasts = np.full(len(combined_actuals), exp_rate)
            
            actual_rate = combined_actuals.mean()
            brier = brier_score_loss(combined_actuals, forecasts)
            l_loss = log_loss(combined_actuals, forecasts, labels=[0, 1])
            
            bias_diff = actual_rate - exp_rate
            eval_tag = "✅ Excellent" if abs(bias_diff) < 0.03 else "⚠️ Loose" if abs(bias_diff) < 0.08 else "❌ Poor"
            
            print(f" P{p:02d} | Expected: {exp_rate*100:04.1f}% | Actual: {actual_rate*100:04.1f}% | Diff: {bias_diff*100:>5.1f}% | BS: {brier:.4f} | {eval_tag}")
            
            results.append({
                'Model': model_name,
                'Percentile': f'P{p}',
                'Expected': exp_rate,
                'Actual Rate': actual_rate,
                'Brier Score': brier,
                'Log Loss': l_loss
            })

    return pd.DataFrame(results)

# ====================================================================
# NQ DATA LOADER
# ====================================================================
def load_nq_csv(file_path):
    print(f"Loading {file_path}...")
    # CSV Format: 20170417 175500,5381.0,5382.0,5380.0,5381.75,334
    # No header, parsing datetime from first column
    df = pd.read_csv(file_path, header=None, names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S')
    df.set_index('DateTime', inplace=True)
    print(f"Successfully loaded {len(df)} rows.")
    return df

if __name__ == "__main__":
    file_path = "ENQH26.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
    else:
        df_1m = load_nq_csv(file_path)
        df_daily, df_retail, df_quant = run_backtest_models(df_1m)
        metrics_df = evaluate_calibration(df_1m, df_daily, df_retail, df_quant)
