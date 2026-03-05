import pandas as pd
import numpy as np
from numba import jit
from core.data_manager import DataManager
from config import DATA_FILES

from core.time_utils import assign_trading_sessions
from core.math_utils import _pine_percentile, _calc_retail_projections, _calc_quant_projections

from sklearn.metrics import brier_score_loss, log_loss
import time
import os
import sys

# FIX 5.2: Single source of truth for all parameters
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BIAS_THRESHOLD, VOL_PERIOD, SKEW_PERIOD, WARMUP_DAYS, HURST_WINDOW, SMA_PERIOD, HURST_TAUS, RISK_PCT, MNQ_POINT_VALUE

# ═════════════════════════════════════════════════════════════════════════════
# 1. CORE NUMBA COMPUTATION ENGINES
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# 2. PANDAS DATAFRAME WRAPPERS
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest_models(df_1m, lookback=90, min_samples=30):
    t0 = time.time()
    print("Preparing and resampling data...")
    
    df_1m = assign_trading_sessions(df_1m)
    df_daily = df_1m.groupby('session_date').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    df_daily['up_move'] = df_daily['High'] - df_daily['Open']
    df_daily['dn_move'] = df_daily['Open'] - df_daily['Low']
    df_daily['daily_range'] = df_daily['High'] - df_daily['Low']
    
    avg_vol = df_daily['Volume'].rolling(20).mean()
    vol_ratio_filter = np.where(avg_vol > 0, df_daily['Volume'] / avg_vol, 1.0)
    # FIX 1.2: Dynamic validity filter instead of hardcoded 50-point minimum
    rolling_median_range = df_daily['daily_range'].rolling(60, min_periods=20).median()
    is_valid = (df_daily['daily_range'] >= rolling_median_range * 0.25) & (vol_ratio_filter >= 0.5)
    
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
    file_path = DATA_FILES["NQ"]
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
    else:
        df_1m = DataManager().load_data(file_path)
        df_daily, df_retail, df_quant = run_backtest_models(df_1m)
        metrics_df = evaluate_calibration(df_1m, df_daily, df_retail, df_quant)
