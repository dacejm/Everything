import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import os

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
SYMBOL = "@ENQ"
ACCOUNT_SIZE = 50000.0
RISK_AMOUNT = ACCOUNT_SIZE * 0.01  # $500
MNQ_POINT_VALUE = 2.0              # $2 per point for Micros
WARMUP_DAYS = 45
NY_TZ = pytz.timezone("America/New_York")

def run_dry_run():
    # --- MT5 CONNECTION ---
    if not mt5.initialize():
        print("MT5 Initialization failed.")
        return
    
    if not mt5.symbol_select(SYMBOL, True):
        print(f"Failed to select {SYMBOL}. Ensure it is in Market Watch.")
        mt5.shutdown()
        return

    # Fetch 65,000 bars (~3 months of M1)
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 65000)
    if rates is None:
        print("Could not fetch data.")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert(NY_TZ)
    df['date'] = df.index.date

    # --- FEATURE ENGINEERING ---
    unique_dates = np.unique(df['date'])
    daily_records = []
    
    for d in unique_dates:
        day_data = df[df['date'] == d]
        session = day_data.between_time('09:30', '15:59')
        if session.empty: continue
        
        ib_data = day_data.between_time('09:30', '10:29')
        if ib_data.empty: continue
        
        daily_records.append({
            'date': d,
            'open': session['open'].iloc[0],
            'high': session['high'].max(),
            'low': session['low'].min(),
            'close': session['close'].iloc[-1],
            'IBH': ib_data['high'].max(),
            'IBL': ib_data['low'].min()
        })
        
    df_daily = pd.DataFrame(daily_records).set_index('date')
    df_daily['up_move'] = df_daily['high'] - df_daily['open']
    df_daily['dn_move'] = df_daily['open'] - df_daily['low']
    df_daily['ext_up'] = (df_daily['high'] - df_daily['IBH']).clip(lower=0)
    df_daily['ext_dn'] = (df_daily['IBL'] - df_daily['low']).clip(lower=0)
    
    # 20-session log-vol
    df_daily['vol'] = np.log(df_daily['close'] / df_daily['close'].shift(1)).rolling(20).std(ddof=1)
    
    # 5-day Skewness
    u_var = df_daily['up_move'].rolling(5).var()
    d_var = df_daily['dn_move'].rolling(5).var()
    df_daily['bias'] = 0
    df_daily.loc[d_var > u_var * 1.2, 'bias'] = -1 # Neg Skew (Short)
    df_daily.loc[u_var > d_var * 1.2, 'bias'] = 1  # Pos Skew (Long)
    
    df_daily = df_daily.dropna()
    
    # --- TODAY'S PROJECTIONS ---
    today_date = unique_dates[-1]
    today = df_daily.iloc[-1]
    # Check if today's data is actually available in the daily set
    if today.name != today_date:
        # Today's IB might not be formed yet in the aggregated data
        print(f"🕒 TODAY ({today_date}) is still forming. Projections based on most recent full session.")
        # We can't really do "Today" if IB isn't finished.
    
    history = df_daily.iloc[-WARMUP_DAYS-1:-1] # Prev 45 days
    
    cur_vol = today['vol']
    cur_bias = today['bias']
    
    # Normalization Scaling
    scaled_up = history['ext_up'].values * (cur_vol / history['vol'].values)
    scaled_dn = history['ext_dn'].values * (cur_vol / history['vol'].values)
    
    p50_up, p75_up, p95_up = np.percentile(scaled_up, [50, 75, 95])
    p50_dn, p75_dn, p95_dn = np.percentile(scaled_dn, [50, 75, 95])
    
    # Set Levels based on Bias
    bias_str = "NEUTRAL"
    entry, target, stop = 0.0, 0.0, 0.0
    
    if cur_bias == -1:
        bias_str = "SHORT (Neg Skew)"
        entry = today['IBH'] + p75_up
        target = today['IBH'] + p50_up
        stop = today['IBH'] + p95_up
    elif cur_bias == 1:
        bias_str = "LONG (Pos Skew)"
        entry = today['IBL'] - p75_dn
        target = today['IBL'] - p50_dn
        stop = today['IBL'] - p95_dn

    # --- DYNAMIC LOT SIZING ---
    risk_pts = abs(entry - stop)
    contracts = int(np.floor(RISK_AMOUNT / (risk_pts * MNQ_POINT_VALUE))) if risk_pts > 0 else 0
    contracts = max(1, contracts)

    # Round to tick size (0.25 for NQ)
    def round_t(val): return round(val * 4) / 4

    # --- REPORTING ---
    print("\n" + "═"*60)
    print(f" QUANT MODEL DRY-RUN | {today_date} ")
    print("═"*60)
    print(f"Skew Bias:      {bias_str}")
    print(f"Current Vol:    {cur_vol:.6f}")
    print(f"IB High/Low:    {today['IBH']:.2f} / {today['IBL']:.2f}")
    print("-" * 60)
    if entry > 0:
        print(f"P75 ENTRY:      {round_t(entry):.2f}")
        print(f"P50 TARGET:     {round_t(target):.2f}")
        print(f"P95 STOP:       {round_t(stop):.2f}")
        print("-" * 60)
        print(f"Risk Per Trade: ${RISK_AMOUNT:.2f}")
        print(f"Risk Distance:  {risk_pts:.2f} pts")
        print(f"POSITION SIZE:  {contracts} MICRO CONTRACTS")
    else:
        print("NO TRADE SIGNAL (NEUTRAL SKEW)")
    print("═"*60)

    # --- STATUS CHECK ---
    post_ib_bars = df[df['date'] == today_date].between_time('10:30', '23:59')
    if not post_ib_bars.empty and entry > 0:
        triggered = False
        if cur_bias == -1 and post_ib_bars['high'].max() >= entry: triggered = True
        if cur_bias == 1 and post_ib_bars['low'].min() <= entry: triggered = True
        
        if triggered:
            print("STATUS: ⚠️ ENTRY TRIGGERED (Price has already touched the level)")
        else:
            print("STATUS: ✅ RESTING (Price has not touched the entry level yet)")
    elif entry > 0:
        print("STATUS: 🕒 WAITING (Initial Balance still forming or no post-IB data)")
    else:
        print("STATUS: 💤 STANDING ASIDE")

    mt5.shutdown()

if __name__ == "__main__":
    run_dry_run()
