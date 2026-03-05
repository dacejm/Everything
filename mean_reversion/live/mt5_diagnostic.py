import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
SYMBOL = "MNQH26"  # Target symbol
NUM_BARS = 50000   # Request depth
TIMEFRAME = mt5.TIMEFRAME_M1

def run_diagnostic():
    print(f"[{datetime.now()}] --- STARTING MT5 DATA DIAGNOSTIC ---")

    # 1. Initialize MT5 Connection
    if not mt5.initialize():
        print(f"CRITICAL: mt5.initialize() failed.")
        print(f"Error Code: {mt5.last_error()}")
        return

    # 2. Check Terminal Info
    terminal_info = mt5.terminal_info()
    if terminal_info is not None:
        print(f"Connected to Terminal: {terminal_info.company}")
        print(f"Trade Allowed: {terminal_info.trade_allowed}")
    else:
        print("WARNING: Could not retrieve terminal info.")

    # 3. Symbol Selection (Crucial for copy_rates)
    if not mt5.symbol_select(SYMBOL, True):
        print(f"ERROR: Failed to select symbol '{SYMBOL}'.")
        print(f"Check spelling or ensure the contract is not expired.")
        print(f"Internal Error: {mt5.last_error()}")
        mt5.shutdown()
        return
    else:
        print(f"SUCCESS: Symbol '{SYMBOL}' is active in Market Watch.")

    # 4. Attempt Data Fetch
    print(f"Attempting to fetch last {NUM_BARS} bars of M1 data...")
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, NUM_BARS)

    # 5. Error Handling & Verbose Output
    if rates is None:
        error_code, error_desc = mt5.last_error()
        print("\nFETCH FAILED!")
        print(f"MT5 Error Code: {error_code}")
        
        # Mapping common silent failure codes
        if error_code == -5:
            print("Description: ERR_INVALID_PARAMS (Likely timeframe or count error)")
        elif error_code == -4:
            print("Description: ERR_NO_MEMORY (Terminal out of RAM)")
        elif error_code == -1:
            print("Description: ERR_INTERNAL_ERROR (Terminal communication failure)")
        else:
            print(f"Description: Internal MT5 Error occurred.")
    else:
        # 6. Success: Process and Report
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print("\nFETCH SUCCESSFUL!")
        print(f"Total Rows Retrieved: {df.shape[0]}")
        print(f"Columns: {list(df.columns)}")
        
        if not df.empty:
            start_time = df['time'].iloc[0]
            end_time = df['time'].iloc[-1]
            print(f"History Start: {start_time}")
            print(f"History End:   {end_time}")
            print(f"Total Span:    {end_time - start_time}")
            
            print("\nFirst 5 Rows Sample:")
            print(df.head())
        else:
            print("WARNING: Request returned an empty array.")

    # Shutdown
    mt5.shutdown()
    print(f"\n[{datetime.now()}] --- DIAGNOSTIC COMPLETE ---")

if __name__ == "__main__":
    run_diagnostic()
