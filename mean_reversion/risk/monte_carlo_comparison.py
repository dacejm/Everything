import random
import matplotlib.pyplot as plt
import os

# ═════════════════════════════════════════════════════════════════════════════
# 1. HARDCODED SIMULATION PARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
ACCOUNT_SIZE = 50000
PROFIT_TARGET = 3000
MAX_DRAWDOWN = 2500
SIMULATIONS = 10000
MAX_TRADES = 5000

# NORMAL METRICS (Standard 100% TP logic)
N_WIN_RATE = 0.55   
N_AVG_WIN = 350.00
N_AVG_LOSS = 500.00

# RUNNER METRICS (NQ Runner Optimized Backtest)
R_WIN_RATE = 0.642
R_AVG_WIN = 482.79
R_AVG_LOSS = 500.00

# ═════════════════════════════════════════════════════════════════════════════
# 2. THE MONTE CARLO ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(win_rate, avg_win, avg_loss):
    """
    Simulates a Prop Firm evaluation with a Trailing Drawdown.
    Uses simple Python loops for maximum readability.
    """
    passed_count = 0
    total_trades_to_pass = 0
    equity_curves_to_plot = [] # Store first 100
    
    for s in range(SIMULATIONS):
        current_balance = ACCOUNT_SIZE
        peak_balance = ACCOUNT_SIZE
        history = [current_balance]
        status = "FAILED" # Default
        
        for t in range(MAX_TRADES):
            # 1. Simulate Trade Outcome
            if random.random() < win_rate:
                current_balance += avg_win
            else:
                current_balance -= avg_loss
            
            history.append(current_balance)
            
            # 2. Update Peak for Trailing Drawdown
            if current_balance > peak_balance:
                peak_balance = current_balance
            
            # 3. Check Failure (Trailing Drawdown)
            if (peak_balance - current_balance) >= MAX_DRAWDOWN:
                status = "FAILED"
                break
                
            # 4. Check Success (Profit Target)
            if current_balance >= (ACCOUNT_SIZE + PROFIT_TARGET):
                status = "PASSED"
                passed_count += 1
                total_trades_to_pass += (t + 1)
                break
        
        # Keep track of sample curves for plotting
        if s < 100:
            equity_curves_to_plot.append({'history': history, 'status': status})
            
    pass_rate = (passed_count / SIMULATIONS) * 100
    avg_trades = total_trades_to_pass / passed_count if passed_count > 0 else 0
    
    return pass_rate, avg_trades, equity_curves_to_plot

# ═════════════════════════════════════════════════════════════════════════════
# 3. EXECUTION & VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def execute():
    print("Running Monte Carlo Simulations (10,000 iterations each)...")
    
    # Run Normal
    n_pass, n_trades, n_curves = run_monte_carlo(N_WIN_RATE, N_AVG_WIN, N_AVG_LOSS)
    print(f"\nNORMAL LOGIC:")
    print(f"Pass Rate: {n_pass:.2f}%")
    print(f"Avg Trades to Pass: {n_trades:.1f}")
    
    # Run Runner
    r_pass, r_trades, r_curves = run_monte_carlo(R_WIN_RATE, R_AVG_WIN, R_AVG_LOSS)
    print(f"\nRUNNER LOGIC:")
    print(f"Pass Rate: {r_pass:.2f}%")
    print(f"Avg Trades to Pass: {r_trades:.1f}")
    
    # Create Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Normal Logic
    for item in n_curves:
        color = 'green' if item['status'] == "PASSED" else 'red'
        ax1.plot(item['history'], color=color, alpha=0.3, linewidth=1)
    ax1.axhline(ACCOUNT_SIZE + PROFIT_TARGET, color='blue', linestyle='--', label='Target')
    ax1.axhline(ACCOUNT_SIZE - MAX_DRAWDOWN, color='black', linestyle='--', label='Initial Floor')
    ax1.set_title(f"Normal Logic (Pass Rate: {n_pass:.1f}%)")
    ax1.set_ylabel("Account Balance ($)")
    ax1.set_xlabel("Number of Trades")
    
    # Plot Runner Logic
    for item in r_curves:
        color = 'green' if item['status'] == "PASSED" else 'red'
        ax2.plot(item['history'], color=color, alpha=0.3, linewidth=1)
    ax2.axhline(ACCOUNT_SIZE + PROFIT_TARGET, color='blue', linestyle='--')
    ax2.axhline(ACCOUNT_SIZE - MAX_DRAWDOWN, color='black', linestyle='--')
    ax2.set_title(f"Runner Logic (Pass Rate: {r_pass:.1f}%)")
    ax2.set_xlabel("Number of Trades")
    
    plt.tight_layout()
    plt.savefig("prop_firm_monte_carlo_comparison.png")
    print("\nComparison chart saved as: prop_firm_monte_carlo_comparison.png")

if __name__ == "__main__":
    execute()
