import pandas as pd
import matplotlib.pyplot as plt

# Load the backtest results
df = pd.read_csv('YOUR_BACKTEST_RESULTS_FILE.csv')  # Replace with your results file

# Ensure we have a sequential index to simulate time steps
df.reset_index(drop=True, inplace=True)

# Filter for UP and DOWN signals (Long and Short trades)
# Based on the strategy file, we are interested in both LONG and SHORT signals
long_signals = df[df['signal'] == 'UP'].copy()
short_signals = df[df['signal'] == 'DOWN'].copy()

# Dictionary to store results
results = []

# We want to check the performance for holding 1 to 10 bars
holding_periods = list(range(1, 11))

for n_bars in holding_periods:
    # We look ahead 'n' bars to get the exit price
    # The 'shift(-n)' gets the value from n rows ahead
    # We align this with the original signal index
    
    # Calculate PnL for both Long and Short trades
    future_close = df['close'].shift(-n_bars)
    
    # Process LONG signals (BUY trades)
    long_signals[f'close_plus_{n_bars}'] = future_close.loc[long_signals.index]
    long_pnl_column = f'long_pnl_{n_bars}'
    # For LONG: Positive if Price Increases (Exit - Entry)
    long_signals[long_pnl_column] = long_signals[f'close_plus_{n_bars}'] - long_signals['close']
    
    # Process SHORT signals (SELL trades)
    short_signals[f'close_plus_{n_bars}'] = future_close.loc[short_signals.index]
    short_pnl_column = f'short_pnl_{n_bars}'
    # For SHORT: Positive if Price Decreases (Entry - Exit)
    short_signals[short_pnl_column] = short_signals['close'] - short_signals[f'close_plus_{n_bars}']
    
    # Aggregate stats for LONG trades
    long_avg_pnl = long_signals[long_pnl_column].mean()
    long_median_pnl = long_signals[long_pnl_column].median()
    long_win_rate = (long_signals[long_pnl_column] > 0).mean() * 100
    
    # Aggregate stats for SHORT trades
    short_avg_pnl = short_signals[short_pnl_column].mean()
    short_median_pnl = short_signals[short_pnl_column].median()
    short_win_rate = (short_signals[short_pnl_column] > 0).mean() * 100
    
    # Combined stats (both long and short)
    combined_pnl = pd.concat([long_signals[long_pnl_column], short_signals[short_pnl_column]])
    combined_avg_pnl = combined_pnl.mean()
    combined_median_pnl = combined_pnl.median()
    combined_win_rate = (combined_pnl > 0).mean() * 100
    
    results.append({
        'Bars Held': n_bars,
        'Long Avg PnL': long_avg_pnl,
        'Long Median PnL': long_median_pnl,
        'Long Win Rate (%)': long_win_rate,
        'Short Avg PnL': short_avg_pnl,
        'Short Median PnL': short_median_pnl,
        'Short Win Rate (%)': short_win_rate,
        'Combined Avg PnL': combined_avg_pnl,
        'Combined Median PnL': combined_median_pnl,
        'Combined Win Rate (%)': combined_win_rate
    })

# Create a DataFrame for the results
results_df = pd.DataFrame(results)

# Display the results table to console
print("\n" + "="*100)
print("Time-Based Exit Optimization (Both Long and Short Trades)")
print("="*100)
print(results_df.to_string(index=False))
print("="*100 + "\n")

# Plotting the decay curves for both long and short trades
plt.figure(figsize=(12, 8))

# Plot Long trades
plt.plot(results_df['Bars Held'], results_df['Long Avg PnL'], marker='o', linestyle='-', color='green', label='Long Avg PnL', linewidth=2)

# Plot Short trades
plt.plot(results_df['Bars Held'], results_df['Short Avg PnL'], marker='s', linestyle='-', color='red', label='Short Avg PnL', linewidth=2)

# Plot Combined
plt.plot(results_df['Bars Held'], results_df['Combined Avg PnL'], marker='^', linestyle='-', color='blue', label='Combined Avg PnL', linewidth=2)

plt.axhline(0, color='black', linewidth=1, linestyle='--')
plt.title('Signal Decay: Average PnL vs Bars Held (Long, Short, and Combined)')
plt.xlabel('Bars Held')
plt.ylabel('Average PnL (Price Points)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()