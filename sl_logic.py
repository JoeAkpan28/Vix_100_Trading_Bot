"""
Standalone Stop Loss (SL) Logic Module
=======================================

This module provides reusable Stop Loss calculation logic that can be integrated 
into any trading strategy or analysis script.

Features:
- Calculate entry-side wicks (volatility noise)
- Compute rolling 95% confidence buffer
- Generate dynamic SL prices based on trade direction
- Minimal dependencies (pandas, numpy)

Usage:
------
from sl_logic import StopLossCalculator

# Initialize
sl_calc = StopLossCalculator(dataframe=df, window_size=100, pip_size=1.0)

# Calculate
df = sl_calc.calculate_all(df)

# Or use individual methods:
df = sl_calc.calculate_entry_wick(df)
df = sl_calc.calculate_sl_buffer(df)
df = sl_calc.calculate_sl_price(df, prediction_col='prediction')
"""

import pandas as pd
import numpy as np


class StopLossCalculator:
    """
    A modular class for calculating dynamic stop loss levels.
    
    Parameters:
    -----------
    window_size : int, default=100
        Rolling window size for 95th percentile calculation
    pip_size : float, default=1.0
        Pip size for your asset (0.0001 for FX, 1.0 for indices, etc.)
    percentile : float, default=0.95
        Confidence level for SL buffer (0.95 = 95th percentile)
    """
    
    def __init__(self, window_size=100, pip_size=1.0, percentile=0.95):
        self.window_size = window_size
        self.pip_size = pip_size
        self.percentile = percentile
        
    def calculate_entry_wick(self, df, open_col='Open', close_col='Close', 
                           high_col='High', low_col='Low', bullish_col='is_bullish'):
        """
        Calculate entry-side wicks (the noise that threatens SL).
        
        For Bullish candles: entry-side = Open - Low (lower wick)
        For Bearish candles: entry-side = High - Open (upper wick)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            OHLC dataframe
        open_col, close_col, high_col, low_col : str
            Column names for OHLC data
        bullish_col : str
            Column name identifying bullish candles (boolean)
            
        Returns:
        --------
        df : pandas.DataFrame
            Dataframe with added 'entry_wick' column
        """
        df = df.copy()
        
        # Ensure bullish column exists
        if bullish_col not in df.columns:
            df[bullish_col] = df[close_col] >= df[open_col]
        
        # Calculate entry-side wick
        df['entry_wick'] = np.where(
            df[bullish_col],
            df[open_col] - df[low_col],      # Bullish: lower wick
            df[high_col] - df[open_col]      # Bearish: upper wick
        )
        
        return df
    
    def calculate_sl_buffer(self, df, entry_wick_col='entry_wick'):
        """
        Calculate rolling SL buffer using percentile of entry wicks.
        
        This represents the 95% confidence interval for volatility noise,
        which ensures your SL won't get stopped out by normal market jitter.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe with entry_wick column
        entry_wick_col : str
            Column name for entry wicks
            
        Returns:
        --------
        df : pandas.DataFrame
            Dataframe with added 'sl_buffer' column
        """
        df = df.copy()
        
        # Calculate rolling percentile
        df['sl_buffer'] = df[entry_wick_col].rolling(
            window=self.window_size
        ).quantile(self.percentile)
        
        # Fill NaN values at start (forward fill, then backward fill)
        df['sl_buffer'] = df['sl_buffer'].ffill().bfill()
        
        return df
    
    def calculate_sl_price(self, df, prediction_col='prediction', 
                          open_col='Open', sl_buffer_col='sl_buffer'):
        """
        Calculate dynamic SL price based on prediction direction.
        
        For Long (prediction=1): SL = Open - sl_buffer
        For Short (prediction=-1): SL = Open + sl_buffer
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe with prediction and sl_buffer columns
        prediction_col : str
            Column with predictions (1=Long, -1=Short, 0=No trade)
        open_col : str
            Column name for Open price
        sl_buffer_col : str
            Column name for SL buffer
            
        Returns:
        --------
        df : pandas.DataFrame
            Dataframe with added 'sl_price' column
        """
        df = df.copy()
        
        df['sl_price'] = np.where(
            df[prediction_col] == 1,
            df[open_col] - df[sl_buffer_col],      # Long: SL below
            np.where(
                df[prediction_col] == -1,
                df[open_col] + df[sl_buffer_col],  # Short: SL above
                df[open_col]                        # No trade: SL at open
            )
        )
        
        return df
    
    def calculate_all(self, df, open_col='Open', close_col='Close',
                     high_col='High', low_col='Low', bullish_col='is_bullish',
                     prediction_col='prediction'):
        """
        Execute all SL calculations in sequence.
        
        This is a convenience method that runs all three steps:
        1. Calculate entry wicks
        2. Calculate SL buffer
        3. Calculate SL price
        
        Parameters:
        -----------
        df : pandas.DataFrame
            OHLC dataframe with Open, High, Low, Close columns
        open_col, close_col, high_col, low_col : str
            Column names for OHLC data
        bullish_col : str
            Column name for bullish indicator
        prediction_col : str
            Column name for predictions (1=Long, -1=Short)
            
        Returns:
        --------
        df : pandas.DataFrame
            Dataframe with all SL columns added
        """
        df = self.calculate_entry_wick(df, open_col, close_col, high_col, low_col, bullish_col)
        df = self.calculate_sl_buffer(df, entry_wick_col='entry_wick')
        df = self.calculate_sl_price(df, prediction_col, open_col, sl_buffer_col='sl_buffer')
        
        return df


# ============================================================================
# CONVENIENCE FUNCTIONS - Use these for quick integration
# ============================================================================

def apply_sl_logic(df, window_size=100, pip_size=1.0, 
                   open_col='Open', close_col='Close', 
                   high_col='High', low_col='Low',
                   prediction_col='prediction'):
    """
    Quick function to apply all SL logic in one call.
    
    Example:
    --------
    df = apply_sl_logic(df, window_size=100, pip_size=1.0)
    print(df[['Open', 'sl_buffer', 'sl_price']].head())
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OHLC trading data
    window_size : int
        Rolling window for buffer calculation
    pip_size : float
        Pip size for your asset
    open_col, close_col, high_col, low_col : str
        OHLC column names
    prediction_col : str
        Column with direction predictions
        
    Returns:
    --------
    df : pandas.DataFrame
        Enhanced dataframe with SL columns
    """
    calculator = StopLossCalculator(window_size=window_size, pip_size=pip_size)
    return calculator.calculate_all(df, open_col, close_col, high_col, low_col, 
                                   prediction_col=prediction_col)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of the SL logic with sample data.
    """
    print("=" * 70)
    print("STOP LOSS LOGIC - DEMONSTRATION")
    print("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 200
    
    sample_data = {
        'Open': 1300 + np.cumsum(np.random.randn(n_samples) * 2),
        'Close': 1300 + np.cumsum(np.random.randn(n_samples) * 2),
        'High': 1310 + np.cumsum(np.random.randn(n_samples) * 2),
        'Low': 1290 + np.cumsum(np.random.randn(n_samples) * 2),
        'prediction': np.random.choice([1, -1, 0], size=n_samples, p=[0.4, 0.4, 0.2])
    }
    
    df_sample = pd.DataFrame(sample_data)
    
    # Method 1: Using convenience function
    print("\nMethod 1: Using convenience function")
    print("-" * 70)
    df_result = apply_sl_logic(df_sample, window_size=100, pip_size=1.0)
    
    print(f"Dataframe shape: {df_result.shape}")
    print(f"\nNew columns added:")
    print(f"  - entry_wick: Entry-side volatility noise")
    print(f"  - sl_buffer: 95% rolling confidence interval")
    print(f"  - sl_price: Dynamic SL level based on prediction")
    
    print(f"\nSample output:")
    print(df_result[['Open', 'Close', 'prediction', 'entry_wick', 'sl_buffer', 'sl_price']].head(10))
    
    print(f"\nStatistics:")
    print(f"  Average SL Buffer: {df_result['sl_buffer'].mean():.4f}")
    print(f"  Min SL Buffer: {df_result['sl_buffer'].min():.4f}")
    print(f"  Max SL Buffer: {df_result['sl_buffer'].max():.4f}")
    
    # Method 2: Using the class directly for more control
    print("\n" + "=" * 70)
    print("Method 2: Using StopLossCalculator class for custom control")
    print("-" * 70)
    
    calculator = StopLossCalculator(window_size=50, pip_size=1.0, percentile=0.90)
    df_result2 = calculator.calculate_all(df_sample)
    
    print(f"Custom settings: 50-period window, 90th percentile")
    print(f"Average SL Buffer: {df_result2['sl_buffer'].mean():.4f}")
    print(f"Total trades with SL: {(df_result2['prediction'] != 0).sum()}")
    
    print("\n" + "=" * 70)
    print("Ready to integrate into your trading strategy!")
    print("=" * 70)
