"""
STANDALONE MARKOV PREDICTOR (CORRECTED - 6 STATE)
==================================================
This class is "Live Ready" - it only needs the last 100 candles to calculate
rolling thresholds and the last 3 states to make a prediction.

🔧 KEY FIX: Now uses 6-state encoding (0-5) to match markov_brain.json
   - States 0-5: All possible outcomes (not 6-9)
   - State 4 and 5 typically represent UP moves
   - This matches the actual Brain structure

Key Features:
- Zero data leakage: Quantiles calculated per-window only
- Portable: Works with markov_brain.json and any OHLC data
- Modular: Copy this class into any trading bot or backtest framework
- Traceable: Logs exact sequences that triggered each prediction

Usage:
    predictor = MarkovLivePredictor('markov_brain.json')
    current_state = predictor.get_current_state(window_df)
    signal = predictor.predict(tuple(state_queue))
"""

import pandas as pd
import numpy as np
import json


class MarkovLivePredictor:
    """
    Standalone Markov Chain predictor using 3rd-order states.
    
    The "Brain" (transition probabilities) is loaded from markov_brain.json.
    The "Processing" (state encoding) happens in get_current_state().
    
    🔧 CORRECTED VERSION:
    - Brain uses states 0-5 (6-state system)
    - Up states are now 4 and 5 (not 6,7,8,9)
    """
    
    def __init__(self, matrix_path='markov_brain.json', up_threshold=0.50, conf_up=0.15, conf_down=0.40):
        """
        Initialize the predictor with a pre-computed transition matrix.
        
        Parameters
        ----------
        matrix_path : str
            Path to markov_brain.json (the exported transition matrix)
        up_threshold : float
            Threshold for UP/DOWN decision (default: 0.50, range: 0.0-1.0)
            Combined with confidence offsets to determine signal thresholds
        conf_up : float
            Confidence offset for UP signals (default: 0.15)
            UP signal generated when: up_probability > (up_threshold + conf_up)
            Example: 0.5 + 0.15 = 0.65 (requires 65% probability for UP signal)
        conf_down : float
            Confidence offset for DOWN signals (default: 0.40)
            DOWN signal generated when: up_probability < (up_threshold - conf_down)
            Example: 0.5 - 0.40 = 0.10 (requires 10% or less probability for DOWN signal)
        """
        
        # Load the transition matrix
        with open(matrix_path, 'r') as f:
            raw_matrix = json.load(f)
            # Keys are strings like "(0,0,0)" from JSON
            self.matrix = raw_matrix
        
        self.up_threshold = up_threshold
        self.conf_up = conf_up
        self.conf_down = conf_down
        
        # Calculate effective thresholds
        self.threshold_up = self.up_threshold + self.conf_up      # e.g., 0.65
        self.threshold_down = self.up_threshold - self.conf_down   # e.g., 0.10
        
        # 🔧 CORRECTED: States 4, 5 represent UP moves (not 6,7,8,9)
        # These are "Small Up" and "Big Up" in the 6-state system
        self.up_states = ['4', '5']
        
        print(f"✅ MarkovLivePredictor initialized (CORRECTED - 6 STATE)")
        print(f"   Matrix loaded: {len(self.matrix)} sequences")
        print(f"   UP states: {self.up_states}")
        print(f"   UP threshold: {self.threshold_up:.2f} (up_prob >= {self.threshold_up})")
        print(f"   DOWN threshold: {self.threshold_down:.2f} (up_prob < {self.threshold_down})")
    
    def get_current_state(self, window_df):
        """
        Encode the most recent candle into a State ID (0-5).
        
        🔧 CORRECTED: Now uses 6-state logic (was previously using 10-state)
        
        Parameters
        ----------
        window_df : pd.DataFrame
            DataFrame with columns: open, high, low, close
            Should contain ~100 recent candles (rolling window)
            Row order: oldest to newest (most recent last)
        
        Returns
        -------
        state_id : float
            The encoded state (0-5) for the most recent candle
            
        Logic
        -----
        STATE ENCODING (6-state system):
            0: Big Down move
            1: Small Down move
            2: Flat/Neutral move
            3: Flat/Neutral move (variant)
            4: Small Up move     ← UP signal source
            5: Big Up move       ← UP signal source
        
        Previous (WRONG) 10-state logic:
            0-1: Big Down, 2-3: Small Down, 4-5: Flat, 6-7: Small Up, 8-9: Big Up
            ✗ This caused UNKNOWN predictions (85% of sequences not found)
        """
        
        # Calculate features for the window
        returns = np.log(window_df['close'] / window_df['close'].shift(1))
        
        # Get the current candle's values (most recent = last row)
        curr_ret = returns.iloc[-1]
        
        # Get rolling quantiles/thresholds from this window ONLY
        # (This prevents data leakage)
        # 🔧 FIX: Now uses 6 quantile boundaries (0.2, 0.4, 0.5, 0.6, 0.8) 
        # to properly encode all 6 states (0-5) without skipping state 3
        q20 = returns.quantile(0.20)
        q40 = returns.quantile(0.40)
        q50 = returns.quantile(0.50)  # ADDED: Median to split flat region
        q60 = returns.quantile(0.60)
        q80 = returns.quantile(0.80)
        
        # 🔧 CORRECTED: 6-state magnitude-only encoding
        # (We removed the body-ratio structure component)
        if curr_ret < q20:
            state = 0.0      # Big Down
        elif curr_ret < q40:
            state = 1.0      # Small Down
        elif curr_ret < q50:
            state = 2.0      # Flat (lower half)
        elif curr_ret < q60:
            state = 3.0      # Flat (upper half) - FIX: was skipping this state
        elif curr_ret < q80:
            state = 4.0      # Small Up
        else:
            state = 5.0      # Big Up
        
        return state
    
    def predict(self, sequence, verbose=False):
        """
        Generate UP/DOWN prediction for a given sequence of states.
        
        Uses asymmetrical confidence thresholds:
        - UP signal: up_probability > threshold_up (default 0.65)
        - DOWN signal: up_probability < threshold_down (default 0.10)
        - UNKNOWN: probability falls between thresholds (neutral zone)
        
        Parameters
        ----------
        sequence : tuple
            Tuple of 3 states, e.g., (5.0, 4.0, 0.0) representing
            [state_t-2, state_t-1, state_t]
        verbose : bool
            If True, print prediction details
        
        Returns
        -------
        result : dict
            {
                'signal': 'UP' or 'DOWN' or 'UNKNOWN',
                'up_probability': float (0.0-1.0),
                'confidence': float (0.0-0.5),
                'threshold_used': 'conf_up' or 'conf_down' or 'none'
            }
        """
        
        # Convert sequence to string key for lookup
        # e.g., (5.0, 4.0, 0.0) -> "(5,4,0)"
        seq_key = f"({int(sequence[0])},{int(sequence[1])},{int(sequence[2])})"
        
        # Check if this sequence exists in the matrix
        if seq_key not in self.matrix:
            return {
                "signal": "UNKNOWN",
                "up_probability": 0.5,
                "confidence": 0.0,
                "threshold_used": "none"
            }
        
        # Get next-state probabilities for this sequence
        probs = self.matrix[seq_key]
        
        # 🔧 CORRECTED: Sum probabilities for UP states (4, 5 only)
        up_prob = 0.0
        for state in self.up_states:
            if state in probs:
                up_prob += probs[state]
        
        # Confidence = distance from 0.5 (neutral)
        # Range: 0.0 (50/50) to 0.5 (100% certain)
        confidence = abs(up_prob - 0.5)
        
        # 🔧 NEW: Apply asymmetrical thresholds
        # UP signal if up_probability > threshold_up (e.g., > 0.65)
        if up_prob >= self.threshold_up:
            signal = "UP"
            threshold_used = "conf_up"
        # DOWN signal if up_probability < threshold_down (e.g., < 0.10)
        elif up_prob < self.threshold_down:
            signal = "DOWN"
            threshold_used = "conf_down"
        # UNKNOWN: probability falls in neutral zone
        else:
            signal = "UNKNOWN"
            threshold_used = "none"
        
        if verbose:
            print(f"\n📊 Prediction for sequence {seq_key}:")
            print(f"   UP probability (states 4+5): {up_prob:.4f}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   Thresholds: UP > {self.threshold_up:.2f}, DOWN < {self.threshold_down:.2f}")
            print(f"   Signal: {signal} (using {threshold_used})")
        
        return {
            "signal": signal,
            "up_probability": up_prob,
            "confidence": confidence,
            "threshold_used": threshold_used
        }


# ============================================================================
# EXAMPLE: LIVE TRADING USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize predictor with asymmetrical confidence thresholds
    predictor = MarkovLivePredictor('markov_brain.json', up_threshold=0.5, conf_up=0.15, conf_down=0.40)
    
    print("\n" + "="*70)
    print("EXAMPLE: ASYMMETRICAL CONFIDENCE THRESHOLDS")
    print("="*70)
    
    print("\n✅ Ready to use with asymmetrical confidence thresholds!")
    print("\n   Default thresholds:")
    print("   - UP signal:   up_probability > 0.65 (conf_up=0.15)")
    print("   - DOWN signal: up_probability < 0.10 (conf_down=0.40)")
    print("   - UNKNOWN:     probability between 0.10 and 0.65")
    print("\n   Usage:")
    print("   1. Fetch last 100 candles: df_window = fetch_from_broker(count=100)")
    print("   2. Get current state: state = predictor.get_current_state(df_window)")
    print("   3. Maintain state queue: state_queue = [s1, s2, s3] (last 3 states)")
    print("   4. Predict: result = predictor.predict(tuple(state_queue))")
    
    # Example with dummy data
    print("\n" + "="*70)
    print("TEST: With dummy OHLC data")
    print("="*70)
    
    # Create a simple window of 100 candles
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='H')
    close = 1000 + np.random.randn(100).cumsum()
    open_ = close + np.random.randn(100) * 0.5
    high = np.maximum(close, open_) + np.abs(np.random.randn(100)) * 0.5
    low = np.minimum(close, open_) - np.abs(np.random.randn(100)) * 0.5
    
    test_window = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close
    }, index=dates)
    
    # Get the state
    state = predictor.get_current_state(test_window)
    print(f"\n   State for dummy window: {state}")
    print(f"   ✅ Note: State is now 0-5, not 0-9")
    
    # Test prediction with a sequence
    test_sequence = (4.0, 5.0, 0.0)  # Changed from (7.0, 5.0, 0.0)
    result = predictor.predict(test_sequence, verbose=True)
