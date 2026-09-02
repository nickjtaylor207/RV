## Framework for Event Premium Analysis


### 1. **Establish the Baseline (Non-Event Vol)**

You need to determine what overnight vol *should be* without the event. Three approaches:

**A. Historical Average Method**
- Take average O/N vol for the same currency pair over past 30-60 days
- Exclude other known event days
- **Pro**: Simple, intuitive
- **Con**: Doesn't account for current regime

**B. Vol Surface Interpolation**
- Use 1W and 1M implied vols to interpolate an "event-free" overnight vol
- Account for the fact that 1W contains ~5 O/N periods
- **Formula**: `Baseline_ON = sqrt((1W_vol² × 7 - EventON_vol² × 1) / 6)`
- **Pro**: Reflects current vol regime
- **Con**: Requires solving for the unknown

**C. Weighted Combo**
- Blend historical average with regime-adjusted estimate
- Weight more recent data higher
- **Pro**: Balanced approach
- **Con**: More parameters to tune

**My preference**: **Method B** (interpolation) is most theoretically sound, but I'd validate against Method A.

---

### 2. **Calculate Event Premium Components**

Once you have baseline, calculate:

```python
# Implied Event Premium (what market prices in)
Event_Premium_Implied = EventON_Vol - Baseline_ON_Vol
Event_Premium_Implied_pct = (EventON_Vol / Baseline_ON_Vol - 1) × 100

# Realized Event Vol (what actually happened)
Realized_Event_Vol = calculate_realized_vol(price_data, 5pm_to_930am)

# Actual Event Impact
Event_Impact = Realized_Event_Vol - Baseline_ON_Vol

# Premium vs Reality (the key metric!)
Premium_vs_Realized = Event_Premium_Implied - Event_Impact
Premium_Ratio = Event_Premium_Implied / Event_Impact
```

---

### 3. **Key Metrics to Track**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Event Premium** | Implied - Baseline | How much extra vol is priced in |
| **Realization Ratio** | Realized / Implied | Did event deliver on implied vol? |
| **Premium Efficiency** | (Implied - Realized) / Implied | How much of premium was wasted |
| **Surprise Adjustment** | (Realized - Implied) / |Surprise| | Does surprise magnitude explain miss? |

---

### 4. **Statistical Analysis Over Time**

For each event type (US CPI, NFP, FOMC, etc.), track:

**A. Systematic Mispricing?**
```python
# Across N events:
avg_premium_vs_realized = mean(Event_Premium_Implied - Event_Impact)

if avg_premium_vs_realized > 0:
    print("Market OVERPRICES this event on average")
    # Potential trade: SELL event vol
elif avg_premium_vs_realized < 0:
    print("Market UNDERPRICES this event")
    # Potential trade: BUY event vol
```

**Statistical test**: t-test on whether mean(Premium - Realized) ≠ 0

**B. Regime Dependence**
- Does overpricing occur more in high/low vol regimes?
- Does it depend on surprise magnitude?
- Time trends (has market become more/less efficient?)

**C. Cross-Currency Patterns**
- Which pairs show most systematic mispricing?
- Is EUR/USD event premium more accurate than AUD/USD?

---

### 5. **My Interpretation Framework**

**Scenario 1: Market Overprices Event (Premium > Realized)**
- **Implied O/N**: 15% annualized
- **Baseline**: 8% annualized  
- **Realized**: 10% annualized
- **Event Premium**: 15 - 8 = 7%
- **Actual Impact**: 10 - 8 = 2%
- **Waste**: 5% (71% of premium wasted!)

**Interpretation**: Market is pricing in more movement than actually occurs. Systematic opportunity to **sell O/N straddles before CPI**.

**Scenario 2: Market Underprices Event (Premium < Realized)**
- **Implied O/N**: 12%
- **Baseline**: 8%
- **Realized**: 18%
- **Event Premium**: 4%
- **Actual Impact**: 10%
- **Shortfall**: -6%

**Interpretation**: Market systematically underestimates this event. Opportunity to **buy O/N straddles**.

**Scenario 3: Efficient Pricing (Premium ≈ Realized)**
- **Implied O/N**: 14%
- **Baseline**: 8%
- **Realized**: 15%
- **Event Premium**: 6%
- **Actual Impact**: 7%
- **Difference**: -1%

**Interpretation**: Market is fairly efficient. No systematic edge, but might trade on individual surprises.

---

### 6. **Conditional Analysis (Most Important!)**

Don't just look at averages - condition on:

**A. Surprise Magnitude**
```python
if abs(actual - consensus) > 1.5 × std_dev:
    # Large surprise events
    avg_realized_large_surprise = ?
elif abs(actual - consensus) < 0.5 × std_dev:
    # Small surprise events  
    avg_realized_small_surprise = ?
```

**Key insight**: Market might overprice events that end up being "in line" but efficiently price events with large surprises.

**B. Pre-Event Vol Regime**
- High vol regime (VIX > 20): Does market overprice even more?
- Low vol regime (VIX < 15): Does market underprice tail risk?

**C. Time of Day Effects**
- Does 8:30am EST timing matter vs. 10:00am data?
- European morning positioning effects?

---

### 7. **Trading Signal Generation**

Based on historical analysis:

```python
# Example thresholds from backtesting
if historical_avg_premium_vs_realized > 2% and statistical_significance > 95%:
    signal = "SELL overnight straddle before event"
    expected_edge = historical_avg_premium_vs_realized
    
# Position sizing based on confidence
position_size = base_size × (statistical_significance / 100) × min(1, vol_of_edge / expected_edge)
```

---

### 8. **Visualization for Analysis**

**Chart 1: Premium vs Realized Scatter**
- X-axis: Event Premium Implied
- Y-axis: Actual Event Impact
- 45° line = perfect pricing
- Points above line = underpriced
- Points below line = overpriced

**Chart 2: Distribution of Premium Efficiency**
```
Histogram of (Implied - Realized) / Implied
- Centered at 0 = efficient market
- Centered above 0 = systematic overpricing
- Fat tails = occasional large surprises
```

**Chart 3: Time Series**
- Rolling average of Premium - Realized
- Shows if market is learning/adapting

---

## Summary: What I'd Implement

1. **Calculate Baseline** using vol surface interpolation (validated against historical avg)
2. **Measure 3 windows** of realized vol: full O/N, event-focused (12am-9:30am), narrow (8-9am)
3. **Track 4 key metrics**: Event Premium, Realization Ratio, Premium Efficiency, Vol Risk Ratio
4. **Statistical testing** across 50+ events to find systematic patterns
5. **Conditional analysis** by surprise, regime, cross-currency
6. **Backtest trading strategy**: Sell event vol when historical data shows systematic overpricing

The **gold standard metric** is:
```
Vol_Risk_Ratio = (Implied - Realized) / Baseline
```
This tells you "per unit of normal volatility, how much premium was wasted/earned."

If Vol_Risk_Ratio is consistently positive (>0.3 with t-stat > 2), you have a systematic edge to sell event vol.

Would you like me to help code up this framework for your FXEventStatistics class?









# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------




