## Plot 1: Raw Value Distribution (Top Left)

**What it shows:** Histogram of all raw `u` values across all samples before normalization.

**Key elements:**

- Red dashed line: The computed mean (μ)
- Orange dashed lines: One standard deviation away from mean (μ±σ)
- Y-axis is logarithmic to show the tail distribution

**What to look for:**

- Is the distribution roughly Gaussian (bell-shaped)?
- Are there extreme outliers far from the mean?
- Does σ capture the spread of most values?

## Plot 2: Normalized Value Distribution (Top Middle)

**What it shows:** Histogram of all values after normalization: `(u - μ) / σ`

**Key elements:**

- Red dashed line: Zero (the normalized mean)
- Orange dashed lines: ±3σ in normalized space
- Y-axis is logarithmic

**What to look for:**

- Ideally, most values should fall within ±3σ (99.7% for Gaussian)
- If you see significant mass beyond ±3σ, your normalization may not be working well
- Heavy tails indicate extreme values that might cause training issues

## Plot 3: Per-Sample Range Distribution (Top Right)

**What it shows:** Histogram of the range (max - min) for each individual sample.

**Key elements:**

- Red dashed line: Median range across all samples
- X-axis: Range magnitude
- Y-axis: Number of samples with that range

**What to look for:**

- How spread out are the sample ranges?
- Are most samples similar in scale, or highly variable?
- Outlier samples with extremely large ranges may be problematic

## Plot 4: Normalized Min/Max Scatter (Bottom Left)

**What it shows:** Each point represents one sample, plotting its normalized min vs normalized max.

**Key elements:**

- Orange dashed lines: ±3σ boundaries forming a box
- Each dot is one sample
- Diagonal pattern expected (larger max usually means more negative min)

**What to look for:**

- Points inside the orange box: normalized values stay within ±3σ ✅
- Points outside: extreme samples that go beyond typical range ⚠️
- How many samples escape the "typical" region?

## Plot 5: CDF of Absolute Normalized Values (Bottom Middle)

**What it shows:** Cumulative Distribution Function - what percentage of all normalized values fall below a given threshold.

**Key elements:**

- X-axis: Absolute value of normalized data (|σ|)
- Y-axis: Cumulative percentage (0-100%)
- Vertical lines at 3σ, 5σ, 10σ for reference

**What to look for:**

- At 3σ: Should see ~99.7% for normal distribution
- Steep rise = most values concentrated near zero (good)
- Long tail = many extreme values (potentially problematic)
- Example: If CDF at 5σ is 95%, then 5% of values are beyond ±5σ

## Plot 6: Sample Range vs Index (Bottom Right)

**What it shows:** Scatter plot of sample range against sample index (order in dataset).

**Key elements:**

- X-axis: Sample number (0 to N)
- Y-axis: Range of that sample
- Each dot is one sample

**What to look for:**

- Any patterns or trends? (shouldn't be any - should look random)
- Are extreme-range samples clustered together or scattered?
- Helps verify that samples are well-shuffled
- If you see stripes or patterns, it might indicate non-random generation

---

## Summary

These 6 plots together tell you:

1. **Raw distribution** - what your actual data looks like
2. **Normalized distribution** - how well normalization works
3. **Per-sample ranges** - variability between samples
4. **Min/max scatter** - which samples are extreme
5. **CDF** - quantitative measure of tail behavior
6. **Index plot** - sanity check for randomness

The key insight from your analysis was that plots 2, 4, and 5 showed that normalization was failing - with values reaching ±240σ instead of staying within ±3σ, which is why your model struggled with large-amplitude functions!
