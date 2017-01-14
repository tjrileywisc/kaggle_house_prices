# Experiment log

1. Simple random forest regressor from sklearn, skewing from dataset not removed
Score: 0.16194


2. Simple random forest regressor from sklearn, skewing from dataset removed by
simple heuristics that choose from no transformation, scipy's boxcox1p(x), sqrt(x) and np.log1p(x).
Unskewing is done if the absolute value of the skew is greater than 1, and we choose the best
unskewing transformation. See [preprocess_data.py](preprocess_data.py) for more information.

Score: 0.1622

3. Filtered out some of the values with low feature importance scores (< 1e-3) and refit
Some improvment here so obviously some features need to be filtered out.

Score: 0.15864

