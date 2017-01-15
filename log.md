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

4. Filtered out some values (more agressively, < 1e-2) and changed scorer to rsme

I don't expect the change of the scorer to matter much, so filtering too many features
is more likely to have caused the score to go down.

Score: 0.16624

5. Less filtering (< 1e-4). Slight score improvement.

Score: 0.15899

6. Even less filtering (< 1-e5). Score got worse.

Score: 0.16345

7. Added a deep learning model, with 128 neurons. I deliberately tried to overfit so I knew where to reduce from.
And indeed I did!

Score: 0.53179
