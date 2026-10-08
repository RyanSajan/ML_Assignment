# Polynomial Regression

Predicts a continuous target `y` for two datasets (`var1`, `var2`) using polynomial regression with regularization.

## Approach

- Pipeline: `PolynomialFeatures` → `StandardScaler` → Linear / Ridge / Lasso / ElasticNet
- Degrees searched: 1–10 (var1), 1–20 (var2)
- Best degree and hyperparameters picked by lowest 5-fold CV MSE, then refit on the full training set
- Lasso/ElasticNet only tried at lower degrees (≤ 6 for var1, ≤ 9 for var2)

## Results

| Dataset | Degree | Model | Alpha | CV MSE | CV R² |
|---------|--------|-------|-------|--------|-------|
| var1 | 5 | Lasso | 0.01 | 0.33936 | 0.96652 |
| var2 | 10 | Ridge | 1 | 0.22769 | 0.99513 |

## Files

- `model.py`: full pipeline
- `data/`: train/test CSVs (`IMT2024036_{train,test}_var{1,2}.csv`)
- `output/`: test predictions, per-degree CV results, and CV MSE vs degree plots

## Usage

```bash
pip install numpy pandas matplotlib scikit-learn
python model.py
```
