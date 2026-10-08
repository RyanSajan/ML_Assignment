# Polynomial Regression: ML Assignment 1

Predicts a continuous target $y$ for two datasets (`var1`, `var2`) using polynomial regression with regularization. The polynomial degree and regularization settings are chosen by 5-fold cross-validation (lowest MSE).

## Approach

1. **Polynomial expansion** of the input features up to degree $d$ (`PolynomialFeatures`)
2. **Standardization** of the expanded features (`StandardScaler`)
3. **Regression model**: Linear, Ridge, Lasso or ElasticNet
4. **Selection**: for each degree, the best model and hyperparameters are picked by 5-fold shuffled CV MSE. The overall best degree is the one with the lowest CV MSE.
5. **Final fit** on the full training set, then predict on the test set

## Models and Formulas

Let $\Phi(X)$ be the standardized polynomial feature matrix with $n$ samples, $w$ the coefficient vector, and $\alpha$ the regularization strength. 

**Polynomial expansion.** For features $x_1, \dots, x_p$ and degree $d$, all terms of the form

$$x_1^{k_1} x_2^{k_2} \cdots x_p^{k_p}, \quad 1 \le k_1 + k_2 + \dots + k_p \le d$$

are generated, including interaction terms.

**Linear Regression** (ordinary least squares)

$$\min_{w} \; \lVert y - \Phi w \rVert_2^2$$

**Ridge Regression** ($L_2$ penalty)

$$\min_{w} \; \lVert y - \Phi w \rVert_2^2 + \alpha \lVert w \rVert_2^2$$

**Lasso Regression** ($L_1$ penalty)

$$\min_{w} \; \frac{1}{2n} \lVert y - \Phi w \rVert_2^2 + \alpha \lVert w \rVert_1$$

**ElasticNet** (mix of $L_1$ and $L_2$, controlled by $\rho$ = `l1_ratio`)

$$\min_{w} \; \frac{1}{2n} \lVert y - \Phi w \rVert_2^2 + \alpha \rho \lVert w \rVert_1 + \frac{\alpha (1 - \rho)}{2} \lVert w \rVert_2^2$$

Setting $\rho = 1$ gives Lasso, and $\rho = 0$ gives a Ridge-like penalty.

## Results

| Dataset | Best degree | Model | $\alpha$ | CV MSE | CV $R^2$ |
|---------|-------------|-------|----------|--------|----------|
| var1 | 5 | Lasso | 0.01 | 0.33936 | 0.96652 |
| var2 | 10 | Ridge | 1 | 0.22769 | 0.99513 |

## Project Structure

```
.
├── model.py            
├── data/                 
└── output/                
    ├── IMT2024036_pred_var{1,2}.csv   
    ├── degree_search_var{1,2}.csv   
    └── optimal_degree_{1,2}.png      
```

## Usage

```bash
pip install numpy pandas matplotlib scikit-learn
python model.py
```
