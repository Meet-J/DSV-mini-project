# Feature ablation (cross-validated ROC-AUC, training split only)

5-fold stratified CV repeated 3 times, seed 42. `delta` is the paired mean AUC difference against the reference configuration (`vs`); `se` is its standard error (optimistic, see module docstring). Differences under about 0.002 are noise.

## Logistic Regression

| Configuration | Mean AUC | Std | vs | Delta AUC | SE |
|---|---|---|---|---|---|
| A base features only | 0.8462 | 0.0121 |  |  |  |
| B A + TenureGroup | 0.8465 | 0.0122 | A | +0.0004 | 0.0005 |
| C A + ProtectiveAddOnCount | 0.8462 | 0.0121 | A | +0.0000 | 0.0000 |
| D A + AutoPay | 0.8461 | 0.0121 | A | -0.0000 | 0.0000 |
| E A + all three engineered | 0.8465 | 0.0122 | A | +0.0004 | 0.0005 |
| F A - five low-signal columns | 0.8441 | 0.0115 | A | -0.0021 | 0.0006 |
| G A - gender | 0.8465 | 0.0121 | A | +0.0004 | 0.0001 |
| H A - TotalCharges | 0.8450 | 0.0120 | A | -0.0012 | 0.0003 |
| I A - TotalCharges - gender | 0.8453 | 0.0120 | A | -0.0008 | 0.0003 |
| J E - TotalCharges | 0.8464 | 0.0120 | E | -0.0002 | 0.0001 |
| K E - TotalCharges - gender | 0.8467 | 0.0120 | E | +0.0002 | 0.0001 |
| L A - TotalCharges - gender - MonthlyCharges (adopted) | 0.8454 | 0.0120 | I | +0.0000 | 0.0000 |

## Gradient Boosting

| Configuration | Mean AUC | Std | vs | Delta AUC | SE |
|---|---|---|---|---|---|
| A base features only | 0.8471 | 0.0137 |  |  |  |
| B A + TenureGroup | 0.8469 | 0.0136 | A | -0.0002 | 0.0001 |
| C A + ProtectiveAddOnCount | 0.8475 | 0.0135 | A | +0.0004 | 0.0003 |
| D A + AutoPay | 0.8469 | 0.0137 | A | -0.0002 | 0.0001 |
| E A + all three engineered | 0.8479 | 0.0134 | A | +0.0008 | 0.0004 |
| F A - five low-signal columns | 0.8461 | 0.0132 | A | -0.0010 | 0.0006 |
| G A - gender | 0.8473 | 0.0134 | A | +0.0002 | 0.0002 |
| H A - TotalCharges | 0.8475 | 0.0135 | A | +0.0004 | 0.0004 |
| I A - TotalCharges - gender | 0.8479 | 0.0133 | A | +0.0008 | 0.0004 |
| J E - TotalCharges | 0.8475 | 0.0134 | E | -0.0004 | 0.0005 |
| K E - TotalCharges - gender | 0.8478 | 0.0134 | E | -0.0002 | 0.0005 |
| L A - TotalCharges - gender - MonthlyCharges (adopted) | 0.8487 | 0.0128 | I | +0.0008 | 0.0006 |

