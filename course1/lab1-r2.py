from sklearn.metrics import r2_score

y_true = [100, 150, 200]
y_pred = [110, 140, 190]

r2 = r2_score(y_true, y_pred)
print(f"R² Score: {r2:.3f}")

