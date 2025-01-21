from chemml.datasets import load_xyz_polarizability
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

molecules, polarizabilities = load_xyz_polarizability()
polarizability_values = polarizabilities.values.ravel()


molecular_features = np.random.rand(len(molecules), 200)

X_train, X_test, y_train, y_test = train_test_split(
    molecular_features, polarizability_values, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Root Mean Squared Error (RMSE):", rmse)

import joblib
joblib.dump(model, "polarizability_model.pkl")

new_molecule_descriptor = np.random.rand(1, 200) 
predicted_polarizability = model.predict(new_molecule_descriptor)
print("Predicted Polarizability (Bohr^3):", predicted_polarizability[0])

results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
results_df.to_csv("results/polarizability.csv", index=False)
print("Actual vs Predicted results exported to 'actual_vs_predicted.csv'")
