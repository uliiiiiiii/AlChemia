from chemml.utils import load_chemml_model
loaded_MLP = load_chemml_model("./saved_MLP_chemml_model.json")
print(loaded_MLP)

# Access the underlying TensorFlow model of the loaded MLP for inspection if needed.
loaded_MLP.model

# Make predictions on the test data using the loaded MLP model.
# The predictions are in the standardized form, so they need to be inverse transformed.
import numpy as np
y_pred = loaded_MLP.predict(X_test)
y_pred = yscale.inverse_transform(y_pred.reshape(-1,1))  # Reshape and inverse transform to get original scale.

# Calculate regression performance metrics to evaluate the model's predictions.
from chemml.utils import regression_metrics

# Ensure the inputs for metrics calculation have the same data type.
metrics_df = regression_metrics(y_test, y_pred)
print("Metrics: \n")
print(metrics_df[["MAE", "RMSE", "MAPE", "r_squared"]])

# Visualize the relationship between actual and predicted values using scatter plots.
from chemml.visualization import scatter2D, SavePlot, decorator
import pandas as pd
import matplotlib.pyplot as plt

# Prepare data for visualization by combining actual and predicted values in a DataFrame.
df = pd.DataFrame()
df["Actual"] = y_test.reshape(-1,)
df["Predicted"] = y_pred.reshape(-1,)

# Create a scatter plot with red markers for 'Actual vs Predicted' values.
sc = scatter2D('r', marker='.')
fig = sc.plot(dfx=df, dfy=df, x="Actual", y="Predicted")

# Decorate the plot with labels, limits, grid settings, and a title.
dec = decorator(title='Actual vs. Predicted', xlabel='Actual Density', ylabel='Predicted Density',
                xlim=(950,1550), ylim=(950,1550), grid=True,
                grid_color='g', grid_linestyle=':', grid_linewidth=0.5)
fig = dec.fit(fig)

# Save the plot as an image file in the specified directory with custom settings.
sa = SavePlot(filename='Parity', output_directory='images',
              kwargs={'facecolor':'w', 'dpi':330, 'pad_inches':0.1, 'bbox_inches':'tight'})
sa.save(obj=fig)

# Display the plot in the current session.
fig.show()
