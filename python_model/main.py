import warnings
warnings.filterwarnings("ignore")

# Set TensorFlow environment variable to enable XLA (Accelerated Linear Algebra) devices for optimization.
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# dataset for predicting organic density using ChemML.
from chemml.datasets import load_organic_density
molecules, target, dragon_subset = load_organic_density()

# libraries for data splitting and preprocessing.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Initialize scalers for standardizing input features (X) and target values (y).
xscale = StandardScaler()
yscale = StandardScaler()

# Split the data into training and testing sets.
# `dragon_subset` contains input features, and `target` contains the corresponding labels.
# Use 25% of the data for testing, with a fixed random state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(dragon_subset.values, target.values, test_size=0.25, random_state=42)

# Standardize the training and testing data for input features.
# Fit the scaler on the training data and transform both training and test data.
X_train = xscale.fit_transform(X_train)
X_test = xscale.transform(X_test)

# Standardize the training data for the target variable.
# Note: The test target values are not transformed at this stage.
y_train = yscale.fit_transform(y_train)

# Import the MLP (Multi-Layer Perceptron) model from ChemML for regression.
from chemml.models import MLP

# Initialize the MLP model with the following configurations:
# - Engine: TensorFlow
# - Architecture: 3 hidden layers with 32, 64, and 128 neurons respectively, all using ReLU activation.
# - Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.01.
# - Regularization: L2 regularization with an alpha value of 0.002.
# - Epochs: Train for 100 epochs with a batch size of 50.
# - Loss function: Mean Squared Error (MSE) for regression.
mlp = MLP(engine='tensorflow',nfeatures=X_train.shape[1], nneurons=[32,64,128], activations=['ReLU','ReLU','ReLU'],
                learning_rate=0.01, alpha=0.002, nepochs=100, batch_size=50, loss='mean_squared_error',
                is_regression=True, nclasses=None, layer_config_file=None, opt_config='sgd')

# Train the MLP model using the training data.
mlp.fit(X=X_train, y=y_train)

# Save the trained model to the current directory with the filename "saved_MLP".
mlp.save(path=".", filename="saved_MLP")

# Load the saved MLP model using ChemML's utility function for loading models.
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
