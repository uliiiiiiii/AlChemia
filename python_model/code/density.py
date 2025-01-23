import warnings
warnings.filterwarnings("ignore")
import os
from chemml.datasets import load_organic_density
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from chemml.models import MLP
import numpy as np
import pandas as pd
from chemml.utils import regression_metrics
from chemml.visualization import scatter2D, SavePlot, decorator
import matplotlib.pyplot as plt

# Enable TensorFlow XLA (Accelerated Linear Algebra) for optimization
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Load dataset for organic density prediction using ChemML
molecules, target, dragon_subset = load_organic_density()  # molecules: molecular structures, target: density values, dragon_subset: features

# Initialize scalers to standardize input features (X) and target values (y)
xscale = StandardScaler()  # For feature scaling
yscale = StandardScaler()  # For target value scaling

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    dragon_subset.values, target.values, test_size=0.3, random_state=42
)

# Standardize the feature (x) and target (y) values
X_train = xscale.fit_transform(X_train)
X_test = xscale.transform(X_test)
y_train = yscale.fit_transform(y_train)

# Initialize the MLP (Multi-Layer Perceptron) model
mlp = MLP(
    engine='tensorflow',
    nfeatures=X_train.shape[1],  # Number of input features
    nneurons=[32, 64, 128],  # Architecture: number of neurons in each layer
    activations=['ReLU', 'ReLU', 'ReLU'],  # Activation functions for each layer
    learning_rate=0.01,  # Learning rate for optimization
    alpha=0.002,  # L2 regularization parameter
    nepochs=100,  # Number of training epochs
    batch_size=50,  # Batch size for training
    loss='mean_squared_error',  # Loss function for regression
    is_regression=True,
    opt_config='sgd'  # Optimization algorithm: Stochastic Gradient Descent
)

# Train the model
mlp.fit(X=X_train, y=y_train)

# Save the trained model to the current directory
mlp.save(path="../trained_models/", filename="density_MLP")

# # Load the saved model for inference or further use
# from chemml.utils import load_chemml_model
# loaded_MLP = load_chemml_model("./saved_MLP_chemml_model.json")

y_pred = mlp.predict(X_test)  # Predict density values (standardized)

# Inverse transform the predictions to the original scale
y_pred = yscale.inverse_transform(y_pred.reshape(-1, 1))

# Convert actual and predicted values into a readable format
actual_values = y_test.reshape(-1,)
predicted_values = y_pred.reshape(-1,)

# Create a DataFrame combining molecules, actual values, and predictions
results_df = pd.DataFrame({
    "Molecule": molecules['smiles'][len(X_train):].values, 
    "Actual Density": actual_values,
    "Predicted Density": predicted_values
})

# Save the results to a CSV
results_df.to_csv("../results/density_prediction.csv", index=False)

# Calculate regression performance metrics
metrics_df = regression_metrics(y_test, y_pred)
print("Metrics:\n")
print(metrics_df[["MAE", "RMSE", "MAPE", "r_squared"]])  # Display key metrics


# Prepare data for visualization
visualization_df = pd.DataFrame({
    "Actual": actual_values, 
    "Predicted": predicted_values
})

# Create scatter plot
scatter = scatter2D('r', marker='.') 
fig = scatter.plot(dfx=visualization_df, dfy=visualization_df, x="Actual", y="Predicted")

# Decorate the plot with labels, limits, and grid settings
decorate = decorator(
    title='Реальна vs. передбачувана густина',
    xlabel='Реальна густина',
    ylabel='Передбачувана густина', 
    xlim=(950, 1550),
    ylim=(950, 1550),
    grid=True,
    grid_color='g',
    grid_linestyle=':',
    grid_linewidth=0.5
)
fig = decorate.fit(fig)

# Save the plot to a file
save_plot = SavePlot(
    filename='Density_Prediction_Parity_Plot',
    output_directory='../images',
    kwargs={
        'facecolor': 'w',
        'dpi': 330,
        'pad_inches': 0.1,
        'bbox_inches': 'tight'
    }
)
save_plot.save(obj=fig)
