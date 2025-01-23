import numpy as np
import pandas as pd
from chemml.datasets import load_cep_homo
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from chemml.visualization import scatter2D, SavePlot, decorator
import joblib

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = {}
    for name, function in Descriptors._descList:
        try:
            value = function(mol)
            descriptors[name] = float(value) 
        except:
            descriptors[name] = 0.0 
    
    morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
    fp = morgan_gen.GetFingerprintAsNumPy(mol)
    
    for i, value in enumerate(fp):
        descriptors[f'Morgan_bit_{i}'] = float(value)
    
    return descriptors

print("Loading dataset...")
smiles_df, homo_df = load_cep_homo()

print("Calculating molecular descriptors...")
descriptor_list = []
for smiles in smiles_df['smiles']:
    desc = calculate_descriptors(smiles)
    if desc is None:
        desc = {name: 0.0 for name in descriptor_list[0].keys()} if descriptor_list else {}
    descriptor_list.append(desc)

X = pd.DataFrame(descriptor_list)
X = X.fillna(0.0) 

y = homo_df.values.ravel()

print("Preprocessing features...")
variance = X.var()
X = X.loc[:, variance > 0]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Making predictions...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nResults:")
print(f"Training RMSE: {train_rmse:.4f} eV")
print(f"Testing RMSE: {test_rmse:.4f} eV")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")

test_results = pd.DataFrame({
    'SMILES': smiles_df.iloc[X_test.index]['smiles'],
    'Actual_HOMO_eV': y_test,
    'Predicted_HOMO_eV': y_pred_test
})

test_results.to_csv('../results/HOMO.csv', index=False)
joblib.dump(model, '../trained_models/HOMO_model.pkl')

# Prepare data for visualization
visualization_df = pd.DataFrame({
    "Actual": y_test, 
    "Predicted": y_pred_test
})


# Create scatter plot
scatter = scatter2D('r', marker='.') 
fig = scatter.plot(dfx=visualization_df, dfy=visualization_df, x="Actual", y="Predicted")

# Decorate the plot with labels, limits, and grid settings
decorate = decorator(
    title='Реальна vs. передбачувана HOMO',
    xlabel='Реальна HOMO',
    ylabel='Передбачувана HOMO', 
    xlim=(min(y_test), max(y_test)),
    ylim=(min(y_test), max(y_test)),
    grid=True,
    grid_color='g',
    grid_linestyle=':',
    grid_linewidth=0.5
)
fig = decorate.fit(fig)


# Save the plot to a file
save_plot = SavePlot(
    filename='HOMO_Prediction_Parity_Plot',
    output_directory='../images',
    kwargs={
        'facecolor': 'w',
        'dpi': 330,
        'pad_inches': 0.1,
        'bbox_inches': 'tight'
    }
)
save_plot.save(obj=fig)