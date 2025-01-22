from fastapi import FastAPI
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from chemml.utils import load_chemml_model
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model
model = load_chemml_model("../python_model/saved_MLP_chemml_model.json")

x_scaler = StandardScaler()
y_scaler = StandardScaler()

def convert_smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string.")
    # Replace this with your actual feature extraction logic
    features = []
    return features

def fetch_actual_density(smiles: str) -> float:
    """
    Placeholder for fetching the actual density from a database or API.
    Replace this with your logic to fetch the actual density.
    """
    # Example: Mocked actual density
    return 1.23  # Replace with actual density fetching logic

def generate_3d_coordinates(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string.")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    atoms = []
    bonds = []
    conformer = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        atoms.append({
            "id": atom.GetIdx(),
            "element": atom.GetAtomicNum(),
            "x": pos.x,
            "y": pos.y,
            "z": pos.z
        })
    for bond in mol.GetBonds():
        bonds.append({
            "aid1": bond.GetBeginAtomIdx(),
            "aid2": bond.GetEndAtomIdx(),
            "order": bond.GetBondTypeAsDouble()
        })
    return {"atoms": atoms, "bonds": bonds}

@app.get("/api/fetchMolecule")
async def predict_density(smiles: str):
    try:
        # # Feature extraction and prediction
        # features = convert_smiles_to_features(smiles)
        # scaled_features = x_scaler.transform([features])
        # scaled_prediction = model.predict(scaled_features)
        # predicted_density = y_scaler.inverse_transform(scaled_prediction)[0][0]

        # # Fetch actual density
        # actual_density = fetch_actual_density(smiles)

        # Generate 3D coordinates
        molecule_data = generate_3d_coordinates(smiles)

        # return {
        #     "smiles": smiles,
        #     "predicted_density": predicted_density,
        #     "actual_density": actual_density,
        #     "molecule_data": molecule_data
        # }
        return {
            "smiles": smiles,
            "predicted_density": 1203,
            "actual_density": 204,
            "molecule_data": molecule_data
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
