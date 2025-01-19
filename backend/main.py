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

# Define input schema
class MoleculeInput(BaseModel):
    smiles: str

def convert_smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Invalid SMILES string.")
    features = []
    return features

@app.post("/predict")
async def predict_density(input_data: MoleculeInput):
    smiles = input_data.smiles
    try:
        features = convert_smiles_to_features(smiles)
        scaled_features = x_scaler.transform([features])
        scaled_prediction = model.predict(scaled_features)
        predicted_density = y_scaler.inverse_transform(scaled_prediction)
        return {"smiles": smiles, "predicted_density": predicted_density[0][0]}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/api/fetchMolecule")
async def fetch_molecule(smiles: str):
    try:
        # Generate 3D coordinates from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return JSONResponse(status_code=400, content={"error": "Invalid SMILES string."})
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
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
