from fastapi import FastAPI
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import StandardScaler
from chemml.utils import load_chemml_model
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

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
        # Generate 3D coordinates
        molecule_data = generate_3d_coordinates(smiles)

        return {
            "smiles": smiles,
            "molecule_data": molecule_data
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
