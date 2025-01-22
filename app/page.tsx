"use client";

import React, { useState } from "react";
import MoleculeViewer from "@/components/MoleculeViewer";
import styles from "./page.module.css";

const MoleculePage: React.FC = () => {
  const [moleculeName, setMoleculeName] = useState("");
  const [moleculeData, setMoleculeData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFetchMolecule = async () => {
    setError(null);
    setMoleculeData(null);

    try {
      const response = await fetch(`/api/fetchMolecule?smiles=${moleculeName}`);
      const data = await response.json();
      if (response.ok) {
        setMoleculeData(data);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError("Failed to fetch molecule data.");
    }
  };

  return (
    <div>
      <div className={styles.container}>
        <h1 className={styles.headline}>Molecule Viewer</h1>
        <div className={styles.inputContainer}>
          <input
            type="text"
            className={styles.inputField}
            placeholder="Enter SMILES string (e.g. O)"
            value={moleculeName}
            onChange={(e) => setMoleculeName(e.target.value)}
          />
          <button className={styles.inputButton} onClick={handleFetchMolecule}>
            Search
          </button>
        </div>
        {error && <p className="text-red-500 mt-2">{error}</p>}
        <div className={styles.moleculeInfo}>
          <p>
            <b>Some properties</b> (actual/predicted by model)
          </p>
          <p>
            <b>Density: </b>
            {moleculeData &&
              `${
                moleculeData.predicted_density
                  ? moleculeData.predicted_density
                  : "N/A"
              }/${
                moleculeData.actual_density
                  ? moleculeData.actual_density
                  : "N/A"
              } g/cm^3`}
          </p>
          <p>
            <b>Polarization: </b>
          </p>
          <p>
            <b>HOMO: </b>
          </p>
        </div>
      </div>
      {moleculeData && (
        <MoleculeViewer moleculeData={moleculeData.molecule_data} />
      )}
    </div>
  );
};

export default MoleculePage;
