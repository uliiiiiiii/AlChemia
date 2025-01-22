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
            placeholder="Enter molecule name (e.g., NaCl)"
            value={moleculeName}
            onChange={(e) => setMoleculeName(e.target.value)}
          />
          <button className={styles.inputButton} onClick={handleFetchMolecule}>
            Search
          </button>
        </div>
        {error && <p className="text-red-500 mt-2">{error}</p>}
      </div>
      {moleculeData && <MoleculeViewer moleculeData={moleculeData} />}
    </div>
  );
};

export default MoleculePage;
