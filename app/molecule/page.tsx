'use client';

import React, { useState } from 'react';
import MoleculeViewer from '@/components/MoleculeViewer';

const MoleculePage: React.FC = () => {
  const [moleculeName, setMoleculeName] = useState('');
  const [moleculeData, setMoleculeData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFetchMolecule = async () => {
    setError(null);
    setMoleculeData(null);

    try {
      const response = await fetch(`/api/fetchMolecule?molecule=${moleculeName}`);
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
    <div className="w-full h-screen">
      <div className="absolute top-0 left-0 z-10 p-4 bg-black/50">
        <h1 className="text-white text-2xl mb-4">Molecule Viewer</h1>
        <div className="flex gap-2">
          <input
            type="text"
            className="px-2 py-1 rounded"
            placeholder="Enter molecule name (e.g., NaCl)"
            value={moleculeName}
            onChange={(e) => setMoleculeName(e.target.value)}
          />
          <button 
            className="px-4 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
            onClick={handleFetchMolecule}
          >
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