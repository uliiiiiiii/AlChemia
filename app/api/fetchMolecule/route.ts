import { NextResponse } from 'next/server';
import axios from 'axios';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const molecule = searchParams.get('molecule');
  console.log(molecule);

  if (!molecule) {
    return NextResponse.json({ error: "Molecule name is required." }, { status: 400 });
  }

  try {
    const response = await axios.get(
      `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${encodeURIComponent(
        molecule
      )}/record/JSON`
    );

    return NextResponse.json(response.data.PC_Compounds);
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Failed to fetch molecule data." }, { status: 500 });
  }
}







