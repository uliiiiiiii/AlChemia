import { NextResponse } from 'next/server'

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const smiles = searchParams.get('smiles')

  if (!smiles) {
    return NextResponse.json(
      { error: 'SMILES parameter is required' },
      { status: 400 }
    )
  }

  try {
    const response = await fetch(
      `${process.env.NEXT_PUBLIC_PYTHON_BACKEND_LINK}/api/fetchMolecule?smiles=${smiles}`,
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(
        { error: data.error || 'Failed to fetch molecule data' },
        { status: response.status }
      )
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error('Error fetching molecule data:', error)
    return NextResponse.json(
      { error: 'Failed to fetch molecule data' },
      { status: 500 }
    )
  }
}