from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_atoms(smiles: str, seed: int = 0) -> Atoms:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise RuntimeError(f"RDKit embedding failed for {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=positions)
