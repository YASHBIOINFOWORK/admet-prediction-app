# ================================================
# ✅ Imports
# ================================================
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
import py3Dmol
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import files
import io
import streamlit as st # (if running as notebook ignore)
# ================================================

# =========================
# ✅ 1) Lipinski Function
# =========================
def lipinski(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([None]*5, index=["MW", "LogP", "HBD", "HBA","Lipinski_pass"])

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Chem.GetNHydrogenBondDonors(mol)
    hba = Chem.GetNHydrogenBondAcceptors(mol)

    lip_pass = (mw < 500) and (logp < 5) and (hbd <= 5) and (hba <= 10)

    return pd.Series([mw, logp, hbd, hba, lip_pass],
                     index=["MW","LogP","HBD","HBA","Lipinski_pass"])


# =========================
# ✅ 2) PAINS Filter (SMILES check - no RDKit FilterCatalog)
# =========================
PAINS_smarts = [
    "c1ccccc1[N+](=O)[O-]",  # Nitroaromatics
    "[#6]=[#6]-[#6]=[#6]",   # Michael acceptor
    "O=C-O-CO",              # Carbonate alerts
]

def pains_check(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    for smarts in PAINS_smarts:
        patt = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(patt):
            return True
    return False


# =========================
# ✅ 3) Display 3D molecule
# =========================
def show_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.EmbedMolecule(mol)
    Chem.UFFOptimizeMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mblock, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.zoomTo()
    return viewer.show()


# =========================
# ✅ 4) Upload CSV or SDF
# =========================
uploaded = files.upload()
filename = next(iter(uploaded))
filetype = filename.split(".")[-1]

if filetype == "csv":
    df = pd.read_csv(io.BytesIO(uploaded[filename]))
elif filetype == "sdf":
    df = PandasTools.LoadSDF(io.BytesIO(uploaded[filename]), smilesName="SMILES")
else:
    raise ValueError("Upload only .csv or .sdf")

df = df.dropna(subset=["SMILES"]).reset_index(drop=True)

# =========================
# ✅ 5) Compute properties
# =========================
lip = df["SMILES"].apply(lipinski)
df = pd.concat([df, lip], axis=1)
df["PAINS_flag"] = df["SMILES"].apply(pains_check)

print("✅ ADMET + Lipinski + PAINS computed!")

# =========================
# ✅ 6) Train Quick Classifier for Demo
# =========================
features = df[["MW","LogP","HBD","HBA"]]
labels = np.where(df["Lipinski_pass"]==True, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
model = RandomForestClassifier()
model.fit(X_train,y_train)
preds = model.predict(X_test)

print("✅ Demo ML Accuracy:", accuracy_score(y_test,preds))

df["Predicted_ADMET"] = model.predict(features)

# =========================
# ✅ 7) Visualization
# =========================
fig = px.scatter(df,
                 x="MW", y="LogP",
                 color="Predicted_ADMET",
                 title="Drug-likeness Plot (MW vs LogP)")
fig.show()

df.head()
