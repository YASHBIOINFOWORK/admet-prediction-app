# =========================================================
# 🧬 ADMET + Docking Prioritizer Notebook
# Lipinski + PAINS + 2D/3D Viewer + Ranking
# =========================================================

# 1️⃣ IMPORT LIBRARIES
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, PandasTools, FilterCatalog, FilterCatalogParams
import py3Dmol
import plotly.express as px
from google.colab import files
from io import StringIO

# 2️⃣ LOAD SAMPLE DATA OR UPLOAD CSV
print("📂 Upload your CSV file with columns: SMILES,Docking_Score")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)
df.head()

# If you want sample data instead, uncomment below
"""
sample_data = \"\"\"SMILES,Docking_Score
CC(=O)Oc1ccccc1C(=O)O,-7.2
COc1ccc(C(C)Nc2ccc(C)cc2)cc1,-6.5
O=C1CCc2c(C)nc(C)c2N1C1CC1,-4.1
CC(C)CN(C)CC(O)C(C)c1ccc(O)c(Cl)c1,-5.9
C1=CC=C2C(=C1)C=CC(=O)C2=O,-7.5
\"\"\"
df = pd.read_csv(StringIO(sample_data))
"""

# 3️⃣ LIPINSKI + PAINS FILTERING
catalog = FilterCatalog.FilterCatalog(FilterCatalogParams.FilterCatalogParams(FilterCatalogParams.FilterCatalogParams.PAINS))

results = []
for _, row in df.iterrows():
    smi = row["SMILES"]
    dock = row["Docking_Score"]
    mol = Chem.MolFromSmiles(smi)

    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        lipinski_viol = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10
        ])

        pains = catalog.HasMatch(mol)

        results.append([
            smi, dock, round(mw,2), round(logp,2), hbd, hba, lipinski_viol, pains
        ])

df_res = pd.DataFrame(results,
    columns=["SMILES","Docking","MW","LogP","HBD","HBA","Lipinski_Viol","PAINS"]
)
df_res["Druglike"] = df_res.apply(lambda x: "Pass" if x.Lipinski_Viol <= 1 and not x.PAINS else "Fail", axis=1)
df_res.head()

# 4️⃣ RANK MOLECULES (only passing ones)
df_rank = df_res[df_res["Druglike"]=="Pass"].sort_values(by="Docking", ascending=True).reset_index(drop=True)
df_rank["Rank"] = range(1, len(df_rank) + 1)
df_rank

# 5️⃣ ADD 2D STRUCTURES
PandasTools.AddMoleculeColumnToFrame(df_rank, smilesCol="SMILES")
df_rank[["SMILES","Molecule","Docking","Rank"]]

# 6️⃣ OPTIONAL: 3D SDF FILE VIEWER
print("📂 (Optional) Upload .sdf for 3D view")
sdf_upload = files.upload()

sdf_file = list(sdf_upload.keys())[0]
mols = PandasTools.LoadSDF(sdf_file)
mol3d = mols["ROMol"][0]
mol_block = Chem.MolToMolBlock(mol3d)

view = py3Dmol.view(width=400, height=400)
view.addModel(mol_block,'mol')
view.setStyle({'stick':{}})
view.zoomTo()
view.show()

# 7️⃣ Plot MW vs Docking Score
fig = px.scatter(df_rank, x="MW", y="Docking", hover_data=["SMILES"],
                 title="MW vs Docking Score (Only Passing Molecules)")
fig.show()

# 8️⃣ DOWNLOAD FINAL RESULTS
df_rank.to_csv("ADMET_ranked.csv", index=False)
files.download("ADMET_ranked.csv")

print("✅ Completed — results saved as ADMET_ranked.csv")
