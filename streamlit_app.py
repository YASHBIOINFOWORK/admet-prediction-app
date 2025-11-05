import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, FilterCatalog, FilterCatalogParams
from rdkit.Chem.Draw import rdMolDraw2D
import py3Dmol
from io import StringIO

# =====================================
# PAGE CONFIG & THEME
# =====================================
st.set_page_config(page_title="ADMET & Docking Prioritizer", page_icon="🧬", layout="wide")

st.markdown("""
<style>
body { background-color:#0e1117; color:#fafafa }
h1,h2,h3 { color:#00c3ff }
</style>
""", unsafe_allow_html=True)

st.title("🧬 ADMET + Docking Prioritizer")
st.caption("Drug-likeness, PAINS filter, 3D Viewer & Docking Score Ranking")

# ========== PAINS FILTER ==========
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog = FilterCatalog(params)

def pains_check(mol):
    """Return PAINS alerts"""
    entries = catalog.GetMatches(mol)
    return "Yes" if len(entries) > 0 else "No"

# ========== 3D VIEWER ==========

def show_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    try:
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        block = Chem.MolToMolBlock(mol)
        
        viewer = py3Dmol.view(width=500, height=400)
        viewer.addModel(block,"mol")
        viewer.setStyle({'stick': {}})
        viewer.zoomTo()
        viewer.show()
    except:
        st.warning("3D structure could not be generated for this SMILES.")

# ========== INPUT SECTION ==========
st.header("📥 Input Molecules")

input_method = st.radio("Choose input:", ["Example", "Paste", "Upload CSV", "Upload SDF"], horizontal=True)

EXAMPLE = """SMILES,Docking
CC(=O)Oc1ccccc1C(=O)O,-7.2
COc1ccc(C(C)Nc2ccc(C)cc2)cc1,-6.5
O=C1CCc2c(C)nc(C)c2N1C1CC1,-4.1
"""

df = None

if input_method == "Example":
    df = pd.read_csv(StringIO(EXAMPLE.strip()))

elif input_method == "Paste":
    txt = st.text_area("Paste SMILES,Docking:", EXAMPLE, height=200)
    df = pd.read_csv(StringIO(txt.strip()))

elif input_method == "Upload CSV":
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f:
        df = pd.read_csv(f)

elif input_method == "Upload SDF":
    sdf = st.file_uploader("Upload .sdf", type=["sdf"])
    if sdf:
        suppl = Chem.SDMolSupplier(sdf)
        mols = [m for m in suppl if m is not None]
        df = pd.DataFrame({
            "SMILES":[Chem.MolToSmiles(m) for m in mols],
            "Docking":[0]*len(mols)   # user adds docking later
        })
        st.success(f"✅ Loaded {len(df)} molecules from SDF")

if df is not None:
    st.dataframe(df, use_container_width=True)

# ========== RUN ANALYSIS ==========
if st.button("🚀 Run ADMET Prioritization") and df is not None:
    results = []
    
    for i,row in df.iterrows():
        smi = row["SMILES"]
        dock = row["Docking"]
        mol = Chem.MolFromSmiles(smi)
        
        if mol is None:
            continue
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hdon = Descriptors.NumHDonors(mol)
        hacp = Descriptors.NumHAcceptors(mol)
        
        # Lipinski violations
        viol = sum([
            mw > 500,
            logp > 5,
            hdon > 5,
            hacp > 10
        ])

        # PAINS check
        pains = pains_check(mol)

        status = "Pass" if viol <= 1 and pains=="No" else "Fail"

        results.append([smi, dock, mw, logp, hdon, hacp, viol, pains, status])

    cols = ["SMILES","Docking","MW","LogP","HDonors","HAcceptors","Violations","PAINS","Status"]
    out = pd.DataFrame(results, columns=cols)

    # Ranking (better docking = lower score)
    out_pass = out[out["Status"]=="Pass"].sort_values(by="Docking")
    out_pass["Rank"] = range(1,len(out_pass)+1)
    out_fail = out[out["Status"]!="Pass"]
    out_fail["Rank"] = "-"
    
    final = pd.concat([out_pass,out_fail])
    
    st.success("✅ Analysis Complete")
    st.dataframe(final, use_container_width=True)

    st.download_button("💾 Download Results", final.to_csv(index=False).encode(),
                       "admet_results.csv", "text/csv")

    # View 3D
    st.subheader("🧬 3D Molecule Viewer")
    smi_sel = st.selectbox("Select molecule", final["SMILES"].tolist())
    show_3d(smi_sel)
