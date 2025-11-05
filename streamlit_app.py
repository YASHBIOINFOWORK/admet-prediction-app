import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import plotly.express as px
from io import StringIO
from PIL import Image

# ✅ PAINS-safe fallback (no RDKit crash)
def pains_check(mol):
    return "Not Supported in this environment"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="ADMET & Docking Prioritizer", page_icon="🧬", layout="wide")
st.title("🧬 ADMET & Docking Prioritizer")
st.caption("Cheminformatics + ADMET + Docking Prioritization")

EXAMPLE_DATA = """SMILES,Docking_Score
CC(=O)Oc1ccccc1C(=O)O,-7.2
COc1ccc(C(C)Nc2ccc(C)cc2)cc1,-6.5
O=C1CCc2c(C)nc(C)c2N1C1CC1,-4.1
CC(C)CN(C)CC(O)C(C)c1ccc(O)c(Cl)c1,-5.9
C1=CC=C2C(=C1)C=CC(=O)C2=O,-7.5
InvalidSMILES,-8.0
"""

# ========================= INPUT
input_method = st.radio("Input molecules", 
                        ('Example Data','Paste Data','Upload CSV'),horizontal=True)

if input_method=="Example Data":
    df = pd.read_csv(StringIO(EXAMPLE_DATA))

elif input_method=="Paste Data":
    txt = st.text_area("Paste SMILES,Docking_Score", EXAMPLE_DATA, height=200)
    df = pd.read_csv(StringIO(txt))

else:
    file = st.file_uploader("Upload CSV",type=["csv"])
    df = pd.read_csv(file) if file else None

# ========================= PROCESS
if st.button("🚀 Analyze") and df is not None:

    results, images = [], []

    for _, row in df.iterrows():
        sm = row.SMILES
        score = row.Docking_Score
        mol = Chem.MolFromSmiles(sm)

        if mol is None:
            results.append([sm,score,"Invalid",None,None,None,None,None,"No"])
            images.append(None)
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        violations = sum([mw>500, logp>5, hbd>5, hba>10])
        status = "Pass" if violations<=1 else "Fail"

        pains_flag = pains_check(mol)

        results.append([sm,score,status,round(mw,2),round(logp,2),hbd,hba,violations,pains_flag])
        images.append(Draw.MolToImage(mol,size=(200,200)))

    cols = ["SMILES","Docking_Score","Status","MW","LogP","HDonors","HAcceptors","Violations","PAINS"]
    df_res = pd.DataFrame(results,columns=cols)

    st.success("✅ Done")
    st.dataframe(df_res,use_container_width=True)

    st.subheader("Structures")
    for i,img in enumerate(images):
        if img: st.image(img,caption=df_res.iloc[i].SMILES,width=150)

    st.subheader("MW vs Docking Score")
    fig = px.scatter(df_res[df_res.Status=="Pass"],x="MW",y="Docking_Score",
                     hover_data=["SMILES"],color="Violations")
    st.plotly_chart(fig,use_container_width=True)

    st.download_button("💾 Download Results", df_res.to_csv(index=False), "results.csv")

else:
    st.info("Upload or paste molecules, then click Analyze")
