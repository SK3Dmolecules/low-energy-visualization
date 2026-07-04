import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import py3Dmol
from stmol import showmol

# Page Configurations
st.set_page_config(
    page_title="3D Molecule Conformer Generator",
    page_icon="🧬",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #2E4057; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.1rem; color: #666; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Layout Header
st.markdown('<div class="main-title">🧬 3D Low-Energy Conformation Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Input a SMILES string to automatically generate, optimize, and visualize its 3D molecular structure.</div>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("⚙️ Optimization Settings")
max_iters = st.sidebar.slider("Energy Minimization Iterations", min_value=100, max_value=1000, value=500, step=100)
style_choice = st.sidebar.selectbox("3D Representation Style", ["Stick", "Sphere/CPK", "Ball and Stick"])

# Main Input Layout
smiles_input = st.text_input(
    "Enter SMILES String:", 
    value="CC(=O)OC1=CC=CC=C1C(=O)O", 
    placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)"
)

if st.button("Generate & Optimize 3D Structure", type="primary"):
    if not smiles_input.strip():
        st.error("Please provide a valid SMILES string.")
    else:
        with st.spinner("⏳ Parsing SMILES, creating 3D coordinates, and running energy minimization..."):
            try:
                # 1. Convert SMILES to Mol Object
                mol = Chem.MolFromSmiles(smiles_input)
                
                if mol is None:
                    st.error("❌ Invalid SMILES notation. Please double-check your syntax.")
                else:
                    # 2. Add explicit Hydrogens (critical for accurate physical properties/energy calculation)
                    mol_with_h = Chem.AddHs(mol)
                    
                    # 3. Embed 3D Coordinates using ETKDG v3
                    params = AllChem.ETKDGv3()
                    embed_result = AllChem.EmbedMolecule(mol_with_h, params)
                    
                    if embed_result == -1:
                        st.error("❌ 3D Embedding failed. The molecule structure may be physically impossible or overly constrained.")
                    else:
                        # 4. Energy Minimization Strategy (MMFF94 with a UFF fallback)
                        ff_properties = AllChem.MMFFGetMoleculeProperties(mol_with_h)
                        if ff_properties:
                            ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, ff_properties)
                            ff.Initialize()
                            ff.Minimize(maxIts=max_iters)
                            ff_used = "MMFF94 (Merck Molecular Force Field)"
                        else:
                            # Fallback if atoms aren't supported by MMFF94
                            ff = AllChem.UFFGetMoleculeForceField(mol_with_h)
                            ff.Initialize()
                            ff.Minimize(maxIts=max_iters)
                            ff_used = "UFF (Universal Force Field - Fallback)"
                        
                        # 5. Extract Details for UI Display
                        mol_weight = Descriptors.MolWt(mol)
                        formula = Chem.rdmolops.CalcMolFormula(mol)
                        pdb_block = Chem.MolToPDBBlock(mol_with_h)
                        
                        # --- Display Results ---
                        st.success(f"✅ Optimization complete using {ff_used}!")
                        
                        # Metrics Columns
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Chemical Formula", formula)
                        col2.metric("Molecular Weight", f"{mol_weight:.2f} g/mol")
                        col3.metric("Total Atoms (with H)", mol_with_h.GetNumAtoms())
                        
                        st.divider()
                        
                        # 3D Visualizer Render
                        st.subheader("Interactive 3D View")
                        st.caption("Click and drag to rotate. Scroll to zoom.")
                        
                        viewer = py3Dmol.view(width=800, height=500)
                        viewer.addModel(pdb_block, 'pdb')
                        
                        # Map style choices
                        if style_choice == "Stick":
                            viewer.setStyle({'stick': {'radius': 0.2}})
                        elif style_choice == "Sphere/CPK":
                            viewer.setStyle({'sphere': {'scale': 0.9}})
                        else:  # Ball and Stick
                            viewer.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.3}})
                            
                        viewer.zoomTo()
                        showmol(viewer, height=500, width=800)
                        
                        # Download Button
                        st.download_button(
                            label="📥 Download Low-Energy .PDB File",
                            data=pdb_block,
                            file_name="optimized_conformation.pdb",
                            mime="chemical/x-pdb"
                        )
            except Exception as e:
                st.error(f"An unexpected system error occurred: {e}")