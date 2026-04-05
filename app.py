import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import io

import aeroml_notebook_common as common
from aeroml_forward_v3_runtime import ForwardV3Predictor
from aeroml_reverse_runtime import ReverseV3Designer

st.set_page_config(
    page_title="AeroML Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- THEME & STYLING ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

    /* Dark Theme Optimization + Dribbble Cyber/Pro Aesthetics */
    .stApp {
        background-color: #050505;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #111111;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    
    /* Subtle volt green border glow on hover */
    .metric-card:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(209, 255, 0, 0.3);
        box-shadow: 0 12px 40px rgba(209, 255, 0, 0.12);
    }
    
    .metric-title {
        font-size: 0.85rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        font-weight: 600;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #D1FF00;
        text-shadow: 0 0 20px rgba(209, 255, 0, 0.3);
    }
    
    .metric-value-warn {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF3B30;
        text-shadow: 0 0 20px rgba(255, 59, 48, 0.3);
    }
    
    .metric-std {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #6C757D;
        margin-top: 8px;
    }
    
    /* Streamlit Tabs Glassmorphism overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        background: rgba(20, 20, 20, 0.5);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        padding: 0 20px;
        color: #777777;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #222222 !important;
        color: #D1FF00 !important;
        border-bottom: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Primary Button Styling overrides */
    .stButton > button {
        background-color: rgba(209, 255, 0, 0.1);
        color: #D1FF00;
        font-weight: 800;
        border-radius: 12px;
        border: 1px solid rgba(209, 255, 0, 0.5);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #D1FF00;
        color: #050505;
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(209, 255, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_forward_predictor():
    return ForwardV3Predictor()

@st.cache_resource
def load_reverse_designer():
    return ReverseV3Designer()

def load_precomputed_results():
    summary_path = Path("Reverse_outputs/reverse_refinement_summary.json")
    csv_path = Path("Reverse_outputs/reverse_refinement_candidate_1.csv")
    if not summary_path.exists() or not csv_path.exists():
        return None, None, None
        
    import json
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        
    top = summary["candidates"][0]
    df = pd.read_csv(csv_path)
    
    bc = {
        "label": top["label"] + " (Pre-computed Kaggle)",
        "LDMax_pred": top["LDMax_pred"],
        "ClMax_pred": top["ClMax_pred"],
        "CdMin_pred": top["CdMin_pred"],
        "LDMax_std": top["LDMax_std"],
        "ClMax_std": top["ClMax_std"],
        "CdMin_std": top["CdMin_std"],
        "CdMin_rel_std": top["CdMin_rel_std"],
        "passes_uncertainty": top["passes_uncertainty"],
        "geometry": {
            "x": df["x"].values,
            "y_upper": df["y_upper"].values,
            "y_lower": df["y_lower"].values,
            "thickness": df["thickness"].values,
            "camber": df["camber"].values,
        },
        "is_precomputed": True
    }
    
    target = summary.get("target", {"LDMax": 180.0, "ClMax": 1.85, "CdMin": 0.02})
    flow = summary.get("flow", {"Re": 2000000.0, "Mach": 0.1})
    
    return bc, target, flow


def plot_geometry(x, y_upper, y_lower, title="Airfoil Geometry"):
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    fig.patch.set_facecolor('#050505')
    ax.set_facecolor('#050505')
    
    # Glowing neon lines
    ax.plot(x, y_upper, label='Upper Surface', color='#D1FF00', linewidth=2.5, alpha=0.9)
    ax.plot(x, y_lower, label='Lower Surface', color='#00F0FF', linewidth=2.5, alpha=0.8)
    ax.plot(x, 0.5 * (y_upper + y_lower), '--', color='#444444', label='Camber Line', linewidth=1)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='solid', alpha=0.08, color='#FFFFFF')
    ax.set_xlabel('x/c', color='#888888', fontsize=9, family='monospace')
    ax.set_ylabel('y/c', color='#888888', fontsize=9, family='monospace')
    ax.tick_params(colors='#666666', labelsize=8)
    
    # Hide spines
    for spine in ax.spines.values():
        spine.set_color('#111111')
        
    ax.set_title(title, color='#FAFAFA', fontsize=12, pad=15, weight='bold')
    fig.tight_layout()
    return fig


def dat_to_csv_download(x, y_upper, y_lower):
    # Assemble to .dat format like XFOIL
    # Reverse upper for wrapping LE to TE
    x_upper_rev = x[::-1]
    y_upper_rev = y_upper[::-1]
    
    x_all = np.concatenate([x_upper_rev, x[1:]])
    y_all = np.concatenate([y_upper_rev, y_lower[1:]])
    
    df = pd.DataFrame({'x': x_all, 'y': y_all})
    csv = df.to_csv(index=False, header=False, sep='\t')
    return csv


def render_metric(title, value, uncertainty, rel_uncertainty=None, is_warn=False):
    val_class = "metric-value-warn" if is_warn else "metric-value"
    
    std_text = f"± {uncertainty:.4f}"
    if rel_uncertainty is not None:
        std_text += f" ({rel_uncertainty*100:.1f}%)"
        
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="{val_class}">{value:.4f}</div>
            <div class="metric-std">{std_text}</div>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.title("AeroML Dashboard ✈️")
    st.markdown("<p style='color: #9BA1A6;'>Physics-Informed Deep Learning for Airfoil Analysis & Design</p>", unsafe_allow_html=True)
    
    with st.spinner("Loading AI Models..."):
        try:
            forward_predictor = load_forward_predictor()
            reverse_designer = load_reverse_designer()
            
            # Load precomputed Kaggle outputs
            if 'best_candidate' not in st.session_state:
                bc, targ, fl = load_precomputed_results()
                if bc is not None:
                    st.session_state['best_candidate'] = bc
                    st.session_state['target'] = targ
                    st.session_state['flow'] = fl
                    
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}\nPlease check if artifacts are present in the search roots.")
            return

    tab1, tab2 = st.tabs(["🔮 Forward Prediction", "🛠️ Reverse Design"])

    # ------------------ FORWARD PREDICTION ------------------ #
    with tab1:
        st.header("Forward Analysis")
        st.markdown("Upload a standard `.dat` airfoil file to predict aerodynamic performance.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Flow Conditions")
            file_upload = st.file_uploader("Upload Airfoil .dat", type=["dat"])
            
            re_val = st.number_input("Reynolds Number (Re)", min_value=1e4, max_value=2e7, value=1e6, format="%e")
            mach_val = st.number_input("Mach Number", min_value=0.0, max_value=0.8, value=0.10, step=0.01)
            
            predict_btn = st.button("Predict Aerodynamics", type="primary", use_container_width=True)
            
        with col2:
            if file_upload is not None and predict_btn:
                # Save uploaded file to temp file to read it
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dat') as tmp:
                    tmp.write(file_upload.getvalue())
                    tmp_path = tmp.name
                    
                with st.spinner("Running Forward v3 Ensemble..."):
                    try:
                        result = forward_predictor.predict_from_dat_file(tmp_path, re_val, mach_val)
                        preds = result["predictions"]
                        unc = result["uncertainty"]
                        
                        st.subheader("Predictions")
                        mc1, mc2, mc3 = st.columns(3)
                        
                        with mc1:
                            render_metric("LDMax", preds["LDMax"], unc["LDMax_std"])
                        with mc2:
                            render_metric("ClMax", preds["ClMax"], unc["ClMax_std"])
                        with mc3:
                            is_warn = unc["CdMin_rel_std"] > 0.40
                            render_metric("CdMin", preds["CdMin"], unc["CdMin_std"], unc["CdMin_rel_std"], is_warn=is_warn)
                        
                        if unc["CdMin_rel_std"] > 0.40:
                            st.warning("⚠️ **Low Confidence on CdMin:** The ensemble standard deviation is high for the drag prediction. This often happens in deep low-drag regimes or edges of the training manifold.")
                        
                        # Plot Geometry
                        geom = result["geometry"]
                        
                        # To plot, we need to reconstruct coordinates from the dat file. 
                        # aeroml parser already normalizes it, let's use the raw normalize_coords from aeroml.
                        coords_raw = common.read_dat_file(tmp_path)
                        if coords_raw is not None:
                            coords_norm = common.normalize_coords(coords_raw)
                            upper, lower = common.split_upper_lower(coords_norm)
                            
                            fig = plt.figure(figsize=(10, 3), dpi=150)
                            fig.patch.set_facecolor('#0E1117')
                            ax = fig.add_subplot(111)
                            ax.set_facecolor('#0E1117')
                            ax.plot(upper[:,0], upper[:,1], color='#4facfe')
                            ax.plot(lower[:,0], lower[:,1], color='#00f2fe')
                            ax.set_aspect('equal')
                            for spine in ax.spines.values(): spine.set_color('#333333')
                            ax.tick_params(colors='white', labelsize=8)
                            ax.grid(True, linestyle=':', alpha=0.2, color='white')
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                
                # cleanup
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            elif predict_btn:
                st.info("Please upload a .dat file first.")
            else:
                st.info("Awaiting input...")

    # ------------------ REVERSE DESIGN ------------------ #
    with tab2:
        st.header("Reverse Airfoil Design")
        st.markdown("Specify operating conditions and aerodynamic targets. The system will search the latent space for a capable airfoil.")
        
        rcol1, rcol2 = st.columns([1, 2])
        
        with rcol1:
            st.subheader("Aerodynamic Targets")
            init_target = st.session_state.get('target', {"LDMax": 150.0, "ClMax": 1.5, "CdMin": 0.015})
            init_flow = st.session_state.get('flow', {"Re": 1e6, "Mach": 0.1})
            
            target_ld = st.number_input("Target LDMax", min_value=1.0, max_value=300.0, value=float(init_target["LDMax"]))
            target_cl = st.number_input("Target ClMax", min_value=0.1, max_value=3.0, value=float(init_target["ClMax"]))
            target_cd = st.number_input("Target CdMin", min_value=0.001, max_value=0.1, value=float(init_target["CdMin"]), format="%.4f")
            
            st.subheader("Operating Conditions")
            r_re = st.number_input("Design Re", min_value=1e4, max_value=2e7, value=float(init_flow["Re"]), format="%e")
            r_mach = st.number_input("Design Mach", min_value=0.0, max_value=0.8, value=float(init_flow["Mach"]), step=0.01)
            
            st.subheader("Search Mode")
            mode = st.selectbox("Search Mode", ["Fast Demo", "Balanced", "High Quality"], index=1, 
                                help="Fast Demo: Quick search. \nBalanced: Realistic defaults. \nHigh Quality: Deeper search space exploration.")
            
            if mode == "Fast Demo":
                n_restarts, opt_maxiter = 3, 15
            elif mode == "Balanced":
                n_restarts, opt_maxiter = 8, 35
            else:
                n_restarts, opt_maxiter = 16, 75
                
            search_btn = st.button("Search for Airfoil", type="primary", use_container_width=True)
            
        with rcol2:
            if search_btn:
                target = {"LDMax": target_ld, "ClMax": target_cl, "CdMin": target_cd}
                flow = {"Re": r_re, "Mach": r_mach}
                
                with st.spinner(f"Running Flow-Aware Reverse Search ({mode})..."):
                    res = reverse_designer.run_reverse_search(target, flow, n_restarts=n_restarts, opt_maxiter=opt_maxiter)
                    
                    feas = res["feasibility"]
                    st.write("### Search Diagnostics")
                    if feas["count"] < 100:
                        st.warning(f"⚠️ **Sparse Data Region:** Only {feas['count']} similar flow records found.")
                    else:
                        st.success(f"✓ Found {feas['count']} nearby flow records for initialization.")
                        
                    best_candidate = res["candidates"][0]
                    st.session_state['best_candidate'] = best_candidate
                    st.session_state['target'] = target
                    st.session_state['flow'] = flow
                    
            if 'best_candidate' in st.session_state:
                bc = st.session_state['best_candidate']
                st.subheader("Top Candidate: " + bc["label"])
                
                if not bc["passes_uncertainty"]:
                    st.error("⚠️ **Low Confidence Result:** This candidate violates our uncertainty thresholds. It may not perform as predicted in reality.")
                
                cm1, cm2, cm3 = st.columns(3)
                with cm1:
                    render_metric("Predicted LDMax", bc["LDMax_pred"], bc["LDMax_std"])
                with cm2:
                    render_metric("Predicted ClMax", bc["ClMax_pred"], bc["ClMax_std"])
                with cm3:
                    render_metric("Predicted CdMin", bc["CdMin_pred"], bc["CdMin_std"], bc["CdMin_rel_std"], is_warn=(not bc["passes_uncertainty"]))
                
                # Plot
                geom = bc["geometry"]
                fig = plot_geometry(geom["x"], geom["y_upper"], geom["y_lower"], title=f"Reverse Candidate ({bc['label']})")
                st.pyplot(fig)
                
                # Download
                csv_data = dat_to_csv_download(geom["x"], geom["y_upper"], geom["y_lower"])
                st.download_button("Download Candidate .dat file", data=csv_data, file_name=f"candidate_{bc.get('label', 'custom')}.dat", mime='text/plain')
                
                # Refinement option
                if not bc.get("is_precomputed", False):
                    st.divider()
                    st.write("### Local Refinement")
                    st.write("Optimize the current top candidate further using local gradient search. This takes longer but yields better target matching.")
                    
                    refine_btn = st.button("Refine Top Candidate", use_container_width=True)
                    if refine_btn:
                        with st.spinner("Refining..."):
                            targ = st.session_state['target']
                            fl = st.session_state['flow']
                            ref_res = reverse_designer.refine_candidate(bc, targ, fl)
                            best_ref = ref_res[0]
                            st.session_state['best_candidate'] = best_ref
                            st.rerun()
                else:
                    st.info("💡 **Pre-computed Output Shown:** You are looking at a cached result to save time. Enter new targets and click 'Search' to run a fresh live design pipeline.")

if __name__ == "__main__":
    main()
