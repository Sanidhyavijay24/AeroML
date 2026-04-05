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
    /* Dark Theme Optimization */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    .metric-card {
        background: rgba(30, 30, 40, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.2);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #9BA1A6;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-value-warn {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #fceABB 0%, #f8b500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-std {
        font-size: 0.85rem;
        color: #6C757D;
        margin-top: 5px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #9BA1A6;
    }
    
    .stTabs [aria-selected="true"] {
        color: #FAFAFA !important;
        border-bottom: 3px solid #00f2fe !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_forward_predictor():
    return ForwardV3Predictor()

@st.cache_resource
def load_reverse_designer():
    return ReverseV3Designer()


def plot_geometry(x, y_upper, y_lower, title="Airfoil Geometry"):
    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    ax.plot(x, y_upper, label='Upper Surface', color='#4facfe', linewidth=2)
    ax.plot(x, y_lower, label='Lower Surface', color='#00f2fe', linewidth=2)
    ax.plot(x, 0.5 * (y_upper + y_lower), '--', color='#6e757c', label='Camber Line', linewidth=1)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.2, color='white')
    ax.set_xlabel('x/c', color='white', fontsize=9)
    ax.set_ylabel('y/c', color='white', fontsize=9)
    ax.tick_params(colors='white', labelsize=8)
    
    # Hide spines
    for spine in ax.spines.values():
        spine.set_color('#333333')
        
    ax.set_title(title, color='white', fontsize=12, pad=15)
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
            target_ld = st.number_input("Target LDMax", min_value=1.0, max_value=300.0, value=150.0)
            target_cl = st.number_input("Target ClMax", min_value=0.1, max_value=3.0, value=1.5)
            target_cd = st.number_input("Target CdMin", min_value=0.001, max_value=0.1, value=0.015, format="%.4f")
            
            st.subheader("Operating Conditions")
            r_re = st.number_input("Design Re", min_value=1e4, max_value=2e7, value=1e6, format="%e")
            r_mach = st.number_input("Design Mach", min_value=0.0, max_value=0.8, value=0.1, step=0.01)
            
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
                st.download_button("Download Candidate .dat file", data=csv_data, file_name=f"candidate_{bc['label']}.dat", mime='text/plain')
                
                # Refinement option
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

if __name__ == "__main__":
    main()
