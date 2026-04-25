import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import awkward as ak
import os
import uproot
from data_utils import load_root_file, create_dummy_root_file

st.set_page_config(page_title="Particle Event Display", layout="wide")

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'event_index' not in st.session_state:
    st.session_state.event_index = 0
if 'file_path' not in st.session_state:
    st.session_state.file_path = ""
if 'selected_pair_idx' not in st.session_state:
    st.session_state.selected_pair_idx = 0

# --- Sidebar: Controls ---
st.sidebar.title("Controls")

# File Loading
default_file = "/home/tasheng/tpc/rootfiles/selected_events.root"
if not os.path.exists(default_file):
    default_file = "/home/tasheng/tpc/rootfiles/EnergyCut_ge_100_ntrk5_stratified_mcclosure_eff100_skim_all_sampled_1x.root"

file_path = st.sidebar.text_input("ROOT File Path", value=default_file)
if st.sidebar.button("Load ROOT File"):
    with st.spinner("Loading..."):
        data, error = load_root_file(file_path)
        if error:
            st.sidebar.error(f"Error: {error}")
        else:
            st.session_state.data = data
            st.session_state.file_path = file_path
            st.session_state.event_index = 0
            st.session_state.selected_pair_idx = 0
            st.sidebar.success("File loaded!")

if st.sidebar.button("Create Dummy File"):
    success, msg = create_dummy_root_file()
    if success: st.sidebar.success(msg)
    else: st.sidebar.error(msg)

if st.session_state.data:
    data = st.session_state.data
    num_events = data['num_events']
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Event Navigation")
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("< Prev") and st.session_state.event_index > 0:
        st.session_state.event_index -= 1
        st.session_state.selected_pair_idx = 0
    if col2.button("Next >") and st.session_state.event_index < num_events - 1:
        st.session_state.event_index += 1
        st.session_state.selected_pair_idx = 0
    
    st.session_state.event_index = st.sidebar.number_input("Event Index", min_value=0, max_value=num_events-1, value=st.session_state.event_index)
    
    # Jump by Run/Event
    st.sidebar.markdown("Jump to Event:")
    jump_run = st.sidebar.text_input("Run No")
    jump_event = st.sidebar.text_input("Event No")
    if st.sidebar.button("Jump"):
        source = data['tgen'] if data['tgen'] is not None else data['t']
        if source is not None:
            try:
                r, e = int(jump_run), int(jump_event)
                match = (source.RunNo == r) & (source.EventNo == e)
                indices = np.where(ak.to_numpy(match))[0]
                if len(indices) > 0:
                    st.session_state.event_index = int(indices[0])
                    st.session_state.selected_pair_idx = 0
                else:
                    st.sidebar.warning("Event not found.")
            except ValueError:
                st.sidebar.error("Invalid input.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Source")
    available_sources = []
    if data['tgen'] is not None: available_sources.append("Gen (tgen)")
    if data['t'] is not None: available_sources.append("Reco (t)")
    source_choice = st.sidebar.radio("Source", available_sources) if available_sources else None
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    pt_filter = st.sidebar.checkbox("Filter pT > 0.2 GeV", value=False)
    charge_options = st.sidebar.multiselect("Charges", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative"])
    exclude_pdgs = st.sidebar.text_input("Exclude PDG IDs (comma-sep)")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Options")
    show_pid = st.sidebar.checkbox("Show PID Labels", value=True)
    wrt_thrust = st.sidebar.checkbox("Draw wrt Thrust Axis", value=False)
    log_pt = st.sidebar.checkbox("Log Scale for pT_wrt", value=False)
    draw_miss_p = st.sidebar.checkbox("Draw Missing Momentum", value=True)

    # Main Display
    st.title(f"Event Display: {os.path.basename(st.session_state.file_path)}")
    
    ev_idx = st.session_state.event_index
    particles = None
    if source_choice == "Gen (tgen)": particles = data['tgen'][ev_idx]
    elif source_choice == "Reco (t)": particles = data['t'][ev_idx]
    
    # Summary info
    if particles is not None:
        m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
        m1.metric("Run", int(particles.RunNo))
        m2.metric("Event", int(particles.EventNo))
        m3.metric("nParticles", int(particles.nParticle))
        m4.metric("Miss P", f"{float(particles.missP):.2f}")
        # Use get() or check fields explicitly to avoid attribute errors on Awkward Records
        src_val = particles["source"] if "source" in particles.fields else "N/A"
        m5.metric("Source", f"{src_val}")
        if 'Thrust' in particles.fields: m6.metric("Thrust", f"{float(particles.Thrust):.3f}")
        if 'Sphericity' in particles.fields: m7.metric("Sphericity", f"{float(particles.Sphericity):.3f}")

    # --- Shared Logic for Masking and Highlighting ---
    mask = None
    highlight_indices = []
    run_no = int(particles.RunNo) if particles is not None else None
    event_no = int(particles.EventNo) if particles is not None else None
    event_pairs = None

    if particles is not None:
        mask = np.ones(len(particles['px']), dtype=bool)
        try: excluded_pdgids = {int(pid.strip()) for pid in exclude_pdgs.split(',') if pid.strip()}
        except ValueError: excluded_pdgids = set()
        
        if excluded_pdgids: mask &= ak.to_numpy(~ak.is_in(particles['pid'], list(excluded_pdgids)))
        if pt_filter: mask &= ak.to_numpy(particles['pt'] > 0.2)
        
        charge_mask = np.zeros_like(mask, dtype=bool)
        if "Positive" in charge_options: charge_mask |= ak.to_numpy(particles['charge'] > 0)
        if "Negative" in charge_options: charge_mask |= ak.to_numpy(particles['charge'] < 0)
        if "Neutral" in charge_options: charge_mask |= ak.to_numpy(particles['charge'] == 0)
        mask &= charge_mask

    if data['pair_info'] is not None and run_no is not None:
        mask_pairs = (data['pair_info'].runno == run_no) & (data['pair_info'].eventno == event_no)
        event_pairs = data['pair_info'][mask_pairs]
        if len(event_pairs) > 0:
            p_idx = st.session_state.selected_pair_idx
            if p_idx < len(event_pairs):
                pair = event_pairs[p_idx]
                highlight_indices = [int(pair.track1_index), int(pair.track2_index)]

    # Layout for plots
    col_main, col_side = st.columns([2, 1])

    with col_side:
        st.subheader("Candidate Pairs")
        if event_pairs is not None and len(event_pairs) > 0:
            pair_df_disp = ak.to_dataframe(event_pairs[['track1_index', 'track2_index', 'track1_pt_wrt_thrust', 'track2_pt_wrt_thrust', 'abs_deta', 'abs_dphi', 'peak_minus_valley_payload_density']])
            
            # Use dataframe selection
            selection = st.dataframe(
                pair_df_disp,
                on_select="rerun",
                selection_mode="single-row",
                width='stretch'
            )
            
            if selection and "rows" in selection.selection and len(selection.selection.rows) > 0:
                new_idx = selection.selection.rows[0]
                if new_idx != st.session_state.selected_pair_idx:
                    st.session_state.selected_pair_idx = new_idx
                    st.rerun()

            # Sync Dropdown
            st.session_state.selected_pair_idx = st.selectbox(
                "Select Pair (Sync)", 
                options=range(len(pair_df_disp)), 
                index=st.session_state.selected_pair_idx if st.session_state.selected_pair_idx < len(pair_df_disp) else 0
            )
        else:
            st.info("No candidate pairs for this event.")

    with col_main:
        # 3D Visualization
        fig = go.Figure()

        def get_color(charge, is_highlighted=False):
            if is_highlighted: return 'yellow'
            if charge > 0: return 'red'
            if charge < 0: return 'blue'
            return 'gray'

        # Thrust Axis
        thrust_axis = None
        if particles is not None and 'ThrustWithMissP' in particles.fields:
            r_t, theta_t, phi_t = float(particles.ThrustWithMissP), float(particles.TThetaWithMissP), float(particles.TPhiWithMissP)
            t_vec = np.array([r_t * np.sin(theta_t) * np.cos(phi_t), r_t * np.sin(theta_t) * np.sin(phi_t), r_t * np.cos(theta_t)])
            norm = np.linalg.norm(t_vec)
            if norm > 0: thrust_axis = t_vec / norm

        if wrt_thrust and particles is not None:
            pt = ak.to_numpy(particles['pt_wrtThrMissP'][mask])
            theta = ak.to_numpy(particles['theta_wrtThrMissP'][mask])
            phi = ak.to_numpy(particles['phi_wrtThrMissP'][mask])
            charges = ak.to_numpy(particles['charge'][mask])
            pids = ak.to_numpy(particles['pid'][mask])
            orig_indices = np.where(mask)[0]
            
            plot_pt = np.log10(pt) if log_pt else pt
            cos_theta = np.cos(theta)
            
            for i in range(len(pt)):
                is_h = orig_indices[i] in highlight_indices
                fig.add_trace(go.Scatter3d(
                    x=[0, plot_pt[i] * np.cos(phi[i])], y=[0, plot_pt[i] * np.sin(phi[i])], z=[0, cos_theta[i]],
                    mode='lines+text' if (show_pid or is_h) else 'lines',
                    line=dict(color=get_color(charges[i], is_h), width=6 if is_h else 4),
                    text=[None, str(pids[i])] if (show_pid or is_h) else None,
                    name=f"PID {pids[i]}" + (" (Selected)" if is_h else ""),
                    hoverinfo='text',
                    hovertext=f"PID: {pids[i]}<br>pT_wrt: {pt[i]:.2f}<br>cos_theta: {cos_theta[i]:.2f}<br>phi_wrt: {phi[i]:.2f}"
                ))
            fig.update_layout(scene=dict(xaxis_title='pT_wrt*cos(phi)', yaxis_title='pT_wrt*sin(phi)', zaxis_title='cos(theta)', zaxis=dict(range=[-1.1, 1.1])))
        else:
            if particles is not None:
                px, py, pz = ak.to_numpy(particles.px[mask]), ak.to_numpy(particles.py[mask]), ak.to_numpy(particles.pz[mask])
                charges, pids = ak.to_numpy(particles.charge[mask]), ak.to_numpy(particles.pid[mask])
                orig_indices = np.where(mask)[0]
                
                for i in range(len(px)):
                    is_h = orig_indices[i] in highlight_indices
                    fig.add_trace(go.Scatter3d(
                        x=[0, px[i]], y=[0, py[i]], z=[0, pz[i]],
                        mode='lines+text' if (show_pid or is_h) else 'lines',
                        line=dict(color=get_color(charges[i], is_h), width=6 if is_h else 4),
                        text=[None, str(pids[i])] if (show_pid or is_h) else None,
                        name=f"PID {pids[i]}" + (" (Selected)" if is_h else ""),
                        hoverinfo='text',
                        hovertext=f"PID: {pids[i]}<br>px: {px[i]:.2f}<br>py: {py[i]:.2f}<br>pz: {pz[i]:.2f}"
                    ))
                if thrust_axis is not None:
                    max_p = np.max(np.sqrt(px**2 + py**2 + pz**2)) if len(px) > 0 else 10
                    t = thrust_axis * max_p
                    fig.add_trace(go.Scatter3d(x=[-t[0], t[0]], y=[-t[1], t[1]], z=[-t[2], t[2]], mode='lines', line=dict(color='lime', width=3, dash='dash'), name="Thrust Axis"))
                if draw_miss_p and 'missPx' in particles.fields:
                    fig.add_trace(go.Scatter3d(x=[0, float(particles.missPx)], y=[0, float(particles.missPy)], z=[0, float(particles.missPz)], mode='lines', line=dict(color='cyan', width=6, dash='dot'), name="Missing P"))
            fig.update_layout(scene=dict(xaxis_title='px', yaxis_title='py', zaxis_title='pz'))

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=700)
        st.plotly_chart(fig, width='stretch')

    # Correlation Plot below candidates
    with col_side:
        st.subheader("Correlation Data")
        if event_pairs is not None and len(event_pairs) > 0:
            with uproot.open(st.session_state.file_path) as f:
                all_keys = f.keys(recursive=True)
                hist_key = None
                for k in all_keys:
                    if f"run{run_no}_evt{event_no}" in k and "signal2PCSelected" in k:
                        hist_key = k
                        break
                
                if hist_key:
                    h = f[hist_key]
                    v = h.values()
                    x_edges = h.axes[0].edges()
                    y_edges = h.axes[1].edges()
                    
                    fig_2pc = go.Figure(data=go.Heatmap(
                        z=v.T, x=x_edges, y=y_edges,
                        colorscale='Viridis'
                    ))
                    
                    # Highlight selected pair
                    p_idx = st.session_state.selected_pair_idx
                    if p_idx < len(event_pairs):
                        pair = event_pairs[p_idx]
                        fig_2pc.add_trace(go.Scatter(
                            x=[pair.deta], y=[pair.dphi_analysis_range],
                            mode='markers',
                            marker=dict(color='red', size=15, symbol='x', line=dict(color='white', width=2)),
                            name="Selected Pair"
                        ))

                    fig_2pc.update_layout(
                        title=f"2PC: {hist_key.split('/')[-1]}", 
                        xaxis_title="dEta", yaxis_title="dPhi", 
                        height=400, margin=dict(l=0, r=0, b=0, t=30),
                        showlegend=False
                    )
                    st.plotly_chart(fig_2pc, width='stretch')
                else:
                    st.info("No signal2PC histogram found for this event.")
        else:
            st.info("No candidate pairs for this event.")

    # Particle Info Table
    if particles is not None:
        st.subheader("Particle Details")
        n_p = len(particles['px'])
        jagged_data = {}
        for field in particles.fields:
            val = particles[field]
            if hasattr(val, "__len__") and not isinstance(val, (str, bytes)) and len(val) == n_p:
                jagged_data[field] = ak.to_numpy(val[mask])
        
        if jagged_data:
            df = pd.DataFrame(jagged_data)
            orig_indices = np.where(mask)[0]
            
            # Function to highlight rows
            def highlight_selected_tracks(row):
                orig_idx = orig_indices[row.name]
                if orig_idx in highlight_indices:
                    return ['background-color: #555500; color: white'] * len(row)
                return [''] * len(row)

            st.dataframe(df.style.apply(highlight_selected_tracks, axis=1), width='stretch')
else:
    st.info("Please load a ROOT file from the sidebar.")
