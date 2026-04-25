import uproot
import numpy as np
import awkward as ak
import os

def load_root_file(path, max_events=None):
    """Loads ROOT file data into a structured dictionary."""
    try:
        # We don't use 'with' here because we want to keep the file open if needed,
        # but actually for streamlit it's better to load what we need and close it,
        # or use a persistent object. Let's load pair_info and keep the path for histograms.
        file = uproot.open(path)
        has_tgen = "tgen" in file
        has_t = "t" in file
        has_rntuple = "Events" in file or "Events;1" in file

        if not has_tgen and not has_t and not has_rntuple:
            file.close()
            return None, "No valid TTree ('tgen' or 't') or RNTuple ('Events') found."

        data = {
            'tgen': None,
            't': None,
            'rntuple': None,
            'num_events': 0,
            'pair_info': None,
            'file_path': path
        }

        if "selectedCandidatePairInfo" in file:
            data['pair_info'] = file["selectedCandidatePairInfo"].arrays(library="ak")

        if has_tgen:
            tree = file["tgen"]
            scalar_branches = ["RunNo", "EventNo", "nParticle", "source", "ThrustWithMissP", "TThetaWithMissP", "TPhiWithMissP", "missP", "missPhi", "missTheta"]

            jagged_branches = ["px", "py", "pz", "pt", "charge", "pid", "pt_wrtThrMissP", "theta_wrtThrMissP", "phi_wrtThrMissP"]
            
            existing_scalars = [f for f in scalar_branches if f in tree.keys()]
            existing_jagged = [f for f in jagged_branches if f in tree.keys()]
            
            scalar_data = tree.arrays(existing_scalars, library="ak", entry_stop=max_events)
            jagged_data = tree.arrays(existing_jagged, library="ak", entry_stop=max_events)
            
            event_data = {**{f: scalar_data[f] for f in scalar_data.fields}, 
                          **{f: jagged_data[f] for f in jagged_data.fields}}
            
            if "missP" in event_data and ("missPhi" not in event_data or "missTheta" not in event_data):
                px_sum = ak.sum(event_data["px"], axis=-1)
                py_sum = ak.sum(event_data["py"], axis=-1)
                pz_sum = ak.sum(event_data["pz"], axis=-1)
                event_data["missPx"] = -px_sum
                event_data["missPy"] = -py_sum
                event_data["missPz"] = -pz_sum
            elif "missPhi" in event_data and "missTheta" in event_data:
                r, theta, phi = event_data["missP"], event_data["missTheta"], event_data["missPhi"]
                event_data["missPx"] = r * np.sin(theta) * np.cos(phi)
                event_data["missPy"] = r * np.sin(theta) * np.sin(phi)
                event_data["missPz"] = r * np.cos(theta)

            data['tgen'] = ak.zip(event_data, depth_limit=1)
            data['num_events'] = len(data['tgen'])

        if has_t:
            tree = file["t"]
            scalar_branches = ["RunNo", "EventNo", "nParticle", "source", "ThrustWithMissP", "TThetaWithMissP", "TPhiWithMissP", "missP", "missPhi", "missTheta"]
            jagged_branches = ["px", "py", "pz", "pt", "charge", "pid", "pt_wrtThrMissP", "theta_wrtThrMissP", "phi_wrtThrMissP"]
            existing_scalars = [f for f in scalar_branches if f in tree.keys()]
            existing_jagged = [f for f in jagged_branches if f in tree.keys()]

            scalar_data = tree.arrays(existing_scalars, library="ak", entry_stop=max_events)
            jagged_data = tree.arrays(existing_jagged, library="ak", entry_stop=max_events)
            
            event_data = {**{f: scalar_data[f] for f in scalar_data.fields},
                          **{f: jagged_data[f] for f in jagged_data.fields}}

            if "missP" in event_data and ("missPhi" not in event_data or "missTheta" not in event_data):
                px_sum = ak.sum(event_data["px"], axis=-1)
                py_sum = ak.sum(event_data["py"], axis=-1)
                pz_sum = ak.sum(event_data["pz"], axis=-1)
                event_data["missPx"] = -px_sum
                event_data["missPy"] = -py_sum
                event_data["missPz"] = -pz_sum
            elif "missPhi" in event_data and "missTheta" in event_data:
                r, theta, phi = event_data["missP"], event_data["missTheta"], event_data["missPhi"]
                event_data["missPx"] = r * np.sin(theta) * np.cos(phi)
                event_data["missPy"] = r * np.sin(theta) * np.sin(phi)
                event_data["missPz"] = r * np.cos(theta)

            data['t'] = ak.zip(event_data, depth_limit=1)
            data['num_events'] = max(data['num_events'], len(data['t']))

        if has_rntuple:
            rntuple = file["Events"]
            keys = rntuple.keys()
            branches = []
            if "GenPart_pdgId" in keys: branches.append("GenPart_pdgId")
            if "GenPart_vector._0.fCoordinates.fX" in keys:
                 branches.extend(["GenPart_vector._0.fCoordinates.fX", "GenPart_vector._0.fCoordinates.fY", "GenPart_vector._0.fCoordinates.fZ"])
            
            if branches:
                r_data = rntuple.arrays(branches, entry_stop=max_events)
                px = r_data["GenPart_vector._0.fCoordinates.fX"]
                py = r_data["GenPart_vector._0.fCoordinates.fY"]
                pz = r_data["GenPart_vector._0.fCoordinates.fZ"]
                data['rntuple'] = ak.zip({
                    "pid": r_data["GenPart_pdgId"], "px": px, "py": py, "pz": pz, "pt": np.sqrt(px**2 + py**2)
                })
                data['num_events'] = max(data['num_events'], len(data['rntuple']))

        file.close()
        return data, None
    except Exception as e:
        return None, str(e)

def create_dummy_root_file(file_name="dummy_events.root"):
    """Creates a dummy ROOT file for testing."""
    if os.path.exists(file_name):
        return True, f"'{file_name}' already exists."
    try:
        with uproot.recreate(file_name) as file:
            num_events = 50
            events_data = {
                "RunNo": [], "EventNo": [], "nParticle": [], "px": [], "py": [], "pz": [], "pt": [], "charge": [], "pid": [],
                "ThrustWithMissP": [], "TThetaWithMissP": [], "TPhiWithMissP": [],
                "missP": [], "missPhi": [], "missTheta": [],
                "pt_wrtThrMissP": [], "theta_wrtThrMissP": [], "phi_wrtThrMissP": []
            }
            run_number = 12345
            for i in range(num_events):
                if i > 0 and i % 10 == 0: run_number += 1
                events_data["RunNo"].append(run_number); events_data["EventNo"].append(i + 101)
                n_particles = np.random.randint(10, 30); events_data["nParticle"].append(n_particles)
                px_vals, py_vals = np.random.normal(0, 5, n_particles), np.random.normal(0, 5, n_particles)
                events_data["px"].append(px_vals); events_data["py"].append(py_vals)
                events_data["pz"].append(np.random.normal(0, 10, n_particles)); events_data["pt"].append(np.sqrt(px_vals**2 + py_vals**2))
                events_data["charge"].append(np.random.choice([-1, 0, 1], n_particles, p=[0.45, 0.1, 0.45]))
                events_data["pid"].append(np.random.choice([211, -211, 11, -11, 22, 130], n_particles))
                events_data["ThrustWithMissP"].append(np.random.uniform(0.8, 1.2))
                events_data["TThetaWithMissP"].append(np.arccos(2 * np.random.uniform(0, 1) - 1))
                events_data["TPhiWithMissP"].append(np.random.uniform(0, 2 * np.pi))
                events_data["missP"].append(np.random.uniform(0, 5)); events_data["missPhi"].append(np.random.uniform(0, 2*np.pi)); events_data["missTheta"].append(np.arccos(2*np.random.uniform(0,1)-1))
                events_data["pt_wrtThrMissP"].append(np.random.gamma(2, 2, n_particles))
                events_data["theta_wrtThrMissP"].append(np.random.uniform(0, np.pi, n_particles))
                events_data["phi_wrtThrMissP"].append(np.random.uniform(-np.pi, np.pi, n_particles))
            file["tgen"] = {k: (ak.Array(v) if isinstance(v[0], (list, (np.ndarray, ak.Array))) else np.array(v)) for k, v in events_data.items()}
        return True, f"'{file_name}' created successfully."
    except Exception as e:
        return False, str(e)
