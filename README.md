# Particle Event Display Web UI

A Streamlit-based web application for visualizing particle collider events from ROOT files.

## Features

- **Interactive 3D Visualization**: Built with Plotly for smooth rotation, zooming, and hover info.
- **Support for Multiple Formats**: Reads TTree (`tgen`, `t`) and RNTuple (`Events`) formats.
- **Dual Views**:
    - **Cartesian**: Standard $p_x, p_y, p_z$ view.
    - **Cylindrical**: Visualization with respect to the thrust axis ($p_T^{wrt}, \cos(\theta_{wrt}), \phi_{wrt}$).
- **Event Navigation**: Next/Prev, Jump by index, or Jump by Run/Event number.
- **Advanced Filtering**: Filter by $p_T$, charge, or specific PDG IDs.
- **Detailed Info**: View full particle properties in a searchable table.

## How to Run

1. Ensure you have the required dependencies:
   ```bash
   pip install streamlit uproot awkward numpy pandas plotly
   ```

2. Run the application from this directory:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to the URL displayed in the terminal (usually `http://localhost:8501`).
