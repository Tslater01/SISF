# oversight_app.py
"""
An impressive Streamlit web application for human-in-the-loop (HITL) oversight
and visualization of the Self-Improving Safety Framework (SISF).
"""
import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"
PAGE_TITLE = "SISF Oversight Dashboard"
PAGE_ICON = "🛡️"
POLL_INTERVAL_SECONDS = 5

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Communication Functions ---
@st.cache_data(ttl=POLL_INTERVAL_SECONDS)
def get_policies_data():
    """Fetches all policies from the SISF API."""
    try:
        response = httpx.get(f"{API_BASE_URL}/v1/policies", timeout=10.0)
        response.raise_for_status()
        st.session_state.last_update_time = datetime.now()
        # The response now contains the 'is_active' state, no guessing needed.
        return response.json()
    except httpx.RequestError as e:
        st.error(f"Connection Error: Could not connect to the SISF API at {API_BASE_URL}. Is the server running? Details: {e}", icon="🚨")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}", icon="🔥")
        return None

def toggle_policy_status_api(policy_id: str, new_status: bool):
    """Sends a request to activate or deactivate a policy and refreshes the page."""
    try:
        response = httpx.post(f"{API_BASE_URL}/v1/policies/toggle/{policy_id}?active={new_status}", timeout=10.0)
        response.raise_for_status()
        st.toast(f"Policy {policy_id} status updated.", icon="✅")
        st.cache_data.clear() # Clear cache to force a refresh
        st.rerun() # Force an immediate page reload to show the change
    except Exception as e:
        st.error(f"Failed to toggle policy {policy_id}: {e}", icon="🚨")

# --- Sidebar ---
st.sidebar.header("Dashboard Controls")
auto_refresh = st.sidebar.toggle(f"Auto-Refresh (every {POLL_INTERVAL_SECONDS}s)", value=False)
if st.sidebar.button("Manual Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("System Status")
if 'last_update_time' in st.session_state and st.session_state.last_update_time:
    last_update_str = st.session_state.last_update_time.strftime("%H:%M:%S")
    st.sidebar.metric("Last API Update", last_update_str)
else:
    st.sidebar.warning("API connection not yet established.")

# --- Main Dashboard ---
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("Monitor and manage the autonomously generated safety policies of the SISF.")

policies = get_policies_data()

if policies is None: pass
elif not policies:
    st.info("No policies found in the store. Run `python main_loop.py` in another terminal to generate some.")
else:
    # --- KPIs and Visualizations ---
    st.header("System Overview")
    
    df_policies = pd.DataFrame(policies)
    total_policies = len(df_policies)
    # --- Calculate active policies from the real data ---
    active_policies_count = df_policies['is_active'].sum() if 'is_active' in df_policies.columns else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Policies Generated", total_policies)
    col2.metric("Currently Active Policies", active_policies_count)
    col3.metric("Breach Rate (Placeholder)", "N/A", help="This will be calculated in Phase 3")

    st.subheader("Policy Type Distribution")
    if total_policies > 0:
        type_counts = df_policies['type'].value_counts().reset_index()
        fig = px.pie(type_counts, names='type', values='count', 
                     title="Distribution of Policy Types",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.header("Manage Policies")

    for policy in sorted(policies, key=lambda p: p['id']):
        # --- Use the 'is_active' field from the API as the source of truth ---
        is_active = policy.get('is_active', False)
        status_icon = "🟢" if is_active else "⚪"
        
        with st.expander(f"{status_icon} Policy `{policy['id']}` ({policy['type']})"):
            st.markdown(f"**Description:** {policy['description']}")
            st.markdown(f"**Action:** `{policy['action']}`")
            if policy['type'] == 'HEURISTIC': st.code(f"Regex: {policy.get('regex_pattern', 'N/A')}", language="regex")
            elif policy['type'] == 'EMBEDDING_SIMILARITY': st.markdown(f"**Threshold:** `{policy.get('similarity_threshold', 'N/A')}`")

            st.markdown("---")
            st.write("**Manage Status:**")
            
            toggle_col1, toggle_col2 = st.columns(2)
            with toggle_col1:
                if st.button("Activate", key=f"act_{policy['id']}", use_container_width=True, disabled=is_active):
                    toggle_policy_status_api(policy['id'], True)
            with toggle_col2:
                if st.button("Deactivate", key=f"deact_{policy['id']}", use_container_width=True, disabled=not is_active):
                    toggle_policy_status_api(policy['id'], False)

# --- Auto-refresh logic ---
if auto_refresh:
    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()