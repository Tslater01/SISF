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
POLL_INTERVAL_SECONDS = 5  # <-- THE FIX IS HERE

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- State Management ---
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'policies_data' not in st.session_state:
    st.session_state.policies_data = [] # Cache fetched data
if 'active_status_override' not in st.session_state:
     st.session_state.active_status_override = {} # Track local toggle state

# --- API Communication Functions ---
@st.cache_data(ttl=POLL_INTERVAL_SECONDS) # Use the variable for cache TTL
def get_policies_data():
    """Fetches all policies from the SISF API."""
    try:
        response = httpx.get(f"{API_BASE_URL}/v1/policies", timeout=10.0)
        response.raise_for_status()
        st.session_state.last_update_time = datetime.now()
        policies = response.json()
        for policy in policies:
             if policy['id'] in st.session_state.active_status_override:
                 policy['is_active'] = st.session_state.active_status_override[policy['id']]
             else:
                 policy['is_active'] = True 
        return policies
    except httpx.RequestError as e:
        st.error(f"Connection Error: Could not connect to the SISF API at {API_BASE_URL}. Is the server running? Details: {e}", icon="🚨")
        return None
    except httpx.HTTPStatusError as e:
         st.error(f"API Error: Received status {e.response.status_code}. Details: {e.response.text}", icon="🔥")
         return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching policies: {e}", icon="🔥")
        return None

def toggle_policy_status_api(policy_id: str, new_status: bool):
    """Sends a request to activate or deactivate a policy."""
    try:
        response = httpx.post(f"{API_BASE_URL}/v1/policies/toggle/{policy_id}?active={new_status}", timeout=10.0)
        response.raise_for_status()
        st.toast(f"Policy {policy_id} status update requested.", icon="✅")
        st.session_state.active_status_override[policy_id] = new_status
        st.cache_data.clear()
        return True
    except Exception as e:
        st.error(f"Failed to toggle policy {policy_id}: {e}", icon="🚨")
        return False

# --- Sidebar ---
st.sidebar.header("Dashboard Controls")
auto_refresh = st.sidebar.toggle(f"Auto-Refresh (every {POLL_INTERVAL_SECONDS}s)", value=False)
if st.sidebar.button("Manual Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("System Status")
if st.session_state.last_update_time:
    last_update_str = st.session_state.last_update_time.strftime("%H:%M:%S")
    st.sidebar.metric("Last API Update", last_update_str)
else:
    st.sidebar.warning("API connection not yet established.")

# --- Main Dashboard ---
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("Monitor and manage the autonomously generated safety policies of the SISF.")

policies = get_policies_data()

if policies is None:
    pass
elif not policies:
    st.info("No policies found in the store. Run `python main_loop.py` in another terminal to generate some.")
else:
    st.header("System Overview")
    
    total_policies = len(policies)
    active_policies_count = sum(1 for p in policies if p.get('is_active', True)) 
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Policies Generated", total_policies)
    col2.metric("Currently Active Policies", active_policies_count)
    col3.metric("Breach Rate (Placeholder)", "N/A", help="This will be calculated in Phase 3")

    st.subheader("Policy Type Distribution")
    if total_policies > 0:
        df_policies = pd.DataFrame(policies)
        type_counts = df_policies['type'].value_counts().reset_index()
        type_counts.columns = ['type', 'count']

        fig = px.pie(type_counts, names='type', values='count', 
                     title="Distribution of Policy Types",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(legend_title_text='Policy Type')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.header("Manage Policies")

    for policy in sorted(policies, key=lambda p: p['id']):
        policy_id = policy['id']
        is_active = policy.get('is_active', True)
        status_icon = "🟢" if is_active else "⚪"
        
        with st.expander(f"{status_icon} Policy `{policy_id}` ({policy['type']})"):
            st.markdown(f"**Description:** {policy['description']}")
            st.markdown(f"**Action:** `{policy['action']}`")

            if policy['type'] == 'HEURISTIC':
                st.code(f"Regex Pattern: {policy.get('regex_pattern', 'N/A')}", language="regex")
            elif policy['type'] == 'EMBEDDING_SIMILARITY':
                st.markdown(f"**Similarity Threshold:** `{policy.get('similarity_threshold', 'N/A')}`")

            st.markdown("---")
            st.write("**Manage Status:**")
            
            toggle_col1, toggle_col2 = st.columns(2)
            
            with toggle_col1:
                if st.button("Activate", key=f"act_{policy_id}", use_container_width=True, disabled=is_active):
                    toggle_policy_status_api(policy_id, True)
                    st.rerun()

            with toggle_col2:
                if st.button("Deactivate", key=f"deact_{policy_id}", use_container_width=True, disabled=not is_active):
                    toggle_policy_status_api(policy_id, False)
                    st.rerun()

# --- Auto-refresh logic ---
if auto_refresh:
    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()