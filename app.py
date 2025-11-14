import streamlit as st
import pandas as pd
import requests
import re
import io
import time
from datetime import datetime
import json
import os
from groq import Groq

CONFIG_FILE = "config.json"

# --- Helper Functions ---

def save_config(api_key: str, groq_api_key: str):
    """Saves the API keys to the config file."""
    config = {"api_key": api_key, "groq_api_key": groq_api_key}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def load_config() -> dict:
    """Loads the config from the config file if it exists."""
    default_config = {"api_key": "", "groq_api_key": ""}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Ensure all keys are present
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except (json.JSONDecodeError, IOError):
            return default_config
    return default_config

def to_excel(df: pd.DataFrame):
    """Converts a DataFrame to an in-memory Excel file."""
    output = io.BytesIO()
    # Use xlsxwriter as the engine to auto-adjust column widths
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        # Auto-adjust columns' width
        for column in df:
            column_length = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_length)
    processed_data = output.getvalue()
    return processed_data

def get_query_details(base_url: str, query_id: int, api_key: str) -> list:
    """Fetches query metadata to find its parameters."""
    api_endpoint = f"{base_url}/api/queries/{query_id}"
    headers = {'Authorization': f'Key {api_key}'}
    try:
        response = requests.get(api_endpoint, headers=headers, timeout=30)
        response.raise_for_status()
        query_info = response.json()
        return query_info.get("options", {}).get("parameters", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch query details: {e}")
        return []

def poll_job(base_url: str, job: dict, api_key: str) -> dict:
    """Polls a Redash job until it's completed."""
    headers = {'Authorization': f'Key {api_key}'}
    job_id = job['id']
    while job['status'] < 3: # 1: PENDING, 2: STARTED, 3: SUCCESS, 4: FAILURE, 5: CANCELLED
        response = requests.get(f"{base_url}/api/jobs/{job_id}", headers=headers, timeout=30)
        response.raise_for_status()
        job = response.json()['job']
        time.sleep(1)
    return job

def get_query_results(redash_url: str, query_id: int, api_key: str, parameters: dict) -> pd.DataFrame:
    """
    Fetches query results from Redash API and returns a pandas DataFrame.
    """
    # The base URL of your Redash instance (e.g., https://redash.yourcompany.com)
    base_url = "/".join(redash_url.split('/')[:3])
    # Use the dedicated 'refresh' endpoint to execute the query.
    api_endpoint = f"{base_url}/api/queries/{query_id}/refresh"

    # The /refresh endpoint expects parameters inside a 'parameters' object.
    headers = {'Authorization': f'Key {api_key}', 'Content-Type': 'application/json'}
    payload = {"parameters": parameters} if parameters else {}
    
    try:
        # Use POST to force a refresh, which works for parameterized queries.
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=180)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        data = response.json()
        
        # Handle job polling for long-running queries
        if 'job' in data:
            with st.spinner("Query is executing, waiting for results..."):
                job = poll_job(base_url, data['job'], api_key)
            
            if job['status'] == 4: # FAILURE
                st.error(f"Query execution failed: {job.get('error', 'No error details provided.')}")
                return pd.DataFrame()
            
            # After job success, fetch the actual results
            result_id = job['query_result_id']
            result_endpoint = f"{base_url}/api/query_results/{result_id}.json"
            response = requests.get(result_endpoint, headers=headers, timeout=180)
            response.raise_for_status()
            data = response.json()

        if 'query_result' not in data:
            st.error("Could not find 'query_result' in the API response.")
            return pd.DataFrame()

        rows = data['query_result']['data']['rows']
        columns = data['query_result']['data']['columns']
        column_names = [col['name'] for col in columns]
        return pd.DataFrame(rows, columns=column_names)

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(
                "API Request Failed: 404 Not Found. Please check the following:\n"
                "1. The Redash Query URL is correct and the query exists.\n"
                "2. The query has not been archived.\n"
                "3. Your API key has permission to access this query."
            )
        elif e.response.status_code == 401:
            st.error("API Request Failed: 401 Unauthorized. Please check if your API Key is correct and has not expired.")
        else:
            st.error(f"API Request Failed with status code {e.response.status_code}: {e}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"A network error occurred: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

def get_llm_analysis_code(client: Groq, question: str, df: pd.DataFrame) -> str:
    """
    Uses an LLM to convert a natural language question into executable Python code.
    """
    # Provide the LLM with the structure of the dataframe (head and column types)
    df_head = df.head().to_string()
    df_info = df.dtypes.to_string()

    prompt = f"""
    You are an expert Python data analyst. Your task is to generate Python code to answer a user's question based on a pandas DataFrame.

    **Instructions:**
    1.  The DataFrame is available in a variable named `df`.
    2.  The user's question is: "{question}"
    3.  The first 5 rows of the DataFrame (`df.head()`) are:
        ```
        {df_head}
        ```
    4.  The column data types (`df.dtypes`) are:
        ```
        {df_info}
        ```
    5.  Your code should produce a result that can be displayed. This could be a print statement, a DataFrame, a Series, or a Streamlit chart.
    6.  For plotting, **you must use Streamlit's charting functions**. For example: `st.bar_chart(data)`, `st.line_chart(data)`, `st.pyplot(fig)`. Do NOT use `plt.show()`.
    7.  Wrap your final Python code in a single markdown code block (e.g., ```python ... ```). Do not include any explanations outside of the code block.
    8.  If the question is unclear or cannot be answered with the given data, generate code that prints a clarification question, like `print("Could you please clarify what you mean by 'best performing'?")`.
    """

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    response_content = chat_completion.choices[0].message.content
    # Extract code from the markdown block
    code_match = re.search(r"```python\n(.*?)```", response_content, re.DOTALL)
    return code_match.group(1).strip() if code_match else ""

# --- Streamlit App UI ---

st.set_page_config(page_title="Redash Query Analyser", layout="wide")

st.title("📊 Redash Query Analyser")
st.write("Run Redash queries, analyze the results with pandas, and download them as Excel.")

# --- Session State for Multi-Tab ---
if 'tabs' not in st.session_state:
    # Initialize with a single default tab
    st.session_state.next_tab_id = 1
    st.session_state.tabs = [
        {
            "id": 0,
            "query_id_str": "",
            "params": [],
            "df": pd.DataFrame(),
            "title": "Query 1"
        }
    ]

with st.sidebar:
    st.header("Connection Details")
    # As requested, hardcode the base URL and only ask for the query number.
    REDASH_BASE_URL = "https://common-redash.mmt.live"
    
    config = load_config()
    api_key = st.text_input("Redash User API Key", value=config.get("api_key", ""), type="password")
    groq_api_key = st.text_input("Groq API Key", value=config.get("groq_api_key", ""), type="password", help="Get a free key from https://console.groq.com/keys")
    
    save_keys = st.checkbox("Save API Keys for future use", value=(config.get("api_key") != "" or config.get("groq_api_key") != ""))

    st.divider()

    if st.button("➕ New Analysis", use_container_width=True):
        new_tab_id = st.session_state.next_tab_id
        st.session_state.tabs.append({
            "id": new_tab_id,
            "query_id_str": "",
            "params": [],
            "df": pd.DataFrame(),
            "title": f"Query {len(st.session_state.tabs) + 1}"
        })
        st.session_state.next_tab_id += 1
        # To make the new tab active, we would ideally set it here, but Streamlit's st.tabs
        # doesn't support programmatic switching. The user will see the new tab appear.

def render_tab_content(tab_state):
    """Renders the UI and logic for a single analysis tab."""
    
    # --- Close Tab Button ---
    # Place it in columns to have it on the right side.
    if len(st.session_state.tabs) > 1:
        # This is the closest we can get to a browser-like tab closing experience with st.tabs
        _, col2 = st.columns([0.9, 0.1])
        with col2:
            if st.button("❌", key=f"close_tab_{tab_state['id']}", help="Close this analysis tab", use_container_width=True):
                st.session_state.tabs.remove(tab_state)
                # Re-number the titles of the remaining tabs
                for i, tab in enumerate(st.session_state.tabs):
                    tab["title"] = f"Query {i + 1}" if "(" not in tab["title"] else f"Query {i + 1} ({tab['title'].split('(')[1]}"
                st.rerun()

    tab_state["query_id_str"] = st.text_input(
        "Redash Query Number", 
        value=tab_state["query_id_str"], 
        placeholder="e.g., 71328",
        key=f"query_id_{tab_state['id']}"
    )

    query_id = int(tab_state["query_id_str"]) if tab_state["query_id_str"].isdigit() else None

    if query_id and api_key:
        if st.button("Load Query Parameters", use_container_width=True, key=f"load_params_{tab_state['id']}"):
            with st.spinner("Loading query parameters..."):
                tab_state["params"] = get_query_details(REDASH_BASE_URL, query_id, api_key)
                if not tab_state["params"]:
                    st.info("This query has no parameters.")
    elif tab_state["query_id_str"]:
        st.warning("Please enter a valid query number.")
    
    param_values = {}
    if tab_state["params"]:
        st.subheader("Query Parameters")
        for param in tab_state["params"]:
            param_name = param['name']
            param_title = param.get('title', param_name)
            param_type = param.get('type', 'text')
            default_value = param.get('value')
            
            # The Redash API expects date-range parameters to be split into two
            # separate parameters with .start and .end suffixes.
            if param_type == 'date-range':
                st.write(f"**{param_title}**") # Sub-header for the date range
                start_key = f"{param_name}.start"
                end_key = f"{param_name}.end"
                
                param_values[start_key] = st.text_input(
                    label="Start Date",
                    value=datetime.today().strftime('%Y-%m-%d'), # Default to today
                    key=f"param_{start_key}_{tab_state['id']}"
                )
                param_values[end_key] = st.text_input(
                    label="End Date",
                    value=datetime.today().strftime('%Y-%m-%d'), # Default to today
                    key=f"param_{end_key}_{tab_state['id']}"
                )
            else:
                # For all other parameter types, use a standard text input.
                param_values[param_name] = st.text_input(
                    label=f"{param_title} (Type: {param_type})",
                    value=str(default_value) if default_value is not None else "",
                    key=f"param_{param_name}_{tab_state['id']}"
                )

    if st.button("Fetch Query Results", use_container_width=True, key=f"fetch_results_{tab_state['id']}"):
        if not tab_state["query_id_str"] or not api_key:
            st.warning("Please provide both the Query Number and API Key.")
        else:
            if query_id:
                # Save the API key if the user requested it
                if save_keys:
                    save_config(api_key, groq_api_key)
                    st.toast("API Keys saved!", icon="🔑")

                redash_url = f"{REDASH_BASE_URL}/queries/{query_id}"
                with st.spinner(f"Fetching results for Query ID: {query_id}..."):
                    tab_state["df"] = get_query_results(redash_url, query_id, api_key, param_values)
                    # Update tab title with query number
                    current_index = st.session_state.tabs.index(tab_state)
                    # Keep the sequential numbering but add the query ID for clarity
                    tab_state["title"] = f"Query {current_index + 1} ({query_id})"
                    st.rerun() # Rerun to update the tab title in the UI
            else:
                st.error("Invalid Query Number. Please enter a valid number.")

    # --- Main Content Area for the tab ---
    if not tab_state["df"].empty:
        st.success(f"Successfully loaded {len(tab_state['df'])} rows.")
        st.dataframe(tab_state["df"])

        # --- Download Section ---
        st.download_button(
            label="📥 Download as Excel",
            data=to_excel(tab_state["df"]),
            file_name=f"redash_query_{tab_state['query_id_str']}_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"download_{tab_state['id']}"
        )

        # --- Analysis Section ---
        st.subheader("💬 Natural Language Analysis")
        st.info("Ask a question about the data in plain English. The AI will generate and run the code to get the answer.")
        user_query = st.text_area(
            "Your question:",
            height=100,
            placeholder="e.g., 'What are the top 10 countries by order count?' or 'Plot a bar chart of sales by category'",
            key=f"analysis_query_{tab_state['id']}"
        )

        if st.button("Analyze", use_container_width=True, key=f"analyze_{tab_state['id']}"):
            if not groq_api_key:
                st.error("Please enter your Groq API Key in the sidebar to use this feature.")
            elif user_query:
                with st.spinner("AI is analyzing your question..."):
                    try:
                        client = Groq(api_key=groq_api_key)
                        analysis_code = get_llm_analysis_code(client, user_query, tab_state["df"])
                        
                        if analysis_code:
                            st.markdown("---")
                            st.markdown("##### 🤖 Answer:")
                            # IMPORTANT: exec() is a security risk. Only use in trusted environments.
                            exec(analysis_code, {'df': tab_state["df"], 'st': st, 'pd': pd})
                        else:
                            st.warning("The AI could not generate code for your request. Please try rephrasing your question.")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
            else:
                st.warning("Please enter a pandas expression to analyze.")    
    elif query_id and len(tab_state["params"]) > 0 and not param_values:
        # This case handles when parameters are loaded but not yet filled.
        st.info("Fill in the required parameters above and click 'Fetch Query Results'.")
    else:
        st.info("Enter a Query Number and click 'Fetch Query Results' to begin.")

# --- Render all tabs ---
tab_titles = [tab["title"] for tab in st.session_state.tabs]
created_tabs = st.tabs(tab_titles)

for i, tab_ui in enumerate(created_tabs):
    with tab_ui:
        render_tab_content(st.session_state.tabs[i])