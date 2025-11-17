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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



CONFIG_FILE = "config.json"

# --- Helper Functions ---

def save_config(api_key: str, groq_api_key: str, redash_base_url: str):
    """Saves the config to the config file."""
    config = {"api_key": api_key, "groq_api_key": groq_api_key, "redash_base_url": redash_base_url}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def load_config() -> dict:
    """Loads the config from the config file if it exists."""
    default_config = {"api_key": "", "groq_api_key": "", "redash_base_url": "https://common-redash.mmt.live"}
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
    # Use the dedicated 'refresh' endpoint to execute the query, which is more robust for parameterized queries.
    api_endpoint = f"{base_url}/api/queries/{query_id}/refresh"

    headers = {'Authorization': f'Key {api_key}', 'Content-Type': 'application/json'}
    payload = {"parameters": parameters} if parameters else {}
    
    try:
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

def get_llm_analysis_code(client: Groq, chat_history: list, df: pd.DataFrame) -> str:
    """
    Uses an LLM to convert a natural language question into executable Python code.
    """
    # Provide the LLM with the structure of the dataframe (head and column types)
    df_head = df.head().to_string()
    df_info = df.dtypes.to_string()

    prompt_data = {
        "role": "Expert Python Data Analyst",
        "task": "Generate Python code to answer a user's question based on a pandas DataFrame.",
        "context": {
            "dataframe_variable": "df",
            "available_libraries": ["pandas as pd", "numpy as np", "matplotlib.pyplot as plt", "seaborn as sns", "plotly.express as px"],
            "chat_history": chat_history,
            "dataframe_head": df_head,
            "dataframe_info": df_info
        },
        "instructions": [
            "CRITICAL: To avoid errors, `st.dataframe` and `st.plotly_chart` MUST have a unique `key` argument. You MUST generate this key using the `key_counter` variable provided (e.g., `key=f'element_{next(key_counter)}'`). Do NOT use hardcoded string keys like `key='my_chart'`. The `st.pyplot` function does NOT accept a key. Static elements like `st.markdown` or `st.error` also do not need a key.",
            "Your primary output should be a descriptive, multi-sentence textual answer written with `st.markdown()`. Only generate a chart (`st.pyplot`, `st.plotly_chart`) or a table (`st.dataframe`) if the user explicitly asks for one (e.g., 'show me a table', 'plot a bar chart of...').",
            "When the user asks about a column, you MUST try to infer the correct column name. First, check for a case-insensitive match in `df.columns`. If no match is found, check if any column name contains the user's term as a substring. Use your best judgment to select the most likely column. Only if you are very unsure after these checks should you ask for clarification using `st.markdown()`.",
            "Before performing mathematical operations, you MUST ensure the column is a numeric type. If it is an object/string type, you must first clean it and convert it to a numeric type using `pd.to_numeric(df['column'], errors='coerce')`.",
            "When plotting aggregated data (e.g., from `value_counts()` or `groupby()`), you must prepare it for plotting. If the data to be plotted is in the index, you MUST call `.reset_index()` to turn it into a column. If `.reset_index()` causes a `ValueError` because the column already exists, you should instead drop the index by calling `data.index.name = None`.",
            "For plotting, you must use Streamlit's charting functions. When using `matplotlib` or `seaborn`, you must create a figure and pass it to `st.pyplot()`. Example: `fig, ax = plt.subplots(); sns.histplot(df['column'], ax=ax); st.pyplot(fig)`. Do NOT use `plt.show()`. For `plotly`, use `st.plotly_chart(fig)`.",
            "Prefer using `seaborn` or `plotly.express` for creating plots as they are more visually appealing.",
            "CRITICAL: Your code MUST NOT perform any file I/O operations (e.g., reading from or writing to files like `data.csv`). All operations must be performed on the in-memory `df` DataFrame.",
            "If the question is unclear or cannot be answered with the given data, use `st.markdown()` to ask a clarification question."
        ],
        "output_format": {
            "format": "A single Python code block in markdown.",
            "example": "```python\n# your python code here\n```",
            "notes": "Do not include any explanations or text outside of the markdown code block."
        }
    }
    
    prompt = json.dumps(prompt_data, indent=2)

    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
    )
    response_content = chat_completion.choices[0].message.content
    # Extract code from the markdown block
    code_match = re.search(r"```python\n(.*?)```", response_content, re.DOTALL)
    return code_match.group(1).strip() if code_match else ""

# --- Streamlit App UI ---

st.set_page_config(page_title="Redash Query Analyser", layout="wide")

# --- Session State for Multi-Tab ---
if 'tabs' not in st.session_state:
    # Initialize with a single default tab
    st.session_state.next_tab_id = 1
    st.session_state.tabs = [
        {
            "id": 0,
            "data_source": "redash",
            "query_id_str": "",
            "params": [],
            "df": pd.DataFrame(),
            "title": "Query 1",
            "chat_history": [],
            "uploaded_filename": ""
        }
    ]
if 'active_tab_index' not in st.session_state:
    st.session_state.active_tab_index = 0

with st.sidebar:
    st.title("📊 Redash Query Analyser")
    st.write("Configure queries on the left, see results on the right.")
    st.divider()

    with st.expander("Connection Details", expanded=True):
        config = load_config()
        redash_base_url = st.text_input("Redash Base URL", value=config.get("redash_base_url", "https://common-redash.mmt.live"))
        api_key = st.text_input("Redash User API Key", value=config.get("api_key", ""), type="password")
        groq_api_key = st.text_input("Groq API Key", value=config.get("groq_api_key", ""), type="password", help="Get a free key from https://console.groq.com/keys")
        
        save_keys = st.checkbox("Save Connection Details for future use", value=(config.get("api_key") != "" or config.get("groq_api_key") != ""))

    st.divider()

    if st.button("➕ New Analysis", use_container_width=True):
        new_tab_id = st.session_state.next_tab_id
        st.session_state.tabs.append({
            "id": new_tab_id,
            "data_source": "redash",
            "query_id_str": "",
            "params": [],
            "df": pd.DataFrame(),
            "title": f"Query {len(st.session_state.tabs) + 1}",
            "chat_history": [],
            "uploaded_filename": ""
        })
        st.session_state.next_tab_id += 1
        # To make the new tab active, we would ideally set it here, but Streamlit's st.tabs
        # doesn't support programmatic switching. The user will see the new tab appear.
        # We will switch to it manually by setting the active index.
        st.session_state.active_tab_index = len(st.session_state.tabs) - 1
        st.rerun()

    st.divider()
    st.header("Analysis Tabs")

    # Use st.radio as a tab selector in the sidebar
    tab_titles = [tab["title"] for tab in st.session_state.tabs]
    # The index of the selected radio button will determine the active tab
    selected_title = st.radio("Select an analysis to view/edit:", tab_titles, index=st.session_state.active_tab_index, key="tab_selector")
    st.session_state.active_tab_index = tab_titles.index(selected_title)

def render_tab_content(tab_state):
    """Renders the UI and logic for a single analysis tab."""    
    
    # --- All input controls are now in the sidebar ---
    with st.sidebar:
        with st.expander(f"Controls for {tab_state['title']}", expanded=True):
            # --- Data Source Selection ---
            tab_state["data_source"] = st.selectbox(
                "Data Source",
                ["redash", "upload"],
                index=["redash", "upload"].index(tab_state["data_source"]),
                format_func=lambda x: "Redash Query" if x == "redash" else "Upload File",
                key=f"data_source_{tab_state['id']}"
            )

            # --- Close Tab Button ---

            if len(st.session_state.tabs) > 1:
                if st.button("❌ Close this Analysis", key=f"close_tab_{tab_state['id']}", help="Close this analysis tab", use_container_width=True):
                    st.session_state.tabs.remove(tab_state)
                    # Re-number the titles and reset active tab index
                    for i, tab in enumerate(st.session_state.tabs):
                        if tab["data_source"] == "redash":
                            tab["title"] = f"Query {i + 1}" if "(" not in tab["title"] else f"Query {i + 1} ({tab['title'].split('(')[1]}"
                        else:
                            tab["title"] = f"Upload {i + 1}" if "(" not in tab["title"] else f"Upload {i + 1} ({tab['title'].split('(')[1]}"
                    st.session_state.active_tab_index = 0
                    st.rerun()

            if tab_state["data_source"] == "redash":
                tab_state["query_id_str"] = st.text_input(
                    "Redash Query Number",
                    value=tab_state["query_id_str"],
                    placeholder="e.g., 71328",
                    key=f"query_id_{tab_state['id']}"
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload CSV or Excel file",
                    type=["csv", "xlsx", "xls"],
                    key=f"upload_{tab_state['id']}"
                )
                if uploaded_file is not None and st.button("Load File", use_container_width=True, key=f"load_file_{tab_state['id']}"):
                    with st.spinner("Loading file..."):
                        try:
                            if uploaded_file.name.endswith('.csv'):
                                tab_state["df"] = pd.read_csv(uploaded_file)
                            else:
                                tab_state["df"] = pd.read_excel(uploaded_file, engine='openpyxl')
                            tab_state["uploaded_filename"] = uploaded_file.name
                            tab_state["title"] = f"Upload {st.session_state.tabs.index(tab_state) + 1} ({uploaded_file.name})"
                            st.success(f"Loaded {len(tab_state['df'])} rows from {uploaded_file.name}.")
                        except Exception as e:
                            st.error(f"Failed to load file: {e}")

            if tab_state["data_source"] == "redash":
                query_id = int(tab_state["query_id_str"]) if tab_state["query_id_str"].isdigit() else None

                if tab_state["query_id_str"] and not query_id:
                    st.warning("Please enter a valid query number.")
                # --- Parameter and Fetch controls ---
                if query_id and api_key:
                    if st.button("Load Query Parameters", use_container_width=True, key=f"load_params_{tab_state['id']}"):
                        with st.spinner("Loading query parameters..."):
                            tab_state["params"] = get_query_details(redash_base_url, query_id, api_key)
                            if not tab_state["params"]:
                                st.info("This query has no parameters.")

                if tab_state["params"]:
                    st.write("---")
                    st.subheader("Query Parameters")
                    for param in tab_state["params"]:
                        param_name = param['name']
                        param_title = param.get('title', param_name)
                        param_type = param.get('type', 'text')
                        default_value = param.get('value')
                        
                        if param_type == 'date-range':
                            st.write(f"**{param_title}**")
                            start_key = f"param_{param_name}.start_{tab_state['id']}"
                            end_key = f"param_{param_name}.end_{tab_state['id']}"
                            st.text_input(
                                label="Start Date",
                                value=datetime.today().strftime('%Y-%m-%d'),
                                key=start_key
                            )
                            st.text_input(
                                label="End Date",
                                value=datetime.today().strftime('%Y-%m-%d'),
                                key=end_key
                            )
                        else:
                            param_key = f"param_{param_name}_{tab_state['id']}"
                            st.text_input(
                                label=f"{param_title} (Type: {param_type})",
                                value=str(default_value) if default_value is not None else "",
                                key=param_key
                            )

                if st.button("Fetch Query Results", use_container_width=True, key=f"fetch_results_{tab_state['id']}", type="primary"):
                    param_values = {}
                    # Construct param_values from session_state before making the API call
                    for param in tab_state.get("params", []):
                        param_name = param['name']
                        if param['type'] == 'date-range':
                            # Date-range parameters must be sent as a nested dictionary for the /refresh endpoint.
                            param_values[param_name] = {
                                "start": st.session_state.get(f"param_{param_name}.start_{tab_state['id']}", ""),
                                "end": st.session_state.get(f"param_{param_name}.end_{tab_state['id']}", "")
                            }
                        else:
                            # Simple parameters get the 'p_' prefix.
                            param_values[f"p_{param_name}"] = st.session_state.get(f"param_{param_name}_{tab_state['id']}", "")

                    if not query_id or not api_key:
                        st.warning("Please provide a valid Query Number and API Key.")
                    else:
                        if save_keys:
                            save_config(api_key, groq_api_key, redash_base_url)
                            st.toast("Connection Details saved!", icon="🔑")
                        redash_url = f"{redash_base_url}/queries/{query_id}"
                        with st.spinner(f"Fetching results for Query ID: {query_id}..."):
                            tab_state["df"] = get_query_results(redash_url, query_id, api_key, param_values)
                            tab_state["title"] = f"Query {st.session_state.tabs.index(tab_state) + 1} ({query_id})"

    # --- Main Content Area for the tab ---
    if not tab_state["df"].empty:
        st.success(f"Successfully loaded {len(tab_state['df'])} rows.")
        st.dataframe(tab_state["df"])

        # --- Download Section ---
        download_filename = f"{tab_state['uploaded_filename'].replace('.csv', '').replace('.xlsx', '').replace('.xls', '')}_results.xlsx" if tab_state["data_source"] == "upload" else f"redash_query_{tab_state['query_id_str']}_results.xlsx"
        st.download_button(
            label="📥 Download as Excel",
            data=to_excel(tab_state["df"]),
            file_name=download_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"download_{tab_state['id']}"
        )

    if not tab_state["df"].empty:
        # --- Analysis Section ---
        st.subheader("💬 Chat with your Data")

        # To keep the chat input bar at the bottom, the chat history must be in a scrollable container.
        chat_container = st.container(height=500)
        with chat_container:
            key_counter = iter(range(1000)) # Create a counter for unique keys
            # Display chat history
            for i, message in enumerate(tab_state["chat_history"]):
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        # Handle different types of assistant messages
                        if message["content"]["type"] == "code":
                            try:
                                exec(message["content"]["code"], {'df': tab_state["df"], 'st': st, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'px': px, 'key_counter': key_counter})
                            except KeyError as e:
                                st.error(f"Oops! I looked everywhere for a column named {e}, but it seems to be playing hide-and-seek. 🙈 Could you try asking about one of the columns I can see in the table above?")
                            except Exception as e:
                                st.error("Whoops! It looks like my circuits got a little crossed trying to answer that. 🤖 Sometimes I make a mistake. Could you try rephrasing your question? That usually helps me get back on track!")
                        elif message["content"]["type"] == "warning":
                            st.warning(message["content"]["message"])
                        elif message["content"]["type"] == "error":
                            st.error(message["content"]["message"])
                            if st.button("🔄 Retry", key=f"retry_{tab_state['id']}_{i}"):
                                # Remove the last error message and retry
                                tab_state["chat_history"].pop() 
                                handle_chat_submit(tab_state, groq_api_key, is_retry=True)
                    else: # For user messages, just display the text.
                        st.markdown(message["content"])

        # User input using st.chat_input for a WhatsApp-like experience
        if prompt := st.chat_input("Ask a question about the data..."):
            # Add user message to chat history for context
            tab_state["chat_history"].append({"role": "user", "content": prompt})
            handle_chat_submit(tab_state, groq_api_key)

    else:
        st.info("Welcome! Configure your query or upload a file in the sidebar to get started.")

def handle_chat_submit(tab_state, groq_api_key, is_retry=False):
    """Handles the logic for submitting a prompt to the AI and updating the chat history."""
    if not groq_api_key:
        tab_state["chat_history"].append({"role": "assistant", "content": {"type": "error", "message": "Please enter your Groq API Key in the sidebar to use this feature."}})
    else:
        with st.spinner("AI is thinking..."):
            try:
                client = Groq(api_key=groq_api_key)
                # If retrying, the user prompt is already in the history.
                analysis_code = get_llm_analysis_code(client, tab_state["chat_history"], tab_state["df"])
                if analysis_code:
                    tab_state["chat_history"].append({"role": "assistant", "content": {"type": "code", "code": analysis_code}})
                else:
                    tab_state["chat_history"].append({"role": "assistant", "content": {"type": "warning", "message": "The AI could not generate a response. Please try rephrasing your question."}})
            except Exception as e:
                tab_state["chat_history"].append({"role": "assistant", "content": {"type": "error", "message": f"An error occurred during analysis: {e}"}})
    
    if not is_retry:
        # Rerun the script to display the new messages inside the container
        st.rerun()
    else:
        # For retry, we need to rerun to show the new result
        st.rerun()

# --- Render the active tab's content ---
if 'tabs' in st.session_state and len(st.session_state.tabs) > st.session_state.active_tab_index:
    active_tab_state = st.session_state.tabs[st.session_state.active_tab_index]
    render_tab_content(active_tab_state)