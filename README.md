# redash-query-analyser

A simple web-based tool to fetch Redash query results, analyze them using pandas, and download the data locally.

This tool provides a user-friendly interface built with Streamlit to interact with your Redash instance without needing to write code for every analysis.

## Features

- **Fetch Data**: Input a Redash query URL and your API key to fetch the latest results.
- **View Data**: Display the query results in an interactive table.
- **Analyze Data**: Use a text box to run pandas expressions directly on the fetched data for quick analysis.
- **Download Data**: Download the full query result as an Excel (`.xlsx`) file with a single click.

## Setup and Installation

1.  **Clone the repository (or create the files):**
    ```bash
    git clone <your-repo-url>
    cd redash-query-analyser
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.7+ installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    Your browser will automatically open a new tab with the application running.
