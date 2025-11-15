# 📊 Redash Query Analyser

This tool lets you connect to your company's Redash dashboards, pull data from a specific query, and then chat with an AI assistant in plain English to get insights, summaries, and charts from that data. No coding required!

## What is this for?

Imagine you have data in a Redash query (like sales numbers, user activity, etc.) and you want to ask questions about it without needing to write complex code. This tool is your personal data analyst.

-   **Connect**: Securely connect to your Redash query.
-   **Fetch**: Pull the latest data with the click of a button.
-   **Chat**: Ask questions like "What was the total sales last month?" or "Show me a bar chart of users by country" and get instant answers.

## Features

- **Simple Connection**: Just enter your Redash query number and API key.
- **Multi-Query Workspace**: Analyze multiple queries at once in separate, closable tabs.
- **Conversational AI**: Chat with your data in plain English. The AI understands follow-up questions and can generate charts on request.
- **Error Handling**: Friendly error messages and a "Retry" button make the experience smooth.
- **Download Data**: Easily download your query results as an Excel file.

## How to Run This (For Non-Coders)

You don't need to be a programmer to use this! Just follow these steps carefully.

### Step 1: Install Python

If you don't have Python on your computer, you'll need to install it.

1.  Go to the official Python website: python.org/downloads
2.  Download the latest version for your operating system (Windows or macOS).
3.  Run the installer.
    -   **On Windows**: **IMPORTANT!** On the first screen of the installer, make sure to check the box that says **"Add Python to PATH"**. This is crucial.
    -   **On macOS**: The default installation settings are fine.

### Step 2: Get the Code

1.  Go to the GitHub page for this repository.
2.  Click the green `<> Code` button.
3.  Click `Download ZIP`.
4.  Find the downloaded ZIP file on your computer and unzip it. This will create a folder named `redash-query-analyser-main` (or similar).

### Step 3: Open the Terminal

This is the command-line interface for your computer.

-   **On macOS**: Open the "Terminal" app (you can find it in Applications > Utilities, or by searching for it).
-   **On Windows**: Open the Start Menu, type `cmd`, and open the "Command Prompt" app.

### Step 4: Navigate to the Code Folder

1.  In the terminal, type `cd ` (that's `c`, `d`, and a space).
2.  Drag the unzipped code folder from your file explorer (Finder on Mac, File Explorer on Windows) and drop it directly into the terminal window. The path to the folder will appear.
3.  Press `Enter`. Your terminal is now "inside" the project folder.

### Step 5: Install the Required Tools

With your terminal still open and inside the project folder, copy and paste the following command and press `Enter`.

```bash
python -m pip install -r requirements.txt
```

You will see text scrolling as the necessary libraries are downloaded and installed.

### Step 6: Get Your API Keys

You need two secret keys to use the app. **Never share these with anyone.**

1.  **Redash API Key**:
    -   Log in to your Redash account.
    -   Click on your profile icon in the top right and go to your profile page.
    -   You will find your "API Key" there. Copy it.
2.  **Groq API Key** (for the AI):
    -   Go to console.groq.com/keys.
    -   Sign up for a free account.
    -   Click "Create API Key" and copy the new key.

### Step 7: Run the App!

In your terminal (which should still be in the project folder), run this final command:

```bash
python -m streamlit run app.py
```

Your web browser should automatically open a new tab with the Redash Query Analyser running.

### Step 8: Use the App

1.  Paste your Redash and Groq API keys into the input boxes on the left sidebar.
2.  Check the "Save API Keys" box so you don't have to enter them next time.
3.  Enter the number of the Redash query you want to analyze.
4.  Click "Fetch Query Results".
5.  Once the data loads, you can start chatting with the AI in the "Chat with your Data" section on the main page!
