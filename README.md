# StreetCred AI - How to Run Guide

This project consists of a Python Machine Learning backend (FastAPI) and a static HTML/JS frontend.

## Prerequisites
Ensure you have Python installed on your Windows machine.

1. Open a Command Prompt or PowerShell in this directory (C:\Streetcred).
2. Install the required dependencies by running:
   pip install -r requirements.txt

## The Easiest Way to Run (1-Click)
Simply double-click the run.bat file in this folder. 
- It will automatically launch the backend server in a command window.
- It will wait 3 seconds for the server to start, and then automatically open index.html in your default web browser.

## Running Manually
If you prefer to start the services manually:

**Step 1: Start the Backend Server**
Open a terminal in C:\Streetcred and run:
uvicorn backend:app --host 127.0.0.1 --port 8000
Leave this terminal window open. It serves the ML models to the frontend.

**Step 2: Open the Frontend Dashboard**
Double-click index.html to open it in your browser. 
Once it opens, you can select models, fill in the vendor's financial data, and click 'Generate Credit Score' to see the dynamic output.

## Training the Models (Optional)
If you ever want to re-train the machine learning models or re-generate the EDA and SHAP plots from the dataset, run:
python train_pipeline.py
This will overwrite the models in the models folder and update the graphs in the plots folder.
