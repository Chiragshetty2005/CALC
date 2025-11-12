# Heart disease model and UI

This workspace contains a trained RandomForest model and a small Flask frontend to present predictions by patient id.

How to run the web UI (dev):

1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Start the app:

   python app.py

4. Open http://127.0.0.1:5000 in your browser.

Notes:
- The app expects `final_model.pkl`, `model_metadata.json`, and `heart_disease_uci_encoded_with_id.csv` to be present in the project root.
- If you see an import error, install the missing package listed in `requirements.txt`.
