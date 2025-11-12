from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import pickle
import json
import os

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, 'final_model.pkl')
META_PATH = os.path.join(HERE, 'model_metadata.json')
DATA_PATH = os.path.join(HERE, 'heart_disease_uci_encoded_with_id.csv')
FI_PATH = os.path.join(HERE, 'feature_importances.csv')

app = Flask(__name__)


def load_resources():
    model = None
    meta = None
    df = None
    fi = None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    if os.path.exists(FI_PATH):
        try:
            fi = pd.read_csv(FI_PATH)
        except Exception:
            fi = None
    return model, meta, df, fi


MODEL, META, DF, FI = load_resources()


@app.route('/', methods=['GET'])
def index():
    # pass a small sample of ids for client convenience
    id_col = 'id' if 'id' in DF.columns else None
    sample_ids = []
    if id_col is not None:
        sample_ids = DF[id_col].dropna().astype(int).head(100).tolist()
    # render index and provide sample ids to the client for in-page autocomplete
    return render_template('index.html', sample_ids=sample_ids)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint that returns JSON result for a given patient id.
    Request JSON: {"patient_id": <id>, "id_col": "id"}
    """
    if DF is None or MODEL is None or META is None:
        return jsonify({"error": "Model, metadata or data file not found."}), 500

    data = request.get_json() or {}
    id_val = data.get('patient_id') or data.get('id')
    id_col = data.get('id_col', 'id')

    if id_val is None or str(id_val).strip() == '':
        return jsonify({"error": "patient_id is required"}), 400

    # try integer conversion if possible
    try:
        id_val_int = int(id_val)
        id_val = id_val_int
    except Exception:
        pass

    # locate row
    if id_col in DF.columns:
        matches = DF[DF[id_col] == id_val]
        if matches.empty:
            return jsonify({"error": f"No patient found with {id_col} = {id_val}"}), 404
        row = matches.iloc[0]
    else:
        # treat id_val as 1-based row index
        try:
            idx = int(id_val) - 1
            row = DF.iloc[idx]
        except Exception:
            return jsonify({"error": f"Could not locate id {id_val} using id-col '{id_col}'"}), 400

    # Build feature vector
    feature_names = META.get('feature_names', [])
    if not feature_names and hasattr(MODEL, 'feature_names_in_'):
        feature_names = list(MODEL.feature_names_in_)

    try:
        X = row.reindex(feature_names).to_frame().T
    except Exception:
        X = row.to_frame().T

    raw_pred = MODEL.predict(X)[0]
    label_map = META.get('label_map', {"0":"No Heart Disease","1":"Heart Disease"})
    pred_label = label_map.get(str(raw_pred), label_map.get(raw_pred, str(raw_pred)))

    reasons = []
    if FI is not None and 'feature' in FI.columns:
        reasons = FI['feature'].astype(str).head(5).tolist()
    elif feature_names:
        reasons = feature_names[:5]

    patient_info = prepare_patient_info(row)

    return jsonify({
        "patient_id": id_val,
        "id_col": id_col,
        "prediction": pred_label,
        "raw": int(raw_pred),
        "patient_info": patient_info,
        "reasons": reasons
    })


def prepare_patient_info(row):
    # Robustly select a small subset of human-friendly fields if available.
    # Normalize column names (lowercase, alphanumeric) to match variants like
    # 'RestingBP', 'resting_bp', 'restingbp', etc.
    def normalize(s):
        return ''.join(ch.lower() for ch in str(s) if ch.isalnum())

    # Map dataset columns to friendly display labels in a specific order.
    # Desired order and labels (matching the screenshot):
    display_order = [
        ('Age', ['age']),
        ('Sex', ['sex', 'gender']),
        ('RestingBP', ['restingbp', 'resting_bp', 'restbp', 'restingbloodpressure']),
        ('Cholesterol', ['chol', 'cholesterol']),
        ('MaxHeartRate', ['thalach', 'maxheartrate', 'max_hr', 'thalch']),
        ('ExerciseAngina', ['exerciseangina', 'exang', 'exercise_angina']),
        ('ST_Depression', ['stdepression', 'oldpeak', 'st_depression']),
        ('NumMajorVessels', ['num_major_vessels', 'ca', 'numvessels']),
        ('ChestPainType', ['chestpaintype', 'cp', 'cp_type', 'chest_pain_type']),
        ('Thalassemia', ['thal', 'thalassemia'])
    ]

    col_map = {normalize(c): c for c in row.index}
    info = {}

    for label, candidates in display_order:
        found = None
        for cand in candidates:
            n = normalize(cand)
            if n in col_map:
                found = col_map[n]
                break
        # substring fallback
        if not found:
            for nkey, orig in col_map.items():
                for cand in candidates:
                    if normalize(cand) in nkey or nkey in normalize(cand):
                        found = orig
                        break
                if found:
                    break
        if found:
            val = row[found]
            try:
                if pd.isna(val):
                    pyval = None
                elif hasattr(val, 'item'):
                    pyval = val.item()
                else:
                    pyval = val
            except Exception:
                pyval = str(val)
            info[label] = pyval

    # If some fields are missing, also append up to 3 other columns to give context
    if len(info) < len(display_order):
        added = 0
        for c in row.index:
            if added >= 3:
                break
            # skip if already included
            if any(c in orig for orig in info.values()):
                continue
            v = row[c]
            try:
                if pd.isna(v):
                    pv = None
                elif hasattr(v, 'item'):
                    pv = v.item()
                else:
                    pv = v
            except Exception:
                pv = str(v)
            # use the original column name as label
            if c not in info:
                info[c] = pv
                added += 1

    return info


@app.route('/predict', methods=['POST'])
def predict():
    if DF is None or MODEL is None or META is None:
        return render_template('error.html', message='Model, metadata or data file not found. Ensure final_model.pkl, model_metadata.json and heart_disease_uci_encoded_with_id.csv exist in the project folder.')

    id_str = request.form.get('patient_id', '').strip()
    id_col = request.form.get('id_col', 'id')
    if not id_str:
        return redirect(url_for('index'))

    # try integer
    try:
        id_val = int(id_str)
    except ValueError:
        id_val = id_str

    # find the row
    if id_col in DF.columns:
        mask = DF[id_col] == id_val
    else:
        # if id column not present, treat id as 1-based row index
        try:
            idx = int(id_val) - 1
            row = DF.iloc[idx]
            mask = None
        except Exception:
            return render_template('error.html', message=f'Could not locate id {id_val} using id-col "{id_col}"')

    if 'mask' in locals() and mask is not None:
        matches = DF[mask]
        if matches.empty:
            return render_template('error.html', message=f'No patient found with {id_col} = {id_val}')
        row = matches.iloc[0]

    # Build feature vector in correct order
    feature_names = META.get('feature_names', [])
    # if feature names missing, try using model's feature_names_in_
    if not feature_names and hasattr(MODEL, 'feature_names_in_'):
        feature_names = list(MODEL.feature_names_in_)

    X = None
    try:
        X = row.reindex(feature_names).to_frame().T
    except Exception:
        # fallback: attempt to drop label column if present and use whole row
        X = row.to_frame().T

    # predict
    raw_pred = MODEL.predict(X)[0]
    label_map = META.get('label_map', {"0":"No Heart Disease","1":"Heart Disease"})
    # label_map may be mapping of strings; ensure key type matches
    pred_label = label_map.get(str(raw_pred), label_map.get(raw_pred, str(raw_pred)))

    # prepare short reason: use feature_importances.csv if available else meta
    reasons = []
    if FI is not None and 'feature' in FI.columns:
        reasons = FI['feature'].astype(str).head(5).tolist()
    elif feature_names:
        reasons = feature_names[:5]

    patient_info = prepare_patient_info(row)

    return render_template('result.html', patient_id=id_val, id_col=id_col, prediction=pred_label, raw=raw_pred, patient_info=patient_info, reasons=reasons)


if __name__ == '__main__':
    # simple dev server
    app.run(host='127.0.0.1', port=5000, debug=True)
