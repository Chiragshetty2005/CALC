from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.exceptions import HTTPException
import pandas as pd
import pickle
import json
import os
import logging
import traceback
from werkzeug.exceptions import HTTPException

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, 'final_model.pkl')
META_PATH = os.path.join(HERE, 'model_metadata.json')
DATA_PATH = os.path.join(HERE, 'heart_disease_uci_encoded_with_id.csv')
FI_PATH = os.path.join(HERE, 'feature_importances.csv')

app = Flask(__name__)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


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
    try:
        logger.info('/api/predict called')
        if DF is None or MODEL is None or META is None:
            logger.error('Missing resources: MODEL=%s META=%s DF=%s', MODEL is not None, META is not None, DF is not None)
            return jsonify({"error": "Model, metadata or data file not found."}), 500

        data = request.get_json() or {}
        id_val = data.get('patient_id') or data.get('id')
        id_col = data.get('id_col', 'id')

        logger.info('Request payload: id=%s id_col=%s', id_val, id_col)

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
                logger.warning('No matching row for %s=%s', id_col, id_val)
                return jsonify({"error": f"No patient found with {id_col} = {id_val}"}), 404
            row = matches.iloc[0]
        else:
            # treat id_val as 1-based row index
            try:
                idx = int(id_val) - 1
                row = DF.iloc[idx]
            except Exception as e:
                logger.exception('Index error locating row')
                return jsonify({"error": f"Could not locate id {id_val} using id-col '{id_col}'", "details": str(e)}), 400

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
        # convert encoded values to human-friendly labels for presentation
        patient_info = humanize_patient_info(patient_info)

        resp = {
            "patient_id": id_val,
            "id_col": id_col,
            "prediction": pred_label,
            "patient_info": patient_info,
            "reasons": reasons
        }
        logger.info('Prediction result: %s', resp)
        return jsonify(resp)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error('Unhandled exception in /api/predict: %s\n%s', e, tb)
        return jsonify({"error": "internal server error", "details": str(e), "trace": tb}), 500


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
    used_cols = set()

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
                        if orig in used_cols:
                            continue
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
            used_cols.add(found)
            # If this came from a one-hot / prefixed column like 'cp_non-anginal' or 'thal_normal',
            # prefer a readable suffix (e.g. 'non-anginal', 'normal') instead of the raw numeric flag.
            if label in ('ChestPainType', 'Thalassemia') and isinstance(found, str) and any(sep in found for sep in ('_', '-', ' ')):
                try:
                    v_check = row[found]
                    truthy = False
                    if isinstance(v_check, (int, float)):
                        truthy = float(v_check) != 0.0
                    else:
                        truthy = str(v_check).strip().lower() in ('1', 'true', 'yes')
                    if truthy:
                        display = found
                        if '_' in found:
                            display = found.split('_', 1)[1]
                        elif '-' in found:
                            display = found.split('-', 1)[1]
                        elif ' ' in found:
                            display = found.split(' ', 1)[1]
                        display = display.replace('_', ' ').replace('-', ' ')
                        info[label] = display
                except Exception:
                    pass

        # If still not found, try to detect one-hot / prefixed categorical columns
        if not found and label in ('ChestPainType', 'Thalassemia'):
            for cand in candidates:
                ncand = normalize(cand)
                for nkey, orig in col_map.items():
                    if orig in used_cols:
                        continue
                    if ncand in nkey or nkey.startswith(ncand):
                        # check if this column indicates a truthy one-hot flag
                        try:
                            v = row[orig]
                            truthy = False
                            if isinstance(v, (int, float)):
                                truthy = float(v) != 0.0
                            else:
                                sv = str(v).strip().lower()
                                truthy = sv in ('1', 'true', 'yes')
                        except Exception:
                            truthy = False
                        if truthy:
                            # derive a readable label from the column name suffix
                            display = orig
                            if '_' in orig:
                                display = orig.split('_', 1)[1]
                            elif '-' in orig:
                                display = orig.split('-', 1)[1]
                            elif ' ' in orig:
                                display = orig.split(' ', 1)[1]
                            display = display.replace('_', ' ').replace('-', ' ')
                            info[label] = display
                            used_cols.add(orig)
                            found = orig
                            break
                if found:
                    break

    # If some fields are missing, also append up to 3 other columns to give context
    if len(info) < len(display_order):
        added = 0
        for c in row.index:
            if added >= 3:
                break
            # skip if already included
            if c in used_cols:
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
            if c not in info and c not in used_cols:
                info[c] = pv
                used_cols.add(c)
                added += 1

    return info


def humanize_patient_info(info: dict) -> dict:
    """Convert encoded numeric values into human-friendly labels and format numbers."""
    out = {}
    # chest pain mapping (common UCI mapping variants)
    cp_map = {
        0: 'non-anginal', 1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal', 4: 'asymptomatic'
    }
    thal_map = {3: 'normal', 6: 'fixed defect', 7: 'reversible defect', 0: 'unknown'}
    for k, v in info.items():
        # None stays None (but convert to empty string for UI consistency)
        if v is None:
            out[k] = ''
            continue

        # determine a normalized key for checking
        key = k.lower()

        # Sex mapping (accept numeric or numeric-like values such as 1, 1.0, '1', '1.0')
        if key == 'sex':
            try:
                num = float(v)
                if int(num) == 1:
                    out[k] = 'Male'
                elif int(num) == 0:
                    out[k] = 'Female'
                else:
                    out[k] = str(v)
            except Exception:
                sv = str(v).strip().lower()
                if sv in ('male', 'm'):
                    out[k] = 'Male'
                elif sv in ('female', 'f'):
                    out[k] = 'Female'
                else:
                    out[k] = str(v)
            continue

        # Exercise angina mapping (handle numeric or numeric-like values)
        if key in ('exerciseangina', 'exercise_angina', 'exang'):
            try:
                num = float(v)
                if int(num) == 1:
                    out[k] = 'Yes'
                elif int(num) == 0:
                    out[k] = 'No'
                else:
                    out[k] = str(v)
            except Exception:
                sv = str(v).strip().lower()
                if sv in ('yes', 'y', 'true'):
                    out[k] = 'Yes'
                elif sv in ('no', 'n', 'false'):
                    out[k] = 'No'
                else:
                    out[k] = str(v)
            continue

        # Chest pain mapping
        if key in ('chestpaintype', 'chest_pain_type', 'cp'):
            try:
                num = int(float(v))
                out[k] = cp_map.get(num, str(v))
            except Exception:
                out[k] = str(v)
            continue

        # Thalassemia mapping
        if key in ('thalassemia', 'thal'):
            try:
                num = int(float(v))
                out[k] = thal_map.get(num, str(v))
            except Exception:
                out[k] = str(v)
            continue

        # Numeric vitals and counts: format them as readable strings (units added where sensible)
        try:
            # Treat integer-like floats as ints for nicer formatting
            if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer()):
                iv = int(v)
                if key == 'age':
                    out[k] = f"{iv} years"
                elif key in ('restingbp', 'resting_bp', 'restbp'):
                    out[k] = f"{iv} mmHg"
                elif key in ('cholesterol', 'chol'):
                    out[k] = f"{iv} mg/dL"
                elif key in ('maxheartrate', 'thalach', 'max_hr'):
                    out[k] = f"{iv} bpm"
                elif key in ('nummajorvessels', 'ca', 'numvessels'):
                    # friendly text for vessel counts
                    if iv == 0:
                        out[k] = 'none'
                    elif iv == 1:
                        out[k] = '1 vessel'
                    else:
                        out[k] = f"{iv} vessels"
                else:
                    out[k] = str(iv)
            elif isinstance(v, float):
                # show ST depression or other decimals with 2 dp and unit where sensible
                fv = round(float(v), 2)
                if key in ('st_depression', 'stdepression', 'oldpeak'):
                    out[k] = f"{fv} mm"
                else:
                    out[k] = str(fv)
            else:
                out[k] = str(v)
        except Exception:
            out[k] = str(v)

    # ensure every output value is a string (avoid sending numeric codes to UI)
    for kk in list(out.keys()):
        if out[kk] is None:
            out[kk] = ''
        else:
            out[kk] = str(out[kk])

    return out


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
    # convert encoded values to human-friendly labels for presentation
    patient_info = humanize_patient_info(patient_info)

    return render_template('result.html', patient_id=id_val, id_col=id_col, prediction=pred_label, raw=raw_pred, patient_info=patient_info, reasons=reasons)


@app.errorhandler(Exception)
def handle_all_exceptions(e):
    # Return JSON for all uncaught exceptions (including HTTPException)
    if isinstance(e, HTTPException):
        logger.warning('HTTP exception: %s', e)
        return jsonify({"error": e.description}), e.code
    tb = traceback.format_exc()
    logger.error('Unhandled exception: %s\n%s', e, tb)
    return jsonify({"error": "internal server error", "details": str(e)}), 500


if __name__ == '__main__':
    # Ensure the Flask app runs without the interactive debugger so we return JSON errors
    app.config['ENV'] = 'production'
    app.config['DEBUG'] = False
    app.debug = False
    # Disable the reloader to avoid multiple processes interfering with logging
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
