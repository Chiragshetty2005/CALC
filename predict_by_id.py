import argparse
import json
import pickle
import pandas as pd
import sys


def load_model_and_metadata(model_path="final_model.pkl", metadata_path="model_metadata.json"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def predict_by_id(csv_path, patient_id, model, metadata, id_col=None):
    df = pd.read_csv(csv_path)

    if id_col:
        if id_col not in df.columns:
            print(f"id column '{id_col}' not found in CSV columns: {df.columns.tolist()}")
            sys.exit(1)
        matches = df[df[id_col] == patient_id]
        if matches.empty:
            print(f"No rows found where {id_col} == {patient_id}")
            sys.exit(1)
        if len(matches) > 1:
            print(f"Multiple rows ({len(matches)}) found where {id_col} == {patient_id}; using the first match.")
        row = matches.iloc[0]
    else:
        if not (isinstance(patient_id, int) and 1 <= patient_id <= len(df)):
            print(f"Patient id {patient_id} is out of range. CSV has {len(df)} rows.")
            sys.exit(1)
        row = df.iloc[patient_id - 1]

    feature_names = metadata.get("feature_names")
    if feature_names is None:
        print("No feature_names found in metadata.")
        sys.exit(1)

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        print("The input CSV is missing required feature columns:", missing)
        sys.exit(1)

    # Prepare a DataFrame with correct feature names to avoid sklearn "feature names" warning
    df_input = pd.DataFrame([row[feature_names].values], columns=feature_names)
    pred = model.predict(df_input)[0]
    label_map = metadata.get("label_map", {"0": "No Heart Disease", "1": "Heart Disease"})
    human_label = label_map.get(str(int(pred)), str(int(pred)))

    print(f"\nPatient lookup ({'col='+id_col if id_col else 'row number'}): {patient_id} -> Prediction: {human_label}")

    # Print basic patient info only (no numeric comparisons)
    print("\nBasic patient info:")
    patient_row = row.copy()
    # helper extractors
    def get_cp_label(r):
        # chest pain types encoded as separate one-hot columns in this dataset
        cp_cols = [c for c in ['cp_atypical angina', 'cp_non-anginal', 'cp_typical angina'] if c in r.index]
        for c in cp_cols:
            try:
                if float(r[c]) == 1:
                    return c.replace('cp_', '').strip()
            except Exception:
                pass
        return 'unknown'

    def get_thal_label(r):
        thal_cols = [c for c in ['thal_normal', 'thal_reversable defect'] if c in r.index]
        for c in thal_cols:
            try:
                if float(r[c]) == 1:
                    return c.replace('thal_', '').strip()
            except Exception:
                pass
        return 'unknown'

    essentials = [
        ('age', 'Age'),
        ('sex_Male', 'Sex'),
        ('trestbps', 'RestingBP'),
        ('chol', 'Cholesterol'),
        ('thalch', 'MaxHeartRate'),
        ('exang', 'ExerciseAngina'),
        ('oldpeak', 'ST_Depression'),
        ('ca', 'NumMajorVessels'),
    ]

    for col, label in essentials:
        if col in patient_row.index:
            val = patient_row[col]
            if col == 'sex_Male':
                val = 'Male' if float(val) == 1 else 'Female'
            if col == 'exang':
                val = 'Yes' if float(val) == 1 else 'No'
            print(f" - {label}: {val}")

    # chest pain and thal
    cp_label = get_cp_label(patient_row)
    thal_label = get_thal_label(patient_row)
    print(f" - ChestPainType: {cp_label}")
    print(f" - Thalassemia: {thal_label}")

    # Short human-readable reason: list top features considered (names only)
    try:
        fi = pd.read_csv("feature_importances.csv")
        top_feats = fi.head(5)['Feature'].tolist()
    except Exception:
        top_feats = metadata.get('feature_names', [])[:5]

    print("\nReason (features the model most considered):")
    for f in top_feats:
        print(f" - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="heart_disease_uci_encoded_with_id.csv", help="Path to CSV with features (default: heart_disease_uci_encoded_with_id.csv)")
    parser.add_argument("--id", help="Patient id: integer 1-based row number or a value in id-column")
    parser.add_argument("--id-col", default="id", help="Column to use for id lookup when providing --id as a column value (default: id)")
    parser.add_argument("--model", default="final_model.pkl", help="Path to trained model pickle")
    parser.add_argument("--metadata", default="model_metadata.json", help="Path to model metadata JSON")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode and type the id when prompted")
    args = parser.parse_args()

    model, metadata = load_model_and_metadata(args.model, args.metadata)

    # If no id provided on the command line, enter interactive prompt mode
    if args.interactive or not args.id:
        # Only prompt for the patient id; use the default CSV and id-col unless provided via flags
        print("Which patient id do you want to predict? (enter the id number or value)")
        id_input = input("Patient id: ").strip()
        try:
            parsed_id = int(id_input)
        except Exception:
            parsed_id = id_input
        predict_by_id(args.csv, parsed_id, model, metadata, id_col=args.id_col)
    else:
        # Non-interactive: use provided args
        try:
            parsed_id = int(args.id)
        except Exception:
            parsed_id = args.id
        predict_by_id(args.csv, parsed_id, model, metadata, id_col=args.id_col)
