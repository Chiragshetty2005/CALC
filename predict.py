import argparse
import json
import pickle
import pandas as pd
import argparse
import json
import pickle
import pandas as pd
import sys


def load_model_and_metadata(model_path="final_model.pkl", metadata_path="model_metadata.json"):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Trained model not found at '{model_path}'. Run model.py first to train and save the model.")
        sys.exit(1)

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Model metadata not found at '{metadata_path}'. Run model.py to save metadata.")
        sys.exit(1)

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

    X = row[feature_names].values.reshape(1, -1)
    pred = model.predict(X)[0]

    label_map = metadata.get("label_map", {"0": "No Heart Disease", "1": "Heart Disease"})
    human_label = label_map.get(str(int(pred)), str(int(pred)))

    print(f"Patient lookup ({'col='+id_col if id_col else 'row number'}): {patient_id} -> Prediction: {human_label} (raw: {pred})")


def main():
    parser = argparse.ArgumentParser(description="Predict heart disease for a patient by row id in the CSV (1-based) or by id-column value.")
    parser.add_argument("--csv", default="heart_disease_uci_encoded.csv", help="Path to the encoded CSV (default: heart_disease_uci_encoded.csv)")
    parser.add_argument("--id", required=True, help="Patient id: integer 1-based row number or a value in id-column (use --id-col to lookup by column)")
    parser.add_argument("--id-col", default=None, help="If provided, look up the patient by this column's value instead of row number (e.g., --id-col id)")
    parser.add_argument("--model", default="final_model.pkl", help="Path to trained model pickle")
    parser.add_argument("--metadata", default="model_metadata.json", help="Path to model metadata JSON")

    args = parser.parse_args()

    model, metadata = load_model_and_metadata(args.model, args.metadata)

    if args.id_col:
        try:
            parsed_id = int(args.id)
        except Exception:
            parsed_id = args.id
    else:
        try:
            parsed_id = int(args.id)
        except Exception:
            print("When not using --id-col, --id must be an integer 1-based row number.")
            sys.exit(1)

    predict_by_id(args.csv, parsed_id, model, metadata, id_col=args.id_col)


if __name__ == "__main__":
    main()

def load_model_and_metadata(model_path="final_model.pkl", metadata_path="model_metadata.json"):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Trained model not found at '{model_path}'. Run model.py first to train and save the model.")
        sys.exit(1)

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Model metadata not found at '{metadata_path}'. Run model.py to save metadata.")
        sys.exit(1)

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

    X = row[feature_names].values.reshape(1, -1)
    pred = model.predict(X)[0]

    label_map = metadata.get("label_map", {"0": "No Heart Disease", "1": "Heart Disease"})
    human_label = label_map.get(str(int(pred)), str(int(pred)))

    print(f"Patient lookup ({'col='+id_col if id_col else 'row number'}): {patient_id} -> Prediction: {human_label} (raw: {pred})")


def main():
    parser = argparse.ArgumentParser(description="Predict heart disease for a patient by row id in the CSV (1-based) or by id-column value.")
    parser.add_argument("--csv", default="heart_disease_uci_encoded.csv", help="Path to the encoded CSV (default: heart_disease_uci_encoded.csv)")
    parser.add_argument("--id", required=True, help="Patient id: integer 1-based row number or a value in id-column (use --id-col to lookup by column)")
    parser.add_argument("--id-col", default=None, help="If provided, look up the patient by this column's value instead of row number (e.g., --id-col id)")
    parser.add_argument("--model", default="final_model.pkl", help="Path to trained model pickle")
    parser.add_argument("--metadata", default="model_metadata.json", help="Path to model metadata JSON")

    args = parser.parse_args()

    model, metadata = load_model_and_metadata(args.model, args.metadata)

    if args.id_col:
        try:
            parsed_id = int(args.id)
        except Exception:
            parsed_id = args.id
    else:
        try:
            parsed_id = int(args.id)
        except Exception:
            print("When not using --id-col, --id must be an integer 1-based row number.")
            sys.exit(1)

    predict_by_id(args.csv, parsed_id, model, metadata, id_col=args.id_col)


if __name__ == "__main__":
    main()
        except FileNotFoundError:
            print(f"Trained model not found at '{model_path}'. Run model.py first to train and save the model.")
            sys.exit(1)

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f"Model metadata not found at '{metadata_path}'. Run model.py to save metadata.")
            sys.exit(1)

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

        X = row[feature_names].values.reshape(1, -1)
        pred = model.predict(X)[0]

        label_map = metadata.get("label_map", {"0": "No Heart Disease", "1": "Heart Disease"})
        key = str(int(pred))
        human_label = label_map.get(key, key)

        print(f"Patient lookup ({'col='+id_col if id_col else 'row number'}): {patient_id} -> Prediction: {human_label} (raw: {pred})")


    def main():
        parser = argparse.ArgumentParser(description="Predict heart disease for a patient by row id in the CSV (1-based) or by id-column value.")
        parser.add_argument("--csv", default="heart_disease_uci_encoded.csv", help="Path to the encoded CSV (default: heart_disease_uci_encoded.csv)")
        parser.add_argument("--id", required=True, help="Patient id: integer 1-based row number or a value in id-column (use --id-col to lookup by column)")
        parser.add_argument("--id-col", default=None, help="If provided, look up the patient by this column's value instead of row number (e.g., --id-col id)")
        parser.add_argument("--model", default="final_model.pkl", help="Path to trained model pickle")
        parser.add_argument("--metadata", default="model_metadata.json", help="Path to model metadata JSON")

        args = parser.parse_args()

        model, metadata = load_model_and_metadata(args.model, args.metadata)

        id_col = args.id_col
        if id_col:
            try:
                parsed_id = int(args.id)
            except Exception:
                parsed_id = args.id
        else:
            try:
                parsed_id = int(args.id)
            except Exception:
                print("When not using --id-col, --id must be an integer 1-based row number.")
                sys.exit(1)

        predict_by_id(args.csv, parsed_id, model, metadata, id_col=id_col)


    if __name__ == "__main__":
        main()
            parsed_id = args.id
    else:
        try:
            parsed_id = int(args.id)
        except Exception:
            print("When not using --id-col, --id must be an integer 1-based row number.")
            sys.exit(1)

    predict_by_id(args.csv, parsed_id, model, metadata, id_col=id_col)


if __name__ == "__main__":
    main()
import argparse
import json
import pickle
import pandas as pd
import sys


def load_model_and_metadata(model_path='final_model.pkl', metadata_path='model_metadata.json'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return model, metadata


def prepare_input_dataframe(df, feature_columns):
    # If the dataset still contains 'num' or 'HeartDisease', drop them
    for col in ['num', 'HeartDisease']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure all required feature columns are present
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")

    # Reorder and select only the model features
    df_prepared = df[feature_columns].copy()
    return df_prepared


def predict_from_csv(model, metadata, csv_path):
    df = pd.read_csv(csv_path)
    df_in = prepare_input_dataframe(df, metadata['feature_columns'])
    preds = model.predict(df_in)
    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(df_in)

    for i, p in enumerate(preds):
        # metadata JSON keys may be strings, so try both int and str keys
        label = metadata['target_mapping'].get(int(p)) if isinstance(metadata['target_mapping'], dict) else None
        if label is None:
            label = metadata['target_mapping'].get(str(int(p)), str(p))
        if probs is not None:
            prob_pos = probs[i][1] if probs.shape[1] > 1 else probs[i][0]
            print(f"Row {i}: {label} (probability={prob_pos:.3f})")
        else:
            print(f"Row {i}: {label}")


def predict_single_interactive(model, metadata):
    print("Enter values for each feature in order. Press Enter after each value.")
    feature_columns = metadata['feature_columns']
    values = []
    for col in feature_columns:
        val = input(f"{col}: ")
        # try to convert to numeric when possible
        try:
            val = float(val)
        except Exception:
            pass
        values.append(val)

    df = pd.DataFrame([values], columns=feature_columns)
    df_in = prepare_input_dataframe(df, feature_columns)
    pred = model.predict(df_in)[0]
    prob = None
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(df_in)[0]
    label = metadata['target_mapping'].get(int(pred)) if isinstance(metadata['target_mapping'], dict) else None
    if label is None:
        label = metadata['target_mapping'].get(str(int(pred)), str(pred))
    if prob is not None:
        prob_pos = prob[1] if len(prob) > 1 else prob[0]
        print(f"Prediction: {label} (probability={prob_pos:.3f})")
    else:
        print(f"Prediction: {label}")


def main():
    parser = argparse.ArgumentParser(description='Load trained model and predict heart disease from input CSV or interactively.')
    parser.add_argument('--input_csv', help='Path to input CSV file containing rows to predict')
    parser.add_argument('--interactive', action='store_true', help='Enter a single sample interactively')
    parser.add_argument('--model', default='final_model.pkl', help='Path to the trained model pickle')
    parser.add_argument('--metadata', default='model_metadata.json', help='Path to model metadata JSON')

    args = parser.parse_args()

    try:
        model, metadata = load_model_and_metadata(args.model, args.metadata)
    except FileNotFoundError as e:
        print(f"Required file not found: {e}")
        sys.exit(2)

    if args.input_csv:
        predict_from_csv(model, metadata, args.input_csv)
    elif args.interactive:
        predict_single_interactive(model, metadata)
    else:
        print("No input provided. Use --input_csv <path> to predict from a CSV or --interactive to enter a single sample.")
        parser.print_help()


if __name__ == '__main__':
    main()
