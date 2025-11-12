import pandas as pd

IN = "heart_disease_uci_encoded.csv"
OUT = "heart_disease_uci_encoded_with_id.csv"

df = pd.read_csv(IN)
if 'id' in df.columns:
    print(f"'id' column already exists in {IN}, writing copy to {OUT} with same ids.")
else:
    df.insert(0, 'id', range(1, len(df) + 1))
    print(f"Inserted 1-based 'id' column with {len(df)} rows.")

# save copy
df.to_csv(OUT, index=False)
print(f"Wrote {OUT}")
