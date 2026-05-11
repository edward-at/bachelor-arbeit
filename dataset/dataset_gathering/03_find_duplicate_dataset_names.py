import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

FINAL_DIR = Path("dataset_gathering_final")

review_file = FINAL_DIR / "dataset_name_review.csv"
output_file = FINAL_DIR / "duplicate_dataset_candidates.csv"

df = pd.read_csv(review_file, dtype=str)

df = df.dropna(subset=["Dataset_ID", "Dataset"])
df = df.drop_duplicates(subset=["Dataset_ID"])

rows = df.to_dict("records")

candidates = []

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        id1 = rows[i]["Dataset_ID"]
        id2 = rows[j]["Dataset_ID"]

        name1 = rows[i]["Dataset"]
        name2 = rows[j]["Dataset"]

        score = similarity(id1, id2)

        # Only keep suspiciously similar names
        if score >= 0.78:
            candidates.append({
                "Dataset_ID_1": id1,
                "Dataset_1": name1,
                "Dataset_ID_2": id2,
                "Dataset_2": name2,
                "Similarity": round(score, 3),
                "Same_Dataset?": "",
                "Canonical_Dataset_ID": "",
                "Canonical_Dataset": "",
                "Notes": ""
            })

out = pd.DataFrame(candidates)
out = out.sort_values("Similarity", ascending=False)

out.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print(f"Candidate pairs found: {len(out)}")