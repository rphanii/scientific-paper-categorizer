import json
import pandas as pd
import os

category_map = {
    "cs": "Computer Science",
    "physics": "Physics",
    "math": "Mathematics",
    "q-bio": "Biology",
    "bio": "Biology",
    "chem": "Chemistry",
    "hep": "Physics",
    "astro-ph": "Physics",
    "cond-mat": "Physics",
}

filename = "arxiv-metadata-oai-snapshot.json"
if not os.path.exists(filename):
    exit()

data = []
with open(filename, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 50000:
            break
        try:
            paper = json.loads(line)
            abstract = paper.get("abstract", "").replace("\n", " ").strip()
            categories = paper.get("categories", "")
            main_cat = categories.split()[0].split(".")[0]
            label = category_map.get(main_cat)
            if label and abstract:
                data.append([abstract, label])
        except:
            continue

if not data:
    exit()

df = pd.DataFrame(data, columns=["abstract", "category"])
df = df.groupby("category").apply(lambda x: x.sample(n=200, random_state=42)).reset_index(drop=True)
df.to_csv("data.csv", index=False)
