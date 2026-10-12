"""
Export the trained Random Forest model to a JSON format
that can be loaded and executed entirely in JavaScript.
"""
import pickle
import json
import numpy as np

# Load artifacts
with open("model/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/stats.json", "r") as f:
    stats = json.load(f)

# ── Export scaler params ────────────────────────────────────────────────────────
scaler_data = {
    "mean":  scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}

# ── Export each decision tree ───────────────────────────────────────────────────
def export_tree(tree):
    t = tree.tree_
    def recurse(node_id):
        # Leaf node
        if t.children_left[node_id] == -1:
            values = t.value[node_id][0]
            total  = sum(values)
            return {
                "leaf": True,
                "prob": [round(v / total, 6) for v in values]
            }
        return {
            "leaf":    False,
            "feature": int(t.feature[node_id]),
            "threshold": round(float(t.threshold[node_id]), 6),
            "left":  recurse(int(t.children_left[node_id])),
            "right": recurse(int(t.children_right[node_id]))
        }
    return recurse(0)

print(f"Exporting {len(model.estimators_)} trees...")
trees = [export_tree(est) for est in model.estimators_]
print("Done exporting trees.")

# ── Bundle everything ───────────────────────────────────────────────────────────
bundle = {
    "scaler": scaler_data,
    "trees":  trees,
    "stats":  stats,
    "n_classes": 2
}

out_path = "model/rf_bundle.json"
with open(out_path, "w") as f:
    json.dump(bundle, f, separators=(",", ":"))

import os
size_kb = os.path.getsize(out_path) / 1024
print(f"Saved to {out_path}  ({size_kb:.1f} KB)")
