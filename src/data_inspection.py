"""
Step 2 - Data understanding and schema inspection for the Home Credit Default Risk data.

Design goals
------------
* Never load every file at once. Each file is streamed in chunks, so the large
  monthly/behavioural tables (bureau_balance, POS_CASH_balance,
  installments_payments, credit_card_balance) fit in a few GB of RAM.
* Treat primary keys and table grain as hypotheses to TEST, not facts to assume.
* Write every result to disk (reports/schema/) so later stages and the
  presentation quote measured numbers only.

Usage
-----
    python -m src.data_inspection --data-dir data/raw --out reports/schema
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 500_000
SENTINEL_DAYS = 365243  # placeholder reported by community analyses; we test for it, not assume it

# Candidate grain (columns expected to identify one row) and foreign keys per table.
# These are HYPOTHESES derived from the official schema diagram; the script verifies them.
TABLES = {
    "application_train":    {"file": "application_train.csv",    "grain": ["SK_ID_CURR"], "fk": {}},
    "application_test":     {"file": "application_test.csv",     "grain": ["SK_ID_CURR"], "fk": {}},
    "bureau":               {"file": "bureau.csv",               "grain": ["SK_ID_BUREAU"],
                             "fk": {"SK_ID_CURR": "application"}},
    "bureau_balance":       {"file": "bureau_balance.csv",       "grain": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
                             "fk": {"SK_ID_BUREAU": "bureau"}},
    "previous_application": {"file": "previous_application.csv", "grain": ["SK_ID_PREV"],
                             "fk": {"SK_ID_CURR": "application"}},
    "POS_CASH_balance":     {"file": "POS_CASH_balance.csv",     "grain": ["SK_ID_PREV", "MONTHS_BALANCE"],
                             "fk": {"SK_ID_CURR": "application", "SK_ID_PREV": "previous_application"}},
    "credit_card_balance":  {"file": "credit_card_balance.csv",  "grain": ["SK_ID_PREV", "MONTHS_BALANCE"],
                             "fk": {"SK_ID_CURR": "application", "SK_ID_PREV": "previous_application"}},
    "installments_payments": {"file": "installments_payments.csv",
                              "grain": ["SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"],
                              "fk": {"SK_ID_CURR": "application", "SK_ID_PREV": "previous_application"}},
}
ID_COLS = {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"}


def resolve(data_dir: Path, fname: str) -> Path:
    """Return the CSV path, or a compressed variant (.zip/.gz) if only that exists."""
    for cand in (fname, fname + ".zip", fname + ".gz", fname.replace(".csv", ".zip")):
        if (data_dir / cand).exists():
            return data_dir / cand
    return data_dir / fname


def verify_files(data_dir: Path) -> pd.DataFrame:
    """List every expected file, whether it exists and its size on disk."""
    rows = []
    for name, cfg in TABLES.items():
        p = resolve(data_dir, cfg["file"])
        rows.append({"table": name, "file": cfg["file"], "exists": p.exists(),
                     "size_mb": round(p.stat().st_size / 2**20, 1) if p.exists() else None})
    extras = sorted(f.name for f in data_dir.glob("*.csv")
                    if f.name not in {c["file"] for c in TABLES.values()})
    df = pd.DataFrame(rows)
    df.attrs["extra_files"] = extras
    return df


def _hash_rows(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def _n_duplicates(hashes: list[np.ndarray]) -> int:
    if not hashes:
        return 0
    allh = np.concatenate(hashes)
    return int(allh.size - np.unique(allh).size)


def inspect_table(name: str, data_dir: Path, collect_ids: dict) -> dict:
    """Stream one CSV and compute schema, missingness, duplicates, cardinality, ranges, key checks."""
    cfg = TABLES[name]
    path = resolve(data_dir, cfg["file"])
    n_rows = 0
    na = None
    dtypes: dict[str, str] = {}
    row_hashes, grain_hashes = [], []
    cat_counts: dict[str, Counter] = defaultdict(Counter)
    num_min, num_max, num_neg, num_sentinel = {}, {}, Counter(), Counter()
    ids: dict[str, set] = defaultdict(set)
    children_per_parent: dict[str, Counter] = defaultdict(Counter)
    prev_to_curr = defaultdict(set) if {"SK_ID_PREV", "SK_ID_CURR"} <= set(cfg["fk"]) else None

    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        n_rows += len(chunk)
        na = chunk.isna().sum() if na is None else na.add(chunk.isna().sum(), fill_value=0)
        for c, t in chunk.dtypes.items():  # a column is text if ANY chunk reads it as text
            is_text = not pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_bool_dtype(t)
            dtypes[c] = "text" if (dtypes.get(c) == "text" or is_text) else str(t)

        row_hashes.append(_hash_rows(chunk))
        grain = [g for g in cfg["grain"] if g in chunk.columns]
        if len(grain) == len(cfg["grain"]):
            grain_hashes.append(_hash_rows(chunk[grain]))

        for c in chunk.columns:
            s = chunk[c]
            if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                cat_counts[c].update(s.dropna().astype(str).value_counts().to_dict())
            elif c not in ID_COLS and pd.api.types.is_numeric_dtype(s):
                if s.notna().any():
                    num_min[c] = min(num_min.get(c, np.inf), s.min())
                    num_max[c] = max(num_max.get(c, -np.inf), s.max())
                num_neg[c] += int((s < 0).sum())
                if c.startswith("DAYS_"):
                    num_sentinel[c] += int((s == SENTINEL_DAYS).sum())

        for c in ID_COLS & set(chunk.columns):
            ids[c].update(chunk[c].dropna().unique().tolist())
        for fk in cfg["fk"]:
            children_per_parent[fk].update(chunk[fk].value_counts().to_dict())
        if prev_to_curr is not None:
            for p, cur in chunk[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates().itertuples(index=False):
                prev_to_curr[p].add(cur)

    for c, s in ids.items():
        collect_ids.setdefault(name, {})[c] = s

    report = {
        "table": name,
        "n_rows": n_rows,
        "n_cols": len(dtypes),
        "dtype_counts": dict(Counter(dtypes.values())),
        "dtypes": dtypes,
        "full_row_duplicates": _n_duplicates(row_hashes),
        "grain_tested": cfg["grain"],
        "grain_duplicate_rows": _n_duplicates(grain_hashes) if grain_hashes else None,
        "missing": {c: {"n": int(v), "pct": round(100 * v / max(n_rows, 1), 2)}
                    for c, v in na.sort_values(ascending=False).items() if v > 0},
        "categorical_cardinality": {c: {"n_unique": len(cnt), "top": cnt.most_common(5)}
                                    for c, cnt in cat_counts.items()},
        "numeric_range": {c: {"min": float(num_min[c]), "max": float(num_max[c]),
                              "n_negative": int(num_neg[c])} for c in num_min},
        "days_sentinel_365243": {c: v for c, v in num_sentinel.items() if v > 0},
        "children_per_parent": {fk: pd.Series(list(cnt.values())).describe().round(2).to_dict()
                                for fk, cnt in children_per_parent.items()},
    }
    if prev_to_curr is not None:
        report["SK_ID_PREV_mapped_to_multiple_SK_ID_CURR"] = sum(len(v) > 1 for v in prev_to_curr.values())
    return report


def referential_integrity(collect_ids: dict) -> list[dict]:
    """How many child keys exist in the parent, and how many parents have any child rows."""
    app_train = collect_ids.get("application_train", {}).get("SK_ID_CURR", set())
    app_test = collect_ids.get("application_test", {}).get("SK_ID_CURR", set())
    parents = {
        "application": app_train | app_test,
        "bureau": collect_ids.get("bureau", {}).get("SK_ID_BUREAU", set()),
        "previous_application": collect_ids.get("previous_application", {}).get("SK_ID_PREV", set()),
    }
    print(f"SK_ID_CURR overlap between train and test: {len(app_train & app_test)}")
    out = []
    for name, cfg in TABLES.items():
        for fk, parent in cfg["fk"].items():
            child = collect_ids.get(name, {}).get(fk, set())
            par = parents[parent]
            if not child or not par:
                continue
            row = {"child": name, "fk": fk, "parent": parent,
                   "child_keys": len(child),
                   "child_keys_found_in_parent_pct": round(100 * len(child & par) / len(child), 2),
                   "parents_with_children_pct": round(100 * len(par & child) / len(par), 2)}
            if parent == "application":
                row["train_coverage_pct"] = round(100 * len(app_train & child) / max(len(app_train), 1), 2)
                row["test_coverage_pct"] = round(100 * len(app_test & child) / max(len(app_test), 1), 2)
            out.append(row)
    return out


def target_summary(data_dir: Path) -> dict:
    t = pd.read_csv(data_dir / "application_train.csv", usecols=["TARGET"])["TARGET"]
    vc = t.value_counts(dropna=False).sort_index()
    return {"counts": {str(k): int(v) for k, v in vc.items()},
            "pct": {str(k): round(100 * v / len(t), 3) for k, v in vc.items()},
            "values_observed": sorted(map(str, t.dropna().unique()))}


def main(data_dir: str, out: str) -> None:
    data_dir, out = Path(data_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)

    files = verify_files(data_dir)
    print(files.to_string(index=False))
    print("Extra CSVs present:", files.attrs["extra_files"])
    files.to_csv(out / "file_inventory.csv", index=False)
    missing = files.loc[~files.exists, "file"].tolist()
    if missing:
        print(f"\nWARNING - missing files, inspection limited: {missing}")

    collect_ids: dict = {}
    summary = []
    for name in TABLES:
        if not resolve(data_dir, TABLES[name]["file"]).exists():
            continue
        print(f"Inspecting {name} ...", flush=True)
        rep = inspect_table(name, data_dir, collect_ids)
        (out / f"{name}.json").write_text(json.dumps(rep, indent=2, default=str))
        summary.append({k: rep[k] for k in ["table", "n_rows", "n_cols", "full_row_duplicates",
                                             "grain_duplicate_rows"]}
                       | {"cols_with_missing": len(rep["missing"]),
                          "n_categorical": len(rep["categorical_cardinality"])})

    summ = pd.DataFrame(summary)
    summ.to_csv(out / "table_summary.csv", index=False)
    print("\n", summ.to_string(index=False))

    ri = referential_integrity(collect_ids)
    pd.DataFrame(ri).to_csv(out / "referential_integrity.csv", index=False)
    print("\n", pd.DataFrame(ri).to_string(index=False))

    if (data_dir / "application_train.csv").exists():
        ts = target_summary(data_dir)
        (out / "target_summary.json").write_text(json.dumps(ts, indent=2))
        print("\nTARGET distribution:", ts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="reports/schema")
    a = ap.parse_args()
    main(a.data_dir, a.out)
