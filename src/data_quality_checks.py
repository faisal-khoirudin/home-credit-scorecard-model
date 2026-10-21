"""
Step 2b - Targeted data-quality checks that follow up on data_inspection.py findings.
Usage: python -m src.data_quality_checks --data-dir data/raw
Every number quoted in the Step 2 write-up is printed by this script.
"""
import argparse
import pandas as pd

SENTINEL = 365243


def application_checks(d):
    tr = pd.read_csv(f"{d}/application_train.csv")
    te = pd.read_csv(f"{d}/application_test.csv")
    s = tr.DAYS_EMPLOYED == SENTINEL
    print("TARGET rate %:", round(100 * tr.TARGET.mean(), 3))
    print("\nDAYS_EMPLOYED sentinel by income type:\n", pd.crosstab(tr.NAME_INCOME_TYPE, s))
    print("sentinel rows with ORGANIZATION_TYPE == 'XNA':", (tr.loc[s, "ORGANIZATION_TYPE"] == "XNA").sum(), "/", s.sum())
    print("TARGET % sentinel vs not:", round(100 * tr.TARGET[s].mean(), 2), round(100 * tr.TARGET[~s].mean(), 2))
    print("\nincome >1M:", (tr.AMT_INCOME_TOTAL > 1e6).sum(), " >10M:", (tr.AMT_INCOME_TOTAL > 1e7).sum())
    print(tr.nlargest(3, "AMT_INCOME_TOTAL")[["AMT_INCOME_TOTAL", "AMT_CREDIT", "NAME_INCOME_TYPE", "TARGET"]])
    print("CODE_GENDER == 'XNA':", (tr.CODE_GENDER == "XNA").sum(), "(train);", (te.CODE_GENDER == "XNA").sum(), "(test)")
    print("\nMissingness vs TARGET (train) and test missing %:")
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "OWN_CAR_AGE", "COMMONAREA_AVG",
              "OCCUPATION_TYPE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
        m = tr[c].isna()
        print(f"  {c:28s} train_miss={100*m.mean():5.1f} test_miss={100*te[c].isna().mean():5.1f} "
              f"TARGET%_missing={100*tr.TARGET[m].mean():5.2f} TARGET%_present={100*tr.TARGET[~m].mean():5.2f}")
    hb = [c for c in tr.columns if c.endswith(("_AVG", "_MODE", "_MEDI"))]
    print(f"building-info columns: {len(hb)}; rows with all missing: {100*tr[hb].isna().all(axis=1).mean():.2f}%")
    print("\nRevolving share % train/test:", round(100 * (tr.NAME_CONTRACT_TYPE == "Revolving loans").mean(), 2),
          round(100 * (te.NAME_CONTRACT_TYPE == "Revolving loans").mean(), 2))
    print("absolute-date columns:", [c for c in tr.columns if "DATE" in c.upper() and not c.startswith("DAYS")])


def bureau_checks(d):
    b = pd.read_csv(f"{d}/bureau.csv")
    n = len(b)
    pct = lambda x: f"{x:,} ({100*x/n:.2f}%)"
    print("\nbureau DAYS_CREDIT_UPDATE > 0:", pct((b.DAYS_CREDIT_UPDATE > 0).sum()))
    print("bureau DAYS_CREDIT_ENDDATE < -2922:", pct((b.DAYS_CREDIT_ENDDATE < -2922).sum()))
    print("bureau DAYS_ENDDATE_FACT < DAYS_CREDIT:", pct((b.DAYS_ENDDATE_FACT < b.DAYS_CREDIT).sum()))
    print("bureau AMT_CREDIT_SUM_DEBT < 0:", pct((b.AMT_CREDIT_SUM_DEBT < 0).sum()))
    print("bureau AMT_CREDIT_SUM == 0:", pct((b.AMT_CREDIT_SUM == 0).sum()))


def previous_checks(d):
    p = pd.read_csv(f"{d}/previous_application.csv",
                    usecols=["NAME_CONTRACT_STATUS", "AMT_CREDIT", "FLAG_LAST_APPL_PER_CONTRACT", "DAYS_DECISION"])
    print("\nprevious status x AMT_CREDIT==0:\n", pd.crosstab(p.NAME_CONTRACT_STATUS, p.AMT_CREDIT == 0))
    print("FLAG_LAST_APPL_PER_CONTRACT == 'N':", (p.FLAG_LAST_APPL_PER_CONTRACT == "N").sum())
    print("DAYS_DECISION range:", p.DAYS_DECISION.min(), p.DAYS_DECISION.max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    d = ap.parse_args().data_dir
    application_checks(d); bureau_checks(d); previous_checks(d)
