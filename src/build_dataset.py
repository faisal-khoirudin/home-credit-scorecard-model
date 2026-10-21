"""
Build the applicant-level modelling dataset (one row per SK_ID_CURR) from the Home Credit files provided
for this project (same structure as the public Home Credit Default Risk dataset, used only as a reference).

Why this is safe to build before modelling
------------------------------------------
Every step here is a deterministic, per-applicant transformation:
* rule-based cleaning (coded placeholders -> missing, impossible dates -> missing);
* aggregation of each history table over that applicant's own records.
No step uses TARGET or statistics computed across applicants, so nothing can leak between train,
validation and holdout. Learned steps (imputation, capping, encoding, scaling) stay inside the
model pipeline and are fitted on training folds only (Step 4).

Granularity rules (Step 2 findings)
-----------------------------------
* Every history table is aggregated to SK_ID_CURR BEFORE joining (no row multiplication).
* installments_payments has one row per PAYMENT: payments are summed per instalment first.
* previous_application: only final application versions are used.
* bureau records updated after the application (DAYS_CREDIT_UPDATE > 0) are excluded.
* Count features are 0 when an applicant has no records (a factual zero); ratios, means and
  maxima stay missing and a HAS_<TABLE> flag records the absence of history.
* Amount sums use min_count=1: if every input amount is missing the sum is missing, not 0
  (otherwise "not reported" would be indistinguishable from "zero obligations").

Usage
-----
    python -m src.build_dataset --data-dir data/raw --out data/processed
Outputs: abt_train.parquet, abt_test.parquet, data_dictionary.csv, build_report.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

SENTINEL = 365243
DICT: list[dict] = []          # data dictionary rows, filled by reg()


def reg(name, source, formula, meaning):
    DICT.append({"feature": name, "source": source, "formula": formula, "meaning": meaning})


def resolve(data_dir: Path, fname: str) -> Path:
    for cand in (fname, fname + ".zip", fname + ".gz", fname.replace(".csv", ".zip")):
        if (data_dir / cand).exists():
            return data_dir / cand
    raise FileNotFoundError(fname)


def safe_div(a, b):
    b = b.astype("float64")
    return a / b.where(b != 0)


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- application
def build_application(data_dir: Path) -> pd.DataFrame:
    tr = pd.read_csv(resolve(data_dir, "application_train.csv"))
    te = pd.read_csv(resolve(data_dir, "application_test.csv"))
    n_tr = len(tr)
    app = pd.concat([tr, te], ignore_index=True).copy()      # .copy() consolidates memory blocks
    app["DATASET"] = np.where(app.index < n_tr, "train", "test")
    del tr, te

    # Rule-based cleaning (Section 6.1 findings)
    app["FLAG_NOT_EMPLOYED"] = (app.DAYS_EMPLOYED == SENTINEL).astype("int8")
    app["DAYS_EMPLOYED"] = app.DAYS_EMPLOYED.replace(SENTINEL, np.nan)
    app["CODE_GENDER"] = app.CODE_GENDER.replace("XNA", np.nan)
    app["NAME_FAMILY_STATUS"] = app.NAME_FAMILY_STATUS.replace("Unknown", np.nan)
    app["ORGANIZATION_TYPE"] = app.ORGANIZATION_TYPE.replace("XNA", np.nan)
    reg("FLAG_NOT_EMPLOYED", "application", "DAYS_EMPLOYED == 365243",
        "Coded 'not employed' (pensioners, unemployed); DAYS_EMPLOYED set to missing for these rows")

    ext = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    docs = [c for c in app.columns if c.startswith("FLAG_DOCUMENT_")]
    new = {
        "APP_AGE_YEARS": (-app.DAYS_BIRTH / 365.25,
                          "-DAYS_BIRTH / 365.25", "Applicant age in years"),
        "APP_EMPLOYED_YEARS": (-app.DAYS_EMPLOYED / 365.25,
                               "-DAYS_EMPLOYED / 365.25", "Current employment length in years"),
        "APP_EMPLOYED_TO_AGE": (safe_div(app.DAYS_EMPLOYED, app.DAYS_BIRTH),
                                "DAYS_EMPLOYED / DAYS_BIRTH", "Share of life in current job (stability)"),
        "APP_CREDIT_TO_INCOME": (safe_div(app.AMT_CREDIT, app.AMT_INCOME_TOTAL),
                                 "AMT_CREDIT / AMT_INCOME_TOTAL", "Loan size relative to annual income"),
        "APP_ANNUITY_TO_INCOME": (safe_div(app.AMT_ANNUITY, app.AMT_INCOME_TOTAL),
                                  "AMT_ANNUITY / AMT_INCOME_TOTAL", "Payment burden of this loan"),
        "APP_CREDIT_TO_GOODS": (safe_div(app.AMT_CREDIT, app.AMT_GOODS_PRICE),
                                "AMT_CREDIT / AMT_GOODS_PRICE", "Financing above the goods price (fees, insurance, cash)"),
        "APP_CREDIT_TERM": (safe_div(app.AMT_CREDIT, app.AMT_ANNUITY),
                            "AMT_CREDIT / AMT_ANNUITY", "Implied term; mixes contract length and pricing"),
        "APP_INCOME_PER_PERSON": (safe_div(app.AMT_INCOME_TOTAL, app.CNT_FAM_MEMBERS),
                                  "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS", "Household income per member"),
        "APP_EXT_MEAN": (ext.mean(axis=1), "row mean of available EXT_SOURCE_1..3", "Average external score"),
        "APP_EXT_MIN": (ext.min(axis=1), "row min of available EXT_SOURCE_1..3", "Weakest external score"),
        "APP_EXT_MAX": (ext.max(axis=1), "row max of available EXT_SOURCE_1..3", "Strongest external score"),
        "APP_EXT_STD": (ext.std(axis=1), "row std of available EXT_SOURCE_1..3", "Disagreement between external scores"),
        "APP_EXT_N_MISSING": (ext.isna().sum(axis=1).astype("int8"),
                              "count of missing EXT_SOURCE_1..3", "External-score coverage"),
        "APP_N_DOCUMENTS": (app[docs].sum(axis=1).astype("int8"),
                            "sum of FLAG_DOCUMENT_2..21", "Number of documents provided"),
    }
    for k, (s, f, m) in new.items():
        reg(k, "application", f, m)
    app = pd.concat([app, pd.DataFrame({k: v[0] for k, v in new.items()})], axis=1)
    return app.copy()


# --------------------------------------------------------------------------- previous_application
def build_previous(data_dir: Path) -> pd.DataFrame:
    cols = ["SK_ID_CURR", "NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT",
            "AMT_ANNUITY", "RATE_DOWN_PAYMENT", "CNT_PAYMENT", "DAYS_DECISION", "NAME_YIELD_GROUP",
            "DAYS_LAST_DUE_1ST_VERSION", "FLAG_LAST_APPL_PER_CONTRACT", "NFLAG_LAST_APPL_IN_DAY"]
    p = pd.read_csv(resolve(data_dir, "previous_application.csv"), usecols=cols)
    p = p[(p.FLAG_LAST_APPL_PER_CONTRACT == "Y") & (p.NFLAG_LAST_APPL_IN_DAY == 1)].copy()
    p["DAYS_LAST_DUE_1ST_VERSION"] = p.DAYS_LAST_DUE_1ST_VERSION.replace(SENTINEL, np.nan)
    st = p.NAME_CONTRACT_STATUS
    p["is_appr"], p["is_ref"], p["is_can"] = st.eq("Approved"), st.eq("Refused"), st.eq("Canceled")
    p["ref_1y"] = p.is_ref & (p.DAYS_DECISION >= -365)
    p["is_cash"], p["is_rev"] = p.NAME_CONTRACT_TYPE.eq("Cash loans"), p.NAME_CONTRACT_TYPE.eq("Revolving loans")
    g = p.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "PREV_N_APPS": g.size(),
        "PREV_N_APPROVED": g.is_appr.sum(), "PREV_N_REFUSED": g.is_ref.sum(), "PREV_N_CANCELED": g.is_can.sum(),
        "PREV_N_REFUSED_1Y": g.ref_1y.sum(),
        "PREV_N_CASH": g.is_cash.sum(), "PREV_N_REVOLVING": g.is_rev.sum(),
        "PREV_DAYS_LAST_DECISION": g.DAYS_DECISION.max(),
        "PREV_AMT_APPLICATION_MEAN": g.AMT_APPLICATION.mean(),
    })
    out["PREV_APPROVAL_RATE"] = out.PREV_N_APPROVED / out.PREV_N_APPS
    out["PREV_REFUSAL_RATE"] = out.PREV_N_REFUSED / out.PREV_N_APPS
    out["PREV_DAYS_LAST_REFUSAL"] = p[p.is_ref].groupby("SK_ID_CURR").DAYS_DECISION.max()

    a = p[p.is_appr].copy()
    a["credit_to_app"] = safe_div(a.AMT_CREDIT, a.AMT_APPLICATION)
    a["active"] = a.DAYS_LAST_DUE_1ST_VERSION > 0
    a["high_yield"] = a.NAME_YIELD_GROUP.eq("high")
    ga = a.groupby("SK_ID_CURR")
    out = out.join(pd.DataFrame({
        "PREV_APPR_AMT_CREDIT_MEAN": ga.AMT_CREDIT.mean(),
        "PREV_APPR_AMT_CREDIT_SUM": ga.AMT_CREDIT.sum(min_count=1),
        "PREV_APPR_CREDIT_TO_APP_MEAN": ga.credit_to_app.mean(),
        "PREV_APPR_CREDIT_TO_APP_MIN": ga.credit_to_app.min(),
        "PREV_APPR_CNT_PAYMENT_MEAN": ga.CNT_PAYMENT.mean(),
        "PREV_APPR_DOWN_PAYMENT_RATE_MEAN": ga.RATE_DOWN_PAYMENT.mean(),
        "PREV_APPR_SHARE_HIGH_YIELD": ga.high_yield.mean(),
        "PREV_APPR_N_ACTIVE": ga.active.sum(),
        "PREV_APPR_ACTIVE_ANNUITY_SUM": a[a.active].groupby("SK_ID_CURR").AMT_ANNUITY.sum(min_count=1),
    }))
    S = "previous_application (final versions)"
    for k, f, m in [
        ("PREV_N_APPS", "count of applications", "Volume of previous Home Credit applications"),
        ("PREV_N_APPROVED", "count status == Approved", "Previously approved contracts"),
        ("PREV_N_REFUSED", "count status == Refused", "Previously refused applications"),
        ("PREV_N_CANCELED", "count status == Canceled", "Applications cancelled by client or lender"),
        ("PREV_N_REFUSED_1Y", "refused with DAYS_DECISION >= -365", "Recent refusals (last 12 months)"),
        ("PREV_N_CASH", "count contract type == Cash loans", "Previous cash-loan applications"),
        ("PREV_N_REVOLVING", "count contract type == Revolving loans", "Previous revolving applications"),
        ("PREV_DAYS_LAST_DECISION", "max DAYS_DECISION", "Recency of last application"),
        ("PREV_AMT_APPLICATION_MEAN", "mean AMT_APPLICATION", "Typical amount requested"),
        ("PREV_APPROVAL_RATE", "approved / applications", "Share of past applications approved"),
        ("PREV_REFUSAL_RATE", "refused / applications", "Share of past applications refused"),
        ("PREV_DAYS_LAST_REFUSAL", "max DAYS_DECISION where refused", "Recency of last refusal"),
        ("PREV_APPR_AMT_CREDIT_MEAN", "mean AMT_CREDIT, approved", "Typical previously granted amount"),
        ("PREV_APPR_AMT_CREDIT_SUM", "sum AMT_CREDIT, approved", "Total previously granted credit"),
        ("PREV_APPR_CREDIT_TO_APP_MEAN", "mean AMT_CREDIT / AMT_APPLICATION, approved",
         "Granted vs requested amount (below 1 = lender cut the amount)"),
        ("PREV_APPR_CREDIT_TO_APP_MIN", "min AMT_CREDIT / AMT_APPLICATION, approved", "Largest past amount cut"),
        ("PREV_APPR_CNT_PAYMENT_MEAN", "mean CNT_PAYMENT, approved", "Typical previous loan term"),
        ("PREV_APPR_DOWN_PAYMENT_RATE_MEAN", "mean RATE_DOWN_PAYMENT, approved", "Typical down-payment share"),
        ("PREV_APPR_SHARE_HIGH_YIELD", "share NAME_YIELD_GROUP == high, approved",
         "Share of past loans priced in the high-yield group (lender's earlier risk pricing)"),
        ("PREV_APPR_N_ACTIVE", "approved with DAYS_LAST_DUE_1ST_VERSION > 0",
         "Previous contracts scheduled to run past this application"),
        ("PREV_APPR_ACTIVE_ANNUITY_SUM", "sum AMT_ANNUITY of active approved contracts",
         "Existing Home Credit instalment obligations"),
    ]:
        reg(k, S, f, m)
    return out


# --------------------------------------------------------------------------- bureau + bureau_balance
def build_bureau(data_dir: Path, report: dict) -> pd.DataFrame:
    b = pd.read_csv(resolve(data_dir, "bureau.csv"))
    n0 = len(b)
    b = b[~(b.DAYS_CREDIT_UPDATE > 0)].copy()                    # leakage rule (Section 6.2)
    report["bureau_rows_excluded_updated_after_application"] = n0 - len(b)
    bad_end = b.DAYS_CREDIT_ENDDATE < b.DAYS_CREDIT
    bad_fact = b.DAYS_ENDDATE_FACT < b.DAYS_CREDIT
    report["bureau_impossible_dates_set_missing"] = int(bad_end.sum() + bad_fact.sum())
    b.loc[bad_end, "DAYS_CREDIT_ENDDATE"] = np.nan
    b.loc[bad_fact, "DAYS_ENDDATE_FACT"] = np.nan

    b["active"] = b.CREDIT_ACTIVE.eq("Active")
    b["overdue_now"] = b.CREDIT_DAY_OVERDUE > 0
    b["opened_1y"] = b.DAYS_CREDIT >= -365
    b["is_card"] = b.CREDIT_TYPE.eq("Credit card")
    b["is_micro"] = b.CREDIT_TYPE.eq("Microloan")
    b["bad_debt"] = b.CREDIT_ACTIVE.eq("Bad debt")
    g = b.groupby("SK_ID_CURR")
    act = b[b.active].groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "BUR_N_CREDITS": g.size(), "BUR_N_ACTIVE": g.active.sum(),
        "BUR_N_BAD_DEBT": g.bad_debt.sum(),
        "BUR_N_CREDIT_TYPES": g.CREDIT_TYPE.nunique(),
        "BUR_N_CARDS": g.is_card.sum(), "BUR_N_MICROLOANS": g.is_micro.sum(),
        "BUR_N_OPENED_1Y": g.opened_1y.sum(), "BUR_N_OVERDUE_NOW": g.overdue_now.sum(),
        "BUR_DAYS_LAST_OPENED": g.DAYS_CREDIT.max(), "BUR_DAYS_FIRST_OPENED": g.DAYS_CREDIT.min(),
        "BUR_MAX_DAYS_OVERDUE_NOW": g.CREDIT_DAY_OVERDUE.max(),
        "BUR_AMT_OVERDUE_SUM": g.AMT_CREDIT_SUM_OVERDUE.sum(min_count=1),
        "BUR_MAX_OVERDUE_AMT_EVER": g.AMT_CREDIT_MAX_OVERDUE.max(),
        "BUR_N_PROLONGED": g.CNT_CREDIT_PROLONG.sum(min_count=1),
        "BUR_AMT_CREDIT_SUM": g.AMT_CREDIT_SUM.sum(min_count=1),
        "BUR_ACTIVE_DEBT_SUM": act.AMT_CREDIT_SUM_DEBT.sum(min_count=1),
        "BUR_ACTIVE_CREDIT_SUM": act.AMT_CREDIT_SUM.sum(min_count=1),
        "BUR_ACTIVE_LIMIT_SUM": act.AMT_CREDIT_SUM_LIMIT.sum(min_count=1),
        "BUR_ACTIVE_ANNUITY_SUM": act.AMT_ANNUITY.sum(min_count=1),
        "BUR_ACTIVE_MAX_DAYS_TO_END": act.DAYS_CREDIT_ENDDATE.max(),
    })
    out["BUR_SHARE_ACTIVE"] = out.BUR_N_ACTIVE / out.BUR_N_CREDITS
    out["BUR_ACTIVE_DEBT_RATIO"] = safe_div(out.BUR_ACTIVE_DEBT_SUM, out.BUR_ACTIVE_CREDIT_SUM).clip(-1, 5)
    for c in ["BUR_ACTIVE_DEBT_SUM", "BUR_ACTIVE_CREDIT_SUM", "BUR_ACTIVE_LIMIT_SUM", "BUR_ACTIVE_ANNUITY_SUM"]:
        out.loc[out.BUR_N_ACTIVE == 0, c] = np.nan            # no active credit: sums are undefined, not zero

    # bureau_balance: worst DPD bucket, streamed (27M rows); orphan bureau IDs drop out at the join
    rank = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    parts = []
    for ch in pd.read_csv(resolve(data_dir, "bureau_balance.csv"), chunksize=5_000_000):
        s = ch.STATUS.map(rank)
        f = pd.DataFrame({"SK_ID_BUREAU": ch.SK_ID_BUREAU, "known": s.notna(), "dpd": s >= 1,
                          "worst": s, "worst12": s.where(ch.MONTHS_BALANCE >= -12)})
        parts.append(f.groupby("SK_ID_BUREAU").agg(known=("known", "sum"), dpd=("dpd", "sum"),
                                                   worst=("worst", "max"), worst12=("worst12", "max")))
    bb = pd.concat(parts).groupby(level=0).agg(known=("known", "sum"), dpd=("dpd", "sum"),
                                               worst=("worst", "max"), worst12=("worst12", "max"))
    del parts
    bb = b[["SK_ID_CURR", "SK_ID_BUREAU"]].merge(bb, left_on="SK_ID_BUREAU", right_index=True)
    gb = bb.groupby("SK_ID_CURR")
    out = out.join(pd.DataFrame({
        "BB_N_MONTHS_KNOWN": gb.known.sum(), "BB_WORST_STATUS": gb.worst.max(),
        "BB_WORST_STATUS_12M": gb.worst12.max(),
        "BB_SHARE_MONTHS_DPD": safe_div(gb.dpd.sum(), gb.known.sum())}))
    out["HAS_BUREAU_BALANCE"] = out.BB_N_MONTHS_KNOWN.notna().astype("int8")

    S = "bureau (records updated after application excluded)"
    for k, f, m in [
        ("BUR_N_CREDITS", "count of bureau credits", "Breadth of external credit history"),
        ("BUR_N_ACTIVE", "count CREDIT_ACTIVE == Active", "Open external credits"),
        ("BUR_N_BAD_DEBT", "count CREDIT_ACTIVE == Bad debt", "External credits written off as bad debt"),
        ("BUR_N_CREDIT_TYPES", "distinct CREDIT_TYPE", "Diversity of credit products"),
        ("BUR_N_CARDS", "count CREDIT_TYPE == Credit card", "External credit cards"),
        ("BUR_N_MICROLOANS", "count CREDIT_TYPE == Microloan", "External microloans"),
        ("BUR_N_OPENED_1Y", "count DAYS_CREDIT >= -365", "Credits opened in last 12 months (credit hunger)"),
        ("BUR_N_OVERDUE_NOW", "count CREDIT_DAY_OVERDUE > 0", "Credits overdue at application"),
        ("BUR_DAYS_LAST_OPENED", "max DAYS_CREDIT", "Recency of newest external credit"),
        ("BUR_DAYS_FIRST_OPENED", "min DAYS_CREDIT", "Length of external credit history"),
        ("BUR_MAX_DAYS_OVERDUE_NOW", "max CREDIT_DAY_OVERDUE", "Worst current delinquency"),
        ("BUR_AMT_OVERDUE_SUM", "sum AMT_CREDIT_SUM_OVERDUE", "Amount currently overdue"),
        ("BUR_MAX_OVERDUE_AMT_EVER", "max AMT_CREDIT_MAX_OVERDUE", "Largest overdue amount ever"),
        ("BUR_N_PROLONGED", "sum CNT_CREDIT_PROLONG", "Times credits were prolonged"),
        ("BUR_AMT_CREDIT_SUM", "sum AMT_CREDIT_SUM", "Total external credit ever reported"),
        ("BUR_ACTIVE_DEBT_SUM", "sum AMT_CREDIT_SUM_DEBT, active", "Outstanding external debt (negative = overpaid)"),
        ("BUR_ACTIVE_CREDIT_SUM", "sum AMT_CREDIT_SUM, active", "Size of open external credits"),
        ("BUR_ACTIVE_LIMIT_SUM", "sum AMT_CREDIT_SUM_LIMIT, active", "Open external card limits"),
        ("BUR_ACTIVE_ANNUITY_SUM", "sum AMT_ANNUITY, active", "Known external instalment obligations"),
        ("BUR_ACTIVE_MAX_DAYS_TO_END", "max DAYS_CREDIT_ENDDATE, active", "Longest remaining external commitment"),
        ("BUR_SHARE_ACTIVE", "active / credits", "Share of external credits still open"),
        ("BUR_ACTIVE_DEBT_RATIO", "active debt / active credit, clipped [-1, 5]", "How much of open credit is still owed"),
    ]:
        reg(k, S, f, m)
    for k, f, m in [
        ("BB_N_MONTHS_KNOWN", "count months with status 0-5", "Months of known bureau payment status"),
        ("BB_WORST_STATUS", "max status bucket (0 = no DPD ... 5 = 120+ / sold)", "Worst external delinquency ever"),
        ("BB_WORST_STATUS_12M", "max status bucket, MONTHS_BALANCE >= -12", "Worst external delinquency, last 12 months"),
        ("BB_SHARE_MONTHS_DPD", "months with DPD / months with known status", "Frequency of external delinquency"),
        ("HAS_BUREAU_BALANCE", "1 if any monthly status known", "Monthly bureau history available"),
    ]:
        reg(k, "bureau_balance (via bureau)", f, m)
    return out


# --------------------------------------------------------------------------- POS_CASH_balance
def build_pos(data_dir: Path) -> pd.DataFrame:
    pos = pd.read_csv(resolve(data_dir, "POS_CASH_balance.csv"),
                      usecols=["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE",
                               "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"],
                      dtype={"SK_ID_PREV": "int32", "SK_ID_CURR": "int32", "MONTHS_BALANCE": "int16",
                             "CNT_INSTALMENT_FUTURE": "float32", "NAME_CONTRACT_STATUS": "category",
                             "SK_DPD": "int32", "SK_DPD_DEF": "int32"})
    pos["dpd"], pos["dpd_def"] = pos.SK_DPD > 0, pos.SK_DPD_DEF > 0
    pos["dpd_12m"] = pos.SK_DPD.where(pos.MONTHS_BALANCE >= -12)
    g = pos.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "POS_N_CONTRACTS": g.SK_ID_PREV.nunique(), "POS_N_MONTHS": g.size(),
        "POS_MAX_DPD": g.SK_DPD.max(), "POS_MAX_DPD_DEF": g.SK_DPD_DEF.max(),
        "POS_SHARE_MONTHS_DPD": g.dpd.mean(), "POS_SHARE_MONTHS_DPD_DEF": g.dpd_def.mean(),
        "POS_MAX_DPD_12M": g.dpd_12m.max()})
    latest = pos.sort_values("MONTHS_BALANCE").groupby("SK_ID_PREV").tail(1)
    act = latest[latest.NAME_CONTRACT_STATUS == "Active"]
    out["POS_N_ACTIVE"] = act.groupby("SK_ID_CURR").size()
    out["POS_REMAINING_INSTALMENTS"] = act.groupby("SK_ID_CURR").CNT_INSTALMENT_FUTURE.sum(min_count=1)
    out["POS_SHARE_COMPLETED"] = (latest.NAME_CONTRACT_STATUS == "Completed").groupby(latest.SK_ID_CURR).mean()
    out["POS_N_ACTIVE"] = out.POS_N_ACTIVE.fillna(0)
    S = "POS_CASH_balance"
    for k, f, m in [
        ("POS_N_CONTRACTS", "distinct SK_ID_PREV", "POS/cash contracts with monthly history"),
        ("POS_N_MONTHS", "count monthly rows", "Length of POS/cash history"),
        ("POS_MAX_DPD", "max SK_DPD", "Worst POS/cash delinquency"),
        ("POS_MAX_DPD_DEF", "max SK_DPD_DEF", "Worst delinquency beyond tolerance"),
        ("POS_SHARE_MONTHS_DPD", "mean(SK_DPD > 0)", "Frequency of POS/cash delinquency"),
        ("POS_SHARE_MONTHS_DPD_DEF", "mean(SK_DPD_DEF > 0)", "Frequency of delinquency beyond tolerance"),
        ("POS_MAX_DPD_12M", "max SK_DPD, MONTHS_BALANCE >= -12", "Recent POS/cash delinquency"),
        ("POS_N_ACTIVE", "contracts whose latest status == Active", "Open POS/cash contracts"),
        ("POS_REMAINING_INSTALMENTS", "sum latest CNT_INSTALMENT_FUTURE, active contracts", "Remaining POS/cash obligations"),
        ("POS_SHARE_COMPLETED", "share of contracts whose latest status == Completed", "Track record of finishing contracts"),
    ]:
        reg(k, S, f, m)
    return out


# --------------------------------------------------------------------------- credit_card_balance
def build_cc(data_dir: Path) -> pd.DataFrame:
    cc = pd.read_csv(resolve(data_dir, "credit_card_balance.csv"),
                     usecols=["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
                              "AMT_DRAWINGS_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT", "AMT_INST_MIN_REGULARITY",
                              "SK_DPD", "SK_DPD_DEF"])
    cc["util"] = safe_div(cc.AMT_BALANCE, cc.AMT_CREDIT_LIMIT_ACTUAL.where(cc.AMT_CREDIT_LIMIT_ACTUAL > 0)).clip(-1, 3)
    cc["util_6m"] = cc.util.where(cc.MONTHS_BALANCE >= -6)
    due = cc.AMT_INST_MIN_REGULARITY > 0
    cc["below_min"] = (cc.AMT_PAYMENT_TOTAL_CURRENT.fillna(0) < cc.AMT_INST_MIN_REGULARITY).astype("float32").where(due)
    cc["dpd"] = cc.SK_DPD > 0
    cc["dpd_12m"] = cc.SK_DPD.where(cc.MONTHS_BALANCE >= -12)
    g = cc.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "CC_N_MONTHS": g.size(), "CC_UTIL_MEAN": g.util.mean(), "CC_UTIL_MAX": g.util.max(),
        "CC_UTIL_MEAN_6M": g.util_6m.mean(), "CC_BALANCE_MEAN": g.AMT_BALANCE.mean(),
        "CC_LIMIT_MEAN": g.AMT_CREDIT_LIMIT_ACTUAL.mean(), "CC_DRAWINGS_MEAN": g.AMT_DRAWINGS_CURRENT.mean(),
        "CC_SHARE_BELOW_MIN_PAYMENT": g.below_min.mean(), "CC_MAX_DPD": g.SK_DPD.max(),
        "CC_SHARE_MONTHS_DPD": g.dpd.mean(), "CC_MAX_DPD_12M": g.dpd_12m.max()})
    S = "credit_card_balance"
    for k, f, m in [
        ("CC_N_MONTHS", "count monthly rows", "Length of card history"),
        ("CC_UTIL_MEAN", "mean AMT_BALANCE / limit (limit > 0, clipped [-1, 3])", "Average card utilisation"),
        ("CC_UTIL_MAX", "max monthly utilisation", "Peak card utilisation"),
        ("CC_UTIL_MEAN_6M", "mean utilisation, MONTHS_BALANCE >= -6", "Recent card utilisation"),
        ("CC_BALANCE_MEAN", "mean AMT_BALANCE", "Typical card balance"),
        ("CC_LIMIT_MEAN", "mean AMT_CREDIT_LIMIT_ACTUAL", "Typical card limit"),
        ("CC_DRAWINGS_MEAN", "mean AMT_DRAWINGS_CURRENT", "Typical monthly drawings"),
        ("CC_SHARE_BELOW_MIN_PAYMENT", "share of months with min due > 0 and total payment < min due",
         "Frequency of paying less than the minimum (repayment stress)"),
        ("CC_MAX_DPD", "max SK_DPD", "Worst card delinquency"),
        ("CC_SHARE_MONTHS_DPD", "mean(SK_DPD > 0)", "Frequency of card delinquency"),
        ("CC_MAX_DPD_12M", "max SK_DPD, MONTHS_BALANCE >= -12", "Recent card delinquency"),
    ]:
        reg(k, S, f, m)
    return out


# --------------------------------------------------------------------------- installments_payments
def build_installments(data_dir: Path) -> pd.DataFrame:
    inst = pd.read_csv(resolve(data_dir, "installments_payments.csv"),
                       dtype={"SK_ID_PREV": "int32", "SK_ID_CURR": "int32", "NUM_INSTALMENT_VERSION": "float32",
                              "NUM_INSTALMENT_NUMBER": "int16", "DAYS_INSTALMENT": "float32",
                              "DAYS_ENTRY_PAYMENT": "float32", "AMT_INSTALMENT": "float64", "AMT_PAYMENT": "float64"})
    n = len(inst)
    key = (inst.SK_ID_PREV.to_numpy(np.int64) * 1_000_000
           + inst.NUM_INSTALMENT_VERSION.to_numpy().astype(np.int64) * 1_000
           + inst.NUM_INSTALMENT_NUMBER.to_numpy(np.int64))
    order = np.argsort(key, kind="stable")
    key = key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    del key
    take = lambda c: inst[c].to_numpy()[order]
    per = pd.DataFrame({
        "SK_ID_CURR": take("SK_ID_CURR")[starts],
        "due_amt": np.maximum.reduceat(take("AMT_INSTALMENT"), starts),
        "paid": np.add.reduceat(np.nan_to_num(take("AMT_PAYMENT")), starts),
        "due_day": np.maximum.reduceat(take("DAYS_INSTALMENT"), starts).astype("float32"),
        "last_pay": np.fmax.reduceat(take("DAYS_ENTRY_PAYMENT"), starts).astype("float32")})
    del inst, order, starts
    gc.collect()
    per["missed"] = per.last_pay.isna()
    per["late"] = (per.last_pay > per.due_day) | per.missed
    per["days_late"] = (per.last_pay - per.due_day).clip(lower=0)
    per["days_early"] = (per.due_day - per.last_pay).clip(lower=0, upper=365)   # cap extreme early payments
    per["underpaid"] = per.paid < per.due_amt - 0.01
    recent = per.due_day >= -365
    per["late_1y"] = per.late.astype("float32").where(recent)
    per["days_late_1y"] = per.days_late.where(recent)
    g = per.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "INST_N": g.size(), "INST_N_LATE": g.late.sum(), "INST_N_MISSED": g.missed.sum(),
        "INST_SHARE_LATE": g.late.mean(), "INST_SHARE_LATE_1Y": g.late_1y.mean(),
        "INST_MAX_DAYS_LATE": g.days_late.max(), "INST_MEAN_DAYS_LATE": g.days_late.mean(),
        "INST_MAX_DAYS_LATE_1Y": g.days_late_1y.max(), "INST_MEAN_DAYS_EARLY": g.days_early.mean(),
        "INST_SHARE_UNDERPAID": g.underpaid.mean(),
        "INST_PAID_TO_DUE": safe_div(g.paid.sum(), g.due_amt.sum()),
        "INST_DAYS_LAST_DUE": g.due_day.max()})
    S = "installments_payments (payments summed per instalment first)"
    for k, f, m in [
        ("INST_N", "count of instalments", "Length of repayment history"),
        ("INST_N_LATE", "count instalments with last payment after due date or no payment", "Late instalments"),
        ("INST_N_MISSED", "count instalments with no recorded payment", "Missed instalments"),
        ("INST_SHARE_LATE", "late / instalments", "Repayment discipline"),
        ("INST_SHARE_LATE_1Y", "late / instalments, due in last 365 days", "Recent repayment discipline"),
        ("INST_MAX_DAYS_LATE", "max(last payment day - due day, 0)", "Worst lateness"),
        ("INST_MEAN_DAYS_LATE", "mean(last payment day - due day, floored at 0)", "Typical lateness"),
        ("INST_MAX_DAYS_LATE_1Y", "max days late, due in last 365 days", "Recent worst lateness"),
        ("INST_MEAN_DAYS_EARLY", "mean(due day - last payment day), clipped [0, 365]", "Habit of paying ahead"),
        ("INST_SHARE_UNDERPAID", "share instalments with total paid < amount due", "Partial payment frequency"),
        ("INST_PAID_TO_DUE", "sum paid / sum due", "Overall repayment completeness"),
        ("INST_DAYS_LAST_DUE", "max DAYS_INSTALMENT", "Recency of last scheduled instalment"),
    ]:
        reg(k, S, f, m)
    report_rows = {"installment_rows": n, "instalments": int(len(per))}
    del per, g
    gc.collect()
    out.attrs["report"] = report_rows
    return out


# --------------------------------------------------------------------------- assembly
def main(data_dir: str, out_dir: str, split_file: str | None) -> None:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    log("application"); app = build_application(data_dir)
    n_train, n_test = (app.DATASET == "train").sum(), (app.DATASET == "test").sum()
    report.update(app_train_rows=int(n_train), app_test_rows=int(n_test))

    blocks = [("PREV", build_previous, (data_dir,)), ("BUREAU", build_bureau, (data_dir, report)),
              ("POS", build_pos, (data_dir,)), ("CC", build_cc, (data_dir,)),
              ("INST", build_installments, (data_dir,))]
    for flag, fn, args in blocks:
        log(fn.__name__)
        agg = fn(*args)
        assert agg.index.is_unique, f"{fn.__name__} not one row per applicant"
        report.update(agg.attrs.get("report", {}))
        agg = agg.astype("float32")
        agg[f"HAS_{flag}"] = np.int8(1)
        src = {"PREV": "previous_application", "BUREAU": "bureau", "POS": "POS_CASH_balance",
               "CC": "credit_card_balance", "INST": "installments_payments"}[flag]
        reg(f"HAS_{flag}", src, "1 if applicant has any records", "History available (absence is information)")
        before = len(app)
        app = app.merge(agg, left_on="SK_ID_CURR", right_index=True, how="left", validate="one_to_one")
        assert len(app) == before, "join changed row count"
        app[f"HAS_{flag}"] = app[f"HAS_{flag}"].fillna(0).astype("int8")
        counts = [c for c in agg.columns if ("_N_" in c or c.endswith("_N")) and not c.startswith("HAS_")]
        app[counts] = app[counts].fillna(0)
        report[f"coverage_pct_{flag}"] = round(100 * app[f"HAS_{flag}"].mean(), 2)
        del agg
        gc.collect()
    app["HAS_BUREAU_BALANCE"] = app["HAS_BUREAU_BALANCE"].fillna(0).astype("int8")

    # Cross-table affordability: known obligations relative to income
    inc = app.AMT_INCOME_TOTAL
    app["X_BUR_DEBT_TO_INCOME"] = safe_div(app.BUR_ACTIVE_DEBT_SUM, inc)
    app["X_TOTAL_ANNUITY_TO_INCOME"] = safe_div(
        app.AMT_ANNUITY + app.BUR_ACTIVE_ANNUITY_SUM.fillna(0) + app.PREV_APPR_ACTIVE_ANNUITY_SUM.fillna(0), inc)
    reg("X_BUR_DEBT_TO_INCOME", "application + bureau", "BUR_ACTIVE_DEBT_SUM / AMT_INCOME_TOTAL",
        "Outstanding external debt relative to income")
    reg("X_TOTAL_ANNUITY_TO_INCOME", "application + bureau + previous_application",
        "(AMT_ANNUITY + active bureau annuities + active Home Credit annuities) / AMT_INCOME_TOTAL",
        "Total known instalment burden; unknown external annuities count as 0, so this is a lower bound")

    # Split assignment (train only): -1 = locked holdout, 0..4 = CV folds
    if split_file and Path(split_file).exists():
        sp = pd.read_csv(split_file)
        app = app.merge(sp, on="SK_ID_CURR", how="left", validate="one_to_one")
        assert app.loc[app.DATASET == "train", "fold"].notna().all()
        app["fold"] = app.fold.astype("Int8")

    # Final checks
    assert app.SK_ID_CURR.is_unique
    tr_abt = app[app.DATASET == "train"].drop(columns="DATASET").reset_index(drop=True)
    te_abt = app[app.DATASET == "test"].drop(columns=["DATASET", "TARGET", "fold"], errors="ignore").reset_index(drop=True)
    assert len(tr_abt) == n_train and len(te_abt) == n_test
    assert set(tr_abt.SK_ID_CURR).isdisjoint(te_abt.SK_ID_CURR)
    assert tr_abt.TARGET.notna().all()

    tr_abt.to_parquet(out_dir / "abt_train.parquet", index=False)
    te_abt.to_parquet(out_dir / "abt_test.parquet", index=False)
    dd = pd.DataFrame(DICT).drop_duplicates("feature")
    dd.to_csv(out_dir / "data_dictionary.csv", index=False)
    report.update(abt_train_shape=list(tr_abt.shape), abt_test_shape=list(te_abt.shape),
                  engineered_features=len(dd), target_rate_pct=round(100 * tr_abt.TARGET.mean(), 3))
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"done: train {tr_abt.shape}, test {te_abt.shape}, engineered {len(dd)}")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--split-file", default="data/processed/split_assignment.csv")
    a = ap.parse_args()
    main(a.data_dir, a.out, a.split_file)
