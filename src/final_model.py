"""
Final Diabetes 30-Day Readmission Model
=======================================

Dataset:
    UCI Diabetes 130-US Hospitals for Years 1999-2008

Goal:
    Predict whether a patient will be readmitted within 30 days.

Final modeling decisions:
    - Positive target: readmitted == "<30"
    - Remove expired/hospice discharge outcomes
    - Group rare medical specialties into "Other" using TRAINING data only
    - Exclude discharge_disposition_id from predictive features
    - Split by patient ID to reduce leakage
    - Train XGBoost with class-imbalance adjustment
    - Evaluate ROC-AUC, PR-AUC, precision, recall, F1
    - Create polished SHAP visualizations for portfolio use

Project structure:
    diabetes-readmission/
    ├── data/
    │   └── diabetic_data.csv
    ├── results/
    ├── src/
    │   └── final_model.py
    └── visuals/

Run from the project root:
    python src/final_model.py

Outputs:
    - final_model_metrics.csv
    - final_model_summary.txt
    - shap_beeswarm_final.png
    - shap_bar_final.png
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
@dataclass
class DataConfig:
    target_col: str = "readmitted_30d"
    patient_id_col: str = "patient_nbr"

    numeric_cols: list[str] = field(
        default_factory=lambda: [
            "age_midpoint",
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
        ]
    )

    categorical_cols: list[str] = field(
        default_factory=lambda: [
            "race",
            "gender",
            "admission_type_id",
            "admission_source_id",
            "medical_specialty",
            "max_glu_serum",
            "A1Cresult",
            "change",
            "diabetesMed",
            "insulin",
            "diag_1_group",
        ]
    )

    missing_flag_cols: list[str] = field(
        default_factory=lambda: ["race", "medical_specialty"]
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "diabetic_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALS_DIR = PROJECT_ROOT / "visuals"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

# Excluded because these outcomes make ordinary 30-day readmission analysis
# inappropriate or structurally impossible.
EXCLUDED_DISCHARGE_CODES = {11, 13, 14, 19, 20, 21}

# Specialties appearing fewer than this many times in TRAINING data
# are grouped into "Other".
MIN_SPECIALTY_COUNT = 200

RANDOM_STATE = 42
MAX_SHAP_SAMPLES = 2000
MAX_DISPLAY_FEATURES = 12


# ----------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------
def age_to_midpoint(value) -> float:
    """Convert age bins such as '[60-70)' into numeric midpoints."""
    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    try:
        left, right = text.strip("[]()").split("-")
        return (float(left) + float(right)) / 2.0
    except Exception:
        return np.nan


def group_icd9(code) -> str:
    """Map an ICD-9 diagnosis code into a broad diagnostic category."""
    if pd.isna(code):
        return "Missing"

    text = str(code).strip()

    if text.startswith("V"):
        return "Supplementary"
    if text.startswith("E"):
        return "ExternalCause"

    try:
        value = float(text)
    except ValueError:
        return "Other"

    if 390 <= value < 460 or value == 785:
        return "Circulatory"
    if 460 <= value < 520 or value == 786:
        return "Respiratory"
    if 520 <= value < 580 or value == 787:
        return "Digestive"
    if 250 <= value < 251:
        return "Diabetes"
    if 800 <= value < 1000:
        return "Injury"
    if 710 <= value < 740:
        return "Musculoskeletal"
    if 580 <= value < 630 or value == 788:
        return "Genitourinary"
    if 140 <= value < 240:
        return "Neoplasms"
    if 240 <= value < 280:
        return "Endocrine"
    if 680 <= value < 710 or value == 782:
        return "Skin"
    if 1 <= value < 140:
        return "Infectious"
    if 290 <= value < 320:
        return "Mental"
    if 280 <= value < 290:
        return "Blood"
    if 320 <= value < 390:
        return "Nervous"
    if 630 <= value < 680:
        return "Pregnancy"
    if 740 <= value < 760:
        return "Congenital"
    if 760 <= value < 780:
        return "Perinatal"
    if 780 <= value < 800:
        return "Symptoms"

    return "Other"


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load, prepare, and clean the UCI dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. Place diabetic_data.csv "
            "in the same folder as final_model.py."
        )

    df = pd.read_csv(
        csv_path,
        na_values=["?"],
        low_memory=False,
    )

    required = {
        "patient_nbr",
        "readmitted",
        "age",
        "diag_1",
        "discharge_disposition_id",
        "medical_specialty",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Binary target: readmitted within 30 days.
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)

    # Feature engineering.
    df["age_midpoint"] = df["age"].apply(age_to_midpoint)
    df["diag_1_group"] = df["diag_1"].apply(group_icd9)

    df["discharge_disposition_id"] = pd.to_numeric(
        df["discharge_disposition_id"],
        errors="coerce",
    )

    original_rows = len(df)

    df = df.loc[
        ~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_CODES)
    ].copy()

    removed_rows = original_rows - len(df)

    logger.info(f"Original rows: {original_rows:,}")
    logger.info(f"Removed expired/hospice rows: {removed_rows:,}")
    logger.info(f"Final cohort rows: {len(df):,}")
    logger.info(
        f"30-day readmission rate: {df['readmitted_30d'].mean():.2%}"
    )

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. Preprocessing
# ----------------------------------------------------------------------
class MedicalDataPreprocessor:
    """Preprocess tabular clinical data while reducing leakage risk."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.pipeline: Optional[ColumnTransformer] = None

    def validate_columns(self, df: pd.DataFrame) -> None:
        needed = (
            set(self.config.numeric_cols)
            | set(self.config.categorical_cols)
            | {self.config.target_col, self.config.patient_id_col}
        )

        missing = needed - set(df.columns)

        if missing:
            raise ValueError(
                "Missing configured columns: "
                + ", ".join(sorted(missing))
            )

    def add_missing_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.config.missing_flag_cols:
            if col in df.columns:
                df[f"{col}_is_missing"] = df[col].isna().astype(int)

        return df

    def patient_level_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )

        groups = df[self.config.patient_id_col]
        train_idx, test_idx = next(splitter.split(df, groups=groups))

        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )

    def build_transformer(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="Unknown",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                    ),
                ),
            ]
        )

        missing_flag_names = [
            f"{col}_is_missing"
            for col in self.config.missing_flag_cols
        ]

        transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.config.numeric_cols),
                ("cat", categorical_pipeline, self.config.categorical_cols),
                ("flags", "passthrough", missing_flag_names),
            ],
            remainder="drop",
        )

        self.pipeline = transformer
        return transformer

    def get_feature_names(self) -> list[str]:
        if self.pipeline is None:
            raise RuntimeError(
                "The transformer must be fit before retrieving feature names."
            )

        numeric_names = list(self.config.numeric_cols)

        encoder = (
            self.pipeline
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )

        categorical_names = list(
            encoder.get_feature_names_out(
                self.config.categorical_cols
            )
        )

        flag_names = [
            f"{col}_is_missing"
            for col in self.config.missing_flag_cols
        ]

        return numeric_names + categorical_names + flag_names


def group_rare_specialties(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Learn specialty frequency thresholds from training data only,
    then apply the learned categories to both splits.
    """
    train = train_df.copy()
    test = test_df.copy()

    train["medical_specialty"] = (
        train["medical_specialty"]
        .fillna("Unknown")
    )

    test["medical_specialty"] = (
        test["medical_specialty"]
        .fillna("Unknown")
    )

    counts = train["medical_specialty"].value_counts()

    kept_specialties = set(
        counts[counts >= MIN_SPECIALTY_COUNT].index
    )

    train["medical_specialty"] = train["medical_specialty"].where(
        train["medical_specialty"].isin(kept_specialties),
        "Other",
    )

    test["medical_specialty"] = test["medical_specialty"].where(
        test["medical_specialty"].isin(kept_specialties),
        "Other",
    )

    logger.info(
        f"Medical specialties: {len(counts)} original -> "
        f"{len(kept_specialties) + 1} modeled categories "
        f"(including Other)"
    )

    return train, test


# ----------------------------------------------------------------------
# 4. Model
# ----------------------------------------------------------------------
def build_xgboost(y_train: np.ndarray) -> XGBClassifier:
    """Create the final XGBoost model with class-imbalance adjustment."""
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)

    scale_pos_weight = negatives / max(positives, 1)

    logger.info(
        f"Class balance: 1 positive to {scale_pos_weight:.1f} negatives"
    )

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_model(
    model: XGBClassifier,
    X_test,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Evaluate the final model."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
    }


# ----------------------------------------------------------------------
# 5. Plotting helpers
# ----------------------------------------------------------------------
def pretty_feature_name(name: str) -> str:
    """Convert encoded feature names into clean, human-readable plot labels."""
    direct_names = {
        "age_midpoint": "Age",
        "time_in_hospital": "Time in hospital",
        "num_lab_procedures": "Lab procedures",
        "num_procedures": "Procedures",
        "num_medications": "Medications",
        "number_outpatient": "Prior outpatient visits",
        "number_emergency": "Prior emergency visits",
        "number_inpatient": "Prior inpatient visits",
        "number_diagnoses": "Number of diagnoses",
        "race_is_missing": "Race missing",
        "medical_specialty_is_missing": "Medical specialty missing",
    }

    if name in direct_names:
        return direct_names[name]

    admission_type_labels = {
        "1": "Emergency admission",
        "2": "Urgent admission",
        "3": "Elective admission",
        "4": "Newborn admission",
        "5": "Admission type unavailable",
        "6": "Admission type missing",
        "7": "Trauma center admission",
        "8": "Admission type not mapped",
    }

    admission_source_labels = {
        "1": "Physician referral",
        "2": "Clinic referral",
        "3": "HMO referral",
        "4": "Transfer from another hospital",
        "5": "Transfer from skilled nursing facility",
        "6": "Transfer from another health facility",
        "7": "Emergency room",
        "8": "Court / law enforcement",
        "9": "Admission source unavailable",
    }

    specialty_labels = {
        "PhysicalMedicineandRehabilitation": "Physical Medicine & Rehabilitation",
        "ObstetricsandGynecology": "Obstetrics & Gynecology",
        "InternalMedicine": "Internal Medicine",
        "Family/GeneralPractice": "Family / General Practice",
        "Surgery-Cardiovascular/Thoracic": "Cardiovascular / Thoracic Surgery",
        "Orthopedics-Reconstructive": "Reconstructive Orthopedics",
        "Hematology/Oncology": "Hematology / Oncology",
        "Pediatrics-Endocrinology": "Pediatric Endocrinology",
    }

    if name.startswith("admission_type_id_"):
        code = name.removeprefix("admission_type_id_")
        return admission_type_labels.get(code, f"Admission type {code}")

    if name.startswith("admission_source_id_"):
        code = name.removeprefix("admission_source_id_")
        return admission_source_labels.get(code, f"Admission source {code}")

    if name.startswith("medical_specialty_"):
        specialty = name.removeprefix("medical_specialty_")
        specialty = specialty_labels.get(
            specialty,
            specialty.replace("_", " "),
        )
        return f"Specialty: {specialty}"

    prefix_map = {
        "diag_1_group_": "Diagnosis: ",
        "race_": "Race: ",
        "gender_": "Gender: ",
        "max_glu_serum_": "Max glucose: ",
        "A1Cresult_": "A1C result: ",
        "change_": "Medication change: ",
        "diabetesMed_": "Diabetes medication: ",
        "insulin_": "Insulin: ",
    }

    for prefix, replacement in prefix_map.items():
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            return replacement + suffix.replace("_", " ")

    return name.replace("_", " ").title()


def normalize_shap_values(shap_values) -> np.ndarray:
    """Normalize SHAP output into a 2D array of shape (rows, features)."""
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    values = np.asarray(shap_values)

    if values.ndim == 3:
        # Handle explainers that return an output-class dimension.
        values = values[:, :, -1]

    if values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP value shape: {values.shape}"
        )

    return values


def generate_shap_plots(
    model: XGBClassifier,
    X_test,
    feature_names: list[str],
) -> None:
    """Create polished SHAP beeswarm and bar plots."""
    import shap

    rng = np.random.default_rng(RANDOM_STATE)

    sample_size = min(MAX_SHAP_SAMPLES, X_test.shape[0])

    if sample_size < X_test.shape[0]:
        sample_idx = rng.choice(
            X_test.shape[0],
            size=sample_size,
            replace=False,
        )
        X_sample = X_test[sample_idx]
    else:
        X_sample = X_test

    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = normalize_shap_values(
        explainer.shap_values(X_sample)
    )

    pretty_names = [
        pretty_feature_name(name)
        for name in feature_names
    ]

    # --------------------------------------------------------------
    # A. Beeswarm plot
    # --------------------------------------------------------------
    plt.figure(figsize=(10.5, 7.5))

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=pretty_names,
        max_display=MAX_DISPLAY_FEATURES,
        plot_size=None,
        show=False,
    )

    plt.title(
        "Drivers of 30-Day Readmission Predictions",
        fontsize=16,
        pad=18,
        weight="semibold",
    )
    plt.xlabel(
        "SHAP value (impact on predicted readmission risk)",
        fontsize=11,
    )
    plt.ylabel("")
    plt.tight_layout()

    beeswarm_path = VISUALS_DIR / "shap_beeswarm_final.png"

    plt.savefig(
        beeswarm_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # --------------------------------------------------------------
    # B. Clean mean-absolute-SHAP bar plot
    # --------------------------------------------------------------
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": pretty_names,
            "importance": mean_abs_shap,
        }
    )

    importance = (
        importance
        .sort_values("importance", ascending=False)
        .head(MAX_DISPLAY_FEATURES)
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.8))

    ax.barh(
        importance["feature"],
        importance["importance"],
    )

    ax.set_title(
        "Top Predictors of 30-Day Readmission",
        fontsize=16,
        pad=16,
        weight="semibold",
    )
    ax.set_xlabel(
        "Mean absolute SHAP value",
        fontsize=11,
    )
    ax.set_ylabel("")
    ax.grid(
        axis="x",
        alpha=0.2,
    )

    # Remove unnecessary chart borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    bar_path = VISUALS_DIR / "shap_bar_final.png"

    plt.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    logger.info(f"Saved SHAP beeswarm plot: {beeswarm_path.name}")
    logger.info(f"Saved SHAP bar plot: {bar_path.name}")


# ----------------------------------------------------------------------
# 6. Final pipeline
# ----------------------------------------------------------------------
def main() -> None:
    df = load_and_clean_data(DATA_PATH)

    config = DataConfig()
    preprocessor = MedicalDataPreprocessor(config)

    # Patient-level split before any frequency-based category grouping.
    train_df, test_df = preprocessor.patient_level_split(df)

    logger.info(f"Training rows: {len(train_df):,}")
    logger.info(f"Test rows: {len(test_df):,}")

    # Prevent rare-category information from leaking from test to train.
    train_df, test_df = group_rare_specialties(
        train_df,
        test_df,
    )

    preprocessor.validate_columns(train_df)
    preprocessor.validate_columns(test_df)

    train_df = preprocessor.add_missing_flags(train_df)
    test_df = preprocessor.add_missing_flags(test_df)

    transformer = preprocessor.build_transformer()

    columns_to_drop = [
        config.target_col,
        config.patient_id_col,
    ]

    X_train = transformer.fit_transform(
        train_df.drop(columns=columns_to_drop)
    )

    X_test = transformer.transform(
        test_df.drop(columns=columns_to_drop)
    )

    y_train = train_df[config.target_col].to_numpy()
    y_test = test_df[config.target_col].to_numpy()

    feature_names = preprocessor.get_feature_names()

    if len(feature_names) != X_test.shape[1]:
        raise RuntimeError(
            f"Feature-name mismatch: {len(feature_names)} names for "
            f"{X_test.shape[1]} transformed columns."
        )

    logger.info(f"Final encoded feature count: {len(feature_names)}")

    model = build_xgboost(y_train)
    model.fit(X_train, y_train)

    results = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=0.5,
    )

    print("\n=== Final Model Results ===")
    print(f"ROC-AUC:   {results['roc_auc']:.3f}")
    print(f"PR-AUC:    {results['pr_auc']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall:    {results['recall']:.3f}")
    print(f"F1-score:  {results['f1']:.3f}")

    print("\nConfusion Matrix:")
    print(np.array(results["confusion_matrix"]))

    # Save structured metrics.
    metrics_df = pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "Cohort": "Expired/hospice removed",
                "Rare_Specialties_Grouped": True,
                "Discharge_Disposition_Excluded": True,
                "Training_Rows": len(train_df),
                "Test_Rows": len(test_df),
                "Feature_Count": len(feature_names),
                "ROC_AUC": results["roc_auc"],
                "PR_AUC": results["pr_auc"],
                "Precision": results["precision"],
                "Recall": results["recall"],
                "F1_Score": results["f1"],
            }
        ]
    )

    metrics_path = RESULTS_DIR / "final_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Save human-readable summary.
    summary_path = RESULTS_DIR / "final_model_summary.txt"

    with summary_path.open("w") as f:
        f.write("Final Diabetes 30-Day Readmission Model\n")
        f.write("=======================================\n\n")
        f.write("Dataset: UCI Diabetes 130-US Hospitals\n")
        f.write("Model: XGBoost\n")
        f.write("Target: 30-day readmission\n")
        f.write("Expired/hospice outcomes removed: yes\n")
        f.write("Rare medical specialties grouped: yes\n")
        f.write("Discharge disposition excluded: yes\n")
        f.write("Patient-level train/test split: yes\n\n")
        f.write(f"Training rows: {len(train_df):,}\n")
        f.write(f"Test rows: {len(test_df):,}\n")
        f.write(f"Encoded features: {len(feature_names)}\n\n")
        f.write(f"ROC-AUC: {results['roc_auc']:.3f}\n")
        f.write(f"PR-AUC: {results['pr_auc']:.3f}\n")
        f.write(f"Precision: {results['precision']:.3f}\n")
        f.write(f"Recall: {results['recall']:.3f}\n")
        f.write(f"F1-score: {results['f1']:.3f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(np.array(results["confusion_matrix"])))
        f.write("\n")

    logger.info(f"Saved metrics: {metrics_path.name}")
    logger.info(f"Saved summary: {summary_path.name}")

    # Create portfolio-ready interpretation plots.
    logger.info("Generating polished SHAP plots...")
    generate_shap_plots(
        model,
        X_test,
        feature_names,
    )

    print("\n=== Output Files ===")
    print("results/final_model_metrics.csv")
    print("results/final_model_summary.txt")
    print("visuals/shap_beeswarm_final.png")
    print("visuals/shap_bar_final.png")
    print("\nFinal model pipeline completed successfully.")


if __name__ == "__main__":
    main()
