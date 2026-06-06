from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("xgboost is required. Install dependencies with: pip install -r requirements.txt") from exc


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a repeat-listening recommender from Last.fm logs.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parent / "Last.fm_data.csv",
        help="Path to the Last.fm listening history CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory where models and reports will be saved.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving visualization PNG files.",
    )
    return parser.parse_args()


def load_and_clean_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_raw = pd.read_csv(data_path)
    df_raw = df_raw.loc[:, ~df_raw.columns.astype(str).str.contains("^Unnamed", case=False, na=False)]
    if "" in df_raw.columns:
        df_raw = df_raw.drop(columns=[""])
    df_raw.columns = [str(col).strip() for col in df_raw.columns]

    df = df_raw.copy()
    for column in ["Username", "Artist", "Track", "Album", "Date", "Time"]:
        df[column] = df[column].astype("string").str.strip()

    df["timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d %b %Y %H:%M",
        errors="coerce",
    )

    df = df.dropna(subset=["Username", "Artist", "Track", "Album", "timestamp"])
    df = df.drop_duplicates(subset=["Username", "Artist", "Track", "Album", "timestamp"])
    df = df.sort_values(["Username", "timestamp", "Artist", "Track"]).reset_index(drop=True)
    return df_raw, df


def cumulative_unique_before(series: pd.Series) -> pd.Series:
    newly_seen = ~series.duplicated()
    return newly_seen.cumsum().shift(fill_value=0)


def rolling_count_before(group: pd.DataFrame, window_days: int = 30) -> pd.Series:
    timestamps = group["timestamp"].to_numpy(dtype="datetime64[ns]")
    counts = np.zeros(len(group), dtype=np.int32)
    left = 0
    window = np.timedelta64(window_days, "D")

    for idx, current_ts in enumerate(timestamps):
        lower_bound = current_ts - window
        while left < idx and timestamps[left] < lower_bound:
            left += 1
        counts[idx] = idx - left

    return pd.Series(counts, index=group.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy().sort_values(["Username", "timestamp", "Artist", "Track"]).reset_index(drop=True)

    model_df["day_of_week"] = model_df["timestamp"].dt.day_name()
    model_df["hour_of_day"] = model_df["timestamp"].dt.hour
    model_df["hour_bucket"] = pd.cut(
        model_df["hour_of_day"],
        bins=[-1, 5, 11, 17, 21, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"],
        ordered=True,
    )

    model_df["next_same_track_timestamp"] = model_df.groupby(["Username", "Track"])["timestamp"].shift(-1)
    model_df["days_to_next_same_track"] = (
        model_df["next_same_track_timestamp"] - model_df["timestamp"]
    ).dt.total_seconds() / 86400
    model_df["target_repeat_30d"] = (
        model_df["days_to_next_same_track"].between(0, 30, inclusive="both").fillna(False).astype(int)
    )

    model_df["user_total_plays_before"] = model_df.groupby("Username").cumcount()
    model_df["track_plays_by_user_before"] = model_df.groupby(["Username", "Track"]).cumcount()
    model_df["days_since_prev_user_listen"] = model_df.groupby("Username")["timestamp"].diff().dt.total_seconds() / 86400
    model_df["days_since_prev_same_track"] = (
        model_df.groupby(["Username", "Track"])["timestamp"].diff().dt.total_seconds() / 86400
    )
    model_df["user_unique_tracks_before"] = model_df.groupby("Username")["Track"].transform(cumulative_unique_before)
    model_df["user_unique_artists_before"] = model_df.groupby("Username")["Artist"].transform(cumulative_unique_before)
    model_df["user_activity_30d"] = (
        model_df.groupby("Username", group_keys=False).apply(rolling_count_before).sort_index().astype(int)
    )
    model_df["user_track_frequency_30d"] = (
        model_df.groupby(["Username", "Track"], group_keys=False).apply(rolling_count_before).sort_index().astype(int)
    )
    model_df["artist_popularity"] = model_df.groupby("Artist")["Artist"].transform("count")
    model_df["album_popularity"] = model_df.groupby("Album")["Album"].transform("count")

    model_df["days_since_prev_user_listen"] = model_df["days_since_prev_user_listen"].fillna(-1)
    model_df["days_since_prev_same_track"] = model_df["days_since_prev_same_track"].fillna(-1)
    model_df["days_to_next_same_track"] = model_df["days_to_next_same_track"].fillna(-1)

    return model_df


def create_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def split_data(model_df: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    cutoff_timestamp = model_df["timestamp"].quantile(0.80)
    train_df = model_df[model_df["timestamp"] <= cutoff_timestamp].copy()
    test_df = model_df[model_df["timestamp"] > cutoff_timestamp].copy()

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]
    return train_df, test_df, X_train, X_test, y_train, y_test


def fit_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    positive_rate = y_train.mean()
    scale_pos_weight = (1 - positive_rate) / positive_rate if positive_rate > 0 else 1.0

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
        ),
    }

    results = []
    trained_models: dict[str, Pipeline] = {}
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
        )
        trained_models[name] = pipeline
        roc_data[name] = roc_curve(y_test, y_proba)

    results_df = pd.DataFrame(results).sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)
    return results_df, trained_models, roc_data


def plot_summary_eda(model_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    track_counts = model_df["Track"].value_counts().head(15).sort_values()
    axes[0, 0].barh(track_counts.index, track_counts.values, color="#2E86AB")
    axes[0, 0].set_title("Most Listened Tracks")
    axes[0, 0].set_xlabel("Plays")

    artist_counts = model_df["Artist"].value_counts().head(15).sort_values()
    axes[0, 1].barh(artist_counts.index, artist_counts.values, color="#A23B72")
    axes[0, 1].set_title("Most Listened Artists")
    axes[0, 1].set_xlabel("Plays")

    user_activity = model_df.groupby("Username").size()
    sns.histplot(user_activity, bins=30, ax=axes[1, 0], color="#F18F01")
    axes[1, 0].set_title("User Activity Distribution")
    axes[1, 0].set_xlabel("Listening Events per User")

    hour_counts = model_df.groupby("hour_of_day").size().reindex(range(24), fill_value=0)
    axes[1, 1].plot(hour_counts.index, hour_counts.values, marker="o", linewidth=2, color="#3B7A57")
    axes[1, 1].set_title("Listening Patterns by Hour")
    axes[1, 1].set_xlabel("Hour of Day")
    axes[1, 1].set_ylabel("Plays")
    axes[1, 1].set_xticks(range(0, 24, 2))

    plt.tight_layout()
    plt.savefig(output_dir / "eda_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(results_df: pd.DataFrame, roc_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], output_dir: Path) -> None:
    plt.figure(figsize=(10, 7))
    for name, (fpr, tpr, _) in roc_data.items():
        auc_score = results_df.loc[results_df["model"] == name, "roc_auc"].iloc[0]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()


def plot_feature_importance(xgb_pipeline: Pipeline, output_dir: Path) -> pd.DataFrame:
    feature_names = xgb_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = xgb_pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    top_importance_df = importance_df.head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_importance_df, y="feature", x="importance", color="#4C72B0")
    plt.title("Top Feature Importances - XGBoost")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    return top_importance_df.reset_index(drop=True)


def generate_recommendations(
    ranking_model: Pipeline,
    model_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    top_n_users: int = 5,
    top_n_tracks: int = 5,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    candidate_df = (
        model_df.sort_values("timestamp")
        .groupby(["Username", "Track"], as_index=False)
        .tail(1)
        .copy()
    )

    candidate_features = candidate_df[numeric_features + categorical_features]
    candidate_df["repeat_probability"] = ranking_model.predict_proba(candidate_features)[:, 1]

    example_users = candidate_df["Username"].drop_duplicates().head(top_n_users)
    recommendation_examples: dict[str, pd.DataFrame] = {}

    for user in example_users:
        user_candidates = candidate_df[candidate_df["Username"] == user].copy()
        recommendation_examples[user] = (
            user_candidates.sort_values("repeat_probability", ascending=False)
            .head(top_n_tracks)[["Artist", "Track", "Album", "repeat_probability"]]
            .reset_index(drop=True)
        )

    recommendation_table = candidate_df[["Username", "Artist", "Track", "Album", "repeat_probability"]].copy()
    return recommendation_table, recommendation_examples


def save_outputs(
    output_dir: Path,
    classification_model_pipeline: Pipeline,
    ranking_model_pipeline: Pipeline,
    results_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    recommendation_table: pd.DataFrame,
    recommendation_examples: dict[str, pd.DataFrame],
    artifact_metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classification_model_pipeline, output_dir / "repeat_listening_classification_model.joblib")
    joblib.dump(ranking_model_pipeline, output_dir / "repeat_listening_ranking_model.joblib")

    results_df.to_csv(output_dir / "model_comparison_results.csv", index=False)
    importance_df.to_csv(output_dir / "xgboost_feature_importance.csv", index=False)
    recommendation_table.sort_values(["Username", "repeat_probability"], ascending=[True, False]).to_csv(
        output_dir / "all_candidate_recommendations.csv",
        index=False,
    )

    example_frames = []
    for user_name, user_frame in recommendation_examples.items():
        frame = user_frame.copy()
        frame.insert(0, "Username", user_name)
        example_frames.append(frame)

    if example_frames:
        pd.concat(example_frames, ignore_index=True).to_csv(output_dir / "top_5_user_recommendations.csv", index=False)

    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(artifact_metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    df_raw, df = load_and_clean_data(args.data_path)
    model_df = engineer_features(df)

    print("Raw rows:", len(df_raw))
    print("Clean rows:", len(df))
    print("Users:", df["Username"].nunique())
    print("Tracks:", df["Track"].nunique())
    print("Artists:", df["Artist"].nunique())
    print("Date range:", df["timestamp"].min(), "->", df["timestamp"].max())

    print("\nTarget balance:")
    print(model_df["target_repeat_30d"].value_counts(normalize=True).rename({0: "No repeat", 1: "Repeat"}).round(3))

    numeric_features = [
        "user_total_plays_before",
        "track_plays_by_user_before",
        "days_since_prev_user_listen",
        "days_since_prev_same_track",
        "user_unique_tracks_before",
        "user_unique_artists_before",
        "user_activity_30d",
        "user_track_frequency_30d",
        "artist_popularity",
        "album_popularity",
        "hour_of_day",
    ]
    categorical_features = ["day_of_week", "hour_bucket"]
    feature_columns = numeric_features + categorical_features
    target_column = "target_repeat_30d"

    if not args.no_plots:
        plot_summary_eda(model_df, args.output_dir)

    train_df, test_df, X_train, X_test, y_train, y_test = split_data(model_df, feature_columns, target_column)
    preprocessor = create_preprocessor(numeric_features, categorical_features)

    results_df, trained_models, roc_data = fit_and_evaluate_models(X_train, y_train, X_test, y_test, preprocessor)

    print("\nModel comparison:\n")
    print(results_df.round(4))

    classification_model_name = results_df.iloc[0]["model"]
    ranking_model_name = results_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    classification_model = trained_models[classification_model_name]
    ranking_model = trained_models[ranking_model_name]

    print("\nBest model for threshold-based classification:", classification_model_name)
    print("Best model for probability ranking:", ranking_model_name)
    print("\nClassification report for the best classification model:\n")
    print(classification_report(y_test, classification_model.predict(X_test), zero_division=0))

    if not args.no_plots:
        plot_roc_curves(results_df, roc_data, args.output_dir)
    importance_df = plot_feature_importance(trained_models["XGBoost"], args.output_dir)

    recommendation_table, recommendation_examples = generate_recommendations(
        ranking_model=ranking_model,
        model_df=model_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    print("\nTop recommendations for sample users:\n")
    for user_name, user_frame in recommendation_examples.items():
        print(f"User: {user_name}")
        print(user_frame.to_string(index=False))
        print()

    artifact_metadata = {
        "classification_model": classification_model_name,
        "ranking_model": ranking_model_name,
        "cutoff_timestamp": str(model_df["timestamp"].quantile(0.80)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "total_users": int(df["Username"].nunique()),
        "total_tracks": int(df["Track"].nunique()),
    }

    save_outputs(
        output_dir=args.output_dir,
        classification_model_pipeline=classification_model,
        ranking_model_pipeline=ranking_model,
        results_df=results_df,
        importance_df=importance_df,
        recommendation_table=recommendation_table,
        recommendation_examples=recommendation_examples,
        artifact_metadata=artifact_metadata,
    )

    print(f"Saved model artifacts and recommendation files to: {args.output_dir}")


if __name__ == "__main__":
    main()