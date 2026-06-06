# Last.fm Repeat-Listening Recommender

This project builds a machine learning pipeline that predicts whether a user will listen to the same song again within 30 days, then ranks tracks by repeat-listening probability to generate recommendations.

## What It Does

- Cleans the raw Last.fm listening log.
- Combines `Date` and `Time` into a timestamp.
- Builds user-track history features.
- Trains and compares Logistic Regression, Random Forest, and XGBoost.
- Exports model artifacts and recommendation outputs.
- Produces EDA and diagnostic plots.

## Files

- [music_recommendation_pipeline.py](music_recommendation_pipeline.py) - standalone training and export script.
- [Rhombix_Tech.ipynb](Rhombix_Tech.ipynb) - notebook version of the same workflow.
- [requirements.txt](requirements.txt) - Python dependencies.
- `artifacts/` - generated models, CSV outputs, and plots.

## Run

```bash
python music_recommendation_pipeline.py
```

## Output Artifacts

- `repeat_listening_classification_model.joblib`
- `repeat_listening_ranking_model.joblib`
- `model_comparison_results.csv`
- `xgboost_feature_importance.csv`
- `all_candidate_recommendations.csv`
- `top_5_user_recommendations.csv`
- `run_metadata.json`
- `eda_summary.png`
- `roc_curve_comparison.png`
- `xgboost_feature_importance.png`

## Notes

The classification winner is chosen by F1 score, while the ranking model is chosen by ROC-AUC. This keeps the thresholded decision task and the recommendation task aligned with their respective goals.