# Person B: Candidate Classifier

Final model: Tuned XGBoost (sklearn Pipeline: preprocessing + classifier).

- Sampled GroupKFold (by Trip_ID) macro-F1 = 0.827 (tuned Random Forest = 0.822, comparable).
- Student-test macro-F1: 0.816 (tuned, sample-fitted), 0.832 (final model fitted on all training rows).
- Out-of-fold (5-fold GroupKFold, all 1.94M rows): macro-F1 0.803, AUC 0.939.

## Where things are

- models/PathSnap_PersonB_All_In_One.ipynb : full training, comparison and tuning notebook
- models/feature_engineering.py : feature engineering used by the model
- models/requirements_modeling.txt : exact package versions
- saved_models/best_candidate_classifier.joblib, model_metadata.json : final model
- results/ : CV results, Optuna trial logs, student-test comparison, tuned vs untuned
- visualizations/ : comparison plots and confusion matrices

## Usage

    import joblib
    from models.feature_engineering import engineer_features

    model = joblib.load("saved_models/best_candidate_classifier.joblib")
    X = engineer_features(candidates_df)
    prob = model.predict_proba(X)[:, 1]   # P(Label = 1)

## Required input columns (from candidate generation)

distance_m, search_radius_m, gps_speed_mps, segment_length_m,
heading_difference_deg, gps_heading_deg, segment_heading_deg, speed_quality

## Notes

- Label 1 = candidate segment is the correct match (confirm with Person A).
- Probabilities are NOT calibrated (scale_pos_weight ~ 2.46; log loss 0.18 vs 0.15 untuned). Calibrate before using as HMM emissions.
- The final model is trained on all 80 training trips. For a fair classifier-only vs HMM comparison on training trips, use the out-of-fold predictions (shared separately).
- Required versions: scikit-learn 1.6.1, xgboost 3.4.1, pandas 2.2.3, numpy 2.4.6.
