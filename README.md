# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap is an end-to-end machine learning system developed for the PES University Machine Learning Hackathon. The system reconstructs realistic vehicle routes by matching noisy, sparse GPS observations to road-network segments, combining supervised candidate-ranking classifiers with Hidden Markov Model (HMM) Viterbi sequential decoding.

---

## 1. Team Responsibilities & Algorithm Families

The hackathon requires developing, comparing, and benchmarking distinct machine learning families on consistent data:

- **Teammate 2 (This Implementation)**: **Bagging & Tree Ensemble Learning** (`RandomForestClassifier`)
- **Teammate 3**: **Gradient Boosting** (`HistGradientBoostingClassifier`)
- Other teammates: Linear/Probabilistic, Neural Networks, Instance-based (KNN).

---

## 2. Repository Structure

```text
PathSnap/
├── datasets/                                 # Raw trip packages (80 train + 10 test)
│   ├── train/                                # 80 training trips (.track, .nodes, .arcs, .route)
│   └── student_test/                         # 10 student testing trips
├── data/
│   └── processed/
│       ├── train_features.csv.gz             # Shared feature cache (1.85M samples)
│       └── student_test_features.csv.gz       # Shared test feature cache (195K samples)
├── src/
│   ├── preprocessing/                        # SHARED TEAM PREPROCESSING PIPELINE
│   │   ├── parser.py                         # Fast parsing of .track, .nodes, .arcs, .route
│   │   ├── projection.py                     # WGS84 to local metric Cartesian projection
│   │   ├── candidate_gen.py                  # Spatial KDTree search with top-K fallback
│   │   ├── labeler.py                        # Monotonic route sequence ground-truth mapper
│   │   └── feature_engine.py                 # 10 spatial, angular, kinematic features
│   ├── models/
│   │   ├── random_forest.py                  # Teammate 2: Random Forest implementation
│   │   ├── gradient_boosting.py              # Teammate 3: Gradient Boosting implementation
│   │   ├── evaluation.py                     # Candidate metrics & Top-1 accuracy
│   │   ├── calibration.py                    # Brier score & reliability diagrams
│   │   └── bayesian_optimizer.py             # Gaussian Process Expected Improvement engine
│   ├── hmm/
│   │   ├── transition.py                     # NetworkX directed road graph transitions
│   │   └── viterbi.py                        # Log-space Viterbi dynamic programming decoder
│   └── visualization/
│       └── map_generator.py                  # Interactive Leaflet / Folium map builder
├── scripts/
│   ├── build_features.py                     # Generates shared processed feature caches
│   ├── train_random_forest.py                # Teammate 2: 5-Fold GroupKFold CV & model training
│   ├── train_gradient_boosting.py            # Teammate 3: 5-Fold GroupKFold CV & model training
│   ├── reconstruct_route_rf.py               # Evaluates RF-only vs. RF + HMM reconstruction
│   ├── evaluate_calibration_rf.py            # Evaluates probability calibration & Brier score
│   ├── bayesian_tune_rf.py                   # Bayesian hyperparameter optimization
│   ├── compare_models.py                     # Generates side-by-side benchmark table
│   └── generate_report_visuals.py            # Generates figures for the technical report
├── api/
│   └── main.py                               # FastAPI backend service (Laptop 1)
├── ui/
│   └── index.html                            # Evaluator user interface (Laptop 2)
├── models/
│   ├── random_forest.joblib                  # Trained Random Forest model weights
│   └── gradient_boosting.joblib              # Trained Gradient Boosting model weights
└── reports/
    ├── random_forest_summary.json            # Teammate 2 evaluation metrics
    ├── gradient_boosting_summary.json        # Teammate 3 evaluation metrics
    ├── teammate_model_comparison.csv         # Side-by-side comparison table
    ├── calibration_analysis_rf.json          # Brier score & reliability curve coordinates
    ├── bayesian_trials_rf.csv                # Bayesian optimization search history
    ├── feature_importances_rf.png            # Feature importance horizontal bar chart
    ├── model_comparison_rf_vs_gb.png         # Model comparison bar chart
    └── map_trip_00000007.html                # Interactive map for trip 00000007
```

---

## 3. Quickstart & Execution Guide

### Step 1: Shared Feature Extraction
To re-generate the shared feature matrices across all 80 training trips and 10 testing trips:
```powershell
python scripts/build_features.py --partition train
python scripts/build_features.py --partition student_test
```

### Step 2: Train Teammate 2's Random Forest Model
Runs Group-Aware 5-Fold Cross-Validation on the 80 training trips and evaluates on unseen student-test trips:
```powershell
python scripts/train_random_forest.py
```

### Step 3: Probability Calibration Evaluation
Evaluates reliability curves, Brier scores, and Expected Calibration Error (ECE):
```powershell
python scripts/evaluate_calibration_rf.py
```

### Step 4: Full Route Reconstruction (RF vs. RF + HMM)
Runs sequential Viterbi decoding on any test trip and evaluates route continuity:
```powershell
python scripts/reconstruct_route_rf.py --trip-id 00000007
```

### Step 5: Generate Comparison Tables & Report Figures
```powershell
python scripts/compare_models.py
python scripts/generate_report_visuals.py
```

---

## 4. Two-Laptop Evaluation Architecture

- **Laptop 1 (Model & FastAPI Backend)**:
  ```powershell
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
  ```
- **Laptop 2 (Evaluator UI Client)**:
  Open `ui/index.html` in any web browser on Laptop 2, enter Laptop 1's LAN IP address, select an evaluation trip, and click **Run Reconstruction**. Ground-truth labels remain strictly server-side on Laptop 1.
