# AI-Based Predictive Maintenance System

An academic prototype that explores failure prediction, interpretable condition assessment, and maintenance scheduling for industrial machines. The project compares artificial neural networks (ANN) with adaptive neuro-fuzzy inference (ANFIS), then uses a Mamdani fuzzy inference system and a genetic algorithm (GA) to turn machine condition estimates into a maintenance plan.

**This is a research prototype, not a validated industrial decision system.** The saved metrics describe this repository's dataset and evaluation procedures. The repository currently does not record the raw dataset's external source or license; do not attribute it to a public benchmark without confirming that provenance.

## Project submission materials

- [Project report (Markdown)](docs/project_report.md)
- [Project report (Word)](docs/project_report.docx)
- [Project presentation (PowerPoint)](docs/Predictive_Maintenance_Project_Presentation.pptx)
- [Verification summary](docs/verification_summary.md)
- [Modification log](docs/modification_log.md)

The report covers the project background, literature, system design, implementation, saved results, limitations, and references. The presentation uses the supplied dashboard screenshots and the repository's saved dataset, model, and scheduling results. Figures are identified as prototype evidence; no new model training was performed.

## System overview

```text
Machine records
      │
      ├── preprocessing: validation, chronological 80:20 split, missing-value handling in model pipelines
      │
      ├── ANN failure probability ── Mamdani fuzzy urgency ── Genetic-algorithm job order
      │
      └── ANFIS model comparison (separate evaluation path)
```

ANN and ANFIS are evaluated as alternative prediction approaches. The saved dashboard/scheduler workflow uses the ANN pipeline for machine probabilities, the Mamdani FIS for urgency, and the GA for scheduling. ANFIS is a model comparison; it is not currently the prediction model consumed by the GA.

## Repository layout

```text
Predictive_Maintenance/
├── dashboard/app.py             Streamlit dashboard and saved-result views
├── dataset/
│   ├── raw/predictive_maintenance_v3.csv
│   └── processed/train.csv, test.csv
├── models/                       Saved ANN and ANFIS estimators
├── results/                      Existing metrics, predictions, and schedules
├── src/
│   ├── data_preprocessing.py     Validation and chronological split
│   ├── ann_model.py              Primary single-split ANN
│   ├── machine_independent_ann.py Group-aware five-fold ANN evaluation
│   ├── ann_optimization.py       Four group-aware ANN configurations
│   ├── ann_fair_comparison.py    Four-input ANN comparison
│   ├── anfis_model.py            ANFIS without oversampling
│   ├── anfis_balanced.py         ANFIS with training-only oversampling
│   ├── fuzzy_logic.py            Mamdani fuzzy assessment
│   ├── genetic_algorithm.py      Capacity-constrained job ordering
│   └── *_threshold_analysis*.py Exploratory probability-threshold sweeps
├── run_all_tests.py              Existing verification suite
└── requirements.txt              Python dependencies
```

## Data and preprocessing

The checked-in CSV contains **24,042 records from 20 machine IDs**. The target is `failure_within_24h` (0/1). The saved preprocessing summary reports 20,482 negative and 3,560 positive records (14.81% positives overall). The chronological 80:20 split contains 19,233 training records and 4,809 test records; the test partition has 300 positive records (6.24%), so the class balance differs substantially between partitions.

The nine model inputs are seven numeric sensor/usage variables—vibration RMS, motor temperature, average phase current, pressure, RPM, hours since maintenance, and ambient temperature—and two categorical variables, machine type and operating mode. `failure_type`, repair cost, and remaining useful life are excluded from the ANN feature list. Numeric imputation and scaling, categorical imputation and one-hot encoding, and training-only random oversampling are placed in model pipelines where applicable.

The repository documents no external dataset citation, acquisition procedure, or license. Add and verify these before describing the data as a named public benchmark or redistributing it.

## Setup and execution

Use a Python environment, then install the listed dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run scripts from the project root. Training scripts write to the existing `models/` and `results/` filenames, so **do not rerun them if you need to preserve the checked-in artifacts**. Back up the project or change output paths on a separate experiment copy before generating new model files.

```bash
python src/data_preprocessing.py
python src/ann_model.py
python src/machine_independent_ann.py
python src/ann_optimization.py
python src/ann_fair_comparison.py
python src/anfis_model.py
python src/anfis_balanced.py
python src/fuzzy_logic.py
python src/genetic_algorithm.py
python run_all_tests.py
streamlit run dashboard/app.py
```

The scripts are independent experiments after preprocessing; the list is a guide, not a single required run sequence. The GA script expects the saved `models/ann_pipeline.joblib`. The dashboard reads saved models and files from `models/` and `results/`.

## Evaluation protocol and saved results

Use the **machine-independent ANN** as the primary predictive result when discussing unseen-machine generalization. It uses five-fold `StratifiedGroupKFold` keyed by machine ID, so a machine is kept wholly within either the training or validation side of a fold. Its pooled out-of-fold results are:

| Metric | Score |
|---|---:|
| Accuracy | 89.77% |
| Precision | 61.71% |
| Recall | 81.43% |
| F1-score | 70.21% |
| ROC-AUC | 0.9428 |
| PR-AUC | 0.7778 |

The pooled confusion matrix is TN 18,683; FP 1,799; FN 661; TP 2,899. These out-of-fold results address machine overlap, but the group folds are not a prospective future-time holdout. They do not establish generalization to future operating periods, new factories, or field deployment.

The primary ANN's chronological split metrics (97.57% accuracy, 76.99% precision, 87.00% recall, 0.9894 ROC-AUC) are **not the recommended headline result**: the same machine IDs occur in both partitions. The README and dashboard retain these results as a comparison while explicitly flagging this limitation.

The saved four-feature, chronological-test comparison shows a different sensitivity/false-alarm trade-off:

| Model | Precision | Recall | F1 | ROC-AUC | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| Four-feature ANN | 38.07% | 72.33% | 49.89% | 0.9240 | 353 | 83 |
| Original ANFIS | 66.81% | 53.00% | 59.11% | 0.9390 | 79 | 141 |
| Balanced ANFIS | 40.47% | 92.67% | 56.33% | 0.9565 | 409 | 22 |

Balanced ANFIS detects more positive test cases but creates substantially more false alarms. The appropriate operating point depends on maintenance capacity and the real costs of missed failures and unnecessary interventions. Threshold CSVs are exploratory sweeps on out-of-fold probabilities; selecting and reporting a threshold on the same OOF predictions is not independent confirmation.

For scheduling, the saved GA objective values are **dimensionless prototype costs**, not currency:

| Strategy | Objective cost | Improvement vs FIFO |
|---|---:|---:|
| FIFO | 4.7758 | — |
| Priority order | 3.1505 | 34.0% |
| Genetic algorithm | 3.0284 | 36.6% |

The GA improves the saved objective by 3.88% over priority ordering. Capacity (four machines per day) and the machine-type changeover penalty (0.03) are assumptions, not factory-calibrated constraints or costs.

## Dashboard

Launch with `streamlit run dashboard/app.py`. The dashboard provides dataset analysis, model metric comparisons, interactive Mamdani FIS assessment, machine monitoring, a saved maintenance schedule, and an experimental-results/limitations view. It reads the local saved CSV, JSON, and model artifacts; it is not a connected live sensor service.

## Limitations and next research steps

- Establish and cite the dataset's provenance, collection process, and redistribution rights.
- Validate with a truly future-time holdout and external machines or plants; account for temporal drift.
- Report confidence intervals and per-machine/fold variation, and use nested validation for threshold or model selection.
- Calibrate probabilities and decision thresholds against intervention costs, false alarms, and missed failures.
- Validate fuzzy membership boundaries and rules with maintenance engineers.
- Replace assumed GA objectives with measured labor, downtime, parts, deadlines, skill, shift, and production constraints; compare across repeated seeds and additional baselines.
- Conduct a documented literature search and prospective field evaluation before making deployment or savings claims.

## Branch policy

- `main` preserves the original project state.
- `testing-audit` is the isolated branch for audit documentation and approved experiments.
- Do not change ANN architecture, fuzzy rules, GA settings, saved model files, or prior experiment results without the required approval and a separate experiment plan.


