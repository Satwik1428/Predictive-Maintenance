# AI-Based Predictive Maintenance System

An academic prototype for machine failure prediction and maintenance planning. The project combines artificial neural network (ANN) and adaptive neuro-fuzzy inference system (ANFIS) comparisons with a Mamdani fuzzy condition assessment and a genetic-algorithm (GA) maintenance schedule.

**Project authors:** Y Santhosh Satwik (24BCE8605) and Richard issac (24BCE8536)

> **Branch and evidence note:** `main` contains the baseline repository. The report, dashboard screenshots, verification summary, and saved evaluation discussed below were prepared from the separate `testing-audit` branch. The documentation links therefore point to that branch. The audit commit does not change model files, source code, or existing experiment outputs on `main`.

## Project materials

- [Project report (Markdown)](https://github.com/Satwik1428/Predictive-Maintenance/blob/testing-audit/docs/project_report.md)
- [Project report (Word)](https://github.com/Satwik1428/Predictive-Maintenance/blob/testing-audit/docs/project_report.docx)
- [Project presentation (PowerPoint)](https://github.com/Satwik1428/Predictive-Maintenance/blob/testing-audit/docs/Predictive_Maintenance_Project_Presentation.pptx)
- [Verification summary](https://github.com/Satwik1428/Predictive-Maintenance/blob/testing-audit/docs/verification_summary.md)
- [Modification log](https://github.com/Satwik1428/Predictive-Maintenance/blob/testing-audit/docs/modification_log.md)

## Approach

```text
Sensor and machine history
        ├── ANN or ANFIS failure-probability evaluation
        └── ANN probability + selected condition inputs
                    ↓
             Mamdani fuzzy urgency
                    ↓
          Genetic-algorithm job schedule
```

ANN and ANFIS are alternative failure-prediction approaches. In the audited dashboard workflow, the ANN probability feeds the Mamdani assessment and the GA schedule; ANFIS is a separate comparison model. The fuzzy layer reports urgency, health, severity, and a maintenance recommendation. The GA orders jobs under an assumed daily capacity and machine-type changeover penalty.

## Dataset and saved evaluation

The audited branch's checked-in CSV contains 24,042 rows, 20 machine IDs, and 15 columns. Its target is failure within 24 hours; 3,560 records are positive (14.81%). The chronological train/test split contains 19,233 and 4,809 rows, respectively. Test prevalence is 6.24%.

The primary ANN result in the report is the saved five-fold machine-grouped evaluation: 89.77% accuracy, 61.71% precision, 81.43% recall, 70.21% F1, 0.9428 ROC-AUC, and 0.7778 PR-AUC. Grouping prevents the same machine from appearing on both sides of a fold. It does not test future-time or external-plant generalization.

In the saved four-input chronological comparison, balanced ANFIS reaches 92.67% recall (278 of 300 positive test records) with 409 false alarms and 22 missed failures. Its operating trade-off should be evaluated against real maintenance costs and capacity. The report explains why the higher 97.57% accuracy from the machine-overlapping ANN split is not treated as the primary generalization result.

The GA's saved objective is a dimensionless prototype cost: 4.7758 for FIFO, 3.1505 for priority ordering, and 3.0284 for the GA. The daily capacity and changeover penalty are assumptions, not measured factory costs.

## Dashboard and local setup

The Streamlit dashboard in the audited branch presents dataset analysis, model comparisons, saved machine assessments, interactive fuzzy inference, sensor history, and the optimized maintenance schedule. It reads project files locally and is not a live sensor service.

Create a Python environment, install dependencies, and run from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

Training scripts write to existing model and results paths. To preserve saved artifacts, run experiments only on a separate project copy with separate output paths.

## Limitations

- The dataset source, acquisition protocol, and license are not documented in the repository.
- The machine-grouped evaluation holds out machine IDs but does not hold out later time periods.
- Saved results are dataset-specific and are not evidence of field performance or financial savings.
- Fuzzy rules, score boundaries, probability thresholds, and GA cost assumptions require validation with maintenance practitioners.
- The prototype requires future-time, external-machine, and prospective validation before operational use.

See the linked report for implementation details, literature context, references, figures, and the complete results analysis.


