# Title Page

# AI-Based Predictive Maintenance System

**Project Report**

**Prepared by**  
Y Santhosh Satwik — 24BCE8605  
Richard issac — 24BCE8536

**Project area:** Soft computing for machine failure prediction and maintenance planning  
**Repository:** [Satwik1428/Predictive-Maintenance](https://github.com/Satwik1428/Predictive-Maintenance)  
**Prepared:** 10 November 2026

**Project status:** Academic prototype. Results are specific to the saved data and evaluation protocols; they are not field validated.

---

## Abstract

This report describes an academic predictive-maintenance prototype that uses machine sensor and operating records to estimate failure risk and prioritize maintenance. The project compares an artificial neural network (ANN) with adaptive neuro-fuzzy inference systems (ANFIS). A separate Mamdani fuzzy inference system converts selected condition inputs and ANN failure probability into urgency, health, severity, and a recommendation. A genetic algorithm (GA) then orders machine jobs under an assumed daily capacity and machine-type changeover penalty. A Streamlit dashboard displays the dataset, saved evaluation results, machine assessments, and maintenance schedule.

The repository dataset contains 24,042 records from 20 machines, with 3,560 positive failure-within-24-hour labels (14.81%). The strongest saved machine-independent ANN evaluation is five-fold group-aware cross-validation: 89.77% accuracy, 61.71% precision, 81.43% recall, 70.21% F1, 0.9428 ROC-AUC, and 0.7778 PR-AUC. It holds out machine identities but does not hold out future time periods. In a separate chronological test comparison containing 300 positive cases, balanced ANFIS obtains 92.67% recall, detecting 278 failures, with 409 false alarms and 22 missed failures. The GA lowers its saved prototype objective by 3.88% relative to priority ordering. Dataset provenance, external validity, fuzzy rules, and scheduling costs require further verification. No industrial performance or financial-savings claim is supported by these results.

## Table of Contents

- [Introduction](#introduction)
- [Literature Review](#literature-review)
- [Motivation](#motivation)
- [Proposed Work](#proposed-work)
- [Methodology](#methodology)
- [Model / Architecture / Design](#model--architecture--design)
- [Implementation](#implementation)
- [Results and Analysis](#results-and-analysis)
- [Comparison with Existing Methods](#comparison-with-existing-methods)
- [Conclusion](#conclusion)
- [Scope for Future Work](#scope-for-future-work)
- [References](#references)

## Introduction

Industrial equipment can fail between scheduled inspections. Reactive maintenance may cause unplanned downtime, while fixed preventive schedules can service equipment before it needs attention. Predictive maintenance uses operational measurements and machine history to support decisions about inspection and service timing.

This project addresses three connected tasks:

1. Estimate whether a record is associated with failure within the next 24 hours.
2. Convert risk and condition readings into an interpretable maintenance urgency and health score.
3. Schedule machine jobs under a limited daily service capacity.

The work is a software prototype evaluated on the checked-in dataset and saved experiment artifacts. It is not a real-time sensor system, a validated safety controller, or an autonomous work-order issuer. All displayed recommendations require human review before any operational use.

![Dashboard overview with record counts, machine count, health summary, and system flow.](assets/dashboard_overview.png)

*Figure 1. Project dashboard overview. The displayed average health value is a UI snapshot, not an evaluation metric.*

## Literature Review

Predictive-maintenance research connects condition monitoring, fault prediction, and maintenance decisions. The cited papers informed the methods and the limits emphasized in this report. Their results are not direct benchmarks for this repository because their datasets, tasks, and evaluation settings differ.

### Research papers used and lessons learned

1. **Susto et al. (2015), multiple-classifier predictive maintenance [1].** The paper combines classifiers with different prediction horizons and connects their outputs to cost-based maintenance decisions. **Lesson for this project:** prediction quality should be assessed using missed failures and false alarms as well as accuracy. We therefore report precision, recall, and confusion counts; plant-specific costs are still needed before setting a decision threshold.
2. **Guidotti, Pandolfo, and Pulina (2025), systematic review of supervised predictive maintenance [2].** Their review covers 216 peer-reviewed studies and identifies limited public data availability and cross-domain generalization as continuing concerns. **Lesson for this project:** document data provenance and evaluation design. The repository’s dataset source and license are not recorded, and the machine-grouped evaluation is distinguished from future-time validation.
3. **Jang (1993), ANFIS [3].** The paper presents a hybrid learning procedure that combines fuzzy-rule structure with input/output data. **Lesson for this project:** adaptive neuro-fuzzy models can be compared as a distinct prediction approach. The ANFIS experiments are kept separate from the ANN-to-FIS-to-GA dashboard path.
4. **Mamdani and Assilian (1975), linguistic fuzzy control [4].** The study converts an operator’s linguistic rules into a fuzzy control strategy. **Lesson for this project:** a rule-based layer can express urgency in terms that a person can inspect. The prototype uses a Mamdani FIS for this role, but its rule base and membership boundaries still need expert validation.
5. **Mota, Faria, and Ramos (2023), genetic-algorithm production and maintenance scheduling [5].** Their method considers maintenance together with production constraints and cost and overload objectives. **Lesson for this project:** scheduling quality depends on realistic operational constraints. The current GA demonstrates capacity and changeover terms only; its objective is simplified and is not calibrated to factory costs.

These sources provide a focused background, not a complete systematic literature review. A publication-ready review would require documented databases, search terms, inclusion criteria, and a structured comparison across additional studies.

## Motivation

A failure probability by itself does not state what action to take, and a list of high-risk machines does not account for limited maintenance capacity. The project was motivated by the need to demonstrate a connected decision-support workflow: estimate risk, explain condition urgency, and produce a feasible job order for a small machine set.

The prototype also makes evaluation trade-offs visible. In the saved comparison, balanced ANFIS detects more positive cases but generates more false alarms. This illustrates why maintenance teams need to consider the relative impact of a missed failure, an unnecessary inspection, and a constrained service schedule. The project does not assign monetary values to those outcomes.

## Proposed Work

The proposed work is an integrated demonstration with two alternative failure-prediction models and two downstream decision components:

- Train and evaluate ANN and ANFIS prediction approaches using the repository’s saved data and evaluation procedures.
- Use ANN risk with sensor and maintenance-history inputs in a Mamdani FIS to estimate urgency and health.
- Use a GA to order machine jobs under the prototype’s daily-capacity and changeover assumptions.
- Present dataset summaries, saved model metrics, machine histories, fuzzy assessments, and the saved schedule in a Streamlit dashboard.

In the currently audited workflow, ANN probability feeds the fuzzy assessment and GA scheduler. ANFIS is a separate comparison model; it is not the model consumed by the scheduler. The contribution is integration and transparent comparison in a prototype, not a new learning algorithm or a validated industrial policy.

## Methodology

### Dataset and preparation

The project uses `dataset/raw/predictive_maintenance_v3.csv` and processed training and test files. The raw data has 24,042 rows, 20 machine IDs, and 15 columns. The target is `failure_within_24h`: 20,482 records are negative and 3,560 are positive. The preprocessing pipeline validates required columns and target values, removes exact duplicates and invalid timestamps, sorts records by time, and makes an 80:20 chronological split. The saved partitions contain 19,233 training rows and 4,809 test rows. Positive prevalence is 16.95% in training and 6.24% in test.

The ANN uses seven numerical inputs—vibration RMS, motor temperature, average phase current, pressure, RPM, hours since maintenance, and ambient temperature—and two categorical inputs: machine type and operating mode. The project excludes failure type, estimated repair cost, and remaining useful life from these ANN predictors. Numeric imputation and scaling, categorical imputation and one-hot encoding, and training-only oversampling are applied in the relevant pipelines.

The repository does not identify the dataset creator, source URL, acquisition protocol, or license. This report therefore refers to it as the repository dataset and does not attribute it to a public benchmark.

![Dataset analysis view and example records.](assets/dataset_analysis.png)

*Figure 2. Dashboard view of dataset size, fields, and example sensor records.*

![Target distribution displayed by the dashboard.](assets/failure_distribution.png)

*Figure 3. Label distribution: 20,482 records without failure and 3,560 records with failure within 24 hours.*

### Prediction evaluation

The primary ANN is a scikit-learn MLP classifier with ReLU activations and hidden layers (32, 16, 8), trained with Adam. Its saved primary experiment uses a chronological split. Since the same machine IDs occur in both partitions, this result is reported as a comparison rather than as the main unseen-machine estimate.

The machine-independent ANN evaluation uses five-fold `StratifiedGroupKFold` grouped by `machine_id`. Each machine appears in either the training side or validation side of a fold, not both. Each fold fits its own preprocessing, sampling, and ANN pipeline; pooled out-of-fold probabilities produce the saved metrics. This addresses machine overlap but mixes time periods across folds. It is not a prospective future-time test. The group-aware methodology follows the purpose of keeping groups intact across folds [6].

The ANFIS experiments use four numerical features: motor temperature, vibration RMS, average phase current, and hours since maintenance. Each input has two Gaussian membership functions, resulting in 16 rule combinations. The implementation fits premise and consequent parameters using training data. The balanced variant applies random oversampling to the training data only; the held-out test data remains unchanged.

### Fuzzy assessment and scheduling

The Mamdani FIS takes motor temperature, vibration RMS, ANN failure probability, and hours since maintenance. It maps inputs to low/medium/high fuzzy memberships, evaluates its explicit IF–THEN rules, aggregates outputs, and uses centroid defuzzification to obtain urgency on a 0–100 scale. Health is defined by the implementation as 100 minus urgency; score bands determine severity and recommendation. The rules are software assumptions and have not been validated by maintenance experts.

The GA scheduler evaluates the latest record for each machine, calculates ANN probability and fuzzy urgency, then orders jobs under a capacity of four machines per day. Its saved configuration uses population 80, 150 generations, a 15% swap-mutation probability, four elites, and seed 42. The objective combines risk-weighted waiting position with a machine-type changeover penalty. It is compared with FIFO and descending-priority order. Objective values are dimensionless prototype costs, not money.

## Model / Architecture / Design

The system separates predictive models from the decision-support stages:

```text
Machine sensor and history records
                 │
       ┌─────────┴─────────┐
       │                   │
     ANN path          ANFIS path
       │              model comparison
       ▼
ANN failure probability
       │
       ▼
Mamdani fuzzy assessment
(urgency, health, severity)
       │
       ▼
Genetic-algorithm schedule
(capacity and changeover terms)
       │
       ▼
Streamlit dashboard and saved outputs
```

| Component | Inputs | Output and role |
|---|---|---|
| ANN | Seven numeric and two categorical variables | Failure-within-24-hour probability; main dashboard/scheduling prediction path |
| ANFIS | Temperature, vibration, current, maintenance hours | Alternative failure prediction; separate comparison experiment |
| Mamdani FIS | Temperature, vibration, ANN probability, maintenance hours | Urgency score, health score, severity, and recommendation |
| Genetic algorithm | Machine-level risk and urgency, machine type | Maintenance order under assumed capacity and changeover penalty |
| Streamlit dashboard | Dataset, saved models, metrics, and schedules | Interactive project views; local prototype rather than live telemetry |

The stages are loosely coupled through saved files and Python modules. The ANN and ANFIS comparisons use different validation protocols in some experiments; their metrics must therefore be interpreted with those protocol differences in mind.

## Implementation

The implementation is organized into `src/` for preprocessing, model experiments, fuzzy logic, and scheduling; `dashboard/app.py` for the Streamlit interface; `dataset/` for raw and processed data; `models/` for serialized estimators; and `results/` for saved metrics, predictions, and schedules. The dashboard displays dataset analysis, model comparisons, sensor trends, individual machine assessments, fuzzy logic output, and the optimized maintenance schedule.

A displayed fuzzy example produces urgency 79.37/100, health 20.63/100, HIGH severity, and the recommendation “Arrange maintenance soon.” This is an example of the implemented rule system, not a validated maintenance threshold.

![Interactive fuzzy machine assessment.](assets/fuzzy_assessment.png)

*Figure 4. Dashboard example of the Mamdani assessment and resulting recommendation.*

![Individual machine monitoring view.](assets/machine_monitoring.png)

*Figure 5. Selected machine assessment and sensor history in the dashboard.*

The existing verification suite was run with UTF-8 console output and passed all eight sections. It compiled 16 Python modules, checked fuzzy edge cases and GA critical-job ordering, loaded both ANFIS artifacts, validated the saved group-aware ANN and ANN-configuration results, checked six threshold result files, and received HTTP 200 from the headless dashboard health endpoint. No model training, preprocessing, or threshold experiment was run to prepare this report. Training scripts write to existing artifact paths; future experiments should use a separate copy and separate output paths.

## Results and Analysis

### Dataset and machine-level overview

The dataset contains 24,042 observations across 20 machines and four machine types. There are 3,560 positive records (14.81%) overall. The chronological test partition contains 300 positives (6.24% of its 4,809 records), compared with 16.95% positives in training. This prevalence difference affects precision and the number of alerts expected at a fixed threshold.

### Machine-independent ANN results

| Metric | Saved result |
|---|---:|
| Accuracy | 89.77% |
| Precision | 61.71% |
| Recall | 81.43% |
| F1-score | 70.21% |
| ROC-AUC | 0.9428 |
| PR-AUC | 0.7778 |
| True negatives | 18,683 |
| False positives | 1,799 |
| False negatives | 661 |
| True positives | 2,899 |

These are pooled out-of-fold results from the machine-grouped ANN evaluation. They are the primary ANN figures for discussing unseen machine IDs. They should not be described as future-time or external-plant performance.

![Dashboard display of the group-aware ANN metrics.](assets/grouped_ann_metrics.png)

*Figure 6. Machine-independent ANN evaluation shown in the dashboard.*

### ANN configuration comparison

| Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (32, 16, 8) | 89.77% | 61.71% | 81.43% | 70.21% | 0.9428 | 0.7778 |
| Deep (64, 32, 16) | 89.17% | 60.48% | 77.50% | 67.94% | 0.9260 | 0.6359 |
| Regularized (64, 32) | 89.38% | 60.82% | 79.52% | 68.92% | 0.9384 | 0.7583 |
| Large (128, 64, 32) | 89.63% | 62.29% | 75.87% | 68.41% | 0.9225 | 0.6861 |

The baseline configuration has the best recall, F1, ROC-AUC, and PR-AUC among the saved configurations. The larger network has slightly higher precision but lower recall. These four saved configurations are not an exhaustive search.

### Four-feature model comparison

The following models use four inputs and the same chronological test partition with 300 positive records. These results are not directly comparable to the group-aware ANN metrics above.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Four-feature ANN | 90.93% | 38.07% | 72.33% | 49.89% | 0.9240 | 353 | 83 |
| Original ANFIS | 95.43% | 66.81% | 53.00% | 59.11% | 0.9390 | 79 | 141 |
| Balanced ANFIS | 91.04% | 40.47% | 92.67% | 56.33% | 0.9565 | 409 | 22 |

Balanced ANFIS detects 278 of the 300 positive test records and misses 22, while producing 409 false alarms. The original ANFIS produces fewer false alarms but misses 141 positives. The model choice depends on the relative cost of missed failures and unnecessary interventions; those costs are not known here.

![Saved comparison of ANN and ANFIS results.](assets/model_comparison.png)

*Figure 7. Dashboard comparison of saved prediction metrics. The feature set and split used for each row should be considered when interpreting it.*

### Chronological ANN comparison result

The nine-feature ANN chronological split reports 97.57% accuracy, 76.99% precision, 87.00% recall, 81.69% F1, and 0.9894 ROC-AUC. Since machines occur on both sides of that split, this number is not the primary unseen-machine result. It is retained as a separate saved comparison only.

### Genetic-algorithm schedule

| Scheduling strategy | Saved objective |
|---|---:|
| FIFO | 4.7758 |
| Descending priority | 3.1505 |
| Genetic algorithm | 3.0284 |

The GA objective is 3.88% lower than priority ordering and 36.59% lower than FIFO for the saved run. This comparison is limited to the implemented synthetic objective, capacity of four machines per day, and changeover penalty. It does not measure money, downtime reduction, or factory savings.

![Saved GA progress curve.](assets/ga_progress.png)

*Figure 8. Best saved objective over 150 generations; the y-axis is a prototype cost, not currency.*

![Saved optimized maintenance schedule.](assets/maintenance_schedule.png)

*Figure 9. The schedule assigns 20 machines to five days at the assumed capacity of four machines per day.*

## Comparison with Existing Methods

This is a qualitative comparison with the cited literature, not a head-to-head experiment. Each paper uses its own data and objectives.

| Existing work | Method and focus | Relation to this project | Main difference / caution |
|---|---|---|---|
| Susto et al. [1] | Multiple classifiers and prediction horizons linked to cost-based maintenance decisions | Motivates comparing prediction trade-offs and relating false alarms/missed failures to maintenance decisions | The repository does not implement their classifier ensemble or cost model, and its data provenance is undocumented |
| Guidotti et al. [2] | Systematic review of supervised predictive-maintenance studies | Motivates explicit data and validation caveats | A project prototype is not comparable to a literature review; external generalization remains untested here |
| Jang [3] | Adaptive neuro-fuzzy inference with hybrid learning | Provides the ANFIS method foundation for the comparison experiments | The saved ANFIS model and four-feature dataset setup differ from Jang’s demonstration tasks |
| Mamdani and Assilian [4] | Linguistic fuzzy rules used to construct a controller | Motivates the interpretable urgency layer | Current memberships/rules are project-defined, not expert-elicited or validated |
| Mota et al. [5] | Genetic algorithm for joint production and maintenance scheduling under constraints | Motivates scheduling with capacity and changeover considerations | This project uses a simplified job-ordering objective and does not include production, energy, labor, or measured maintenance costs |

## Conclusion

The project demonstrates a complete software path from sensor records to model evaluation, fuzzy condition assessment, a capacity-limited schedule, and dashboard presentation. The machine-grouped ANN evaluation provides the most defensible saved evidence for prediction on machine IDs excluded from training. Balanced ANFIS demonstrates higher recall in a separate four-feature chronological test, with a substantial false-alarm burden. The GA improves the saved prototype objective relative to FIFO and priority order under its stated assumptions.

The results establish that the pipeline runs on the repository artifacts; they do not establish prospective performance, a universally better model, factory savings, or deployment readiness. Dataset provenance, future-time validation, domain validation of fuzzy rules, calibrated decision costs, and operational scheduling constraints remain unresolved.

## Scope for Future Work

1. Verify and document the dataset source, collection process, sensor definitions, calibration, and redistribution license.
2. Evaluate on a locked future-time holdout and on machines or plants absent from model development; report per-machine variation and uncertainty.
3. Use nested or independent validation for feature, model, and threshold selection; report calibration and threshold-specific confusion counts.
4. Estimate the real costs of missed failures, inspections, parts, downtime, and interventions before choosing an operating threshold.
5. Review membership functions, fuzzy rules, severity boundaries, and recommendations with maintenance engineers and safety stakeholders.
6. Extend scheduling to job duration, staff skills, shifts, parts availability, deadlines, production plans, and machine dependencies; repeat GA runs across seeds and compare richer baselines.
7. Conduct a prospective, human-supervised pilot before making operational or commercial claims.

## References

[1] G. A. Susto, A. Schirru, S. Pampuri, S. McLoone, and A. Beghi, “Machine Learning for Predictive Maintenance: A Multiple Classifiers Approach,” *IEEE Transactions on Industrial Informatics*, vol. 11, no. 3, pp. 812–820, 2015. https://doi.org/10.1109/TII.2014.2349359

[2] D. Guidotti, L. Pandolfo, and L. Pulina, “A Systematic Literature Review of Supervised Machine Learning Techniques for Predictive Maintenance in Industry 4.0,” *IEEE Access*, vol. 13, pp. 102479–102504, 2025. https://doi.org/10.1109/ACCESS.2025.3578686

[3] J.-S. R. Jang, “ANFIS: Adaptive-Network-Based Fuzzy Inference System,” *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 23, no. 3, pp. 665–685, 1993. https://doi.org/10.1109/21.256541

[4] E. H. Mamdani and S. Assilian, “An Experiment in Linguistic Synthesis with a Fuzzy Logic Controller,” *International Journal of Man-Machine Studies*, vol. 7, no. 1, pp. 1–13, 1975. https://doi.org/10.1016/S0020-7373(75)80002-2

[5] B. Mota, P. Faria, and C. Ramos, “Joint Production and Maintenance Scheduling for Total Cost and Machine Overload Reduction in Manufacturing: A Genetic Algorithm Approach,” *IEEE Access*, vol. 11, pp. 98070–98081, 2023. https://doi.org/10.1109/ACCESS.2023.3312557

[6] scikit-learn developers, “StratifiedGroupKFold,” *scikit-learn User Guide*. https://scikit-learn.org/stable/modules/cross_validation.html#stratifiedgroupkfold
