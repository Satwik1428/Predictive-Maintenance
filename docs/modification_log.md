# Modification Log

All modifications below were made on the existing `testing-audit` branch. This branch was clean before the documentation changes. The repository's `main` branch, source code, dataset, trained models, and existing experiment outputs were not modified.

## 2026-11-10 — Documentation audit

### Updated

- `README.md` — Rewrote the overview and setup guide to describe the actual ANN/ANFIS alternatives, Mamdani fuzzy assessment, GA scheduling flow, dashboard, data split, evaluation protocols, and saved results. Added a warning that training scripts write to existing artifact paths and should be run only on a separate experiment copy when preserving current models/results. Clarified that the GA consumes the ANN prediction plus FIS priority; ANFIS is currently a comparison model.

### Added

- `docs/project_report.md` — Prepared a report covering the abstract, problem definition, literature context and research gap, methodology, implementation, results and analysis, discussion, limitations, future scope, article/startup framing, references, and artifact map. Values are transcribed from saved results and code; no experiment was rerun.
- `docs/modification_log.md` — Recorded the scope and boundaries of this documentation change.

### Audit findings recorded in the documentation

- The repository does not state the source, license, or collection protocol for its raw dataset.
- The headline primary ANN chronological split overlaps machine IDs; machine-grouped OOF results are presented as the stronger unseen-machine evaluation, with the limitation that they do not hold out future time periods.
- The test class prevalence is 6.24%, compared with 16.95% in training.
- OOF threshold sweeps are exploratory when a threshold is selected and assessed on the same OOF predictions.
- GA objective values, capacity, and changeover penalty are assumptions; objective costs are not currency.

## Preservation and verification

- Existing files under `models/` and `results/` were treated as read-only during this documentation task.
- No ANN architecture, fuzzy membership/rule, or GA parameter was changed.
- The existing verification suite was run with UTF-8 output: all eight sections passed. The first invocation stopped at a Windows console encoding error before any project check; see `docs/verification_summary.md`.
- No preprocessing, model training, or threshold experiment was run.
- Changes were reviewed by reading the final documentation and comparing claims against checked-in source and metric files.

## 2026-11-10 — Report and presentation deliverables

- Added nine selected user-provided dashboard screenshots under `docs/assets/` for report evidence.
- Added `docs/project_report.docx`, formatted from the audited Markdown report with the supplied 20-mark rubric, saved-result tables, references, and UI figures.
- Added `docs/Predictive_Maintenance_Project_Presentation.pptx`, a 15-slide deck with UI, dataset, evaluation, fuzzy assessment, GA, and schedule evidence.
- Added `docs/verification_summary.md`. The existing verification suite passed all eight sections with UTF-8 console output; no training or threshold experiment was rerun.
- Source code, dataset, models, and saved experiment results remain unchanged.
