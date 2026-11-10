# Verification Summary

**Date:** 10 November 2026  
**Command:** `python run_all_tests.py`  
**Outcome:** 8 of 8 verification sections passed after running with UTF-8 console output.

## Checks completed

| Check | Result |
|---|---|
| Python compilation | 16 modules compiled |
| Mamdani fuzzy cases | 4 of 4 passed: normal conditions, high-temperature override, extreme vibration, and overdue maintenance |
| GA critical-job ordering | Passed with 1, 4, and 6 critical jobs; critical jobs occupied the first positions |
| ANFIS artifacts | Both saved models loaded; each had 16 rules and centers shaped `(4, 2)` |
| Group-aware ANN result | Five saved folds found; reported accuracy 89.77%, recall 81.43%, F1 70.21%, ROC-AUC 0.9428, PR-AUC 0.7778 |
| ANN configuration outputs | Four saved configurations read and summarized |
| Threshold analysis files | Six saved CSV files were present and non-empty |
| Dashboard health | Headless Streamlit endpoint returned HTTP 200 (`ok`) |

The first invocation stopped before the first project assertion because the Windows console's CP1252 encoding could not print the suite's checkmark character. Re-running with UTF-8 console output completed all eight sections. This was an output-encoding issue, not a model or test assertion failure.

The suite checks existing code and saved artifacts. It does not rerun model training or validate the project dataset's provenance, future-time generalization, fuzzy rule validity with plant experts, or real-world scheduling costs. No model or experiment result file was overwritten during verification.
