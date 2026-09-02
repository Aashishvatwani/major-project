# 🔬 Satellite Health Management — Research Ablation Study Results

| Experiment | Architecture Configuration | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | PR-AUC (%) | False Alarm Rate (%) | Latency (ms) | Safety Compliance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | XGBoost Only | 100.00% | 100.00% | **100.00%** | 100.00% | 100.00% | 0.000% | 0.00 ms | **100.0%** |
| **Exp B** | Random Forest + XGBoost | 100.00% | 100.00% | **100.00%** | 100.00% | 100.00% | 0.000% | 0.00 ms | **100.0%** |
| **Exp C** | RF + XGB + Extra Trees Ensemble | 100.00% | 100.00% | **100.00%** | 100.00% | 100.00% | 0.000% | 0.00 ms | **100.0%** |
| **Exp D** | Ensemble + Safety Engine | 98.90% | 100.00% | **99.45%** | 100.00% | 100.00% | 0.028% | 0.06 ms | **100.0%** |
| **Exp E** | Ensemble + Safety + RL Mitigation | 98.90% | 100.00% | **99.45%** | 100.00% | 100.00% | 0.028% | 0.26 ms | **100.0%** |
| **Exp F** | FULL SYSTEM (+ Digital Twin + RAG) | 98.90% | 100.00% | **99.45%** | 100.00% | 100.00% | 0.028% | 2.05 ms | **100.0%** |
