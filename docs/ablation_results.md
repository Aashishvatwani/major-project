# 🔬 Satellite Health Management — Research Ablation Study Results

| Experiment | Architecture Configuration | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) | PR-AUC (%) | False Alarm Rate (%) | Latency (ms) | Safety Compliance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | XGBoost Only | 100.00% | 99.86% | **99.93%** | 99.99% | 99.95% | 0.000% | 0.00 ms | **100.0%** |
| **Exp B** | Random Forest + XGBoost | 100.00% | 99.86% | **99.93%** | 100.00% | 99.98% | 0.000% | 0.00 ms | **100.0%** |
| **Exp C** | RF + XGB + Extra Trees Ensemble | 100.00% | 99.86% | **99.93%** | 100.00% | 99.98% | 0.000% | 0.00 ms | **100.0%** |
| **Exp D** | Ensemble + Safety Engine | 98.76% | 99.86% | **99.31%** | 100.00% | 99.98% | 0.192% | 0.08 ms | **100.0%** |
| **Exp E** | Ensemble + Safety + RL Mitigation | 98.76% | 99.58% | **99.17%** | 100.00% | 99.98% | 0.192% | 0.23 ms | **100.0%** |
| **Exp F** | FULL SYSTEM (+ Digital Twin + RAG) | 98.76% | 99.58% | **99.17%** | 100.00% | 99.98% | 0.192% | 12.71 ms | **100.0%** |
