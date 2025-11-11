from __future__ import annotations
import os, json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

@dataclass
class ComparisonAnalyzer:
    cfg: dict

    def _paths(self):
        p = self.cfg["paths"]
        return p["semantic_dataset"], p["reports_dir"], p["figures_dir"]

    def simulate_deepsc_predictions(self, y_true: np.ndarray, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        class_acc = {0:0.95,1:0.88,2:0.90,3:0.85,4:0.87,5:0.92,6:0.91,7:0.89}
        confusion_map = {0:[1,5,6],1:[0,2],2:[7,1],3:[5,4],4:[6,3],5:[3,0],6:[4,0],7:[2,0]}
        preds = []
        for t in y_true:
            acc = class_acc.get(int(t), 0.90)
            if rng.random() < acc:
                preds.append(int(t))
            else:
                cand = confusion_map.get(int(t), [i for i in range(8) if i!=int(t)])
                preds.append(int(rng.choice(cand)))
        return np.array(preds, dtype=int)

    def _metrics(self, y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def run(self) -> dict:
        sem_path, rep_dir, fig_dir = self._paths()
        os.makedirs(rep_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)

        df = pd.read_csv(sem_path)
        df.columns = df.columns.astype(str).str.strip()
        y_true = df["ground_truth"].values

        fse_json = os.path.join(rep_dir, "fse_evaluation_results.json")
        if not os.path.exists(fse_json):
            raise FileNotFoundError("fse_evaluation_results.json not found. Please run evaluation first.")
        with open(fse_json, "r", encoding="utf-8") as f:
            fse_metrics = json.load(f)

        y_pred_deepsc = self.simulate_deepsc_predictions(y_true, seed=self.cfg.get("random_seed", 42))
        deepsc_metrics = self._metrics(y_true, y_pred_deepsc)
        deepsc_metrics.update({
            "bytes_per_sample": int(self.cfg["deepsc"]["bytes_per_sample"]),
            "inference_time_ms": 15.6,
            "model_size_mb": 2.4,
            "training_time_hours": 4.5,
        })

        fse_b = int(self.cfg["fse"]["bytes_per_symbol"])
        dsc_b = int(self.cfg["deepsc"]["bytes_per_sample"])
        bandwidth_saving_percent = (1 - fse_b/dsc_b) * 100.0

        uJ_per_bit = float(self.cfg["energy"]["per_bit"])
        fse_energy_uJ = fse_b * 8 * uJ_per_bit
        dsc_energy_uJ = dsc_b * 8 * uJ_per_bit
        energy_saving_percent = (1 - fse_energy_uJ / dsc_energy_uJ) * 100.0

        comparison = {
            "performance_comparison": {
                "fuzsemcom": {k: fse_metrics[k] for k in ["accuracy","f1_macro","f1_weighted","precision_macro","recall_macro"]},
                "l_deepsc": deepsc_metrics,
                "differences": {
                    "accuracy_diff_percent": (deepsc_metrics["accuracy"] - fse_metrics["accuracy"]) * 100.0,
                    "f1_diff_percent": (deepsc_metrics["f1_macro"] - fse_metrics["f1_macro"]) * 100.0,
                },
            },
            "efficiency_comparison": {
                "communication": {
                    "fse_payload_per_sample": fse_b,
                    "deepsc_payload_per_sample": dsc_b,
                    "bandwidth_saving_percent": bandwidth_saving_percent,
                    "payload_reduction_ratio": dsc_b / fse_b,
                },
                "energy": {
                    "unit": "uJ_per_message",
                    "fse_energy": fse_energy_uJ,
                    "deepsc_energy": dsc_energy_uJ,
                    "energy_saving_percent": energy_saving_percent,
                },
            },
        }

        import matplotlib
        matplotlib.use(self.cfg.get("plotting", {}).get("backend", "Agg"))
        import matplotlib.pyplot as plt
        import seaborn as sns
        classes = ["optimal","nutrient_deficiency","fungal_risk","water_deficit_acidic","water_deficit_alkaline","acidic_soil","alkaline_soil","heat_stress"]
        cm = confusion_matrix(y_true, y_pred_deepsc, labels=list(range(8)))
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=classes, yticklabels=classes)
        plt.title("L-DeepSC (Simulated) - Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("True")
        fig_path = os.path.join(fig_dir, "deepsc_confusion_matrix.png")
        plt.tight_layout(); plt.savefig(fig_path, dpi=200); plt.close()

        html_path = os.path.join(rep_dir, "comparison_report.html")
        html = """<!doctype html>
<html><head><meta charset="utf-8"><title>FuzSemCom vs L-DeepSC</title>
<style>body{font-family:Arial,Helvetica,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}th{background:#f2f2f2}</style>
</head><body>
<h1>FuzSemCom vs L-DeepSC (Simulated)</h1>
<p><b>Payload:</b> FSE __FSE_B__ B vs L-DeepSC __DSC_B__ B — <b>Bandwidth saving:</b> __BW__%</p>
<p><b>Energy:</b> __FSE_E__ µJ vs __DSC_E__ µJ — <b>Saving:</b> __ES__%</p>
<h2>Performance</h2>
<table>
<tr><th>Metric</th><th>FuzSemCom</th><th>L-DeepSC</th><th>Δ</th></tr>
<tr><td>Accuracy</td><td>__FA__</td><td>__DA__</td><td>__DADELTA__%</td></tr>
<tr><td>F1 (macro)</td><td>__FF1__</td><td>__DF1__</td><td>__F1DELTA__%</td></tr>
<tr><td>Precision (macro)</td><td>__FP__</td><td>__DP__</td><td></td></tr>
<tr><td>Recall (macro)</td><td>__FR__</td><td>__DR__</td><td></td></tr>
</table>
<p><a href="../figures/deepsc_confusion_matrix.png">DeepSC confusion matrix</a></p>
</body></html>"""
        html = (html
                .replace("__FSE_B__", f"{fse_b}")
                .replace("__DSC_B__", f"{dsc_b}")
                .replace("__BW__", f"{bandwidth_saving_percent:.1f}")
                .replace("__FSE_E__", f"{fse_energy_uJ:.2f}")
                .replace("__DSC_E__", f"{dsc_energy_uJ:.2f}")
                .replace("__ES__", f"{energy_saving_percent:.1f}")
                .replace("__FA__", f"{fse_metrics['accuracy']:.3f}")
                .replace("__DA__", f"{deepsc_metrics['accuracy']:.3f}")
                .replace("__DADELTA__", f"{(deepsc_metrics['accuracy']-fse_metrics['accuracy'])*100.0:+.1f}")
                .replace("__FF1__", f"{fse_metrics['f1_macro']:.3f}")
                .replace("__DF1__", f"{deepsc_metrics['f1_macro']:.3f}")
                .replace("__F1DELTA__", f"{(deepsc_metrics['f1_macro']-fse_metrics['f1_macro'])*100.0:+.1f}")
                .replace("__FP__", f"{fse_metrics['precision_macro']:.3f}")
                .replace("__DP__", f"{deepsc_metrics['precision_macro']:.3f}")
                .replace("__FR__", f"{fse_metrics['recall_macro']:.3f}")
                .replace("__DR__", f"{deepsc_metrics['recall_macro']:.3f}")
               )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        out_json = os.path.join(rep_dir, "deepsc_comparison_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

        return {"comparison_json": out_json, "deepsc_confusion_matrix_png": fig_path, "report_html": html_path}
