# On MICRO vs MACRO: https://stephenallwright.com/micro-vs-macro-f1-score/
from typing import List, Dict, Any
from collections import defaultdict
import glob, os, json, statistics
import pandas as pd

from utils.classes import IntaviaDocument
from utils_general import get_gold_annotations, INTAVIA_JSON_ROOT


def evaluate_bionet_intavia(nlp_systems: List[str], valid_labels: List[str], eval_type: str):
    intavia_files_root = f"{INTAVIA_JSON_ROOT}/*"
    gold_docs = get_gold_annotations()
    gold_method = "human_gold"

    # Here we must know which system_keys are present in the NLP JSONs
    sys_general_dict = {}
    for sys in nlp_systems:
        sys_general_dict[sys] = {"TOTAL (micro)": {"TP": 0, "FP": 0, "FN": 0}}
        for l in valid_labels:
            sys_general_dict[sys][l] = {"TP": 0, "FP": 0, "FN": 0}

    # For Corpus Micro Eval
    for src_path in glob.glob(intavia_files_root):
        for bio_path in glob.glob(f"{src_path}/*"):
            bio_id = os.path.basename(bio_path).strip(".json")
            if bio_id in gold_docs:
                # Predictions INFO
                bio = IntaviaDocument(json.load(open(bio_path)))
                for sys_name in nlp_systems:
                    eval_dict = bio.evaluate_ner(gold_method, sys_name, eval_type, valid_labels, ignore_text_after_gold=False)
                    if eval_dict:
                        per_label = eval_dict["metrics"]
                        for lbl, metrics in per_label.items():
                            if lbl not in ["MICRO", "MACRO"]:
                                sys_general_dict[sys_name][lbl]["TP"] += metrics["TP"]
                                sys_general_dict[sys_name][lbl]["FP"] += metrics["FP"]
                                sys_general_dict[sys_name][lbl]["FN"] += metrics["FN"]
    # For Corpus-Level Eval
    for sys_name, metrics in sys_general_dict.items():
        macro_p, macro_r, macro_f1 = [], [], []
        corpus_tp, corpus_fp, corpus_fn = 0, 0, 0
        evaluated_labels = []
        for lbl, m in metrics.items():
            if lbl != "TOTAL (micro)":
                prec = 0 if m["TP"]+m["FP"] == 0 else 100*m["TP"]/(m["TP"]+m["FP"])
                rec = 0 if m["TP"]+m["FN"] == 0 else 100*m["TP"]/(m["TP"]+m["FN"])
                f1 = 0 if prec+rec == 0 else 2*(prec*rec)/(prec+rec)
                m["P"] = round(prec, 2) 
                m["R"] = round(rec, 2) 
                m["F1"] = round(f1, 2) 
                corpus_tp += m["TP"] 
                corpus_fp += m["FP"] 
                corpus_fn += m["FN"]
                macro_p.append(prec)
                macro_r.append(rec)
                macro_f1.append(f1)
                evaluated_labels.append(lbl)
        # Macro Eval
        metrics["TOTAL (macro)"] = {"TP": corpus_tp, "FP": corpus_fp, "FN": corpus_fn}
        print(f"------ {sys_name} --------\n{evaluated_labels}\n{macro_p}\n{macro_r}\n{macro_f1}")
        metrics["TOTAL (macro)"]["P"] = round(statistics.mean(macro_p), 2)
        metrics["TOTAL (macro)"]["R"] = round(statistics.mean(macro_r), 2)
        metrics["TOTAL (macro)"]["F1"] = round(statistics.mean(macro_f1), 2)
        # Micro Eval
        micro_prec = 0 if corpus_tp+corpus_fp == 0 else 100*corpus_tp/(corpus_tp+corpus_fp)
        micro_rec = 0 if corpus_tp+corpus_fn == 0 else 100*corpus_tp/(corpus_tp+corpus_fn)
        micro_f1 = 0 if micro_prec+micro_rec == 0 else 2*(micro_prec*micro_rec)/(micro_prec+micro_rec)
        metrics["TOTAL (micro)"]["P"] = round(micro_prec, 2)
        metrics["TOTAL (micro)"]["R"] = round(micro_rec, 2)
        metrics["TOTAL (micro)"]["F1"] = round(micro_f1, 2)

     # Eval can be dumped as JSON
    # json.dump(sys_general_dict, open("local_outputs/sys_general_dict.json", "w"), indent=2)
    # Make a Human Readable Table for sys_general_dict
    final_eval_table = []
    for sys_name, category_dict in sys_general_dict.items():
        for cat, metrics_dict in category_dict.items():
            row = {"System Name": sys_name, "Category": cat}
            for metric_name, metric_val in metrics_dict.items():
                row[metric_name] = metric_val
            final_eval_table.append(row)
    pd.DataFrame.from_dict(final_eval_table).to_csv(f"local_outputs/BiographyNet_Systems_Eval_Final_{eval_type}.csv")


def system_label_report(systems_metrics: Dict[str, Any]) -> List[List[Any]]:
    report_table = defaultdict(list)
    for sys_vs_sys_names, obj in systems_metrics.items():
        sys_name = sys_vs_sys_names.split("_vs_")[1]
        label_metrics = obj["metrics"]
        for label_name, metrics in label_metrics.items():
            if label_name not in ["MICRO", "MACRO"]:
                report_table[label_name].append({
                    "M": sys_name,
                    "P": metrics["P"],
                    "R": metrics["R"],
                    "F1": metrics["F1"]
                })
    return report_table


if __name__ == "__main__":
    systems = ["stanza_nl", "human_gold", "flair/ner-dutch-large_0.12.2", "gpt-3.5-turbo"]
    valid_labels = ["PER", "LOC", "ORG"]
    evaluate_bionet_intavia(systems, valid_labels, eval_type="full_match")
    evaluate_bionet_intavia(systems, valid_labels, eval_type="bag_of_entities")