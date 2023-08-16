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
        if sys == "gysbert_hist_fx_finetuned_epoch2":
            for l in ["PER", "LOC"]:
                sys_general_dict[sys][l] = {"TP": 0, "FP": 0, "FN": 0}
        else:
            for l in valid_labels:
                sys_general_dict[sys][l] = {"TP": 0, "FP": 0, "FN": 0}

    # For Corpus Micro Eval
    entity_errors_table = []
    for src_path in glob.glob(intavia_files_root):
        for bio_path in glob.glob(f"{src_path}/*"):
            bio_id = os.path.basename(bio_path).strip(".json")
            if bio_id in gold_docs:
                # Predictions INFO
                bio = IntaviaDocument(json.load(open(bio_path)))
                for sys_name in nlp_systems:
                    if sys_name == "gysbert_hist_fx_finetuned_epoch2":
                        eval_dict = bio.evaluate_ner(gold_method, sys_name, eval_type, ["PER", "LOC"], ignore_text_after_gold=False)
                    else:
                        eval_dict = bio.evaluate_ner(gold_method, sys_name, eval_type, valid_labels, ignore_text_after_gold=False)
                    if eval_dict:
                        entity_errors_table.append({"text_id":bio.text_id, "method": sys_name, "errors": eval_dict["errors"]})
                        per_label = eval_dict["metrics"]
                        for lbl, metrics in per_label.items():
                            if lbl not in ["MICRO", "MACRO"]:
                                sys_general_dict[sys_name][lbl]["TP"] += metrics["TP"]
                                sys_general_dict[sys_name][lbl]["FP"] += metrics["FP"]
                                sys_general_dict[sys_name][lbl]["FN"] += metrics["FN"]
    save_entity_errors_table(entity_errors_table, eval_type)

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
    pd.DataFrame.from_dict(final_eval_table).to_csv(f"local_outputs/BiographyNet_Systems_Eval_Final_{eval_type}.csv", index=False)


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


def save_entity_errors_table(collected_errors: List[Dict[str, Any]], eval_type: str) -> List[Dict[str, Any]]:
    if eval_type in ["full_match"]:
        fp_fn_errors, sp_lb_errors = [], []
        for i, row in enumerate(collected_errors):
            text_id = row["text_id"]
            sys_name = row["method"]
            for fp in row["errors"]["Full Errors (not in Gold)"]: # FP_Exact_Match: ('3_17', 'Paul-Henri Spaak', 'PER', 'FP')
                #print("FP", fp)
                fp_fn_errors.append({"text_id": text_id, "method": sys_name, "type": "FP", "error_span": fp[0], "error_text": fp[1], "error_label": fp[2]})
            for fn in row["errors"]["Missed Entities"]: # FN_Exact_Match: (553, 'Buitenlandse Zaken van Belgie', 'ORG', 'FN')
                #print("FN", fn)
                fp_fn_errors.append({"text_id": text_id, "method": sys_name, "type": "FN", "error_span": fp[0], "error_text": fn[1], "error_label": fn[2]})
            for sp_err in row["errors"]["Span Errors"]: # SP ('J.M. Pfeil ,', 'J.M. Pfeil', 'FP')
                #print("SP", sp_err)
                sp_lb_errors.append({"text_id": text_id, "method": sys_name, "type": "SPAN", "true_text": sp_err[0], "error_text": sp_err[1], "true_label": "-", "error_label": "-"})
            for lbl_err in row["errors"]["Label Errors"]: # LB ('Woesthoven', 'PER', 'LOC', 'FP')
                #print("LB", lbl_err)
                sp_lb_errors.append({"text_id": text_id, "method": sys_name, "type": "LABEL", "true_text": lbl_err[0], "error_text": "-", "true_label": lbl_err[1], "error_label": lbl_err[2]})
            #print("-----")
        pd.DataFrame(fp_fn_errors).sort_values("error_text").to_csv(f"local_outputs/1_Errors_FP_FN_{eval_type}.csv", index=False)
        pd.DataFrame(sp_lb_errors).sort_values("true_text").to_csv(f"local_outputs/1_Errors_Span_Label_{eval_type}.csv", index=False)
    elif eval_type == "bag_of_entities":
        fp_fn_errors = []
        for i, row in enumerate(collected_errors):
            text_id = row["text_id"]
            sys_name = row["method"]
            for fp in row["errors"]["Full Errors (not in Gold)"]: # FP: ('ROYER', 'PER')
                #print("FP", fp)
                fp_fn_errors.append({"text_id": text_id, "method": sys_name, "type": "FP", "text": fp[0], "label": fp[1]})
            for fn in row["errors"]["Missed Entities"]: # FN: ('ROYER', 'PER')
                #print("FN", fn)
                fp_fn_errors.append({"text_id": text_id, "method": sys_name, "type": "FN", "text": fn[0], "label": fn[1]})
        pd.DataFrame(fp_fn_errors).sort_values("text").to_csv(f"local_outputs/1_Errors_FP_FN_{eval_type}.csv", index=False)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    systems = ["stanza_nl", "human_gold", "flair/ner-dutch-large_0.12.2", "gpt-3.5-turbo", "gysbert_hist_fx_finetuned_epoch2", "xlmr_ner_"]
    valid_labels = ["PER", "LOC", "ORG"]
    evaluate_bionet_intavia(systems, valid_labels, eval_type="full_match")
    evaluate_bionet_intavia(systems, valid_labels, eval_type="bag_of_entities")