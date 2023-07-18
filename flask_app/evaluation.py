from typing import List, Dict, Any
from classes_mirror import IntaviaEntity
import statistics
from collections import defaultdict


def document_ner_evaluation(reference_entities: List[IntaviaEntity], predicted_entities: List[IntaviaEntity], valid_labels: List[str] = None) -> Dict[str, Any]:
    macro_p, macro_r, macro_f = [], [], []
    total_freq = 0
    # Filter for Valid Labels ONLY
    if not valid_labels: valid_labels = [x.category for x in reference_entities]
    reference_entities = [x for x in reference_entities if x.category in valid_labels]
    predicted_entities = [x for x in predicted_entities if x.category in valid_labels]
    # F1 SCORES PER LABEL
    eval_per_label = {}
    per_label_table = []
    for lbl in valid_labels:
        lbl_gold = [x for x in reference_entities if x.category == lbl]
        lbl_pred = [x for x in predicted_entities if x.category == lbl]
        lbl_metrics = evaluate_ner(lbl_gold, lbl_pred)
        if  lbl_metrics["Frequency"] > 0:
            macro_p.append(float(lbl_metrics["Precision"]))
            macro_r.append(float(lbl_metrics["Recall"]))
            macro_f.append(float(lbl_metrics["F1"]))
            total_freq += lbl_metrics["Frequency"]
            eval_per_label[lbl] = {"P": lbl_metrics["Precision"], "R": lbl_metrics["Recall"], "F1": lbl_metrics["F1"], "Frequency": lbl_metrics["Frequency"],
                                   "TP": lbl_metrics["TP"], "FP": lbl_metrics["FP"], "FN": lbl_metrics["FN"]}
            per_label_table.append([lbl, lbl_metrics["Precision"], lbl_metrics["Recall"], lbl_metrics["F1"], lbl_metrics["Frequency"]])
        else:
            per_label_table.append([lbl, "-", "-", "-", "-"])
    # TOTAL MACRO (Unweighted Average of Scores)
    if  total_freq > 0:
        macro_p = statistics.mean(macro_p)
        macro_r = statistics.mean(macro_r)
        macro_f = statistics.mean(macro_f)
        macro_metrics = {"P": "{:.1f}".format(macro_p), "R": "{:.1f}".format(macro_r), "F1": "{:.1f}".format(macro_f), "Frequency": total_freq}
        eval_per_label["MACRO"] = macro_metrics
        per_label_table.append(["MACRO", macro_metrics["P"], macro_metrics["R"], macro_metrics["F1"], macro_metrics["Frequency"]])
    # TOTAL EVAL (MICRO -> Accuracy)
    micro_metrics = evaluate_ner(reference_entities, predicted_entities)
    eval_per_label["MICRO"] = {"P": micro_metrics["Precision"], "R": micro_metrics["Recall"], "F1": micro_metrics["F1"], "Frequency": micro_metrics["Frequency"],
                                "TP": micro_metrics["TP"], "FP": micro_metrics["FP"], "FN": micro_metrics["FN"]}
    per_label_table.append(["MICRO", micro_metrics["Precision"], micro_metrics["Recall"], micro_metrics["F1"], micro_metrics["Frequency"]])

    # FINAL REPORT DICT
    eval_metrics = micro_metrics # The full dict returned by evaluate_ner()
    eval_metrics["Macro_P"] = macro_metrics["P"]
    eval_metrics["Macro_R"] = macro_metrics["R"]
    eval_metrics["Macro_F1"] = macro_metrics["F1"]
    
    return {
        "eval_metrics": eval_metrics,
        "per_label_dict": eval_per_label,
        "per_label_table": per_label_table
    }


def evaluate_ner(reference: List[IntaviaEntity], hypothesis: List[IntaviaEntity]) -> Dict[str, Any]:
    sorted_ref = sorted(reference, key = lambda ent: ent.locationStart)
    sorted_hyp = sorted(hypothesis, key = lambda ent: ent.locationStart)     
    
    full_match = []     # TP) TruePositives (Exactly the same in both) - match
    hallucination = []  # FP) FalsePositives (Missing in Gold) - error
    missed = []         # FN) FalseNegative (Missing in System Output) - missed
    label_error = []    #   -   The subset of errors that has the correct span but WRONG LABEL
    span_error = []     #   -   Right Label but WRONG SPAN. TP or FP/FN? Depends on Strictness: TP is partial matches allowed, or FP/FN if only exact matches count

    for ref in sorted_ref:
        for hyp in sorted_hyp:
            if hyp.locationStart > ref.locationEnd:
                break
            if ref == hyp:
                full_match.append(ref)
            elif ref.span_match(hyp) and ref.category != hyp.category:
                missed.append(ref)
                label_error.append((ref, hyp))
            elif ref.span_partial_match(hyp):
                missed.append(ref)
                if ref.category == hyp.category:
                    span_error.append((ref, hyp))
                else:
                    label_error.append((ref, hyp))

    # span_err_hyp = [y for x,y in span_error]
    # label_err_hyp = [y for x,y in label_error]
    for hyp in sorted_hyp:
        # if hyp not in sorted_ref and hyp not in missed and hyp not in span_err_hyp and hyp not in label_err_hyp:
        if hyp not in sorted_ref and hyp not in missed:
            hallucination.append(hyp)

    # Double-check the Missed and LabelError Array, in case the overlapped entities were already counted in the TruePositives
    # e.g. (1489, 'landing der Engelsche in Zeeland', 'MISC', 'Zeeland', 'LOC'), AND (1514, 'Zeeland', 'LOC', 'landing der Engelsche in Zeeland', 'MISC')
    filtered_missed = []
    for m in missed:
        if m not in full_match:
            filtered_missed.append(m)
    missed = filtered_missed
    filtered_label_err = []
    for x,y in label_error:
        if y not in full_match:
            filtered_label_err.append((x,y))
    label_error = filtered_label_err
    # Compute Metrics
    tp, fp, fn = len(full_match), len(hallucination), len(missed)
    prec = 0 if tp+fp == 0 else 100*tp/(tp+fp)
    rec = 0 if tp+fn == 0 else 100*tp/(tp+fn)
    f1 = 0 if prec+rec == 0 else 2*(prec*rec)/(prec+rec)
    # Return Everything
    return {
        # "reference": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in sorted_ref],
        # "hypothesis": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in sorted_hyp],
        "Full Match": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in full_match],
        "Span Errors": [(ent1.surfaceForm, ent2.surfaceForm) for (ent1, ent2) in span_error],
        "Label Errors": [(ent1.surfaceForm, ent1.category, ent2.surfaceForm, ent2.category) for (ent1, ent2) in label_error],
        "Full Errors (not in Gold)": [(f"{ent.locationStart}_{ent.locationEnd}", ent.surfaceForm, ent.category) for ent in hallucination],
        "Missed Entities": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in missed],
        "Frequency": len(reference),
        "TP": tp, # True Positives
        "FP": fp, # False Positives
        "FN": fn, # False Negatives
        "Precision": "{:.1f}".format(prec),
        "Recall": "{:.1f}".format(rec),
        "F1": "{:.1f}".format(f1)
    }


def system_label_report(systems_metrics: Dict[str, Any]) -> List[List[Any]]:
    report_table = defaultdict(list)
    for sys_vs_sys_names, label_metrics in systems_metrics.items():
        sys_name = sys_vs_sys_names.split("_vs_")[1]
        for label_name, metrics in label_metrics.items():
            if label_name not in ["MICRO", "MACRO"]:
                report_table[label_name].append({
                    "M": sys_name,
                    "P": metrics["P"],
                    "R": metrics["R"],
                    "F1": metrics["F1"]
                })
    return report_table
