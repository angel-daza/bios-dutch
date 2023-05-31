from typing import List, Dict, Any
from classes_mirror import IntaviaEntity
from seqeval.metrics import classification_report

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


    tp, fp, fn = len(full_match), len(hallucination), len(missed)
    prec = 0 if tp+fp == 0 else 100*tp/(tp+fp)
    rec = 0 if tp+fn == 0 else 100*tp/(tp+fn)
    f1 = 0 if prec+rec == 0 else 2*(prec*rec)/(prec+rec)

    return {
        # "reference": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in sorted_ref],
        # "hypothesis": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in sorted_hyp],
        "Full Match": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in full_match],
        "Span Errors": [(ent1.surfaceForm, ent2.surfaceForm) for (ent1, ent2) in span_error],
        "Label Errors": [(ent1.surfaceForm, ent1.category, ent2.surfaceForm, ent2.category) for (ent1, ent2) in label_error],
        "Full Errors (not in Gold)": [(f"{ent.locationStart}_{ent.locationEnd}", ent.surfaceForm, ent.category) for ent in hallucination],
        "Missed Entities": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in missed],
        "Frequency": len(reference),
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "Precision": "{:.1f}".format(prec),
        "Recall": "{:.1f}".format(rec),
        "F1": "{:.1f}".format(f1)
    }