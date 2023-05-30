from typing import List, Dict, Any
from classes_mirror import IntaviaEntity
from seqeval.metrics import classification_report

def evaluate_ner(reference: List[IntaviaEntity], hypothesis: List[IntaviaEntity]) -> Dict[str, Any]:
    ref_set = set(reference)
    hyp_set = set(hypothesis)     
    
    full_match = ref_set.intersection(hyp_set)  # TP) TruePositives (Exactly the same in both)
    hallucination = hyp_set.difference(ref_set) # FP) FalsePositives (Missing in Gold)
    missed = ref_set.difference(hyp_set)        # FN) FalseNegative (Missing in System Output)
    
    partial_match = []  # TP or FP/FN? Depends on Strictness: TP is partial matches allowed, or FP/FN if only exact matches count

    # We can Define if Strict or Lenient (are partial matches errors or matches?)
    match = full_match
    error = hallucination

    tp, fp, fn = len(match), len(error), len(missed)
    prec = 0 if tp+fp == 0 else 100*tp/(tp+fp)
    rec = 0 if tp+fn == 0 else 100*tp/(tp+fn)
    f1 = 0 if prec+rec == 0 else 2*(prec*rec)/(prec+rec)

    return {
        "reference": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in ref_set],
        "hypothesis": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in hyp_set],
        "match": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in full_match],
        "match_partial": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in partial_match],
        "error": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in error],
        "missed": [(ent.locationStart, ent.surfaceForm, ent.category) for ent in missed],
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": "{:.2f}".format(prec),
        "recall": "{:.2f}".format(rec),
        "f1": "{:.2f}".format(f1)
    }
