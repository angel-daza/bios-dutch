# On MICRO vs MACRO: https://stephenallwright.com/micro-vs-macro-f1-score/
from typing import List, Dict, Any
from collections import defaultdict, Counter
import glob, os, json, statistics
import pandas as pd

from utils.classes import IntaviaDocument
from utils_general import get_gold_annotations, INTAVIA_JSON_ROOT, conll_data_reader
from utils.nlp_tasks import unify_wordpiece_predictions

from flair.data import Sentence
from flair.models import SequenceTagger
import stanza
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report


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
        print(f"------ {sys_name} {eval_type} --------\n{evaluated_labels}\n{macro_p}\n{macro_r}\n{macro_f1}")
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
    if eval_type in ["full_match", "partial_match"]:
        fp_fn_errors, sp_lb_errors = [], []
        for i, row in enumerate(collected_errors):
            text_id = row["text_id"]
            sys_name = row["method"]
            for fp in row["errors"]["Full Errors (not in Gold)"]: # FP_Exact_Match: ('3_17', 'Paul-Henri Spaak', 'PER', 'FP')
                #print("FP", fp)
                fp_fn_errors.append({"text_id": text_id, "method": sys_name, "type": "FP", "error_span": fp[0], "error_text": fp[1], "error_label": fp[2]})
            for fn in row["errors"]["Missed Entities"]: # FN_Exact_Match: (553, 'Buitenlandse Zaken van Belgie', 'ORG', 'FN')
                #print("FN", fn)
                fp_fn_errors.append({"text_id": text_id, "method": sys_name, "type": "FN", "error_span": fn[0], "error_text": fn[1], "error_label": fn[2]})
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

def get_conll_stats():
    conll_files = [
        "data/bionet_gold/biographynet_test_A_gold.conll",
        "data/bionet_gold/biographynet_test_B_gold.conll",
        "data/bionet_gold/biographynet_test_C_gold.conll"
    ]
    sents_dict = {}
    for path in conll_files:
        sents_dict.update(conll_data_reader(path))
    # Global Stats
    tot_sents, tot_tokens =  0, 0
    doc_sizes, sent_sizes, doc_sent_sizes = [], [], []
    per_sizes, loc_sizes, org_sizes, ner_sizes = [], [], [], []
    all_labels = []
    for doc_id, sents in sents_dict.items():
        tot_sents += len(sents)
        doc_toks = 0
        per_c, loc_c, org_c, ner_c = 0, 0, 0, 0
        for s in sents:
            tot_tokens += len(s)
            sent_sizes.append(len(s))
            for sid, tok, lbl in s:
                all_labels.append(lbl)
                doc_toks += 1
                if lbl == "B-PER":
                    per_c += 1
                elif lbl == "B-LOC":
                    loc_c += 1
                elif lbl == "B-ORG":
                    org_c += 1
                if lbl in ["B-PER", "B-LOC", "B-ORG"]:
                    ner_c += 1
        per_sizes.append(per_c)
        loc_sizes.append(loc_c)
        org_sizes.append(org_c)
        ner_sizes.append(ner_c)
        doc_sizes.append(doc_toks)
    
    print(f"Total Documents = {len(doc_sizes)} == {len(sents_dict)}")
    print(f"Longest Doc = {max(doc_sizes)} | Shortest Doc = {min(doc_sizes)} | Average Doc = {statistics.mean(doc_sizes)} | Median = {statistics.median(doc_sizes)}")
    print(f"Total Sentences = {tot_sents} | MAX = {max(sent_sizes)} | Mean = {statistics.mean(sent_sizes)}  | Median = {statistics.median(sent_sizes)}")
    print(f"Total Tokens = {tot_tokens}")
    print(Counter(all_labels))
    print(f"PER = {sum(per_sizes)} | MAX = {max(per_sizes)} | Mean = {statistics.mean(per_sizes)}  | Median = {statistics.median(per_sizes)}")
    print(f"LOC = {sum(loc_sizes)} | MAX = {max(loc_sizes)} | Mean = {statistics.mean(loc_sizes)}  | Median = {statistics.median(loc_sizes)}")
    print(f"ORG = {sum(org_sizes)} | MAX = {max(org_sizes)} | Mean = {statistics.mean(org_sizes)}  | Median = {statistics.median(org_sizes)}")
    print(f"ALL = {sum(ner_sizes)} | MAX = {max(ner_sizes)} | Mean = {statistics.mean(ner_sizes)}  | Median = {statistics.median(ner_sizes)}")


def evaluate_conll_files():

    def _run_flair_pretok(tokens:List[str], tagger: SequenceTagger):
        text = " ".join(tokens)
        labels = []
        sentence = Sentence(text, use_tokenizer=False)
        tagger.predict(sentence)

        # transfer entity labels to token level
        for entity in sentence.get_spans('ner'):
            prefix = 'B-'
            for token in entity:
                token.set_label('ner-bio', prefix + entity.tag, entity.score)
                prefix = 'I-'

        for tok in sentence:
            lbl = tok.get_label()._value
            if "PER" in lbl or "ORG" in lbl or "LOC" in lbl:
                labels.append(lbl)
            else:
                labels.append("O")
        return labels

    def _run_stanza_pretok(tokens:List[str], stanza_nlp):
        text = " ".join(tokens)
        doc = stanza_nlp(text)
        labels = []
        for s in doc.sentences:
            for tok in s.tokens:
                lbl = tok.ner.replace("E-", "I-").replace("S-", "B-")
                if "PER" in lbl or "ORG" in lbl or "LOC" in lbl:
                    labels.append(lbl)
                else:
                    labels.append("O")
        return labels
    
    def _run_xlmr_pretok(tokens: List[str], bert_nlp):
        sentence = " ".join(tokens)
        predictions = bert_nlp(sentence)
        tagged_ents = unify_wordpiece_predictions(predictions, '▁')
        return tagged_ents

    ## Flair Model
    flair_model = "flair/ner-dutch-large"
    flair_tagger = SequenceTagger.load(flair_model)
    
    ## Stanza Model
    stanza_nlp = stanza.Pipeline(lang="nl", processors="tokenize,lemma,pos,depparse,ner", model_dir="/Users/daza/stanza_resources/", tokenize_pretokenized=True)
    
    # ## XLM-R Model
    # checkpoint = "Davlan/xlm-roberta-base-ner-hrl"
    # xlmr_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # xlmr_model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    # xlmr_nlp = pipeline('token-classification', model=xlmr_model, tokenizer=xlmr_tokenizer, device=-1)
    
    # Pupulate Data
    conll_files = [
        "data/bionet_gold/biographynet_test_A_gold.conll",
        "data/bionet_gold/biographynet_test_B_gold.conll",
        "data/bionet_gold/biographynet_test_C_gold.conll"
    ]
    sents_dict = {}
    for path in conll_files:
        sents_dict.update(conll_data_reader(path))
    # Iterate and Run NLP Systems on pre-tokenized data
    c = 0
    all_all_gold, all_all_stanza, all_all_flair, all_all_xlmr = [], [], [], []
    all_doc_gold, all_doc_stanza, all_doc_flair, all_doc_xlmr = [], [], [], []
    all_sent_gold, all_sent_stanza, all_sent_flair, all_sent_xlmr = [], [], [], []
    for doc_id, sents in sents_dict.items():
        #print(doc_id)
        doc_gold_labels, doc_stanza_labels, doc_flair_labels, doc_xlmr_labels = [], [], [], []
        for s in sents:
            tokens, gold_labels = [], []
            for sid, tok, lbl in s:
                tokens.append(tok)
                if "PER" in lbl or "ORG" in lbl or "LOC" in lbl:
                    gold_labels.append(lbl)
                else:
                    gold_labels.append("O")
            flair_labels = _run_flair_pretok(tokens, flair_tagger)
            stanza_labels = _run_stanza_pretok(tokens, stanza_nlp)
            xlmr_labels = [] #_run_xlmr_pretok(tokens, xlmr_nlp)
            # print(tokens)
            # print(gold_labels)
            # print(flair_labels)
            # print(stanza_labels)
            # print(xlmr_labels)
            all_all_gold += gold_labels
            all_sent_gold.append(gold_labels)
            doc_gold_labels += gold_labels
            #
            all_all_stanza += stanza_labels
            all_sent_stanza.append(stanza_labels)
            doc_stanza_labels += stanza_labels
            #
            all_all_flair += flair_labels
            all_sent_flair.append(flair_labels)
            doc_flair_labels += flair_labels
            #
            all_all_xlmr += xlmr_labels
            all_sent_xlmr.append(xlmr_labels)
            doc_xlmr_labels += xlmr_labels
        #print("----")
        all_doc_gold.append(doc_gold_labels) 
        all_doc_stanza.append(doc_stanza_labels) 
        all_doc_flair.append(doc_flair_labels)
        all_doc_xlmr.append(doc_xlmr_labels)
        c += 1
        # if c ==5: break

    print("Default with All Labels Flattened")
    stanza_report = classification_report([all_all_gold], [all_all_stanza], digits=4)
    flair_report = classification_report([all_all_gold], [all_all_flair], digits=4)
    print(stanza_report)
    print(flair_report)
    # xlmr_report = classification_report([all_all_gold], [all_all_xlmr])
    # print(xlmr_report)
    
    print("Strict with All Labels Flattened")
    stanza_report = classification_report([all_all_gold], [all_all_stanza], mode='strict', digits=4)
    flair_report = classification_report([all_all_gold], [all_all_flair], mode='strict', digits=4)
    print(stanza_report)
    print(flair_report)
    # xlmr_report = classification_report([all_all_gold], [all_all_xlmr], mode='strict')
    # print(xlmr_report)
    
    print("Default with Doc-Level Labels")
    stanza_report = classification_report(all_doc_gold, all_doc_stanza, digits=4)
    flair_report = classification_report(all_doc_gold, all_doc_flair, digits=4)
    print(stanza_report)
    print(flair_report)
    # xlmr_report = classification_report(all_doc_gold, all_doc_xlmr)
    # print(xlmr_report)
    
    print("Strict with Doc-Level Labels")
    stanza_report = classification_report(all_doc_gold, all_doc_stanza, mode='strict', digits=4)
    flair_report = classification_report(all_doc_gold, all_doc_flair, mode='strict', digits=4)
    print(stanza_report)
    print(flair_report)
    # xlmr_report = classification_report(all_doc_gold, all_doc_xlmr, mode='strict')
    # print(xlmr_report)
    
    print("Default with Sentence-Level Labels")
    stanza_report = classification_report(all_sent_gold, all_sent_stanza, digits=4)
    flair_report = classification_report(all_sent_gold, all_sent_flair, digits=4)
    print(stanza_report)
    print(flair_report)
    # xlmr_report = classification_report(all_sent_xlmr, all_sent_xlmr)
    # print(xlmr_report)
    
    print("Strict with Sentence-Level Labels")
    stanza_report = classification_report(all_sent_gold, all_sent_stanza, mode='strict', digits=4)
    flair_report = classification_report(all_sent_gold, all_sent_flair, mode='strict', digits=4)
    print(stanza_report)
    print(flair_report)
    # xlmr_report = classification_report(all_sent_xlmr, all_sent_xlmr, mode='strict')
    # print(xlmr_report)
    



if __name__ == "__main__":
    systems = ["stanza_nl", "flair/ner-dutch-large_0.12.2", "gpt-3.5-turbo", "xlmr_ner_"] # ["stanza_nl", "human_gold", "flair/ner-dutch-large_0.12.2", "gpt-3.5-turbo", "gysbert_hist_fx_finetuned_epoch2", "xlmr_ner_"]
    valid_labels = ["PER", "LOC", "ORG"]
    #evaluate_bionet_intavia(systems, valid_labels, eval_type="full_match")
    #evaluate_bionet_intavia(systems, valid_labels, eval_type="bag_of_entities")
    #evaluate_bionet_intavia(systems, valid_labels, eval_type="partial_match")

    # get_conll_stats()
    evaluate_conll_files()