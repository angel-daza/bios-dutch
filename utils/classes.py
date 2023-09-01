"""
    Mirror Class definitions from a different project - BiographyNet/utils/classes.py - DO NOT CHANGE BEHAVIOR HERE!!! ONLY COPY from there
"""

from typing import Dict, List, NamedTuple, Union, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
import re, statistics


# System Keys: 'human_gold', 'stanza_nl', 'flair/ner-dutch-large_0.12.2', 'gpt-3.5-turbo', 'gysbert_hist_fx_finetuned_epoch2'
NER_METHOD_DISPLAY = {'human_gold': 'gold', 
                    'stanza_nl': 'stanza', 
                    'flair/ner-dutch-large_0.12.2': 'flair', 
                    'gpt-3.5-turbo': 'gpt', 
                    'gysbert_hist_fx_finetuned_epoch2': 'gysbert',
                    'xlmr_ner_': 'xlmr_ner',
                    }


@dataclass 
class IntaviaToken:
    ID: int
    FORM: str
    LEMMA: str
    UPOS: str
    XPOS: str
    HEAD: int
    DEPREL: str
    DEPS: str
    MISC: Dict[str, str] = None
    FEATS: List[str] = None


@dataclass
class IntaviaEntity:
    ID: str
    surfaceForm: str
    category: str
    locationStart: int
    locationEnd: int
    tokenStart: int = None
    tokenEnd: int = None
    method: str = None
    label_dict = {"PER": "PER", "PERSON": "PER", 
                      "LOC": "LOC", "LOCATION": "LOC", 
                      "ORG": "ORG", "ORGANIZATION": "ORG",
                      "MISC": "MISC",
                      "ARTWORK": "ARTWORK", "WORK_OF_ART": "ARTWORK",
                      "TIME": "TIME", "DATE": "TIME"}

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return self.locationStart == other.locationStart and self.locationEnd == other.locationEnd and self.category == other.category
        else:
            return False
    
    def __hash__(self) -> int:
        return hash((self.locationStart, self.surfaceForm, self.category))

    def span_match(self, other: object):
        if type(other) is type(self):
            return self.locationStart == other.locationStart and self.locationEnd == other.locationEnd
        else:
            return False
    # same Span Start OR same Span End
    def span_partial_match(self, other: object):
        if type(other) is type(self):
            return self.locationStart == other.locationStart or self.locationEnd == other.locationEnd
        else:
            return False
    
    def normalize_label(self):
        unnorm = self.category
        self.category = self.label_dict.get(self.category, "MISC")
        # if self.category == "MISC":
        #     if unnorm != "MISC": # to avoid printing the error more than once (e.g. when the normalized entities are again normalized)
        #         print(f"{unnorm} --> MISC")

    def get_displacy_format(self):
        return {"start": self.locationStart, "end": self.locationEnd, "label": self.category}


@dataclass
class IntaviaTimex:
    ID: str
    surfaceForm: str
    value: str
    category: str
    locationStart: int
    locationEnd: int
    method: str

@dataclass
class IntaviaSentence:
    paragraph: int
    sentence: int
    text: str
    words: List[IntaviaToken]

# Strict NER Evaluation (Only Exact Span Matches == TP)
def _evaluate_ner(reference: List[IntaviaEntity], hypothesis: List[IntaviaEntity]) -> Dict[str, Any]:
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
        if hyp not in sorted_ref and hyp not in missed and hyp not in hallucination:
            hallucination.append(hyp)

    # Double-check the Missed and LabelError Array, in case the overlapped entities were already counted in the TruePositives
    # e.g. (1489, 'landing der Engelsche in Zeeland', 'MISC', 'Zeeland', 'LOC'), AND (1514, 'Zeeland', 'LOC', 'landing der Engelsche in Zeeland', 'MISC')
    filtered_missed = []
    for m in missed:
        if m not in full_match and m not in filtered_missed:
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
        "Full Match": [(ent.locationStart, ent.surfaceForm, ent.category, "TP") for ent in full_match],
        "Span Errors": [(ent1.surfaceForm, ent2.surfaceForm, "FP") for (ent1, ent2) in span_error],
        "Label Errors": [(ent1.surfaceForm, ent1.category, ent2.category, "FP") for (ent1, ent2) in label_error],
        "Full Errors (not in Gold)": [(f"{ent.locationStart}_{ent.locationEnd}", ent.surfaceForm, ent.category, "FP") for ent in hallucination],
        "Missed Entities": [(ent.locationStart, ent.surfaceForm, ent.category, "FN") for ent in missed],
        "Support": len(reference),
        "TP": tp, # True Positives
        "FP": fp, # False Positives
        "FN": fn, # False Negatives
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1": round(f1, 2)
    }

# NER Evaluation as Bag-of-Entities
def _evaluate_ner_boe(reference: List[IntaviaEntity], hypothesis: List[IntaviaEntity]) -> Dict[str, Any]:
    # Get Only NER (Text, Label) Info
    ref_entities = set([(ent.surfaceForm, ent.category) for ent in reference]) # IF model lowercases things: ent.surfaceForm.lower()
    hyp_entities = set([(ent.surfaceForm, ent.category) for ent in hypothesis])
    # Compute Set Operations
    match = ref_entities.intersection(hyp_entities)
    error = hyp_entities.difference(ref_entities)
    missed = ref_entities.difference(hyp_entities)
    tp, fp, fn = len(match), len(error), len(missed)
    prec = 0 if tp+fp == 0 else 100*tp/(tp+fp)
    rec = 0 if tp+fn == 0 else 100*tp/(tp+fn)
    f1 = 0 if prec+rec == 0 else 2*(prec*rec)/(prec+rec)
    # Return Metrics
    return {
        "Full Match": match,
        "Full Errors (not in Gold)": error,
        "Missed Entities": missed,
        "Support": len(reference),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1": round(f1, 2)
    }

class IntaviaDocument:
    def __init__(self, intavia_dict: Dict[str, Any], nlp_base_model: str = 'stanza_nl'):
        self.text_id: str = intavia_dict['text_id']
        self.text:str = intavia_dict['data']['text']
        if isinstance(intavia_dict['data']['tokenization'], List): # For Compatibility with previous IntaviaJSON Template
            self.tokenization: List[str] = intavia_dict['data']['tokenization']
            morpho = intavia_dict['data']['morpho_syntax']
        else:
            self.tokenization: List[str] = intavia_dict['data']['tokenization'][nlp_base_model]
            morpho = intavia_dict['data']['morpho_syntax'][nlp_base_model]
        self.morpho_syntax: List[IntaviaSentence] = [] 
        for sent_obj in morpho:
            tokens = [IntaviaToken(**word_obj) for word_obj in sent_obj['words']]
            pid = sent_obj.get('paragraphID') or sent_obj.get('paragraph')
            sid = sent_obj.get('sentenceID') or sent_obj.get('sentence')
            sentence = IntaviaSentence(pid, sid, sent_obj['text'], tokens)
            self.morpho_syntax.append(sentence)
        self.entities: List[IntaviaEntity] = [IntaviaEntity(**ent) for ent in intavia_dict['data'].get('entities', [])]
        self.time_expressions: List[IntaviaTimex] = [IntaviaTimex(**tim) for tim in intavia_dict['data'].get('time_expressions', [])]
        self.semantic_roles: List[Dict[str, Any]] = intavia_dict['data'].get('semantic_roles', [])
        self.metadata = {}
        for k,v in intavia_dict.items():
            if k not in ['text_id', 'data']:
                self.metadata[k] = v
    
    def get_basic_stats(self) -> Dict[str, Any]:
        sentences = []
        verbs, nouns, adjs = [], [], []
        for sent in self.morpho_syntax:
            sentences.append(sent.text)
            for tok in sent.words:
                lemma = tok.LEMMA
                if tok.UPOS == "VERB":
                    verbs.append(lemma)
                elif tok.UPOS in ["NOUN", "PROPN"]:
                    nouns.append(lemma)
                elif tok.UPOS == "ADJ":
                    adjs.append(lemma)
        return {
            'sentences': sentences,
            "top_verbs": Counter(verbs).most_common(10),
            "top_nouns": Counter(nouns).most_common(10),
            "top_adjs": Counter(adjs).most_common(10),
        }
    
    def get_available_methods(self, task_layer: str) -> List[str]:
        if task_layer == "entities":
            return list(set([ent.method for ent in self.entities]))
        elif task_layer == "time_expressions":
            return list(set([ent.method for ent in self.time_expressions]))
        elif task_layer == "semantic_roles":
            return list(set([ent['method'] for ent in self.semantic_roles]))
        else:
            raise ValueError(f"NLP Layer {task_layer} is not a valid layer in the IntaviaDocument") 

    def get_entities(self, methods: List[str] = ['all'], valid_labels: List[str] = None) -> List[IntaviaEntity]:
        """_summary_
        Args:
            methods (List[str], optional): Filter entitities according to one or more <methods> | 'all' (everything in the list) | 'intersection' (only entities produced by all models listed in <methods>)
        Returns:
            List[IntaviaEntity]: The requested list of Entities.
        """
        normalized_entities = []
        if valid_labels:
            for ent in self.entities:
                ent.normalize_label()
                if ent.category in valid_labels:
                    normalized_entities.append(ent)
        else:
            for ent in self.entities:
                ent.normalize_label()
                normalized_entities.append(ent)
        if 'all' in methods:
            entities = normalized_entities
        elif 'intersection' in methods:
            raise NotImplementedError
        else:
            entities = [ent for ent in normalized_entities if ent.method in methods]
        
        return entities
    
    def get_entity_counts(self, methods: List[str] = ['all'], valid_labels: List[str] = None, top_k: int = -1) -> Dict[str, List[Tuple[str, int]]]:
        "Returns a dictionary of Methods [KEY = 'method_name'] and entity distribution [VALUE = List of (label, freq) pairs]"
        entity_src_dict = defaultdict(list)
        entities = self.get_entities(methods, valid_labels)
        for ent_obj in entities:
            if valid_labels: 
                if ent_obj.category in valid_labels:
                    entity_src_dict[ent_obj.method].append(ent_obj.category)
            else:
                entity_src_dict[ent_obj.method].append(ent_obj.category)
        entity_dict = {}
        for src, ents in entity_src_dict.items():
            if top_k > 0:
                entity_dict[src] = Counter(ents).most_common(top_k)
            else:
                entity_dict[src] = Counter(ents).most_common()
        return entity_dict

    def get_entity_category_matrix(self):
        all_methods, all_labels = set(), set()
        entity_info = defaultdict(list)
        for ent in self.get_entities():
            all_labels.add(ent.category)
            all_methods.add(ent.method)
            entity_info[f"{ent.method}_{ent.category}"].append(ent)
        
        
        sorted_found_labels = sorted(all_labels)
        sorted_found_methods = sorted(all_methods)
        entity_table = [["Category"] + [method for method in sorted_found_methods]]
        for label in sorted_found_labels:
            row = [label]
            for method in sorted_found_methods:
                key = f"{method}_{label}"
                if key in entity_info:
                    ent_counts = len(entity_info[key])
                    row.append(ent_counts)
                else:
                    row.append(0)
            entity_table.append(row)
        
        return entity_table

    def get_confidence_entities(self, mode: str = "spans") -> List[Dict]:
        " mode = 'spans' or 'ents' "
       
        if mode not in ["spans", "ents"]:
            raise NotImplementedError

        entity_agreement = []
        methods = [m for m in self.get_available_methods("entities") if m != "gpt-3.5-turbo"]
        max_agreement = len(methods)
        
        if max_agreement == 0: return []

        if mode == "spans":
            charstart2token, charend2token = {}, {}
            for sent in self.morpho_syntax:
                for token in sent.words:
                    charstart2token[token.MISC['StartChar']] = token.ID
                    charend2token[token.MISC['EndChar']] = token.ID

        for ent_obj in self.get_entities(methods=methods):
            key = f"{ent_obj.surfaceForm}#{ent_obj.locationStart}#{ent_obj.locationEnd}#{ent_obj.category}"
            entity_agreement.append(key)
        entity_agreement = Counter(entity_agreement).most_common()
        entity_confidence_spans = []
        for ent_key, freq in entity_agreement:
            agreement_ratio = freq/max_agreement
            text, start, end, label = ent_key.split("#")
            if agreement_ratio <= 0.3:
                confidence_cat = "LOW"
            elif 0.3 < agreement_ratio <= 0.5:
                confidence_cat = "WEAK"
            elif 0.5 < agreement_ratio <= 0.75:
                confidence_cat = "MEDIUM"
            elif 0.75 < agreement_ratio <= 0.89:
                confidence_cat = "HIGH"
            else:
                confidence_cat = "VERY HIGH"
            
            if mode == "spans":
                token_start = charstart2token.get(int(start))
                token_end = charend2token.get(int(end))
                if type(token_start) == int and type(token_end) == int:
                    entity_confidence_spans.append({"start_token": int(token_start), "end_token": int(token_end) + 1, "label": label})
            else:
                entity_confidence_spans.append({"text": text, "start": int(start), "end": int(end), "label": confidence_cat})
        
        return entity_confidence_spans
    
    def get_entities_IOB(self) -> List[str]:
        raise NotImplementedError

    def evaluate_ner(self, reference_method: str, eval_method: str, evaluation_type: str, valid_labels: List[str] = None, ignore_text_after_gold:bool = False) -> Dict[str, Any]:
        """Evaluate NER In The whole Document
        Args:
            reference_method (str): NER model-specific outputs considered as reference. Dictionary Key should be in the IntaviaDocument.get_available_methods("entities")
            eval_method (str): NER model-specific outputs considered as predictions. Dictionary Key should be in the IntaviaDocument.get_available_methods("entities")
            evaluation_type (str): How strict to be when evaluating. Possible types: ['full_match', 'bag_of_entities']
            valid_labels (List[str], optional): List of Labels to consider for evaluation, all other labels will be ignored. If None then all labels will be considered. Defaults to None.
            ignore_text_after_gold (bool, optional): _description_. DO NOT Evaluate entities predicted after the last Reference Entity (e.g. to avoid considerinf doc references). Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary with metrics and errors found during the evaluation
        """
        nlp_systems = self.get_available_methods("entities")
        # Gold INFO
        reference_entities = sorted(self.get_entities([reference_method], valid_labels=valid_labels), key = lambda ent: ent.locationStart)
        # Return Empty if there is no Gold or if the method to evaluate is not available in the File
        if len(reference_entities) == 0 or eval_method not in nlp_systems: return None
        if not valid_labels: 
            valid_labels = list(set([e.category for e in reference_entities]))
        # Predictions INFO
        if ignore_text_after_gold:
            last_char_to_eval = reference_entities[-1].locationEnd + 1
        else:
            last_char_to_eval = len(self.text)
        predicted_entities = sorted([ent for ent in self.get_entities(valid_labels=valid_labels) if ent.method == eval_method], key = lambda ent: ent.locationStart)
        predicted_entities = [pe for pe in predicted_entities if pe.locationStart < last_char_to_eval]
        # --- EVALUATION ---
        all_metrics_dict = {}
        # TOTAL EVAL (MICRO -> Accuracy)
        if evaluation_type == 'full_match':
            micro_metrics = _evaluate_ner(reference_entities, predicted_entities)
        elif evaluation_type == 'bag_of_entities':
            micro_metrics = _evaluate_ner_boe(reference_entities, predicted_entities)
        else:
            raise NotImplementedError("Evaluation type shouls be 'full_match' or 'bag_of_entities'")
        all_metrics_dict["MICRO"] = {"P": micro_metrics["Precision"], "R": micro_metrics["Recall"], "F1": micro_metrics["F1"], "Support": micro_metrics["Support"],
                                    "TP": micro_metrics["TP"], "FP": micro_metrics["FP"], "FN": micro_metrics["FN"]}
        errors = {k:v for k,v in micro_metrics.items() if k not in all_metrics_dict["MICRO"].keys()}
        # TOTAL MACRO (Unweighted Average of Scores)
        total_freq = 0
        macro_p, macro_r, macro_f = [], [], []
        all_metrics_dict["MACRO"] = {}
        for lbl in valid_labels:
            lbl_gold = [x for x in reference_entities if x.category == lbl]
            lbl_pred = [x for x in predicted_entities if x.category == lbl]
            if evaluation_type == 'full_match':
                lbl_metrics = _evaluate_ner(lbl_gold, lbl_pred)
            elif evaluation_type == 'bag_of_entities':
                lbl_metrics = _evaluate_ner_boe(lbl_gold, lbl_pred)
            if  lbl_metrics["Support"] > 0:
                macro_p.append(float(lbl_metrics["Precision"]))
                macro_r.append(float(lbl_metrics["Recall"]))
                macro_f.append(float(lbl_metrics["F1"]))
                total_freq += lbl_metrics["Support"]
                all_metrics_dict[lbl] = {"P": lbl_metrics["Precision"], "R": lbl_metrics["Recall"], "F1": lbl_metrics["F1"], "Support": lbl_metrics["Support"],
                                    "TP": lbl_metrics["TP"], "FP": lbl_metrics["FP"], "FN": lbl_metrics["FN"]}
        if  total_freq > 0:
            all_metrics_dict["MACRO"]["Support"] = total_freq
            all_metrics_dict["MACRO"]["P"] = round(statistics.mean(macro_p), 2)
            all_metrics_dict["MACRO"]["R"] = round(statistics.mean(macro_r), 2)
            all_metrics_dict["MACRO"]["F1"] = round(statistics.mean(macro_f), 2)
        
        return {"metrics": all_metrics_dict, "errors": errors}

    def get_ner_variance(self, valid_labels: List[str] = ['PER', 'LOC', 'ORG', 'MISC']) -> Dict[str, float]:
        token_len = len(self.tokenization)
        row = {
                'text_id': self.text_id,
                'tokens': token_len,
                }
        entity_counts = self.get_entity_counts(valid_labels=valid_labels)
        label2method_mapping = defaultdict(list)
        for method, entity_dist in entity_counts.items():
            total_freq = 0
            for lbl, freq in entity_dist:
                row[f"{lbl.lower()}_freq_{method}"] = freq
                total_freq += freq
                label2method_mapping[lbl].append(freq)
            row[f"entity_freq_{method}"] = total_freq
            row[f"entity_density_{method}"] = round(total_freq*100 / token_len, 2)
        # Get Info from label distribution across methods
        for lbl, method_dist in label2method_mapping.items():
            if len(method_dist) > 1:
                row[f"{lbl.lower()}_stdev"] = round(statistics.stdev(method_dist), 4)
            else:
                row[f"{lbl.lower()}_stdev"] = 0
        
        return row

class Event:
    '''Object that can describe an event (time, place, description)'''
    
    def __init__(self, label, location, date):
        self.label: str = label
        self.location: str = location
        if date and isinstance(date, str):
            date = date.strip()
        elif date and isinstance(date, float):
            date = str(int(date))
        self.date: str = date if date else None
        self.date_tuple: Tuple = (0, 0, 0)
        self.date_range: Tuple = (0, 0)
        self.date_is_certain = True
        # The Date Field is too dirty. Here we pack everything in a tuple to later make calculations easier
        if self.date and len(self.date) > 0:
            info_full = re.search(r"(\d{4})-(\d{2})-(\d{2})?", self.date) # Exact Full Date Known
            if info_full: 
                self.date_tuple = (int(info_full.group(1)), int(info_full.group(2)), int(info_full.group(3))) # (1708, 10, 11)
            elif len(self.date) == 4 or len(self.date) == 3: # Only the year is known
                self.date_tuple = (int(self.date), 0, 0) # (1708, )
            elif self.date == '?':
                self.date_is_certain = False
            else:
                info_year_month = re.search(r"(\d{4})-(\d{2})", self.date)
                if info_year_month:
                    self.date_tuple = (int(info_year_month.group(1)), int(info_year_month.group(2)), 0) 
                else:
                    info_range_4 = re.search(r"(\d{4})~(\d{4})", self.date) # Event happened sometime between two years (e.g. 1919~1934)
                    info_range_3 = re.search(r"(\d{3})~(\d{3})", self.date) # Event happened sometime between two years (e.g. 519~534)
                    if info_range_4:
                        self.date_tuple = (int(info_range_4.group(1)), 0, 0) # Arbitrarily choose the first date
                        self.date_range = (int(info_range_4.group(1)),int(info_range_4.group(2)))
                        self.date_is_certain = False
                    elif info_range_3:
                        self.date_tuple = (int(info_range_3.group(1)), 0, 0) # Arbitrarily choose the first date
                        self.date_range = (int(info_range_3.group(1)),int(info_range_3.group(2)))
                        self.date_is_certain = False
                    else:
                        info_year = re.search(r"(\d{4})", self.date)
                        info_year_3 = re.search(r"(\d{3})", self.date)
                        self.date_is_certain = False
                        try:
                            self.date_tuple = (int(info_year.group(1)), 0, 0)
                        except:
                            try:
                                self.date_tuple = (int(info_year_3.group(1)), 0, 0)
                            except:
                                # TODO: Comment the following lines. For now, they are here to explicitly catch "strange" date formats
                                if not any([x.isalpha() for x in self.date]):
                                    print("DATE -->",self.date, len(self.date))
        else:
            self.date_is_certain = False



    def __repr__(self):
        if self.date and self.location:
            return f"date: {self.date}, location: {self.location}"
        elif self.date:
            return f"date: {self.date}"
        elif self.date_tuple:
            return f"{self.date_tuple}"
        else:
            return f"date: UNK"
    
    def to_json(self):
        return {
            'label': self.label,
            'location': self.location,
            'date': self.date
        }

    def setDate(self, date):
        self.date = date

    def setLocation(self, location):
        self.location = location
    
    def getYear(self):
        if self.date_tuple:
            return self.date_tuple[0]
        else:
            return None
    
    def getDate(self):
        return self.date_tuple
    
    def getLocation(self):
        if self.location and len(self.location) > 0:
            return self.location
        else:
            return None


class State:
    '''Object that can describe a state (begin time, end time, place, description)'''
    
    def __init__(self, label, description = '', location = '', beginDate = '', endDate = ''):
        self.label = label
        self.location = location
        self.beginDate = beginDate
        self.endDate = endDate
        self.description = description
    
    def __repr__(self) -> str:
        state_info = f"{self.label.title()}"
        if self.beginDate and self.beginDate != "None":
            state_info += f": from {self.beginDate}"
        if self.endDate and self.endDate != "None":
            state_info += f" until {self.endDate}"
        if self.location and len(self.location) > 0:
            state_info += f" at {self.location}"
        return state_info

    
    def to_json(self):
        return {
            'label': self.label,
            'location': self.location,
            'beginDate': self.beginDate,
            'endDate': self.endDate,
            'description': self.description
        }

    def setBeginDate(self, date):
        self.beginDate = date
    
    def setEndDate(self, date):
        self.endDate = date
    
    def setLocation(self, location):
        self.location = location

    def setDescription(self, description):
        self.description = description
    
    def getLocation(self):
        if self.location and len(self.location) > 0:
            return self.location
        else:
            return None


def _process_dates_from_events(date_events: List[Event], method: str) -> Tuple[int, int, int]:
        """
            method: 'valid_full_dates' | 'valid_years' | 'most_likely_date'
        """

        valid_dates = set()
        for event in date_events:
            if event.date_tuple and event.date_tuple != (0, 0, 0): 
                valid_dates.add(event.date_tuple)

        if method == 'valid_full_dates':
            valid_full = set([d for d in valid_dates if d[1] != 0 and d[2] != 0])
            return list(valid_full)
        elif method == 'valid_years':
            valid_years = set([d[0] for d in valid_dates])
            return list(valid_years)
        elif method == 'most_likely_date':
            valid_years = [d[0] for d in valid_dates if d[0] > 0]
            valid_months = [d[1] for d in valid_dates if d[1] > 0]
            valid_days = [d[2] for d in valid_dates if d[2] > 0]
            if len(valid_years) > 0:
                most_repeated_year = Counter(valid_years).most_common(1)[0][0]
                if len(valid_months) > 0:
                    most_repeated_month = Counter(valid_months).most_common(1)[0][0]
                else:
                    most_repeated_month = 0
                if len(valid_days) > 0:
                    most_repeated_day = Counter(valid_days).most_common(1)[0][0]
                else:
                    most_repeated_day = 0
                my_date = (int(most_repeated_year), int(most_repeated_month), int(most_repeated_day)) # Ensemble a Full-Date with the most frequent data
                return my_date
            else:
                return (0, 0, 0)
        else:
            print(f"Invalid Date Processing Method {method}")
            raise NotImplementedError


def _get_state_info(states: List[State], method: str):
    states_str = []
    for st in states:
        if st: states_str.append(st.label.title())
    if len(states_str) == 0:
        return None
    else:
        if method == 'most_common':
            return Counter(states_str).most_common(1)[0][0]
        elif method == 'stringified_all':
            return ", ".join(states_str)
        else:
            raise NotImplementedError


def _get_century(year: int):
    'Return a Century String according to the year Int'
    century = ''
    if year == -1 or year == 0:
        century = None
    elif year < 0:
        century = 'OLD'
    elif 0 < year <= 1000:
        century = 'X (or less)'
    elif 1000 < year <= 1100:
        century = 'XI'
    elif 1100 < year <= 1200:
        century = 'XII'
    elif 1200 < year <= 1300:
        century = 'XIII'
    elif 1300 < year <= 1400:
        century = 'XIV'
    elif 1400 < year <= 1500:
        century = 'XV'
    elif 1500 < year <= 1600:
        century = 'XVI'
    elif 1600 < year <= 1700:
        century = 'XVII'
    elif 1700 < year <= 1800:
        century = 'XVIII'
    elif 1800 < year <= 1900:
        century = 'XIX'
    elif 1900 < year <= 2000:
        century = 'XX'
    elif year > 2000:
        century = 'XXI'
    else:
        century = None
    return century


def normalize_name(name: str, sep: str = " ") -> str:
    norm_name = []
    toks = name.split(" ")
    for t in toks:
        if t.lower() not in ["de", "den", "der", "en", "of", "ten", "ter", "van", "von"]:
            if all([c.upper() == c for c in t if c not in [".", ",", "-"]]):
                norm_name.append(t.strip())
            else:
                norm_name.append(t.title().strip())
        else:
            norm_name.append(t.lower().strip())
    return sep.join(norm_name).strip()

class MetadataComplete:
    '''Object that represents all available metadata for an individual. All except id number are represented as lists'''
    
    def __init__(self, idNr):
        self.person_id: str = idNr
        self.versions: List[str] = [] # This allows to 'map back' to the original source of metadata since all lists are equally ordered
        self.sources: List[str] = []
        self.names: List[str] = []
        self.partitions: List[str] = []
        self.births: List[Event] = []
        self.deaths: List[Event] = []
        self.fathers: List[str] = []
        self.mothers: List[str] = []
        self.partners: List[str] = []
        self.educations: List[State] = []
        self.occupations: List[State] = []
        self.genders: List[str] = []
        self.religions: List[str] = []
        self.faiths: List[State] = []
        self.residences: List[State] = []
        self.otherEvents: List[Event] = []
        self.otherStates: List[State] = []
        self.texts: List[str] = []
        self.texts_tokens: List[List[str]] = []
        self.texts_entities: List[List[str]] = [] # ent_list_item ~ ['Balduinus', 'LOC', 8, 17]
        self.texts_timex: List[List[Dict]] = [] # timex_dict ~ {'tid': 't3', 'type': 'DATE', 'value': '1789-08-11', 'text': '11 Aug. 1789', 'start': 48, 'end': 59}

    def __str__(self) -> str:
        s1 = f" ----- PERSON {self.person_id} -----\n\tName: {self.names}\n\tGender: {self.genders}\n\tBirth: {self.births}\n\tDeath: {self.deaths}\n\tFather: {self.fathers}"
        s2 = f"\n\tMother: {self.mothers}\n\tPartner: {self.partners}\n\tEducation: {self.educations}\n\tOccupation: {self.occupations}\n\tReligion: {self.religions}"
        s3 = f"\n\tN_TEXTS: {len(self.texts)}"
        return s1+s2+s3

    @classmethod
    def from_json(cls, info: Dict):
        person = cls(info['person_id'])
        person.versions = info['versions']
        person.sources = info['sources']
        person.names = info['names']
        person.partitions = info['partitions']
        person.births = [Event(**e) for e in info['births']]
        person.deaths = [Event(**e) for e in info['deaths']]
        person.fathers = info['fathers']
        person.mothers = info['mothers']
        person.partners = info['partners']
        person.educations = [State(**s) for s in info['educations']]
        person.occupations = [State(**s) for s in info['occupations']]
        person.genders = info['genders']
        person.religions = info['religions']
        person.faiths = [State(**s) for s in info['faiths']]
        person.residences = [State(**s) for s in info['residences']]
        person.otherEvents = [Event(**e) for e in info['otherEvents']]
        person.otherStates = [State(**s) for s in info['otherStates']]
        person.texts = info['texts']
        person.texts_tokens = info['texts_tokens']
        person.texts_entities = info['texts_entities']
        person.texts_timex = info['texts_timex']
        return person

    
    def to_json(self):
        return {
            'person_id': self.person_id,
            'versions': self.versions,
            'sources': self.sources,
            'names': self.names,
            'partitions': self.partitions,
            'births': [e.to_json() for e in self.births],
            'deaths': [e.to_json() for e in self.deaths],
            'fathers': self.fathers,
            'mothers': self.mothers,
            'partners': self.partners,
            'educations': [st.to_json() for st in self.educations],
            'occupations': [st.to_json() for st in self.occupations],
            'genders': self.genders,
            'religions': self.religions,
            'faiths': [st.to_json() for st in self.faiths],
            'residences': [st.to_json() for st in self.residences],
            'otherEvents': [e.to_json() for e in self.otherEvents],
            'otherStates': [st.to_json() for st in self.otherStates],
            'texts': self.texts,
            'texts_tokens': self.texts_tokens,
            'texts_entities': self.texts_entities,
            'texts_timex': self.texts_timex
        }

    def addName(self, name):
        self.names.append(name)
    
    def addVersion(self, version):
        self.versions.append(version)
    
    def addSource(self, source):
        self.sources.append(source)

    def addPartition(self, partition):
        self.partitions.append(partition)

    def addBirthDay(self, birthEvent):
        if birthEvent is not None:
            self.births.append(birthEvent)
    
    def addDeathDay(self, deathEvent):
        if deathEvent is not None:
            self.deaths.append(deathEvent)
    
    def addFather(self, fatherName):
        if fatherName is not None:
            self.fathers.append(fatherName)
    
    def defineMother(self, motherName):
        if motherName is not None:
            self.mothers.append(motherName)
    
    def definePartner(self, partnerName):
        if partnerName is not None:
            self.mothers.append(partnerName)
    
    def defineGender(self, gender):
        if gender is not None:
            self.genders.append(gender)
    
    def addReligion(self, religion):
        if religion is not None:
            self.religions.append(religion)
    
    def addOtherEvents(self, otherElist):
        self.otherEvents.append(otherElist)
    
    def addOtherStates(self, otherSlist):
        self.otherStates.append(otherSlist)

    def addText(self, text):
        if text is not None and len(text) > 0:
            self.texts.append(text)
    
    def addPreTokenized(self, tokens):
        if tokens is not None and len(tokens) > 0:
            self.texts_tokens.append(tokens)
    
    def addEntities(self, entities):
        self.texts_entities.append(entities)
    
    def addTimex(self, timex_list):
        self.texts_timex.append(timex_list)
    
    def addEducation(self, edu: State) -> None:
        if edu is not None:
            self.educations.append(edu)
    
    def addFaith(self, faithList: State) -> None:
        if faithList is not None:
            self.faiths.append(faithList)
    
    def addOccupation(self, occ: State) -> None:
        if occ is not None:
            self.occupations.append(occ)
    
    def addResidence(self, res: State) -> None:
        if res is not None:
            self.residences.append(res)
    
    def getName(self, mode: str = 'unique_shortest') -> Union[str, List]:
        """
        Args:
            mode (str): 'unique_shortest' | 'unique_longest' | 'all_names'

        Returns:
            Union[str, List]: String containing a unique name or a List of all possible names mentioned in the metadata
        """
        nicest_name = "UNK"
        if len(self.names) == 0: return nicest_name

        expanded_names_possibilites = []
        for name in self.names:
            if "|" in name:
                subnames = name.split("|")
            elif name != "0":
                subnames = [name]
            else:
                continue
            expanded_names_possibilites.extend(subnames)
        if len(expanded_names_possibilites) == 0: return nicest_name

        ordered_names = sorted(expanded_names_possibilites, key= lambda x: len(x)) # Shortest to Longest

        if mode == 'unique_shortest':
            nicest_name = ordered_names[0]
        elif mode == 'unique_longest':
            nicest_name = ordered_names[-1]
        elif mode == 'all_names':
            return [n for n in ordered_names if n]
        else:
            raise NotImplementedError
        
        return nicest_name
    

    def getGender(self) -> str:
        """
            method: ?
        """
        for g in self.genders:
            if g == 1.0:
                return 'male'
            elif g == 2.0:
                return 'female'
        return None

    
    def getGender_predicted_pronoun_votes(self) -> str:
        masc_votes, fem_votes = 0, 0
        masc_weights = {'hij': 1, 'hem': 1, 'broeder': 1, 'broeder van': 2, 'zn. van': 2, 'zoon van': 2}
        fem_weights = {'zij': 4, 'haar': 2, 'dochter': 2, 'dochter van': 10, 'vrouw': 2}
        masc_patterns = "|".join(masc_weights.keys())
        fem_patterns = "|".join(fem_weights.keys())
        # Get the votes in all available texts
        for text in self.texts:
            init_text = text[:200]
            masc_matches = re.finditer(masc_patterns, init_text, re.IGNORECASE)
            for m in masc_matches:
                t = m.group(0).lower()
                if t in masc_patterns:
                    masc_votes += masc_weights[t]
            fem_matches = re.finditer(fem_patterns, init_text, re.IGNORECASE)
            for m in fem_matches:
                t = m.group(0).lower()
                if t in fem_patterns:
                    fem_votes += fem_weights[t]
        # Return the most voted gender
        if fem_votes == 0 and masc_votes == 0: # Most biographies are from men, so if we know nothing then 'male' is a 'safe choice'...
            gender_str = 'male'
        elif fem_votes >= masc_votes:
            gender_str = 'female'
        else:
            gender_str = 'male'
        # print(f"{self.getName()}\t{len(self.texts)}\t{masc_votes}\t{fem_votes}\t{self.getGender()}\t{gender_str}\t{init_text}")
        return gender_str
    
    def getGender_predicted_first_pronoun(self) -> str:
        # Get the votes in all available texts
        gender_str = None
        for text in self.texts:
            init_toks = text.split()[:30]
            for tok in init_toks:
                clean_tok = tok.strip().replace(",", "").replace(";", "").replace("(", "").replace(".", "").lower()
                if clean_tok in ["hij", "hem", "zoon"]:
                    gender_str = 'male'
                    break
                elif clean_tok in ["zij", "haar", "dochter"]:
                    gender_str = 'female'
                    break
            if gender_str:
                break
        # The most likley gender is male so if there was no info then assign male
        if not gender_str:
            gender_str = 'male'
        # print(f"{self.getName()}\t{len(self.texts)}\t{self.getGender()}\t{gender_str}\t{init_toks}")
        return gender_str


    def getFather(self) -> str:
        fathers = []
        for f in self.fathers:
            if f: fathers.append(f)
        if len(fathers) == 0: return None
        return Counter(fathers).most_common(1)[0]

    def getMother(self) -> str:
        mothers = []
        for m in self.mothers:
            if m: mothers.append(m)
        if len(mothers) == 0: return None
        return Counter(mothers).most_common(1)[0]

    def getPartners(self) -> List[str]:
        partners = []
        for p in self.partners:
            if p: partners.append(p)
        return partners

    def getCenturyLived(self) -> Optional[str]:
        """This calculates the century in which a person lived according to the average between birth and death (if both are known). 
         Otherwise, if only one of the dates is known some simple heuristics are used. It Return None for both unknown dates.
         The heuristic is birth + 10 years since their life achievements coulcn't been in their early childhood, like wise death - 10 for the same reason. This is 
         to avoid 'borderline' cases: e.g. a person dying in 1901, should be classified as XIX century. A person born in 1799 should be also XIX century...
         Conflicts still exist but should be less.
        """
        birth_year = self.getBirthDate()[0]
        death_year = self.getDeathDate()[0]
        if birth_year != 0 and death_year != 0:
            year = (birth_year+death_year)/2
        elif birth_year != 0:
            year = birth_year + 10
        elif death_year != 0:
            year = death_year - 10
        else:
            year = -1
        return _get_century(year)

    def getOccupation(self, method: str = 'most_common') -> Optional[str]:
        """ method = 'most_common' | 'stringified_all' """
        return _get_state_info(self.occupations, method)
    
    def getResidence(self, method: str = 'most_common') -> Optional[str]:
        """ method = 'most_common' | 'stringified_all' """
        return _get_state_info(self.residences, method)

    def getEducation(self, method: str = 'most_common') -> Optional[str]:
        """ method = 'most_common' | 'stringified_all' """
        return _get_state_info(self.educations, method)
    
    def getReligion(self, method: str = 'most_common') -> Optional[str]:
        """ method = 'most_common' | 'stringified_all' """
        religions = []
        for rel in self.religions:
            if rel: religions.append(rel)
        if len(religions) == 0: return None
        if method == 'most_common':
            return Counter(religions).most_common(1)[0]
        else:
            return ", ".join(religions)
    
    def getFaith(self, method: str = 'most_common') -> Optional[str]:
        """ method = 'most_common' | 'stringified_all' """
        return _get_state_info(self.faiths, method)


    def getBirthDate(self, method: str = 'most_likely_date') -> Tuple[int, int, int]:
        """ Returns a Tuple(year, month, day) with the date. If it is Fully Unknown it returns None. The default tuple is (0, 0, 0)
        method (str, optional): 'all_valid_dates' | 'valid_full_dates' | 'valid_years' | 'most_likely_date' | 'stringified_all'
        """
        if method == 'stringified_all' or method == 'all_valid_dates':
            births = set()
            for bev in self.births:
                if bev and bev.date and len(bev.date) > 0:
                    births.add(str(bev))
            if method == 'stringified_all':
                return ", ".join(births)
            else:
                return births
        else:
            return _process_dates_from_events(self.births, method=method)

    
    def birthDate_is_intext(self) -> bool:
        """
            Returns True if the birth_date from metadata is explicitly inside one of the biography texts. Returns False otherwise
        """
        found_in_text = False
        years_from_metadata = [b.date_tuple for b in self.births if b.date_tuple is not None]
        years_from_metadata = [b_tup[0] for b_tup in years_from_metadata]
        for text in self.texts:
            if len(text) == 0: continue
            # Quantify how many times the metadata year is actually in the text
            for birth_year in years_from_metadata:
                if re.search(str(birth_year), text):
                    found_in_text = True
        return found_in_text

    def getBirthDate_baseline1(self) -> int:
        """
            BASELINE 1: Return The first year-like text (\d{4} or \d{3}) found in one of the biography texts
        """
        candidates = []
        birth_date = -1
        for text in self.texts:
            if len(text) == 0: continue
            if len(text) > 200: text = text[:200]
            inferred = re.search(r'\d{4}', text)
            if inferred: 
                candidates.append(int(inferred.group()))
            else:
                inferred = re.search(r'\d{3}', text)
                if inferred: 
                    candidates.append(int(inferred.group()))
        #  Get most common date found in all texts
        if len(candidates) > 0:
            proposed_dates = Counter(candidates).most_common() 
            birth_date = proposed_dates[0][0]
        return birth_date
    
    def getBirthDate_baseline2(self) -> int:
        """
            BASELINE 2: Return The first year-like text closer to the birth verbs
        """
        keywords = '|'.join(['geb.', 'geboren', 'Geb.', 'Geboren'])
        candidates = []
        birth_date = -1
        for text in self.texts:
            if len(text) == 0: continue
            # Get a Narrow window around one of the provided keywords ...
            keyword_match = re.search(keywords, text)
            chargram_window = 30
            if keyword_match:
                m_start, m_end = keyword_match.span()
                start = m_start - chargram_window if m_start > chargram_window else 0 # Maybe the window should only go forward! do not move starting point...
                end = m_end + chargram_window if m_end - chargram_window < len(text) else len(text)
                text = text[start:end]
                # Search for Year as usual ...
                inferred = re.search(r'\d{4}', text)
                if inferred: 
                    candidates.append(int(inferred.group()))
                else:
                    inferred = re.search(r'\d{3}', text)
                    if inferred: 
                        candidates.append(int(inferred.group()))
        #  Get most common date found in all texts
        if len(candidates) > 0:
            proposed_dates = Counter(candidates).most_common() 
            birth_date = proposed_dates[0][0]
        return birth_date

    def getBirthDate_from_timexps(self) -> int:
        """
            BASELINE TIMEX: Return The first HeidelTime year identified in text
        """
        birth_year = -1
        # if len(self.texts_timex) == 0: return birth_year
        first_ones = [tim[0] for tim in self.texts_timex if len(tim) > 0]
        if len(first_ones) == 0: 
            return birth_year

        candidates = []
        for first_one in first_ones:
            inferred = re.search(r'\d{4}', first_one['value'])
            if inferred:
                candidates.append(int(inferred.group()))
            else:
                inferred = re.search(r'\d{3}', first_one['value'])
                if inferred: 
                    candidates.append(int(inferred.group()))
        #  Get most common date found in all texts
        if len(candidates) > 0:
            proposed_dates = Counter(candidates).most_common() 
            birth_year = proposed_dates[0][0]
        return birth_year
    
    def getDeathDate(self, method: str = 'most_likely_date') -> str:
        """ Returns a Tuple(year, month, day) with the date. If it is Fully Unknown it returns None. The default tuple is (-1, -1, -1)
        method (str, optional):  'all_valid_dates' | 'valid_full_dates' | 'valid_years' | 'most_likely_date' | 'stringified_all'
        """
        if method == 'stringified_all' or method == 'all_valid_dates':
            deaths = set()
            for dev in self.deaths:
                if dev and dev.date and len(dev.date) > 0:
                    deaths.add(str(dev))
            if method == 'stringified_all':
                return ", ".join(deaths)
            else:
                return deaths
        else:
            return _process_dates_from_events(self.deaths, method=method)
    
    def getBirthPlace(self) -> str:
        if self.births and len(self.births) > 0:
            known_locations = [ev.getLocation() for ev in self.births if ev.getLocation()]
            if len(known_locations) > 0:
                return Counter([ev.getLocation() for ev in self.births if ev.getLocation()]).most_common(1)[0][0]
            else:
                return None
        else:
            return None

    def getDeathPlace(self) -> str:
        if self.deaths and len(self.deaths) > 0:
            known_locations = [ev.getLocation() for ev in self.deaths if ev.getLocation()]
            if len(known_locations) > 0:
                return Counter([ev.getLocation() for ev in self.deaths if ev.getLocation()]).most_common(1)[0][0]
            else:
                return None
        else:
            return None
    
    def getEntityMentions(self, entity_label: str, text_ix: int = -1) -> List[str]:
        ents = []
        if text_ix < 0:
            for ent_list in self.texts_entities:
                if ent_list:
                    for ent in ent_list:
                        if ent and ent['label'] == entity_label:
                            ents.append(ent['text'])
        elif text_ix < len(self.texts):
            if not self.texts_entities[text_ix]: return []
            for ent in self.texts_entities[text_ix]:
                if ent and ent['label'] == entity_label:
                        ents.append(ent['text'])
        return ents

    def getRelatedMetadataPlaces(self) -> List[str]:
        ' All events and states metadata can have locations associated to them. Here we return everything we find...'
        # Events: self.births, self.deaths, self.otherEvents
        # States: self.educations, self.occupations, self.faiths, self.residences, self.otherStates
        places = set()
        for ev in self.births:
            places.add(ev.getLocation())
        for ev in self.deaths:
            places.add(ev.getLocation())
        for ev in self.otherEvents:
            places.add(ev.getLocation())
        for st in self.educations:
            places.add(st.getLocation())
        for st in self.occupations:
            places.add(st.getLocation())
        for st in self.faiths:
            places.add(st.getLocation())
        for st in self.residences:
            places.add(st.getLocation())
        for st in self.otherStates:
            places.add(st.getLocation())
        places.remove(None)
        return list(places)
    
    def getFullMetadataDict(self, autocomplete=True):
        all_names_normalized = list(set([normalize_name(n) for n in self.getName("all_names")]))
        bd = "-".join([str(i) for i in self.getBirthDate()]) 
        if bd == "0-0-0": bd = None
        dd = "-".join([str(i) for i in self.getDeathDate()])
        if dd == "0-0-0": dd = None
        norm_name_main = normalize_name(self.getName(), sep="_")
        metadata_facts = {
            'person_id': self.person_id,
            'name': norm_name_main,
            'names_all': all_names_normalized + [norm_name_main],
            'birth_date': bd,
            'birth_place': self.getBirthPlace(),
            'death_date': dd,
            'death_place': self.getDeathPlace(),
            'fathers': self.getFather(),
            'mothers': self.getMother(),
            'partners': self.getPartners(),
            'education': self.getEducation('stringified_all'),
            'occupation': self.getOccupation('stringified_all'),
            'gender': self.getGender(),
            'religion': self.getReligion('stringified_all'),
            'faith': self.getFaith('stringified_all'),
            'residence': self.getResidence('stringified_all'),
        }
        if autocomplete:
            metadata_facts['birth_year_pred'] = self.getBirthDate_baseline1()
            metadata_facts['gender_pred'] = self.getGender_predicted_first_pronoun()
            metadata_facts['century_pred'] = self.getCenturyLived()
        
        return metadata_facts



# This class is used to map to DataFrame and do Filtered Queries based on Enriched data 
# The enrichment can be obtained with Metadata, Automatic Heuristics and NLP methods and should be stated when instanciating the object
# The class stays lean (only strings, and numbers) to make fast queries in the dataframe
class EnrichedText(NamedTuple):
    person_id: str
    text_id: str
    person_name: str
    source: str
    birth_date: Tuple[int, int, int] # [YYYY, MM, DD]
    death_date: Tuple[int, int, int] # [YYYY, MM, DD]
    century: str
    birth_place: str
    death_place: str
    occupations: str # joins a List[str] into a long str that can be searched by pandas
    places: str # joins a List[str] into a long str that can be searched by pandas
    person_mentions: str # joins a List[str] into a long str that can be searched by pandas
    place_mentions: str # joins a List[str] into a long str that can be searched by pandas
    text: str