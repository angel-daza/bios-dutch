from dataclasses import dataclass
import json, os
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, TypeVar, Union
import logging
logger = logging.getLogger(__name__)

from flair.data import Sentence
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter

def flair2bio(flair_obj: Dict) -> List[List[str]]:
    sent_entities = flair_obj["tagged_ner"]
    sentences = flair_obj["sentences"]
    doc_bio_labels = []

    for entities, sentence in zip(sent_entities, sentences):
        sentence = sentence.split()
        sent_bio = ["O"]*len(sentence)
        # Now translate the Entity spans into BIO Notation
        print(entities)
        for ent in entities:
            token_start = ent['start_token']
            token_end =  ent['end_token']
            label = ent['entity']
            for j in range(token_start, token_end):
                if j == token_start:
                    sent_bio[j] = f"B-{label}"
                else:
                    sent_bio[j] = f"I-{label}"
        doc_bio_labels.append(sent_bio)
    return doc_bio_labels


def run_flair(text: Union[str, List[str]], tagger: SequenceTagger, splitter: SegtokSentenceSplitter = None) -> Dict:
    if splitter:
        sentences = splitter.split(text)
        tagger.predict(sentences)
        texts, ner, offsets = [], [], []
        for sentence in sentences:
            tagged_ents = []
            for entity in sentence.get_spans('ner'):
                token_indices = [t.idx for t in entity]
                tagged_ents.append({"text": entity.text, "start": entity.start_position, "end": entity.end_position, "entity": entity.get_label("ner").value, 
                                "start_token": token_indices[0]-1, "end_token": token_indices[-1], "score": entity.get_label("ner").score})
            ner.append(tagged_ents)
            texts.append(sentence.to_tokenized_string())
            offsets.append(len(sentence.to_plain_string()))
        return {'tagged_ner':  ner, 'sentences': texts, 'offsets': offsets}
    else:
        sentence = Sentence(text)
        tagger.predict(sentence)
        tagged_ents = []
        for entity in sentence.get_spans('ner'):
            token_indices = [t.idx for t in entity]
            tagged_ents.append({"text": entity.text, "start": entity.start_position, "end": entity.end_position, "entity": entity.get_label("ner").value, 
                                "start_token": token_indices[0]-1, "end_token": token_indices[-1], "score": entity.get_label("ner").score})
        return {'tagged_ner': [tagged_ents], 'sentences': [sentence.to_tokenized_string()], 'offsets': [0]}


def run_bert_ner(bert_nlp, stanza_nlp, text, wordpiece_chars):
    doc = stanza_nlp(text)
    texts, ner, offsets = [], [], []
    for stanza_sent in doc.sentences:
        tagged_ents = []
        sentence = stanza_sent.text #" ".join([tok.text for tok in stanza_sent.tokens])
        # print(sentence)
        if len(sentence) > 1:
            predictions = bert_nlp(sentence)
            tagged_ents = unify_wordpiece_predictions(predictions, wordpiece_chars)
            ner.append(tagged_ents)
            texts.append(sentence)
            offsets.append(len(stanza_sent.text))
    return {'tagged_ner':  ner, 'sentences': texts, 'offsets': offsets}


def match_proper_names(matcher, nlp_doc, text):
    matches = matcher(nlp_doc)
    spans_matched = []
    ix = 0
    spans = []

    for match_id, start, end in matches:
        start_char = nlp_doc[start].idx
        end_char = nlp_doc[end-1].idx + len(nlp_doc[end-1].text)
        surface_form = text[start_char:end_char]
        spans_matched.append({
            "ID": f"spacy_matcher_nl_{ix}",
            "surfaceForm": surface_form,
            "category": "PER",
            "locationStart": start_char,
            "locationEnd": end_char,
            "method": "spacy_matcher_nl"
        })
        ix += 1
    
    return spans_matched


def unify_wordpiece_predictions(prediction_list: List, wordpiece_chars: str) -> List:
    """
     This function is written to fix models that return predictions as:
     Also looking to unify for the visualization tool build on top!
        EXAMPLE: SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
        PREDS:    [{'end': 14, 'entity': 'B-LOC', 'index': 5, 'score': 0.9960225820541382, 'start': 8, 'word': '▁JAPAN'}
                    {'end': 33, 'entity': 'B-LOC', 'index': 15, 'score': 0.9985975623130798, 'start': 30, 'word': '▁CH'}
                    {'end': 36, 'entity': 'B-LOC', 'index': 16, 'score': 0.9762864708900452, 'start': 33, 'word': 'INA'}]
    """

    def _merge_objs(obj_list):
        merged_word = "".join([o['word'].replace(wordpiece_chars, '') for o in obj_list])
        if wordpiece_chars == "▁":
            real_start = obj_list[0]['start'] + 1 # The +1 is to avoid the underscore
        else:
            real_start = obj_list[0]['start']
        if real_start == 1: real_start = 0 # For some reason the first underscore is not counted...
        real_end = obj_list[-1]['end']
        real_entity = obj_list[0]['entity']
        scores = [o['score'] for o in obj_list]
        real_score = sum(scores) / len(scores)
        return {'start': real_start, 'end': real_end, 'entity': real_entity, 'score': real_score, 'text': merged_word}


    if len(prediction_list) == 0: return []
    unified_predictions= []
    tmp_unif = []

    if wordpiece_chars == "▁":
        for pred_obj in sorted(prediction_list, key=lambda x: x['index']):
            if pred_obj['word'].startswith('▁'):
                if len(tmp_unif) > 0:
                    unified_predictions.append(_merge_objs(tmp_unif)) 
                    tmp_unif = []
                tmp_unif.append(pred_obj)
            else:
                tmp_unif.append(pred_obj)
            # print(pred_obj)
        if len(tmp_unif) > 0: unified_predictions.append(_merge_objs(tmp_unif))
        # print("\nUNIFIED:")
        # [print(x) for x in unified_predictions]
    else:
        ordered_preds = sorted(prediction_list, key=lambda x: x['index'])
        head_indices = [ix for ix, pred in enumerate(ordered_preds) if not pred['word'].startswith(wordpiece_chars)]
        for ix, pred_obj in enumerate(ordered_preds):
            if ix > 0 and ix in head_indices:
                if len(tmp_unif) > 0:
                    unified_predictions.append(_merge_objs(tmp_unif)) 
                    tmp_unif = []
                tmp_unif.append(pred_obj)
            else:
                tmp_unif.append(pred_obj)
            # print(pred_obj)
        if len(tmp_unif) > 0:
            unified_predictions.append(_merge_objs(tmp_unif))
        # print("\nUNIFIED:")
        # [print(x) for x in unified_predictions]
    
    #     # In this step we further unify this time the IOB into FULL-LABEL
    full_labeled = []
    label, tmp_entity = "", []
    entity_head_indices = [ix for ix, pred in enumerate(unified_predictions) if pred['entity'].startswith("B-")]
    for ix, pred in enumerate(unified_predictions):
        if ix in entity_head_indices:
            if len(tmp_entity) > 0:
                text = " ".join([e['text'] for e in tmp_entity])
                full_labeled.append({'text': text, 'entity': label, 'start': tmp_entity[0]['start'], 'end': tmp_entity[-1]['end']})
                tmp_entity = []
            label = pred['entity'][2:]
            tmp_entity.append(pred)
        else:
            tmp_entity.append(pred)

    if len(tmp_entity) > 0:
        text = " ".join([e['text'] for e in tmp_entity])
        full_labeled.append({'text': text, 'entity': label, 'start': tmp_entity[0]['start'], 'end': tmp_entity[-1]['end']})

    # print("\nMORE UNIFIED:")
    # [print(x) for x in full_labeled]

    return full_labeled


def unify_wordpiece_predictions_iob(prediction_list: List, wordpiece_chars: str, tokens: List[str]) -> List:
    """
     This function is written to fix models that return predictions as:
     Also looking to unify for the visualization tool build on top!
        EXAMPLE: SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .
        PREDS:    [{'end': 14, 'entity': 'B-LOC', 'index': 5, 'score': 0.9960225820541382, 'start': 8, 'word': '▁JAPAN'}
                    {'end': 33, 'entity': 'B-LOC', 'index': 15, 'score': 0.9985975623130798, 'start': 30, 'word': '▁CH'}
                    {'end': 36, 'entity': 'B-LOC', 'index': 16, 'score': 0.9762864708900452, 'start': 33, 'word': 'INA'}]
    """

    def _merge_objs(obj_list, tok_index):
        merged_word = "".join([o['word'].replace(wordpiece_chars, '') for o in obj_list])
        real_start = tok_index
        real_end = tok_index + len(obj_list)
        real_entity = obj_list[0]['entity']
        scores = [o['score'] for o in obj_list]
        real_score = sum(scores) / len(scores)
        return {'start': real_start, 'end': real_end, 'entity': real_entity, 'score': real_score, 'text': merged_word}


    full_iob_labeled = ['O' for tok in tokens]

    if len(prediction_list) == 0: return full_iob_labeled
    unified_predictions= []
    tmp_unif = []

    unified_token_index = 0
    for pred_obj in sorted(prediction_list, key=lambda x: x['index']):
        if pred_obj['word'].startswith('▁'):
            if len(tmp_unif) > 0:
                unified_predictions.append(_merge_objs(tmp_unif, unified_token_index)) 
                tmp_unif = []
                unified_token_index += 1
            tmp_unif.append(pred_obj)
        else:
            tmp_unif.append(pred_obj)
        # print(pred_obj)
    if len(tmp_unif) > 0: unified_predictions.append(_merge_objs(tmp_unif, unified_token_index))
    # print("\nUNIFIED:")
    # [print(x) for x in unified_predictions]

    for pred in unified_predictions:
        ent_start, ent_end = pred['start'], pred['end']
        label = pred['entity'][2:]
        for ix in range(ent_start, ent_end):
            if ix == ent_start:
                full_iob_labeled[ix] = f"B-{label}"
            else:
                full_iob_labeled[ix] = f"I-{label}"

    print(unified_predictions)

    return full_iob_labeled