from typing import List,Dict, Any, Tuple
import glob, json, os
import argparse
import pandas as pd
from collections import Counter

import spacy
from spacy.matcher import Matcher
from spacy import __version__ as spacy_version

from flair import __version__ as flair_version
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter

from utils.nlp_tasks import run_flair, match_proper_names

import stanza
from utils.nlp_tasks import run_bert_ner
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

from utils_general import INTAVIA_JSON_ROOT
INTAVIA_JSON_ROOT = f"/Users/Daza/intavia_json_v1_all"

from pymongo import MongoClient
COLLECTION_NAME = f"bionet_intavia"
DB_NAME = "biographies"


# A cheat to add layers only on biographies present in the Test Set
gold_paths = ["data/bionet_gold/biographynet_test_A_gold.json",
                  "data/bionet_gold/biographynet_test_B_gold.json", 
                  "data/bionet_gold/biographynet_test_C_gold.json"]
gold_docs = {}
for gold_path in gold_paths:
    gold_docs.update(json.load(open(gold_path)))


def main_bionet_intavia_files(nlp_config: Dict[str, Any]):
    # Iterate through all files and add the Requested <apply_tasks>
    ctr = 0
    for src_path in glob.glob(f"{INTAVIA_JSON_ROOT}/*"):
        for filepath in glob.glob(f"{src_path}/*.json"):
            bio_id = os.path.basename(filepath).strip(".json")
            # if bio_id not in gold_docs: continue # TODO: This is only A TRICK to only add for the Test Set. Should be disabled or parametrized!
            print(f"[{ctr}] {filepath}")
            intavia_obj = json.load(open(filepath))
            included_models = set([ent['method'] for ent in intavia_obj['data'].get('entities', [])])
            if "gold_ner" in nlp_config and "human_gold" not in included_models:
                gold_obj = nlp_config["gold_ner"]["annotations_human"].get(intavia_obj['text_id'])
                if gold_obj:
                    print("Adding Gold")
                    intavia_obj = add_bionet_gold_ner(intavia_obj, gold_obj)
            if "spacy_matcher" in nlp_config:
                print("Adding Spacy Matcher")
                ents = [e for e in intavia_obj['data']['entities'] if e["method"] != "spacy_matcher_nl"]
                intavia_obj['data']['entities'] = ents
                intavia_obj = add_spacy_matcher(nlp_config["spacy_matcher"]["matcher"], nlp_config["spacy_matcher"]["nlp"], intavia_obj)
            if "flair_ner" in nlp_config and not any(["flair" in m for m in included_models]):
                print("Adding Flair")
                flair_model = nlp_config["flair_ner"]["flair_model"]
                flair_tagger = nlp_config["flair_ner"]["flair_tagger"]
                flair_splitter = nlp_config["flair_ner"]["flair_splitter"]
                intavia_obj = add_flair_ner(intavia_obj, flair_model, flair_tagger, flair_splitter)
            if "bert_ner" in nlp_config:
                model_label = nlp_config["bert_ner"]["model_json_label"]
                print(f"Adding {model_label}")
                model_version = nlp_config["bert_ner"]["model_json_version"]
                stanza_nlp = nlp_config["bert_ner"]["stanza"]
                bert_nlp = nlp_config["bert_ner"]["bert_pipeline"]
                wordpiece_chars= nlp_config["bert_ner"]["wordpiece_chars"]
                intavia_obj = add_bert_based_ner(intavia_obj, stanza_nlp, bert_nlp, model_label, model_version, wordpiece_chars)
            if "chatgpt_ner" in nlp_config and "gpt-3.5-turbo" not in included_models:
                gpt_outputs = nlp_config["chatgpt_ner"]["model_outputs"]
                model_label = nlp_config["chatgpt_ner"]["model_json_label"]
                if intavia_obj["text_id"] in gpt_outputs:
                    print("Adding GPT3.5 NER")
                    intavia_obj = add_chatgpt_ner(intavia_obj, gpt_outputs[intavia_obj["text_id"]], model_label)
            # Override File with New Object
            json.dump(intavia_obj, open(filepath, "w"), indent=2, ensure_ascii=False)
            ctr += 1

def main_bionet_intavia_mongo(nlp_config: Dict[str, Any]):
    client = MongoClient("mongodb://localhost:27017/")
    db = client[DB_NAME]
    bionet_collection = db[COLLECTION_NAME]
    batched_docs, batched_ids, batch_size = [], [], 10

    original_count = bionet_collection.count_documents({})
    print(f"Initial MongoDB has {original_count} items in Collection: {COLLECTION_NAME}. DataBase: {DB_NAME}")

    # Iterate through all files and add the Requested <apply_tasks>
    for ix, intavia_obj in enumerate(bionet_collection.find({})):
        included_models = set([ent['method'] for ent in intavia_obj['data'].get('entities', [])])
        print(f"{intavia_obj['text_id']} [{ix}]")

        if "gold_ner" in nlp_config and "human_gold" not in included_models:
            gold_obj = nlp_config["gold_ner"]["annotations_human"].get(intavia_obj['text_id'])
            if gold_obj:
                print(f"Adding Gold [{ix}]")
                intavia_obj = add_bionet_gold_ner(intavia_obj, gold_obj)
        if "flair_ner" in nlp_config and not any(["flair" in m for m in included_models]):
            print(f"Adding Flair [{ix}]")
            flair_model = nlp_config["flair_ner"]["flair_model"]
            flair_tagger = nlp_config["flair_ner"]["flair_tagger"]
            flair_splitter = nlp_config["flair_ner"]["flair_splitter"]
            intavia_obj = add_flair_ner(intavia_obj, flair_model, flair_tagger, flair_splitter)
        if "bert_ner" in nlp_config:
            model_label = nlp_config["bert_ner"]["model_json_label"]
            print(f"Adding {model_label}")
            model_version = nlp_config["bert_ner"]["model_json_version"]
            stanza_nlp = nlp_config["bert_ner"]["stanza"]
            bert_nlp = nlp_config["bert_ner"]["bert_pipeline"]
            wordpiece_chars= nlp_config["bert_ner"]["wordpiece_chars"]
            intavia_obj = add_bert_based_ner(intavia_obj, stanza_nlp, bert_nlp, model_label, model_version, wordpiece_chars)
        # Accumulate batch with NEW objects
        batched_docs.append(intavia_obj)
        batched_ids.append(intavia_obj['_id'])
        # Override Batched Records with New Objects
        if ix > 0 and ix % batch_size == 0:
            d = bionet_collection.delete_many({'_id': {'$in': batched_ids}})
            i = bionet_collection.insert_many(batched_docs)
            # print(f"Deleted {d.deleted_count} docs!\n[{batched_ids}]\nInserted:\n{i.inserted_ids}")
            batched_docs = []
            batched_ids = []
        if ix == original_count: break
    
    if len(batched_ids) > 0:
        bionet_collection.delete_many({'_id': {'$in': batched_ids}})
        bionet_collection.insert_many(batched_docs)

    print(f"Final MongoDB has {bionet_collection.count_documents({})} items in Collection: {COLLECTION_NAME}. DataBase: {DB_NAME}")


def load_gold_annotations(gold_paths: List[str]) -> Dict[str, Dict]:
    gold_docs = {}
    for gold_path in gold_paths:
        gold_docs.update(json.load(open(gold_path)))
        print(len(gold_docs))
    return gold_docs

def add_bionet_gold_ner(intavia_obj: Dict[str, Any], gold_obj: Dict[str, Any]) -> Dict[str, Any]:
    for ent in gold_obj['entities']:
        intavia_obj['data']['entities'].append(ent)
    intavia_obj['data']['entities'] = sorted(intavia_obj['data']['entities'], key=lambda x: x['locationStart'])
    return intavia_obj


def add_flair_ner(intavia_obj, flair_model, flair_tagger, flair_splitter) -> Dict[str, Any]:

    def _add_json_flair_ner(flair_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        ner_all = []
        doc_offset, doc_tok_offset = 0, 0
        for i, sent_objs in enumerate(flair_output['tagged_ner']):
            for ner_obj in sent_objs:
                ner_all.append({'ID': f"flair_{i}", 
                        'category': ner_obj['entity'], 
                        'surfaceForm': ner_obj['text'], 
                        'locationStart': doc_offset + ner_obj['start'], 
                        'locationEnd': doc_offset + ner_obj['end'],
                        # 'tokenStart': doc_tok_offset + ner_obj['start_token'], # These tokens ar enot the same as the MAIN TOKENS so why calculate them?
                        # 'tokenEnd': doc_tok_offset + ner_obj['end_token'],
                        'method': f'{flair_model}_{flair_version}'
                        })
            doc_tok_offset += len(flair_output['sentences'][i].split())
            doc_offset += flair_output['offsets'][i] + 1
        return ner_all

    flair_nlp = run_flair(intavia_obj["data"]["text"], flair_tagger, flair_splitter)
    flair_ents = _add_json_flair_ner(flair_nlp)
    if len(intavia_obj['data'].get('entities', [])) > 0:
        intavia_obj['data']['entities'] += flair_ents
    else:
        intavia_obj['data']['entities'] = flair_ents
    intavia_obj['data']['entities'] = sorted(intavia_obj['data']['entities'], key=lambda x: x['locationStart'])
    return intavia_obj


def add_bert_based_ner(intavia_obj: Dict[str, Any], stanza_nlp: Any, bert_nlp: str, model_label: str, model_version: str, wordpiece_chars: str):

    def _add_json_bert_ner(bert_ner_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        ner_all = []
        doc_offset, doc_tok_offset = 0, 0
        for i, sent_objs in enumerate(bert_ner_output['tagged_ner']):
            for ner_obj in sent_objs:
                ner_all.append({'ID': f"{model_label}_{i}", 
                        'category': ner_obj['entity'], 
                        'surfaceForm': ner_obj['text'], 
                        'locationStart': doc_offset + ner_obj['start'], 
                        'locationEnd': doc_offset + ner_obj['end'],
                        'method': f'{model_label}_{model_version}'
                        })
            doc_tok_offset += len(bert_ner_output['sentences'][i].split())
            doc_offset += bert_ner_output['offsets'][i] + 1
        return ner_all

    # Process NER with a BERT-based model (only if needed)
    bert_nlp = run_bert_ner(bert_nlp, stanza_nlp, intavia_obj["data"]["text"], wordpiece_chars)

    bert_ents = _add_json_bert_ner(bert_nlp)
    if len(intavia_obj['data'].get('entities', [])) > 0:
        intavia_obj['data']['entities'] += bert_ents
    else:
        intavia_obj['data']['entities'] = bert_ents
    intavia_obj['data']['entities'] = sorted(intavia_obj['data']['entities'], key=lambda x: x['locationStart'])
    
    return intavia_obj


def add_chatgpt_ner(intavia_obj, gpt_outputs, model_label):
    gpt_ents = []
    for i, ner_obj in enumerate(gpt_outputs):
        gpt_ents.append({'ID': f"{model_label}_{i}", 
                        'category': ner_obj['Label'], 
                        'surfaceForm': ner_obj['Entity'], 
                        'locationStart': ner_obj['Span Start'], 
                        'locationEnd': ner_obj['Span End'],
                        'method': f'{model_label}'
                        })
    intavia_obj['data']['entities'] += gpt_ents
    return intavia_obj

def read_gpt_content(filepath: str, valid_labels: List[str]) -> List[Dict[str, Any]]:
    data = []
    with open(filepath) as f:
        for ix, line in enumerate(f):
            if ix == 0: continue
            elems = line.split("\t")
            if len(elems) > 3:
                label = elems[1].upper()
                try:
                    start = int(elems[2])
                    end = int(elems[3])
                except:
                    start, end = -1, -1
                if label in valid_labels and start >= 0:
                    row = {
                        "Entity": elems[0],	
                        "Label": label,
                        "Span Start": start,	
                        "Span End": end
                    }
                    data.append(row)
                # else:
                #     row = {"Label": "ERR"}
                #     print("CHECHENTON",filepath, ix, elems)
                #     data.append(row)
                    
    return data

def add_spacy_matcher(matcher, nlp, intavia_obj: Dict[str, Any]):
    nlp_doc = nlp(intavia_obj["data"]["text"])
    match_ents = match_proper_names(matcher, nlp_doc, intavia_obj["data"]["text"])
    if len(intavia_obj['data'].get('entities', [])) > 0:
        intavia_obj['data']['entities'] += match_ents
    else:
        intavia_obj['data']['entities'] = match_ents
    intavia_obj['data']['entities'] = sorted(intavia_obj['data']['entities'], key=lambda x: x['locationStart'])
    return intavia_obj


if __name__ == "__main__":
    """
        Running Examples:

            python add_nlp_layers.py --mode files --gold_ner --flair_ner --gysbert_ner 

            python add_nlp_layers.py --mode mongo --gold_ner

            python add_nlp_layers.py --mode files --gysbert_ner 
            python add_nlp_layers.py --mode files --chatgpt_ner 
            python add_nlp_layers.py --mode files --xlm_roberta_ner
            python add_nlp_layers.py --mode files --spacy_matcher

    """

    NLP_CONFIG = {}

    # Read arguments from command line
    parser = argparse.ArgumentParser()

    # GENERAL SYSTEM PARAMS
    parser.add_argument('-m', '--mode', help='Mode: mongo | files ', required=True)
    parser.add_argument('-gn', '--gold_ner', help='', action='store_true', default=False)
    parser.add_argument('-fn', '--flair_ner', help='', action='store_true', default=False)
    parser.add_argument('-bn', '--gysbert_ner', help='', action='store_true', default=False)
    parser.add_argument('-xlmr', '--xlm_roberta_ner', help='', action='store_true', default=False)
    parser.add_argument('-gpt', '--chatgpt_ner', help='', action='store_true', default=False)
    parser.add_argument('-sp', '--spacy_matcher', help='', action='store_true', default=False)

    args = parser.parse_args()

    # Load Necessary Models according to the Requested Tasks
    if args.gold_ner:
        gold_paths = ["data/bionet_gold/biographynet_test_A_gold.json",
                        "data/bionet_gold/biographynet_test_B_gold.json", 
                        "data/bionet_gold/biographynet_test_C_gold.json"]
        NLP_CONFIG["gold_ner"] = {
            "model_config_label": "Human Annotations NER",
            "annotations_human": load_gold_annotations(gold_paths)
        }
    if args.spacy_matcher:
        nlp = spacy.load("nl_core_news_lg")
        matcher = Matcher(nlp.vocab)
        pattern = [[{'IS_TITLE': True, 'OP': '+'}, {'IS_LOWER': True, 'OP': '?'}, {'IS_LOWER': True, 'OP': '?'}, {'IS_TITLE': True, 'OP': '+'}],
                   [{'IS_TITLE': True, 'OP': '+'}, {'IS_LOWER': True, 'OP': '?'}, {'IS_LOWER': True, 'OP': '?'}, {'IS_UPPER': True, 'OP': '+'}],
                   ]
        pattern_id = "proper_names_greedy"
        matcher.add(pattern_id, pattern)
        NLP_CONFIG["spacy_matcher"] = {
            "model_config_label": "Spacy Matcher",
            "matcher": matcher,
            "nlp": nlp,
            "model": "nl_core_news_lg",
            "version": spacy_version 
        }
    if args.flair_ner:
        flair_model = "flair/ner-dutch-large"
        flair_tagger = SequenceTagger.load(flair_model)
        flair_splitter = SegtokSentenceSplitter()
        NLP_CONFIG["flair_ner"] = {
            "model_config_label": "Flair NER (PER, ORG, LOC, MISC)",
            "flair_model": flair_model,
            "flair_version": flair_version,
            "flair_tagger": flair_tagger,
            "flair_splitter": flair_splitter
        }
    if args.gysbert_ner:
        gysbert_checkpoint = "/Users/daza/gysbert_saved_models/EPOCH_2"
        bert_tokenizer = AutoTokenizer.from_pretrained(gysbert_checkpoint)
        bert_model = AutoModelForTokenClassification.from_pretrained(gysbert_checkpoint)
        stanza_nlp = stanza.Pipeline(lang="nl", processors="tokenize,lemma,pos,depparse", model_dir="/Users/daza/stanza_resources/")
        bert_nlp = pipeline('token-classification', model=bert_model, tokenizer=bert_tokenizer, device=-1)
        NLP_CONFIG["bert_ner"] = {
            "model_config_label": "GysBERT NER",
            "model_json_label": "gysbert_hist",
            "model_json_version": "fx_finetuned_epoch2",
            "stanza": stanza_nlp,
            "bert_pipeline": bert_nlp,
            "wordpiece_chars": "##"
        }
    if args.xlm_roberta_ner:
        checkpoint = "Davlan/xlm-roberta-base-ner-hrl"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
        stanza_nlp = stanza.Pipeline(lang="nl", processors="tokenize,lemma,pos,depparse", model_dir="/Users/daza/stanza_resources/")
        bert_nlp = pipeline('token-classification', model=model, tokenizer=tokenizer, device=-1)
        NLP_CONFIG["bert_ner"] = {
            "model_config_label": "XLM RoBERTa NER",
            "model_json_label": "xlmr_ner",
            "model_json_version": "",
            "stanza": stanza_nlp,
            "bert_pipeline": bert_nlp,
            "wordpiece_chars": '▁'
                }
    if args.chatgpt_ner:
        # For security, this mode sssumes the model predictions were already run.
        # Here we only read the outputs as they were saved in the disk
        NLP_CONFIG["chatgpt_ner"] = {
            "gpt_outputs_dir": "data/gpt-3/test_set_structured",
            "model_json_label": "gpt-3.5-turbo"
        }
        err = 0
        gpt_outputs = {}
        label_dist = []
        for filepath in glob.glob(f"{NLP_CONFIG['chatgpt_ner']['gpt_outputs_dir']}/*.tsv"):
            bio_id = os.path.basename(filepath).split(".")[0]
            try:
                gpt_outputs[bio_id] = read_gpt_content(filepath, valid_labels=['PERSON', 'LOCATION', 'ORGANIZATION', 'TIME', 'ARTWORK', 'DATE', 'MISC', 'NUMBER'])
                for row in gpt_outputs[bio_id]:
                    label_dist.append(row["Label"])
            except:
                print(f"Pandas CSV READ ERROR [{err}]: {filepath}")
                err += 1
                gpt_outputs[bio_id] = []
        NLP_CONFIG["chatgpt_ner"]["model_outputs"] = gpt_outputs
        print(Counter(label_dist))

    # More BERT-BASED
    # "surferfelix/ner-bertje-tagdetekst", "GroNLP/bert-base-dutch-cased", "bertje_hist"
    # RoBERTa NER! (see performancer code)

    if len(NLP_CONFIG) == 0:
        print("\nNo models were requested. Please check the valid flags to request models\\n")
    else:
        user_continue = input(f"\nThe following models were requested and will be applied to the IntaVia Files and overwite them:\n\n{list(NLP_CONFIG.keys())}\n\nAre you sure[y/N]? ")

        if user_continue.lower() == "y":
            if args.mode == "files":
                main_bionet_intavia_files(NLP_CONFIG)
            elif args.mode == "mongo":
                main_bionet_intavia_mongo(NLP_CONFIG)
        else:
            print("\nNo changes were applied\n")