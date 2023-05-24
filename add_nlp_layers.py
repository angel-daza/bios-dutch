from typing import List,Dict, Any, Tuple
import glob, json
from utils.nlp_tasks import run_flair
import argparse

from flair import __version__ as flair_version
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

from pymongo import MongoClient
COLLECTION_NAME = f"bionet_intavia"
DB_NAME = "biographies"


def main_bionet_intavia_files(nlp_config: Dict[str, Any]):
    JSON_BASEPATH = "flask_app/backend_data/intavia_json/*"
    # Iterate through all files and add the Requested <apply_tasks>
    for src_path in glob.glob(JSON_BASEPATH):
        for filepath in glob.glob(f"{src_path}/*.json"):
            print(filepath)
            intavia_obj = json.load(open(filepath))
            included_models = set([ent['method'] for ent in intavia_obj['data'].get('entities', [])])
            if "gold_ner" in nlp_config and "human_gold" not in included_models:
                gold_obj = nlp_config["gold_ner"]["annotations_human"].get(intavia_obj['text_id'])
                if gold_obj:
                    print("Adding Gold")
                    intavia_obj = add_bionet_gold_ner(intavia_obj, gold_obj)
            if "flair_ner" in nlp_config and not any(["flair" in m for m in included_models]):
                print("Adding Flair")
                flair_model = nlp_config["flair_ner"]["flair_model"]
                flair_tagger = nlp_config["flair_ner"]["flair_tagger"]
                flair_splitter = nlp_config["flair_ner"]["flair_splitter"]
                intavia_obj = add_flair_ner(intavia_obj, flair_model, flair_tagger, flair_splitter)
            # Override File with New Object
            json.dump(intavia_obj, open(filepath, "w"), indent=2, ensure_ascii=False)


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
        doc_offset = 0
        for i, sent_objs in enumerate(flair_output['tagged_ner']):
            for ner_obj in sent_objs:
                ner_all.append({'ID': f"flair_{i}", 
                        'category': ner_obj['entity'], 
                        'surfaceForm': ner_obj['text'], 
                        'locationStart': doc_offset + ner_obj['start'], 
                        'locationEnd': doc_offset + ner_obj['end'], 
                        'method': f'{flair_model}_{flair_version}'
                        })
            doc_offset += len(flair_output['sentences'][i]) + 1
        return ner_all

    flair_nlp = run_flair(intavia_obj["data"]["text"], flair_tagger, flair_splitter)
    flair_ents = _add_json_flair_ner(flair_nlp)
    if len(intavia_obj['data'].get('entities', [])) > 0:
        intavia_obj['data']['entities'] += flair_ents
    else:
        intavia_obj['data']['entities'] = flair_ents
    intavia_obj['data']['entities'] = sorted(intavia_obj['data']['entities'], key=lambda x: x['locationStart'])
    return intavia_obj


if __name__ == "__main__":
    """
        Running Examples:

            python add_nlp_layers.py --mode files --gold_ner --flair_ner

            python add_nlp_layers.py --mode mongo --gold_ner

    """

    NLP_CONFIG = {}

    # Read arguments from command line
    parser = argparse.ArgumentParser()

    # GENERAL SYSTEM PARAMS
    parser.add_argument('-m', '--mode', help='Mode: mongo | files ', required=True)
    parser.add_argument('-gn', '--gold_ner', help='Filepath containing the Training JSON', action='store_true', default=False)
    parser.add_argument('-fn', '--flair_ner', help='Filepath containing the Validation JSON', action='store_true', default=False)

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
    if args.flair_ner:
        flair_model = "flair/ner-dutch-large"
        NLP_CONFIG["flair_ner"] = {
            "model_config_label": "Flair NER (PER, ORG, LOC, MISC)",
            "flair_model": flair_model,
            "flair_version": flair_version,
            "flair_tagger": SequenceTagger.load(flair_model),
            "flair_splitter": SegtokSentenceSplitter()
        }

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