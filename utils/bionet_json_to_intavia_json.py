"""
    EXAMPLE RUN:
        python3 utils/bionet_json_to_intavia_json.py files "data/seed_data/AllBios.jsonl" "flask_app/backend_data/intavia_json" 
        python3 utils/bionet_json_to_intavia_json.py files "data/seed_data/biographynet_test.jsonl" "flask_app/backend_data/test/intavia_json_v2"
"""

import json, os, sys
from typing import Dict, Any, List, TypeVar, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from classes import IntaviaToken

# IF USING MONGO (more robust, but it is not required):
# START MONGO IN MAC: mongod --config /usr/local/etc/mongod.conf
# START MONGO IN UBUNTU: sudo systemctl start mongod
# https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/
import pymongo
from pymongo import MongoClient
MongoCollection = TypeVar("MongoCollection")

COLLECTION_NAME = f"bionet_intavia"
DB_NAME = "biographies"

metadata_keys = ['id_person', 'version', 'source', 'name', 'partition', 'birth_pl', 'birth_tm', 'baptism_pl', 'baptism_tm', 'death_pl', 'death_tm', 'funeral_pl', 'funeral_tm', 
                            'marriage_pl', 'marriage_tm', 'gender', 'category', 'father', 'mother', 'partner', 'religion', 'educations', 'faiths', 'occupations', 'residences']
nlp_keys = ['text_clean', 'text_original', 'text_tokens', 'text_token_objects', 'text_sentences', 'text_entities', 'text_timex']

sources = ['bioport', 'raa', 'rkdartists', 'dvn', 'vdaa', 'weyerman', 'bwsa', 'bwsa_archives', 'dbnl', 'nnbw', 'rijksmuseum', 
                    'pdc', 'blnp', 'knaw', 'wikipedia', 'nbwv', 'schilderkunst', 'portraits', 'glasius', 'schouburg', 
                    'smoelenboek', 'na', 'bwn', 'IAV_MISC-biographie', 'jews', 'bwg', 'bwn_1780tot1830', 'elias']


def main(bionet_json:str, output_mode: str, output_root: str):
    if output_mode == "mongo":
        convert_to_intavia_mongo(bionet_filepath=bionet_json)
    elif output_mode == "files":
        convert_to_intavia_files(bionet_filepath=bionet_json, output_path=output_root)
    else:
        raise NotImplementedError


def convert_to_intavia_files(bionet_filepath: str, output_path: str):
    print(f"Generating InTaVia files for {sources}")
    for src in sources: os.makedirs(f"{output_path}/{src}", exist_ok=True)

    with open(bionet_filepath) as in_file_bionet:
        for i, line in enumerate(in_file_bionet):
            nlp_instance = json.loads(line)
            text_id = nlp_instance["id_composed"]
            print(f"{i} --> {text_id}")
            if nlp_instance['source'] == "foo": continue
            if 'IAV_' in nlp_instance['source']:
                intavia_output_file = f"{output_path}/IAV_MISC-biographie/{text_id}.json"
            else:
                intavia_output_file = f"{output_path}/{nlp_instance['source']}/{text_id}.json"
            # Create basic document in the new format
            doc = get_basic_doc_schema(text_id, nlp_instance["text_clean"], "stanza_nl")
            # Add Metadata
            for key in metadata_keys:
                doc[key] = nlp_instance[key]
            # Add NLP Related Info
            if len(nlp_instance['text_token_objects']) > 0:
                doc['data']['tokenization'] = {"stanza_nl": nlp_instance["text_tokens"]}
                doc['data']['morpho_syntax'] = {"stanza_nl": nlp_to_dict(nlp_instance)}
                doc['data']['entities'] = legacy_entity_mapper(nlp_instance.get('text_entities', []), "stanza_nl")
                doc['data']['time_expressions'] = legacy_timex_mapper(nlp_instance.get('text_timex', []), "heideltime")
            # Write the Output File
            json.dump(doc, open(intavia_output_file, "w"), indent=2, ensure_ascii=False)


def convert_to_intavia_mongo(bionet_filepath: str):
    client = MongoClient("mongodb://localhost:27017/")
    ORIGIN_DF = bionet_filepath # This has EVERY BIOGRAPHY
    
    db = client[DB_NAME]
    bionet_collection = db[COLLECTION_NAME]

    # Save data into Mongo if it doesn't exist or if we wish to Override the complete Collection with new Data ...
    if bionet_collection.count_documents({}) == 0:
        print("Creating Collection in Database ...")
        transfer_json_to_mongo(ORIGIN_DF, bionet_collection)
    else:
        override_db = input(f"Database already exists. Are you sure you want to override it [yes,NO]?  ") 
        if override_db.lower() == "yes":
            print(f"Replacing it with the new data from {ORIGIN_DF} ...")
            db.drop_collection(bionet_collection)
            transfer_json_to_mongo(ORIGIN_DF, bionet_collection)     
    
    # --- Do Analysis directly on the MongoDB ---
    print(f"MongoDB has {bionet_collection.count_documents({})} items in Collection: {COLLECTION_NAME}. DataBase: {DB_NAME}")


def get_basic_doc_schema(text_id: str, text: str, basic_nlp_processor: str):
    json_doc = {
                "text_id": text_id,
                "nlp_preprocessor": basic_nlp_processor,
                "data": {
                    "text": text,
                    "tokenization": [],
                    "morpho_syntax": [],
                    "entities": [],
                    "time_expressions": []
                }
            }
    return json_doc


def sentence_from_token_objects(token_objects: List[str]) -> Tuple[str, List[str]]:
    sentence_original, sentence_tokenized = "", []
    for tok in token_objects:
        sentence_tokenized.append(tok['text'])
        if tok["space_after"]:
            sentence_original += f"{tok['text']} "
        else:
            sentence_original += f"{tok['text']}"
    return sentence_original, sentence_tokenized


def nlp_to_dict(nlp_dict: Dict[str, Any]) -> Dict[str, Any]:
    sentencized, token_objs = defaultdict(list), []
    for tok in nlp_dict['text_token_objects']:
        try:
            sentencized[tok['sent_id']].append(tok)
        except:
            return []
    for sent_id, sentence in sentencized.items():
        sent_text, tokens = sentence_from_token_objects(sentence)
        token_objs.append({
            "paragraphID": None,
            "sentenceID": sent_id,
            "text": sent_text,
            "tokenized": " ".join(tokens),
            "words": [asdict(nlp_token2json_token(tok)) for tok in sentence]
        })
    return token_objs


def legacy_entity_mapper(old_entities: List[Dict[str, Any]], entity_processor: str) -> List[Dict[str, Any]]:
    new_entities = []
    for i, old_obj in enumerate(old_entities):
        new_obj = {
            "ID": f"{entity_processor}_{i}",
            "surfaceForm": old_obj["text"],
            "category": old_obj["label"],
            "locationStart": old_obj["start"],
            "locationEnd": old_obj["end"],
            "tokenStart": old_obj["start_token"],
            "tokenEnd": old_obj["end_token"],
            "method": entity_processor
        }
        new_entities.append(new_obj)
    return new_entities


def legacy_timex_mapper(old_timex: List[Dict[str, Any]], timex_processor: str) -> List[Dict[str, Any]]:
    new_timex = []
    for i, old_obj in enumerate(old_timex):
        new_obj = {
            "ID": f"t{i}",
            "category": old_obj["type"],
            "value": old_obj["value"],
            "surfaceForm": old_obj["text"],
            "locationStart": old_obj["start"],
            "locationEnd": old_obj["end"],
            "method": timex_processor
        }
        new_timex.append(new_obj)
    return new_timex


def nlp_token2json_token(nlp_token: Dict[str, Any]):
    return IntaviaToken(
        ID=nlp_token['id'],
        FORM=nlp_token['text'],
        LEMMA=nlp_token['lemma'],
        UPOS=nlp_token['upos'],
        XPOS=nlp_token['xpos'],
        HEAD=nlp_token['dep_head'],
        DEPREL=nlp_token['dep_rel'],
        DEPS=None,
        FEATS=nlp_token.get('morph', []),
        MISC={
            'SpaceAfter': nlp_token['space_after'],
            'StartChar': nlp_token['start_char'],
            'EndChar': nlp_token['end_char']
        }
    )


 ##### Mongo DB Auxiliary Functions #####
def load_json_data(filepath):
    with open(filepath) as f:
        for line in f:
            yield json.loads(line)

def insert_to_mongo(items: list, collection: MongoCollection):
    if isinstance(items, list):
        collection.insert_many(items) 
    collection.create_index([('ID', pymongo.ASCENDING), ('Version', pymongo.ASCENDING)])

def collection_is_empty(coll):
    return coll.count() == 0


def transfer_json_to_mongo(filepath: str, collection: MongoCollection) -> bool:
    total_data = 0
    with open(filepath) as f:
        for i, line in enumerate(f):
            nlp_instance = json.loads(line)
            text_id = nlp_instance["id_composed"]
            print(f"{i} --> {text_id}")
            if nlp_instance['source'] == "foo": continue
            # Create basic document in the new format
            doc = get_basic_doc_schema(text_id, nlp_instance["text_clean"], "stanza_nl")
            # Add Metadata
            for key in metadata_keys:
                doc[key] = nlp_instance[key]
            # Add NLP Related Info
            if len(nlp_instance['text_token_objects']) > 0:
                doc['data']['tokenization'] = nlp_instance["text_tokens"]
                doc['data']['morpho_syntax'] = nlp_to_dict(nlp_instance)
                doc['data']['entities'] = legacy_entity_mapper(nlp_instance.get('text_entities', []), "stanza_nl")
                doc['data']['time_expressions'] = legacy_timex_mapper(nlp_instance.get('text_timex', []), "heideltime")
            collection.insert_one(doc)
            total_data += 1
    print(f"Successfully inserted {total_data} rows from JSON file")
    return True


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("You need to provide three arguments: <storage_type> ['mongo' | 'files'] <input_json> ['path/to/BiosFile.jsonl'] <output_path> ['path/to/filestorage']")
    else:
        output_mode = sys.argv[1] # "mongo" | "files"
        bionet_json = sys.argv[2] 
        output_root = sys.argv[3] # Will be ignored if mode == mongo 
        main(bionet_json, output_mode, output_root)