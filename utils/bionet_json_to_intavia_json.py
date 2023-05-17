import json, os
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from collections import defaultdict
from classes import IntaviaToken


def main():
    convert_to_intavia(bionet_filepath=f"data/seed_data/AllBios.jsonl", output_path=f"flask_app/backend_data/intavia_json")


def convert_to_intavia(bionet_filepath: str, output_path: str):
    metadata_keys = ['id_person', 'version', 'source', 'name', 'partition', 'birth_pl', 'birth_tm', 'baptism_pl', 'baptism_tm', 'death_pl', 'death_tm', 'funeral_pl', 'funeral_tm', 
                            'marriage_pl', 'marriage_tm', 'gender', 'category', 'father', 'mother', 'partner', 'religion', 'educations', 'faiths', 'occupations', 'residences']
    nlp_keys = ['text_clean', 'text_original', 'text_tokens', 'text_token_objects', 'text_sentences', 'text_entities', 'text_timex']

    sources = ['bioport', 'raa', 'rkdartists', 'dvn', 'vdaa', 'weyerman', 'bwsa', 'bwsa_archives', 'dbnl', 'nnbw', 'rijksmuseum', 
                       'pdc', 'blnp', 'knaw', 'wikipedia', 'nbwv', 'schilderkunst', 'portraits', 'glasius', 'schouburg', 
                       'smoelenboek', 'na', 'bwn', 'IAV_MISC-biographie', 'jews', 'bwg', 'bwn_1780tot1830', 'elias']
    
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
                doc['data']['tokenization'] = nlp_instance["text_tokens"]
                doc['data']['morpho_syntax'] = nlp_to_dict(nlp_instance)
                doc['data']['entities'] = legacy_entity_mapper(nlp_instance.get('text_entities', []), "stanza_nl")
                doc['data']['time_expressions'] = legacy_timex_mapper(nlp_instance.get('text_timex', []), "heideltime")
            # Write the Output File
            json.dump(doc, open(intavia_output_file, "w"), indent=2, ensure_ascii=False)

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


def nlp_to_dict(nlp_dict: Dict[str, Any]) -> Dict[str, Any]:
    sentencized, token_objs = defaultdict(list), []
    for tok in nlp_dict['text_token_objects']:
        try:
            sentencized[tok['sent_id']].append(tok)
        except:
            return []
    for sent_id, sentence in sentencized.items():
        sent_text = " ".join([tok['text'] for tok in sentence])
        token_objs.append({
            "paragraph": None,
            "sentence": sent_id,
            "text": sent_text,
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



if __name__ == "__main__":
    main()