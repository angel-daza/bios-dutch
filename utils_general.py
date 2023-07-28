from typing import Dict, Any
import json
from utils.classes import IntaviaEntity

FLASK_ROOT = "flask_app/backend_data"
INTAVIA_JSON_ROOT = f"{FLASK_ROOT}/intavia_json/"

# System Keys: 'human_gold', 'stanza_nl', 'flair/ner-dutch-large_0.12.2', 'gpt-3.5-turbo', 'gysbert_hist_fx_finetuned_epoch2'
NER_METHOD_DISPLAY = {'human_gold': 'gold', 
                    'stanza_nl': 'stanza', 
                    'flair/ner-dutch-large_0.12.2': 'flair', 
                    'gpt-3.5-turbo': 'gpt', 
                    'gysbert_hist_fx_finetuned_epoch2': 'gysbert'
                    }

def get_gold_annotations() -> Dict[str, Any]:
    gold_paths = ["data/bionet_gold/biographynet_test_A_gold.json",
                  "data/bionet_gold/biographynet_test_B_gold.json", 
                  "data/bionet_gold/biographynet_test_C_gold.json"]
    raw_docs = {}
    for gold_path in gold_paths:
        raw_docs.update(json.load(open(gold_path)))

    gold_docs = {}
    for doc_id, obj in raw_docs.items():
        gold_docs[doc_id] = sorted([IntaviaEntity(**e) for e in obj["entities"]], key= lambda x: x.locationStart)
    return gold_docs