from typing import Dict, Any
import json, re
from utils.classes import IntaviaEntity

FLASK_ROOT = "flask_app/backend_data"
INTAVIA_JSON_ROOT = f"{FLASK_ROOT}/intavia_json/"

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


def normalize_metadata_name(full_name: str) -> str:
    names = full_name.split(" ")
    if len(names) < 2:
        if "|" in full_name:
            names = full_name.split("|")
            return names[0].title()
        else:
            return full_name.title()
    first_name = names[0].title()
    last_name = names[-1].title()
    return f"{first_name}_{last_name}"


def normalize_entity_per_name(full_name: str, default_name: str = None):
    # Process default_name
    if default_name:
        if not full_name or len(full_name) == 0: return ""
        if "_" in default_name:
            norm_first, norm_last = default_name.split("_")
        else:
            norm_first, norm_last = "", default_name
    else:
        if not full_name or len(full_name) == 0: return default_name
        norm_first, norm_last = None, None
    # Process full_name
    if len(full_name) == 2 and full_name[-1] == ".":
       initials = full_name.strip(".")
       if default_name and initials in default_name:
            return default_name
       else:
           return full_name.title().replace(" ", "_")
    elif "(" in full_name and ")" in full_name:
        "NIEUWENAAR ( gravin Anna van )"
        norm_last = full_name.split()[0]
        match = re.search(r'\((.*?)\)', full_name)
        first_names = match.group(1) if match else ""
        first_names_list = first_names.split()
        norm_first = first_names_list[0] if first_names_list else ""
        if len(norm_first) > 0:
            return f"{norm_first.title()}_{norm_last.title()}"
        else:
            return norm_last.title()
    elif "," in full_name:
        "FRANSEN , ELIAS"
        names = full_name.split(",")
        norm_first = names[1].strip()
        norm_last = names[0].strip()
        return f"{norm_first.title()}_{norm_last.title()}"
    elif norm_first is not None and norm_last is not None:
        if len(norm_first) > 0:
            return f"{norm_first.title()}_{norm_last.title()}"
        else:
            return norm_last.title()
    else:
        names = full_name.split()
        if len(names) == 1:
            return full_name.title()
        else:
            return f"{names[0].title()}_{names[-1].title()}"

