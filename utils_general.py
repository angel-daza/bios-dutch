from typing import Dict, Any, Tuple
import json, re
from utils.classes import IntaviaEntity

FLASK_ROOT = "flask_app/backend_data"
INTAVIA_JSON_ROOT = f"{FLASK_ROOT}/intavia_json/"
BIOS_MAIN_DATAFRAME = f"{FLASK_ROOT}/biographies/AllBios_unified_enriched.jsonl"


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


def normalize_entity_person_name(full_name: str, default_name: str = None):
    # Process default_name
    if default_name:
        if not full_name or len(full_name) == 0: return ""
        if "_" in default_name:
            norm_first, norm_last = default_name.split("_")
        else:
            norm_first, norm_last = "", default_name
    else:
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


def get_lifespan_from_meta(meta: Dict) -> Tuple[int, int]:
    birth = meta["birth_date"]
    death = meta["death_date"]
    if birth:
        b_year, b_month, b_day = birth.split("-")
        b_year = int(b_year)
    else:
        b_year = None
        if meta.get("birth_year_pred"): b_year = meta["birth_year_pred"]
    if death:
        d_year, d_month, d_day = death.split("-")
        d_year = int(d_year)
    else:
        d_year = None
    return (b_year, d_year)


def split_person_name(full_name: str, sep: str) -> Tuple[str, str, str]:
    possible_titles = ["Dr", "Prof", "Jhr", "Med", "Mr", "DrMed", "JhrMr", "JhrDr", "ProfDr"]
    title, first_name, last_name = None, None, None
    toks = full_name.split(sep)
    if len(toks) == 1:
        last_name = [toks[0]]
    elif len(toks) == 2:
        if toks[0].replace(".", "").strip() in possible_titles:
            title = [toks[0]]
        else:
            first_name = [toks[0]]
        last_name = [toks[1]]
    else:
        title_indices = []
        lastname_index = len(toks) - 1
        for i, t in enumerate(toks):
            if t.replace(".", "").strip() in possible_titles:
                title_indices.append(i)
            if t.lower() in ["ab", "de", "den", "der", "en", "le", "of", "te", "ten", "ter", "van", "von"]:
                lastname_index = i
                break
        name_start_ix = 0
        if len(title_indices) > 0:
            title = toks[title_indices[0]: title_indices[-1] + 1]
            name_start_ix = title_indices[-1] + 1
        first_name = toks[name_start_ix:lastname_index]
        last_name = toks[lastname_index:]
    if title: title = sep.join(title)
    if first_name or len(first_name) == 0: first_name = sep.join(first_name)
    if last_name or len(last_name) == 0: last_name = sep.join(last_name)
    return title, first_name, last_name