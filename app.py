# Google Charts: https://developers.google.com/chart/interactive/docs/gallery/combochart
from flask import Flask, render_template, request, redirect, jsonify, url_for
from collections import defaultdict
from spacy import displacy
from typing import Counter, List, Dict, NamedTuple
import json
import flask_app.virtual_pandas_dataset as my_data
from flask_paginate import Pagination, get_page_args

import ner_evaluation
from utils.classes import IntaviaDocument
from utils_general import FLASK_ROOT, INTAVIA_JSON_ROOT


STATISTICS = None

GLOBAL_QUERY = []

SORTED_BIOS = None

ALL_LABEL_COLORS = {
    "PER": "#add8e6",
    "LOC": "#c3e9bb",
    "ORG": "#fdfd96",
    "MISC": "#e9bbbb", 
    "GPE": "#e1e3ae",
    "DEATH_YEAR": "#e1e3ae", 
    "BIRTH_PLACE": "#e3aeda", 
    "DEATH_PLACE": "#aee3e0",
    "BIRTH_PLACE": "#bab6b1",
    "DEATH_PLACE": "#c497df",
    "OCCUPATION_NLP": "#e3cdae"
}

CONFIDENCE_COLORS = {
    "LOW": "#EBB9B5", 
    "WEAK": "#F2F1AE",
    "MEDIUM": "#c3e9bb",
    "HIGH":  "#A8E8A2",
    "VERY HIGH": "#2B7424",
}

def process_annotations(annos: List[Dict]):
    new_annos = []
    for a in annos:
        s = a["span"]
        entry = {"start": int(s["start"]), "end": int(s["end"]), "label": a["type"]}
        new_annos.append(entry)
    return new_annos


def process_query_string(query: str, search_type: List[str]) -> List[str]:
    if len(query) == 0:
        queried_elems = ["ALL"]
    elif "," in query:
        queried_elems = [q.strip() for q in query.split(',')]
    else:
        queried_elems = [query]
    if search_type is None:
        queried_ids = None
        queried_names = None
    elif search_type[0] == 'option_search_id':
            queried_ids = queried_elems
            queried_names = None
    elif search_type[0] == 'option_search_name':
        queried_ids = None
        queried_names = queried_elems
    return queried_ids, queried_names



app = Flask(__name__, template_folder="flask_app/templates", static_folder="flask_app/static")
render_docs = []
biographies_unified = []


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/bio_detail_google_charts/<source>/<text_id>")
def bio_detail_google_charts(source: str, text_id: str):

    bio_info = json.load(open(f"{INTAVIA_JSON_ROOT}/{source}/{text_id}.json"))
    bio = IntaviaDocument(bio_info)

    # General Entity Frequency Chart
    entity_counts = bio.get_entity_counts()
    cols = [
        {"id":"", "label":"Type",      "pattern":"",  "type":"string"},
        {"id":"", "label":"Frequency", "pattern":"",  "type":"number"}
    ]
    all_labels = defaultdict(int)
    for method, entity_list in entity_counts.items():
        for label, freq in entity_list:
            all_labels[label] += freq
    rows = []
    for label, freq in all_labels.items():
        rows.append({"c": [{"v":label, "f":None}, {"v": freq,"f":None}]})
    entity_freq_table = {"cols": cols, "rows": rows}

    # All Models Entity Distribution (Histogram of Label-Method Matrix)
    entity_cat_matrix = bio.get_entity_category_matrix() # [['Category', 'flair/ner-dutch-large_0.12.2', 'stanza_nl'], ['LOC', 14, 10], ['MISC', 18, 21], ['ORG', 10, 4], ['PER', 26, 28]]
    print(entity_cat_matrix)

    response = {
        "entity_freq_table": entity_freq_table,
        "entity_freq_title": "Distribution of Entities",
        "model_entity_dist": entity_cat_matrix
    }

    return jsonify(response)

@app.route("/bio_viewer", methods=['GET','POST'])
def bio_viewer():
    global biographies_search
    global GLOBAL_QUERY
    global SORTED_BIOS

    page, per_page, offset = get_page_args(page_parameter="page", per_page_parameter="per_page")

    used_bios = biographies_search if SORTED_BIOS is None else SORTED_BIOS

    if request.method == "GET":
        if len(GLOBAL_QUERY) == 0:
            n_rows, _ = used_bios.shape
            returned_bios = used_bios.iloc[offset:offset+10,:].to_dict(orient='records')
            pagination = Pagination(page=page, per_page=10, total=n_rows, css_framework="bootstrap4")
        else:
            n_rows = len(GLOBAL_QUERY)
            returned_bios = GLOBAL_QUERY[offset:offset+10]
            pagination = Pagination(page=page, per_page=per_page, total=n_rows, css_framework="bootstrap4")
        
        return render_template('biography_viewer.html', biographies=returned_bios, occupations=occupations_catalogue, 
                locations=locations_catalogue, sources=MY_SOURCES, n_rows=n_rows, page=page, per_page=per_page, pagination=pagination)
    else:
        
        queried_string = request.form.get('search', None)
        search_type=request.form.getlist('search_by_option', None) #['option_search_name'] or ['option_search_id']
        queried_ids, queried_names = process_query_string(queried_string, search_type)
        query_occupation = request.form.get('input_occupation', None)
        query_location = request.form.get('input_location', None)
        query_century = request.form.get('input_century', None)
        query_source = request.form.get('input_source', None)
        query_partition = request.form.get('input_partition', None)

        query_params = {
            'specific_ids': queried_ids,
            'specific_names': queried_names,
            'vals': [query_occupation, query_location, query_century, query_source, query_partition],
            'fields': ['search_occupations', 'search_places', 'search_person_century', 'search_sources', 'search_partitions'],
            'string_exact_match': [False, False, True, False, False]
        }

        # FOR LATER: https://www.statology.org/pandas-loc-multiple-conditions/ and filter in a single query for locations, occupations, and other fields...
        returned_bios = my_data.get_biographies(used_bios, query_params)
        n_rows = len(returned_bios)
        pagination = Pagination(page=1, per_page=10, total=n_rows, css_framework="bootstrap4")
        GLOBAL_QUERY = returned_bios


        return render_template('biography_viewer.html', biographies=returned_bios, occupations=occupations_catalogue, 
            locations=locations_catalogue, sources=MY_SOURCES, n_rows=n_rows, page=page, per_page=per_page, pagination=pagination)


@app.route("/bio_detail/<source>/<text_id>", methods=['GET'])
def bio_detail(source: str, text_id: str):
    global FLASK_ROOT
    global ALL_LABEL_COLORS
    display_labels = list(ALL_LABEL_COLORS.keys())
    global CONFIDENCE_COLORS
    display_conf_labels = list(CONFIDENCE_COLORS.keys())
    
    if request.method == "GET":
        bio_info = json.load(open(f"{INTAVIA_JSON_ROOT}/{source}/{text_id}.json"))
        bio = IntaviaDocument(bio_info)
        # Basic Info
        basic = bio.get_basic_stats()
        # Named Entities
        entity_dict = {}
        entity_count = 0
        valid_labels = ["PER", "LOC", "ORG", "MISC"]
        for ent in bio.get_entities():
            entity_count += 1
            if ent.category in valid_labels:
                if ent.method in entity_dict:
                    entity_dict[ent.method].append((ent.surfaceForm, ent.category))
                else:
                    entity_dict[ent.method] = [(ent.surfaceForm, ent.category)]
            
        # entity_categories_dict = bio.get_entity_counts()
        entity_cat_matrix = bio.get_entity_category_matrix()

        stats_dict = {
            "text_id": bio.text_id,
            "name": bio_info['name'],
            "source": bio_info['source'],
            "sentences": basic['sentences'],
            "total_tokens": len(bio.tokenization),
            "total_sentences": len(bio.morpho_syntax),
            "total_entities": entity_count,
            "total_timexp": len(bio.time_expressions),
            "top_verbs": basic['top_verbs'],
            "top_nouns":  basic['top_nouns'],
            "top_adjs":  basic['top_adjs'],
            "entity_cats": entity_cat_matrix,
            "entities": entity_dict
        }

        ## Code for Showing Evaluation Detail
        methods = bio.get_available_methods("entities")
        systems_eval, html_annotated = {}, {}
        # eval_per_label = {}
        per_label_tables = defaultdict(list)
        
        # Evaluate ONLY if there are HUMAN ANNOTATIONS
        gold_method = "human_gold"
        if gold_method in methods:
            for hypo_method in methods:
                predicted_entities = bio.get_entities([hypo_method], valid_labels)
                eval_dict = bio.evaluate_ner(gold_method, hypo_method, "full_match", valid_labels, ignore_text_after_gold=False)
                per_label_table = []
                for lbl, metrics_obj in eval_dict["metrics"].items():
                    if metrics_obj["Support"] > 0:
                        per_label_table.append([lbl, metrics_obj["P"], metrics_obj["R"], metrics_obj["F1"], metrics_obj["Support"]])
                    else:
                        per_label_table.append([lbl, "-", "-", "-", "-"])
                per_label_tables[f"{gold_method}_vs_{hypo_method}"] = per_label_table
                systems_eval[f"{gold_method}_vs_{hypo_method}"] = eval_dict
                # Create Entity Listing with Badges if there is gold
                # create_entity_badges(entity_dict, macro_evals)

                # Display Spans with Displacy
                display_obj = {"text": bio.text, "ents": [ent.get_displacy_format() for ent in predicted_entities]}
                html_annotated[hypo_method] = displacy.render(display_obj, style="ent", manual=True, page=False, options={"ents": display_labels, "colors": ALL_LABEL_COLORS})
        
        # Generate System Summary Table
        summary_table = ner_evaluation.system_label_report(systems_eval)
        for lbl, table in summary_table.items():
            print("-----------", lbl, "-----------")
            [print(t) for t in table]

        # Display CONFIDENCE Spans with Displacy
        confidence_list = bio.get_confidence_entities(mode="spans")
        display_obj = {"text": bio.text, "tokens": bio.tokenization, "spans": confidence_list}
        html_annotated["entity_overlap"] = displacy.render(display_obj, style="span", manual=True, page=False, options={"ents": display_labels, "colors": ALL_LABEL_COLORS})
        # Display "HeatMap"
        confidence_list = bio.get_confidence_entities(mode="ents")
        display_obj = {"text": bio.text, "ents": confidence_list}
        html_annotated["entity_heatmap"] = displacy.render(display_obj, style="ent", manual=True, page=False, options={"ents": display_conf_labels, "colors": CONFIDENCE_COLORS})

        return render_template("biography_detail.html", stats=stats_dict, evaluation={"eval": systems_eval, "per_label_tables": per_label_tables,
                                                                                      "summary_label_model": summary_table}, 
                                                                        html_annotated=html_annotated)

@app.route("/bio_overview_google_charts")
def bio_overview_google_charts():

    statistics = json.load(open(f"{FLASK_ROOT}/biographies/statistics.json"))
    entities_per_method = statistics["method_total"]
    data_array = [["method", "total"]]
    for k,v in entities_per_method.items():
        data_array.append([k, v])

    category_total = statistics["category_total"]
    category_array = [["category", "total"]]
    for k,v in category_total.items():
        category_array.append([k, v])

    category_per_method_total = statistics["category_per_method_total"]
    category_dict = {}
    for k,v in category_per_method_total.items():
        category_dict[k] = [[f"{k} entities summary", "total"]]
        for k2, v2 in v.items():
            category_dict[k].append([k2, v2])

    options = {
        "title" : "Total #entities per method", 
        "pieSliceText" : "percentage",
        "categoryTitle" : "#Entities per category"
    }

    response = {
        "method_total": data_array,
        "category_total" : category_array,
        "options": options,
        "per_method_category" : category_dict
    }

    return jsonify(response)

@app.route("/bio_viewer_sort", methods=['GET', 'POST'])
def biography_sort():
    sorting = request.json["sorting"]
    method = request.json["method"]
    global biographies_search
    global STATISTICS
    global SORTED_BIOS

    page, per_page, offset = get_page_args(page_parameter="page", per_page_parameter="per_page")

    if sorting == "entities":
        ids_sorted_per_method = STATISTICS["ids_sorted_per_method"]
        sorted_ids = ids_sorted_per_method[method]
    elif sorting == "distance":
        sorted_ids = STATISTICS["dist_sorted"]
    else:
        sorted_ids = STATISTICS["gold_dist_sorted"]

    sorted_ids = [id.split("_")[0] for id in sorted_ids]
    all_ids = biographies_search["display_id"].tolist()
    diff = list(set(all_ids) - set(sorted_ids))
    deep_copy = biographies_search.copy(deep=True)

    deep_copy.drop( index=diff, inplace=True)
    deep_copy.sort_values(by="display_id", key=lambda column: column.map(lambda e: sorted_ids.index(e)), inplace=True)

    SORTED_BIOS = deep_copy
    
    n_rows, _ = deep_copy.shape
    returned_bios = deep_copy.iloc[offset:offset+10,:].to_dict(orient='records')
    pagination = Pagination(page=page, per_page=10, total=n_rows, css_framework="bootstrap4")
    return render_template('biography_viewer.html', biographies=returned_bios, occupations=occupations_catalogue, 
                locations=locations_catalogue, sources=MY_SOURCES, n_rows=n_rows, page=page, per_page=per_page, pagination=pagination)

@app.route("/bio_stat_google_charts")
def bio_stat_google_charts():

    statistics = json.load(open(f"{FLASK_ROOT}/biographies/statistics.json"))

    dist_dict = statistics["max_dist_per_id"]
    array_dict = {}
    for k,v in dist_dict.items():
        k_norm = k.split("_")[0]
        array_dict[k_norm] = [
            ["method", "values"],
            [v["max"][0], v["max"][1]],
            [v["min"][0], v["min"][1]],
            ["gold", v["gold"]]

        ]

    options = {
        "title" : "Discrepancies", 
        "pieSliceText" : "percentage",
    }

    response = {
        "array_of_dicts": array_dict,
        "options": options,
    }

    return jsonify(response)
if __name__ == '__main__':
    # Load pre-computed statistics
    STATISTICS = my_data.open_json(f"{FLASK_ROOT}/biographies/statistics.json")

    # Get individual biography statistics, trim ids
    biography_stats = STATISTICS["per_id"]
    biography_stats = {k.split("_")[0] : v for k,v in biography_stats.items()}

    # First of all, load Full DataFrame in Memory just ONCE!
    biographies_search = my_data.load_bios_dataset(f"{FLASK_ROOT}/biographies/AllBios_unified_enriched.jsonl", biography_stats)


    # Load Catalogues to Choose from pre-defined fields (Bio Viewer)
    MY_SOURCES= open(f"{FLASK_ROOT}/sources.txt").read().split("\n")

    # In case we want to take directly from the DF
    occupations_catalogue =  biographies_search['list_occupations'].explode('list_occupations').value_counts()
    occupations_catalogue = sorted([occ_name for occ_name, occ_count in occupations_catalogue.items() if occ_count > 10])

    locations_catalogue =  [loc.split('\t') for loc in open(f"{FLASK_ROOT}/locations.txt").readlines()]
    locations_catalogue = sorted([loc_name for loc_name, loc_count in locations_catalogue if int(loc_count.strip('\n')) > 50])
    
    # Load WebApp
    app.config.from_object('flask_app.flask_config.DevelopmentConfig')
    app.run(host='localhost', port=8000)
