from flask import Flask, render_template, request, redirect
import spacy
from spacy import displacy
from typing import Counter, List, Dict, NamedTuple
import json
import re
import random
import virtual_pandas_dataset as my_data
from classes_mirror import AnnoFlask, IntaviaDocument
import pandas as pd
from flask_paginate import Pagination, get_page_args
from collections import defaultdict
import evaluation as eval_module


GLOBAL_QUERY = []

ALL_LABEL_COLORS = {
    "BIO_PER": "#c3e9bb", 
    "BIRTH_DATE": "#e9bbbb", 
    "DEATH_DATE": "#4285F4", 
    "BIRTH_YEAR": "#aeb8e3", 
    "DEATH_YEAR": "#e1e3ae", 
    "BIRTH_PLACE": "#e3aeda", 
    "DEATH_PLACE": "#aee3e0",
    "BIRTH_PLACE": "#bab6b1",
    "DEATH_PLACE": "#c497df",
    "OCCUPATION_NLP": "#e3cdae"
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


app = Flask(__name__)
render_docs = []
biographies_unified = []


@app.route("/")
def home():
    return render_template('home.html')


@app.route("/bio_viewer", methods=['GET','POST'])
def bio_viewer():
    global biographies_search
    global GLOBAL_QUERY
    page, per_page, offset = get_page_args(page_parameter="page", per_page_parameter="per_page")
    if request.method == "GET":
        if len(GLOBAL_QUERY) == 0:
            n_rows, _ = biographies_search.shape
            returned_bios = biographies_search.iloc[offset:offset+10,:].to_dict(orient='records')
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
        returned_bios = my_data.get_biographies(biographies_search, query_params)
        n_rows = len(returned_bios)
        pagination = Pagination(page=1, per_page=10, total=n_rows, css_framework="bootstrap4")
        GLOBAL_QUERY = returned_bios

        return render_template('biography_viewer.html', biographies=returned_bios, occupations=occupations_catalogue, 
            locations=locations_catalogue, sources=MY_SOURCES, n_rows=n_rows, page=page, per_page=per_page, pagination=pagination)


@app.route("/bio_detail/<source>/<text_id>", methods=['GET'])
def bio_detail(source: str, text_id: str):
    global FLASK_ROOT
    if request.method == "GET":
        bio_info = json.load(open(f"{FLASK_ROOT}/intavia_json/{source}/{text_id}.json"))
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
            
        entity_categories_dict = bio.get_entity_counts()

        stats_dict = {
            "text_id": bio.text_id,
            "name": bio_info['name'],
            "sentences": basic['sentences'],
            "total_tokens": len(bio.tokenization),
            "total_sentences": len(bio.morpho_syntax),
            "total_entities": entity_count,
            "total_timexp": len(bio.time_expressions),
            "top_verbs": basic['top_verbs'],
            "top_nouns":  basic['top_nouns'],
            "top_adjs":  basic['top_adjs'],
            "entity_cats": entity_categories_dict,
            "entities": entity_dict
        }

        methods = bio.get_available_methods("entities")
        evals = {}
        if "human_gold" in methods:
            gold_method = "human_gold"
            for hypo_method in methods:
                reference_entities, predicted_entities = [], []
                for ent in bio.entities:
                    if ent.category in valid_labels:
                        if ent.method == gold_method:
                            reference_entities.append(ent)
                        if ent.method == hypo_method: # replace for elif after debugging
                            predicted_entities.append(ent)
                evals[f"{gold_method}_vs_{hypo_method}"] = eval_module.evaluate_ner(reference_entities, predicted_entities)

        return render_template("biography_detail.html", stats=stats_dict, evaluation=evals)

if __name__ == '__main__':
    FLASK_ROOT = "flask_app/backend_data"

    # First of all, load Full DataFrame in Memory just ONCE!
    biographies_search = my_data.load_bios_dataset(f"{FLASK_ROOT}/biographies/AllBios_unified_enriched.jsonl")

    # Load Catalogues to Choose from pre-defined fields (Bio Viewer)
    MY_SOURCES= open(f"{FLASK_ROOT}/sources.txt").read().split("\n")

    # In case we want to take directly from the DF
    occupations_catalogue =  biographies_search['list_occupations'].explode('list_occupations').value_counts()
    occupations_catalogue = sorted([occ_name for occ_name, occ_count in occupations_catalogue.iteritems() if occ_count > 10])

    locations_catalogue =  [loc.split('\t') for loc in open(f"{FLASK_ROOT}/locations.txt").readlines()]
    locations_catalogue = sorted([loc_name for loc_name, loc_count in locations_catalogue if int(loc_count.strip('\n')) > 50])
    
    # Load WebApp
    app.config.from_object('flask_config.DevelopmentConfig')
    app.run(host='localhost', port=8000)
