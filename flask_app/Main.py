from flask import Flask, render_template, request, redirect
import spacy
from spacy import displacy
from typing import Counter, List, Dict, NamedTuple
import json
import re
import random
import virtual_pandas_dataset as my_data
from classes_mirror import AnnoFlask
import pandas as pd
from flask_paginate import Pagination, get_page_args


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


def get_biography_slice(offset, page_size):
    return False
     


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

        query_params = {
            'specific_ids': queried_ids,
            'specific_names': queried_names,
            'vals': [query_occupation, query_location, query_century, query_source],
            'fields': ['search_occupations', 'search_places', 'search_person_century', 'search_sources'],
            'string_exact_match': [False, False, True, False]
        }

        # FOR LATER: https://www.statology.org/pandas-loc-multiple-conditions/ and filter in a single query for locations, occupations, and other fields...
        returned_bios = my_data.get_biographies(biographies_search, query_params)
        n_rows = len(returned_bios)
        pagination = Pagination(page=1, per_page=10, total=n_rows, css_framework="bootstrap4")
        GLOBAL_QUERY = returned_bios

        return render_template('biography_viewer.html', biographies=returned_bios, occupations=occupations_catalogue, 
            locations=locations_catalogue, sources=MY_SOURCES, n_rows=n_rows, page=page, per_page=per_page, pagination=pagination)


@app.route("/tag_sentence", methods=['GET','POST'])
def tag_sentence():
    global render_docs
    if request.method == "POST":
        if request.form.get('selectLanguage', 'None') == 'Dutch':
            nlp = spacy.load('nl_core_news_sm')
        else:
            nlp = spacy.load('en_core_web_sm')
        sentences = [request.form.get('sentence', None)]
        if None in sentences: sentences = []
        get_dep = request.form.get('DepParser', 'False')
        get_ner = request.form.get('NerParser', 'False')
        html_dep, html_ner = '', ''
        for sent in sentences:
            doc = nlp(sent)
            if get_dep == 'DEP':
                html_dep = displacy.render(doc, style="dep", page=False)
            if get_ner == 'NER':
                html_ner = displacy.render(doc, style="ent", page=False)
            render_docs.append({'text': sent, 'html_dep':html_dep, 'html_ner': html_ner})
        render_docs.reverse()
        return render_template('sentence.html', sentences=render_docs)
    else:
        render_docs =[]
        return render_template('sentence.html', sentences=[]) 
    

@app.route("/hhucap_annotations", methods=['GET'])
def hhucap_annotations():
    render_docs = []
    TEST_PATH="/Users/daza/Repos/my-vu-experiments/BiographyNet/outputs/GPU_2022_03_02/biographynet_test.jsonl"
    test_set = [json.loads(line) for line in open(TEST_PATH).readlines()][1:]
    people, profession = [], []
    for ex in test_set:
        labeled_ents = process_annotations(ex["hhucap_annotations"]["annotations"])
        anno = {"text": re.sub("\n", " ", ex["text_clean"]), "ents": labeled_ents, "title": re.sub("\|", " | ", ex["name"])}
        html = displacy.render(anno, style="ent", manual=True, page=False)
        mac_os_path = re.sub("/home/jdazaareva/data/volume_2/Data/BiographyNet/bioport_export_2017-03-10/", 
                             "/Users/daza/DATA/BiographyNet/bioport_export_2017-03-10/", 
                             ex["original_path"]
                             )
        render_docs.append({'doc_id': ex["id_composed"], 'html': html, 'path': mac_os_path})
        # For general Stats
        for e in labeled_ents:
            if e['label'] == "Profession":
                profession.append(ex["text_clean"][e['start']:e['end']])
            elif e['label'] == "Person":
                people.append(ex["text_clean"][e['start']:e['end']])

    stat_dict = {
        "total_person": len(people),
        "total_profession": len(profession),
        "top_person": Counter(people).most_common(50),
        "top_profession": Counter(profession).most_common(50)
    }

    return render_template('hhucap.html', sentences=render_docs, stats=stat_dict)

@app.route("/skweak_annotations", methods=['GET'])
def skweak_annotations():
    """
        This function visualizes the file produced by the BiographyNet/bionet_querier_mongo.py script
    """
    render_docs = []
    MY_DATA="/Users/daza/Repos/my-vu-experiments/BiographyNet/outputs/skweak_annotations.json"
    annotations = [json.loads(line) for line in open(MY_DATA).readlines()]
    for anno in annotations:
        html = displacy.render(anno, style="ent", manual=True, page=False)
        render_docs.append({'doc_id': "ID = ?", 'html': html})
    return render_template('skweak.html', sentences=render_docs)


@app.route("/annotation_visualizer", methods=['GET'])
def annotation_visualizer():
    """
        This function visualizes the file produced by the BiographyNet/bionet_querier_mongo.py script
    """
    render_docs = []
    MY_DATA="/Users/daza/Repos/my-vu-experiments/BiographyNet/outputs/metadata_annotations.json"
    annotations = [AnnoFlask(**json.loads(line)) for line in open(MY_DATA).readlines()]
    display_labels = list(ALL_LABEL_COLORS.keys())
    #colors = {lbl: random.choice(ALL_COLORS) for lbl in display_labels}
    colors = ALL_LABEL_COLORS
    for anno in annotations:
        html = displacy.render(anno._asdict(), style="ent", manual=True, page=False, options={"ents": display_labels, "colors": colors})
        render_docs.append({'html': html, 'doc_id': anno.text_id, 'source': anno.source, 'b_dates': anno.birthday, 'd_dates': anno.death, 'occupations': anno.occupations})
    return render_template('annotation_viz.html', sentences=render_docs, sources=MY_SOURCES)


if __name__ == '__main__':
    FLASK_ROOT = "flask_app/backend_data"

    json_test = "data/json/development/75376904_03.json"

    # First of all, load Full DataFrame in Memory just ONCE!
    biographies_search = my_data.load_bios_dataset(f"{FLASK_ROOT}/biographies/AllBios_unified_enriched.jsonl")

    # Load Catalogues to Choose from pre-defined fields (Bio Viewer)
    MY_SOURCES= open(f"{FLASK_ROOT}/sources.txt").read().split("\n")

    # In case we want to take directly from the DF
    occupations_catalogue =  biographies_search['list_occupations'].explode('list_occupations').value_counts()
    print(occupations_catalogue.head())
    occupations_catalogue = sorted([occ_name for occ_name, occ_count in occupations_catalogue.iteritems() if occ_count > 10])

    locations_catalogue =  [loc.split('\t') for loc in open(f"{FLASK_ROOT}/locations.txt").readlines()]
    locations_catalogue = sorted([loc_name for loc_name, loc_count in locations_catalogue if int(loc_count.strip('\n')) > 50])
    
    # Load WebApp
    app.config.from_object('flask_config.DevelopmentConfig')
    app.run(host='localhost', port=8000)
