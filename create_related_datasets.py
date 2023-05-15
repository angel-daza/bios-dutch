from typing import List, Dict, Any
import os, json
import pandas as pd
from utils.classes import MetadataComplete
from stats_unique_people import collect_global_info

INPUT_JSON = "data/AllBios_Unified.jsonl"
SAVE_RESULTS_PATH = "data/BioNetStats"

if not os.path.exists(SAVE_RESULTS_PATH): os.mkdir(SAVE_RESULTS_PATH)

BIONET_XML_FLASK_PATH = "static/bioport_export_2017-03-10"
BIONET_INTAVIA_JSON_FLASK_PATH = "static/intavia_json"

def main():
    bionet_people = [MetadataComplete.from_json(json.loads(l)) for l in open(INPUT_JSON).readlines()]
    print(f"Analyzing {len(bionet_people)} individuals")

    # #### Collect Global Info
    # global_dicts = collect_global_info(bionet_people)

    # ### Create Datasets for other Tasks (Web Visualization, Text classification...)
    create_flask_searchable_people_data(bionet_people, "flask_app/backend_data/biographies/AllBios_unified_enriched.jsonl")
    # create_text_classification_dataset(bionet_people, SAVE_RESULTS_PATH, global_dicts['occup2coarse'])
    # create_wikipedia_dataset(bionet_people, SAVE_RESULTS_PATH)



def create_flask_searchable_people_data(people: List[MetadataComplete], dataset_filepath: str):
    with open(dataset_filepath, 'w') as fout:
        for p in people:
            save_fields = {}
            save_fields['person_id'] = p.person_id
            save_fields['sources'] = p.sources
            birth_year = p.getBirthDate(method='most_likely_date')
            century = p.getCenturyLived()
            century_search = century.lower() if century else None
            save_fields['display_gender'] = p.getGender()
            save_fields['display_person_name'] = p.getName('unique_longest')
            save_fields['display_person_century'] = p.getCenturyLived()
            save_fields['display_birth_year'] = birth_year
            save_fields['display_death_year'] = p.getDeathDate(method='most_likely_date')
            save_fields['display_birth_place'] = p.getBirthPlace()
            save_fields['display_death_place'] = p.getDeathPlace()
            save_fields['list_person_names'] = p.getName('all_names')
            save_fields['list_genders'] = p.genders
            save_fields['list_occupations'] = [st.label for st in p.occupations]
            save_fields['list_places'] = p.getRelatedMetadataPlaces()
            save_fields['list_birth_years'] = [str(bd) for bd in p.births]
            save_fields['list_death_years'] = [str(dd) for dd in p.deaths]
            save_fields['list_educations'] = [str(ed) for ed in p.educations]
            save_fields['list_religions'] = [str(rel) for rel in p.religions]
            save_fields['list_faiths'] = [str(fa) for fa in p.faiths] # To display individual: person_bios.getFaith('stringified_all')
            save_fields['list_residences'] = [str(res) for res in p.residences]
            save_fields['search_sources'] = "|".join([x.lower() for x in p.sources])
            save_fields['search_person_century'] = century_search
            save_fields['search_person_names'] = "|".join([x.lower() for x in p.getName('all_names')])
            save_fields['search_occupations'] = "|".join([st.label.lower() for st in p.occupations])
            save_fields['search_places'] = "|".join([x.lower() for x in p.getRelatedMetadataPlaces()])

            # Extract NLP Info Directly Tied to Individual Texts...
            person_ments, place_ments = [], []
            texts_with_content = []
            text_ids, xml_paths, intavia_jsons = [], [], []
            sources_with_text = []
            for i, text in enumerate(p.texts):
                tid = f"{p.person_id}_{p.versions[i]}"
                text_ids.append(tid)
                xml_paths.append(f"{BIONET_XML_FLASK_PATH}/{tid}.xml")
                intavia_jsons.append(f"{BIONET_INTAVIA_JSON_FLASK_PATH}/{tid}.{p.sources[i]}")
                if len(text.strip().strip("\n")) > 0:
                    person_ments.append("|".join([x.lower() for x in p.getEntityMentions(entity_label='PER', text_ix=i)]))
                    place_ments.append("|".join([x.lower() for x in p.getEntityMentions(entity_label='LOC', text_ix=i)]))
                    texts_with_content.append(text)
                    sources_with_text.append(p.sources[i])
            # Attach at the PERSON level
            save_fields['text_ids'] = text_ids
            save_fields['original_files'] = xml_paths
            save_fields['nlp_files'] = intavia_jsons
            save_fields['sources_with_text'] = sources_with_text
            save_fields['list_entity_per'] = list(set([x for x in p.getEntityMentions(entity_label='PER')]))
            save_fields['list_entity_loc'] = list(set([x for x in p.getEntityMentions(entity_label='LOC')]))
            save_fields['search_person_mentions'] = "|".join(person_ments)
            save_fields['search_place_mentions'] = "|".join(place_ments)
            save_fields['texts'] = texts_with_content
            fout.write(json.dumps(save_fields)+'\n')


def create_text_classification_dataset(people: List[MetadataComplete], dataset_filepath: str, occupations_dict: Dict[str, str]):
    data = []
    for p in people:
        occupation = occupations_dict.get(p.getOccupation(), None)
        gender = p.getGender()
        century = p.getCenturyLived()
        for i, text in enumerate(p.texts):
            if len(text) > 0:
                semi_toks = text.split()
                truncated_text = " ".join(semi_toks[:200]) if len(semi_toks) > 200 else text
                dataset_row = {
                    'person_id': p.person_id,
                    'text_id': f"{p.person_id}_{p.versions[i]}",
                    'person_name': p.getName('unique_longest'),
                    'gender': gender,
                    'source': p.sources[i],
                    'century': century,
                    'occupation': occupation,
                    'is_labeled_occupation': True if occupation else False,
                    'is_labeled_gender': True if gender else False,
                    'is_labeled_century': True if century else False,
                    'text': truncated_text
                }
                data.append(dataset_row)
    
    df = pd.DataFrame(data)
    df.to_json(f"{dataset_filepath}/occupations_text_classification.jsonl", orient='records', lines=True)

    df_labeled = df.loc[df["is_labeled_occupation"] == True]
    df_unlabeled = df.loc[df["is_labeled_occupation"] == False]
    
    df_train = df_labeled.groupby('occupation', group_keys=False).apply(lambda x: x.sample(frac=0.9))
    df_dev = df_labeled.drop(df_train.index)

    df_unlabeled.to_json(f"{dataset_filepath}/occup_unlabeled.jsonl", orient='records', lines=True)
    df_train.to_json(f"{dataset_filepath}/occup_train.jsonl", orient='records', lines=True)
    df_dev.to_json(f"{dataset_filepath}/occup_dev.jsonl", orient='records', lines=True)


def create_wikipedia_dataset(people: List[MetadataComplete], dataset_filepath: str):
    included_ids_tr, included_ids_dv, included_ids_ts  = set(), set(), set()
    with open(f"{dataset_filepath}/wiki_search_train.tsv", "w") as f_train, open(f"{dataset_filepath}/wiki_search_dev.tsv", "w") as f_dev, open(f"{dataset_filepath}/wiki_search_test.tsv", "w") as f_test:
        for p in people:
            names = p.getName('all_names') # Ordered shortest to longest
            name_str = "||".join(names)
            b_date = p.getBirthDate()
            if b_date:
                b_year = b_date[0]
            else:
                b_year = p.getBirthDate_baseline1()
            d_date = p.getDeathDate()
            if d_date:
                d_year = d_date[0]
            else:
                d_year = -1
            for partition in p.partitions:
                row_str = f"{p.person_id}\t{name_str}\t{b_year}\t{d_year}\n"
                if partition == "train" and p.person_id not in included_ids_tr:
                    f_train.write(row_str)
                    included_ids_tr.add(p.person_id)
                elif partition == "development" and p.person_id not in included_ids_dv:
                    f_dev.write(row_str)
                    included_ids_dv.add(p.person_id)
                elif partition == "test" and p.person_id not in included_ids_ts:
                    f_test.write(row_str)
                    included_ids_ts.add(p.person_id)


if __name__ == "__main__":
    main()