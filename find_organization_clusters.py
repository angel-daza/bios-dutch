import json
from collections import defaultdict

# dict_keys(['id_person', 'version', 'id_composed', 'source', 'name', 'partition', 'nlp_processor', 'meta_keys', 'had_html', 'original_path', 
            # 'birth_pl', 'birth_tm', 'baptism_pl', 'baptism_tm', 'death_pl', 'death_tm', 'funeral_pl', 'funeral_tm', 'marriage_pl', 'marriage_tm', 'gender', 
            # 'category', 'father', 'mother', 'partner', 'religion', 'educations', 'faiths', 'occupations', 'residences', 'text_clean', 'text_original', 
            # 'text_tokens', 'text_token_objects', 'text_sentences', 'text_entities', 'text_timex', 'tokens_len', 'meta_len', 'hhucap_annotations'])

def main():
    org_dict, person_dict = get_person_orgs("data/biographynet_development.jsonl")
    org_clusters = get_org_clusters(org_dict, person_dict)

    sorted_org_dict = {_get_person_name(person_dict, k):v for k,v in sorted(org_dict.items(), key=lambda x: len(x[1]), reverse=True)}
    sorted_org_clusters = {k:[_get_person_name(person_dict, v) for v in vals] for k,vals in sorted(org_clusters.items(), key=lambda x: len(x[1]), reverse=True)}

    json.dump(person_dict, open("data/person_dict.json", "w", encoding="utf-8"), indent=2)
    json.dump(sorted_org_dict, open("data/org_mentions_per_person.json", "w", encoding="utf-8"), indent=2)
    json.dump(sorted_org_clusters, open("data/organization_clusters.json", "w", encoding="utf-8"), indent=2)
    

def get_person_orgs(filepath):
    org_mapper = defaultdict(list)
    person_mapper = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            bio = json.loads(line)
            entities = bio.get('text_entities', [])
            # Dealing with people here ...
            if bio['name'] not in person_mapper[bio['id_person']]:
                person_mapper[bio['id_person']].append(bio['name'])
            # Dealing with entities here ...
            # ENTITY: {'text': 'N.B.A.C. Wb.', 'label': 'MISC', 'start': 172, 'end': 184, 'start_token': 39, 'end_token': 41}
            for org in [ent['text'] for ent in entities if ent['label'] == 'ORG']:
                if org not in org_mapper[bio['id_person']]:
                    org_mapper[bio['id_person']].append(org)
            if bio['source'] == 'wikipedia':
                print(bio['text_clean'][:200])

    return org_mapper, person_mapper


def get_org_clusters(org_dict, person_dict):
    org_clusters = defaultdict(list)
    for person_id, orgs in org_dict.items():
        for org in orgs:
            if person_id not in org_clusters[org]:
                org_clusters[org].append(person_id)
    return org_clusters


def _get_person_name(person_dict, person_id):
    name_list = person_dict[person_id]
    if len(name_list) > 0:
        sorted_names = sorted(name_list, key=lambda x: len(x))
        return sorted_names[0]
    return person_id

if __name__ == '__main__':
    main()