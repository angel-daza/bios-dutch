import json
from collections import defaultdict
from utils.classes import MetadataComplete



def main():
    org_dict, person_dict = get_person_orgs("data/Dev_Bios_Unified.jsonl")
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
            row = json.loads(line)
            person = MetadataComplete.from_json(row)
            # Entities PER-TEXT, so we flatten the list to have access to all mentions...
            if person.texts_entities:
                entities = []
                for text_ents in person.texts_entities:
                    if text_ents:
                        for ent in text_ents:
                            if ent: 
                                entities.append(ent)
            # Dealing with people here ...
            name = person.getName()
            if name not in person_mapper[person.person_id]:
                person_mapper[person.person_id].append(name)
            # Dealing with entities here ...
            # ENTITY: {'text': 'N.B.A.C. Wb.', 'label': 'MISC', 'start': 172, 'end': 184, 'start_token': 39, 'end_token': 41}
            for org in [ent['text'] for ent in entities if ent['label'] == 'ORG']:
                if org not in org_mapper[person.person_id]:
                    org_mapper[person.person_id].append(org)

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