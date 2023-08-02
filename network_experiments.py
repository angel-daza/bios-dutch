from typing import List, Dict, Any
from collections import defaultdict, Counter
import glob, os, json, statistics
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from utils.classes import IntaviaEntity, IntaviaDocument, MetadataComplete, NER_METHOD_DISPLAY
from stats_unique_people import build_names_dictionaries
from utils_general import get_gold_annotations, INTAVIA_JSON_ROOT


def main():
    # Get Dataset
    gold_docs = get_gold_annotations() # 347 Documents
    documents = get_intavia_documents(f"{INTAVIA_JSON_ROOT}/*", gold_docs, keep_gold_limits=True)
    
    # Statistics
    get_entity_stats(documents)
    get_method_divergence_stats(documents)

    # Get Name Dictionaries
    name_dict_path = f"data/BioNetStats/bionet_id2names.json"
    name_dict_path_inv = f"data/BioNetStats/bionet_names2id.json"
    bionet_json = "data/seed_data/biographynet_test.jsonl"
    name2id, id2names = get_names_dict(bionet_json, name_dict_path, name_dict_path_inv)
    
    # Build Networks
    get_network_of_person_mentions(documents["37716498_02"], "human_gold") # hendrik hendicus huisman
    for sys in NER_METHOD_DISPLAY:
        print(f"----- {sys} -----")
        get_network_of_person_mentions(documents["37716498_02"], sys)



def get_intavia_documents(intavia_files_root:str, gold_docs: Dict[str, List[IntaviaEntity]] = {}, keep_gold_limits: bool = False) -> Dict[str, Any]:
    all_docs = {}
    for src_path in glob.glob(intavia_files_root):
        for bio_path in glob.glob(f"{src_path}/*"):
            bio_id = os.path.basename(bio_path).strip(".json")
            if bio_id in gold_docs:
                intavia_doc = IntaviaDocument(json.load(open(bio_path)))
                # If we have gold info, then we filter only predictions within the gold range 
                # (This is to avoid on purpose predicting entities in the references)
                if keep_gold_limits:
                    last_gold_limit = gold_docs[bio_id][-1].locationEnd
                    filtered_ents = []
                    for e in sorted(intavia_doc.entities, key= lambda x: x.locationStart):
                        if e.locationStart <= last_gold_limit:
                            filtered_ents.append(e)
                    intavia_doc.entities = filtered_ents
                # Add IntaviaDoc to Dict
                all_docs[bio_id] = intavia_doc
    return all_docs


def get_entity_stats(documents: Dict[str, IntaviaDocument]):
    data = []
    for doc_id, doc in documents.items():
        entity_methods = doc.get_available_methods(task_layer='entities')
        token_len = len(doc.tokenization)
        for method in entity_methods:
            entity_dict = defaultdict(int)
            doc_ents = doc.get_entities(methods=[method], valid_labels=['PER', 'LOC', 'ORG'])
            for ent in doc_ents:
                entity_dict[ent.category] += 1
            entity_len = len(doc_ents)
            data.append({
                'text_id': doc_id,
                'name': doc.metadata['name'],
                'birth_time': doc.metadata['birth_tm'],
                'death_time': doc.metadata['death_tm'],
                'method': method,
                'tokens': token_len,
                'per_freq': entity_dict['PER'],
                'loc_freq': entity_dict['LOC'],
                'org_freq': entity_dict['ORG'],
                'entity_freq': entity_len,
                'entity_density': round(entity_len*100 / token_len, 2)
                })
    df = pd.DataFrame(data).sort_values(by=['entity_freq', 'per_freq'], ascending=False)
    df.to_csv("local_outputs/entity_stats.csv")


def get_method_divergence_stats(documents: Dict[str, IntaviaDocument]):
    data = []
    for doc_id, doc in documents.items():
        row = {
                'name': doc.metadata.get('name'),
                'birth_time': doc.metadata.get('birth_tm'),
                'death_time': doc.metadata.get('death_tm')
                }
        ent_variance = doc.get_ner_variance(valid_labels=['PER', 'LOC', 'ORG'])
        for name, val in ent_variance.items():
            row[name] = val
        # Append the computed fields into the table
        data.append(row)
    df = pd.DataFrame(data).sort_values(by=['per_stdev', 'per_freq_human_gold'], ascending=False)
    df.to_csv("local_outputs/method_divergence_stats.csv")


def get_network_of_person_mentions(document: IntaviaDocument, method: str):
    ego_network = nx.Graph()
    person_id = document.metadata['id_person']

    # Keep Global track of "popularity" measured as PER mentions in all texts
    doc_persons = document.get_entities([method], valid_labels=["PER"])
    related_mentions = defaultdict(int)
    for per in doc_persons:
        related_mentions[per.surfaceForm] += 1

    for m in sorted(related_mentions.items(), key=lambda x: x[1], reverse=True):
        print(m)

    # TODO: Use NetworkX to visualize
    # Build a Simple Network Based on Person Mentions as recognized by the metadata Names Dictionary
    for rel, freq in related_mentions.items():
        ego_network.add_edge(person_id, rel, weight=freq)
        # id_rel = name2id.get(rel.lower())
        # if id_rel and id_rel != p.person_id: # Do NOT add self-mentions 
        #     connected_people[p.person_id].add(id_rel)
        #     people_network.add_edge(p.person_id, id_rel)
    labels = {e: ego_network.edges[e]['weight'] for e in ego_network.edges}
    
    print(nx.info(ego_network))
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off")
    pos = nx.spring_layout(ego_network, iterations=15, seed=1721)
    plot_options = {"node_size": 100, "with_labels": True, "width": 0.15}
    nx.draw_networkx(ego_network, pos=pos, ax=ax, **plot_options)
    nx.draw_networkx_edge_labels(ego_network, pos, edge_labels=labels)
    plt.show()


def get_names_dict(bionet_people_json: str, name_dict_path: str, name_dict_path_inv: str):
    if not os.path.exists(name_dict_path):
        bionet_people = [MetadataComplete.from_json(json.loads(l)) for l in open(bionet_people_json).readlines()]
        name2id, id2names = build_names_dictionaries(bionet_people)
        with open(name_dict_path, "w") as f:
            json.dump(id2names, f, indent=2,  ensure_ascii=False)
        with open(name_dict_path_inv, "w") as f:
            json.dump(name2id, f, indent=2,  ensure_ascii=False)
    else:
        id2names = json.load(open(name_dict_path))
        name2id = {}
        for id, names in id2names.items():
            for name in names:
                name2id[name] = id
    
    return name2id, id2names

if __name__ == "__main__":
    main()