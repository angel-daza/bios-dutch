from typing import List, Dict, Any
from collections import defaultdict, Counter
import glob, os, json, statistics, itertools, re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from lifelines.utils import concordance_index
import rbo

from utils.classes import IntaviaEntity, IntaviaDocument, MetadataComplete, NER_METHOD_DISPLAY
from stats_unique_people import build_names_dictionaries
from utils_general import get_gold_annotations, INTAVIA_JSON_ROOT, normalize_entity_per_name, normalize_metadata_name


def main():
    # Get Dataset
    gold_docs = get_gold_annotations() # 347 Documents
    documents = get_intavia_documents(f"{INTAVIA_JSON_ROOT}/*", gold_docs, keep_gold_limits=True)
    
    # Get Name Dictionaries (This is from the whole 70K People to get more connections!)
    name_dict_path = f"data/BioNetStats/bionet_id2names.json"
    name_dict_path_inv = f"data/BioNetStats/bionet_names2id.json"
    bionet_json = "data/All_Bios_Unified.jsonl"
    name2id, id2names = get_names_dict(bionet_json, name_dict_path, name_dict_path_inv)
    norm_dict = {} # Will Hold {"Possible Name": "Normalized_FirstName_LastName"}
    for name, pid in name2id.items():
        if "_" in name:
            for n in id2names[pid]:
                norm_dict[n] = name
    json.dump(norm_dict, open("data/BioNetStats/bionet_normalized_namedict.json", "w"), indent=2, ensure_ascii=False)


    # Statistics
    get_entity_stats(documents, name2id)
    get_method_divergence_stats(documents)
    
    # Build Networks
    # get_ego_network_of_mentions(documents["37716498_02"], "human_gold", norm_dict, ["PER", "ORG", "LOC"]) # hendrik hendicus huisman
    # get_ego_network_of_mentions(documents["19103689_02"], "human_gold", norm_dict, ["PER", "ORG", "LOC"]) # Charlotte Sophie of Aldenburg
    # get_ego_network_of_mentions(documents["40672923_04"], "human_gold", norm_dict, ["PER", "ORG", "LOC"]) # Helena_Kuipers-Rietberg Network
    # network_per_gold = get_social_network(documents, "human_gold", ["PER"], norm_dict)

    network_analysis_summary = []
    for sys in NER_METHOD_DISPLAY:
        print(f"----- {sys} -----")
        ### EGO NETWORKS
        for id, doc in documents.items():
            if id in ["19103689_02", "40672923_04"]:
                get_ego_network_of_mentions(doc, sys, norm_dict, ["PER"], save_dir=f"local_outputs/{id}")
                get_ego_network_of_mentions(doc, sys, norm_dict, ["LOC"], save_dir=f"local_outputs/{id}")
                get_ego_network_of_mentions(doc, sys, norm_dict, ["ORG"], save_dir=f"local_outputs/{id}")
                get_ego_network_of_mentions(doc, sys, norm_dict, ["PER", "ORG", "LOC"], save_dir=f"local_outputs/{id}")
        ### SOCIAL NETWORK
        network_people = get_social_network(documents, sys, ["PER"], norm_dict)
        get_nodes_and_nbrs(network_people, nodes_of_interest=["Charlotte_Bentinck"], save_dir=f"local_outputs/19103689_02/Charlotte_Nbrs_{NER_METHOD_DISPLAY[sys]}.jpg")
        # network_all = get_social_network(documents, sys, ["PER", "ORG", "LOC"], norm_dict)
        metrics = compute_network_metrics(network_people)
        metrics["method"] = sys
        clique_dist = get_cliques_distribution(network_people)
        print(clique_dist)
        network_analysis_summary.append(metrics)
    
    # pd.DataFrame(network_analysis_summary).to_csv("local_outputs/network_analysis_summary.tsv", sep="\t", index=False)




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


def get_entity_stats(documents: Dict[str, IntaviaDocument], name2id: Dict[str, str]):
    data = []
    all_possible_persons, all_possible_locations, all_possible_orgs = [], [], []
    gld_possible_persons, gld_possible_locations, gld_possible_orgs = [], [], []
    entity_model_matrix = []
    for doc_id, doc in documents.items():
        entity_methods = doc.get_available_methods(task_layer='entities')
        token_len = len(doc.tokenization)
        for method in entity_methods:
            entity_dict = defaultdict(int)
            doc_ents = doc.get_entities(methods=[method], valid_labels=['PER', 'LOC', 'ORG'])
            for ent in doc_ents:
                entity_dict[ent.category] += 1
                if ent.category == "PER":
                    all_possible_persons.append(ent.surfaceForm.title())
                    if ent.method == "human_gold": gld_possible_persons.append(ent.surfaceForm)
                elif ent.category == "LOC":
                    all_possible_locations.append(ent.surfaceForm.title())
                    if ent.method == "human_gold": gld_possible_locations.append(ent.surfaceForm)
                elif ent.category == "ORG":
                    all_possible_orgs.append(ent.surfaceForm.title())
                    if ent.method == "human_gold": gld_possible_orgs.append(ent.surfaceForm)
                entity_model_matrix.append({
                    "method": method,
                    "category": ent.category,
                    "entity": ent.surfaceForm.title()
                })
                
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

    # ENTITY MODEL MEASURES
    emm_df = pd.DataFrame(entity_model_matrix)
    rank_top_n = [10, 50, 100, 500, 1000]
    for top in rank_top_n:
        method_rankings = {}
        for cat in ["PER", "ORG", "LOC"]:
            emm_df_cat = emm_df[emm_df["category"] == cat]
            emm_df_cat = emm_df_cat.groupby("method").value_counts("entity").groupby(level=0, group_keys=False).head(top).reset_index()
            emm_df_cat.columns = ["method",	"category",	"entity", "count"]
            emm_df_cat = emm_df_cat.sort_values(["method", "count"], ascending=False)
            for m in NER_METHOD_DISPLAY:
                method_rankings[m] = list(emm_df_cat[emm_df_cat["method"] == m]["entity"])
            emm_df_cat.to_csv(f"local_outputs/0_entity_method_counter_{cat}_{top}.tsv", sep="\t")
        # Correlation of Rankings
        get_rankings_correlation(method_rankings)

    # PERSON DICT
    ranked_per_all = sorted(Counter(all_possible_persons).most_common(), key=lambda x: - x[1])
    ranked_per_with_id = []
    for per, freq in ranked_per_all:
        per_id = name2id.get(per.lower())
        if per_id:
            ranked_per_with_id.append((per, freq, per_id))
        else:
            ranked_per_with_id.append((per, freq, "9999999999"))
    csv_sep = ","
    # PERSON DICT
    df = pd.DataFrame(ranked_per_all)
    df.to_csv("local_outputs/0_PER_dict.csv", sep=csv_sep, header=["Entity", "Frequency"])
    ranked_per_with_id = sorted(ranked_per_with_id, key= lambda x: (x[2], - x[1]))
    # PER - ID DICT
    df = pd.DataFrame(ranked_per_with_id)
    df.to_csv("local_outputs/0_PER_ID_dict.csv", sep=csv_sep, header=["Entity", "Frequency", "BioNet ID"])
    # LOCATION DICT
    ranked_loc_all = sorted(Counter(all_possible_locations).most_common(), key=lambda x: - x[1])
    df = pd.DataFrame(ranked_loc_all)
    df.to_csv("local_outputs/0_LOC_dict.csv", sep=csv_sep, header=["Entity", "Frequency"])
    # ORGANIZATION DICT
    ranked_org_all = sorted(Counter(all_possible_orgs).most_common(), key=lambda x: - x[1])
    df = pd.DataFrame(ranked_org_all)
    df.to_csv("local_outputs/0_ORG_dict.csv", sep=csv_sep, header=["Entity", "Frequency"])
    # GOLD HISTOGRAMS
    ranked_per_gld = sorted(Counter(gld_possible_persons).most_common(), key=lambda x: - x[1])
    df = pd.DataFrame(ranked_per_gld).to_csv("local_outputs/0_PER_gold_dict.csv", sep=csv_sep, header=["Entity", "Frequency"])
    ranked_loc_gld = sorted(Counter(gld_possible_locations).most_common(), key=lambda x: - x[1])
    df = pd.DataFrame(ranked_loc_gld).to_csv("local_outputs/0_LOC_gold_dict.csv", sep=csv_sep, header=["Entity", "Frequency"])
    ranked_org_gld = sorted(Counter(gld_possible_orgs).most_common(), key=lambda x: - x[1])
    df = pd.DataFrame(ranked_org_gld).to_csv("local_outputs/0_ORG_gold_dict.csv", sep=csv_sep, header=["Entity", "Frequency"])
    


def get_method_divergence_stats(documents: Dict[str, IntaviaDocument]):
    data = []
    for doc_id, doc in documents.items():
        row = {
                'name': doc.metadata.get('name'),
                'birth_time': doc.metadata.get('birth_tm'),
                'death_time': doc.metadata.get('death_tm')
                }
        ent_variance = doc.get_ner_variance(valid_labels=['PER', 'LOC', 'ORG'])
        for name, val in sorted(ent_variance.items(), key=lambda x: x[0]):
            row[name] = val
        # Append the computed fields into the table
        data.append(row)
    df = pd.DataFrame(data).sort_values(by=['per_stdev', 'per_freq_human_gold'], ascending=False)
    df.to_csv("local_outputs/method_divergence_stats.csv")


def get_ego_network_of_mentions(document: IntaviaDocument, method: str, norm_dict: Dict[str, str], valid_labels: List[str], save_dir: str):
    ego_network = nx.Graph() # nx.DiGraph()
    person_id = document.metadata['id_person']
    person_norm_name = norm_dict.get(document.metadata['name'])
    if not person_norm_name:
        person_norm_name = normalize_metadata_name(document.metadata['name'])
    if "_" in person_norm_name: 
        person_firstname, person_lastname = person_norm_name.split("_")
    else:
        person_firstname, person_lastname = person_norm_name, person_norm_name
    node_colors = {"PER": "skyblue", "ORG": "#1f78b4", "LOC": "green", "ERR": "black"}
    print(f"\n------------\n{document.metadata['name']} ({document.text_id})")

    # Keep Global track of "popularity" measured as PER mentions in all texts
    doc_mentions = document.get_entities([method], valid_labels=valid_labels)
    if len(doc_mentions) > 0:
        first_entity_name = doc_mentions[0].surfaceForm
    else:
        first_entity_name = document.metadata['name']
    related_mentions = defaultdict(int)
    for ent in doc_mentions:
        if ent.category == "PER":
            norm_surface_form = normalize_entity_per_name(ent.surfaceForm)
            if norm_surface_form == person_lastname:
                norm_surface_form = person_norm_name
            elif norm_surface_form == f"{person_lastname[0]}.":
                norm_surface_form = person_norm_name
            print(f"\t{ent.surfaceForm} -------> {norm_surface_form}")
        else:
            norm_surface_form = ent.surfaceForm
        related_mentions[(norm_surface_form, ent.category)] += 1
        # related_mentions[(ent.surfaceForm, ent.category)] += 1

    sorted_mentions = sorted(related_mentions.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_mentions) == 0: return

    for m in sorted_mentions:
        print(m)

    # Main Node == Most Mentioned Entity (since it is their biography this assumtution should hold)
    # Backoff: Take the First Entity in the Text
    first_ent = None
    for mention_tuple, freq in sorted_mentions:
        if mention_tuple[1] == "PER":
            first_ent = mention_tuple[0]
            break
    normalized_main_entity = normalize_entity_per_name(first_ent, person_norm_name)
    if not normalized_main_entity or "." in normalized_main_entity:
        normalized_main_entity = normalize_entity_per_name(first_entity_name, person_norm_name)
    print(f"{document.metadata['name']} ({document.text_id}) --> {person_norm_name} | {normalized_main_entity}")
    ego_network.add_node(normalized_main_entity)
    ego_network.nodes[normalized_main_entity]["node_type"] = "PER"

    for rel_tup, freq in related_mentions.items():
        text, category = rel_tup
        ego_network.add_node(text, node_type=category)
        ego_network.add_edge(normalized_main_entity, text, weight=freq)
    
    labels = {e: ego_network.edges[e]['weight'] for e in ego_network.edges}

    node_attributes = nx.get_node_attributes(ego_network, "node_type")
    node_colors_list = [node_colors[node_attributes[node]] for node in ego_network.nodes] 

    print(nx.info(ego_network))

    plt.figure(figsize=(12, 8), num=f"{document.metadata['name'].title()}")
    pos = nx.spring_layout(ego_network, seed=42)
    nx.draw(ego_network, pos, with_labels=False, node_size=100, node_color=node_colors_list, font_size=4)
    nx.draw_networkx_edge_labels(ego_network, pos, edge_labels=labels)

    # Draw the legend shifted away from the noe (useful for longer names)
    center_y = 0
    for k, (node, (x, y)) in enumerate(pos.items()):
        if k == 0: center_y = y
        if y > center_y:
            label_y = y + 0.05
        else:
            label_y = y - 0.05
        plt.text(x, label_y, node, fontsize=10, ha="center", va="center")


    plt.suptitle(f"NER Network {method}", fontsize=10)
    plt.axis("off") 
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/EgoNetwork_{'_'.join(valid_labels)}_{NER_METHOD_DISPLAY[method]}.jpg")
    #plt.show()
    plt.clf()
    
    return ego_network


def get_social_network(documents: List[IntaviaDocument], method: str, valid_labels: List[str], norm_dict: Dict):
    social_network = nx.Graph()
    # Keep Global track of "popularity" measured as PER mentions in all texts
    related_mentions = defaultdict(int)

    # Iterate All Documents and Build Social Network
    for doc_id, doc in documents.items():
        person_id = doc.metadata['id_person']
        person_norm_name = norm_dict.get(doc.metadata['name'])
        if not person_norm_name:
            person_norm_name = normalize_metadata_name(doc.metadata['name'])
        if "_" in person_norm_name: 
            person_firstname, person_lastname = person_norm_name.split("_")
        else:
            person_firstname, person_lastname = person_norm_name, person_norm_name


        doc_mentions = doc.get_entities([method], valid_labels=valid_labels)
        
        for ent in doc_mentions:
            if ent.category == "PER":
                norm_surface_form = normalize_entity_per_name(ent.surfaceForm)
                if norm_surface_form == person_lastname:
                    norm_surface_form = person_norm_name
                elif norm_surface_form == f"{person_lastname[0]}.":
                    norm_surface_form = person_norm_name
                # print(f"\t{ent.surfaceForm} -------> {norm_surface_form}")
            else:
                norm_surface_form = ent.surfaceForm
            
            related_mentions[(norm_surface_form, ent.category)] += 1
            social_network.add_edge(person_norm_name, norm_surface_form)
            #related_mentions[(ent.surfaceForm, ent.category)] += 1
            #social_network.add_edge(person_id, ent.surfaceForm)
            

    # Show the top 100 Popular by Mention
    sorted_links = sorted(related_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
    for m in sorted_links:
        print(m)

    print(nx.info(social_network))

    return social_network


def compute_network_metrics(network):
    metrics = {}
    # Node-level Metrics
    node_degrees = [len(list(network.neighbors(n))) for n in network.nodes()]
    if len(node_degrees) == 0: return {}
    node_deg_centralities = nx.degree_centrality(network)
    node_betw_centralities = nx.betweenness_centrality(network)
    
    # Graph Level Metrics
    metrics['n_nodes'] = network.number_of_nodes()
    metrics['n_edges'] = network.number_of_edges()
    metrics['density'] = round(nx.density(network), 4)
    metrics['degree_average'] = round(statistics.mean(node_degrees), 4)
    metrics['degree_centrality'] = round(statistics.mean(node_deg_centralities.values()), 4)
    metrics['betweenness_centrality'] =  round(statistics.mean(node_betw_centralities.values()), 4)

    cliques = 0

    ## ---- Find Most Prolific Nodes
    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(node_deg_centralities.values()))
    centralities = sorted(list(node_deg_centralities.values()), reverse=True)[:10]
    threshold_dc = centralities[-1]
    # Find the user(s) that have collaborated the most: prolific_collaborators
    prolific_collaborators = [(n, round(dc, 2)) for n, dc in node_deg_centralities.items() if dc >=threshold_dc]
    print("Prolific Entities:", prolific_collaborators)

    return metrics
    

def maximal_cliques(G, size):
	"""
		Finds all maximal cliques in graph 'G' that are of size 'size'.
	"""
	mcs = []
	for clique in nx.find_cliques(G):
		if len(clique) == size:
			mcs.append(clique)
			return mcs


def get_cliques_distribution(G):
    """
    	Finds all maximal cliques in graph 'G' that are of size 'size'.
    """
    cliques = defaultdict(int)
    for clique in nx.find_cliques(G):
        cliques[len(clique)] += 1
    return cliques


# Define get_nodes_and_nbrs()
def get_nodes_and_nbrs(G, nodes_of_interest, save_dir):
    """
	    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.`
    """
    nodes_to_draw = []
    for n in nodes_of_interest:
        nodes_to_draw.append(n)
        print(f"Neighbors of {n}")
        for nbr in G.neighbors(n):
            nodes_to_draw.append(nbr)
            print(f"\t{nbr}")
    G_draw = G.subgraph(nodes_to_draw)

    nx.draw(G, with_labels=False)
    plt.savefig(save_dir)
    plt.clf()
    return G_draw


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


def get_rankings_correlation(method_rankings: Dict[str, List]):

    all_ents_in_ranking = []
    for _, vals in method_rankings.items():
        all_ents_in_ranking += vals
    # Create a mapping of unique strings to integer values
    string_to_int_mapping = {string: i for i, string in enumerate(set(all_ents_in_ranking))}
    methods_analyzed = set()
    metrics_data = []
    ranking_size = -1
    for method_1 in method_rankings:
        for method_2 in method_rankings:
            method_key = tuple(sorted([method_1, method_2]))
            if method_key not in methods_analyzed: # method_1 != method_2 and
                methods_analyzed.add(method_key)
                list_1 = method_rankings[method_1]
                list_2 = method_rankings[method_2]
                #print(list_1)
                #print(list_2)
                if len(list_1) > len(list_2):
                    list_1 = method_rankings[method_1][:len(list_2)]
                elif len(list_1) < len(list_2):
                    list_2 = method_rankings[method_1][:len(list_1)]
                
                list_1 = [string_to_int_mapping[x] for x in list_1]
                list_2 = [string_to_int_mapping[y] for y in list_2]
                if ranking_size == -1: ranking_size = len(list_1)
                # Calculate the rank correlation coefficient between list1 and list2
                tau_1_2, _ = kendalltau(list_1, list_2)
                tau_1_2 = round(tau_1_2, 2)
                rbo_1_2 = round(rbo.RankingSimilarity(list_1, list_2).rbo(), 2)
                metrics_data.append({"model_1": method_1, "model_2": method_2, "tau": tau_1_2, "rbo": rbo_1_2})
                print(f"Kendall's Tau Correlation between {method_1} and {method_2} = {tau_1_2}")
                print(f"Rank Biased Overlap (RBO) between {method_1} and {method_2} = {rbo_1_2}")
                print("---------------")
    pd.DataFrame(metrics_data).to_csv(f"local_outputs/method_ranking_correlations_{ranking_size}.tsv", index=False, sep="\t")

if __name__ == "__main__":
    main()