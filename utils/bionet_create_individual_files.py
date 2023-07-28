import json, os
from typing import List
from collections import defaultdict
from tqdm import tqdm

def create_individual_bios_by_partition(original_bios_path: str, individual_parent_path: str, partition: str = None):
    if not os.path.exists(individual_parent_path): os.makedirs(individual_parent_path)
    if partition: os.makedirs(f"{individual_parent_path}/{partition}", exist_ok=True)
    print(f"Generating files for {partition}")
    with open(original_bios_path) as f:
        for line in tqdm(f.readlines()):
            row = json.loads(line)
            # Write to new file
            if partition:
                if row['partition'] == partition:
                    fout = open(f"{individual_parent_path}/{partition}/{row['id_composed']}.json", "w")
                    json.dump(row, fout, indent=2, ensure_ascii=False)
            else:
                fout = open(f"{individual_parent_path}/{row['id_composed']}.json", "w")
                json.dump(row, fout, indent=2, ensure_ascii=False)


def create_individual_bios_by_source(original_bios_path: str, individual_parent_path: str, sources: List[str] = []):
    source_count = defaultdict(int)
    if len(sources) == 0:
        sources = ['bioport', 'raa', 'rkdartists', 'dvn', 'vdaa', 'weyerman', 'bwsa', 'bwsa_archives', 'dbnl', 'nnbw', 'rijksmuseum', 
                       'pdc', 'blnp', 'knaw', 'wikipedia', 'nbwv', 'schilderkunst', 'portraits', 'glasius', 'schouburg', 
                       'smoelenboek', 'na', 'bwn', 'IAV_MISC-biographie', 'jews', 'bwg', 'bwn_1780tot1830', 'elias']
    
    print(f"Generating files for {sources}")
    for src in sources: os.makedirs(f"{individual_parent_path}/{src}", exist_ok=True)
    
    with open(original_bios_path) as f:
        for line in tqdm(f.readlines()):
            row = json.loads(line)
            # Write to new file
            if len(sources) == 0 or row['source'] in sources:
                source = row['source']
                if 'IAV_' in source:
                    fout = open(f"{individual_parent_path}/IAV_MISC-biographie/{row['id_composed']}.json", "w")
                else:
                    fout = open(f"{individual_parent_path}/{source}/{row['id_composed']}.json", "w")
                json.dump(row, fout, indent=2, ensure_ascii=False)
                source_count[source] += 1
    for item in sorted(source_count.items(), key=lambda x: x[1]):
        print(item)
    print("--------------")



# ### OPTION 1: Generate the individual files of a specific partition only (train, development or test)
# # Reminder: the test/dev/train partitions only include bios with text. In AllBios.jsonl there are really ALL bios, including the ones with metadata only
# create_individual_bios_by_partition("data/biographynet_development.jsonl", "data/json/bionet", "development")
# create_individual_bios_by_partition("data/biographynet_test.jsonl", "data/json/bionet", "test")
# create_individual_bios_by_partition("data/biographynet_train.jsonl", "data/json/bionet", "train")

### OPTION 2: Generate the individual files of specific sources only
# create_individual_bios_by_source("data/seed_data/AllBios.jsonl", "data/json/bionet")
# create_individual_bios_by_source("data/biographynet_train.jsonl", "data/json/bionet", ["weyerman", "glasius", "vdaa", "knaw"])