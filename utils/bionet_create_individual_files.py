import json, os
from typing import List
from collections import defaultdict

def create_individual_bios_by_partition(original_bios_path: str, individual_parent_path: str, partition: str = None):
    if not os.path.exists(individual_parent_path): os.makedirs(individual_parent_path)
    if partition: os.makedirs(f"{individual_parent_path}/{partition}", exist_ok=True)

    with open(original_bios_path) as f:
        for i, line in enumerate(f.readlines()):
            row = json.loads(line)
            # Write to new file
            if partition:
                if row['partition'] == partition:
                    fout = open(f"{individual_parent_path}/{partition}/{row['id_composed']}.json", "w")
                    json.dump(row, fout, indent=2)
            else:
                fout = open(f"{individual_parent_path}/{row['id_composed']}.json", "w")
                json.dump(row, fout, indent=2)
            print(i)


def create_individual_bios_by_source(original_bios_path: str, individual_parent_path: str, sources: List[str]):
    for src in sources: os.makedirs(f"{individual_parent_path}/{src}", exist_ok=True)
    source_count = defaultdict(int)
    with open(original_bios_path) as f:
        for i, line in enumerate(f.readlines()):
            row = json.loads(line)
            # Write to new file
            if row['source'] in sources:
                source = row['source']
                partition = row['partition']
                fout = open(f"{individual_parent_path}/{source}/{row['id_composed']}.{partition}.json", "w")
                json.dump(row, fout, indent=2)
                source_count[source] += 1
    for item in sorted(source_count.items(), key=lambda x: x[1]):
        print(item)
    print("--------------")

# create_individual_bios_by_partition("../data/biographynet_development.jsonl", "../data/json", "development")

# Reminder: the test/dev/train partitions only include bios with text. In AllBios.jsonl there are really ALL bios, including the ones with metadata only
create_individual_bios_by_source("../data/biographynet_test.jsonl", "../data/json", ["weyerman", "glasius", "vdaa", "knaw"])
#create_individual_bios_by_source("../data/biographynet_development.jsonl", "../data/json", ["weyerman", "glasius", "vdaa", "knaw"])
#create_individual_bios_by_source("../data/biographynet_train.jsonl", "../data/json", ["weyerman", "glasius", "vdaa", "knaw"])