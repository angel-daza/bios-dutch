# Import Libs
from typing import Dict, List, Any
import pandas as pd
import json, os, sys

from tqdm import tqdm
from classes import MetadataComplete, Event, State


def main(input_filepath: str, output_filepath: str, create_lite_version: bool):
    if create_lite_version:
        # Create LITE version so it fits in memory for unification
        basename = os.path.basename(input_filepath)
        LITE_INPUT_JSON = f"data/{basename.split('.')[:-1][0]}.lite.jsonl"
        create_bios_lite(input_filepath, LITE_INPUT_JSON)
    else:
        # If the file is small or it is already a lite version
        LITE_INPUT_JSON = input_filepath
    
    # Load the Dataset 
    biographies = pd.read_json(LITE_INPUT_JSON, dtype={"id_person": 'string', 'version': 'string'}, orient='records', lines=True)
    biographies.set_index(["id_person", "version"], inplace=True, append=False, drop=False)
    biographies.sort_index(ascending=True)

    print(biographies.head())
    print(biographies.shape)
    print(biographies.index)

    unique_persons = list(biographies['id_person'].unique())
    metadata_info = get_unique_people(biographies, unique_persons, output_filename=output_filepath)
    json.dump(metadata_info, open("data/unified_metadata_info.json", "w"), indent=2, ensure_ascii=False)


def create_bios_lite(original_bios_path: str, lite_bios_path: str):
    columns_to_keep = [
					'id_composed', 'id_person', 'version', 'source', 'name', 
					'partition', 'meta_keys', 'nlp_processor', 'birth_pl', 'birth_tm', 
					'baptism_pl', 'baptism_tm', 'death_pl', 'death_tm', 'funeral_pl', 'funeral_tm', 
					'marriage_pl', 'marriage_tm', 'gender', 'category', 'father', 'mother', 'partner', 
					'religion', 'educations', 'faiths', 'occupations', 'residences', 
                    'text_original', 'tokens_len',
					'text_clean', 'text_tokens', 'text_sentences', 'text_entities', 'text_timex', 'hhucap_annotations'
					]
    print(f"Creating LITE Version of file '{original_bios_path}' ...")
    with open(original_bios_path) as f:
        with open(lite_bios_path, "w") as fout:
            for line in tqdm(f.readlines()):
                row = json.loads(line)
                # Create new Row with the minimum fields for this experiment...
                new_row = {}
                for k in row.keys():
                    if k in columns_to_keep:
                        new_row[k] = row[k]
                # Write to new file
                fout.write(json.dumps(new_row)+"\n")


def unify_metadata(person_id: str, data_versions: List[Dict]) -> MetadataComplete:
    meta = MetadataComplete(person_id)
    for ver_dct in data_versions:
        meta.addName(ver_dct['name'])
        meta.addVersion(ver_dct['version'])
        meta.addSource(ver_dct['source'])
        meta.addPartition(ver_dct['partition'])
        meta.addText(ver_dct['text_clean'])
        meta.addPreTokenized(ver_dct['text_tokens'])
        meta.addEntities(ver_dct.get('text_entities'))
        meta.addTimex(ver_dct.get('text_timex'))
        meta.addBirthDay(Event('Birth', location = ver_dct['birth_pl'], date = ver_dct['birth_tm']))
        meta.addDeathDay(Event('Death', location = ver_dct['death_pl'], date = ver_dct['death_tm']))
        meta.addFather(ver_dct['father'])
        meta.defineMother(ver_dct['mother'])
        meta.defineGender(ver_dct['gender'])
        meta.definePartner(ver_dct['partner'])
        meta.addReligion(ver_dct['religion'])
        meta.addOtherEvents(Event('Baptism', location = ver_dct['baptism_pl'], date = ver_dct['baptism_tm']))
        meta.addOtherEvents(Event('Funeral', location = ver_dct['funeral_pl'], date = ver_dct['funeral_tm']))
        meta.addOtherEvents(Event('Marriage', location = ver_dct['marriage_pl'], date = ver_dct['marriage_tm']))
        # Multi Data [{Cat: X, CatBegin: XX-XX-XXXX, CatEnd: XX-XX-XXXX} ...] THESE ARE STATES!
        if ver_dct['educations']:
            for educ in ver_dct['educations']:
                meta.addEducation(State(educ['name'], beginDate=educ['begin'], endDate=educ['end']))
        if ver_dct['occupations']:
            for occup in ver_dct['occupations']:
                meta.addOccupation(State(occup['name'], beginDate=occup['begin'], endDate=occup['end']))
        if ver_dct['faiths']:
            for faith in ver_dct['faiths']:
                meta.addFaith(State(faith['name'], beginDate=faith['begin'], endDate=faith['end']))
        if ver_dct['residences']:
            for residence in ver_dct['residences']:
                meta.addResidence(State(residence['name'], beginDate=residence['begin'], endDate=residence['end']))
    return meta



def get_unique_people(bionet_df: pd.DataFrame, people_of_interest: List[str], output_filename: str) -> Dict[str, Any]:
    """Looks in the whole database for more biographies of the people of interest, and unifies all the metadata available under the same ID. 
        Saves it on JSON to re-use later
    """
    bio_counter = 0
    metadata_dict = {}
    with open(output_filename, "w") as fout:
        for person_id in tqdm(people_of_interest):
            diff_versions = []
            for bio_id, version in bionet_df.loc[person_id, :].iterrows():
                bio_counter += 1
                version_as_dict = json.loads(version.to_json())
                diff_versions.append(version_as_dict)
            unified_metadata = unify_metadata(person_id, diff_versions)
            fout.write(json.dumps(unified_metadata.to_json()) + '\n')
            metadata_dict[person_id] = unified_metadata.getFullMetadataDict(autocomplete=True)
    
    print(f"Successfully unified {bio_counter} biographies into {len(people_of_interest)} unique people")
    return metadata_dict

if __name__ == "__main__":
    """
        Run Examples: 
            
            python utils/bionet_unify_people_bios.py "data/seed_data/AllBios.jsonl" "data/All_Bios_Unified.jsonl"
            
            python utils/bionet_unify_people_bios.py "data/seed_data/biographynet_development.jsonl" "data/Dev_Bios_Unified.jsonl"
    """
    if len(sys.argv) < 2:
        raise Exception("You need to provide and input and output file [See Run Examples inside the script]")
    else:
        input_filepath = sys.argv[1]
        output_filepath = sys.argv[2]
        create_lite_version = True
        main(input_filepath, output_filepath, create_lite_version)