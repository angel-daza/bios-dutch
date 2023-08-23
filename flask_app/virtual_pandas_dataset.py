"""
    The idea of these functions is to "simulate" a database by loading a dataset into pandas and querying the DataFrame from the WebApp.
    It is definitely not robust for production but should work fine for prototyping
"""
from typing import Any, Dict, List
import pandas as pd
import json


def load_bios_dataset(json_filename: str, biography_stats=None) -> pd.DataFrame:
    biographies = pd.read_json(json_filename, dtype={"person_id": 'string'}, orient='records', lines=True)
    biographies['display_id'] = biographies['person_id']
    biographies.set_index('person_id', inplace=True)
    biographies.sort_index(ascending=True)

    if biography_stats is not None:
        #first_id = list(biography_stats.keys())[0]
        #print(biographies.loc[[first_id]])
        df_sorted_ids = biographies["display_id"].tolist()

        sorted_bio_stats = [biography_stats[id_] for id_ in df_sorted_ids]
        biographies['stats'] = sorted_bio_stats

    return biographies


def get_biographies(biographies_query: pd.DataFrame, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    all_query_vals, all_query_fields, all_exact_match = query_params['vals'], query_params['fields'], query_params['string_exact_match']
    queried_ids = query_params['specific_ids']
    queried_names = query_params['specific_names']
    for i, (qval, qfield, qmatch) in enumerate(zip(all_query_vals, all_query_fields, all_exact_match)):
        biographies_query = df_query_string_match(biographies_query, qval, filter_field=qfield, exact_match=qmatch)
        print(i, biographies_query.shape)

        # Search for the specific Name/ID provided
        if queried_ids:
            # Filter the queried IDs since they are indexed already this might be better to be done first!
            try:
                if queried_ids[0] == 'ALL':
                    returned_bios = biographies_query.to_dict(orient='records')
                else:
                    returned_bios = biographies_query.loc[queried_ids].to_dict(orient='records')
                n_rows = len(returned_bios)
            except KeyError:
                returned_bios = []
                n_rows = -1
        elif queried_names:
            # In this case Names is a normal field so we construct a complex query...
            if queried_names[0].upper() == 'ALL':
                returned_bios = biographies_query.to_dict(orient='records')
            else:
                biographies_query = df_query_string_match(biographies_query, queried_names[0], filter_field='search_person_names', exact_match=False)
                print(-1, biographies_query.shape)
                returned_bios = biographies_query.to_dict(orient='records')
            n_rows = len(returned_bios)
        else:
            returned_bios = []
        
    return returned_bios



def df_query_string_match(main_df: pd.DataFrame, query_value: str, filter_field: str, exact_match: bool) -> pd.DataFrame:
    if query_value and query_value != 'Choose...':
        if exact_match == True:
            df_mask = main_df[filter_field] == query_value.lower()
        else:
            df_mask = main_df[filter_field].str.contains(query_value.lower()) == True
        query_result = main_df.loc[df_mask]
    else:
        query_result = main_df
    return query_result

def open_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return data