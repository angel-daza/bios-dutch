"""
    This Script Assumes the biographies were already unified (bionet_unify_people_bios.py)
"""
from collections import defaultdict
import json, re
from statistics import mean, median, mode
from typing import Counter, List, Dict, OrderedDict, Any

from utils.classes import MetadataComplete
from tabulate import tabulate
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import os
from nltk.util import ngrams

INPUT_JSON = "data/AllBios_Unified.jsonl"
SAVE_RESULTS_PATH = "data/BioNetStats"


if not os.path.exists(SAVE_RESULTS_PATH): os.mkdir(SAVE_RESULTS_PATH)


def main():
    bionet_people = [MetadataComplete.from_json(json.loads(l)) for l in open(INPUT_JSON).readlines()]
    print(f"Analyzing {len(bionet_people)} individuals")

    # # ### Distribution of unique people across sources
    # source_distribution(bionet_people)
    # source_distribution(bionet_people, partition='train')
    # source_distribution(bionet_people, partition='development')
    # source_distribution(bionet_people, partition='test')

    # # ### Stats about how complete is the metadata even after unifying everyone
    # verify_unified_metadata(bionet_people)


    # #### Collect Global Info
    global_dicts = collect_global_info(bionet_people)

    # Proportion of Males and Females - Using Metadata Only
    gender_list = gender_analysis(bionet_people)
    evaluate_inferred_gender(bionet_people)
    exit()


    # Occupation <--> Gender co-occurrences
    get_crosstab_cats([global_dicts['occupations_coarse_list']], [gender_list], values=[1]*len(gender_list), 
                        val_sort='female', row_limit=50, separate_vals=True)

    # Statistics of People's Birthdates (how many people born in which centuries?)
    print(f"\n\n------------ Biographies thorugh time according to MetaData BirthYear Only ------------\n")
    # Including only the 'Gold Metadata'
    people_century = group_per_century(bionet_people, print_stats=True)
    # BASELINE EVAL: Using Metadata Only vs Using Heuristic (First Date recognized in text)
    evaluate_inferred_birth_dates(bionet_people)
    # Show Stats assuming we 'autocomplete' using the best baseline, for the biographies without 'Gold Metadata'
    people_century_augmented = group_per_century(bionet_people, print_stats=True, include_inferred_dates=True)

    # People's Lifespans per Century
    print(f"\n\n------------ Average Lifespan thorugh centuries according to MetaData Only ------------\n")
    lifespans_dict = show_lifespans_per_century(people_century)
    total_stats = []
    for century, lifespan_list in lifespans_dict.items():
        total_stats.append((century[0], century[1], len(lifespan_list), min(lifespan_list), max(lifespan_list), mean(lifespan_list), median(lifespan_list), mode(lifespan_list)))
    print(tabulate(total_stats, headers=("From", "To", "People", "Min LifeSpan", "Max LifeSpan", "Mean LifeSpan", "Median", "Mode"), floatfmt=".2f"))

    # People's Mention Counter and Network (Based only on NER PERSON Mentions inside the texts)
    get_network_of_person_mentions(bionet_people)


def source_distribution(people: List[MetadataComplete], partition='all'):
    """Compute amount of biographies grouped by source

    Args:
        people (List[MetadataComplete]): List of unified Biographies (one row per person)
        partition (str, optional): restrict the counter to a certain partition. Values: 'train', 'development', 'test'. Defaults to 'all'.
    """
    source_dict_all, source_dict_text, source_dict_tokens = defaultdict(int), defaultdict(int), defaultdict(int)
    source_dict_individuals = defaultdict(set)
    unique_people_set = set()
    for p in people:
        assert len(p.texts) == len(p.sources) == len(p.partitions)
        for i, text in enumerate(p.texts):
            if partition == 'all' or p.partitions[i] == partition:
                src_key = p.sources[i]
                if src_key.startswith('IAV_'):
                    src_key = 'IAV_MISC-biographie'
                source_dict_all[src_key] += 1
                if len(text.strip()) > 0:
                    source_dict_text[src_key] += 1
                    source_dict_tokens[src_key] += len([tok for tok_list in p.texts_tokens for tok in tok_list])
                    source_dict_individuals[src_key].add(p.person_id)
                    unique_people_set.add(p.person_id)

    source_dict_all['** TOTAL **'] = sum(source_dict_all.values())
    source_dict_text['** TOTAL **'] = sum(source_dict_text.values())
    source_dict_individuals['** TOTAL **'] = unique_people_set
    with open(f"{SAVE_RESULTS_PATH}/source_distribution_all_{partition}.tsv", "w") as fout:
        fout.write(tabulate(source_dict_all.items()))
    with open(f"{SAVE_RESULTS_PATH}/source_distribution_with-text_{partition}.tsv", "w") as fout:
        fout.write(tabulate(source_dict_text.items()))
    with open(f"{SAVE_RESULTS_PATH}/source_distribution_token-counts_{partition}.tsv", "w") as fout:
        fout.write(tabulate(source_dict_tokens.items()))
    with open(f"{SAVE_RESULTS_PATH}/source_distribution_individuals_{partition}.tsv", "w") as fout:
        fout.write(tabulate( [(src, len(individuals)) for src, individuals in source_dict_individuals.items()]))


def verify_unified_metadata(people: List[MetadataComplete]):
    """Verify that the Unified Person has AT LEAST ONE value for each metadata property

    Args:
        people (List[MetadataComplete]): List of unified Biographies (one row per person)
    """

    def _generate_ranked_dfs(df: pd.DataFrame, fields_to_rank_by: List[str], file_suffix: str, ascending: bool = True):

        df.sort_values(by=fields_to_rank_by, ascending=ascending, inplace=True)
        df.to_csv(f"{SAVE_RESULTS_PATH}/ranking_metafield_{file_suffix}.tsv", sep='\t')

        for part in ['dev', 'test', 'train']:
            filtered_df = df[df[f"src_with_text_{part}"].str.len() > 0].\
                filter(axis='columns', items=["person_id", "name_long","src_with_text", f"src_with_text_{part}", f"id_with_text_{part}", "meta_fields_populated", "texts"])
            filtered_df.to_csv(f"{SAVE_RESULTS_PATH}/ranked_meta_{file_suffix}_{part}.tsv", sep='\t')

            ranked_filenames = []
            for index, row in filtered_df.iterrows():
                if len(row[f"id_with_text_{part}"]) > 0:
                    for version in row[f"id_with_text_{part}"].split(","):
                        ranked_filenames.append(f"{index}_{version}")
            pd.DataFrame(ranked_filenames).to_csv(f"{SAVE_RESULTS_PATH}/ranked_filenames_{file_suffix}_{part}.tsv", index=False, header=False)


    data = []

    # f_orig = open('all_dates_original.txt', 'w')
    # f_extracted = open('all_dates_extract.txt', 'w')

    for p in people:
        assert len(p.texts) == len(p.sources)
        partitions_with_text = []
        for i, part in enumerate(p.partitions):
            if p.texts[i] and len(p.texts[i]) > 0:
                partitions_with_text.append((i, part, p.sources[i]))
        metadata_counts = {
                            'person_id': p.person_id, 
                            'name_long': p.getName('unique_longest'), 
                            'instances': len(p.sources), 
                            'src_with_text_train': ",".join([s for (i,part,s) in partitions_with_text if part == "train"]),
                            'src_with_text_dev': ",".join([s for (i,part,s) in partitions_with_text if part == "development"]),
                            'src_with_text_test': ",".join([s for (i,part,s) in partitions_with_text if part == "test"]),
                            "id_with_text_train": ",".join([p.versions[i] for (i,part,_) in partitions_with_text if part == "train"]),
                            "id_with_text_dev": ",".join([p.versions[i] for (i,part,_) in partitions_with_text if part == "development"]),
                            "id_with_text_test": ",".join([p.versions[i] for (i,part,_) in partitions_with_text if part == "test"])
                            }
        has_birthdate = 1 if p.getBirthDate() else 0
        has_predicted_birthyear =  1 if p.getBirthDate_baseline1() > 0 else 0
        has_deathdate = 1 if p.getDeathDate() else 0
        text_count = sum([1 if t else 0 for t in p.texts])
        has_gender = 1 if p.getGender() else 0
        religion_counts = len(p.getReligion('all_religions'))
        metadata_counts['names'] = len(p.getName('all_names'))
        metadata_counts['texts'] = text_count
        metadata_counts['birth'] = has_birthdate
        metadata_counts['death'] = has_deathdate
        metadata_counts['birth_completed'] = 1 if (has_birthdate > 0 or has_predicted_birthyear > 0) else 0
        metadata_counts['gender'] = has_gender
        metadata_counts['religion'] = religion_counts
        occupations = p.getOccupation('all_occupations')
        metadata_counts['occupation'] = 0 if not occupations else len(occupations)
        residences = p.getResidence('all_residences')
        metadata_counts['residence'] = 0 if not residences else len(residences)

        has_religion = 1 if len(p.getReligion('all_religions')) > 0 else 0
        has_occupation = 0 if metadata_counts['occupation'] == 0 else 1
        has_residence = 0  if metadata_counts['residence'] == 0 else 1
        has_multi_text = 0 if text_count <= 1 else 1
        metadata_counts['meta_fields_populated'] = has_multi_text + metadata_counts['birth_completed'] + has_deathdate + has_religion + has_gender + has_occupation + has_residence
        data.append(metadata_counts)

        # # Debugging Dates
        # dates_available = p.getBirthDate('all_valid_dates')
        # date_metadata = p.getBirthDate()
        # date_from_text = p.getBirthDate_baseline1()
        # if len(dates_available) == 0 and p.birthDate_is_intext():
        #     f_orig.write(f"{str(dates_available)}\t{p.birthDate_is_intext()}\n")
        #     f_extracted.write(f"{str(date_metadata)}\t{date_from_text}\n")

    df = pd.DataFrame(data)
    for metadata_label in metadata_counts.keys():
        group_count = df.value_counts([metadata_label], normalize=True)
        print(f"\n===============")
        print(group_count)
    
    # Produce the ranking according to the "Best Populated"
    _generate_ranked_dfs(df, fields_to_rank_by=["meta_fields_populated", "texts"], file_suffix="best_populated", ascending=False)

    # Put first the ones without birthdate ! 
    _generate_ranked_dfs(df, fields_to_rank_by=["meta_fields_populated", "texts"], file_suffix="by_birthdate", ascending=True)


def collect_global_info(people: List[MetadataComplete]) -> Dict[str, Counter]:
    occupations, locations, organizations = [], [], []
    persons, misc_ents = [], []
    no_occup, one_occup, mult_ocupp = 0, 0, 0
    for p in people:
        if len(p.occupations) == 0: 
            no_occup += 1
        elif len(p.occupations) == 1:
            one_occup += 1
        else:
            mult_ocupp += 1
        for occ in p.occupations:
            occupations.append(occ.label.title())
        for ent_list in p.texts_entities:
            for ent in ent_list:
                if ent['label'] == 'LOC':
                    locations.append(ent['text'].title())
                elif ent['label'] == 'ORG':
                    organizations.append(ent['text'].title())
                elif ent['label'] == 'PER':
                    persons.append(ent['text'].title())
                else:
                    misc_ents.append(ent['text'].title())

    print(f"Distribution of People with Occupations:\n NO-METADATA: {no_occup}\nONE OCCUPATION: {one_occup}\nMULTIPLE-OCCUPATIONS: {mult_ocupp}")

    # Create a more Coarse-grained occupation labels (e.g. "advocaat en procureur te Amsterdam") -> "Advocaat"
    one_tok_threshold = 2 # Only Keep one-token occupations that occur at least <one_tok_threshold> times!
    occupations_counter = Counter(occupations)
    _counter_to_file(occupations_counter, f'{SAVE_RESULTS_PATH}/occupations_uncompressed_all.txt')
    occup2coarse_converter = {}
    one_token_occupations = [occ for occ in occupations if len(occ.split()) == 1 and occupations_counter[occ] >= one_tok_threshold]
    _counter_to_file(Counter(one_token_occupations), f'{SAVE_RESULTS_PATH}/occupations_one_token.txt')
    coarse_occupations_counter = {occup: {} for occup in one_token_occupations}
    coarse_occupations_counter["OTHER"] = {}

    own_category_threshold = 10
    for occ, freq in occupations_counter.items():
        if occ in one_token_occupations:
            occup2coarse_converter[occ] = occ
            coarse_occupations_counter[occ][occ] = freq
        else:
            coarse_opts = [tok for tok in occ.split() if tok in one_token_occupations]
            if len(coarse_opts) > 0: # Attach it to their corresponding one-token occupation
                for opt in coarse_opts:
                    if occ in coarse_occupations_counter[opt].keys():
                        coarse_occupations_counter[opt][occ] += freq
                    else:
                        coarse_occupations_counter[opt][occ] = freq
            elif freq >= own_category_threshold: # The occupation is multi-word but occurs a lot so it deserves its own category
                coarse_occupations_counter[occ] = {occ: freq}
            else: # Add the long option to 'OTHER' key since it is rare or unknown
                if occ in coarse_occupations_counter["OTHER"].keys():
                    coarse_occupations_counter["OTHER"][occ] += freq
                else:
                    coarse_occupations_counter["OTHER"][occ] = freq

    # The occupations file actually has the coarse_occupations, to make the search "Prettier"
    with open(f'{SAVE_RESULTS_PATH}/occupations_mapper.json', "w") as fout:
        json.dump(coarse_occupations_counter, fout, indent=2,  ensure_ascii=False)

    expanded_coarse_counter = {}
    for occ, fine_grained_dict in coarse_occupations_counter.items():
        tot_freq = 0
        for key, freq in fine_grained_dict.items():
            tot_freq += freq
        expanded_coarse_counter[occ] = tot_freq
    _counter_to_file(Counter(expanded_coarse_counter), 'flask_app/backend_data/occupations.txt')
    _counter_to_file(Counter(expanded_coarse_counter), f'{SAVE_RESULTS_PATH}/occupations_coarse.txt')

    # Iterate again per person to get a list of their "coarse" occupations to later generate correlation charts or matrices
    occupations_coarse_list = []
    for p in people:
        occupations_coarse_list.append(occup2coarse_converter.get(p.getOccupation(), None))

    locations = Counter(locations)
    _counter_to_file(locations, 'flask_app/backend_data/locations.txt')
    _counter_to_file(locations, f'{SAVE_RESULTS_PATH}/ner_locations_all.txt')

    organizations = Counter(organizations)
    # _counter_to_file(organizations, 'flask_app/backend_data/organizations.txt')
    _counter_to_file(locations, f'{SAVE_RESULTS_PATH}/ner_organizations_all.txt')
    _counter_to_file(Counter(persons), f'{SAVE_RESULTS_PATH}/ner_persons_all.txt')
    _counter_to_file(Counter(misc_ents), f'{SAVE_RESULTS_PATH}/ner_misc_all.txt')
    return {'occupations_coarse_list': occupations_coarse_list, 'occup2coarse': occup2coarse_converter, 'locations': locations, 'organizations': organizations}


def gender_analysis(people: List[MetadataComplete]) -> List[str]:
    genders_all = []
    years_born_all = []
    for p in people:
        genders_all.append(p.getGender())
        b_date = p.getBirthDate()
        if b_date:
            byear = b_date[0]
        else:
            byear = -1
        if byear == 5550: print(p.person_id)
        if byear != -1:
            years_born_all.append(byear)
        else:
            years_born_all.append(0)
    print("\n\n")
    print(Counter(genders_all).most_common())

    genders_all_with_unk = [g or "unk" for g in genders_all]

    df = pd.DataFrame({'year_born': years_born_all, 'gender': genders_all_with_unk}) #.dropna()
    # y_range = [y for y in years_born_all if 0 < y < 1980]
    # bins = range(min(y_range), max(y_range), 10)
    y_min = 1400
    y_max = 1960
    bins = range(y_min, y_max, 10)
    groups = df.groupby(['gender', pd.cut(df['year_born'], bins)])
    grouped_df = groups.size().unstack().transpose()

    grouped_df['year_label'] = [interval.left for interval in list(grouped_df.index)]
    grouped_df.set_index('year_label', drop=True, inplace=True)
    grouped_df.to_csv(f"{SAVE_RESULTS_PATH}/gender_decades.csv")
    
    # These are adjustements for nicer plotting
    ax = grouped_df.plot.line(subplots=False, rot=90, xticks=range(y_min, y_max, 20), fontsize=8)
    #plot_selection = list(grouped_df.index)[-50:]
    #ax = grouped_df.filter(items=plot_selection, axis='index').plot.line(subplots=False, rot=45, xticks=range(0,len(plot_selection),5), fontsize=8)
    plt.savefig(f"{SAVE_RESULTS_PATH}/gender_decades.png")

    bins = range(900, 2100, 100)
    print([b for b in bins])
    centuries = ['X','XI','XII','XIII','XIV','XV','XVI','XVII', 'XVIII', 'XIX', 'XX']
    groups = df.groupby(['gender', pd.cut(df['year_born'], bins, labels=centuries)])
    grouped_df = groups.size().unstack().transpose()
    grouped_df.to_csv(f"{SAVE_RESULTS_PATH}/gender_centuries.csv")
    grouped_df.plot.bar()
    plt.savefig(f"{SAVE_RESULTS_PATH}/gender_centuries.png")

    return genders_all



def group_per_century(people: List[MetadataComplete], print_stats: bool = True, include_inferred_dates: bool = False) -> Dict[str, List[MetadataComplete]]:
    """
    Args:
        people (List[MetadataComplete]): The full list of people
        print_stats (bool) = True: Show the distribution in the returned dictionary as a table
        include_inferred_dates (bool) = False: If True, then the date will be 'autocompleted' using the texts, based on a heuristic method 
    Returns:
        Dict[str, List[MetadataComplete]]: People grouped by century, based on the year they were born (using MetaData only)
    """
    century_dict = OrderedDict({
        (0, 1399): [],
        #(1000, 1099): [],
        #(1100, 1199): [],
        #(1200, 1299): [],
        #(1300, 1399): [],
        (1400, 1499): [],
        (1500, 1599): [],
        (1600, 1699): [],
        (1700, 1799): [],
        (1800, 1899): [],
        (1900, 1999): [],
        (-1, -1): []
    })
    for p in people:
        b_date = p.getBirthDate('most_likely_date')
        if b_date:
            year = b_date[0]
        else:
            year = -1
        if year < 0 and include_inferred_dates:
            year = p.getBirthDate_baseline1()
        for k in century_dict.keys():
            if k[0] <= year <= k[1]:
                century_dict[k].append(p)
    
    if print_stats:
        total_stats = [(k[0], k[1], len(v)) for k, v in century_dict.items()]
        print('\n',tabulate(total_stats, headers=(("From", "To", "People"))))

    return century_dict



def get_crosstab_cats(index: List[Any], columns: List[Any], values: List[Any], val_sort: str, row_limit: int, separate_vals: bool = False) -> pd.DataFrame:
    cat1 = 'occupation'
    cat2 = 'gender'
    df = pd.crosstab(index, columns, values=values, rownames=[cat1], colnames=[cat2], aggfunc=sum)
    df = df.fillna(0).sort_values(by=val_sort, ascending=False)
    df.to_csv(f"{SAVE_RESULTS_PATH}/{cat1}_{cat2}_counts.csv")

    if separate_vals:
        cat_vals = list(df.columns)
        for cv in cat_vals:
            df['_'] = df[cv]
            df.sort_index()
            df_val = df.loc[:, ('_',cv)]
            ax = sns.heatmap(df_val, cmap='YlGnBu', robust=True)
            plt.savefig(f"{SAVE_RESULTS_PATH}/{cat1}_val-{cv}_matrix.png")
    else:
        ax = sns.heatmap(df.iloc[:row_limit], cmap='YlGnBu', robust=True)
        plt.savefig(f"{SAVE_RESULTS_PATH}/{cat1}_{cat2}_matrix.png")
    return df
        



def evaluate_inferred_birth_dates(people: List[MetadataComplete]):
    true_births = [] # Here we store the birth_date as expressed in the metadata
    birth_in_text = []
    baseline1_births = [] # Here we put a simple baseline: The first \d{4} or \d{3} found in the text is proposed as the birth_date
    baseline2_births = [] # Any \d{4} or \d{3} that appears around (+-10) "geboren/geb."
    baseline_timex_births = [] # Baseline with HeidelTime Timexp
    skept = 0

    def _eval_births(gold_births, predicted_births):
        tp, fp, tn, fn = 0, 0, 0, 0
        for gb, pb in zip(gold_births, predicted_births):
            if gb == -1 and pb == -1:
                tn += 1
            elif gb == pb:
                tp += 1
            elif pb == -1:
                fn += 1
            else:
                fp += 1
        if tp+fp == 0:
            prec = 0
        else:
            prec = tp/(tp+fp) * 100
        if tp+fn == 0:
            rec = 0
        else:
            rec = tp / (tp+fn) * 100
        print(f"Precision = {prec:.2f} | Recall = {rec:.2f}")
        return True

    for p in people:
        birth_date = p.getBirthDate()
        if birth_date:
            birth_year = birth_date[0]
            true_births.append(birth_year)
            # Test if Metadata Birth Year actually appears in any of the Biography Texts...
            if p.birthDate_is_intext(): 
                birth_in_text.append(birth_year)
            else:
                birth_in_text.append(-1)
            # Baseline 1
            baseline1_births.append(p.getBirthDate_baseline1())
            # Baseline with HeidelTime Timexp
            baseline_timex_births.append(p.getBirthDate_from_timexps())
            # Baseline 2
            baseline2_births.append(p.getBirthDate_baseline2())

        else:
            skept += 1
    
    # This accuracy is evaluated only on the data points for which we have metadata available. Otherwise we skip!
    print("\nBaseline Method for finding Birth Dates ...")
    print(f"SKEPT {skept} instances (UNK metadata date). Analysis is on the {len(true_births)} with valid metadata")
    correct = accuracy_score(true_births, baseline1_births, normalize=False)
    print(f"Accuracy Total = {accuracy_score(true_births, baseline1_births)*100:.2f}% ({correct} / {len(true_births)})")
    _eval_births(true_births, baseline1_births)
    print(f"Metadata Date is present IN TEXT only {accuracy_score(true_births, birth_in_text)*100:.2f}% of the time!")
    print(f"Accuracy when Date is Present {accuracy_score(birth_in_text, baseline1_births)*100:.2f}%")
    _eval_births(birth_in_text, baseline1_births)
    print(f"Accuracy when Date is Present (TIMEX) {accuracy_score(birth_in_text, baseline_timex_births)*100:.2f}%")
    _eval_births(birth_in_text, baseline_timex_births)
    print(f"Accuracy when Date is Present (Baseline geb.) {accuracy_score(birth_in_text, baseline2_births)*100:.2f}%")
    _eval_births(birth_in_text, baseline2_births)


def evaluate_inferred_gender(people: List[MetadataComplete]):
    true_gender, pred_gender1, pred_gender2 = [], [], []
    fem_names, masc_names, all_names = [], [], []

    original_gender, predicted_gender_real, predicted_gender_unk_only = [], [], []

    for p in people:
        # Gender Info
        gender = p.getGender()
        predicted_gen = p.getGender_predicted_first_pronoun()
        if gender:
            pred_gender1.append(p.getGender_predicted_pronoun_votes())
            pred_gender2.append(p.getGender_predicted_first_pronoun())
            true_gender.append(gender)
        # Name Info
        name_onegrams = ["<S>"] + p.getName('unique_longest').split() + ["</S>"]
        name_bigrams = ["_".join(n) for n in ngrams(name_onegrams,2)]
        name_trigrams = ["_".join(n) for n in ngrams(name_onegrams,3)]
        all_names += name_onegrams
        all_names += name_bigrams
        all_names += name_trigrams
        if gender == 'female':
            fem_names += name_onegrams
            fem_names += name_bigrams
            fem_names += name_trigrams
            predicted_gender_unk_only.append('female')
        if gender == 'male':
            masc_names += name_onegrams
            masc_names += name_bigrams
            masc_names += name_trigrams
            predicted_gender_unk_only.append('male')
        elif predicted_gen == 'female':
            fem_names += name_onegrams
            fem_names += name_bigrams
            fem_names += name_trigrams
            predicted_gender_unk_only.append(predicted_gen)
        elif predicted_gen == 'male':
            masc_names += name_onegrams
            masc_names += name_bigrams
            masc_names += name_trigrams
            predicted_gender_unk_only.append(predicted_gen)

        # For general Stats after using the predictor
        original_gender.append(gender)
        predicted_gender_real.append(predicted_gen)
        

    # Evaluation of Gender Predictions
    print("-------- Predicted Gender Evaluation (First Pronoun) --------\n\n")
    print(classification_report(true_gender, pred_gender2, labels=['male', 'female']))
    print("\n")
    print(confusion_matrix(true_gender, pred_gender2))
    print("\n")
    print("-------- Predicted Gender Evaluation (Votes) --------\n\n")
    print(classification_report(true_gender, pred_gender1, labels=['male', 'female']))
    print("\n")
    print(confusion_matrix(true_gender, pred_gender1))
    print("\n")

    print("From Metadata:", Counter(original_gender).most_common())
    print("Adding for UNK Only:", Counter(predicted_gender_unk_only).most_common())
    print("After Predicting All:", Counter(predicted_gender_real).most_common())

    # Name Explorer
    pd.DataFrame(Counter(all_names).most_common()).to_csv("names_all_ngrams.csv")
    pd.DataFrame(Counter(fem_names).most_common()).to_csv("names_fem_ngrams.csv")
    pd.DataFrame(Counter(masc_names).most_common()).to_csv("names_masc_ngrams.csv")
    
    
    

def show_lifespans_per_century(people_dict: Dict[str, List[MetadataComplete]]) -> Dict[str, List[int]]:
    life_spans_dict = defaultdict(list)
    for century, people in people_dict.items():
        if century == (-1, -1): continue
        for p in people:
            birth_date = p.getBirthDate('most_likely_date')
            death_date = p.getDeathDate('most_likely_date')
            try:
                birth_year = birth_date[0]
                death_year = death_date[0]
                lifespan = int(death_year) - int(birth_year)
                if lifespan > 0 and lifespan < 110:
                    life_spans_dict[century].append(lifespan)
                else:
                    pass
                    #print("SPAN ERR",lifespan, "Birth:",p.births, "Death:",p.deaths, p.getName())
            except:
                #print("SPAN ERR",lifespan, "Birth:",p.births, "Death:",p.deaths, p.getName())
                continue
    return life_spans_dict


def get_network_of_person_mentions(people: List[MetadataComplete]):
    people_network = nx.Graph()
    all_bionet_mentions = []
    connected_people = defaultdict(set)
    name2id, id2names = build_names_dictionaries(people) # This dictionaries are built based on the metadata names
    with open(f"{SAVE_RESULTS_PATH}/bionet_id2names.json", "w") as f:
        json.dump(id2names, f, indent=2,  ensure_ascii=False)
    for p in people:
        # Keep Global track of "popularity" measured as PER mentions in all texts
        related_mentions = []
        for text_ents in p.texts_entities:
            persons = [ent['text'] for ent in text_ents if ent['label'] == 'PER'] # ent ~ {'text': 'Amsterdam', 'label': 'LOC', 'start': 70, 'end': 79, 'start_token': 14, 'end_token': 15}
            related_mentions += persons
        related_mentions = set(related_mentions)
        related_mentions = [m for m in related_mentions if " " in m] # Drop 'single name' mentions since they tend to be noisy
        all_bionet_mentions += related_mentions
        # print(f"------ {p.getName()} -------\n{related_mentions}")
        
        # Build a Simple Network Based on Person Mentions as recognized by the metadata Names Dictionary
        for rel in related_mentions:
            id_rel = name2id.get(rel.lower())
            if id_rel and id_rel != p.person_id: # Do NOT add self-mentions 
                connected_people[p.person_id].add(id_rel)
                people_network.add_edge(p.person_id, id_rel)

    print("\n\n---------- People who are mentioned the most across texts ----------")
    popularity_counter = Counter(all_bionet_mentions).most_common(1000)
    print(tabulate(popularity_counter, headers=(['Name', 'Freq'])))

    print("\n\n---------- Disambiguated People connected to the Biographee (showing Name but we have the ID) ----------- ")
    for person_id, connected in sorted(connected_people.items(), key= lambda x: len(x[1]), reverse=True)[:100]:
        person_name = id2names[person_id][0]
        print(f"\n{person_name} --> {[id2names[pid][0] for pid in connected]}")
    
    print(f"Connected a total of {len(connected_people)} people")

    # TODO: Proper Network Analysis using the NetworkX Library
    # print(nx.info(people_network))
    # fig, ax = plt.subplots(figsize=(15, 9))
    # ax.axis("off")
    # pos = nx.spring_layout(people_network, iterations=15, seed=1721)
    # plot_options = {"node_size": 10, "with_labels": False, "width": 0.15}
    # nx.draw_networkx(people_network, pos=pos, ax=ax, **plot_options)
    # plt.show()



def build_names_dictionaries(people: List[MetadataComplete]) -> Dict[str, str]:
    """
        Builds 2 dictionaries mapping the ID's to all recorded names in the metadata and viceversa. 
        EXAMPLE: name2id = {'clara engelen': '43856095'}  ||  id2names = {'43856095': ['clara engelen', 'clara m. engelen']} 
    """
    name2id, id2names = {}, defaultdict(list)
    for p in people:
        all_names = [n.lower() for n in p.getName(mode='all_names')]
        for n in all_names:
            name2id[n] = p.person_id
            id2names[p.person_id].append(n)
    return name2id, id2names




def _counter_to_file(c: Counter, filepath: str, threshold: int = -1):
    with open(filepath, 'w') as fout:
        for element, count in c.most_common():
            if threshold > 0 and count >= threshold:
                fout.write(f"{element}\t{count}\n")
            else:
                fout.write(f"{element}\t{count}\n")


def get_file_from_id(id: str, partition: str, json_parent_path: str = "data/json/"):
        full_path = f"{json_parent_path}/{partition}/{id}.json"
        return json.load(full_path)

if __name__ == '__main__':
    main()