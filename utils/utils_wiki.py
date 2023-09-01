import requests
import re
import wikipedia
import time


def get_bionet_person_wikidata(person_bionet_id: str):
    url = 'https://query.wikidata.org/sparql'
    query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT 
        ?subject ?subjectLabel 
        ?property ?propLabel
        ?object ?objectLabel
        WHERE
        {{
        # Substitute here the ID of the person we are looking at 
        ?subject wdt:P651 '{person_bionet_id}' .

        # General wikidata pattern
        ?subject rdfs:label ?subjectLabel .
        ?subject ?propUrl ?object .
        ?property ?ref ?propUrl .
        ?property rdf:type wikibase:Property .
        ?property rdfs:label ?propLabel.

        # Get object properties (if you want to include data properties too you can wrap the next two lines in an OPTIONAL{{}}, but the data quality goes down
        ?object rdfs:label ?objectLabel .
        FILTER (LANG(?objectLabel) = 'en' ) .

        # Filter only English labels
        FILTER (LANG(?subjectLabel) = "en") .
        FILTER (LANG(?propLabel) = 'en' ) .

        }}
        GROUP BY ?subject ?subjectLabel ?property ?propLabel ?object ?objectLabel
        ORDER BY ?subject ?subjectLabel ?property ?propLabel ?object ?objectLabel
        LIMIT 100
        """

    # Call API
    try:
        r = requests.get(url, params={'format': 'json', 'query': query}, timeout=3)
        data = r.json() if r.status_code == 200 else None
    except:
        print("Failed to query Wikidata")
        data = None
    
    tabular_data = []
    data_dict = {}
    if data:
        # Feed data from Wikidata Response
        person_name = None
        for item in data["results"]["bindings"]:
            subject = item["subjectLabel"]["value"]
            predicate = item["propLabel"]["value"]
            object = item["objectLabel"]["value"]
            tabular_data.append((subject, predicate, object))
            person_name = subject
            data_dict[predicate] = object
        # Find Wikipedia Link and Text
        if person_name:
            wiki_page = get_wikipedia_article(person_name)
            if wiki_page:
                tabular_data.append((subject, "Wikipedia URL", wiki_page.url))
                data_dict["wiki_url"] = wiki_page.url
            time.sleep(3)
    return tabular_data, data_dict


def get_wikipedia_article(query_str: str, language: str = "en") -> wikipedia.WikipediaPage:
    """Get a simple query and return ONE non-ambiugous wikipedia article.
    Args:
        query_str (str): The string to query the Wikipedia API for article title suggestions
        query_restrictions: (Dict[str, Any]): Propertied that bound the search beyond the query string (e.g. birth date)
    Returns:
        wiki_article (str): The content of the wikipedia article
    """

    # Return a list of terms that match our (usually and unintentionally) Fuzzy Term!
    wikipedia.set_lang(language)
    page_names = wikipedia.search(query_str, results=3, suggestion=False)
    print(f"Options: {page_names}")

    if page_names and len(page_names) > 0:
        page_name = page_names[0]
        # Now that we have a valid page name, we retrieve the Page Object
        try:
            print(f"\nRetrieving page for {page_name}")
            page = wikipedia.page(page_name, pageid=None, auto_suggest=False, redirect=True, preload=False)
            return page
        except:
            return None
 

def quick_wiki_cleaner(text: str) -> str:
    """ Quick Cleaner that just erases all formating from wikipedia content to return the actual text in a single line script
    Args:
        text (str): Wikipedia Page Content with Section headers and formatting

    Returns:
        str: Full Text without headers and in a single line string
    """
    text = re.sub(r"=+\s.+\s=+", " ", text)
    text = re.sub("\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

