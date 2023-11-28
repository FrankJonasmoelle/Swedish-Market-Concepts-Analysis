from lxml import etree
from pyriksdagen.utils import protocol_iterators
from concurrent.futures import ProcessPoolExecutor
import nltk    
import spacy
import re
import os
 
nltk.download("stopwords")


def get_protocols(start_date=1971, end_date=2010):
    """Returns a list of protocols"""
    protocols = list(protocol_iterators("corpus/protocols/", start=start_date, end=end_date))
    return protocols

def parse_protocol(protocol):
    """Gets a single protocol, parses it and returns the root"""
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    return root

def parse_speeches(protocol):
    """In comparison to parse_protocol, this function returns only the speeches (defined by <u> tags)"""
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    u_texts = root.findall(".//{http://www.tei-c.org/ns/1.0}u")
    ls_u_texts = []
    for u_tag in u_texts:
        ls_u_texts.append(" ".join(u_tag.itertext()))
    return ls_u_texts

def get_protocol_year(protocol):
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    doc_dates = root.findall(".//{http://www.tei-c.org/ns/1.0}docDate")
    ls_doc_dates = [doc_date.text for doc_date in doc_dates if doc_date.text]
    date = ls_doc_dates[0]
    year = date[:4]
    return year

def remove_whitespace(ls):
    return [item.strip() for item in ls]

def remove_special_characters(ls):
    """removes all special characters and puncations"""
    return [re.sub(r'[^a-zA-ZåäöÅÄÖ\s]', '', item) for item in ls]

def to_lowercase(ls):
    return [item.lower() for item in ls]

def tokenize(ls):
    ls = [item.split() for item in ls]
    flattened_ls = []
    for token in ls:
        flattened_ls.extend(token) # flattens the 'list of lists'
    return flattened_ls

def remove_stopwords(ls):
    """remove words like 'och', 'att', etc."""
    STOPWORDS_SWEDISH = nltk.corpus.stopwords.words('swedish') 
    return [item for item in ls if item not in STOPWORDS_SWEDISH]

def remove_letters(ls):
    """removes single letters"""
    return [item for item in ls if len(item) > 1]

def preprocess_raw(ls):
    """Gets a list of strings and performs the preprocessing on it"""
    ls = remove_whitespace(ls)
    ls = tokenize(ls)
    return ls

def preprocess(ls):
    """Gets a list of strings and performs the preprocessing on it"""
    ls = remove_whitespace(ls)
    ls = remove_special_characters(ls)
    ls = tokenize(ls)
    ls = to_lowercase(ls)
    ls = remove_stopwords(ls)
    ls = remove_letters(ls)
    return ls

def lemmatize_helper(input_directory, output_directory, file, nlp):
    input_file_path = os.path.join(input_directory, file)
    output_file_path = os.path.join(output_directory, file)

    if not os.path.exists(output_file_path):
        with open(input_file_path, "r", encoding="utf-8") as f:
            text = f.read()
            doc = nlp(text)
            lemmatized_text = [item.lemma_ for item in doc]
        with open(output_file_path, "w", encoding="utf-8") as f:
            for item in lemmatized_text:
                f.write(item + " ")
    else:
        print("file was already lemmatized")

def postprocess_lemmatize(input_directory, output_directory):
    nlp = spacy.load("sv_core_news_sm", disable=['ner', 'parser', 'textcat'])
    with ProcessPoolExecutor() as executor:
        futures = []
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                if file.endswith(".txt"):
                    output_dir = output_directory + "/" + root.split("/")[1]
                    input_dir = root 
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    future = executor.submit(lemmatize_helper, input_dir, output_dir, file, nlp)
                    futures.append(future)
            # Wait for all tasks to complete
            for future in futures:
                future.result()
                
def filter_helper(input_directory, output_directory, file, nlp):
    to_keep = ["PROPN", "NOUN", "ADJ", "VERB", "AUX"]
    input_file_path = os.path.join(input_directory, file)
    output_file_path = os.path.join(output_directory, file)

    if not os.path.exists(output_file_path):
        with open(input_file_path, "r", encoding="utf-8") as f:
            text = f.read()
            doc = nlp(text)
            filtered_text = [item.text for item in doc if item.pos_ in to_keep]
        with open(output_file_path, "w", encoding="utf-8") as f:
            for item in filtered_text:
                f.write(item + " ")
    else:
        print("file was already filtered")

def postprocess_filter(input_directory, output_directory):
    nlp = spacy.load("sv_core_news_sm", disable=['ner', 'parser', 'textcat'])
    with ProcessPoolExecutor() as executor:
        futures = []
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                if file.endswith(".txt"):
                    output_dir = output_directory + "/" + root.split("/")[1]
                    input_dir = root 
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    future = executor.submit(filter_helper, input_dir, output_dir, file, nlp)
                    futures.append(future)
        # Wait for all tasks to complete
        for future in futures:
            future.result()

def run_raw(start_date=1971, end_date=2010, foldername="preprocessed_raw"):
    """preprocesses the data by only stripping white space and tokenizing the words using *preprocess_raw*"""
    protocols = get_protocols(start_date, end_date)
    for protocol in protocols:
        #  protocol_year = protocol.split("/")[2] # extract year from file path
        protocol_filename = protocol.split("/")[3] # extract file name
        year = get_protocol_year(protocol)
        if 1971 <= int(year) <= 1980:
            folder_year = "1970s"    
        elif 1981 <= int(year) <= 1990:
            folder_year = "1980s"
        elif 1991 <= int(year) <= 2000:
            folder_year = "1990s"
        elif 2001 <= int(year) <= 2010:
            folder_year = "2000s"
        if not os.path.exists(foldername+"/"+folder_year):
            os.makedirs(foldername+"/"+folder_year)

        filepath = f"{foldername}/{folder_year}/preprocessed_{year}_{protocol_filename}.txt"
        if not os.path.exists(filepath):
            speech_ls = parse_speeches(protocol)
            text = preprocess_raw(speech_ls)
            count = 1
            with open(filepath, "w", encoding="utf-8") as file:
                for item in text:
                    if count % 20 == 0:
                        file.write(item + "\n")
                        if count % 200 == 0: # add line break for readability
                            file.write("\n")
                    else:
                        file.write(item + " ")
                    count+=1
        else:
            print("file already exists")

def run_main(start_date=1971, end_date=2010, foldername="preprocessed"):
    """preprocesses the data the standard way"""
    protocols = get_protocols(start_date, end_date)
    for protocol in protocols:
        #protocol_year = protocol.split("/")[2] # extract year from file path
        protocol_filename = protocol.split("/")[3] # extract file name

        year = get_protocol_year(protocol)
        if 1971 <= int(year) <= 1980:
            folder_year = "1970s"    
        elif 1981 <= int(year) <= 1990:
            folder_year = "1980s"
        elif 1991 <= int(year) <= 2000:
            folder_year = "1990s"
        elif 2001 <= int(year) <= 2010:
            folder_year = "2000s"
        if not os.path.exists(foldername+"/"+folder_year):
            os.makedirs(foldername+"/"+folder_year)

        filepath = f"{foldername}/{folder_year}/preprocessed_{year}_{protocol_filename}.txt"
        if not os.path.exists(filepath):
            speech_ls = parse_speeches(protocol)
            text = preprocess(speech_ls)
            count = 1
            with open(filepath, "w", encoding="utf-8") as file:
                for item in text:
                    if count % 20 == 0:
                        file.write(item + "\n")
                        if count % 200 == 0: # add line break for readability
                            file.write("\n")
                    else:
                        file.write(item + " ")
                    count+=1
        else:
            print("file already exists")

if __name__=="__main__": 
    start_date = 1971
    end_date = 2010
    #run_raw(start_date, end_date, "preprocessed_raw")
    #run_main(start_date, end_date, "preprocessed/")
    # print("started lemmatizing")
    # postprocess_lemmatize("preprocessed/", "preprocessed_lemmatized/")
    # print("started filtering")
    postprocess_filter("preprocessed_lemmatized/", "preprocessed_filtered/")