from lxml import etree
from pyparlaclarin.read import paragraph_iterator
from pyriksdagen.utils import protocol_iterators
import nltk    
import spacy
import re
import os
 
nltk.download("stopwords")


def get_protocols(start_date=1971, end_date=2000):
    """Returns a list of protocols"""
    protocols = list(protocol_iterators("corpus/protocols/", start=start_date, end=end_date))
    return protocols

def parse_protocol(protocol):
    """Gets a single protocol, parses it and returns the root"""
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    return root

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

def preprocess_raw(ls):
    """Gets a list of strings and performs the preprocessing on it"""
    ls = remove_whitespace(ls)
    ls = tokenize(ls)
    return ls

def remove_letters(ls):
    """removes single letters"""
    return [item for item in ls if len(item) > 1]

def preprocess(ls):
    """Gets a list of strings and performs the preprocessing on it"""
    ls = remove_whitespace(ls)
    ls = remove_special_characters(ls)
    ls = tokenize(ls)
    ls = to_lowercase(ls)
    ls = remove_stopwords(ls)
    ls = remove_letters(ls)
    return ls

def postprocess_lemmatize(input_directory, output_directory):
    """Applies POS tagging to keep only nouns, verbs (+ auxilary verbs), and adjectives"""
    nlp = spacy.load("sv_core_news_sm", disable=['ner', 'parser', 'textcat'])
    for file in os.listdir(input_directory):
        filepath = os.path.join(input_directory, file)
        with open(filepath, "r") as f:
            text = f.read()
            # now lemmatize
            doc = nlp(text)
            filtered_text = [item.lemma_ for item in doc]
        testpath = os.path.join(output_directory, file)
        with open(testpath, "w", encoding="utf-8") as f:
            for item in filtered_text:
                f.write(item + " ")
            
def postprocess_filter(input_directory, output_directory):
    """Applies POS tagging to keep only nouns, verbs (+ auxilary verbs), and adjectives"""
    to_keep = ["NOUN", "ADJ", "VERB", "AUX"]
    nlp = spacy.load("sv_core_news_sm", disable=['ner', 'parser', 'textcat'])
    for file in os.listdir(input_directory):
        filepath = os.path.join(input_directory, file)
        with open(filepath, "r") as f:
            text = f.read()
            # now filter based on the POS tagging
            doc = nlp(text)
            filtered_text = [item.text for item in doc if item.pos_ in to_keep]
        testpath = os.path.join(output_directory, file)
        count = 1
        with open(testpath, "w", encoding="utf-8") as f:
            for item in filtered_text:
                if count % 20 == 0:
                    f.write(item + "\n")
                else:
                    f.write(item + " ")
                count+=1

def run_raw(start_date=1971, end_date=2000):
    """preprocesses the data by only stripping white space and tokenizing the words using *preprocess_raw*"""
    protocols = get_protocols(start_date, end_date)
    num_protocol = 0
    for protocol in protocols:
        num_protocol += 1
        root = parse_protocol(protocol)
        ls = []
        for elem in list(paragraph_iterator(root, output="lxml")):
            ls.append(" ".join(elem.itertext()))
        text = preprocess_raw(ls)

        count = 1
        with open(f"preprocessed_raw/preprocessed_text_{start_date}_{end_date}_{num_protocol}.txt", "w", encoding="utf-8") as file:
            for item in text:
                if count % 20 == 0:
                    file.write(item + "\n")
                else:
                    file.write(item + " ")
                count+=1

def run_main(start_date=1971, end_date=2000, foldername="preprocessed"):
    """preprocesses the data the standard way"""
    protocols = get_protocols(start_date, end_date)

    protocols = protocols[:10] # TODO: Remove

    num_protocol = 0
    for protocol in protocols:
        protocol_year = protocol.split("/")[2] # extract year from file path
        num_protocol += 1
        root = parse_protocol(protocol)
        ls = []
        for elem in list(paragraph_iterator(root, output="lxml")):
            ls.append(" ".join(elem.itertext()))
        text = preprocess(ls)

        count = 1
        with open(f"{foldername}/preprocessed_text_{protocol_year}_{num_protocol}.txt", "w", encoding="utf-8") as file:
            for item in text:
                if count % 20 == 0:
                    file.write(item + "\n")
                else:
                    file.write(item + " ")
                count+=1


if __name__=="__main__":
    start_date = 1971
    end_date = 2000    
        
    run_main(start_date, end_date, "preprocessed/")
    print("starting lemmatizing")
    postprocess_lemmatize("preprocessed/", "preprocessed/")
    
  #  run_main(start_date, end_date, "preprocessed_filter")
    print("started filtering")
    postprocess_filter("preprocessed/", "preprocessed_filter/")