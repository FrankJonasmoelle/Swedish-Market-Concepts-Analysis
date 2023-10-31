from lxml import etree
import progressbar, argparse
from pyparlaclarin.read import paragraph_iterator, speeches_with_name
from pyriksdagen.utils import protocol_iterators
import nltk          
import pickle

nltk.download("stopwords")
STOPWORDS_SWEDISH = nltk.corpus.stopwords.words('swedish') 

 
def get_protocols(start_date=1971, end_date=2000):
    """Returns a list of protocols"""
    protocols = list(protocol_iterators("protocols/", start=start_date, end=end_date))
    return protocols


def parse_protocol(protocol):
    """Gets a single protocol, parses it and returns the root"""
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.parse(protocol, parser).getroot()
    return root


def remove_special_characters():
    pass

def remove_punctiation():
    pass

def remove_whitespace():
    pass

def to_lowercase():
    pass

def tokenize():
    pass

def remove_stopwords():
    pass

def stem_words():
    pass

def lemmatize_words():
    pass


def preprocess(ls):
    """Gets a list of strings and performs the following preprocessing steps on it:
    
    1) Stripping of white space
    2) Tokenization
    3) Change to lower case
    4) Stop word removal
    5) Number removal
    6) Lemmatization
    """
    ls = [item.strip() for item in ls]
    ls = [item.split() for item in ls]
    
    flattened_ls = []
    for token in ls:
        flattened_ls.extend(token) # just flattens the 'list of lists'

    ls = [token.lower() for token in flattened_ls]
    
    ls = [item for item in ls if item not in STOPWORDS_SWEDISH]

    ls = [item for item in ls if not item.isdigit()]

    # TODO: Lemmatization

    # TODO: . and , not really removed
    
    return ls


def preprocess_text(start_date=1970, end_date=1971):
    """Creates a list of protocols and preprocesses the text. 
    Returns a list of strings."""
    
    protocols = get_protocols(start_date, end_date)
    preprocessed_protocols_ls = []
    for protocol in protocols:
        root = parse_protocol(protocol)
        ls = []
        for elem in list(paragraph_iterator(root, output="lxml")):
            ls.append(" ".join(elem.itertext()))
        preprocessed_ls = preprocess(ls)
        preprocessed_protocols_ls.extend(preprocessed_ls)
    return preprocessed_protocols_ls


if __name__=="__main__":
    start_date = 1970
    end_date = 1970
    text = preprocess_text(start_date, end_date)
    
    with open(f"preprocessed_text_{start_date}_{end_date}", "w") as file:
        for item in text:
            file.write(item + "\n")

    #with open(f'preprocessed_text_{start_date}_{end_date}', 'wb') as file:
    #    pickle.dump(text, file)

    #with open ('preprocessed_text_{start_date}_{end_date}', 'rb') as fp:
    #    text = pickle.load(fp)

    #print(text)