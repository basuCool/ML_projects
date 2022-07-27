
# coding: utf-8

# In[ ]:


import config

DB_TYPE = config.DB_TYPE
DB_COLS = config.DB_COLS
DB_SUBGROUP_COLS = config.DB_SUBGROUP_COLS

TECH_WORDS = config.TECH_WORDS
DICT_TYPE = config.DICT_TYPE
import config

DB_TYPE = config.DB_TYPE
DB_COLS = config.DB_COLS
DB_SUBGROUP_COLS = config.DB_SUBGROUP_COLS

TECH_WORDS = config.TECH_WORDS
DICT_TYPE = config.DICT_TYPE
TECH_DICT_COLS = config.TECH_DICT_COLS
TECH_DICT_TERM_COL = config.TECH_DICT_TERM_COL
TECH_DICT_DEFN_COL = config.TECH_DICT_DEFN_COL

REGEX_TECHNICAL_WORDS = config.REGEX_TECHNICAL_WORDS
TECH_WORDS_REGEX = config.TECH_WORDS_REGEX

CSV_PATH_DB = config.CSV_PATH_DB
CSV_PATH_DICT = config.CSV_PATH_DICT

XLSX_PATH_DB = config.XLSX_PATH_DB
XLSX_PATH_DICT = config.XLSX_PATH_DICT

MY_SQL_PATH_DB = config.MY_SQL_PATH_DB
MY_SQL_USER_DB = config.MY_SQL_USER_DB
MY_SQL_PASSWORD_DB = config.MY_SQL_PASSWORD_DB
MY_SQL_DB_COLUMNS_QUERY = config.MY_SQL_DB_COLUMNS_QUERY

MY_SQL_PATH_DICT = config.MY_SQL_PATH_DICT
MY_SQL_USER_DICT = config.MY_SQL_USER_DICT
MY_SQL_PASSWORD_DICT = config.MY_SQL_PASSWORD_DICT
MY_SQL_DICT_COLUMNS_QUERY = config.MY_SQL_DICT_COLUMNS_QUERY  

MS_SQL_PATH_DB = config.MS_SQL_PATH_DB
MS_SQL_USER_DB = config.MS_SQL_USER_DB
MS_SQL_PASSWORD_DB = config.MS_SQL_PASSWORD_DB
MS_SQL_DB_COLUMNS_QUERY = config.MS_SQL_DB_COLUMNS_QUERY

MS_SQL_PATH_DICT = config.MS_SQL_PATH_DICT
MS_SQL_USER_DICT = config.MS_SQL_USER_DICT
MS_SQL_PASSWORD_DICT = config.MS_SQL_PASSWORD_DICT
MS_SQL_DICT_COLUMNS_QUERY = config.MS_SQL_DICT_COLUMNS_QUERY  

SAP_HANA_DICT_COLUMNS_QUERY = config.SAP_HANA_DICT_COLUMNS_QUERY
DICT_HOST = config.DICT_HOST
DICT_PORT = config.DICT_PORT
DICT_USER = config.DICT_USER
DICT_PASSWORD = config.DICT_PASSWORD

SAP_HANA_DB_COLUMNS_QUERY = config.SAP_HANA_DB_COLUMNS_QUERY
DB_HOST = config.DB_HOST
DB_PORT = config.DB_PORT
DB_USER = config.DB_USER
DB_PASSWORD = config.DB_PASSWORD

SIZE = config.SIZE
WINDOW = config.WINDOW
MIN_COUNT = config.MIN_COUNT
WORKERS = config.WORKERS
EPOCHS = config.EPOCHS

LOW_TO_HIGH_RANKING_ORDER = config.LOW_TO_HIGH_RANKING_ORDER
NUM_OF_RESULTS = config.NUM_OF_RESULTS
ACCURACY_PERCENTILE = config.ACCURACY_PERCENTILE

"""GLOBAL CONFIG"""
PATH_NON_TECH_DICT = config.PATH_NON_TECH_DICT
PATH_TECH_DICT = config.PATH_TECH_DICT
PATH_WORD2VEC_MODEL = config.PATH_WORD2VEC_MODEL
PATH_WORD2VEC_DOCS = config.PATH_WORD2VEC_DOCS
PATH_DB_SUBGROUPED = config.PATH_DB_SUBGROUPED
PATH_OUTPUTS = config.PATH_OUTPUTS
PATH_TEST_DATA = config.PATH_TEST_DATA

LANGUAGE = config.LANGUAGE
AUTOCORRECT_ON = config.AUTOCORRECT_ON
STEMMED_WORDS = config.STEMMED_WORDS

TECH_DICT_COLS = config.TECH_DICT_COLS
TECH_DICT_TERM_COL = config.TECH_DICT_TERM_COL
TECH_DICT_DEFN_COL = config.TECH_DICT_DEFN_COL

REGEX_TECHNICAL_WORDS = config.REGEX_TECHNICAL_WORDS
TECH_WORDS_REGEX = config.TECH_WORDS_REGEX

CSV_PATH_DB = config.CSV_PATH_DB
CSV_PATH_DICT = config.CSV_PATH_DICT

XLSX_PATH_DB = config.XLSX_PATH_DB
XLSX_PATH_DICT = config.XLSX_PATH_DICT

SQL_PATH_DB = config.SQL_PATH_DB
SQL_USER_DB = config.SQL_USER_DB
SQL_PASSWORD_DB = config.SQL_PASSWORD_DB
SQL_DB_COLUMNS_QUERY = config.SQL_DB_COLUMNS_QUERY

SQL_PATH_DICT = config.SQL_PATH_DICT
SQL_USER_DICT = config.SQL_USER_DICT
SQL_PASSWORD_DICT = config.SQL_PASSWORD_DICT
SQL_DICT_COLUMNS_QUERY = config.SQL_DICT_COLUMNS_QUERY  

SAP_HANA_DICT_COLUMNS_QUERY = config.SAP_HANA_DICT_COLUMNS_QUERY
DICT_HOST = config.DICT_HOST
DICT_PORT = config.DICT_PORT
DICT_USER = config.DICT_USER
DICT_PASSWORD = config.DICT_PASSWORD

SAP_HANA_DB_COLUMNS_QUERY = config.SAP_HANA_DB_COLUMNS_QUERY
DB_HOST = config.DB_HOST
DB_PORT = config.DB_PORT
DB_USER = config.DB_USER
DB_PASSWORD = config.DB_PASSWORD

SIZE = config.SIZE
WINDOW = config.WINDOW
MIN_COUNT = config.MIN_COUNT
WORKERS = config.WORKERS
EPOCHS = config.EPOCHS

LOW_TO_HIGH_RANKING_ORDER = config.LOW_TO_HIGH_RANKING_ORDER
NUM_OF_RESULTS = config.NUM_OF_RESULTS
ACCURACY_PERCENTILE = config.ACCURACY_PERCENTILE

"""GLOBAL CONFIG"""
PATH_NON_TECH_DICT = config.PATH_NON_TECH_DICT
PATH_TECH_DICT = config.PATH_TECH_DICT
PATH_WORD2VEC_MODEL = config.PATH_WORD2VEC_MODEL
PATH_WORD2VEC_DOCS = config.PATH_WORD2VEC_DOCS
PATH_DB_SUBGROUPED = config.PATH_DB_SUBGROUPED
PATH_OUTPUTS = config.PATH_OUTPUTS
PATH_TEST_DATA = config.PATH_TEST_DATA

LANGUAGE = config.LANGUAGE
AUTOCORRECT_ON = config.AUTOCORRECT_ON
STEMMED_WORDS = config.STEMMED_WORDS


# In[ ]:


"""Gensim model"""
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text

"""NLTK natural language processing"""
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import wordnet

import pymssql
import re
import pandas as pd
import pickle
from autocorrect import spell
import pprint
import pyhdb

"""set some defaults when displaying tables"""
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)
"""Gensim model"""
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text

"""NLTK natural language processing"""
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import wordnet

import pymssql
import re
import pandas as pd
import pickle
from autocorrect import spell
import pprint
import pyhdb

"""set some defaults when displaying tables"""
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)

from IPython.core.display import display, HTML


# In[ ]:


f = open(PATH_NON_TECH_DICT,"rb")
non_technical_words=pickle.load(f)
f.close()

f = open(PATH_TECH_DICT,"rb")
technical_words=pickle.load(f)
f.close()

f = open(PATH_DB_SUBGROUPED,"rb")
db_subgrouped=pickle.load(f)
f.close()


# In[ ]:


def customize_filters(stemming):
    if(stemming == 1):
        filters = [lambda x: x.lower(),strip_punctuation,remove_stopwords,strip_multiple_whitespaces,stem_text]
    else:
        filters = [lambda x: x.lower(),strip_punctuation,remove_stopwords,strip_multiple_whitespaces]
    return filters

CUSTOM_FILTERS = customize_filters(STEMMED_WORDS)


# In[ ]:


subgroup_list=list(db_subgrouped.index)
subgroup_num=len(subgroup_list)
subgroup_list.sort()


# In[ ]:


#preprocessing 

def get_words(text):
    return word_tokenize(text)
        
def get_autocorrected(word_list):
    if(AUTOCORRECT_ON == 1):
        for i in range(len(word_list)):
            word = word_list[i]
            if word not in technical_words.keys():
                word_list[i] = spell(word)
    return word_list
        
def preprocess_text(sentence):
    word_list = get_words(sentence)
    word_list = get_autocorrected(word_list)
    sentence = ' '.join(word_list)
    return preprocess_string(sentence, CUSTOM_FILTERS)


# In[ ]:


documents = dict()
default=None

# for each row in the indexed database, create a new (row,doc) into the dict 
for model in subgroup_list:
    documents.setdefault(model,[])

""" for each row in the indexed db"""
for subgroup in subgroup_list:
    """ convert the semantic row into a list of sentences """
    subgroup_semantic=list(db_subgrouped.loc[subgroup,'semantic'])
    """ for each sentence convert it into a list of words """
    for sentence in subgroup_semantic:
        sentence=preprocess_text(sentence)
        model=subgroup
        documents[model].append(sentence)
        """ for each non-tech-word in the sentence, add its (list of synonyms) as sentence into the same document. This enhances the synonymous searching"""
        for word in sentence:
            if word in non_technical_words.keys():
                documents[model].append(non_technical_words[word])
            if word in technical_words.keys():
                for description in technical_words[word]:
                    sent = preprocess_text(description)
                    documents[model].append(sent)
                    
#documents
# In[ ]:
import multiprocessing
WORKERS = multiprocessing.cpu_count()
# In[ ]:
model_num=dict()
"""for each row,list of words in documents"""
for k, v in documents.items():
    """create a model and set its parameters"""
    model = gensim.models.Word2Vec(
        v,
        size=100,
        window=5,
        min_count=0,
        workers=WORKERS,
        hs=1,
        negative=1)
    """train the model on the list of words"""
    model.train(v, total_examples=len(v), epochs=EPOCHS)
    """you can see the vocabulary of the model by"""
    #print(model.wv.vocab)
    """insert the (row,model) into the dictionary"""
    model_num[k]=model

"""model_num[row]=model"""

# In[ ]:


f = open(PATH_WORD2VEC_DOCS,"wb")
pickle.dump(documents,f)
f.close()

f = open(PATH_WORD2VEC_MODEL,"wb")
pickle.dump(model_num,f)
f.close()

