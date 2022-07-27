
# coding: utf-8

# In[ ]:


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
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import wordnet

import re
import pandas as pd
import pickle
from autocorrect import spell
import pprint
import pyhdb

import pymssql
import pymysql

"""set some defaults when displaying tables"""
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


f = open(PATH_NON_TECH_DICT,"rb")
non_technical_words=pickle.load(f)
f.close()

f = open(PATH_TECH_DICT,"rb")
technical_words=pickle.load(f)
f.close()

f = open(PATH_WORD2VEC_MODEL,"rb")
model_num=pickle.load(f)
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

CUSTOM_FILTERS=customize_filters(STEMMED_WORDS)


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

def preprocess_query(text):
    return preprocess_text(text)


# In[ ]:


def search_query(query):
    """create a dict for (row,score)"""
    relevant_rows=dict()
    filter_rows=dict()
    """declare lowest possible score"""
    doc_rank=float('-inf')
    """for each (row,mode) return the score/relevance"""
    for k,v in model_num.items():
            
        model_doc_rank = v.score([query.split()])
        """model.score returns an array [score for each word,score_data_type]"""
        """thus we take a sum of the word-scores to represent the score for the sentence"""
        relevant_rows[k] = sum(model_doc_rank)
        
        if FILTERS:
            filter_text_list = db_subgrouped.loc[k,FILTERS].apply(lambda x: ''.join(map(str,x))).values
            filter_text = ' '.join(filter_text_list)
            filter_text = ' '.join(preprocess_text(filter_text))
            model_filter_rank = v.score([filter_text.split()])
            filter_rows[k] = sum(model_filter_rank)
            relevant_rows[k] = relevant_rows[k] - sum(model_filter_rank)
            
    """create a list of the sorted (row,score)"""
    sorted_rows = sorted(relevant_rows.items(),key=lambda item: (item[1], item[0]),reverse=0)
    """display the top 10 rows in descending order of scores"""
    """sorted_rows[:num] where num = number of rows to display"""
    csv_rows=[]
    
    """selecting number of rows based on percentile accuracy of rows"""
    min_score = sorted_rows[-1][1]
    max_score = sorted_rows[0][1]
    accurate_percentile = (max_score-min_score)*(ACCURACY_PERCENTILE/100)
    accuracy_sorted_rows = [item for item in sorted_rows if accurate_percentile<=item[1]]
    accurate_rows = len(accuracy_sorted_rows)
    
    if(NUM_OF_RESULTS == -1):
        num_rows = min(len(sorted_rows),accurate_rows)
        for k,v in sorted_rows[:num_rows]:
            display(db_subgrouped.loc[k,:])
            csv_rows.append(k)
    else: 
        num_rows = min(NUM_OF_RESULTS,len(sorted_rows),accurate_rows)
        for k,v in sorted_rows[:num_rows]:
            display(db_subgrouped.loc[k,:])
            csv_rows.append(k)
    
    csv_db = db_subgrouped.loc[csv_rows]
    csv_db.to_excel(PATH_OUTPUTS)
        
def process_query(text):
    word_list=preprocess_query(text)
    query=' '.join(word_list)
    search_query(query)
    
def stats(dist):
    pprint("Max:",max(dist))
    pprint("Min:",min(dist))
    pprint("Avg:",sum(dist)/len(dist))
    


# In[ ]:


# MAIN INTERFACE
query = str(sys.argv[1:])
process_query(query)

