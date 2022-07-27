
# coding: utf-8

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

from IPython.core.display import display, HTML


# In[ ]:


def customize_filters(stemming):
    if(stemming==True):
        filters = [lambda x: x.lower(),strip_punctuation,remove_stopwords,strip_multiple_whitespaces,stem_text]
    else:
        filters = [lambda x: x.lower(),strip_punctuation,remove_stopwords,strip_multiple_whitespaces]
    return filters

def read_db(db_type):
    if(db_type==0):
        db = pd.read_csv(CSV_PATH_DB )
    if(db_type==1):
        db = pd.read_excel(XLSX_PATH_DB)
    if(db_type==2):
        conn = pymssql.connect(server=SQL_PATH_DB, user=SQL_USER_DB,password=SQL_PASSWORD_DB)
        db = pd.read_sql(SQL_DB_COLUMNS_QUERY,conn)
    if(db_type == 3):
        conn = pyhdb.connect(host = DB_HOST,port = DB_PORT,user = DB_USER,password = DB_PASSWORD)
        cursor = conn.cursor()
        cursor.execute(SAP_HANA_DB_COLUMNS_QUERY)
        data = cursor.fetchall()
        connection.close()
        db = pd.DataFrame(data)   
    return db

def read_dict(dict_type):
    if(dict_type==0):
        db = pd.read_csv(CSV_PATH_DICT )
    if(dict_type==1):
        db = pd.read_excel(XLSX_PATH_DICT)
    if(dict_type==2):
        conn = pymssql.connect(server=SQL_PATH_DICT, user=SQL_USER_DICT,password=SQL_PASSWORD_DICT)
        db = pd.read_sql(SQL_DICT_COLUMNS_QUERY,conn)
    if(dict_type == 3):
        conn = pyhdb.connect(host = DICT_HOST,port = DICT_PORT,user = DICT_USER,password = DICT_PASSWORD)
        cursor = conn.cursor()
        cursor.execute(SAP_HANA_DICT_COLUMNS_QUERY)
        data = cursor.fetchall()
        connection.close()
        db = pd.DataFrame(data)   
    return db
    


# In[ ]:


#preprocessing functions

LANGUAGE='english'

"""import stopwords (is, has, would, etc) which act as misleading noise"""
STOPWORDS_NLTK_EN=set(stopwords.words(LANGUAGE))
STOPWORDS_PUNCT=set(punctuation)
STOPWORDS_EN_WITHPUNCT=set.union(STOPWORDS_NLTK_EN,STOPWORDS_PUNCT)

def get_dict(word_list):
    word_dict=dict()
    for word in word_list:
        word_dict.setdefault(word,[word])
    return word_dict

def get_words(text):
    return list(word_tokenize(text))
        
def get_autocorrected(word_list):
    if(AUTOCORRECT_ON==True):
        for i in range(len(word_list)):
            word=word_list[i]
            if word not in technical_words.keys():
                word_list[i]=spell(word)
    return word_list
        
def populate_dictionary(dictionary):
    for k,v in dictionary.items():
        for syn in wordnet.synsets(k):
            for l in syn.lemmas():
                v.append(l.name())
    return dictionary

def preprocess_word_list(word_list):
    sentence=' '.join(word_list)
    return preprocess_string(sentence, CUSTOM_FILTERS)


def get_non_technical_word_list(word_list):
    non_tech_word_list=list([word for word in word_list if word not in technical_words.keys()])
    return non_tech_word_list

def get_db_technical_dict():
    tech_dict=dict()
    for k,v in tech_db_dict.items():
        tech_dict.setdefault(str(k).lower(),v)
    return tech_dict

def get_regex_technical_dict():
    text=db.loc[:,'semantic'].str.cat(sep=' ')
    tech_list=list(set(re.findall(TECH_WORDS_REGEX,text)))
    tech_dict=get_dict(tech_list)
    return tech_dict

def get_technical_word_dict():
    tech_dict=dict()
    if(TECH_WORDS==True):
        tech_dict=dict()
        if(REGEX_TECHNICAL_WORDS == True):
            tech_dict = get_regex_technical_word_list()
        else:
            tech_dict = get_db_technical_dict()        
    return tech_dict

def preprocess_text(sentence):
    word_list=get_words(sentence)
    word_list=get_autocorrected(word_list)
    sentence=' '.join(word_list)
    return preprocess_string(sentence, CUSTOM_FILTERS)


# In[ ]:


CUSTOM_FILTERS=customize_filters(STEMMED_WORDS)

db = read_db(DB_TYPE)
db = db.fillna('')

db.columns = DB_COLS
db['semantic']=db.loc[:,DB_COLS].apply(lambda x: ' '.join(map(str,x)), axis=1)

db.loc[db.status == 1,"status" ] = "Approved"
db.loc[db.status == 0,"status" ] = "Unapproved"

db_subgrouped = db.set_index(DB_SUBGROUP_COLS)
db_subgrouped = db_subgrouped.sort_index()
        
if(TECH_WORDS==1):
    tech_db_dict = dict()
    if(REGEX_TECHNICAL_WORDS == 0):
        tech_db = read_dict(DICT_TYPE)
        tech_db.columns = TECH_DICT_COLS
        tech_db_dict = dict(zip(tech_db[TECH_DICT_TERM_COL],tech_db[TECH_DICT_DEFN_COL]))


# In[ ]:


if(TECH_WORDS==1):
    technical_words=get_technical_word_dict()


# In[ ]:


row_sentences=db.loc[:,'semantic'].str.cat(sep=' ')
words=get_words(row_sentences)

non_technical_words=get_non_technical_word_list(words)
non_technical_words=get_autocorrected(non_technical_words)


# In[ ]:


non_tech=preprocess_word_list(non_technical_words)
non_tech=get_dict(non_tech)

non_tech_dict=populate_dictionary(non_tech)            
non_tech_dict.keys()


# In[ ]:


f = open(PATH_NON_TECH_DICT,"wb")
pickle.dump(non_tech_dict,f)
f.close()

f = open(PATH_TECH_DICT,"wb")
pickle.dump(technical_words,f)
f.close()

f = open(PATH_DB_SUBGROUPED,"wb")
pickle.dump(db_subgrouped,f)
f.close()

