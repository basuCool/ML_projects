
# coding: utf-8

# In[ ]:


import config

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

import pickle
import pprint
from IPython.core.display import display, HTML


# In[ ]:


f = open(PATH_WORD2VEC_MODEL,"rb")
model_num=pickle.load(f)
f.close()


# In[ ]:


def stats(dist):
    pprint("Max:",max(dist))
    pprint("Min:",min(dist))
    pprint("Avg:",sum(dist)/len(dist))


# In[ ]:


#ACCURACY TESTER

accuracy = []

for k,v in model_num.items():
    accuracy_list = model.wv.evaluate_word_pairs(PATH_TEST_DATA)
    avg_accuracy = sum(accuracy_list)/len(accuracy_list)
    accuracy.append(avg_accuracy)

stats(accuracy)

