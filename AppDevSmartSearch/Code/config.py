import xml.etree.ElementTree as ET

tree = ET.parse('config.xml')
root = tree.getroot()

"""------------------------------------------------------------------------------------------"""
DB_TYPE = int(root.findall('./preprocessing/db/db_type')[0].text)
DB_COLS = str(root.findall('./preprocessing/db/db_cols')[0].text)
DB_SUBGROUP_COLS = str(root.findall('./preprocessing/db/db_subgroup_cols')[0].text)
"""------------------------------------------------------------------------------------------"""
TECH_WORDS = int(root.findall('./preprocessing/dict/tech_words')[0].text)
DICT_TYPE = int(root.findall('./preprocessing/dict/dict_type')[0].text)
TECH_DICT_COLS = str(root.findall('./preprocessing/dict/dict_cols')[0].text)
TECH_DICT_TERM_COL = str(root.findall('./preprocessing/dict/tech_dict_term_col')[0].text)
TECH_DICT_DEFN_COL = str(root.findall('./preprocessing/dict/tech_dict_defn_col')[0].text)
"""------------------------------------------------------------------------------------------"""
REGEX_TECHNICAL_WORDS = int(root.findall('./preprocessing/dict/regex/regex_tech_words')[0].text)
TECH_WORDS_REGEX = str(root.findall('./preprocessing/dict/regex/tech_words_regex')[0].text)
"""-----------------------------------------------------------------------------------------"""
CSV_PATH_DB = str(root.findall('./preprocessing/db/csv/csv_path_db')[0].text)
CSV_PATH_DICT = str(root.findall('./preprocessing/dict/csv/csv_path_dict')[0].text)
"""-----------------------------------------------------------------------------------------"""
XLSX_PATH_DB = str(root.findall('./preprocessing/db/xlsx/excel_path_db')[0].text)
XLSX_PATH_DICT = str(root.findall('./preprocessing/dict/xlsx/excel_path_dict')[0].text)
"""-----------------------------------------------------------------------------------------"""
MY_SQL_PATH_DB = str(root.findall('./preprocessing/db/mysql/db_server')[0].text)
MY_SQL_USER_DB = str(root.findall('./preprocessing/db/mysql/db_user')[0].text)
MY_SQL_PASSWORD_DB = str(root.findall('./preprocessing/db/mysql/db_password')[0].text)
MY_SQL_DB_COLUMNS_QUERY = str(root.findall('./preprocessing/db/mysql/sql_db_columns_query')[0].text)

MY_SQL_PATH_DICT = str(root.findall('./preprocessing/dict/mysql/dict_server')[0].text)
MY_SQL_USER_DICT = str(root.findall('./preprocessing/dict/mysql/dict_user')[0].text)
MY_SQL_PASSWORD_DICT = str(root.findall('./preprocessing/dict/mysql/dict_password')[0].text)
MY_SQL_DICT_COLUMNS_QUERY = str(root.findall('./preprocessing/dict/mysql/sql_dict_columns_query')[0].text)
"""---------------------------------------------------------------------------------------"""
MS_SQL_PATH_DB = str(root.findall('./preprocessing/db/mssql/db_server')[0].text)
MS_SQL_USER_DB = str(root.findall('./preprocessing/db/mssql/db_user')[0].text)
MS_SQL_PASSWORD_DB = str(root.findall('./preprocessing/db/mssql/db_password')[0].text)
MS_SQL_DB_COLUMNS_QUERY = str(root.findall('./preprocessing/db/mssql/sql_db_columns_query')[0].text)

MS_SQL_PATH_DICT = str(root.findall('./preprocessing/dict/mssql/dict_server')[0].text)
MS_SQL_USER_DICT = str(root.findall('./preprocessing/dict/mssql/dict_user')[0].text)
MS_SQL_PASSWORD_DICT = str(root.findall('./preprocessing/dict/mssql/dict_password')[0].text)
MS_SQL_DICT_COLUMNS_QUERY = str(root.findall('./preprocessing/dict/mssql/sql_dict_columns_query')[0].text)
"""---------------------------------------------------------------------------------------"""
SAP_HANA_DB_COLUMNS_QUERY = str(root.findall('./preprocessing/db/SAPhana/sap_hana_db_columns_query')[0].text)
DB_HOST = str(root.findall('./preprocessing/db/SAPhana/db_host')[0].text)
DB_PORT = int(root.findall('./preprocessing/db/SAPhana/db_port')[0].text)
DB_USER = str(root.findall('./preprocessing/db/SAPhana/db_user')[0].text)
DB_PASSWORD = str(root.findall('./preprocessing/db/SAPhana/db_password')[0].text)

SAP_HANA_DICT_COLUMNS_QUERY = str(root.findall('./preprocessing/dict/SAPhana/sap_hana_dict_columns_query')[0].text)
DICT_HOST = str(root.findall('./preprocessing/dict/SAPhana/dict_host')[0].text)
DICT_PORT = int(root.findall('./preprocessing/dict/SAPhana/dict_port')[0].text)
DICT_USER = str(root.findall('./preprocessing/dict/SAPhana/dict_user')[0].text)
DICT_PASSWORD = str(root.findall('./preprocessing/dict/SAPhana/dict_password')[0].text)
"""-----------------------------------------------------------------------------------"""
DB_COLS = DB_COLS.split(sep=',')
DB_SUBGROUP_COLS = DB_SUBGROUP_COLS.split(sep=',')
TECH_DICT_COLS = TECH_DICT_COLS.split(sep=',')
"""-----------------------------------------------------------------------------------"""
SIZE = int(root.findall('./implementation/size')[0].text)
WINDOW = int(root.findall('./implementation/window')[0].text)
MIN_COUNT = int(root.findall('./implementation/min_count')[0].text)
WORKERS = int(root.findall('./implementation/workers')[0].text)
EPOCHS = int(root.findall('./implementation/epochs')[0].text)
"""-----------------------------------------------------------------------------------"""
LOW_TO_HIGH_RANKING_ORDER = int(root.findall('./interface/low_to_high_ranking_order')[0].text)
NUM_OF_RESULTS = int(root.findall('./interface/num_of_results')[0].text)
ACCURACY_PERCENTILE = int(root.findall('./interface/accuracy_percentile')[0].text)
"""-----------------------------------------------------------------------------------"""
PATH_NON_TECH_DICT = str(root.findall('./global/paths/path_non_tech_dict')[0].text)
PATH_TECH_DICT = str(root.findall('./global/paths/path_tech_dict')[0].text)
PATH_WORD2VEC_MODEL = str(root.findall('./global/paths/path_word2vec_model')[0].text)
PATH_WORD2VEC_DOCS = str(root.findall('./global/paths/path_word2vec_docs')[0].text)
PATH_DB_SUBGROUPED = str(root.findall('./global/paths/path_db_subgrouped')[0].text)
PATH_OUTPUTS = str(root.findall('./global/paths/path_outputs')[0].text)
PATH_TEST_DATA = str(root.findall('./global/paths/path_test_data')[0].text)

LANGUAGE = str(root.findall('./global/language')[0].text)
AUTOCORRECT_ON = int(root.findall('./global/autocorrect_on')[0].text)
STEMMED_WORDS = int(root.findall('./global/stemmed_words')[0].text)
"""-----------------------------------------------------------------------------------"""
GLOBAL_SECTION_UPDATE=int(root.findall('./global/global_section_update')[0].text)
PREPROCESSING_SECTION_UPDATE=int(root.findall('./preprocessing/preprocessing_section_update')[0].text)
IMPLEMENTATION_SECTION_UPDATE=int(root.findall('./implementation/implementation_section_update')[0].text)
INTERFACE_SECTION_UPDATE=int(root.findall('./interface/interface_section_update')[0].text)
