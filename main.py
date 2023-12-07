
from db_utils import RDSDatabaseConnector
from data_extraction import DataExtractor
import psycopg2
import pandas as pd

# BUG:ctrl+f5 okay but py main.py gives 'no module found error'

def milestone2():
    #M2, T2, S1-5- get credentials and connect to database
    db_connector = RDSDatabaseConnector('credentials.yaml')
    db_connector.read_db_creds()
    db_connector.init_db_engine()
    # db_connector.list_db_tables()
    #M2, T2, S6- extract data from database and return as pd.Dataframe in table called failure_data
    db_extractor = DataExtractor()
    table_name = 'failure_data'
    df_extracted= db_extractor.read_rds_table(db_connector, table_name)
    print(df_extracted)
    #M2, T2, S7- save data into .csv format
    df_extracted.to_csv('df_extracted.csv')
    #M2, T2, S1- function to load data from local machine into pandas df. 
    local_df = pd.read_csv('df_extracted.csv')
    # print(local_df.head())
    #M2, T3- print shape of data 
    print(df_extracted.shape)
    
if __name__ == '__main__':
    milestone2()