
from db_utils import RDSDatabaseConnector
from data_extraction import DataExtractor
from data_cleaning import DataFrameInfo, Plotter, DataFrameTransformer
from matplotlib import pyplot as plt
import psycopg2
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats

# BUG:ctrl+f5 okay but py main.py gives 'no module found error'
 
if __name__ == '__main__':
               #M2, T2, S1-5- get credentials and connect to database
     db_connector = RDSDatabaseConnector('credentials.yaml')
     db_connector.read_db_creds()
     db_connector.init_db_engine()
     db_extractor = DataExtractor()
     table_name = 'failure_data'
     df_extracted= db_extractor.read_rds_table(db_connector, table_name)
     df_plotter = Plotter(df_extracted)
     df_plotter.graph_nulls(df_extracted)
     # plt.plot(df_extracted.isna().sum(), 'o')
     # plt.show()
     
    #M2, T2, S7- save data into .csv format
     df_extracted.to_csv('df_extracted.csv')
     
    #M2, T2, S1- function to load data from local machine into pandas df. 
     local_df = pd.read_csv('df_extracted.csv')
    
    #M2, T3- print shape of data 
     # print(df_extracted.shape)
     # #Checking data types and non-null counts
     # print(df_extracted.info())
     
     #M3, T1 convert columns to correct format (think it's all okay)
     #  df_cleaner = DataTransform(df_extracted)
     #  df_cleaned = df_cleaner.clean_columns() #didn't need to 
    
     #M3, T2 get info about dataframe
     df_information = DataFrameInfo(df_extracted)
     
     #M3, T3, S1-3 impute missing data
     nulls_columns_to_sort = df_information.nulls()
     to_sort = df_information.get_median(nulls_columns_to_sort)
     df_transformed = DataFrameTransformer(df_extracted)
     non_null_df = df_transformed.update_tables(to_sort)
     
     #M3, T3, S4 visualise n/a values
     df_plotter =  Plotter(non_null_df)
     # print(non_null_df['Rotational speed [rpm]'], non_null_df['Process temperature [K]'])
     # df_plotter.graph_nulls(non_null_df)
     
     #M3, T4, S1- visualise and find columns with skew
    ##rotational speed skewness probably right skewed because can't be less than 0 and no upper bound https://statisticsbyjim.com/basics/skewed-distribution/, rest of them under 0.5 skew so okay
     skewed = df_plotter.skewness(non_null_df)
     
    #M3, T4, S2-5 apply transformation on skewed and visualise it (TODO:I haven't updated existing df)
     skew_corrected = df_transformed.correct_skew(skewed)    #IF YOU TURN THIS BACK ON IT 
    #  print(non_null_df.head(10))
     skew_corrected.hist(bins=10)
     plt.show()
     
     #M3, T5 visualise and remove outliers from data: TODO: not going to remove outliers because I don't know which are correct- https://stats.stackexchange.com/questions/200534/is-it-ok-to-remove-outliers-from-data
     outliers_removed = df_transformed.find_outliers(table=non_null_df, column='Rotational speed [rpm]')
     outliers_removed = df_transformed.find_outliers(table=non_null_df, column='Process temperature [K]')
     outliers_removed = df_transformed.find_outliers(table=non_null_df, column='Torque [Nm]')
     df_plotter.box_plots(table=outliers_removed)
     
    #M3, T6 correlation matrix- torque/rot speed, air temp/process temp, could remove rot. speed but might keep it just in case- https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6 TODO: may need to come back and drop speed or torque
     numeric_columns = non_null_df.loc[:, ~non_null_df.columns.isin(['UDI', 'Product ID', 'Type'])] 
     df_plotter.matrix(numeric_columns)
     outliers_removed = outliers_removed.drop('Rotational speed [rpm]', axis=1)
     print(outliers_removed)