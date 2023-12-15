
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
     
    #M2, T2, S7- save data into .csv format- make sure .csv file closed
     df_extracted.to_csv('df_extracted.csv')
     
    #M2, T2, S1- function to load data from local machine into pandas df. 
     local_df = pd.read_csv('df_extracted.csv')
    
    #M2, T3- print shape of data 
     print()
     print('Shape of data is:', df_extracted.shape)
     print()
     print('Details of data:') 
     print(df_extracted.info())
     print()
     
     #M3, T1 convert columns to correct format (think it's all okay)
     #  df_cleaner = DataTransform(df_extracted)
     #  df_cleaned = df_cleaner.clean_columns() #didn't need to 
    
     #M3, T2 get info about dataframe
     df_information = DataFrameInfo(df_extracted)
     
     #M3, T3, S1-3 impute missing data
     print('Checking null values...')
     nulls_columns_to_sort = df_information.nulls()
     print('Getting median values...')
     to_sort = df_information.get_median(nulls_columns_to_sort)
     df_transformed = DataFrameTransformer(df_extracted)
     non_null_df = df_transformed.update_tables(to_sort)
     
     #M3, T3, S4 visualise n/a values
    #  df_plotter =  Plotter(non_null_df)
    #  df_plotter.graph_nulls(non_null_df)
     
     #M3, T4, S1- visualise and find columns with skew
     ##rotational speed skewness probably right skewed because can't be less than 0 and no upper bound https://statisticsbyjim.com/basics/skewed-distribution/, rest of them under 0.5 skew so okay
    #  print('Checking skewness...')
    #  skewed = df_plotter.skewness(non_null_df)
     
    #M3, T4, S2-5 apply transformation on skewed and visualise it (TODO:I haven't updated existing df)
     print('Correcting skewed data...')
    #  skew_corrected = df_transformed.correct_skew(skewed)  
    #  df_plotter.skewness(skew_corrected)
     
     #M3, T5 visualise and remove outliers from data: 
     print('Removing outliers...')
     outliers_removed = df_transformed.find_outliers(table=non_null_df, column='Rotational speed [rpm]')
     outliers_removed = df_transformed.find_outliers(table=non_null_df, column='Process temperature [K]')
     outliers_removed = df_transformed.find_outliers(table=non_null_df, column='Torque [Nm]')
    #  df_plotter.box_plots(table=outliers_removed)
     
    #M3, T6 correlation matrix- torque/rot speed, air temp/process temp, could remove rot. https://towardsdatascience.com/are-you-dropping-too-many-correlated-features-d1c96654abe6 TODO: may need to come back and drop speed or torque
     print('Checking correlation of columns...')
     numeric_columns = non_null_df.loc[:, ~non_null_df.columns.isin(['UDI', 'Product ID', 'Type'])] 
    #  df_plotter.matrix(numeric_columns)
    #  outliers_removed = outliers_removed.drop('Rotational speed [rpm]', axis=1)
    #  print()
    #  print('Showing new dataframe...')
    #  print(outliers_removed)
     
     #M4, T1 current operating ranges
     ##find operating ranges fo r5 specified conditions and break them down into different product quality types (H, M, L)
     df_plotter.check_ranges(None)
     df_plotter.check_ranges('H')
     df_plotter.check_ranges('M')
     df_plotter.check_ranges('L')
    ##upper limit of tool wear machines operating it and number of tools operating at different tool wear values
    #  df_plotter.upper_tool_wear_limit('Tool wear [min]')
    
     #MT, T2 failure rate of process
     ##visualise how many failures have happened in the process, what percentage is this of the total? Check if the failures are being caused based on the quality of the product.https://www.pythonfordatascience.org/chi-square-test-of-independence-python/
    #  df_plotter.failure_rate('Machine failure')
     df_plotter.failure_by_product_quality('Machine failure', 'Type')
    #  ##Create a visualisation of the number of failures due to each possible cause during the manufacturing process. leading cause of failure?
    #  df_plotter.failure_cases()
     
     #M4, T3 deeper understanding of failures
     ##
    #  df_plotter.failure_reasons('H') #Machine failure, TWF, HDF
    #  df_plotter.failure_reasons('M') #Machine failure, PWF, HDF
    #  df_plotter.failure_reasons('L') #Machine failure, HDF, OSF
    #['TWF', 'HDF', 'PWF','OSF', 'RNF'] 
    #  df_plotter.failure_corr('TWF')
     df_plotter.failure_corr('HDF')
    #  df_plotter.failure_corr('PWF')
    #  df_plotter.failure_corr('OSF')
    #  df_plotter.failure_corr('RNF')
     
     #plot graphs of values for when machines failed and didn't fail for each failure type- scatter graph but colour points by different colours based on whether failed or not
     
     ##plot values as line then if a 1 print different column to 0 
     
     ###create method, pass in one failure type and plot values by colour based on whether failed or nto for one column of varialbles ex: torque. do it 5 times with each different variable. graph for each data against each data type
     ###
     
     