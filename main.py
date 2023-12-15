
from db_utils import RDSDatabaseConnector
from data_extraction import DataExtractor
from data_cleaning import DataFrameInfo, Plotter, DataFrameTransformer
import pandas as pd

 
if __name__ == '__main__':
    
    #  Get credentials and connect to database
     db_connector = RDSDatabaseConnector('credentials.yaml')
     db_connector.read_db_creds()
     db_connector.init_db_engine()
     db_extractor = DataExtractor()
     table_name = 'failure_data'
     df_extracted= db_extractor.read_rds_table(db_connector, table_name)
     df_plotter = Plotter(df_extracted)
     df_plotter.graph_nulls(df_extracted)
     
    #  Save data into .csv format- make sure .csv file closed
     df_extracted.to_csv('df_extracted.csv')
     
    #  Load data from local machine into pandas df. 
     local_df = pd.read_csv('df_extracted.csv')
    
    #  Get basic overview of the dataframe
     print()
     print('Shape of data is:', df_extracted.shape)
     print()
     print('Details of data:') 
     print(df_extracted.info())
     print()
     
    #  Convert columns to correct format
     #  df_cleaner = DataTransform(df_extracted)
     #  df_cleaned = df_cleaner.clean_columns() #didn't need to 
    
    #  Get info about dataframe
     df_information = DataFrameInfo(df_extracted)
     
    #  Impute missing data
     print('Checking null values...')
     nulls_columns_to_sort = df_information.nulls()
     print('Getting median values...')
     to_sort = df_information.get_median(nulls_columns_to_sort)
     df_transformed = DataFrameTransformer(df_extracted)
     non_null_df = df_transformed.update_tables(to_sort)
     
    #  Visualise missing data 
     df_plotter =  Plotter(non_null_df)
     df_plotter.graph_nulls(non_null_df)
     
    #  Find and visualise skewed data 
     ##rotational speed skewness probably right skewed because can't be less than 0 and no upper bound 
     ##  https://statisticsbyjim.com/basics/skewed-distribution/, rest of them under 0.5 skew so okay
     print('Checking skewness...')
     skewed = df_plotter.skewness(non_null_df)
     
    #  Correct skew and visualise changes
     print('Correcting skewed data...')
     skew_corrected = df_transformed.correct_skew(skewed)  
     df_plotter.skewness(skew_corrected)
     
    #  Visualise and remove outliers 
     print('Removing outliers...')
     outliers_removed = df_transformed.find_outliers(table=non_null_df, column='Rotational speed [rpm]')
     outliers_removed = df_transformed.find_outliers(table=non_null_df, column='Process temperature [K]')
     outliers_removed = df_transformed.find_outliers(table=non_null_df, column='Torque [Nm]')
     df_plotter.box_plots(table=outliers_removed)
     
    #  Find collinearity between columns 
     print('Checking correlation of columns...')
     numeric_columns = non_null_df.loc[:, ~non_null_df.columns.isin(['UDI', 'Product ID', 'Type'])] 
     df_plotter.matrix(numeric_columns)
     #If want to drop rotational speed to help with collinearity- can use below and refer to it in future calls
    #  outliers_removed = outliers_removed.drop('Rotational speed [rpm]', axis=1)
    #  print()
    #  print('Showing new dataframe...')
    #  print(outliers_removed)
     
    #  Find operating ranges of machines
     df_plotter.check_ranges(None)
     df_plotter.check_ranges('H')
     df_plotter.check_ranges('M')
     df_plotter.check_ranges('L')
    #  Find upper limit of tool wear and number of tools operating at it so know which machines may need replacing soon
     df_plotter.upper_tool_wear_limit('Tool wear [min]')
    
    #  Visualise and quantify failure rate of machines 
     df_plotter.failure_rate('Machine failure')
     df_plotter.failure_by_product_quality('Machine failure', 'Type')
     df_plotter.failure_cases()
     
    # Find out what is causing failures 
     df_plotter.failure_corr('TWF')
     df_plotter.failure_corr('HDF')
     df_plotter.failure_corr('PWF')
     df_plotter.failure_corr('OSF') 
     df_plotter.failure_corr('RNF')
    