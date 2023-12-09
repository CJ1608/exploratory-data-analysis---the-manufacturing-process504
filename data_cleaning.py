
import numpy as np

class DataTransform():
    #M3, T1 convert columns to correct format (think it's all okay)
    def __init__(self, table):
        self.table = table
    
    def __convert_column(self, column_name, data_type):
        self.table[column_name] = self.table[column_name].map(data_type)
        
    
    def clean_columns(self):
        #change productid and type to str-- obj classed as str anyway
        #change machine failt to RNF to bool?
        #change tool wear to int64- if has nan values must be float
        pass 
    
    
class DataFrameInfo():
    def __init__(self, table):
        self.table = table
        
    #M3, T2 class to extract info from dataset
    def __df_overview(self):
        #Shape of dataframe
        print('Row count:', self.table.shape[0], '\tColumn count:', self.table.shape[1])
        print()
        #Describe all columns in the DataFrame to check their data types
        print('Information about table: \n"')
        print(self.table.info(), '\n"')
        print()
        
    # def __nulls(self, *columns):
    #     columns = list(*columns)
    #     for column in columns:
    #         print(f'Column "{column}" has {self.table[column].isna().sum()} null values.')
    #         percentage_null = ((self.table[column].isna().sum()/len(self.table[column]))*100)
    #         if percentage_null > 0:
    #             print(self.table[column].isna().sum(),',', len(self.table[column]), '\t\t\tpercentage of null values', percentage_null)
    #     # return self.table[column].isna().sum()
    #     print()
        
    def __distinct_values(self, *columns):
        columns = list(*columns)
        for column in columns:
            #display unique values
            # print(column,'has', np.unique(self.table[column]), 'unique values.')
            #count of unique values
            print(f'Column "{column}" has {len(np.unique(self.table[column]))} unique values.')
        # return len(np.unique(self.table[column]))
        print()
        
    # def __get_mean(self, *columns):
    #     columns = list(*columns)
    #     print('Mean values are:')
    #     for column in columns:
    #         print(column, self.table[column].mean(skipna=True))
    #     print()
           
    def __get_median(self, *columns):
        columns = list(*columns)
        print('Median values are:')
        for column in columns:
            print(column, self.table[column].median(skipna=True))    
        print()
           
    def __get_standard_deviation(self, *columns):
        columns= list(*columns)
        print('Standard deviation values are:')
        for column in columns:
            print(column, self.table[column].std(skipna=True))   
        print()
        
    def get_df_info(self):
        all_columns = self.table.columns.values.tolist()
        all_numeric_columns = self.table.select_dtypes(include=np.number)
        numeric_non_bool_columns = [x for x in self.table.select_dtypes(include=np.number) if x not in ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
        # print('Non bools', numeric_non_bool_columns)
        # print('Numeric columns', all_numeric_columns)
        self.__df_overview()
        #all_columns or ['xxx']
        # self.__nulls(all_columns)
        self.__distinct_values(['Type'])
        # self.__get_mean(numeric_non_bool_columns)
        self.__get_median(numeric_non_bool_columns)
        self.__get_standard_deviation(numeric_non_bool_columns)
        
    def nulls(self):
        nulls = dict()
        columns = list(self.table.columns.values.tolist())#list(*columns)
        #get null values for any number of columns specified
        for column in columns:
            print(f'Column "{column}" has {self.table[column].isna().sum()} null values.')
            percentage_null = ((self.table[column].isna().sum()/len(self.table[column]))*100)
            #if column does have missing values
            if percentage_null > 0:
                print(self.table[column].isna().sum(),',', len(self.table[column]), '\t\t\tpercentage of null values', percentage_null)
                #add column name, percentage_null value to dict
                nulls[column]=percentage_null
        
        for column in nulls.keys():
            print('Null column', column)
        # print(nulls)
        return nulls
        # return self.table[column].isna().sum()
        print()    
    
    def means(self, *columns):
        columns = list(*columns)
        print('Mean values are:')
        for column in columns:
            print(column, self.table[column].mean(skipna=True))
        print()

class Plotter():
    pass


class DataFrameTransformer():
    pass