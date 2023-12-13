
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as stat
import statsmodels #import qqplot
import seaborn as sns

    
class DataFrameInfo():
    #M3, T2 class to extract info from dataset
    def __init__(self, table):
        self.table = table 
   
    def df_overview(self):
        #Shape of dataframe- haven't actually used but might want to in future
        print('Row count:', self.table.shape[0], '\tColumn count:', self.table.shape[1])
        print()
        
    def distinct_values(self, *columns):
        columns = list(*columns)
        for column in columns:
            #display unique values
            print(f'Column {column} has {np.unique(self.table[column])} unique values.')
            #count of unique values
            print(f'Column "{column}" has {len(np.unique(self.table[column]))} unique values.')
        print()
        return len(np.unique(self.table[column]))
    
    def get_means(self, *columns):
        print('Mean values are:')
        for column in columns:
            print(column, self.table[column].mean(skipna=True))
        print()
           
    def get_median(self, *columns):
        columns = list(*columns)
        nulls_to_impute = dict()
        print('Median values are:')
        for column in columns:
            nulls_to_impute[column]= self.table[column].mean(skipna=True)
            print(column, self.table[column].median(skipna=True))    
        print()
        return nulls_to_impute
           
    def get_standard_deviation(self, *columns):
        columns= list(*columns)
        print('Standard deviation values are:')
        for column in columns:
            print(column, self.table[column].std(skipna=True))   
        print()
        
    def nulls(self):
        nulls = dict()
        columns = list(self.table.columns.values.tolist())#list(*columns)
        #get null values for any number of columns specified
        for column in columns:
            print(f'Column "{column}" has {self.table[column].isna().sum()} null values.')
            percentage_null = ((self.table[column].isna().sum()/len(self.table[column]))*100)
            #if column does have missing values
            if percentage_null > 0:
                nulls[column]=percentage_null
                print(self.table[column].isna().sum(), len(self.table[column]), percentage_null)
        print()
        return nulls

class Plotter():
    def __init__(self, table):
        self.table = table
        self.all_columns = self.table.columns.values.tolist()
        self.all_numeric_columns = self.table.select_dtypes(include=np.number)
        self.numeric_non_bool_columns = [x for x in self.table.select_dtypes(include=np.number) if x not in ['UID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
        
    def graph_nulls(self, table):
        plt.plot(table.isna().sum(), 'o')
        plt.title('Null value count')
        plt.show()
        
    def skewness(self, table):
        skewed = list()
        for row in table:
            if row in self.numeric_non_bool_columns:
                print(f'Skewness of "{row}" is {self.table[row].skew()}.')
                self.table[row].hist(bins=100)
                plt.title(row)
                # # qqPLOT = sm.qqplot(self.table[row], scale=1, line='q', fit=True)
                plt.show()
                if self.table[row].skew() > 0.5 or self.table[row].skew() < -0.5:
                    print(row, 'heavily skewed:', self.table[row].skew())
                    skewed.append(row)
            else:
                print('Row not in:', row)
        print()
        return skewed
    
    def box_plots(self, table):
        for row in table:
            if row in self.numeric_non_bool_columns:
                #print visual aid 
                plt.boxplot(self.table[row])
                plt.title(row)
                plt.show()
            
    def matrix(self, table):
        #correlation matrix
        mat = table.corr()
        print(mat)
        #visual correlation
        sns.heatmap(table)
        plt.show()

class DataFrameTransformer():
    
    def __init__(self, table):
        self.table = table
        self.numeric_non_bool_columns = [x for x in self.table.select_dtypes(include=np.number) if x not in ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
        
    def update_tables(self, to_sort):
        for key, value in to_sort.items():
            # print(key, value)
            self.table[key].fillna(value=value, inplace=True)
        # print(self.table)
        # print()
        return self.table
    
    def correct_skew(self, column):
        # print(self.table[column])
        skew_corrected = np.log(self.table[column])
        print('Skewed data corrected (log):', skew_corrected.skew())
        # skew_corrected = np.
        return skew_corrected
    
    def find_outliers(self, table, column):
        drop_rows = list()
        index_location = list() 
        # print(self.table[column])
        q1= np.quantile(self.table[column], 0.25)
        q3 = np.quantile(self.table[column], 0.75)
        iqr = q3 - q1
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)
        
        for index, row in self.table.iterrows():
            ##if outlier has a value outside of upper_bound and is a bigger number that the highest value in 75% of results and spread of % spread of data in Q1-Q3 drop it
            if row[column] <= lower_bound or row[column] >= upper_bound+iqr:
                drop_rows.append(row[column])
                index_location.append(index)
                # print(row[column], index)
        #remove outliers from column by row
        self.table.drop(index_location, axis=0, inplace=True)
        # print(self.table.info())
        #update table
        return self.table
        
        
       
        
    """
            corr = numeric_columns.corr()
        sns.heatmap(corr, xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
        plt.show()
        
        
        # corr = numeric_columns.corr()
        # corr.style.background_gradient(cmap='coolwarm')
        
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
                
            # if row == 'Rotational speed [rpm]':
            #     q1= np.quantile(column, 0.25)
            #     q3 = np.quantile(column, 0.75)
            #     iqr = q3 - q1
            #     upper_bound = q3+(1.5*iqr)
            #     lower_bound = q1-(1.5*iqr)#
            #     print(f'Bounds for {row} {upper_bound}, {lower_bound}, {iqr}')
            #     outliers = column[(column <= lower_bound) | (column >= upper_bound)]#
            #     print(len(outliers))
            #     # print()
            #     # plt.boxplot(column)
            #     # plt.title(row)
            #     # plt.show()
    
    # def __get_mean(self, *columns):
    #     columns = list(*columns)
    #     print('Mean values are:')
    #     for column in columns:
    #         print(column, self.table[column].mean(skipna=True))
    #     print()
           
    # def __nulls(self, *columns):
    #     columns = list(*columns)
    #     for column in columns:
    #         print(f'Column "{column}" has {self.table[column].isna().sum()} null values.')
    #         percentage_null = ((self.table[column].isna().sum()/len(self.table[column]))*100)
    #         if percentage_null > 0:
    #             print(self.table[column].isna().sum(),',', len(self.table[column]), '\t\t\tpercentage of null values', percentage_null)
    #     # return self.table[column].isna().sum()
    #     print()
    
    # def get_df_info(self):
    #     all_columns = self.table.columns.values.tolist()
    #     all_numeric_columns = self.table.select_dtypes(include=np.number)
    #     numeric_non_bool_columns = [x for x in self.table.select_dtypes(include=np.number) if x not in ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
    #     # print('Non bools', numeric_non_bool_columns)
    #     # print('Numeric columns', all_numeric_columns)
    #     self.__df_overview()
    #     #all_columns or ['xxx']
    #     # self.__nulls(all_columns)
    #     self.__distinct_values(['Type'])
    #     # self.__get_mean(numeric_non_bool_columns)
    #     self.__get_median(numeric_non_bool_columns)
    #     self.__get_standard_deviation(numeric_non_bool_columns)
    
        def remove_outliers(self, outliers, row, upper_bound, iqr):
        print(row)
        # print(max(outliers), min(outliers), np.median(outliers))
        to_drop = list()
        # if outlier value is significantly larger or smaller, add to drop list
        for element in outliers:
            # print(element)
           
            if float(element) > float((upper_bound)+(iqr)):
                print(element)
                to_drop.append(element)
        # # print
        # table = table[table.row.isin(to_drop) == False]
        # print(len(table))
        return to_drop     
        #then drop rows from drop list
        
    #return cleaned colummn back to find_outliers
    # return
     
        # for row in table:
        #     if row in self.numeric_non_bool_columns:
        #         # do maths to find outliers
        #         q1= np.quantile(self.table[row], 0.25)
        #         q3 = np.quantile(self.table[row], 0.75)
        #         iqr = q3 - q1
        #         upper_bound = q3+(1.5*iqr)
        #         lower_bound = q1-(1.5*iqr)
        #         print(f'Bounds for {row} upper: {upper_bound}, lower: {lower_bound}, IQrange: {iqr}')
        #         #find outliers using IQRs
        #         outliers = self.table[row][(self.table[row] <= lower_bound) | (self.table[row] >= upper_bound)]
        #         print(f'Number of outliers for {row} is {len(outliers)}')
        #         #if it has outliers
        #         if len(outliers) > 1:
        #             print(row)
        #             # #if outlier has a value outside of upper_bound and is a bigger number that the highest value in 75% of results and spread of % spread of data in Q1-Q3 drop it
        #             upper_outliers = np.where(self.table[row] >= float(upper_bound))[0]
        #             lower_outliers = np.where(self.table[row] <= lower_bound)[0]
        #             #remove outliers
        #             self.table[row].drop(index=upper_outliers, inplace=True)
        #             self.table[row].drop(index=lower_outliers, inplace=True)
        #             print('after', self.table[row].info())
        # return table
        # return self.table
        
    # def find_outliers(self, row):
    #     #find outliers
    #     q1= np.quantile(self.table[row], 0.25)
    #     q3 = np.quantile(self.table[row], 0.75)
    #     iqr = q3 - q1
    #     upper_bound = q3+(1.5*iqr)
    #     lower_bound = q1-(1.5*iqr)
        
    #     outliers = self.table[row][~(self.table[row] <= lower_bound) | (self.table[row] >= upper_bound)]
    #     print(row, 'has ', len(row), 'outliers')
    #     outliers_dropped = self.table[row].dropna()
    #     plt.boxplot(self.table[row])
    #     plt.title(row)
    #     plt.show()
    #     return outliers_dropped
        
        
        
        # upper_bound = q3+(1.5*iqr)
        # lower_bound = q1-(1.5*iqr)
        # print(f'Bounds for {row} upper: {upper_bound}, lower: {lower_bound}, IQrange: {iqr}')
        # #find outliers using IQRs
        # outliers = self.table[row][(self.table[row] <= lower_bound) | (self.table[row] >= upper_bound)]
        # for element in range(len(outliers)):
            
        # outliers_index_positions = self.
        # print(f'Number of outliers for {row} is {len(outliers)}')
        # #remove outliers from row, get their index positions
    

        
    """