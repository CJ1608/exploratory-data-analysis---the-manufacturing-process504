
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as stat
import statsmodels #import qqplot
import seaborn as sns
from tabulate import tabulate
import researchpy as rp

    
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
        
    def check_ranges(self, product_type): 
        range_value_dict = dict()
        product_list = ['H', 'L', 'M']
        if product_type in product_list:
            #filter by column type
            df_new = self.table[self.table['Type'] == product_type]
            #get columns mentioned they want only
            df_columns_wanted = df_new.drop(columns=['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
            #go through every column and get range
            for column in df_columns_wanted:
                if column != 'Type':
                    range_values = float(max((df_columns_wanted[column]))) - float(min((df_columns_wanted[column])))
                    range_value_dict[column] = range_values
            print(f'Values for: {product_type}...')
            print(tabulate(range_value_dict.items(), headers=["Conditon", "Range"], tablefmt='grid'))
            print()
        else:
            df_columns_wanted = self.table.drop(columns=['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
            #go through every column and get range
            for column in df_columns_wanted:
                 range_values = float(max((df_columns_wanted[column]))) - float(min((df_columns_wanted[column])))
                 range_value_dict[column] = range_values
            print(f'Values for all product types...')
            print(tabulate(range_value_dict.items(), headers=["Conditon", "Range"], tablefmt='grid'))
            print()
            
    def upper_tool_wear_limit(self, column_name):
        by_tw = self.table.groupby(column_name).count()
        ypoints = by_tw.iloc[:,0]
        plt.plot(ypoints, linestyle='solid')
        plt.title('Tool wear of machinery')
        plt.xlabel('Tool wear [minutes]')
        plt.ylabel('Count')
        plt.show()
        
    def failure_rate(self, column_name):
        #how many failures happened and what is percentage of failures vs non-failures (machine failure 1-0)
        by_1= self.table.groupby(column_name).count()
        by = by_1.iloc[:,0]
        print(f'Count of failures is {by[1]} out of {by[0]} machining sessions. The percentage total of failures is: {(by[1]/by[0])*100:.2f}')
        print()
        
    def failure_by_product_quality(self, column_name, column_name2):   #chi-square test? 
        #see how failures relate to quality of product      
        crosstab= pd.crosstab(self.table[column_name], self.table[column_name2])
        # print(crosstab) #use this to make a graph
        # print()
        crosstab1, test_results, expected = rp.crosstab(self.table[column_name],self.table[column_name2], test='chi-square', expected_freqs=True, prop='cell')
        print(crosstab1) #showing percentage of each result based on machine failure and type
        print()
        print('Results', test_results) #showing stats tests for it 
        print()
        print(expected)
        print()
        
    def failure_cases(self):
        #check machine failure count then check TWF-RNF columns for failure reason, find each 1 
        failure_dict = dict()
        print('Failure cases method.')
        #get columns want
        df_columns_wanted = self.table.drop(columns=['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
        df_columns_wanted.rename(columns={'Machine failure': 'Total machine failures'}, inplace=True)
        #go through each column and get count of '1's and save column name then count as k:v pairs
        for column in df_columns_wanted:
            failure_dict[column] = df_columns_wanted[column].value_counts()[1]     
        #plot results 
        plt.bar(failure_dict.keys(), failure_dict.values())
        plt.title('Machine failure reasons')
        plt.xlabel('Failure reasons')
        plt.ylabel('Count of occurances')
        plt.show()
        
    def failure_reasons(self, product_type):
        # for specified product type
        df_wanted = self.table.loc[self.table['Type']== product_type]
        df_wanted = df_wanted.drop(columns=['UDI', 'Product ID', 'Type'])
        # print(df_wanted)
        matrix = df_wanted.corr()
        print(matrix)
        corr_pairs = matrix.unstack()
        sorted_pairs = corr_pairs.sort_values(kind='quicksort')
        print()
        positive_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]
        print(positive_pairs) #11
    
    def failure_corr(self, failure_type):
        #pass in table and reduce it down to index, failure types and variables- https://www.w3schools.com/python/matplotlib_scatter.asp
        columns_wanted = self.table.drop(columns=['UDI', 'Product ID', 'Type'])
        variables_to_check = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'] #hdf process temp
        y_axis_fail = list()
        x_axis_fail = list()
        y_axis_pass = list()
        x_axis_pass = list()
        ##for each variable
        for element in variables_to_check:
            # print(element)
            ##for each row, if the value in self.table[failure_type] is 1, add the index to x-array and add value to y-array
            for index, row in self.table.iterrows():
                #if it failed
                if row[failure_type] == 1:
                    y_axis_fail.append(row[element])
                    x_axis_fail.append(index)
                    # print(element, index, row[element])
                else:
                    y_axis_pass.append(row[element])
                    x_axis_pass.append(index)
            #         # print(element, index, row[element])     
            # print('Failed:', y_axis_fail, x_axis_fail)       
            # print('Passed:', y_axis_pass, x_axis_pass) 
            
            plt.scatter(x_axis_pass, y_axis_pass, color='green', alpha=0.7)
            plt.scatter(x_axis_fail, y_axis_fail, color='red')
            plt.xlabel('Row number')
            plt.ylabel(f'{element} values')
            plt.title(f'{failure_type}: {element}')
            plt.show()
            # plt.savefig(f'{failure_type}_{element}.jpg')
                    
                    
                    
                       ###wait until x and y arrays full and add them to plt
                # print(index, row)         
                    
        ##for each row, if the value in self.table[failure_type] is 0, add the index to x-array and add value to y-array
            ###wait until x and y arrays full and then add them to plt
        ##show plt
        
        
        #plot numeric variables by if 0 one colour, and 1 another colour scatterplot
        
            # if self.table[failure_type] == 1:
                
        ##what is x and y variable values- x id one axis, y values the other axis
        
        
        ##get list of values for 1s in specified column 
        
        ##get list of values for 0s in specified column 
        
        #graph for each data against each data type
        

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