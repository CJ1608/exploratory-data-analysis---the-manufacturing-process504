
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tabulate import tabulate
import researchpy as rp

    
class DataFrameInfo():
    """
    Methods for basic understanding of the data within the dataset. 
    """
    def __init__(self, table):
        """
        See help(DataFrameInfo) for more detail.

        Args:
            table (pd.dataframe): dataset to be examined. 
        """
        self.table = table 
   
    def df_overview(self):
        """
        Print number of rows and columns in dataset. 
        """
        print('Row count:', self.table.shape[0], '\tColumn count:', self.table.shape[1])
        print()
        
    def distinct_values(self, *columns):
        """
        Get unique values and count of unique values for each of the specified columns. 
        
        Returns:
            int: number of unique values in column
        """
        columns = list(*columns)
        for column in columns:
            print(f'Column {column} has {np.unique(self.table[column])} unique values.')
            print(f'Column "{column}" has {len(np.unique(self.table[column]))} unique values.')
        print()
        return len(np.unique(self.table[column]))
    
    def get_means(self, *columns):
        """
        Get mean value for each of the specified columns. 
        """
        print('Mean values are:')
        for column in columns:
            print(column, self.table[column].mean(skipna=True))
        print()
           
    def get_median(self, *columns):
        """
        Get median value for each of the specified columns.

        Returns:
            dict: k:v pairs of column name and median value. 
        """
        columns = list(*columns)
        nulls_to_impute = dict()
        print('Median values are:')
        for column in columns:
            nulls_to_impute[column]= self.table[column].median(skipna=True)
            print(column, self.table[column].median(skipna=True))    
        print()
        return nulls_to_impute
           
    def get_standard_deviation(self, *columns):
        """
        Get standard deviation for each of the specified columns. 
        """
        columns= list(*columns)
        print('Standard deviation values are:')
        for column in columns:
            print(column, self.table[column].std(skipna=True))   
        print()
        
    def nulls(self):
        """
        For each of the specified columns, get the percentage of records that have null values in them. 

        Returns:
            dict: k:v pairs of column name and percentage null values for that column
        """
        nulls = dict()
        columns = list(self.table.columns.values.tolist())
        #get null values for the columns specified
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
    """
    Methods to visualise data in dataset. 
    """
    def __init__(self, table):
        """
        See help(Plotter) for more detail. 

        Args:
            table (pd.dataframe): dataset to be visualised. 
        """
        self.table = table
        self.all_columns = self.table.columns.values.tolist()
        self.all_numeric_columns = self.table.select_dtypes(include=np.number)
        self.numeric_non_bool_columns = [x for x in self.table.select_dtypes(include=np.number) if x not in ['UID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
        
    def graph_nulls(self, table):
        """
        Print a point graph showing the count of null values for each column. 

        Args:
            table (pd.dataframe): dataset to be visualised. 
        """
        plt.plot(table.isna().sum(), 'o')
        plt.title('Null value count')
        plt.xlabel('Variable')
        plt.ylabel('Count')
        plt.show()
        
    def skewness(self, table):
        """
        Print histogram showing skew of each column in dataset that is numeric and not one of the Boolean failure types. 

        Args:
            table (pd.dataframe): dataframe used. 

        Returns:
            list: list of all columns that have an absolute skew of more than 0.5. 
        """
        skewed = list()
        for row in self.table:
            for row in self.numeric_non_bool_columns:
                print(f'Skewness of "{row}" is {self.table[row].skew()}.')
                self.table[row].hist(bins=100)
                plt.title(row)
                plt.xlabel('Variable values')
                plt.ylabel('Count')
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
        """
        Plot boxplots to show the spread of values for each column in dataset that is numeric and not one of the Boolean failure types.

        Args:
            table (pd.dataframe): dataframe used.
        """
        for row in self.table:
            if row in self.numeric_non_bool_columns:
                plt.boxplot(self.table[row])
                plt.title(row)
                plt.ylabel('Variable values')
                plt.show()
            
    def matrix(self, table):
        """
        Print correlation matrix of columns in the dataset and plot a heatmap to show the correlation for each record between columns. 

        Args:
            table (pd.dataframe): dataframe used.
        """
        #correlation matrix
        mat = table.corr()
        print(mat)
        sns.heatmap(table)
        plt.title('Correlation between variables')
        plt.xlabel('Variables')
        plt.ylabel('Individual records')
        plt.show()
        
    def check_ranges(self, product_type): 
        """
        For each product type, find the range of operating conditions the machines are operating at. Print the results as a table. 

        Args:
            product_type (str): specific product type to check. 
        """
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
        """
        Plot line graph of tool wear limit and how many machines are operating at the maximum limit. 

        Args:
            column_name (str): name of column to be checked. 
        """
        by_tw = self.table.groupby(column_name).count()
        ypoints = by_tw.iloc[:,0]
        plt.plot(ypoints, linestyle='solid')
        plt.title(f'{column_name}')
        plt.xlabel('Tool wear [minutes]')
        plt.ylabel('Count')
        plt.show()
        
    def failure_rate(self, column_name):
        """
        Get count and percentage total of failed machining sessions. 

        Args:
            column_name (str): _description_
        """
        by_1= self.table.groupby(column_name).count()
        by = by_1.iloc[:,0]
        print(f'Count of failures is {by[1]} out of {by[0]} machining sessions. The percentage total of failures is: {(by[1]/by[0])*100:.2f}')
        print()
        
    def failure_by_product_quality(self, column_name, column_name2): 
        """
        Get chi-squared test result and percentage total for machine failure (0/No, 1/Yes) for each product type. 

        Args:
            column_name (str): column to be checked. 
            column_name2 (str): second column to be checked. 
        """
        #see how failures relate to quality of product      
        crosstab= pd.crosstab(self.table[column_name], self.table[column_name2])
        print('Actual count:\t', crosstab) #use this to make a graph
        # print()
        crosstab1, test_results, expected = rp.crosstab(self.table[column_name],self.table[column_name2], test='chi-square', expected_freqs=True, prop='cell')
        print(crosstab1) #showing percentage of each result based on machine failure and type
        print()
        print('Percentage of results\t', test_results) #showing stats tests for it 
        print()
        
    def failure_cases(self): 
        """
        Print bar chart showing the cause of each machine failure occurance. 
        """
        failure_dict = dict()
        print('Failure cases method.')
        df_columns_wanted = self.table.drop(columns=['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
        df_columns_wanted.rename(columns={'Machine failure': 'Total machine failures'}, inplace=True)
        #go through each column and get count of '1's and save column name then count as k:v pairs
        for column in df_columns_wanted:
            failure_dict[column] = df_columns_wanted[column].value_counts()[1]     
        plt.bar(failure_dict.keys(), failure_dict.values())
        plt.title('Machine failure reasons')
        plt.xlabel('Failure reasons')
        plt.ylabel('Count')
        plt.show()
    
    def failure_corr(self, failure_type):
        """
        Plot scattergraph of variables for each failure type by product type. 

        Args:
            failure_type (str): reason for machine failure. 
        """
        variables_to_check = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'] 
        y_axis_fail_H = list()
        x_axis_fail_H = list()
        y_axis_fail_M = list()
        x_axis_fail_M = list()
        y_axis_fail_L = list()
        x_axis_fail_L = list()        
        y_axis_pass = list()
        x_axis_pass = list()
        ##for each variable
        for element in variables_to_check:
            ##for each row, if the value in self.table[failure_type] is 1, add the index to x-array and add value to y-array
            for index, row in self.table.iterrows():
                #if it failed
                if row[failure_type] == 1:
                    if row['Type'] == 'H':
                        y_axis_fail_H.append(row[element])
                        x_axis_fail_H.append(index)
                    if row['Type'] == 'M':
                        y_axis_fail_M.append(row[element])
                        x_axis_fail_M.append(index)
                    if row['Type'] ==  'L':
                        y_axis_fail_L.append(row[element])
                        x_axis_fail_L.append(index)    
                else:
                    y_axis_pass.append(row[element])
                    x_axis_pass.append(index) 
            plt.scatter(x_axis_pass, y_axis_pass, color='green', alpha=0.7, label='No failure')
            plt.scatter(x_axis_fail_H, y_axis_fail_H, color='red', label='High quality')
            plt.scatter(x_axis_fail_M, y_axis_fail_M, color='darkorange', label='Medium quality')
            plt.scatter(x_axis_fail_L, y_axis_fail_L, color='purple', label='Low quality')
            plt.xlabel('Row number')
            plt.ylabel(f'{element} values')
            plt.title(f'{failure_type} failures for {element} and product quality')
            plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left')
            plt.show()

class DataFrameTransformer():
    """
    Methods to correct dataset to help the analysis and exploration of the dataset. 
    """
    
    def __init__(self, table):
        """
        See help(DataFrameTransformer) for more details.

        Args:
            table (pd.dataframe): dataset to be used. 
        """
        self.table = table
        self.numeric_non_bool_columns = [x for x in self.table.select_dtypes(include=np.number) if x not in ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]
        
    def update_tables(self, to_sort):
        """
        Replace null values in each numeric column with median for that column. 

        Args:
            to_sort (dict): k:v pairs of column names and median values for that column. 

        Returns:
            pd.dataframe: dataframe with no null values
        """
        for key, value in to_sort.items():
            self.table[key].fillna(value=value, inplace=True)
        return self.table
    
    def correct_skew(self, column):
        """
        Correct skew of column specified using log transformation. 

        Args:
            column (str): column with skewed data that needs correcting. 

        Returns:
            pd.dataframe: table with skewed data corrected. 
        """
        skew_corrected = np.log(self.table[column])
        print('Skewed data corrected (log):', skew_corrected.skew())
        return skew_corrected
    
    def find_outliers(self, table, column):
        """
        Find and remove outliers that have a value outside of the upper_bound and % spread of data in Q1-Q3. 

        Args:
            table (pd.dataframe): dataframe to use. 
            column (str): name of column to check. 

        Returns:
            pd.dataframe: pd.dataframe with outlier values removed. 
        """
        drop_rows = list()
        index_location = list() 
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
        self.table.drop(index_location, axis=0, inplace=True)
        return self.table