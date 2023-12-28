# Exploratory Data Analysis


## Project description
Project to load, extract and clean data to analyse the information and generate graphs to visualise the data. 

## Project Breakdown:

### Extract Data 
The data was extracted from an Amazon Web Service database which involved connecting to the database using a SQLAlchemy engine. 

### Clean Data
Once the data was extracted, the columns were renamed and each column was converted to the correct data type. Basic information and descriptive statistics about the columns like the mean were calculated. The null values were removed and/or imputed using the median. Skewed data was transformed using log transform and outliers were removed. Any heavily correlated columns were dropped if appropriate. 
Checking the skew of data...
![image](https://github.com/CJ1608/exploratory-data-analysis---the-manufacturing-process504/assets/128046995/180ef567-f088-4f83-969c-9d99e049c3f2)
Checking for outliers...
![image](https://github.com/CJ1608/exploratory-data-analysis---the-manufacturing-process504/assets/128046995/17568284-cf52-4416-a2a3-d56f5052b245)

### Analyse Data 
Once the data was cleaned, the data was analysed to look at the failure rate of the machines, the potential causes and their relationship to the different product categories. 
The failures were plotted using scatter graphs to visualise any trends. The example below suggests that HDF failures are less likely to happen if machines are run at a rotational speed higher than 1500rpm. 
![image](https://github.com/CJ1608/exploratory-data-analysis---the-manufacturing-process504/assets/128046995/7218c459-dfa3-4ad0-afa1-6324a93e48bc)


## Installation instructions
Made using Python 3.11.5 and VS Code 1.85.1 

- Clone repo : https://github.com/CJ1608/exploratory-data-analysis---the-manufacturing-process504.git
- Check have all the needed modules and packages installed

## File structure of the project:
- .gitignore
- LICENSE.txt
- README.md
- data_cleaning.py
- data_extraction.py
- db_utils.py
- main_1.ipynb- main file that calls on methods within data_cleaning.py, data_extraction.py and db_utils.py
- requirements.txt- packages and libraries to import

  
## License information
Distributed under the MIT License. See LICENSE.txt for more information. 

## Contact 
### Email
128046995+CJ1608@users.noreply.github.com 
### Repo link
https://github.com/CJ1608/exploratory-data-analysis---the-manufacturing-process504.git

## Acknowledgements
- https://choosealicense.com/
- AiCore
