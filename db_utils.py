# import PyYAML
from sqlalchemy import create_engine, inspect
# import sqlalchemy
import psycopg2
import yaml 

class RDSDatabaseConnector():
    """
    M2, T1- Methods to extract data from RDS database. 
    """
    #M2, T4 write init method take in a dict of credentials from T3
    def __init__(self, yaml_file):
        """
        See help(DatabaseConnector) for more detail. 

        Args:
            yaml_file: connection details and passwords to connect to database. 
        """
        self.yaml_file = yaml_file

    #M2, T3 loads credentials and returns data within it, should be passed to RDSDatabaseConnector as an argument
    def read_db_creds(self):
        """
        Reads and returns sensitive database connection credentials, called in the init_db_engine and init_sql_connection methods. 
        Yaml files added to git ignore file.

        Returns: 
            db_credentials: credentials to connect to a database. 
        """
        with open(self.yaml_file, 'r') as file:
            self.db_credentials = yaml.safe_load(file)
            return self.db_credentials

    #M2, T5 initialise SQLALchemy engine 
    def init_db_engine(self):
        """
        Read the credentials from the return of read_db_creds and initialise and return an sqlalchemy database engine to get data from 
        Amazon database. 

        Returns:
            engine (engine): database connection to an Amazon database via a sqlalchemy engine. 
        """
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.db_credentials['RDS_HOST']
        USER = self.db_credentials['RDS_USER']
        PASSWORD = self.db_credentials['RDS_PASSWORD']
        DATABASE = self.db_credentials['RDS_DATABASE']
        PORT = self.db_credentials['RDS_PORT']
        self.engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}") 
        return self.engine 
    
    def list_db_tables(self):
        """
        Use engine returned from init_db_engine to list all the tables in the database. 

        Returns:
            table_names (list[str]): list of table names in Amazon database. 
        """
        inspector = inspect(self.engine)
        self.table_names = inspector.get_table_names()
        return self.table_names