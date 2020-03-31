import numpy as np
import pandas as pd
import pandas.io.sql as pd_sql
import psycopg2 as pg
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def connect_to_psql(host, user, port):
    '''Initial connection to postgreSQL without establishing a database, which has not been created yet.'''
    connection_args = {
                        'host': host,
                       	'user': user,
                        'port': port
                        }
    connection = pg.connect(**connection_args)
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT);
    cursor = connection.cursor()
    return connection, cursor

def create_psql_db(host, user, port, create_db):
    '''Creates postgreSQL database with parameters specififed in the connect_to_psql function'''
    connection, cursor = connect_to_psql(host, user, port)
    db_name = create_db
    command_string_db = "CREATE DATABASE "+db_name+";"
    cursor.execute(command_string_db)
    connection.commit()
    cursor.close()
    connection.close()
    return 

def connect_to_psql_db(host, db, user, port):
    '''Connects to an already established postgreSQL database'''
    connection_args = {
                        'host': host,
                        'dbname': db,   
                       	'user': user,
                        'port': port
                        }
    connection = pg.connect(**connection_args)
    connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT);
    cursor = connection.cursor()
    return connection, cursor

def create_psql_tbl(host, db, user, port, csv_loc_source, create_tbl):
    '''Creates a table in a postgreSQL database with parameters set up in the connect_to_psql_db function'''
    connection, cursor = connect_to_psql_db(host, db, user, port)   
    #ed_visits_df = pd.read_csv(csv_loc_source).replace(np.nan, '', regex=True)
    tbl_name = create_tbl
    command_string_tbl = pd_sql.get_schema(ed_visits_df.reset_index(), create_tbl)    
    cursor.execute(command_string_tbl)
    connection.commit()    
    cursor.execute(f"COPY {create_tbl} FROM '{csv_loc_source}' DELIMITER ',' CSV HEADER;")
    connection.commit()
    cursor.close()
    connection.close()
    return 

def alter_table_psql(host, db, user, port, tbl, col, coltype):
    '''Adds columns to existing postgreSQL tables and allows specification of column types'''
    connection, cursor = connect_to_psql_db(host, db, user, port)
    cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {coltype};")
    connection.commit()
    cursor.close()
    connection.close()
    return


def update_table_psql(host, db, user, port, tbl, col):
    '''Updates a specific column in existing postgreSQL table'''
    connection, cursor = connect_to_psql_db(host, db, user, port)
    command_str = f"""            
                UPDATE {tbl} 
                SET {col} = 
                CASE WHEN disposition = 'Discharge' THEN 0
                WHEN disposition = 'Admit' THEN 1
                END ;
                """
    cursor.execute(command_str)
    connection.commit()
    cursor.close()
    connection.close()
    return


def query_table_psql(host, db, user, port, tbl, cols_str, criteria_str):
    '''Simple postgreSQL query function, specifying search columns, tables, and search criteria'''
    connection, cursor = connect_to_psql_db(host, db, user, port)
    query_str = f"""            
                SELECT {cols_str}
                FROM {tbl}
                WHERE {criteria_str}
                ;
                """
    results = pd_sql.read_sql(query_str, connection)
    cursor.close()
    connection.close()
    return results

