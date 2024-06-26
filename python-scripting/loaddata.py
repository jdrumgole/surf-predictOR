import pandas as pd
from sqlalchemy import create_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection details
db_username = 'postgres'  # Replace with your PostgreSQL username
db_password = 'Drumgole2'  # Replace with your PostgreSQL password
db_host = 'localhost'  # Replace with your PostgreSQL host if different
db_port = '5432'  # Default PostgreSQL port
db_name = 'wsl_surfing'  # Replace with your database name

# Create a database connection
try:
    engine = create_engine(f'postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')
    logger.info("Database connection successful.")
except Exception as e:
    logger.error(f"Error connecting to the database: {e}")
    raise

# Load CSV data into pandas DataFrames
try:
    surfers_df = pd.read_csv('surfers.csv')
    events_df = pd.read_csv('events.csv')
    performances_df = pd.read_csv('performances.csv')
    logger.info("CSV data loaded successfully.")
except Exception as e:
    logger.error(f"Error loading CSV data: {e}")
    raise

# Insert data into PostgreSQL
try:
    surfers_df.to_sql('surfers', engine, if_exists='append', index=False)
    events_df.to_sql('events', engine, if_exists='append', index=False)
    performances_df.to_sql('performances', engine, if_exists='append', index=False)
    logger.info("Data inserted into PostgreSQL successfully.")
except Exception as e:
    logger.error(f"Error inserting data into PostgreSQL: {e}")
    raise
