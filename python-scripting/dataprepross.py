import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging

# Ensure the logs and models directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Configure logging
logging.basicConfig(filename='logs/preprocess_and_train.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection details
db_username = 'postgres'
db_password = 'Drumgole2'
db_host = 'localhost'
db_port = '5432'
db_name = 'wsl_surfing'

# Create a database connection
try:
    engine = create_engine(f'postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')
    logger.info("Database connection successful.")
except Exception as e:
    logger.error(f"Error connecting to the database: {e}")
    raise

# Load data from PostgreSQL
try:
    query = '''
    SELECT p.*, s.name AS surfer_name, e.location, e.wave_height, e.wave_type
    FROM performances p
    JOIN surfers s ON p.surfer_id = s.id
    JOIN events e ON p.event_id = e.id
    '''
    data = pd.read_sql(query, engine)
    logger.info("Data loaded from PostgreSQL successfully.")
except Exception as e:
    logger.error(f"Error loading data from PostgreSQL: {e}")
    raise

# Preprocess data
try:
    # Encode categorical features
    surfer_name_encoder = LabelEncoder()
    location_encoder = LabelEncoder()
    wave_type_encoder = LabelEncoder()
    data['location'] = location_encoder.fit_transform(data['location'])
    data['surfer_name'] = surfer_name_encoder.fit_transform(data['surfer_name'])
    data['wave_type'] = wave_type_encoder.fit_transform(data['wave_type'])

    # Feature Engineering
    data['average_score'] = data.groupby('surfer_name')['score'].transform('mean')
    data['event_count'] = data.groupby('surfer_name')['event_id'].transform('count')

    # Define feature order and save it
    feature_order = ['location', 'surfer_name', 'wave_height', 'wave_type', 'score', 'average_score', 'event_count']
    with open('models/feature_order.pkl', 'wb') as f:
        joblib.dump(feature_order, f)

    # Save label encoders
    joblib.dump(surfer_name_encoder, 'models/surfer_name_encoder.pkl')
    joblib.dump(location_encoder, 'models/location_encoder.pkl')
    joblib.dump(wave_type_encoder, 'models/wave_type_encoder.pkl')

    # Split the data into features and target
    X = data[feature_order]
    y = data['winner']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and scaler
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)  # Handle zero-division
    logger.info(f"Model evaluation completed successfully.\nAccuracy: {accuracy}\n{report}")
except Exception as e:
    logger.error(f"Error preprocessing data: {e}")
    raise
