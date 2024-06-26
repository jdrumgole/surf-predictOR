import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder
import joblib
import logging

# Ensure the logs and models directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Configure logging
logging.basicConfig(filename='logs/make_predictions.log', level=logging.INFO)
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

# Load model, scaler, feature order, and encoders
try:
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_order = joblib.load('models/feature_order.pkl')
    surfer_name_encoder = joblib.load('models/surfer_name_encoder.pkl')
    location_encoder = joblib.load('models/location_encoder.pkl')
    wave_type_encoder = joblib.load('models/wave_type_encoder.pkl')
    logger.info("Model, scaler, feature order, and encoders loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model, scaler, feature order, or encoders: {e}")
    raise

# Function to handle unseen labels
def encode_label(encoder, value):
    if value not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, value)
        encoder.fit(encoder.classes_)
    return encoder.transform([value])[0]

# Example upcoming event data
upcoming_event_data = {
    'location': 'Pipeline',
    'surfer_name': 'John Doe',
    'wave_height': 2.5,
    'wave_type': 'reef',
    'score': 8.0
}

# Convert upcoming event data to DataFrame
upcoming_event_df = pd.DataFrame([upcoming_event_data])
logger.info(f"Upcoming event data DataFrame:\n{upcoming_event_df}")

# Compute necessary features (average_score and event_count)
try:
    upcoming_event_df['average_score'] = 7.5
    upcoming_event_df['event_count'] = 5

    logger.info(f"Upcoming event data with computed features:\n{upcoming_event_df}")

    # Encode categorical features using the loaded encoders
    location_encoded = encode_label(location_encoder, upcoming_event_df.loc[0, 'location'])
    surfer_name_encoded = encode_label(surfer_name_encoder, upcoming_event_df.loc[0, 'surfer_name'])
    wave_type_encoded = encode_label(wave_type_encoder, upcoming_event_df.loc[0, 'wave_type'])

    # Update the DataFrame with encoded values
    upcoming_event_df.loc[0, 'location'] = location_encoded
    upcoming_event_df.loc[0, 'surfer_name'] = surfer_name_encoded
    upcoming_event_df.loc[0, 'wave_type'] = wave_type_encoded

    logger.info(f"Upcoming event data with encoded features:\n{upcoming_event_df}")

    # Ensure the feature order matches the training data
    upcoming_event_df = upcoming_event_df[feature_order]

    logger.info(f"Feature order: {feature_order}")
    logger.info(f"Upcoming event data in feature order:\n{upcoming_event_df}")

    # Normalize numerical features
    upcoming_event_df_normalized = scaler.transform(upcoming_event_df)
    logger.info(f"Upcoming event data after normalization:\n{upcoming_event_df_normalized}")
except Exception as e:
    logger.error(f"Error preprocessing upcoming event data: {e}")
    raise

# Make prediction
try:
    prediction = model.predict(upcoming_event_df_normalized)
    logger.info(f"Prediction array: {prediction}")
    
    # Log the classes to verify mapping
    logger.info(f"Classes in surfer_name_encoder: {surfer_name_encoder.classes_}")
    
    # Decode the prediction
    predicted_winner_index = np.argmax(prediction)
    predicted_winner = surfer_name_encoder.inverse_transform([predicted_winner_index])[0]
        
    logger.info(f"Prediction made successfully. Predicted winner: {predicted_winner}")
except Exception as e:
    logger.error(f"Error making prediction: {e}")
    raise

# Store the prediction in PostgreSQL
try:
    with engine.begin() as connection:
        sql = text('INSERT INTO predictions (event_id, predicted_winner) VALUES (:event_id, :predicted_winner)')
        connection.execute(sql, {'event_id': 1, 'predicted_winner': predicted_winner})
    logger.info("Prediction stored in PostgreSQL successfully.")
except Exception as e:
    logger.error(f"Error storing prediction in PostgreSQL: {e}")
    raise

# Verify the prediction in PostgreSQL
try:
    with engine.connect() as connection:
        result = connection.execute(text('SELECT * FROM predictions')).fetchall()
        logger.info(f"Predictions table content:\n{result}")
except Exception as e:
    logger.error(f"Error retrieving predictions from PostgreSQL: {e}")
    raise
