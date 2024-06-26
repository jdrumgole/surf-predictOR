import pandas as pd

surfers_data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['John John Florence', 'Jane smith', 'Mark Johnson', 'Emily Davis', 'Michael Brown'],
    'country': ['USA','Australia','Africa','USA','Brazil']
}

events_data = {
    'id': [1, 2, 3],
    'name': ['event1', 'event2','event3'],
    'location': ['Pipeline', 'Bells Beach','Supertubos'],
    'wave_height': [2.5, 3.0, 1.6],
    'wave_type': ['reef', 'point break', 'reef']
}

performances_data = {
    'id': [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15],
    'surfer_id': [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5],
    'event_id': [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
    'score': [8.5,7.0,4.0,5.0,9.0,3.0,7.0,6.0,3.0,4.0,8.0,7.0,5.0,6.0,9.0],
    'winner': [False,False,False,False,True,False,True,False,False,False,False,False,False,False,True]
}

surfers_df = pd.DataFrame(surfers_data)
events_df = pd.DataFrame(events_data)
performances_df = pd.DataFrame(performances_data)

surfers_df.to_csv('surfers.csv', index=False)
events_df.to_csv('events.csv', index=False)
performances_df.to_csv('performances.csv', index=False)