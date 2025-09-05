import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


mosq_df = pd.read_csv("mosquitoes.csv", sep=";")
bird_df = pd.read_csv("birds.csv", sep=";")


mosq_df['Total Monthly Precipitation'] = mosq_df['Total Monthly Precipitation'] * 25.4


vector_layer = np.load("vector_layer.npy")
bird_dir = "saved_bird_models"
selected_bird_files = [
    "vector_prob_accipiter_gentilis.npy",
    "vector_prob_aphelocoma_californica.npy",
    "vector_prob_corvus_brachyrhynchos.npy",
    "vector_prob_cyanocitta_cristata.npy",
    "vector_prob_haemorhous_mexicanus.npy",
    "vector_prob_passer_domesticus.npy"
]


scaler = MinMaxScaler()
bird_stack = []
for file in selected_bird_files:
    bird_path = os.path.join(bird_dir, file)
    bird_layer = np.load(bird_path)
    bird_layer = scaler.fit_transform(bird_layer.reshape(-1, 1)).reshape(bird_layer.shape)
    bird_stack.append(bird_layer)

bird_combined = np.mean(bird_stack, axis=0)


max_temperature = mosq_df['Monthly Maximum Temperature'].values.flatten()
min_temperature = mosq_df['Monthly Minimum Temperature'].values.flatten()  
precipitation = mosq_df['Total Monthly Precipitation'].values.flatten()


if len(vector_layer.shape) > 1:
    vector_layer = vector_layer.mean(axis=1).flatten()


min_length = min(len(max_temperature), len(min_temperature), len(precipitation), 
                len(vector_layer), len(bird_combined.flatten()))
max_temperature = max_temperature[:min_length]
min_temperature = min_temperature[:min_length]  
precipitation = precipitation[:min_length]
vector_layer = vector_layer[:min_length]
bird_combined = bird_combined.flatten()[:min_length]

def calculate_optimal_range(values, activity):
    """Calculate optimal range where activity is highest"""
    threshold = np.percentile(activity, 90)
    optimal_mask = activity >= threshold
    return (values[optimal_mask].min(), values[optimal_mask].max())

def calculate_risk_levels(values, activity):
    """Calculate risk level thresholds"""
    return {
        'low': {
            'max': np.percentile(values[activity < np.percentile(activity, 25)], 95),
            'min': values.min()
        },
        'moderate': {
            'max': np.percentile(values[activity < np.percentile(activity, 50)], 95),
            'min': np.percentile(values[activity >= np.percentile(activity, 25)], 5)
        },
        'high': {
            'max': np.percentile(values[activity < np.percentile(activity, 75)], 95),
            'min': np.percentile(values[activity >= np.percentile(activity, 50)], 5)
        },
        'extreme': {
            'max': values.max(),
            'min': np.percentile(values[activity >= np.percentile(activity, 75)], 5)
        }
    }


results = {
    'optimal_temp_ranges': {
        'max_temp': calculate_optimal_range(max_temperature, vector_layer),
        'min_temp': calculate_optimal_range(min_temperature, vector_layer)  
    },
    'optimal_precip_range': calculate_optimal_range(precipitation, vector_layer),
    'high_risk_thresholds': {
        'vector': np.percentile(vector_layer, 75),
        'host': np.percentile(bird_combined, 75)
    },
    'temp_risk_for_vector': {
        'max_temp': calculate_risk_levels(max_temperature, vector_layer),
        'min_temp': calculate_risk_levels(min_temperature, vector_layer)  
    },
    'precip_risk_for_vector': calculate_risk_levels(precipitation, vector_layer),
    'temp_risk_for_host': {
        'max_temp': calculate_risk_levels(max_temperature, bird_combined),
        'min_temp': calculate_risk_levels(min_temperature, bird_combined)  
    },
    'precip_risk_for_host': calculate_risk_levels(precipitation, bird_combined)
}

print(results)

import json
import numpy as np
from datetime import datetime


def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(x) for x in obj]
    return obj

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_filename = f"risk_parameters_{timestamp}.json"

with open(json_filename, 'w') as f:
    json.dump(numpy_to_python(results), f, indent=4)

print(f"Results saved to {json_filename}")


npz_filename = f"risk_parameters_{timestamp}.npz"


np.savez(npz_filename,
         optimal_max_temp_min=results['optimal_temp_ranges']['max_temp'][0],
         optimal_max_temp_max=results['optimal_temp_ranges']['max_temp'][1],
         optimal_min_temp_min=results['optimal_temp_ranges']['min_temp'][0],
         optimal_min_temp_max=results['optimal_temp_ranges']['min_temp'][1],
         optimal_precip_min=results['optimal_precip_range'][0],
         optimal_precip_max=results['optimal_precip_range'][1],
         high_risk_vector=results['high_risk_thresholds']['vector'],
         high_risk_host=results['high_risk_thresholds']['host'],
         
         vector_temp_low_max=results['temp_risk_for_vector']['max_temp']['low']['max'],
         vector_temp_low_min=results['temp_risk_for_vector']['max_temp']['low']['min'],
         vector_temp_moderate_max=results['temp_risk_for_vector']['max_temp']['moderate']['max'],
         vector_temp_moderate_min=results['temp_risk_for_vector']['max_temp']['moderate']['min'],
         vector_temp_high_max=results['temp_risk_for_vector']['max_temp']['high']['max'],
         vector_temp_high_min=results['temp_risk_for_vector']['max_temp']['high']['min'],
         vector_temp_extreme_max=results['temp_risk_for_vector']['max_temp']['extreme']['max'],
         vector_temp_extreme_min=results['temp_risk_for_vector']['max_temp']['extreme']['min'],
         
         )

print(f"Binary parameters saved to {npz_filename}")


csv_filename = f"risk_parameters_{timestamp}.csv"


csv_data = {
    'parameter': [],
    'type': [], 
    'variable': [],  
    'risk_level': [], 
    'min_value': [],
    'max_value': []
}

def add_to_csv(parameter_dict, param_type, variable_name):
    for level, values in parameter_dict.items():
        csv_data['parameter'].append(f"{variable_name}_{level}")
        csv_data['type'].append(param_type)
        csv_data['variable'].append(variable_name)
        csv_data['risk_level'].append(level)
        csv_data['min_value'].append(values['min'])
        csv_data['max_value'].append(values['max'])


add_to_csv(results['temp_risk_for_vector']['max_temp'], 'vector', 'temp_max')
add_to_csv(results['temp_risk_for_vector']['min_temp'], 'vector', 'temp_min')
add_to_csv(results['precip_risk_for_vector'], 'vector', 'precip')
add_to_csv(results['temp_risk_for_host']['max_temp'], 'host', 'temp_max')
add_to_csv(results['temp_risk_for_host']['min_temp'], 'host', 'temp_min')
add_to_csv(results['precip_risk_for_host'], 'host', 'precip')


for var_name in ['max_temp', 'min_temp']:
    csv_data['parameter'].append(f"optimal_{var_name}")
    csv_data['type'].append('both')
    csv_data['variable'].append(var_name)
    csv_data['risk_level'].append('optimal')
    csv_data['min_value'].append(results['optimal_temp_ranges'][var_name][0])
    csv_data['max_value'].append(results['optimal_temp_ranges'][var_name][1])

csv_data['parameter'].append("optimal_precip")
csv_data['type'].append('both')
csv_data['variable'].append('precip')
csv_data['risk_level'].append('optimal')
csv_data['min_value'].append(results['optimal_precip_range'][0])
csv_data['max_value'].append(results['optimal_precip_range'][1])

pd.DataFrame(csv_data).to_csv(csv_filename, index=False)
print(f"CSV parameters saved to {csv_filename}")