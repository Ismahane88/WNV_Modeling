import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.utils.multiclass import unique_labels

# Load your data
data = pd.read_csv('all data.csv', sep=';')
location_data = pd.read_csv('location_coordinates.csv')

# Convert precipitation from inches to millimeters
data['Total Monthly Precipitation'] *= 25.4

# Merge location data
data = pd.merge(data, location_data, how='left', left_on='Geo_Location', right_on='Location')

# Select environmental variables
env_variables = ['Total Monthly Precipitation', 'Monthly Maximum Temperature', 'Monthly Minimum Temperature']

# Filter only Culicidae vectors
vector_data = data[data['Host family'] == 'Culicidae'].dropna(subset=env_variables + ['Host', 'Latitude', 'Longitude']).copy()

# Features (X) and Target (y)
X = vector_data[env_variables + ['Latitude', 'Longitude']]
y = vector_data['Host']

# Encode species names
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Binarize the labels for multiclass ROC analysis
y_binarized = label_binarize(y_encoded, classes=np.arange(len(encoder.classes_)))

# Train/test split
X_train, X_test, y_train_binarized, y_test_binarized = train_test_split(X, y_binarized, test_size=0.2, random_state=42)

# Initialize and train the classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multiclass = OneVsRestClassifier(rf)
rf_multiclass.fit(X_train, y_train_binarized)

# Predict probabilities
y_pred_prob = rf_multiclass.predict_proba(X_test)

# Predict labels
y_pred_binarized = rf_multiclass.predict(X_test)
y_pred_labels = np.argmax(y_pred_binarized, axis=1)
y_test_labels = np.argmax(y_test_binarized, axis=1)



# ROC Curve plotting
plt.figure(figsize=(12, 10))
for i in range(len(encoder.classes_)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{encoder.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multiclass Random Forest')
plt.legend(loc='lower right', fontsize='small')
plt.show()


# Feature Importances
valid_estimators = []
for estimator in rf_multiclass.estimators_:
    if hasattr(estimator, 'feature_importances_'):
        valid_estimators.append(estimator)

if valid_estimators:
    feature_importances = valid_estimators[0].feature_importances_

    feature_df = pd.DataFrame({
        'Feature': env_variables + ['Latitude', 'Longitude'],
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df)
    plt.title('Feature Importances')
    plt.show()
else:
    print("No valid estimators with feature importances found.")

# Ecological Niche Modeling
# Create grid
precip_range = np.linspace(vector_data['Total Monthly Precipitation'].min(), vector_data['Total Monthly Precipitation'].max(), 100)
max_temp_range = np.linspace(vector_data['Monthly Maximum Temperature'].min(), vector_data['Monthly Maximum Temperature'].max(), 100)
min_temp_range = np.linspace(vector_data['Monthly Minimum Temperature'].min(), vector_data['Monthly Minimum Temperature'].max(), 100)

grid_points = np.array(np.meshgrid(precip_range, max_temp_range, min_temp_range)).T.reshape(-1, 3)

avg_latitude = vector_data['Latitude'].mean()
avg_longitude = vector_data['Longitude'].mean()

grid_points_with_location = np.hstack([grid_points, np.full((grid_points.shape[0], 2), [avg_latitude, avg_longitude])])

# Predict probabilities
species_probabilities = {}
for species_label in encoder.classes_:
    species_index = encoder.transform([species_label])[0]
    probabilities = rf_multiclass.predict_proba(grid_points_with_location)[:, species_index]
    species_probabilities[species_label] = probabilities

# Visualization
species_to_plot = 'Aedes japonicus'  # 🔥 Change this to your species name!

if species_to_plot in species_probabilities:
    probabilities = species_probabilities[species_to_plot]
    
    probabilities = probabilities.reshape(len(precip_range), len(max_temp_range), len(min_temp_range))

    plt.figure(figsize=(10, 8))
    sns.heatmap(probabilities.mean(axis=2), cmap='viridis', 
                cbar_kws={'label': f'Probability of WNV presence for {species_to_plot}'},
                xticklabels=precip_range[::10].astype(int),
                yticklabels=max_temp_range[::10].astype(int))
    plt.title(f'Ecological Niche Model for {species_to_plot} (WNV Presence Probability)')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Max Temperature (°C)')
    plt.show()
else:
    print(f"Species '{species_to_plot}' not found in the model.")
