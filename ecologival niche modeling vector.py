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




data = pd.read_csv('mosquitoes.csv', sep=';')
location_data = pd.read_csv('location_coordinates.csv')


data['Total Monthly Precipitation'] *= 25.4


data = pd.merge(data, location_data, how='left', left_on='Geo_Location', right_on='Location')


env_variables = ['Total Monthly Precipitation', 'Monthly Maximum Temperature', 'Monthly Minimum Temperature']


vector_data = data[(data['Host family'] == 'Culicidae') & (data['Host'].notna())].dropna(subset=env_variables).copy()


X = vector_data[env_variables]
y = vector_data['Host']


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


y_binarized = label_binarize(y_encoded, classes=np.arange(len(encoder.classes_)))


X_train, X_test, y_train_binarized, y_test_binarized = train_test_split(X, y_binarized, test_size=0.2, random_state=42)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multiclass = OneVsRestClassifier(rf)
rf_multiclass.fit(X_train, y_train_binarized)


y_pred_prob = rf_multiclass.predict_proba(X_test)


y_pred_binarized = rf_multiclass.predict(X_test)
y_pred_labels = np.argmax(y_pred_binarized, axis=1)
y_test_labels = np.argmax(y_test_binarized, axis=1)



plt.figure(figsize=(12, 10))
for i in range(len(encoder.classes_)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)

    if not np.isnan(roc_auc) and roc_auc >= 0.70:
        plt.plot(fpr, tpr, label=f'{encoder.classes_[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multiclass Random Forest')
plt.legend(loc='lower right', fontsize='small')
plt.show()

valid_estimators = []
for estimator in rf_multiclass.estimators_:
    if hasattr(estimator, 'feature_importances_'):
        valid_estimators.append(estimator)

if valid_estimators:
    feature_importances = valid_estimators[0].feature_importances_

    feature_df = pd.DataFrame({
        'Feature': env_variables,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df)
    plt.title('Feature Importances')
    plt.tight_layout()  
    plt.show()
else:
    print("No valid estimators with feature importances found.")



precip_range = np.linspace(vector_data['Total Monthly Precipitation'].min(), vector_data['Total Monthly Precipitation'].max(), 100)
max_temp_range = np.linspace(vector_data['Monthly Maximum Temperature'].min(), vector_data['Monthly Maximum Temperature'].max(), 100)
min_temp_range = np.linspace(vector_data['Monthly Minimum Temperature'].min(), vector_data['Monthly Minimum Temperature'].max(), 100)

grid_points = np.array(np.meshgrid(precip_range, max_temp_range, min_temp_range)).T.reshape(-1, 3)


species_probabilities = {}
for species_label in encoder.classes_:
    species_index = encoder.transform([species_label])[0]
    probabilities = rf_multiclass.predict_proba(grid_points)[:, species_index]
    species_probabilities[species_label] = probabilities


species_to_plot = 'Culex pipiens'  

if species_to_plot in species_probabilities:
    probabilities = species_probabilities[species_to_plot]
    probabilities = probabilities.reshape(len(precip_range), len(max_temp_range), len(min_temp_range))

    mean_probabilities = probabilities.mean(axis=2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_probabilities, 
                cmap='viridis', 
                cbar_kws={'label': f'Probability of WNV presence for {species_to_plot}'},
                xticklabels=10,  
                yticklabels=10)

    
    plt.xticks(ticks=np.linspace(0, len(precip_range)-1, 10), 
               labels=np.round(np.linspace(precip_range.min(), precip_range.max(), 10)).astype(int), 
               rotation=45)

    plt.yticks(ticks=np.linspace(0, len(max_temp_range)-1, 10), 
               labels=np.round(np.linspace(max_temp_range.min(), max_temp_range.max(), 10)).astype(int))

    plt.title(f'Ecological Niche Model for {species_to_plot} (WNV Presence Probability)')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Max Temperature (°C)')
    plt.tight_layout()
    plt.show()

    np.save("vector_prob_culex_pipiens.npy", mean_probabilities)

species_to_plot = 'Culex quinquefasciatus'  

if species_to_plot in species_probabilities:
    probabilities = species_probabilities[species_to_plot]
    probabilities = probabilities.reshape(len(precip_range), len(max_temp_range), len(min_temp_range))

    mean_probabilities = probabilities.mean(axis=2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_probabilities, 
                cmap='viridis', 
                cbar_kws={'label': f'Probability of WNV presence for {species_to_plot}'},
                xticklabels=10,  
                yticklabels=10)

    
    plt.xticks(ticks=np.linspace(0, len(precip_range)-1, 10), 
               labels=np.round(np.linspace(precip_range.min(), precip_range.max(), 10)).astype(int), 
               rotation=45)

    plt.yticks(ticks=np.linspace(0, len(max_temp_range)-1, 10), 
               labels=np.round(np.linspace(max_temp_range.min(), max_temp_range.max(), 10)).astype(int))

    plt.title(f'Ecological Niche Model for {species_to_plot} (WNV Presence Probability)')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Max Temperature (°C)')
    plt.tight_layout()
    plt.show()

    np.save("vector_prob_culex_quinquefasciatus.npy", mean_probabilities)  



species_to_plot = 'Aedes japonicus'  

if species_to_plot in species_probabilities:
    probabilities = species_probabilities[species_to_plot]
    probabilities = probabilities.reshape(len(precip_range), len(max_temp_range), len(min_temp_range))

    mean_probabilities = probabilities.mean(axis=2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_probabilities, 
                cmap='viridis', 
                cbar_kws={'label': f'Probability of WNV presence for {species_to_plot}'},
                xticklabels=10,  
                yticklabels=10)

    
    plt.xticks(ticks=np.linspace(0, len(precip_range)-1, 10), 
               labels=np.round(np.linspace(precip_range.min(), precip_range.max(), 10)).astype(int), 
               rotation=45)

    plt.yticks(ticks=np.linspace(0, len(max_temp_range)-1, 10), 
               labels=np.round(np.linspace(max_temp_range.min(), max_temp_range.max(), 10)).astype(int))

    plt.title(f'Ecological Niche Model for {species_to_plot} (WNV Presence Probability)')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Max Temperature (°C)')
    plt.tight_layout()
    plt.show()

    np.save("vector_prob_aedes_japonicus.npy", mean_probabilities)  



species_to_plot = 'Ochlerotatus caspius'  

if species_to_plot in species_probabilities:
    probabilities = species_probabilities[species_to_plot]
    probabilities = probabilities.reshape(len(precip_range), len(max_temp_range), len(min_temp_range))

    mean_probabilities = probabilities.mean(axis=2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_probabilities, 
                cmap='viridis', 
                cbar_kws={'label': f'Probability of WNV presence for {species_to_plot}'},
                xticklabels=10,  
                yticklabels=10)

    
    plt.xticks(ticks=np.linspace(0, len(precip_range)-1, 10), 
               labels=np.round(np.linspace(precip_range.min(), precip_range.max(), 10)).astype(int), 
               rotation=45)

    plt.yticks(ticks=np.linspace(0, len(max_temp_range)-1, 10), 
               labels=np.round(np.linspace(max_temp_range.min(), max_temp_range.max(), 10)).astype(int))

    plt.title(f'Ecological Niche Model for {species_to_plot} (WNV Presence Probability)')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Max Temperature (°C)')
    plt.tight_layout()
    plt.show()

    np.save("vector_prob_Ochlerotatus_caspius.npy", mean_probabilities)

species_to_plot = 'Culiseta melanura'  

if species_to_plot in species_probabilities:
    probabilities = species_probabilities[species_to_plot]
    probabilities = probabilities.reshape(len(precip_range), len(max_temp_range), len(min_temp_range))

    mean_probabilities = probabilities.mean(axis=2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_probabilities, 
                cmap='viridis', 
                cbar_kws={'label': f'Probability of WNV presence for {species_to_plot}'},
                xticklabels=10,  
                yticklabels=10)

    
    plt.xticks(ticks=np.linspace(0, len(precip_range)-1, 10), 
               labels=np.round(np.linspace(precip_range.min(), precip_range.max(), 10)).astype(int), 
               rotation=45)

    plt.yticks(ticks=np.linspace(0, len(max_temp_range)-1, 10), 
               labels=np.round(np.linspace(max_temp_range.min(), max_temp_range.max(), 10)).astype(int))

    plt.title(f'Ecological Niche Model for {species_to_plot} (WNV Presence Probability)')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Max Temperature (°C)')
    plt.tight_layout()
    plt.show()

    np.save("vector_prob_Culiseta_melanura.npy", mean_probabilities)


species_to_plot = 'Culex salinarius'  

if species_to_plot in species_probabilities:
    probabilities = species_probabilities[species_to_plot]
    probabilities = probabilities.reshape(len(precip_range), len(max_temp_range), len(min_temp_range))

    mean_probabilities = probabilities.mean(axis=2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_probabilities, 
                cmap='viridis', 
                cbar_kws={'label': f'Probability of WNV presence for {species_to_plot}'},
                xticklabels=10,  
                yticklabels=10)

    
    plt.xticks(ticks=np.linspace(0, len(precip_range)-1, 10), 
               labels=np.round(np.linspace(precip_range.min(), precip_range.max(), 10)).astype(int), 
               rotation=45)

    plt.yticks(ticks=np.linspace(0, len(max_temp_range)-1, 10), 
               labels=np.round(np.linspace(max_temp_range.min(), max_temp_range.max(), 10)).astype(int))

    plt.title(f'Ecological Niche Model for {species_to_plot} (WNV Presence Probability)')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Max Temperature (°C)')
    plt.tight_layout()
    plt.show()

    np.save("vector_prob_Culex_salinarius.npy", mean_probabilities)

species_to_plot = 'Culex restuans'  

if species_to_plot in species_probabilities:
    probabilities = species_probabilities[species_to_plot]
    probabilities = probabilities.reshape(len(precip_range), len(max_temp_range), len(min_temp_range))

    mean_probabilities = probabilities.mean(axis=2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_probabilities, 
                cmap='viridis', 
                cbar_kws={'label': f'Probability of WNV presence for {species_to_plot}'},
                xticklabels=10,  
                yticklabels=10)

    
    plt.xticks(ticks=np.linspace(0, len(precip_range)-1, 10), 
               labels=np.round(np.linspace(precip_range.min(), precip_range.max(), 10)).astype(int), 
               rotation=45)

    plt.yticks(ticks=np.linspace(0, len(max_temp_range)-1, 10), 
               labels=np.round(np.linspace(max_temp_range.min(), max_temp_range.max(), 10)).astype(int))

    plt.title(f'Ecological Niche Model for {species_to_plot} (WNV Presence Probability)')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Max Temperature (°C)')
    plt.tight_layout()
    plt.show()

    np.save("vector_prob_Culex_restuans.npy", mean_probabilities)

species_to_plot = 'Culex tarsalis'  

if species_to_plot in species_probabilities:
    probabilities = species_probabilities[species_to_plot]
    probabilities = probabilities.reshape(len(precip_range), len(max_temp_range), len(min_temp_range))

    mean_probabilities = probabilities.mean(axis=2)

    plt.figure(figsize=(12, 8))
    sns.heatmap(mean_probabilities, 
                cmap='viridis', 
                cbar_kws={'label': f'Probability of WNV presence for {species_to_plot}'},
                xticklabels=10,  
                yticklabels=10)

    
    plt.xticks(ticks=np.linspace(0, len(precip_range)-1, 10), 
               labels=np.round(np.linspace(precip_range.min(), precip_range.max(), 10)).astype(int), 
               rotation=45)

    plt.yticks(ticks=np.linspace(0, len(max_temp_range)-1, 10), 
               labels=np.round(np.linspace(max_temp_range.min(), max_temp_range.max(), 10)).astype(int))

    plt.title(f'Ecological Niche Model for {species_to_plot} (WNV Presence Probability)')
    plt.xlabel('Precipitation (mm)')
    plt.ylabel('Max Temperature (°C)')
    plt.tight_layout()
    plt.show()

    np.save("vector_prob_Culex_tarsalis.npy", mean_probabilities)

else:
    print(f"Species '{species_to_plot}' not found in the model.")

vector_probs = [
    np.load("vector_prob_culex_pipiens.npy"),
    np.load("vector_prob_culex_quinquefasciatus.npy"),
    np.load("vector_prob_aedes_japonicus.npy"),
    np.load("vector_prob_Ochlerotatus_caspius.npy"),
    np.load("vector_prob_Culiseta_melanura.npy"),
    np.load("vector_prob_Culex_salinarius.npy"),
    np.load("vector_prob_Culex_restuans.npy"),
    np.load("vector_prob_Culex_tarsalis.npy")
]
vector_layer = np.mean(vector_probs, axis=0)
np.save("vector_layer.npy", vector_layer)