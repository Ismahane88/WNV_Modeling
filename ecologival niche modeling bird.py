import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Load your data
data = pd.read_csv('birds.csv', sep=';')
location_data = pd.read_csv('location_coordinates.csv')

# Convert precipitation from inches to millimeters
data['Total Monthly Precipitation'] *= 25.4

# Merge location data
data = pd.merge(data, location_data, how='left', left_on='Geo_Location', right_on='Location')

# Select environmental variables
env_variables = ['Total Monthly Precipitation', 'Monthly Maximum Temperature', 'Monthly Minimum Temperature']

# Filter only Culicidae vectors
bird_data = data[(data['Host family'] == 'Bird') & (data['Host'].notna())].dropna(subset=env_variables).copy()

# Features (X) and Target (y)
X = bird_data[env_variables]
y = bird_data['Host']

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

# Decode integer labels to original class names
y_test_species = encoder.inverse_transform(y_test_labels)
y_pred_species = encoder.inverse_transform(y_pred_labels)

# Classification report
print("Classification Report:")
print(classification_report(y_test_species, y_pred_species, digits=3))

# Accuracy
accuracy = accuracy_score(y_test_species, y_pred_species)
print(f"Accuracy: {accuracy:.3f}")

# Precision, Recall, F1 (macro/micro/weighted)
precision_macro = precision_score(y_test_species, y_pred_species, average='macro')
recall_macro = recall_score(y_test_species, y_pred_species, average='macro')
f1_macro = f1_score(y_test_species, y_pred_species, average='macro')

print(f"Precision (Macro): {precision_macro:.3f}")
print(f"Recall (Macro): {recall_macro:.3f}")
print(f"F1 Score (Macro): {f1_macro:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test_species, y_pred_species, labels=encoder.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
plt.figure(figsize=(12, 10))
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# ROC Curve plotting
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
# Feature Importances
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

# Ecological Niche Modeling
# Create grid
precip_range = np.linspace(bird_data['Total Monthly Precipitation'].min(), bird_data['Total Monthly Precipitation'].max(), 100)
max_temp_range = np.linspace(bird_data['Monthly Maximum Temperature'].min(), bird_data['Monthly Maximum Temperature'].max(), 100)
min_temp_range = np.linspace(bird_data['Monthly Minimum Temperature'].min(), bird_data['Monthly Minimum Temperature'].max(), 100)

# Create grid points (make sure these are 1D arrays)
grid_points = np.array(np.meshgrid(precip_range, max_temp_range, min_temp_range)).T.reshape(-1, 3)

# Function to predict in batches
def predict_in_batches(model, X, batch_size=1000):
    probs_list = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        probs = model.predict_proba(batch).astype(np.float32)  # Convert to float32
        probs_list.append(probs)
    return np.vstack(probs_list)

# Predict probabilities for each bird species
Bird_species_probabilities = {}
for Bird_species_label in encoder.classes_:
    Bird_species_index = encoder.transform([Bird_species_label])[0]
    probabilities = predict_in_batches(rf_multiclass, grid_points)[:, Bird_species_index]
    Bird_species_probabilities[Bird_species_label] = probabilities



# Create output directory if it doesn't exist
output_dir = "saved_bird_models"
os.makedirs(output_dir, exist_ok=True)

# Visualization and Saving
selected_Bird_species = [
    'Corvus brachyrhynchos', 'Aphelocoma californica', 'Passer domesticus',
    'Accipiter gentilis', 'Cyanocitta cristata', 'Haemorhous mexicanus'
]

for Bird_species_to_plot in selected_Bird_species:
    if Bird_species_to_plot in Bird_species_probabilities:
        try:
            probabilities = Bird_species_probabilities[Bird_species_to_plot]
            
            # Reshape safely
            expected_shape = (len(precip_range), len(max_temp_range), len(min_temp_range))
            if probabilities.size != np.prod(expected_shape):
                raise ValueError(f"Unexpected size for probabilities array: {probabilities.size}")

            probabilities = probabilities.reshape(expected_shape)
            mean_probabilities = probabilities.mean(axis=2)

            # Plot heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(mean_probabilities, 
                        cmap='viridis', 
                        cbar_kws={'label': f'Probability of WNV presence for {Bird_species_to_plot}'},
                        xticklabels=10,
                        yticklabels=10)

            # Customize tick labels
            plt.xticks(
                ticks=np.linspace(0, len(precip_range)-1, 10), 
                labels=np.round(np.linspace(precip_range.min(), precip_range.max(), 10)).astype(int), 
                rotation=45, ha='right'
            )
            
            plt.yticks(
                ticks=np.linspace(0, len(max_temp_range)-1, 10), 
                labels=np.round(np.linspace(max_temp_range.min(), max_temp_range.max(), 10)).astype(int)
            )

            plt.title(f'Ecological Niche Model for {Bird_species_to_plot} (WNV Presence Probability)', fontsize=16)
            plt.xlabel('Precipitation (mm)', fontsize=12)
            plt.ylabel('Max Temperature (°C)', fontsize=12)
            plt.tight_layout()
            plt.show()

            # Save to .npy
            clean_name = ''.join(c for c in Bird_species_to_plot.lower().replace(" ", "_") if c.isalnum() or c == "_")
            np.save(os.path.join(output_dir, f"bird_prob_{clean_name}.npy"), mean_probabilities)

        except Exception as e:
            print(f"Error processing {Bird_species_to_plot}: {e}")
    else:
        print(f"Species '{Bird_species_to_plot}' not found in the model.")

