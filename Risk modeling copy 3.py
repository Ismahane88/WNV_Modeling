import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap


risk_cmap = LinearSegmentedColormap.from_list('risk_cmap', ['#2ecc71', '#f1c40f', '#e74c3c'])


def load_normalize(path):
    data = np.load(path)
    return MinMaxScaler().fit_transform(data.reshape(-1, 1)).reshape(data.shape)


vector_layer = np.load("vector_layer.npy")
bird_dir = "saved_bird_models"
bird_files = [f for f in os.listdir(bird_dir) if f.endswith('.npy')]
bird_combined = np.mean([load_normalize(os.path.join(bird_dir, f)) for f in bird_files], axis=0) 


vector_layer = (vector_layer - vector_layer.min()) / (vector_layer.max() - vector_layer.min())
bird_combined = (bird_combined - bird_combined.min()) / (bird_combined.max() - bird_combined.min())


temp_range = np.linspace(10, 35, vector_layer.shape[1])  
precip_range = np.linspace(0, 254, vector_layer.shape[0]) 


vector_threshold = 0.7  
bird_threshold = 0.7     


def get_presence_coords(model, threshold):
    ys, xs = np.where(model >= threshold)
    x_coords = temp_range[xs]
    y_coords = precip_range[ys]
    return x_coords, y_coords


mosquito_x, mosquito_y = get_presence_coords(vector_layer, vector_threshold)
bird_x, bird_y = get_presence_coords(bird_combined, bird_threshold)


fig, ax = plt.subplots(figsize=(14, 8))


risk_index = vector_layer * bird_combined
risk_plot = ax.contourf(temp_range, precip_range, risk_index, 
                       levels=20, cmap='RdYlGn_r', alpha=0.6)


ax.scatter(mosquito_x, mosquito_y, color='#0d0d0c', s=15, 
          alpha=0.6, label='Mosquito Presence', edgecolor='white', linewidth=0.5)
ax.scatter(bird_x, bird_y, color='#0e1ded', s=15, 
          alpha=0.6, label='Bird Presence', edgecolor='white', linewidth=0.5)




opt_box = patches.Rectangle((10, 0), 25, 254, linewidth=2,
                           edgecolor='#FF4500', facecolor='none',
                           linestyle='--', label='Optimal Conditions')
ax.add_patch(opt_box)



optimal_text = (
    "Optimal Conditions:\n"
    "• Temp vector and host: 10-35°C\n"
    "• Precip vector: 0-254mm\n"
    "• Precip host: 0-228.3mm\n"
    "• High Vector Activity"
)


text_x = 0.95  
text_y = 0.95  


optimal_box = ax.text(
    12, 220, optimal_text,  
    fontsize=10,
    bbox=dict(facecolor='white', edgecolor='#FF4500', boxstyle='round', alpha=0.8),
    verticalalignment='top'
)


ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Precipitation (mm)')
ax.set_title('Species Presence Points Under Optimal Conditions\n(Mosquito and Bird ENM Threshold ≥0.7)')
ax.legend(loc='upper left')
plt.colorbar(risk_plot, label='Transmission Risk Index')
plt.tight_layout()
plt.show()

np.save("risk_index.npy", risk_index)
