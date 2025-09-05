import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


df = pd.read_csv('risk_parameters_20250505_143516.csv')


risk_levels = ['low', 'moderate', 'high', 'extreme']
vector_color = '#1f77b4'  
host_color = '#ff7f0e'    
both_color = '#FF2E2E'    


plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titleweight': 'bold',
    'axes.titlepad': 20
})


fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)


ax1 = fig.add_subplot(gs[0, 0])  
ax2 = fig.add_subplot(gs[0, 1])  


def plot_ranges(ax, variable):
    bar_height = 0.35
    spacing = 0.1
    
    for i, risk in enumerate(risk_levels):
        
        vector_data = df[(df['type'] == 'vector') & (df['risk_level'] == risk) & 
                        (df['variable'] == variable if variable == 'precip' else True)]
        if not vector_data.empty:
            ax.barh(i + spacing, 
                   vector_data['max_value'].values[0] - vector_data['min_value'].values[0],
                   left=vector_data['min_value'].values[0], 
                   color=vector_color, 
                   alpha=0.8,
                   height=bar_height)
        
        
        host_data = df[(df['type'] == 'host') & (df['risk_level'] == risk) & 
                      (df['variable'] == variable if variable == 'precip' else True)]
        if not host_data.empty:
            ax.barh(i - spacing, 
                   host_data['max_value'].values[0] - host_data['min_value'].values[0],
                   left=host_data['min_value'].values[0], 
                   color=host_color, 
                   alpha=0.8,
                   height=bar_height)
    
    
    both_data = df[(df['type'] == 'both') & 
                  (df['variable'] == variable if variable == 'precip' else True)]
    if not both_data.empty:
        ax.barh(len(risk_levels), 
               both_data['max_value'].values[0] - both_data['min_value'].values[0],
               left=both_data['min_value'].values[0], 
               color=both_color, 
               alpha=0.8,
               height=bar_height)


plot_ranges(ax1, 'temp')
plot_ranges(ax2, 'precip')


y_positions = np.arange(len(risk_levels) + 1)
y_labels = [r.capitalize() for r in risk_levels] + ['Optimal']


ax1.set_xlim(10, 40)  
ax1.set_xticks(np.arange(10, 41, 5))  


ax2.set_xlim(0, 100)  
ax2.set_xticks(np.arange(0, 101, 10))  

for ax, title, xlabel in zip([ax1, ax2], 
                            ['Temperature Ranges by Risk Level', 'Precipitation Ranges by Risk Level'],
                            ['Temperature (°C)', 'Precipitation (mm)']):
    ax.set_title(title, fontsize=14, y=0.95, pad=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    
    

s
legend_elements = [
    Patch(facecolor=vector_color, alpha=0.8, label='Vector'),
    Patch(facecolor=host_color, alpha=0.8, label='Host'),
    Patch(facecolor=both_color, alpha=0.8, label='Both')
]
fig.legend(handles=legend_elements, loc='upper center', 
          bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=14)


fig.suptitle('West Nile Virus Risk Assessment', 
            fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()