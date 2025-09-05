import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


print("Loading and preparing data...")
df = pd.read_csv('all data.csv', sep=';')


print("\nAvailable columns:", df.columns.tolist())


print("\nConverting precipitation from inches to millimeters...")
print("Before conversion - Precipitation range:", 
      f"Min: {df['Total Monthly Precipitation'].min():.2f} inches, ",
      f"Max: {df['Total Monthly Precipitation'].max():.2f} inches")

df['Total Monthly Precipitation'] = df['Total Monthly Precipitation'] * 25.4  

print("After conversion - Precipitation range:", 
      f"Min: {df['Total Monthly Precipitation'].min():.2f} mm, ",
      f"Max: {df['Total Monthly Precipitation'].max():.2f} mm")


df['Temp_Range'] = df['Monthly Maximum Temperature'] - df['Monthly Minimum Temperature']


df['Precip_Quantile'] = pd.qcut(df['Total Monthly Precipitation'], q=5, 
                               labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])


temp_stats = df['Monthly Maximum Temperature'].describe()
min_temp_stats = df['Monthly Minimum Temperature'].describe()
precip_stats = df['Total Monthly Precipitation'].describe()
temp_range_stats = df['Temp_Range'].describe()


precip_threshold = df['Total Monthly Precipitation'].quantile(0.95)
high_precip_count = len(df[df['Total Monthly Precipitation'] > precip_threshold])
total_count = len(df)


plt.figure(figsize=(15, 10))


plt.subplot(2, 2, 1)
sns.histplot(data=df, x='Monthly Maximum Temperature', bins=20, color='firebrick')
plt.axvline(temp_stats['mean'], color='r', linestyle='--', label=f"Mean: {temp_stats['mean']:.1f}°C")
plt.axvline(temp_stats['50%'], color='g', linestyle='--', label=f"Median: {temp_stats['50%']:.1f}°C")
plt.title('Distribution of Maximum Temperature\nin WNV Cases')
plt.xlabel('Maximum Temperature (°C)')
plt.ylabel('Number of Cases')
plt.legend()


plt.subplot(2, 2, 2)
sns.histplot(data=df, x='Monthly Minimum Temperature', bins=20, color='tomato')
plt.axvline(min_temp_stats['mean'], color='r', linestyle='--', label=f"Mean: {min_temp_stats['mean']:.1f}°C")
plt.axvline(min_temp_stats['50%'], color='g', linestyle='--', label=f"Median: {min_temp_stats['50%']:.1f}°C")
plt.title('Distribution of Minimum Temperature\nin WNV Cases')
plt.xlabel('Minimum Temperature (°C)')
plt.ylabel('Number of Cases')
plt.legend()


plt.subplot(2, 2, 3)


sns.histplot(data=df[df['Total Monthly Precipitation'] <= precip_threshold], 
            x='Total Monthly Precipitation', bins=20, color='lightblue')

plt.axvline(precip_stats['mean'], color='r', linestyle='--', 
            label=f"Mean: {precip_stats['mean']:.1f}mm")
plt.axvline(precip_stats['50%'], color='g', linestyle='--', 
            label=f"Median: {precip_stats['50%']:.1f}mm")


plt.text(0.98, 0.95, 
         f"Note: {high_precip_count} cases ({(high_precip_count/total_count*100):.1f}%)\n"
         f"with precipitation > {precip_threshold:.1f}mm",
         transform=plt.gca().transAxes, 
         horizontalalignment='right',
         verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8))

plt.title('Distribution of Precipitation in WNV Cases\n(95th percentile threshold)')
plt.xlabel('Precipitation (mm)')
plt.ylabel('Number of Cases')
plt.legend()


plt.subplot(2, 2, 4)
sns.histplot(data=df, x='Temp_Range', bins=20, color='orangered')
plt.axvline(temp_range_stats['mean'], color='r', linestyle='--', label=f"Mean: {temp_range_stats['mean']:.1f}°C")
plt.axvline(temp_range_stats['50%'], color='g', linestyle='--', label=f"Median: {temp_range_stats['50%']:.1f}°C")
plt.title('Distribution of Temperature Range\nin WNV Cases')
plt.xlabel('Temperature Range (°C)')
plt.ylabel('Number of Cases')
plt.legend()

plt.tight_layout()
plt.savefig('wnv_environmental_conditions.png', dpi=300, bbox_inches='tight')
plt.close()


with open('environmental_conditions_summary.txt', 'w') as f:
    f.write("Environmental Conditions in WNV Cases\n")
    f.write("==================================\n\n")
    
    f.write("Maximum Temperature Statistics (°C):\n")
    f.write("---------------------------------\n")
    for stat, value in temp_stats.items():
        f.write(f"{stat}: {value:.2f}\n")
    f.write("\n")
    
    f.write("Minimum Temperature Statistics (°C):\n")
    f.write("---------------------------------\n")
    for stat, value in min_temp_stats.items():
        f.write(f"{stat}: {value:.2f}\n")
    f.write("\n")
    
    f.write("Precipitation Statistics (mm):\n")
    f.write("---------------------------\n")
    for stat, value in precip_stats.items():
        f.write(f"{stat}: {value:.2f}\n")
    f.write("\nNote: Precipitation values converted from inches to millimeters (1 inch = 25.4 mm)\n")
    f.write(f"High precipitation cases (>{precip_threshold:.1f}mm): {high_precip_count} ({(high_precip_count/total_count*100):.1f}%)\n\n")
    
    f.write("Temperature Range Statistics (°C):\n")
    f.write("-------------------------------\n")
    for stat, value in temp_range_stats.items():
        f.write(f"{stat}: {value:.2f}\n")


print("\nKey Environmental Conditions for WNV Cases:")
print("\nTemperature Conditions:")
print(f"Maximum Temperature: {temp_stats['mean']:.1f}°C (±{temp_stats['std']:.1f}°C)")
print(f"Minimum Temperature: {min_temp_stats['mean']:.1f}°C (±{min_temp_stats['std']:.1f}°C)")
print(f"Temperature Range: {temp_range_stats['mean']:.1f}°C (±{temp_range_stats['std']:.1f}°C)")
print("\nPrecipitation Conditions:")
print(f"Average: {precip_stats['mean']:.1f}mm (±{precip_stats['std']:.1f}mm)")
print(f"Range: {precip_stats['min']:.1f}mm to {precip_stats['max']:.1f}mm")
print(f"High precipitation cases (>{precip_threshold:.1f}mm): {high_precip_count} ({(high_precip_count/total_count*100):.1f}%)") 
