import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def visualize_species_percentage(file_path='mosquitoes.csv', host_col='Host', top_n=8):
    """Professional dot plot with complete minor species list and perfect label alignment"""
    try:
        
        df = pd.read_csv("mosquitoes.csv", sep=";")
        assert host_col in df.columns, f"Column '{host_col}' not found"
        
        
        total = len(df)
        species_counts = df[host_col].value_counts()
        species_pct = (species_counts / total) * 100
        
        
        top_species = species_pct.head(top_n)
        other_pct = species_pct[top_n:].sum()
        minor_species = species_counts[top_n:]
        
        if other_pct > 0:
            top_species = pd.concat([
                top_species,
                pd.Series({'Other species': other_pct})
            ])
        
        
        plot_data = pd.DataFrame({
            'Species': top_species.index,
            'True_Frequency': top_species.values,
            'Display_Frequency': [max(f, 3) for f in top_species.values]  
        })
        
       
        plt.figure(figsize=(12, 3.2))
        ax = sns.stripplot(
            data=plot_data,
            x='Display_Frequency',
            y='Species',
            hue='Species',  
            size=18,
            palette=["#000000", "#222222", "#444444", "#666666", "#888888", 
                    "#AAAAAA", "#C0C0C0", "#D0D0D0", "#E0E0E0"],
            jitter=False,
            linewidth=1,
            edgecolor='grey',
            legend=False  
        )
        
        
        for i, (disp_freq, true_freq) in enumerate(zip(plot_data['Display_Frequency'], plot_data['True_Frequency'])):
            plt.plot([0, true_freq], [i, i], 
                    color='#7e7882', 
                    linestyle=':', 
                    alpha=0.6,
                    linewidth=1.5)
          
            
            ax.text(true_freq + 2.5, i, 
                   f'{true_freq:.1f}%', 
                   fontsize=7,
                   fontweight='bold',
                   color='black',
                   va='center',
                   ha='left')
        
        
        italic_font = FontProperties()
        italic_font.set_style('italic')
        
        
        ax.set_yticks(range(len(plot_data['Species'])))
        ax.set_yticklabels(
            [str(label) for label in plot_data['Species']],
            fontproperties=italic_font,
            va='center',
            ha='right',
            x=-0.01
        )
        
        
        if len(minor_species) > 0:
            minor_text = "Minor Species:\n" + "\n".join(
                [f"• {s} ({species_pct[s]:.2f}%)" for s in minor_species.index])
            
            
            annot = ax.annotate(minor_text,
                        xy=(1.02, 0.5), 
                        xycoords='axes fraction',
                        ha='left',
                        va='center',
                        fontsize=9,
                        bbox=dict(boxstyle='round', 
                                facecolor='#f7f7f7', 
                                alpha=0.8,
                                pad=0.8))
            
            
            annot.set_fontproperties(italic_font)
        
        
        plt.title(f'Relative Frequency (%) of WNV-Positive Mosquito Species found in the Dataset\nTotal Samples: {total:,}', 
                pad=25, fontsize=15, fontweight='bold')
        plt.xlabel('Relative Frequency (%)', fontsize=12, labelpad=10)
        plt.ylabel('')
        plt.xlim(0, 100)
        
        
        plt.subplots_adjust(right=0.75)  
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")


visualize_species_percentage()