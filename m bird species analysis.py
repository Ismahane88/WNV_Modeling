import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def visualize_species_percentage(file_path='birds.csv', host_col='Host', top_n=13):
    """Professional dot plot with complete minor species list and perfect label alignment"""
    try:
        # Load and validate
        df = pd.read_csv("birds.csv", sep=";")
        assert host_col in df.columns, f"Column '{host_col}' not found"
        
         # Calculate percentages
        total = len(df)
        species_counts = df[host_col].value_counts()
        species_pct = (species_counts / total) * 100
        
        # Prepare data
        top_species = species_pct.head(top_n)
        other_pct = species_pct[top_n:].sum()
        minor_species = species_counts[top_n:]
        
        if other_pct > 0:
            top_species = pd.concat([
                top_species,
                pd.Series({'Other species': other_pct})
            ])
        
        # Create plot data with enforced minimum display
        plot_data = pd.DataFrame({
            'Species': top_species.index,
            'True_Frequency': top_species.values,
            'Display_Frequency': [max(f, 3) for f in top_species.values]  # 3% minimum display
        })
        
        # Visualization
        plt.figure(figsize=(12, 4.5))
        ax = sns.stripplot(
            data=plot_data,
            x='Display_Frequency',
            y='Species',
            hue='Species',  # Added to fix palette warning
            size=18,
            palette="Greys_r",
            jitter=False,
            linewidth=1,
            edgecolor='grey',
            legend=False  # Added to fix palette warning
        )
        
        # Add connecting lines and labels
        for i, (disp_freq, true_freq) in enumerate(zip(plot_data['Display_Frequency'], plot_data['True_Frequency'])):
            plt.plot([0, true_freq], [i, i], 
                    color='#7e7882', 
                    linestyle=':', 
                    alpha=0.6,
                    linewidth=1.5)
          
            # Percentage labels
            ax.text(true_freq + 2.5, i, 
                   f'{true_freq:.1f}%', 
                   fontsize=7,
                   fontweight='bold',
                   color='black',
                   va='center',
                   ha='left')
        
        # Create italic font for species names
        italic_font = FontProperties()
        italic_font.set_style('italic')
        
        # Fix yticklabels warning
        ax.set_yticks(range(len(plot_data['Species'])))
        ax.set_yticklabels(
            [str(label) for label in plot_data['Species']],
            fontproperties=italic_font,
            va='center',
            ha='right',
            x=-0.01
        )
        
        # Add complete minor species list with proper italic formatting
        if len(minor_species) > 0:
            minor_text = "Minor Species:\n" + "\n".join(
                [f"• {s} ({species_pct[s]:.2f}%)" for s in minor_species.index])
            
            # Create annotation with italic font
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
            
            # Apply italic style to species names only
            annot.set_fontproperties(italic_font)
        
        # Formatting
        plt.title(f'Relative Frequency (%) of WNV-Positive Bird Species found in the Dataset\nTotal Samples: {total:,}', 
                pad=25, fontsize=15, fontweight='bold')
        plt.xlabel('Relative Frequency (%)', fontsize=12, labelpad=10)
        plt.ylabel('')
        plt.xlim(0, 100)
        
        # Adjust layout
        plt.subplots_adjust(right=0.75)  # Make space for minor species list
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")

# Execute analysis
visualize_species_percentage()