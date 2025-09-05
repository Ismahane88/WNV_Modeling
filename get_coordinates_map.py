import pandas as pd
import folium
import requests
import time
from branca.colormap import LinearColormap
from branca.element import Figure


try:
    df = pd.read_csv('data organization countries.csv', sep=';', encoding='utf-8')
except FileNotFoundError:
    print("Error: Could not find 'data organization countries.csv'")
    exit(1)
except Exception as e:
    print(f"Error reading the file: {str(e)}")
    exit(1)

def get_coordinates(location):
    try:
        
        time.sleep(1)
        
        location = location.replace('_', ' ').split(';')[0]
        
        
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location,
            'format': 'json',
            'limit': 1
        }
        
        
        headers = {
            'User-Agent': 'WNV_Analysis/1.0'
        }
        
        
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return (float(data[0]['lat']), float(data[0]['lon']))
        return None
    except Exception as e:
        print(f"Error with location {location}: {str(e)}")
        return None


unique_locations = df['Geo_Location'].unique()
location_coords = {}

print("Getting coordinates for locations...")
for loc in unique_locations:
    if pd.notna(loc):  
        coords = get_coordinates(loc)
        if coords:
            location_coords[loc] = coords
            print(f"Found coordinates for {loc}: {coords}")


if location_coords:
    all_lats = [coord[0] for coord in location_coords.values()]
    all_lons = [coord[1] for coord in location_coords.values()]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles='cartodbpositron'
    )

    
    location_counts = df['Geo_Location'].value_counts()
    
    
    max_cases = location_counts.max()
    min_cases = location_counts.min()
    
    colormap = LinearColormap(
        colors=['yellow', 'orange', 'red', 'darkred'],
        vmin=min_cases,
        vmax=max_cases,
        caption='Number of WNV Cases'
    )
    
    
    m.add_child(colormap)

    
    for loc, coords in location_coords.items():
        count = location_counts[loc]
        
        radius = max(8, min(25, 8 * (count / max_cases * 100) ** 0.5))
        
        
        color = colormap(count)
        
        
        location_data = df[df['Geo_Location'] == loc]
        hosts = ', '.join(location_data['Host'].dropna().unique())
        tissues = ', '.join(location_data['Tissue_Specimen_Source'].dropna().unique())
        
        popup_html = f"""
        <div style='min-width: 200px'>
            <h4>{loc}</h4>
            <b>Cases:</b> {count}<br>
            <b>Percentage:</b> {(count/sum(location_counts))*100:.1f}%<br>
            <b>Hosts:</b> {hosts}<br>
            <b>Tissues:</b> {tissues}
        </div>
        """
        
        folium.CircleMarker(
            location=coords,
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2
        ).add_to(m)

    
    title_html = '''
        <div style="position: fixed; 
                    top: 10px; 
                    left: 50px; 
                    width: 500px; 
                    height: 120px; 
                    z-index:9999; 
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    border: 2px solid gray;
                    font-size: 16px;
                    font-weight: bold;">
            <h4>West Nile Virus (WNV) Case Distribution</h4>
            <p style="font-size: 14px;">Circle size and color indicate number of cases<br>
            Click on circles to see detailed information including hosts and tissues</p>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    
    m.save('wnv_distribution_map.html')
    print("Map has been created as 'wnv_distribution_map.html'")

    
    coords_df = pd.DataFrame(
        [(loc, coord[0], coord[1], location_counts[loc]) 
         for loc, coord in location_coords.items()],
        columns=['Location', 'Latitude', 'Longitude', 'Cases']
    )
    coords_df.to_csv('location_coordinates.csv', index=False)
    print("Coordinates have been saved to 'location_coordinates.csv'")
else:
    print("No coordinates were found for any location.") 
    