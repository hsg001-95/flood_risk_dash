import pandas as pd
import dash
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
try:
    import dash_bootstrap_components as dbc
except Exception:
    dbc = None
import numpy as np
import re


try:
    df = pd.read_csv(r'C:\Users\hiyan\OneDrive\Desktop\Projects\deepdata\flood_risk_dataset.csv')
except FileNotFoundError:
    print("Warning: Data file not found. Using dummy data for demonstration.")
    data = {
        'country_or_city': np.random.choice(['India (Chennai)', 'Singapore', 'Netherlands (Rotterdam)'], 2963),
        'latitude': np.random.uniform(1, 60, 2963),
        'longitude': np.random.uniform(60, 100, 2963),
        'elevation_m': np.random.uniform(-5, 50, 2963),
        'historical_rainfall_intensity_mm_hr': np.random.uniform(10, 150, 2963),
        'drainage_density_km_per_km2': np.random.uniform(0.1, 5, 2963),
        'land_use': np.random.choice(['Residential', 'Commercial', 'Green Space'], 2963),
        'risk_labels': np.random.choice(['low_risk', 'medium_risk', 'high_risk'], 2963, p=[0.5, 0.3, 0.2]),
        'risk_score': np.random.randint(1, 4, 2963)
    }
    df = pd.DataFrame(data)
    df['hazard_index'] = df['elevation_m'] * df['historical_rainfall_intensity_mm_hr']

COLOR_MAP = {
    'low_risk': '#2ecc71',
    'medium_risk': '#4dd0e1',
    'high_risk': '#1e90ff',
    'low_lying_flooding': '#6f42c1'
}
SHORT_LABEL_MAP = {
    'low_risk': 'Low',
    'medium_risk': 'Medium',
    'high_risk': 'High',
    'low_lying_flooding': 'Low-Lying'
}
COLOR_MAP_SHORT = {
    'Ponding': '#ff7f0e',
    'Low-Lying': '#2ecc71',
    'Extreme Rain': '#d62728',
    'Sparse Drainage': '#9467bd',
    'Monitor': '#1f77b4',
    'Other': '#7f7f7f',
    'Unknown': '#bdbdbd'
}

if 'risk_labels' not in df.columns:
    df['risk_labels'] = pd.NA

def derive_short_label(val):
    if pd.isna(val):
        return 'Unknown'
    s = str(val).lower()
    if 'ponding' in s:
        return 'Ponding'
    if 'low_lying' in s or 'low-lying' in s or 'low lying' in s or 'low_lying' in s:
        return 'Low-Lying'
    if 'extreme_rain' in s or ('extreme' in s and 'rain' in s):
        return 'Extreme Rain'
    if 'sparse_drainage' in s or 'sparse' in s and 'drain' in s:
        return 'Sparse Drainage'
    if 'monitor' in s:
        return 'Monitor'
    if 'low_risk' in s:
        return 'Low'
    if 'medium_risk' in s:
        return 'Medium'
    if 'high_risk' in s:
        return 'High'
    parts = re.split(r"[|]", s)
    for p in parts:
        p = p.strip()
        if 'pond' in p:
            return 'Ponding'
        if 'low' in p and 'lying' in p:
            return 'Low-Lying'
        if 'extreme' in p:
            return 'Extreme Rain'
        if 'sparse' in p:
            return 'Sparse Drainage'
        if 'monitor' in p:
            return 'Monitor'
    return 'Other'

df['risk_label_short'] = df['risk_labels'].apply(derive_short_label)
df['color'] = df['risk_label_short'].map(COLOR_MAP_SHORT).fillna('#7f7f7f')

if 'country_or_city' not in df.columns:
    if 'city_name' in df.columns:
        df['country_or_city'] = df['city_name']
    else:
        df['country_or_city'] = 'Unknown'

locations = sorted(df['country_or_city'].unique())
if 'Global' not in locations:
    locations.insert(0, 'Global')

external_styles = [dbc.themes.BOOTSTRAP] if dbc is not None else []
app = dash.Dash(__name__, external_stylesheets=external_styles,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

CARD_STYLE = {
    "box-shadow": "0 6px 18px rgba(0, 0, 0, 0.08)",
    "padding": "18px",
    "margin-bottom": "20px",
    "border-radius": "12px",
    "background-color": "#ffffff",
    "border": "1px solid #e6eef3"
}

if dbc is not None:
    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1("Urban Flood Resilience Nexus", className="text-center my-4 text-primary"))),
        dbc.Row([
            dbc.Col(dbc.Card([
                html.H3("Select Location for Drill-Down", className="card-title text-dark"),
                dcc.Dropdown(
                    id='location-dropdown',
                    options=[{'label': i, 'value': i} for i in locations],
                    value='Global' if 'Global' in locations else (locations[0] if len(locations) > 0 else None),
                    placeholder="Select a Country or City",
                    style={'color': '#111', 'font-size': '16px'}
                )
            ], body=True, style=CARD_STYLE), md=12),
        ]),

        dbc.Row([
            dbc.Col(
                dcc.Graph(
                    id='main-map-display',
                    config={'displayModeBar': False},
                    style={'height': '75vh', 'width': '100%', 'border-radius': '10px', 'margin': '0'}
                ),
                md=8
            ),
            dbc.Col([
                dbc.Card(
                    dcc.Graph(id='risk-distribution-bar', config={'displayModeBar': False}),
                    body=True, style=CARD_STYLE
                ),
                dbc.Card(
                    dcc.Graph(id='elevation-rainfall-scatter', config={'displayModeBar': False}),
                    body=True, style=CARD_STYLE
                ),
            ], md=4),
        ], align='top'),

        html.Hr(style={'border-color': '#ddd'}),

        dbc.Row(dbc.Col(html.H2(id='detail-title', children='Click a Point on the Map for Micro-Analysis', className='text-center my-4 text-info'))),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dcc.Graph(id='point-analysis-bar', config={'displayModeBar': False}),
                    body=True, style=CARD_STYLE
                ),
                md=6
            ),
            dbc.Col(
                dbc.Card([
                    html.H4("Place Brief", className="text-dark"),
                    html.Div(id='policy-text', children='Select a data point to view details and recommendations.', style={'color': '#111', 'whiteSpace': 'pre-wrap', 'font-size': '14px'}),
                ], body=True, style=CARD_STYLE),
                md=6
            ),
        ]),

    ], fluid=True, className="bg-light", style={'minHeight': '100vh', 'padding': '18px'})
else:
    app.layout = html.Div(children=[
        html.H1("Urban Flood Resilience Nexus"),
        html.P("Dash Bootstrap Components (dbc) not installed; basic layout shown.")
    ], style={'font-family': 'Arial, sans-serif', 'color': '#111'})

@app.callback(
    Output('main-map-display', 'figure'),
    Output('risk-distribution-bar', 'figure'),
    Output('elevation-rainfall-scatter', 'figure'),
    Input('location-dropdown', 'value')
)
def update_country_views(selected_location):
    if selected_location is None or selected_location == 'Global':
        filtered_df = df.copy()
        is_global = True
    else:
        filtered_df = df[df['country_or_city'] == selected_location]
        is_global = False

    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f'No data for {selected_location}', paper_bgcolor='#212529', plot_bgcolor='#212529', font_color='white')
        return empty_fig, empty_fig, empty_fig

    fig_map = px.scatter_geo(
        filtered_df,
        lat='latitude',
        lon='longitude',
        color='risk_label_short',
        hover_name='country_or_city',
        custom_data=['country_or_city'],
        hover_data={'elevation_m': ':.1f', 'historical_rainfall_intensity_mm_hr': ':.1f', 'risk_label_short': True},
        color_discrete_map=COLOR_MAP_SHORT,
        projection='natural earth',
        title=("Global Flood Risk Hotspots" if is_global else f"Flood Risk Hotspots in {selected_location}"),
        height=600
    )
    fig_map.update_traces(marker=dict(size=16, opacity=0.95, line=dict(width=0.6, color='white')))

    fig_map.update_layout(
        margin={"r":0,"t":56,"l":0,"b":0},
        title=dict(font=dict(size=24)),
        geo=dict(
            scope='world',
            showcountries=True,
            showland=True,
            landcolor='#f0fbf9',
            bgcolor='#eaf6f3',
            lakecolor='#eaf6f3'
        ),
        paper_bgcolor='#f7fcfb',
        plot_bgcolor='#f7fcfb',
        font=dict(color='#111', size=16),
    )

    if not is_global and not filtered_df.empty:
        lon_min = float(filtered_df['longitude'].min())
        lon_max = float(filtered_df['longitude'].max())
        lat_min = float(filtered_df['latitude'].min())
        lat_max = float(filtered_df['latitude'].max())
        lon_pad = (lon_max - lon_min) * 0.12 if lon_max > lon_min else 0.5
        lat_pad = (lat_max - lat_min) * 0.12 if lat_max > lat_min else 0.5
        fig_map.update_geos(lonaxis=dict(range=[lon_min - lon_pad, lon_max + lon_pad]), lataxis=dict(range=[lat_min - lat_pad, lat_max + lat_pad]))

    risk_counts = filtered_df['risk_label_short'].value_counts().reindex(list(COLOR_MAP_SHORT.keys()), fill_value=0).reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    bar_height = 420 if is_global else 300
    fig_bar = px.bar(
        risk_counts,
        x='Risk Level',
        y='Count',
        color='Risk Level',
        title=f"Risk Distribution in {selected_location}",
        color_discrete_map=COLOR_MAP_SHORT,
        height=bar_height
    )
    fig_bar.update_layout(xaxis_title=None, font=dict(size=18 if is_global else 16))
    fig_bar.update_layout(
        paper_bgcolor='#f7fcfb', 
        plot_bgcolor='#f7fcfb',
        font=dict(color='#111', size=16),
        xaxis_title=None
    )
    fig_bar.update_traces(marker_line_width=0)

    fig_scatter = px.scatter(
        filtered_df,
        x='elevation_m',
        y='historical_rainfall_intensity_mm_hr',
        color='risk_label_short',
        color_discrete_map=COLOR_MAP_SHORT,
        title='Elevation vs. Rainfall Intensity',
        hover_name='land_use',
    )
    fig_scatter.update_layout(
        paper_bgcolor='#f7fcfb', 
        plot_bgcolor='#f7fcfb',
        font=dict(color='#111', size=16),
        xaxis_title='Elevation (m)',
        yaxis_title='Rainfall Intensity (mm/hr)'
    )

    return fig_map, fig_bar, fig_scatter


@app.callback(
    Output('detail-title', 'children'),
    Output('point-analysis-bar', 'figure'),
    Output('policy-text', 'children'),
    Input('main-map-display', 'clickData')
)
def update_detail_view(clickData):
    if clickData is None:
        empty_fig = go.Figure().update_layout(
            paper_bgcolor='#212529', plot_bgcolor='#212529', font_color='white',
            xaxis={'visible': False}, yaxis={'visible': False}
        )
        return 'Click a Point on the Map for Micro-Analysis', empty_fig, 'Select a data point to view details and recommendations.'

    point = clickData['points'][0]
    clicked_country = None
    if 'customdata' in point and point['customdata']:
        cd = point['customdata']
        if isinstance(cd, (list, tuple)):
            clicked_country = cd[0]
        else:
            clicked_country = cd
    point_index = point.get('pointIndex')
    if point_index is not None:
        clicked_data = df.iloc[point_index]
    else:
        lat_val = point.get('lat') or point.get('y')
        lon_val = point.get('lon') or point.get('x')
        clicked_data = df[(df['latitude'] == lat_val) & (df['longitude'] == lon_val)].iloc[0]

    lat = clicked_data['latitude']
    lon = clicked_data['longitude']
    risk = clicked_data['risk_labels']
    elev = clicked_data['elevation_m']
    rain = clicked_data['historical_rainfall_intensity_mm_hr']
    drain = clicked_data['drainage_density_km_per_km2']
    land = clicked_data['land_use']

    metrics = {
        'Elevation (m)': elev,
        'Rainfall (mm/hr)': rain,
        'Drainage Density': drain,
    }

    fig_detail = px.bar(
        x=list(metrics.keys()), 
        y=list(metrics.values()), 
        title=f"Key Risk Drivers at Lat: {lat:.2f}, Lon: {lon:.2f}",
        color=list(metrics.keys()),
        color_discrete_sequence=['#17a2b8', '#ffc107', '#dc3545']
    )
    fig_detail.update_layout(
        paper_bgcolor='#212529', plot_bgcolor='#212529', font_color='white',
        xaxis_title=None, yaxis_title='Value',
        showlegend=False
    )

    neigh = df[(df['latitude'] >= lat - 0.05) & (df['latitude'] <= lat + 0.05) & (df['longitude'] >= lon - 0.05) & (df['longitude'] <= lon + 0.05)]
    neigh_count = len(neigh)
    pct_high = (neigh['risk_labels'] == 'high_risk').mean() * 100 if neigh_count > 0 else 0
    avg_elev = neigh['elevation_m'].mean() if neigh_count > 0 else elev
    avg_rain = neigh['historical_rainfall_intensity_mm_hr'].mean() if neigh_count > 0 else rain
    predominant_land = neigh['land_use'].mode().iloc[0] if (neigh_count > 0 and not neigh['land_use'].mode().empty) else land

    brief_lines = [
        f"**Location:** {lat:.4f}, {lon:.4f}",
        f"**Risk Level:** {str(risk)}",
        f"**Elevation:** {elev:.1f} m (avg nearby: {avg_elev:.1f} m)",
        f"**Rainfall intensity:** {rain:.1f} mm/hr (avg nearby: {avg_rain:.1f})",
        f"**Nearby segments:** {neigh_count} (high-risk: {pct_high:.1f}% )",
        f"**Predominant land use nearby:** {predominant_land}",
    ]
    place_brief = "\n\n".join(brief_lines)

    return f"Micro-Analysis for Segment (Lat: {lat:.2f}, Lon: {lon:.2f})", fig_detail, html.Div(place_brief, style={'color': '#111', 'whiteSpace': 'pre-wrap'})


@app.callback(
    Output('location-dropdown', 'value'),
    Input('main-map-display', 'clickData'),
    prevent_initial_call=True
)
def sync_dropdown_on_click(clickData):
    if not clickData or 'points' not in clickData:
        raise PreventUpdate
    pt = clickData['points'][0]
    country = None
    if 'customdata' in pt and pt['customdata']:
        cd = pt['customdata']
        country = cd[0] if isinstance(cd, (list, tuple)) else cd
    elif 'hovertext' in pt:
        country = pt['hovertext']
    if country:
        return country
    raise PreventUpdate


if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
import dash
from dash import dcc, html, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
# dash_bootstrap_components may not be installed in all environments; import gracefully
try:
    import dash_bootstrap_components as dbc
except Exception:
    dbc = None
import numpy as np


try:
    df = pd.read_csv(r'C:\Users\hiyan\OneDrive\Desktop\Projects\deepdata\flood_risk_dataset.csv')
except FileNotFoundError:
    # Creating a dummy DataFrame for demonstration if the file is not found
    print("Warning: Data file not found. Using dummy data for demonstration.")
    data = {
        'country_or_city': np.random.choice(['India (Chennai)', 'Singapore', 'Netherlands (Rotterdam)'], 2963),
        'latitude': np.random.uniform(1, 60, 2963),
        'longitude': np.random.uniform(60, 100, 2963),
        'elevation_m': np.random.uniform(-5, 50, 2963),
        'historical_rainfall_intensity_mm_hr': np.random.uniform(10, 150, 2963),
        'drainage_density_km_per_km2': np.random.uniform(0.1, 5, 2963),
        'land_use': np.random.choice(['Residential', 'Commercial', 'Green Space'], 2963),
        'risk_labels': np.random.choice(['low_risk', 'medium_risk', 'high_risk'], 2963, p=[0.5, 0.3, 0.2]),
        'risk_score': np.random.randint(1, 4, 2963)
    }
    df = pd.DataFrame(data)
    # Ensure all columns needed for the app are present
    df['hazard_index'] = df['elevation_m'] * df['historical_rainfall_intensity_mm_hr']

# Map risk labels to colors for consistency and visualization
COLOR_MAP = {
    'low_risk': '#2ecc71',   # sea green
    'medium_risk': '#4dd0e1', # cyan/light blue
    'high_risk': '#1e90ff',   # dodger blue
    'low_lying_flooding': '#6f42c1' # purple as accent
}
# Short, human-friendly labels and colors for UI
SHORT_LABEL_MAP = {
    'low_risk': 'Low',
    'medium_risk': 'Medium',
    'high_risk': 'High',
    'low_lying_flooding': 'Low-Lying'
}
COLOR_MAP_SHORT = {
    'Low': '#2ecc71',
    'Medium': '#4dd0e1',
    'High': '#1e90ff',
    'Low-Lying': '#6f42c1'
}
# Ensure risk_labels exists
if 'risk_labels' not in df.columns:
    df['risk_labels'] = pd.NA

# Create short human-friendly risk labels for plotting and UI
df['risk_label_short'] = df['risk_labels'].map(SHORT_LABEL_MAP).fillna('Unknown')
df['color'] = df['risk_label_short'].map(COLOR_MAP_SHORT).fillna('gray')

# Get unique locations for the dropdown
if 'country_or_city' not in df.columns:
    if 'city_name' in df.columns:
        df['country_or_city'] = df['city_name']
    else:
        df['country_or_city'] = 'Unknown'

locations = sorted(df['country_or_city'].unique())
# Ensure 'Global' is available as the first option and default selection
if 'Global' not in locations:
    locations.insert(0, 'Global')

# --- 2. LAYOUT DEFINITION ---

# Initialize the Dash app with a Bootstrap theme if available
# Switch to a lighter, greener Bootstrap theme when available for a bright UI
external_styles = [dbc.themes.BOOTSTRAP] if dbc is not None else []
app = dash.Dash(__name__, external_stylesheets=external_styles,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

# Custom styles for attractive presentation
CARD_STYLE = {
    "box-shadow": "0 6px 18px rgba(0, 0, 0, 0.08)",
    "padding": "18px",
    "margin-bottom": "20px",
    "border-radius": "12px",
    "background-color": "#ffffff",
    "border": "1px solid #e6eef3"
}

# Build layout using dbc if available, otherwise provide a minimal fallback
if dbc is not None:
    app.layout = dbc.Container([
        # Title Row
        dbc.Row(dbc.Col(html.H1("Urban Flood Resilience Nexus", className="text-center my-4 text-primary"))),
        
        # Dropdown and Map Row (Global/Country View)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H3("Select Location for Drill-Down", className="card-title text-light"),
                    dcc.Dropdown(
                        id='location-dropdown',
                        options=[{'label': i, 'value': i} for i in locations],
                        value='Global' if 'Global' in locations else (locations[0] if len(locations) > 0 else None), 
                        placeholder="Select a Country or City",
                        style={'color': '#111', 'font-size': '16px'}
                    )
                ], body=True, style=CARD_STYLE),
            ], md=12),
        ]),
        
        dbc.Row([
            # Main Map Column (Left/Center)
            dbc.Col(
                dcc.Graph(
                    id='main-map-display',
                    config={'displayModeBar': False},
                    style={'height': '70vh', 'border-radius': '10px'}
                ), 
                md=8
            ),
            
            # Global/Country Graphs Column (Right Side)
            dbc.Col([
                dbc.Card(
                    dcc.Graph(id='risk-distribution-bar', config={'displayModeBar': False}), 
                    body=True, style=CARD_STYLE
                ),
                dbc.Card(
                    dcc.Graph(id='elevation-rainfall-scatter', config={'displayModeBar': False}), 
                    body=True, style=CARD_STYLE
                ),
            ], md=4),
        ]),
        
        html.Hr(style={'border-color': '#555'}),

        # --- Section 3: Detailed Point Analysis (Hidden until click) ---
        dbc.Row(dbc.Col(html.H2(id='detail-title', children='Click a Point on the Map for Micro-Analysis', className='text-center my-4 text-info'))),

        dbc.Row([
            # Detailed Bar Chart for Single Point
            dbc.Col(
                dbc.Card(
                    dcc.Graph(id='point-analysis-bar', config={'displayModeBar': False}),
                    body=True, style=CARD_STYLE
                ),
                md=6
            ),
            # Actionable Recommendation Text
            dbc.Col(
                    dbc.Card([
                    html.H4("Place Brief", className="text-warning"),
                    # Use a Div with dark text so the brief is visible on the light card background
                    html.Div(id='policy-text', children='Select a data point to view details and recommendations.', style={'color': '#111', 'whiteSpace': 'pre-wrap'}),
                ], body=True, style=CARD_STYLE),
                md=6
            ),
       ]),

    ], fluid=True, className="bg-light")
else:
    app.layout = html.Div(children=[
        html.H1("Urban Flood Resilience Nexus"),
        html.P("Dash Bootstrap Components (dbc) not installed; basic layout shown.")
    ], style={'font-family': 'Arial, sans-serif', 'color': '#111'})

# --- 3. CALLBACKS (INTERACTIVITY) ---

# Callback 1: Updates the Main Map, Bar Chart, and Scatter Plot based on Dropdown
@app.callback(
    Output('main-map-display', 'figure'),
    Output('risk-distribution-bar', 'figure'),
    Output('elevation-rainfall-scatter', 'figure'),
    Input('location-dropdown', 'value')
)
def update_country_views(selected_location):
    # --- Filter Data ---
    # Support a 'Global' option which shows all points and global aggregates
    if selected_location is None or selected_location == 'Global':
        filtered_df = df.copy()
        is_global = True
    else:
        filtered_df = df[df['country_or_city'] == selected_location]
        is_global = False

    # If filtered_df empty, return empty placeholder figures
    if filtered_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f'No data for {selected_location}', paper_bgcolor='#212529', plot_bgcolor='#212529', font_color='white')
        return empty_fig, empty_fig, empty_fig

    # --- 1. Map Generation (Country Map or Global view) ---
    # include country_or_city as custom data so clicks can set dropdown
    # Use short labels and slightly larger markers for better visibility
    fig_map = px.scatter_geo(
        filtered_df,
        lat='latitude',
        lon='longitude',
        color='risk_label_short',
        hover_name='country_or_city',
        custom_data=['country_or_city'],
        hover_data={'elevation_m': ':.1f', 'historical_rainfall_intensity_mm_hr': ':.1f', 'risk_label_short': True},
        color_discrete_map=COLOR_MAP_SHORT,
        projection='natural earth',
        title=("Global Flood Risk Hotspots" if is_global else f"Flood Risk Hotspots in {selected_location}"),
        height=600
    )
    # Increase marker size for improved visibility
    fig_map.update_traces(marker=dict(size=6, line_width=0))

    # Layout styling: lighter, colorful theme with dark text
    fig_map.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        geo=dict(
            scope='world',
            showcountries=True,
            showland=True,
            landcolor='#f0fbf9',
            bgcolor='#eaf6f3',
            lakecolor='#eaf6f3'
        ),
        paper_bgcolor='#f7fcfb',
        plot_bgcolor='#f7fcfb',
        font_color='#111',
    )

    # If not global, zoom to bounds for the selected location
    if not is_global and not filtered_df.empty:
        lon_min = float(filtered_df['longitude'].min())
        lon_max = float(filtered_df['longitude'].max())
        lat_min = float(filtered_df['latitude'].min())
        lat_max = float(filtered_df['latitude'].max())
        lon_pad = (lon_max - lon_min) * 0.12 if lon_max > lon_min else 0.5
        lat_pad = (lat_max - lat_min) * 0.12 if lat_max > lat_min else 0.5
        fig_map.update_geos(lonaxis=dict(range=[lon_min - lon_pad, lon_max + lon_pad]), lataxis=dict(range=[lat_min - lat_pad, lat_max + lat_pad]))

    # --- 2. Risk Distribution Bar Chart ---
    # Build risk counts using short labels
    risk_counts = filtered_df['risk_label_short'].value_counts().reindex(list(COLOR_MAP_SHORT.keys()), fill_value=0).reset_index()
    risk_counts.columns = ['Risk Level', 'Count']
    # Make the bar chart larger when viewing Global
    bar_height = 420 if is_global else 300
    fig_bar = px.bar(
        risk_counts,
        x='Risk Level',
        y='Count',
        color='Risk Level',
        title=f"Risk Distribution in {selected_location}",
        color_discrete_map=COLOR_MAP_SHORT,
        height=bar_height
    )
    # Emphasize short labels and readability
    fig_bar.update_layout(xaxis_title=None, font=dict(size=14 if is_global else 12))
    fig_bar.update_layout(
        paper_bgcolor='#f7fcfb', 
        plot_bgcolor='#f7fcfb',
        font_color='#111',
        xaxis_title=None
    )
    fig_bar.update_traces(marker_line_width=0)
    
    # --- 3. Elevation vs. Rainfall Scatter Plot ---
    fig_scatter = px.scatter(
        filtered_df,
        x='elevation_m',
        y='historical_rainfall_intensity_mm_hr',
        color='risk_label_short',
        color_discrete_map=COLOR_MAP_SHORT,
        title='Elevation vs. Rainfall Intensity',
        hover_name='land_use',
    )
    fig_scatter.update_layout(
        paper_bgcolor='#f7fcfb', 
        plot_bgcolor='#f7fcfb',
        font_color='#111',
        xaxis_title='Elevation (m)',
        yaxis_title='Rainfall Intensity (mm/hr)'
    )
    
    return fig_map, fig_bar, fig_scatter

# Callback 2: Updates Detail Bar Chart and Policy Text on Map Click
@app.callback(
    Output('detail-title', 'children'),
    Output('point-analysis-bar', 'figure'),
    Output('policy-text', 'children'),
    Input('main-map-display', 'clickData')
)
def update_detail_view(clickData):
    if clickData is None:
        # Initial state or no click
        empty_fig = go.Figure().update_layout(
            paper_bgcolor='#212529', plot_bgcolor='#212529', font_color='white',
            xaxis={'visible': False}, yaxis={'visible': False}
        )
        return 'Click a Point on the Map for Micro-Analysis', empty_fig, 'Select a data point to view details and recommendations.'

    # Get data for the clicked point
    point = clickData['points'][0]
    # if the map includes customdata (country_or_city), we can use it to set focus
    clicked_country = None
    if 'customdata' in point and point['customdata']:
        cd = point['customdata']
        if isinstance(cd, (list, tuple)):
            clicked_country = cd[0]
        else:
            clicked_country = cd
    # pointIndex can be used to index the dataframe directly
    point_index = point.get('pointIndex')
    if point_index is not None:
        clicked_data = df.iloc[point_index]
    else:
        # fallback: try to lookup by lat/lon
        lat_val = point.get('lat') or point.get('y')
        lon_val = point.get('lon') or point.get('x')
        clicked_data = df[(df['latitude'] == lat_val) & (df['longitude'] == lon_val)].iloc[0]
    
    lat = clicked_data['latitude']
    lon = clicked_data['longitude']
    risk = clicked_data['risk_labels']
    elev = clicked_data['elevation_m']
    rain = clicked_data['historical_rainfall_intensity_mm_hr']
    drain = clicked_data['drainage_density_km_per_km2']
    land = clicked_data['land_use']
    
    # --- Detail Bar Chart ---
    metrics = {
        'Elevation (m)': elev,
        'Rainfall (mm/hr)': rain,
        'Drainage Density': drain,
    }
    
    fig_detail = px.bar(
        x=list(metrics.keys()), 
        y=list(metrics.values()), 
        title=f"Key Risk Drivers at Lat: {lat:.2f}, Lon: {lon:.2f}",
        color=list(metrics.keys()),
        color_discrete_sequence=['#17a2b8', '#ffc107', '#dc3545'] # Info, Warning, Danger colors
    )
    fig_detail.update_layout(
        paper_bgcolor='#212529', plot_bgcolor='#212529', font_color='white',
        xaxis_title=None, yaxis_title='Value',
        showlegend=False
    )

    # --- Place Brief and neighborhood stats ---
    # Neighborhood: compute stats for other points within a small radius (approx bounding box)
    neigh = df[(df['latitude'] >= lat - 0.05) & (df['latitude'] <= lat + 0.05) & (df['longitude'] >= lon - 0.05) & (df['longitude'] <= lon + 0.05)]
    neigh_count = len(neigh)
    pct_high = (neigh['risk_labels'] == 'high_risk').mean() * 100 if neigh_count > 0 else 0
    avg_elev = neigh['elevation_m'].mean() if neigh_count > 0 else elev
    avg_rain = neigh['historical_rainfall_intensity_mm_hr'].mean() if neigh_count > 0 else rain
    predominant_land = neigh['land_use'].mode().iloc[0] if (neigh_count > 0 and not neigh['land_use'].mode().empty) else land

    brief_lines = [
        f"**Location:** {lat:.4f}, {lon:.4f}",
        f"**Risk Level:** {str(risk)}",
        f"**Elevation:** {elev:.1f} m (avg nearby: {avg_elev:.1f} m)",
        f"**Rainfall intensity:** {rain:.1f} mm/hr (avg nearby: {avg_rain:.1f})",
        f"**Nearby segments:** {neigh_count} (high-risk: {pct_high:.1f}% )",
        f"**Predominant land use nearby:** {predominant_land}",
    ]
    place_brief = "\n\n".join(brief_lines)

    # Return the place brief wrapped in a Div with dark text to ensure readability
    return f"Micro-Analysis for Segment (Lat: {lat:.2f}, Lon: {lon:.2f})", fig_detail, html.Div(place_brief, style={'color': '#111', 'whiteSpace': 'pre-wrap'})
    

# New callback: clicking a point on the main map will update the dropdown selection so the app zooms to that country
@app.callback(
    Output('location-dropdown', 'value'),
    Input('main-map-display', 'clickData'),
    prevent_initial_call=True
)
def sync_dropdown_on_click(clickData):
    if not clickData or 'points' not in clickData:
        raise PreventUpdate
    pt = clickData['points'][0]
    country = None
    if 'customdata' in pt and pt['customdata']:
        cd = pt['customdata']
        country = cd[0] if isinstance(cd, (list, tuple)) else cd
    elif 'hovertext' in pt:
        country = pt['hovertext']
    if country:
        return country
    raise PreventUpdate


if __name__ == '__main__':
    # Setting the security check to False for single-user environment safety
    app.run(debug=True)



