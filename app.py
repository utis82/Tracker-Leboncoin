import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import json
import os
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from scraper.leboncoin import LeboncoinScraper
from scraper.cleaning import DataCleaner

# Configuration
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
GEO_CACHE_PATH = os.path.join(CACHE_DIR, "geocode_cache.json")

# Cache de géocodage pour éviter les requêtes répétées
_geocode_cache = {}
if os.path.exists(GEO_CACHE_PATH):
    try:
        with open(GEO_CACHE_PATH, 'r', encoding='utf-8') as f:
            _geocode_cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        _geocode_cache = {}

geolocator = Nominatim(user_agent="moto-leboncoin-analyzer")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, swallow_exceptions=True)

# Initialisation de l'app Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Moto Leboncoin Analyzer"

# Styles CSS inline
COLORS = {
    'background': '#0f1419',
    'card': '#1a1f2e',
    'primary': '#1d9bf0',
    'text': '#e7e9ea',
    'secondary': '#71767b',
    'success': '#00ba7c',
    'border': '#2f3336'
}


def hidden_section_style():
    return {
        'maxWidth': '1400px',
        'margin': '0 auto 40px',
        'display': 'none'
    }


def visible_section_style():
    style = hidden_section_style()
    style['display'] = 'block'
    return style


def save_geocode_cache():
    """Sauvegarde le cache de géocodage sur disque."""
    try:
        with open(GEO_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(_geocode_cache, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def get_coordinates(location: str):
    """Retourne (lat, lon) pour une localisation donnée."""
    if not location or location == "Non spécifié":
        return None, None
    
    location = location.strip()
    
    if location in _geocode_cache:
        return _geocode_cache[location]
    
    try:
        result = geocode(f"{location}, France")
        if result:
            coords = (result.latitude, result.longitude)
        else:
            coords = (None, None)
    except Exception:
        coords = (None, None)
    
    _geocode_cache[location] = coords
    save_geocode_cache()
    return coords


def enrich_with_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes latitude/longitude basées sur la localisation."""
    if 'location' not in df.columns:
        df['latitude'] = None
        df['longitude'] = None
        return df
    
    df = df.copy()
    latitudes = []
    longitudes = []
    
    for location in df['location']:
        lat, lon = get_coordinates(location)
        latitudes.append(lat)
        longitudes.append(lon)
    
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    return df

# Layout de l'application
app.layout = html.Div(style={
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    'color': COLORS['text'],
    'padding': '20px'
}, children=[
    
    # Header
    html.Div(style={
        'textAlign': 'center',
        'marginBottom': '40px',
        'paddingBottom': '20px',
        'borderBottom': f'1px solid {COLORS["border"]}'
    }, children=[
        html.H1('🏍️ Moto Leboncoin Analyzer', style={
            'fontSize': '2.5rem',
            'fontWeight': '700',
            'marginBottom': '10px'
        }),
        html.P('Analysez les prix du marché de l\'occasion en temps réel', style={
            'color': COLORS['secondary'],
            'fontSize': '1.1rem'
        })
    ]),
    
    # Section de recherche
    html.Div(style={
        'maxWidth': '900px',
        'margin': '0 auto 40px',
        'backgroundColor': COLORS['card'],
        'padding': '30px',
        'borderRadius': '16px',
        'border': f'1px solid {COLORS["border"]}'
    }, children=[
        
        html.Label('Modèle de moto', style={
            'fontSize': '1rem',
            'fontWeight': '600',
            'marginBottom': '10px',
            'display': 'block'
        }),
        
        dcc.Input(
            id='input-model',
            type='text',
            placeholder='Ex: triumph street triple 765 rs, yamaha mt09, kawasaki z900',
            style={
                'width': '100%',
                'padding': '15px',
                'fontSize': '1rem',
                'border': f'1px solid {COLORS["border"]}',
                'borderRadius': '8px',
                'backgroundColor': COLORS['background'],
                'color': COLORS['text'],
                'marginBottom': '20px'
            },
            value='triumph street triple 765 rs'
        ),
        
        html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[
            html.Div(style={'flex': '1'}, children=[
                html.Label('Année min', style={'fontSize': '0.9rem', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(
                    id='input-year-min',
                    type='number',
                    min=2000,
                    max=datetime.now().year,
                    value=2017,
                    style={
                        'width': '100%',
                        'padding': '12px',
                        'fontSize': '1rem',
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '8px',
                        'backgroundColor': COLORS['background'],
                        'color': COLORS['text']
                    }
                )
            ]),
            html.Div(style={'flex': '1'}, children=[
                html.Label('Année max', style={'fontSize': '0.9rem', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(
                    id='input-year-max',
                    type='number',
                    min=2000,
                    max=datetime.now().year,
                    value=2020,
                    style={
                        'width': '100%',
                        'padding': '12px',
                        'fontSize': '1rem',
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '8px',
                        'backgroundColor': COLORS['background'],
                        'color': COLORS['text']
                    }
                )
            ]),
            html.Div(style={'flex': '1'}, children=[
                html.Label('Pages à scanner', style={'fontSize': '0.9rem', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Input(
                    id='input-pages',
                    type='number',
                    min=1,
                    max=10,
                    value=3,
                    style={
                        'width': '100%',
                        'padding': '12px',
                        'fontSize': '1rem',
                        'border': f'1px solid {COLORS["border"]}',
                        'borderRadius': '8px',
                        'backgroundColor': COLORS['background'],
                        'color': COLORS['text']
                    }
                )
            ])
        ]),
        
        html.Button('🔍 Scraper les annonces', id='btn-scrape', n_clicks=0, style={
            'width': '100%',
            'padding': '16px',
            'fontSize': '1.1rem',
            'fontWeight': '600',
            'backgroundColor': COLORS['primary'],
            'color': 'white',
            'border': 'none',
            'borderRadius': '8px',
            'cursor': 'pointer',
            'transition': 'all 0.2s'
        })
    ]),
    
    # Zone de loading
    dcc.Loading(
        id="loading",
        type="circle",
        color=COLORS['primary'],
        children=[
            html.Div(id='status-message', style={
                'textAlign': 'center',
                'marginBottom': '30px',
                'fontSize': '1.1rem',
                'color': COLORS['secondary']
            })
        ]
    ),
    
    # Graphique
    html.Div(id='graph-container', style=hidden_section_style()),
    
    # Carte
    html.Div(id='map-container', style=hidden_section_style()),
    
    # Liste des annonces
    html.Div(id='listings-container', style={
        'maxWidth': '1400px',
        'margin': '0 auto'
    }),
    
    # Stores pour les données et actions client-side
    dcc.Store(id='scraped-data'),
    dcc.Store(id='link-opener')
])


@app.callback(
    [Output('scraped-data', 'data'),
     Output('status-message', 'children'),
     Output('graph-container', 'style'),
     Output('graph-container', 'children'),
     Output('map-container', 'style'),
     Output('map-container', 'children'),
     Output('listings-container', 'children')],
    [Input('btn-scrape', 'n_clicks')],
    [State('input-model', 'value'),
     State('input-year-min', 'value'),
     State('input-year-max', 'value'),
     State('input-pages', 'value')]
)
def scrape_and_display(n_clicks, model, year_min, year_max, pages):
    if n_clicks == 0:
        return None, "", hidden_section_style(), None, hidden_section_style(), None, None
    
    if not model or not model.strip():
        return None, html.Div("⚠️ Veuillez entrer un modèle de moto", style={'color': '#f4212e'}), hidden_section_style(), None, hidden_section_style(), None, None
    
    try:
        # Scraping
        scraper = LeboncoinScraper()
        status_msg = html.Div([
            html.Div("🔄 Scraping en cours...", style={'marginBottom': '10px'}),
            html.Div(f"Modèle: {model} | Années: {year_min}-{year_max}", 
                     style={'fontSize': '0.9rem', 'color': COLORS['secondary']})
        ])
        
        raw_data = scraper.scrape(model, year_min, year_max, pages)
        
        if not raw_data:
            return None, html.Div("❌ Aucune annonce trouvée", style={'color': '#f4212e'}), hidden_section_style(), None, hidden_section_style(), None, None
        
        # Nettoyage
        cleaner = DataCleaner()
        df = cleaner.clean(raw_data)
        df = enrich_with_coordinates(df)
        
        if df.empty:
            return None, html.Div("❌ Aucune donnée exploitable après nettoyage", style={'color': '#f4212e'}), hidden_section_style(), None, hidden_section_style(), None, None
        
        # Création du graphique
        fig = create_scatter_plot(df)
        
        # Création de la liste d'annonces
        listings = create_listings_grid(df)
        
        success_msg = html.Div([
            html.Span("✅ ", style={'fontSize': '1.5rem'}),
            html.Span(f"{len(df)} annonces trouvées", style={'color': COLORS['success'], 'fontWeight': '600'})
        ])
        
        graph_div = dcc.Graph(
            id='ads-graph',
            figure=fig,
            style={'height': '600px'},
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
        map_fig = create_map(df)
        if map_fig:
            map_div = dcc.Graph(
                id='map-graph',
                figure=map_fig,
                style={'height': '600px'},
                config={'displayModeBar': False, 'displaylogo': False}
            )
            map_style = visible_section_style()
        else:
            map_style = hidden_section_style()
            map_div = None
        
        return (
            df.to_dict('records'),
            success_msg,
            visible_section_style(),
            graph_div,
            map_style,
            map_div,
            listings
        )
        
    except Exception as e:
        error_msg = html.Div([
            html.Div("❌ Erreur lors du scraping", style={'color': '#f4212e', 'fontWeight': '600'}),
            html.Div(str(e), style={'fontSize': '0.9rem', 'marginTop': '10px'})
        ])
        return None, error_msg, hidden_section_style(), None, hidden_section_style(), None, None


def create_scatter_plot(df):
    """Crée le graphique interactif Plotly"""
    
    # Préparation du texte de survol avec HTML
    hover_text = []
    for _, row in df.iterrows():
        text = f"<b>{row.get('title', 'N/A')[:50]}</b><br>"
        text += f"💰 {row.get('price', 0):,.0f} €<br>"
        text += f"📏 {row.get('mileage', 0):,.0f} km<br>"
        text += f"📅 {row.get('year', 'N/A')}<br>"
        text += f"📍 {row.get('location', 'N/A')}<br>"
        if row.get('link'):
            text += f"🔗 <a href='{row['link']}' target='_blank'>Voir l'annonce</a>"
        hover_text.append(text)
    
    fig = go.Figure()
    
    # Scatter plot avec couleur par année
    fig.add_trace(go.Scatter(
        x=df['mileage'],
        y=df['price'],
        mode='markers',
        marker=dict(
            size=12,
            color=df['year'] if 'year' in df.columns else COLORS['primary'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Année"),
            line=dict(width=1, color='white')
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        customdata=df[['link']].values if 'link' in df.columns else None
    ))
    
    fig.update_layout(
        title=dict(
            text='Prix vs Kilométrage',
            font=dict(size=24, color=COLORS['text'], family='Arial, sans-serif')
        ),
        xaxis=dict(
            title='Kilométrage (km)',
            gridcolor=COLORS['border'],
            color=COLORS['text'],
            tickformat=',',
        ),
        yaxis=dict(
            title='Prix (€)',
            gridcolor=COLORS['border'],
            color=COLORS['text'],
            tickformat=',',
        ),
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        hovermode='closest',
        clickmode='event+select',
        font=dict(color=COLORS['text'])
    )
    
    return fig


def create_map(df):
    """Crée la carte des annonces géolocalisées."""
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    
    map_df = df.dropna(subset=['latitude', 'longitude'])
    if map_df.empty:
        return None
    
    hover_text = []
    for _, row in map_df.iterrows():
        text = f"<b>{row.get('title', 'N/A')[:50]}</b><br>"
        text += f"💰 {row.get('price', 0):,.0f} €<br>"
        text += f"📏 {row.get('mileage', 0):,.0f} km<br>"
        text += f"📍 {row.get('location', 'N/A')}"
        hover_text.append(text)
    
    fig = go.Figure(go.Scattermapbox(
        lat=map_df['latitude'],
        lon=map_df['longitude'],
        mode='markers',
        marker=dict(
            size=14,
            color=map_df['price'],
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title='Prix €'),
            opacity=0.85
        ),
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        customdata=map_df[['link']].values if 'link' in map_df.columns else None
    ))
    
    center_lat = map_df['latitude'].mean()
    center_lon = map_df['longitude'].mean()
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            zoom=5,
            center=dict(lat=center_lat, lon=center_lon)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=COLORS['card'],
        title=dict(
            text='Répartition géographique des annonces',
            font=dict(color=COLORS['text'])
        )
    )
    
    return fig


def create_listings_grid(df):
    """Crée la grille d'annonces"""
    
    listings = []
    
    for _, row in df.iterrows():
        card = html.Div(style={
            'backgroundColor': COLORS['card'],
            'borderRadius': '12px',
            'padding': '20px',
            'border': f'1px solid {COLORS["border"]}',
            'transition': 'transform 0.2s, box-shadow 0.2s',
        }, children=[
            
            # Image
            html.Div(style={
                'width': '100%',
                'height': '200px',
                'backgroundColor': COLORS['background'],
                'borderRadius': '8px',
                'marginBottom': '15px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'overflow': 'hidden'
            }, children=[
                html.Img(
                    src=row.get('photo', ''),
                    style={'width': '100%', 'height': '100%', 'objectFit': 'cover'}
                ) if row.get('photo') else html.Div('📷 Pas d\'image', style={
                    'color': COLORS['secondary'],
                    'fontSize': '1.2rem'
                })
            ]),
            
            # Titre
            html.H3(row.get('title', 'Sans titre')[:60] + ('...' if len(row.get('title', '')) > 60 else ''), 
                    style={
                        'fontSize': '1.1rem',
                        'fontWeight': '600',
                        'marginBottom': '12px',
                        'minHeight': '50px'
                    }),
            
            # Infos
            html.Div(style={'marginBottom': '15px'}, children=[
                html.Div([
                    html.Span('💰 ', style={'marginRight': '5px'}),
                    html.Span(f"{row.get('price', 0):,.0f} €", style={'fontWeight': '700', 'fontSize': '1.3rem', 'color': COLORS['success']})
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Span('📏 ', style={'marginRight': '5px'}),
                    html.Span(f"{row.get('mileage', 0):,.0f} km")
                ], style={'marginBottom': '5px', 'color': COLORS['secondary']}),
                
                html.Div([
                    html.Span('📅 ', style={'marginRight': '5px'}),
                    html.Span(str(row.get('year', 'N/A')))
                ], style={'marginBottom': '5px', 'color': COLORS['secondary']}),
                
                html.Div([
                    html.Span('📍 ', style={'marginRight': '5px'}),
                    html.Span(row.get('location', 'N/A')[:30])
                ], style={'color': COLORS['secondary']})
            ]),
            
            # Bouton
            html.A('Ouvrir l\'annonce →', 
                   href=row.get('link', '#'),
                   target='_blank',
                   style={
                       'display': 'block',
                       'textAlign': 'center',
                       'padding': '12px',
                       'backgroundColor': COLORS['primary'],
                       'color': 'white',
                       'textDecoration': 'none',
                       'borderRadius': '8px',
                       'fontWeight': '600',
                       'transition': 'opacity 0.2s'
                   })
        ])
        
        listings.append(card)
    
    return html.Div(style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fill, minmax(300px, 1fr))',
        'gap': '20px',
        'marginTop': '40px'
    }, children=listings)


# Callback client-side pour ouvrir un nouvel onglet sur clic d'un point du graphique
app.clientside_callback(
    """
    function(scatterClick, mapClick) {
        const ctx = dash_clientside.callback_context;
        if (!ctx || !ctx.triggered || ctx.triggered.length === 0) {
            return window.dash_clientside.no_update;
        }
        const triggered = ctx.triggered[0].prop_id;
        let clickData = null;
        if (triggered === 'ads-graph.clickData') {
            clickData = scatterClick;
        } else if (triggered === 'map-graph.clickData') {
            clickData = mapClick;
        } else {
            return window.dash_clientside.no_update;
        }
        if (!clickData || !clickData.points || clickData.points.length === 0) {
            return window.dash_clientside.no_update;
        }
        const point = clickData.points[0];
        if (point.customdata && point.customdata.length > 0) {
            const url = point.customdata[0];
            if (url) {
                window.open(url, '_blank');
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('link-opener', 'data'),
    Input('ads-graph', 'clickData'),
    Input('map-graph', 'clickData'),
    prevent_initial_call=True
)


if __name__ == '__main__':
    print("🚀 Lancement de l'application Moto Leboncoin Analyzer")
    print("📍 Accédez à l'application sur: http://localhost:8050")
    app.run_server(debug=True, port=8050)
