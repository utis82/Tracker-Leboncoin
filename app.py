import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np
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
SEARCH_CACHE_PATH = os.path.join(CACHE_DIR, "search_cache.json")
LISTING_HISTORY_PATH = os.path.join(CACHE_DIR, "listings_history.json")
TRACE_COLORS = [
    '#1d9bf0', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
]

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


def load_json_file(path: str, default):
    """Charge un fichier JSON et retourne une valeur par défaut en cas d'erreur."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, type(default)) else default
    except (json.JSONDecodeError, OSError):
        return default


def save_json_file(path: str, data):
    """Sauvegarde un fichier JSON."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def normalize_model_name(model: str) -> str:
    """Normalise un libellé de modèle pour les clés de cache."""
    if not model:
        return ''
    return ' '.join(model.strip().lower().split())


def build_search_key(model: str, year_min: int, year_max: int, pages: int) -> str:
    """Construit une clé unique pour une recherche."""
    normalized = normalize_model_name(model)
    return f"{normalized}|{year_min}|{year_max}|{pages}"


def format_timestamp_display(timestamp: str) -> str:
    """Formate un timestamp ISO pour l'affichage."""
    if not timestamp:
        return ''
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime('%d/%m/%Y %H:%M')
    except ValueError:
        return timestamp


def build_search_label(model: str, year_min: int, year_max: int, timestamp: str) -> str:
    """Construit un libellé lisible pour une recherche."""
    base = f"{model.strip()} ({year_min}-{year_max})"
    formatted = format_timestamp_display(timestamp)
    if formatted:
        return f"{base} • {formatted}"
    return base


def load_search_cache() -> dict:
    """Charge le cache des recherches."""
    return load_json_file(SEARCH_CACHE_PATH, {})


def save_search_cache(cache: dict):
    """Sauvegarde le cache des recherches."""
    save_json_file(SEARCH_CACHE_PATH, cache)


def get_history_options(cache: dict) -> list:
    """Construit la liste d'options pour les recherches sauvegardées."""
    options = []
    for key, entry in cache.items():
        label = entry.get('label') or build_search_label(
            entry.get('model', 'N/A'),
            entry.get('year_min', '?'),
            entry.get('year_max', '?'),
            entry.get('timestamp', '')
        )
        options.append({'label': label, 'value': key})
    options.sort(key=lambda opt: cache.get(opt['value'], {}).get('timestamp', ''), reverse=True)
    return options


def parse_iso_datetime(value: str):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def update_listings_history(raw_data, search_key: str, label: str, timestamp: str):
    """Met à jour l'historique des annonces pour estimer la durée de vente."""
    if not raw_data:
        return
    history = load_json_file(LISTING_HISTORY_PATH, {})
    changed = False
    for item in raw_data:
        link = item.get('link')
        if not link:
            continue
        entry = history.get(link, {})
        if not entry.get('first_seen'):
            entry['first_seen'] = timestamp
        entry['last_seen'] = timestamp
        entry.setdefault('searches', [])
        if search_key not in entry['searches']:
            entry['searches'].append(search_key)
        entry.setdefault('labels', [])
        if label not in entry['labels']:
            entry['labels'].append(label)
        history[link] = entry
        changed = True
    if changed:
        save_json_file(LISTING_HISTORY_PATH, history)


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


def annotate_with_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la durée estimée de vente basée sur l'historique."""
    if df.empty or 'link' not in df.columns:
        df['estimated_duration_days'] = None
        return df
    history = load_json_file(LISTING_HISTORY_PATH, {})
    durations = []
    for link in df['link']:
        entry = history.get(link)
        if not entry:
            durations.append(None)
            continue
        first_seen = parse_iso_datetime(entry.get('first_seen'))
        last_seen = parse_iso_datetime(entry.get('last_seen'))
        if first_seen and last_seen:
            delta_days = max((last_seen - first_seen).days, 0) + 1
            durations.append(delta_days)
        else:
            durations.append(None)
    df['estimated_duration_days'] = durations
    return df


def prepare_dataframe(records, label: str, cleaner: DataCleaner) -> pd.DataFrame:
    """Nettoie, enrichit et annote un ensemble de résultats."""
    if not records:
        return pd.DataFrame()
    df = cleaner.clean(records)
    if df.empty:
        return df
    df = enrich_with_coordinates(df)
    df = annotate_with_duration(df)
    df['source'] = label
    return df


def fetch_search_results(model: str, year_min: int, year_max: int, pages: int,
                         use_cache: bool, search_cache: dict):
    """Récupère les résultats (cache ou scraping) et met à jour le cache si besoin."""
    search_key = build_search_key(model, year_min, year_max, pages)
    entry = search_cache.get(search_key)
    if use_cache:
        if entry:
            return entry.get('data', []), entry, True, search_key, search_cache
        return None, None, True, search_key, search_cache
    scraper = LeboncoinScraper()
    raw_data = scraper.scrape(model, year_min, year_max, pages)
    timestamp = datetime.utcnow().isoformat()
    label = build_search_label(model, year_min, year_max, timestamp)
    entry = {
        'model': model,
        'year_min': year_min,
        'year_max': year_max,
        'pages': pages,
        'timestamp': timestamp,
        'label': label,
        'data': raw_data or []
    }
    search_cache[search_key] = entry
    save_search_cache(search_cache)
    return raw_data, entry, False, search_key, search_cache


history_dropdown_options = get_history_options(load_search_cache())

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
        }),

        html.Div(style={'marginTop': '20px', 'display': 'flex', 'flexDirection': 'column', 'gap': '15px'}, children=[
            dcc.Checklist(
                id='use-cache-toggle',
                options=[{'label': "♻️ Utiliser les données en cache si disponibles (pas de nouveau scraping)", 'value': 'use_cache'}],
                value=['use_cache'],
                style={'color': COLORS['secondary']}
            ),
            html.Div(children=[
                html.Label('Superposer des recherches sauvegardées', style={
                    'fontSize': '0.9rem',
                    'marginBottom': '8px',
                    'display': 'block'
                }),
                dcc.Dropdown(
                    id='history-selection',
                    options=history_dropdown_options,
                    value=[],
                    multi=True,
                    placeholder='Sélectionnez des recherches précédentes à ajouter au graphique',
                    style={'backgroundColor': COLORS['background'], 'color': COLORS['text']}
                ),
                html.Div("Les recherches sont automatiquement enregistrées dans le cache.", style={
                    'color': COLORS['secondary'],
                    'fontSize': '0.85rem',
                    'marginTop': '5px'
                })
            ])
        ])
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

    # Graphique durée vs prix
    html.Div(id='duration-container', style=hidden_section_style()),
    
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
     Output('duration-container', 'style'),
     Output('duration-container', 'children'),
     Output('listings-container', 'children'),
     Output('history-selection', 'options')],
    [Input('btn-scrape', 'n_clicks'),
     Input('history-selection', 'value')],
    [State('input-model', 'value'),
     State('input-year-min', 'value'),
     State('input-year-max', 'value'),
     State('input-pages', 'value'),
     State('use-cache-toggle', 'value')]
)
def scrape_and_display(n_clicks, history_selection, model, year_min, year_max, pages, use_cache_values):
    history_selection = history_selection or []
    use_cache = bool(use_cache_values and 'use_cache' in use_cache_values)
    cleaner = DataCleaner()
    search_cache = load_search_cache()
    history_options = get_history_options(search_cache)
    ctx = callback_context
    triggered_prop = ctx.triggered[0]['prop_id'] if ctx and ctx.triggered else ""
    
    def empty_response(message=None):
        return (
            None,
            message or "",
            hidden_section_style(),
            None,
            hidden_section_style(),
            None,
            hidden_section_style(),
            None,
            None,
            history_options
        )
    
    if n_clicks == 0 and not history_selection:
        return empty_response()
    
    if n_clicks > 0 and (not model or not model.strip()):
        error = html.Div("⚠️ Veuillez entrer un modèle de moto", style={'color': '#f4212e'})
        return empty_response(error)
    
    datasets = []
    status_blocks = []
    seen_keys = set()
    
    # Traite la recherche courante si le bouton a été cliqué
    if n_clicks and n_clicks > 0:
        raw_data, entry, used_cache, search_key, search_cache = fetch_search_results(
            model, year_min, year_max, pages, use_cache, search_cache
        )
        history_options = get_history_options(search_cache)
        
        if use_cache and raw_data is None:
            warning = html.Div("⚠️ Aucun cache trouvé pour cette recherche. Décochez l'option ou lancez une autre recherche.", style={'color': '#f9a826'})
            return empty_response(warning)
        
        if not raw_data:
            error = html.Div("❌ Aucune annonce trouvée", style={'color': '#f4212e'})
            return empty_response(error)
        
        if not used_cache:
            update_listings_history(raw_data, search_key, entry['label'], entry['timestamp'])
        
        df_current = prepare_dataframe(raw_data, entry['label'], cleaner)
        if df_current.empty:
            error = html.Div("❌ Aucune donnée exploitable après nettoyage", style={'color': '#f4212e'})
            return empty_response(error)
        
        datasets.append(df_current)
        seen_keys.add(search_key)
        status_blocks.append(html.Div([
            html.Span("♻️ " if used_cache else "✅ ", style={'marginRight': '6px'}),
            html.Span(f"{len(df_current)} annonces • {entry['label']}", style={'fontWeight': '600'}),
            html.Span(" (cache)" if used_cache else " (scraping en direct)", style={'color': COLORS['secondary'], 'marginLeft': '6px'})
        ], style={'padding': '6px 0'}))
    
    # Ajoute les recherches historiques sélectionnées
    for key in history_selection:
        if key in seen_keys:
            continue
        entry = search_cache.get(key)
        if not entry:
            continue
        df_hist = prepare_dataframe(entry.get('data', []), entry.get('label', 'Historique'), cleaner)
        if df_hist.empty:
            continue
        datasets.append(df_hist)
        seen_keys.add(key)
        status_blocks.append(html.Div([
            html.Span("📚 ", style={'marginRight': '6px'}),
            html.Span(f"{len(df_hist)} annonces • {entry.get('label', 'Historique')}", style={'fontWeight': '600'}),
            html.Span(" (cache historique)", style={'color': COLORS['secondary'], 'marginLeft': '6px'})
        ], style={'padding': '6px 0'}))
    
    if not datasets:
        if triggered_prop == 'history-selection.value':
            warning = html.Div("⚠️ Sélection vide ou invalide. Relancez une recherche ou choisissez d'autres éléments.", style={'color': '#f9a826'})
            return empty_response(warning)
        return empty_response()
    
    combined_df = pd.concat(datasets, ignore_index=True)
    if 'price' in combined_df.columns:
        combined_df = combined_df.sort_values('price').reset_index(drop=True)
    combined_records = combined_df.to_dict('records')
    
    status_msg = html.Div(status_blocks, style={'display': 'flex', 'flexDirection': 'column', 'gap': '4px'})
    
    # Graphique principal
    scatter_fig = create_scatter_plot(combined_df)
    graph_div = dcc.Graph(
        id='ads-graph',
        figure=scatter_fig,
        style={'height': '600px'},
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    # Carte
    map_fig = create_map(combined_df)
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
    
    # Durée estimée
    duration_fig = create_duration_plot(combined_df)
    if duration_fig:
        duration_div = dcc.Graph(
            id='duration-graph',
            figure=duration_fig,
            style={'height': '500px'},
            config={'displayModeBar': True, 'displaylogo': False}
        )
        duration_style = visible_section_style()
    else:
        duration_div = None
        duration_style = hidden_section_style()
    
    listings = create_listings_grid(combined_df)
    
    return (
        combined_records,
        status_msg,
        visible_section_style(),
        graph_div,
        map_style,
        map_div,
        duration_style,
        duration_div,
        listings,
        history_options
    )


def create_scatter_plot(df):
    """Crée le graphique interactif Plotly avec superposition multi-sources."""
    if df.empty:
        return go.Figure()
    
    def build_hover(row):
        text = f"<b>{row.get('title', 'N/A')[:50]}</b><br>"
        text += f"💰 {row.get('price', 0):,.0f} €<br>"
        text += f"📏 {row.get('mileage', 0):,.0f} km<br>"
        text += f"📅 {row.get('year', 'N/A')}<br>"
        text += f"📍 {row.get('location', 'N/A')}<br>"
        duration = row.get('estimated_duration_days')
        if duration:
            text += f"⏱️ {duration} jour(s) estimés<br>"
        text += f"🧾 {row.get('source', 'Recherche actuelle')}"
        if row.get('link'):
            text += f"<br>🔗 <a href='{row['link']}' target='_blank'>Voir l'annonce</a>"
        return text
    
    fig = go.Figure()
    if 'source' in df.columns:
        groups = df['source'].fillna('Recherche').unique()
    else:
        groups = ['Recherche']
    
    for idx, group in enumerate(groups):
        subset = df if 'source' not in df.columns else df[df['source'] == group]
        if subset.empty:
            continue
        hover_text = subset.apply(build_hover, axis=1)
        fig.add_trace(go.Scatter(
            x=subset['mileage'],
            y=subset['price'],
            mode='markers',
            name=group,
            marker=dict(
                size=12,
                color=TRACE_COLORS[idx % len(TRACE_COLORS)],
                line=dict(width=1, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            customdata=subset[['link']].values if 'link' in subset.columns else None
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
        text += f"📍 {row.get('location', 'N/A')}<br>"
        text += f"🧾 {row.get('source', 'Recherche')}"
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


def create_duration_plot(df):
    """Crée le graphique durée estimée vs prix."""
    if 'estimated_duration_days' not in df.columns:
        return None
    duration_df = df.dropna(subset=['estimated_duration_days'])
    if duration_df.empty:
        return None
    
    def build_hover(row):
        text = f"<b>{row.get('title', 'N/A')[:50]}</b><br>"
        text += f"💰 {row.get('price', 0):,.0f} €<br>"
        text += f"⏱️ {row.get('estimated_duration_days')} jour(s) estimés<br>"
        text += f"📍 {row.get('location', 'N/A')}<br>"
        text += f"🧾 {row.get('source', 'Recherche')}"
        return text
    
    fig = go.Figure()
    groups = duration_df['source'].fillna('Recherche').unique() if 'source' in duration_df.columns else ['Recherche']
    
    for idx, group in enumerate(groups):
        subset = duration_df if 'source' not in duration_df.columns else duration_df[duration_df['source'] == group]
        if subset.empty:
            continue
        hover_text = subset.apply(build_hover, axis=1)
        fig.add_trace(go.Scatter(
            x=subset['price'],
            y=subset['estimated_duration_days'],
            mode='markers',
            name=f"{group} (durée)",
            marker=dict(
                size=12,
                color=TRACE_COLORS[idx % len(TRACE_COLORS)],
                line=dict(width=1, color='white')
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            customdata=subset[['link']].values if 'link' in subset.columns else None
        ))
    
    if len(duration_df) >= 2:
        try:
            x_vals = duration_df['price'].values
            y_vals = duration_df['estimated_duration_days'].values
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(duration_df['price'].min(), duration_df['price'].max(), 50)
            y_line = slope * x_line + intercept
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode='lines',
                name='Tendance estimée',
                line=dict(color=COLORS['secondary'], dash='dash')
            ))
        except (ValueError, np.linalg.LinAlgError):
            pass
    
    fig.update_layout(
        title=dict(
            text='Durée estimée de vente vs Prix',
            font=dict(size=22, color=COLORS['text'])
        ),
        xaxis=dict(
            title='Prix (€)',
            gridcolor=COLORS['border'],
            color=COLORS['text'],
            tickformat=','
        ),
        yaxis=dict(
            title='Durée estimée (jours)',
            gridcolor=COLORS['border'],
            color=COLORS['text']
        ),
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        hovermode='closest',
        clickmode='event+select',
        font=dict(color=COLORS['text'])
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
            html.Div(row.get('source', 'Recherche'), style={
                'fontSize': '0.85rem',
                'color': COLORS['secondary'],
                'marginBottom': '10px'
            }),
            
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
                ], style={'color': COLORS['secondary']}),
                
                html.Div([
                    html.Span('⏱️ ', style={'marginRight': '5px'}),
                    html.Span(
                        f"{int(row.get('estimated_duration_days'))} j" 
                        if row.get('estimated_duration_days') is not None and not pd.isna(row.get('estimated_duration_days'))
                        else "N/A"
                    )
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
    function(scatterClick, mapClick, durationClick) {
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
        } else if (triggered === 'duration-graph.clickData') {
            clickData = durationClick;
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
    Input('duration-graph', 'clickData'),
    prevent_initial_call=True
)


if __name__ == '__main__':
    print("🚀 Lancement de l'application Moto Leboncoin Analyzer")
    print("📍 Accédez à l'application sur: http://localhost:8050")
    app.run_server(debug=True, port=8050)
