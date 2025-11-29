import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import os
import base64
from uuid import uuid4
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
WATCHLIST_PATH = os.path.join(CACHE_DIR, "watchlist.json")
TRACE_COLORS = [
    '#1d9bf0', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
]
CONDITION_KEYWORDS = {
    'excellent': 1.0,
    'parfait': 0.9,
    'comme neuf': 0.9,
    'neuf': 0.85,
    'propre': 0.7,
    'tres bon etat': 0.8,
    'tres bon état': 0.8,
    'bon etat': 0.6,
    'bon état': 0.6,
    'a revoir': 0.2,
    'project': 0.1,
    'accident': 0.05,
    'panne': 0.1,
    'pour pieces': 0.1,
    'epave': 0.0
}

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


def load_watchlist() -> list:
    """Charge la watchlist des alertes."""
    data = load_json_file(WATCHLIST_PATH, [])
    return data if isinstance(data, list) else []


def save_watchlist(entries: list):
    """Sauvegarde la watchlist."""
    save_json_file(WATCHLIST_PATH, entries)


def decode_uploaded_json(contents):
    """Décode un fichier JSON uploadé via Dash."""
    if not contents:
        return None
    try:
        _, content_string = contents.split(',', 1)
        decoded = base64.b64decode(content_string).decode('utf-8')
        return json.loads(decoded)
    except (ValueError, json.JSONDecodeError, base64.binascii.Error):
        return None


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


def update_listings_history(df: pd.DataFrame, search_key: str, label: str, timestamp: str):
    """Met à jour l'historique des annonces pour estimer la durée de vente."""
    if df is None or df.empty:
        return
    history = load_json_file(LISTING_HISTORY_PATH, {})
    changed = False
    for _, row in df.iterrows():
        link = row.get('link')
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
        observations = entry.setdefault('observations', [])
        observations.append({
            'timestamp': timestamp,
            'price': row.get('price'),
            'mileage': row.get('mileage'),
            'market_score': row.get('market_score'),
            'source': row.get('source')
        })
        entry['last_price'] = row.get('price')
        entry['last_mileage'] = row.get('mileage')
        entry['observation_count'] = len(observations)
        history[link] = entry
        changed = True
    if changed:
        save_json_file(LISTING_HISTORY_PATH, history)


def build_history_dataframe():
    """Construit un DataFrame à partir de l'historique."""
    history = load_json_file(LISTING_HISTORY_PATH, {})
    rows = []
    for link, entry in history.items():
        observations = entry.get('observations', [])
        for obs in observations:
            rows.append({
                'link': link,
                'timestamp': obs.get('timestamp', entry.get('last_seen')),
                'price': obs.get('price'),
                'mileage': obs.get('mileage'),
                'source': obs.get('source') or (entry.get('labels') or ['Historique'])[0],
                'market_score': obs.get('market_score')
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


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


def extract_condition_score(title: str) -> (float, str):
    """Déduit un score d'état depuis le titre."""
    if not title:
        return 0.5, "Standard"
    text = str(title).lower()
    best_score = 0.5
    best_label = "Standard"
    for keyword, score in CONDITION_KEYWORDS.items():
        if keyword in text:
            if score > best_score:
                best_score = score
                best_label = keyword
    return best_score, best_label.title()


def normalize_series(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalise une série entre 0 et 1."""
    if series.empty:
        return pd.Series([], dtype=float)
    valid = series.replace([float('inf'), float('-inf')], pd.NA).dropna()
    if valid.empty:
        return pd.Series([0.5] * len(series), index=series.index)
    min_val = valid.min()
    max_val = valid.max()
    if min_val == max_val:
        normalized = pd.Series([0.5] * len(series), index=series.index)
    else:
        normalized = (series - min_val) / (max_val - min_val)
    if invert:
        normalized = 1 - normalized
    return normalized.fillna(0.5)


def compute_market_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule un score de pertinence marché."""
    if df.empty or 'price' not in df.columns:
        df['market_score'] = None
        df['condition_label'] = "N/A"
        return df
    
    df = df.copy()
    price_norm = normalize_series(df['price'], invert=True)
    mileage_norm = normalize_series(df['mileage'], invert=True) if 'mileage' in df.columns else pd.Series(0.5, index=df.index)
    year_norm = normalize_series(df['year']) if 'year' in df.columns else pd.Series(0.5, index=df.index)
    
    condition_scores = []
    condition_labels = []
    for title in df.get('title', []):
        score, label = extract_condition_score(title)
        condition_scores.append(score)
        condition_labels.append(label)
    condition_series = pd.Series(condition_scores, index=df.index)
    
    market_score = (
        0.35 * price_norm +
        0.25 * mileage_norm +
        0.20 * year_norm +
        0.20 * condition_series
    ) * 100
    df['market_score'] = market_score.round(1)
    df['condition_label'] = condition_labels
    df['score_label'] = pd.cut(
        df['market_score'],
        bins=[0, 40, 60, 75, 100],
        labels=['⚠️ À surveiller', 'Correct', 'Bon plan', '🔥 Immanquable'],
        include_lowest=True
    ).astype(str)
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
    df = compute_market_score(df)
    return df


def apply_watchlist(df: pd.DataFrame, watchlist: list) -> pd.DataFrame:
    """Ajoute les informations de watchlist sur le DataFrame."""
    if df.empty or not watchlist:
        df['watchlist_tags'] = [[] for _ in range(len(df))]
        df['is_watchlist_hit'] = False
        return df
    
    tags_column = []
    for _, row in df.iterrows():
        row_tags = []
        title = str(row.get('title', '')).lower()
        price = row.get('price')
        for entry in watchlist:
            query = entry.get('query', '').lower().strip()
            if query and query not in title:
                continue
            max_price = entry.get('max_price')
            if max_price and price and price > max_price:
                continue
            min_score = entry.get('min_score')
            if min_score and row.get('market_score') and row['market_score'] < min_score:
                continue
            row_tags.append(entry.get('label', 'Watchlist'))
        tags_column.append(row_tags)
    df['watchlist_tags'] = tags_column
    df['is_watchlist_hit'] = df['watchlist_tags'].apply(lambda tags: len(tags) > 0)
    return df


def apply_advanced_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Applique les filtres avancés sur le DataFrame."""
    if df.empty:
        return df
    filtered = df.copy()
    if 'photo' not in filtered.columns:
        filtered['photo'] = None
    price_min = filters.get('price_min')
    price_max = filters.get('price_max')
    mileage_min = filters.get('mileage_min')
    mileage_max = filters.get('mileage_max')
    min_score = filters.get('min_score')
    require_photo = filters.get('require_photo')
    only_watchlist = filters.get('only_watchlist')
    condition_values = filters.get('conditions') or []
    
    if price_min is not None:
        filtered = filtered[filtered['price'] >= price_min]
    if price_max is not None:
        filtered = filtered[filtered['price'] <= price_max]
    if mileage_min is not None and 'mileage' in filtered.columns:
        filtered = filtered[filtered['mileage'] >= mileage_min]
    if mileage_max is not None and 'mileage' in filtered.columns:
        filtered = filtered[filtered['mileage'] <= mileage_max]
    if min_score is not None and 'market_score' in filtered.columns:
        filtered = filtered[filtered['market_score'].fillna(0) >= min_score]
    if require_photo:
        filtered = filtered[filtered['photo'].notna()]
    if only_watchlist:
        filtered = filtered[filtered['is_watchlist_hit']]
    if condition_values:
        filtered = filtered[filtered['condition_label'].isin(condition_values)]
    
    return filtered.reset_index(drop=True)


def build_visual_outputs(data_payload, watchlist_data, filters):
    filter_options = filters.get('options', [])
    watchlist_data = watchlist_data or []
    if not data_payload or not data_payload.get('combined_records'):
        empty_msg = html.Div("Commencez par lancer un scraping ou charger une recherche depuis l'historique.", style={'textAlign': 'center', 'color': COLORS['secondary'], 'padding': '40px'})
        return (hidden_section_style(), empty_msg,
                hidden_section_style(), None,
                hidden_section_style(), None,
                hidden_section_style(), None,
                hidden_section_style(), None,
                hidden_section_style(), None,
                None)
    
    df = ensure_dataframe_ready(data_payload['combined_records'])
    df = apply_watchlist(df, watchlist_data)
    advanced_filters = {
        'price_min': filters.get('price_min'),
        'price_max': filters.get('price_max'),
        'mileage_min': filters.get('mileage_min'),
        'mileage_max': filters.get('mileage_max'),
        'min_score': filters.get('min_score'),
        'require_photo': 'photo' in filter_options,
        'only_watchlist': 'watchlist' in filter_options,
        'conditions': filters.get('conditions')
    }
    filtered_df = apply_advanced_filters(df, advanced_filters)
    
    if filtered_df.empty:
        empty_msg = html.Div("Aucune annonce ne correspond à vos filtres. Ajustez vos critères.", style={'textAlign': 'center', 'color': '#f4212e', 'padding': '40px'})
        return (hidden_section_style(), empty_msg,
                hidden_section_style(), None,
                hidden_section_style(), None,
                hidden_section_style(), None,
                hidden_section_style(), None,
                hidden_section_style(), None,
                None)
    
    scatter_fig = create_scatter_plot(filtered_df)
    graph_div = dcc.Graph(
        id='ads-graph',
        figure=scatter_fig,
        style={'height': '600px'},
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    map_fig = create_map(filtered_df)
    if map_fig:
        map_div = dcc.Graph(
            id='map-graph',
            figure=map_fig,
            style={'height': '600px'},
            config={'displayModeBar': False, 'displaylogo': False}
        )
        map_style = visible_section_style()
    else:
        map_div = None
        map_style = hidden_section_style()
    
    duration_fig = create_duration_plot(filtered_df)
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
    
    comparison_fig = create_comparison_plot(filtered_df)
    if comparison_fig:
        comparison_div = dcc.Graph(
            id='comparison-graph',
            figure=comparison_fig,
            style={'height': '500px'},
            config={'displayModeBar': False, 'displaylogo': False}
        )
        comparison_style = visible_section_style()
    else:
        comparison_div = None
        comparison_style = hidden_section_style()
    
    history_df = build_history_dataframe()
    trend_fig = create_trend_plot(history_df)
    if trend_fig:
        trend_div = dcc.Graph(
            id='trend-graph',
            figure=trend_fig,
            style={'height': '500px'},
            config={'displayModeBar': False, 'displaylogo': False}
        )
        trend_style = visible_section_style()
    else:
        trend_div = None
        trend_style = hidden_section_style()
    
    watchlist_hits = filtered_df[filtered_df['is_watchlist_hit']]
    if not watchlist_hits.empty:
        matches_style = visible_section_style()
        matches_content = html.Div([
            html.H3("Alertes actives", style={'marginBottom': '10px'}),
            html.Ul([
                html.Li(
                    f"{row.get('title', 'Annonce')} • {', '.join(row.get('watchlist_tags', []))} • {format_value(row.get('price'), '€')}",
                    style={'marginBottom': '4px'}
                )
                for _, row in watchlist_hits.iterrows()
            ])
        ], style={'backgroundColor': COLORS['card'], 'padding': '20px', 'borderRadius': '12px'})
    else:
        matches_style = hidden_section_style()
        matches_content = None
    
    listings = create_listings_grid(filtered_df)
    
    return (
        visible_section_style(),
        graph_div,
        map_style,
        map_div,
        duration_style,
        duration_div,
        comparison_style,
        comparison_div,
        trend_style,
        trend_div,
        matches_style,
        matches_content,
        listings
    )


def format_value(value, suffix='', default='N/A'):
    if value is None or value == '':
        return default
    numeric_suffixes = {'€': ' €', 'km': ' km'}
    if suffix in numeric_suffixes:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if np.isnan(numeric):
            return default
        formatted = f"{numeric:,.0f}{numeric_suffixes[suffix]}".replace(',', ' ')
        return formatted
    return str(value)


def build_status_message(entries):
    if not entries:
        return ""
    blocks = []
    for entry in entries:
        icon = entry.get('icon', '✅ ')
        label = entry.get('label', 'Recherche')
        count = entry.get('count', 0)
        tone = entry.get('tone', COLORS['success'])
        note = entry.get('note', '')
        blocks.append(html.Div([
            html.Span(icon, style={'marginRight': '6px'}),
            html.Span(f"{count} annonces • {label}", style={'fontWeight': '600'}),
            html.Span(note, style={'color': COLORS['secondary'], 'marginLeft': '6px'})
        ], style={'padding': '6px 0'}))
    return html.Div(blocks, style={'display': 'flex', 'flexDirection': 'column', 'gap': '4px'})


def ensure_dataframe_ready(records: list) -> pd.DataFrame:
    """Normalise les colonnes numériques/score pour des records potentiellement bruts."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    cleaner = DataCleaner()
    def normalize_numeric(value, func):
        if isinstance(value, (int, float, np.number)):
            return float(value)
        return func(value)
    
    if 'price' in df.columns:
        df['price'] = df['price'].apply(lambda v: normalize_numeric(v, cleaner.clean_price))
    if 'mileage' in df.columns:
        df['mileage'] = df['mileage'].apply(lambda v: normalize_numeric(v, cleaner.clean_mileage))
    if 'year' in df.columns:
        df['year'] = df['year'].apply(lambda v: normalize_numeric(v, cleaner.clean_year))
    # Ajoute colonnes manquantes
    if 'source' not in df.columns:
        df['source'] = 'Recherche'
    else:
        df['source'] = df['source'].fillna('Recherche')
    if 'market_score' not in df.columns or df['market_score'].isna().all():
        df = compute_market_score(df)
    if 'condition_label' not in df.columns:
        df['condition_label'] = 'Standard'
    else:
        df['condition_label'] = df['condition_label'].fillna('Standard')
    if 'score_label' not in df.columns:
        df['score_label'] = 'Correct'
    else:
        df['score_label'] = df['score_label'].fillna('Correct')
    if 'watchlist_tags' not in df.columns:
        df['watchlist_tags'] = [[] for _ in range(len(df))]
    if 'is_watchlist_hit' not in df.columns:
        df['is_watchlist_hit'] = False
    if 'photo' not in df.columns:
        df['photo'] = None
    if 'latitude' not in df.columns:
        df['latitude'] = None
    if 'longitude' not in df.columns:
        df['longitude'] = None
    if 'estimated_duration_days' not in df.columns:
        df['estimated_duration_days'] = None
    if 'link' not in df.columns:
        df['link'] = None
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
    timestamp = datetime.now(timezone.utc).isoformat()
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
            ]),
            html.Details(open=False, style={'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '12px'}, children=[
                html.Summary("Filtres avancés", style={'cursor': 'pointer', 'fontWeight': '600'}),
                html.P("Affinez l'analyse avec des contraintes supplémentaires (prix, kilométrage, score, état...).", style={'color': COLORS['secondary'], 'fontSize': '0.9rem'}),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '12px'}, children=[
                    dcc.Input(id='filter-price-min', type='number', placeholder='Prix min', min=0, style={'padding': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']}),
                    dcc.Input(id='filter-price-max', type='number', placeholder='Prix max', min=0, style={'padding': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']}),
                    dcc.Input(id='filter-mileage-min', type='number', placeholder='Km min', min=0, style={'padding': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']}),
                    dcc.Input(id='filter-mileage-max', type='number', placeholder='Km max', min=0, style={'padding': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']})
                ]),
                html.Div(style={'marginTop': '10px'}, children=[
                    html.Label('Score minimum', style={'fontSize': '0.9rem'}),
                    dcc.Slider(id='filter-score-min', min=0, max=100, step=5, value=0,
                               marks={0: '0', 50: '50', 100: '100'}, tooltip={'placement': 'bottom', 'always_visible': False})
                ]),
                dcc.Checklist(
                    id='filter-options',
                    options=[
                        {'label': '📷 Photos uniquement', 'value': 'photo'},
                        {'label': '🔔 Watchlist uniquement', 'value': 'watchlist'}
                    ],
                    value=[],
                    style={'color': COLORS['secondary'], 'marginTop': '10px'}
                ),
                dcc.Dropdown(
                    id='filter-condition',
                    options=[{'label': label.title(), 'value': label.title()} for label in set(CONDITION_KEYWORDS.keys())],
                    multi=True,
                    placeholder='Filtrer par mention dans le titre (état)',
                    style={'marginTop': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']}
                )
            ]),
            html.Details(open=False, style={'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '12px'}, children=[
                html.Summary("Watchlist & alertes", style={'cursor': 'pointer', 'fontWeight': '600'}),
                html.P("Créez des alertes qui se déclenchent dès qu'une annonce correspond (prix, mot-clé, score).", style={'color': COLORS['secondary'], 'fontSize': '0.9rem'}),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))', 'gap': '12px'}, children=[
                    dcc.Input(id='watchlist-label', type='text', placeholder='Nom', style={'padding': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']}),
                    dcc.Input(id='watchlist-query', type='text', placeholder='Mot-clé (modèle)', style={'padding': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']}),
                    dcc.Input(id='watchlist-max-price', type='number', placeholder='Prix max', min=0, style={'padding': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']}),
                    dcc.Input(id='watchlist-min-score', type='number', placeholder='Score min', min=0, max=100, style={'padding': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']})
                ]),
                html.Div(style={'display': 'flex', 'gap': '10px', 'marginTop': '10px'}, children=[
                    html.Button("Ajouter à la watchlist", id='btn-watchlist-add', n_clicks=0, style={
                        'flex': '1',
                        'padding': '12px',
                        'backgroundColor': COLORS['success'],
                        'border': 'none',
                        'borderRadius': '8px',
                        'color': 'white',
                        'cursor': 'pointer'
                    }),
                    html.Button("Supprimer la sélection", id='btn-watchlist-remove', n_clicks=0, style={
                        'flex': '1',
                        'padding': '12px',
                        'backgroundColor': '#f4212e',
                        'border': 'none',
                        'borderRadius': '8px',
                        'color': 'white',
                        'cursor': 'pointer'
                    })
                ]),
                dcc.Dropdown(
                    id='watchlist-selection',
                    options=[{'label': entry.get('label', 'Watch'), 'value': entry.get('id')} for entry in load_watchlist()],
                    multi=False,
                    placeholder="Sélection pour suppression",
                    style={'marginTop': '10px', 'backgroundColor': COLORS['background'], 'color': COLORS['text']}
                ),
                html.Div(id='watchlist-feedback', style={'marginTop': '8px', 'color': COLORS['secondary']})
            ]),
            html.Details(open=False, style={'backgroundColor': COLORS['card'], 'padding': '15px', 'borderRadius': '12px'}, children=[
                html.Summary("Gestion des caches & exports", style={'cursor': 'pointer', 'fontWeight': '600'}),
                html.P("Sauvegardez vos scrapes ou chargez des caches fournis pour éviter d'utiliser des crédits.", style={'color': COLORS['secondary'], 'fontSize': '0.9rem'}),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))', 'gap': '10px'}, children=[
                    html.Button("⬇️ Exporter cache recherches", id='btn-export-search', n_clicks=0, style={'padding': '12px', 'borderRadius': '8px', 'border': 'none', 'cursor': 'pointer'}),
                    html.Button("⬇️ Exporter historique annonces", id='btn-export-history', n_clicks=0, style={'padding': '12px', 'borderRadius': '8px', 'border': 'none', 'cursor': 'pointer'})
                ]),
                html.Div(style={'marginTop': '10px'}, children=[
                    dcc.Upload(
                        id='upload-search-cache',
                        children=html.Div(['📤 Importer cache recherches (JSON)']),
                        style={
                            'padding': '15px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '8px',
                            'textAlign': 'center',
                            'cursor': 'pointer'
                        },
                        multiple=False
                    ),
                    dcc.Upload(
                        id='upload-history-cache',
                        children=html.Div(['📤 Importer historique annonces (JSON)']),
                        style={
                            'padding': '15px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '8px',
                            'textAlign': 'center',
                            'cursor': 'pointer',
                            'marginTop': '10px'
                        },
                        multiple=False
                    )
                ]),
                html.Div(id='cache-feedback', style={'marginTop': '8px', 'color': COLORS['secondary']})
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

    # Graphiques comparatifs
    html.Div(id='comparison-container', style=hidden_section_style()),
    html.Div(id='trend-container', style=hidden_section_style()),

    # Watchlist matches
    html.Div(id='watchlist-matches', style=hidden_section_style()),
    
    # Liste des annonces
    html.Div(id='listings-container', style={
        'maxWidth': '1400px',
        'margin': '0 auto'
    }),
    
    # Stores pour les données et actions client-side
    dcc.Store(id='scraped-data'),
    dcc.Store(id='link-opener'),
    dcc.Store(id='watchlist-store', data=load_watchlist()),
    dcc.Download(id='download-search-cache'),
    dcc.Download(id='download-history-cache')
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
     Output('comparison-container', 'style'),
     Output('comparison-container', 'children'),
     Output('trend-container', 'style'),
     Output('trend-container', 'children'),
     Output('watchlist-matches', 'style'),
     Output('watchlist-matches', 'children'),
     Output('listings-container', 'children'),
     Output('history-selection', 'options')],
    [Input('btn-scrape', 'n_clicks'),
     Input('history-selection', 'value'),
     Input('watchlist-store', 'data'),
     Input('filter-price-min', 'value'),
     Input('filter-price-max', 'value'),
     Input('filter-mileage-min', 'value'),
     Input('filter-mileage-max', 'value'),
     Input('filter-score-min', 'value'),
     Input('filter-options', 'value'),
     Input('filter-condition', 'value')],
    [State('input-model', 'value'),
     State('input-year-min', 'value'),
     State('input-year-max', 'value'),
     State('input-pages', 'value'),
     State('use-cache-toggle', 'value'),
     State('scraped-data', 'data')]
)
def refresh_data_store(n_clicks, history_selection, watchlist_data,
                       price_min, price_max, mileage_min, mileage_max,
                       score_min, filter_options, condition_values,
                       model, year_min, year_max, pages,
                       use_cache_values, existing_data):
    history_selection = history_selection or []
    filter_options = filter_options or []
    filters = {
        'price_min': price_min,
        'price_max': price_max,
        'mileage_min': mileage_min,
        'mileage_max': mileage_max,
        'min_score': score_min,
        'options': filter_options,
        'conditions': condition_values
    }
    use_cache = bool(use_cache_values and 'use_cache' in use_cache_values)
    cleaner = DataCleaner()
    search_cache = load_search_cache()
    history_options = get_history_options(search_cache)
    ctx = callback_context
    triggered_prop = ctx.triggered[0]['prop_id'] if ctx and ctx.triggered else ""
    triggered_id = triggered_prop.split('.')[0] if triggered_prop else ""
    filter_inputs = {
        'watchlist-store',
        'filter-price-min',
        'filter-price-max',
        'filter-mileage-min',
        'filter-mileage-max',
        'filter-score-min',
        'filter-options',
        'filter-condition'
    }
    only_filters = triggered_id in filter_inputs
    perform_scrape = triggered_id == 'btn-scrape'
    
    existing_payload = existing_data or {}
    if only_filters:
        if not existing_payload.get('combined_records'):
            empty_outputs = build_visual_outputs(existing_payload, watchlist_data, filters)
            return existing_payload, "", *empty_outputs, history_options
        status_msg = build_status_message(existing_payload.get('status_entries', []))
        visuals = build_visual_outputs(existing_payload, watchlist_data, filters)
        return existing_payload, status_msg, *visuals, history_options
    current_records = existing_payload.get('current_records', [])
    current_search_key = existing_payload.get('current_search_key')
    current_label = existing_payload.get('current_label')
    
    if (n_clicks == 0 and not history_selection and not current_records) and not only_filters:
        empty_outputs = build_visual_outputs(existing_payload, watchlist_data, filters)
        return existing_payload, "", *empty_outputs, history_options
    
    if perform_scrape and (not model or not model.strip()):
        error = html.Div("⚠️ Veuillez entrer un modèle de moto", style={'color': '#f4212e'})
        empty_outputs = build_visual_outputs(existing_payload, watchlist_data, filters)
        return existing_payload, error, *empty_outputs, history_options
    
    datasets = []
    status_entries = existing_payload.get('status_entries', [])
    
    if perform_scrape:
        raw_data, entry, used_cache, search_key, search_cache = fetch_search_results(
            model, year_min, year_max, pages, use_cache, search_cache
        )
        history_options = get_history_options(search_cache)
        
        if use_cache and raw_data is None:
            warning = html.Div("⚠️ Aucun cache trouvé pour cette recherche. Décochez l'option ou lancez une autre recherche.", style={'color': '#f9a826'})
            empty_outputs = build_visual_outputs(existing_payload, watchlist_data, filters)
            return existing_payload, warning, *empty_outputs, history_options
        
        if not raw_data:
            error = html.Div("❌ Aucune annonce trouvée", style={'color': '#f4212e'})
            empty_outputs = build_visual_outputs(existing_payload, watchlist_data, filters)
            return existing_payload, error, *empty_outputs, history_options
        
        df_current = prepare_dataframe(raw_data, entry['label'], cleaner)
        if df_current.empty:
            error = html.Div("❌ Aucune donnée exploitable après nettoyage", style={'color': '#f4212e'})
            empty_outputs = build_visual_outputs(existing_payload, watchlist_data, filters)
            return existing_payload, error, *empty_outputs, history_options
        
        if not used_cache:
            update_listings_history(df_current, search_key, entry['label'], entry['timestamp'])
        
        datasets.append(df_current)
        current_records = df_current.to_dict('records')
        current_search_key = search_key
        current_label = entry['label']
        status_entries = [{
            'icon': "♻️ " if used_cache else "✅ ",
            'label': entry['label'],
            'count': len(df_current),
            'note': " (cache)" if used_cache else " (scraping en direct)"
        }]
    elif current_records:
        datasets.append(pd.DataFrame(current_records))
    
    # Ajoute les recherches historiques sélectionnées
    seen_keys = {current_search_key} if current_search_key else set()
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
        status_entries.append({
            'icon': "📚 ",
            'label': entry.get('label', 'Historique'),
            'count': len(df_hist),
            'note': " (cache historique)"
        })
    
    if not datasets:
        warning = html.Div("⚠️ Sélection vide. Lancez un scraping ou choisissez une recherche sauvegardée.", style={'color': '#f9a826'})
        empty_outputs = build_visual_outputs(existing_payload, watchlist_data, filters)
        return existing_payload, warning, *empty_outputs, history_options
    
    combined_df = pd.concat(datasets, ignore_index=True)
    if 'price' in combined_df.columns:
        combined_df = combined_df.sort_values('price').reset_index(drop=True)
    
    payload = {
        'combined_records': combined_df.to_dict('records'),
        'current_records': current_records,
        'current_search_key': current_search_key,
        'current_label': current_label,
        'history_keys': history_selection,
        'status_entries': status_entries
    }
    
    status_msg = build_status_message(status_entries)
    visuals = build_visual_outputs(payload, watchlist_data, filters)
    
    return (payload, status_msg, *visuals, history_options)


def create_scatter_plot(df):
    """Crée le graphique interactif Plotly avec superposition multi-sources."""
    if df.empty:
        return go.Figure()
    
    def build_hover(row):
        text = f"<b>{row.get('title', 'N/A')[:50]}</b><br>"
        text += f"💰 {format_value(row.get('price'), '€')}<br>"
        text += f"📏 {format_value(row.get('mileage'), 'km')}<br>"
        text += f"📅 {row.get('year', 'N/A')}<br>"
        text += f"📍 {row.get('location', 'N/A')}<br>"
        if row.get('market_score') is not None:
            text += f"⭐ Score: {row.get('market_score', 0):.1f}/100 ({row.get('score_label', '')})<br>"
        if row.get('condition_label'):
            text += f"🛠️ {row.get('condition_label')}<br>"
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
        text += f"💰 {format_value(row.get('price'), '€')}<br>"
        text += f"📏 {format_value(row.get('mileage'), 'km')}<br>"
        text += f"📍 {row.get('location', 'N/A')}<br>"
        if row.get('market_score') is not None:
            text += f"⭐ {row.get('market_score', 0):.1f}/100<br>"
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
        text += f"💰 {format_value(row.get('price'), '€')}<br>"
        duration = row.get('estimated_duration_days')
        duration_text = f"{int(duration)} jour(s) estimés" if duration and not pd.isna(duration) else "Durée inconnue"
        text += f"⏱️ {duration_text}<br>"
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


def create_trend_plot(history_df: pd.DataFrame):
    """Crée une courbe d'évolution des prix moyens."""
    if history_df is None or history_df.empty:
        return None
    history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
    recent = history_df[history_df['date'] >= (datetime.utcnow().date() - pd.Timedelta(days=60))]
    if recent.empty:
        recent = history_df
    grouped = recent.groupby(['date', 'source']).agg({'price': 'mean'}).reset_index()
    if grouped.empty:
        return None
    fig = go.Figure()
    for source in grouped['source'].unique():
        subset = grouped[grouped['source'] == source]
        fig.add_trace(go.Scatter(
            x=subset['date'],
            y=subset['price'],
            mode='lines+markers',
            name=source,
            line=dict(shape='spline')
        ))
    fig.update_layout(
        title="Tendance des prix moyens (historique)",
        xaxis_title="Date",
        yaxis_title="Prix moyen (€)",
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'])
    )
    return fig


def create_comparison_plot(df: pd.DataFrame):
    """Compare les distributions de prix par source."""
    if df.empty:
        return None
    if 'source' not in df.columns:
        df['source'] = 'Recherche'
    fig = go.Figure()
    for idx, source in enumerate(df['source'].unique()):
        subset = df[df['source'] == source]
        if subset.empty:
            continue
        fig.add_trace(go.Box(
            y=subset['price'],
            name=source,
            marker_color=TRACE_COLORS[idx % len(TRACE_COLORS)],
            boxmean=True
        ))
    fig.update_layout(
        title="Distribution des prix par recherche",
        yaxis_title="Prix (€)",
        plot_bgcolor=COLORS['card'],
        paper_bgcolor=COLORS['card'],
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
            html.Div([
                html.Span(row.get('source', 'Recherche'), style={
                    'fontSize': '0.85rem',
                    'color': COLORS['secondary'],
                    'marginRight': '8px'
                }),
                html.Span(
                    f"⭐ {row.get('market_score'):.1f}" if row.get('market_score') is not None else "⭐ N/A",
                    style={'fontSize': '0.85rem', 'color': COLORS['success']}
                )
            ], style={'marginBottom': '10px', 'display': 'flex', 'justifyContent': 'space-between'}),
            
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
                    html.Span(format_value(row.get('price'), '€'), style={'fontWeight': '700', 'fontSize': '1.3rem', 'color': COLORS['success']})
                ], style={'marginBottom': '8px'}),
                
                html.Div([
                    html.Span('📏 ', style={'marginRight': '5px'}),
                    html.Span(format_value(row.get('mileage'), 'km'))
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
                ], style={'color': COLORS['secondary']}),
                
                html.Div(f"État: {row.get('condition_label', 'Standard')}", style={'color': COLORS['secondary'], 'marginTop': '6px'})
            ]),
            
            # Tags watchlist
            html.Div([
                html.Span(tag, style={
                    'backgroundColor': COLORS['primary'],
                    'color': 'white',
                    'padding': '4px 8px',
                    'borderRadius': '999px',
                    'fontSize': '0.75rem',
                    'marginRight': '5px'
                }) for tag in row.get('watchlist_tags', [])
            ], style={'marginBottom': '10px'}) if row.get('watchlist_tags') else None,
            
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


@app.callback(
    [Output('watchlist-store', 'data'),
     Output('watchlist-feedback', 'children'),
     Output('watchlist-selection', 'options')],
    [Input('btn-watchlist-add', 'n_clicks'),
     Input('btn-watchlist-remove', 'n_clicks')],
    [State('watchlist-label', 'value'),
     State('watchlist-query', 'value'),
     State('watchlist-max-price', 'value'),
     State('watchlist-min-score', 'value'),
     State('watchlist-selection', 'value'),
     State('watchlist-store', 'data')]
)
def manage_watchlist(add_clicks, remove_clicks, label, query, max_price, min_score, selection, current_store):
    ctx = callback_context
    current_store = current_store or []
    feedback = ""
    triggered = ctx.triggered[0]['prop_id'] if ctx and ctx.triggered else ""
    updated_store = current_store
    
    if triggered == 'btn-watchlist-add.n_clicks':
        if not label or not query:
            feedback = html.Span("❌ Nom et mot-clé requis pour créer une alerte.", style={'color': '#f4212e'})
        else:
            new_entry = {
                'id': str(uuid4()),
                'label': label.strip(),
                'query': query.strip(),
                'max_price': max_price,
                'min_score': min_score
            }
            updated_store = current_store + [new_entry]
            save_watchlist(updated_store)
            feedback = html.Span("✅ Alerte ajoutée.", style={'color': COLORS['success']})
    elif triggered == 'btn-watchlist-remove.n_clicks':
        if not selection:
            feedback = html.Span("⚠️ Sélectionnez une alerte à supprimer.", style={'color': '#f9a826'})
        else:
            updated_store = [entry for entry in current_store if entry.get('id') != selection]
            save_watchlist(updated_store)
            feedback = html.Span("🗑️ Alerte supprimée.", style={'color': COLORS['secondary']})
    else:
        feedback = ""
    
    options = [{'label': entry.get('label', 'Alerte'), 'value': entry.get('id')} for entry in updated_store]
    return updated_store, feedback, options


@app.callback(
    Output('download-search-cache', 'data'),
    Input('btn-export-search', 'n_clicks'),
    prevent_initial_call=True
)
def export_search_cache_data(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    data = load_search_cache()
    return dict(
        content=json.dumps(data, ensure_ascii=False, indent=2),
        filename=f"search_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )


@app.callback(
    Output('download-history-cache', 'data'),
    Input('btn-export-history', 'n_clicks'),
    prevent_initial_call=True
)
def export_history_cache_data(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    data = load_json_file(LISTING_HISTORY_PATH, {})
    return dict(
        content=json.dumps(data, ensure_ascii=False, indent=2),
        filename=f"history_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )


@app.callback(
    Output('cache-feedback', 'children'),
    [Input('upload-search-cache', 'contents'),
     Input('upload-history-cache', 'contents')]
)
def import_caches(search_contents, history_contents):
    message_parts = []
    if search_contents:
        data = decode_uploaded_json(search_contents)
        if data is not None:
            save_json_file(SEARCH_CACHE_PATH, data)
            message_parts.append("✅ Cache recherches importé.")
        else:
            message_parts.append("❌ Fichier de cache recherches invalide.")
    if history_contents:
        data = decode_uploaded_json(history_contents)
        if data is not None:
            save_json_file(LISTING_HISTORY_PATH, data)
            message_parts.append("✅ Historique des annonces importé.")
        else:
            message_parts.append("❌ Fichier d'historique invalide.")
    if not message_parts:
        raise dash.exceptions.PreventUpdate
    return html.Span(" ".join(message_parts), style={'color': COLORS['secondary']})


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
