import pandas as pd
import re
from typing import List, Dict, Optional


class DataCleaner:
    """Nettoyage et structuration des données scrappées"""
    
    def __init__(self):
        self.required_fields = ['price', 'mileage']
    
    def clean_price(self, price_str: any) -> Optional[float]:
        """
        Nettoie et convertit le prix en float
        
        Args:
            price_str: Prix brut (peut être string, int, float)
            
        Returns:
            Prix en float ou None si invalide
        """
        if pd.isna(price_str) or price_str == '' or price_str is None:
            return None
        
        try:
            # Conversion en string
            price_str = str(price_str)
            
            # Suppression des caractères non numériques sauf . et ,
            price_str = re.sub(r'[^\d.,]', '', price_str)
            
            # Remplacement de la virgule par un point
            price_str = price_str.replace(',', '.')
            
            # Conversion en float
            price = float(price_str)
            
            # Validation: prix entre 500€ et 50000€
            if 500 <= price <= 50000:
                return price
            else:
                return None
                
        except (ValueError, AttributeError):
            return None
    
    def clean_mileage(self, mileage_str: any) -> Optional[float]:
        """
        Nettoie et convertit le kilométrage en float
        
        Args:
            mileage_str: Kilométrage brut
            
        Returns:
            Kilométrage en float ou None si invalide
        """
        if pd.isna(mileage_str) or mileage_str == '' or mileage_str is None:
            return None
        
        try:
            # Conversion en string
            mileage_str = str(mileage_str)
            
            # Suppression des caractères non numériques
            mileage_str = re.sub(r'[^\d]', '', mileage_str)
            
            if not mileage_str:
                return None
            
            # Conversion en float
            mileage = float(mileage_str)
            
            # Validation: km entre 0 et 150000
            if 0 <= mileage <= 150000:
                return mileage
            else:
                return None
                
        except (ValueError, AttributeError):
            return None
    
    def clean_year(self, year_str: any) -> Optional[int]:
        """
        Nettoie et convertit l'année en int
        
        Args:
            year_str: Année brute
            
        Returns:
            Année en int ou None si invalide
        """
        if pd.isna(year_str) or year_str == '' or year_str is None:
            return None
        
        try:
            # Si c'est déjà un int
            if isinstance(year_str, int):
                year = year_str
            else:
                # Extraction des 4 chiffres consécutifs
                year_str = str(year_str)
                match = re.search(r'\b(19\d{2}|20\d{2})\b', year_str)
                
                if match:
                    year = int(match.group(1))
                else:
                    return None
            
            # Validation: année entre 2000 et année actuelle + 1
            if 2000 <= year <= 2026:
                return year
            else:
                return None
                
        except (ValueError, AttributeError):
            return None
    
    def clean_location(self, location_str: any) -> str:
        """
        Nettoie la localisation
        
        Args:
            location_str: Localisation brute
            
        Returns:
            Localisation nettoyée
        """
        if pd.isna(location_str) or location_str == '' or location_str is None:
            return "Non spécifié"
        
        location = str(location_str).strip()
        
        # Limitation à 50 caractères
        if len(location) > 50:
            location = location[:47] + "..."
        
        return location
    
    def clean_title(self, title_str: any) -> str:
        """
        Nettoie le titre
        
        Args:
            title_str: Titre brut
            
        Returns:
            Titre nettoyé
        """
        if pd.isna(title_str) or title_str == '' or title_str is None:
            return "Sans titre"
        
        title = str(title_str).strip()
        
        # Suppression des doubles espaces
        title = re.sub(r'\s+', ' ', title)
        
        return title
    
    def clean_url(self, url_str: any) -> Optional[str]:
        """
        Nettoie et valide l'URL
        
        Args:
            url_str: URL brute
            
        Returns:
            URL nettoyée ou None
        """
        if pd.isna(url_str) or url_str == '' or url_str is None:
            return None
        
        url = str(url_str).strip()
        
        # Ajout du préfixe si manquant
        if url.startswith('/'):
            url = f"https://www.leboncoin.fr{url}"
        elif not url.startswith('http'):
            url = f"https://www.leboncoin.fr/{url}"
        
        return url
    
    def clean_photo_url(self, photo_str: any) -> Optional[str]:
        """
        Nettoie l'URL de la photo
        
        Args:
            photo_str: URL photo brute
            
        Returns:
            URL photo nettoyée ou None
        """
        if pd.isna(photo_str) or photo_str == '' or photo_str is None:
            return None
        
        photo = str(photo_str).strip()
        
        # Validation basique d'URL
        if photo.startswith('http'):
            return photo
        
        return None
    
    def clean(self, raw_data: List[Dict]) -> pd.DataFrame:
        """
        Nettoie l'ensemble des données
        
        Args:
            raw_data: Liste de dictionnaires bruts
            
        Returns:
            DataFrame pandas nettoyé
        """
        if not raw_data:
            print("⚠️ Aucune donnée à nettoyer")
            return pd.DataFrame()
        
        print(f"🧹 Nettoyage de {len(raw_data)} annonces brutes...")
        
        # Conversion en DataFrame
        df = pd.DataFrame(raw_data)
        
        print(f"📊 Colonnes disponibles: {list(df.columns)}")
        
        # Nettoyage des colonnes essentielles
        df['price_clean'] = df.get('price', pd.Series()).apply(self.clean_price)
        df['mileage_clean'] = df.get('mileage', pd.Series()).apply(self.clean_mileage)
        
        # Suppression des lignes sans prix ou kilométrage valide
        df_valid = df[df['price_clean'].notna() & df['mileage_clean'].notna()].copy()
        
        print(f"✅ {len(df_valid)} annonces valides après filtrage")
        
        if df_valid.empty:
            return pd.DataFrame()
        
        # Nettoyage des colonnes optionnelles
        df_valid['year'] = df_valid.get('year', pd.Series()).apply(self.clean_year)
        df_valid['location'] = df_valid.get('location', pd.Series()).apply(self.clean_location)
        df_valid['title'] = df_valid.get('title', pd.Series()).apply(self.clean_title)
        df_valid['link'] = df_valid.get('link', pd.Series()).apply(self.clean_url)
        df_valid['photo'] = df_valid.get('photo', pd.Series()).apply(self.clean_photo_url)
        
        # Renommage final
        df_final = df_valid[[
            'title', 'price_clean', 'mileage_clean', 
            'year', 'location', 'link', 'photo'
        ]].copy()
        
        df_final.columns = ['title', 'price', 'mileage', 'year', 'location', 'link', 'photo']
        
        # Tri par prix croissant
        df_final = df_final.sort_values('price').reset_index(drop=True)
        
        # Statistiques
        print(f"\n📈 Statistiques finales:")
        print(f"  Prix moyen: {df_final['price'].mean():,.0f} €")
        print(f"  Prix min/max: {df_final['price'].min():,.0f} € / {df_final['price'].max():,.0f} €")
        print(f"  Km moyen: {df_final['mileage'].mean():,.0f} km")
        print(f"  Années: {df_final['year'].min()}-{df_final['year'].max()}")
        print(f"  Photos disponibles: {df_final['photo'].notna().sum()}/{len(df_final)}\n")
        
        return df_final


# Test du module
if __name__ == "__main__":
    # Données de test
    test_data = [
        {
            'title': 'Triumph Street Triple 765 RS',
            'price': '10 500 €',
            'mileage': '15 000 km',
            'year': '2019',
            'location': 'Paris 75',
            'link': '/motos/123456.htm',
            'photo': 'https://example.com/photo.jpg'
        },
        {
            'title': 'Street Triple RS',
            'price': '9500',
            'mileage': '22000',
            'year': 2018,
            'location': 'Lyon',
            'link': 'https://www.leboncoin.fr/motos/789012.htm',
            'photo': None
        },
        {
            'title': 'Moto sans prix',
            'price': None,
            'mileage': '10000',
            'year': '2020'
        }
    ]
    
    cleaner = DataCleaner()
    df = cleaner.clean(test_data)
    
    print("\n📋 DataFrame final:")
    print(df)