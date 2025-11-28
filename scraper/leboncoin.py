from firecrawl import Firecrawl
import time
import os
from urllib.parse import quote_plus
from typing import List, Dict, Optional


class LeboncoinScraper:
    """Scraper pour les annonces Leboncoin via Firecrawl"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le scraper Firecrawl
        
        Args:
            api_key: Clé API Firecrawl (ou via variable d'environnement FIRECRAWL_API_KEY)
        """
        if api_key is None:
            api_key = os.getenv('FIRECRAWL_API_KEY', 'fc-779c2dbbb7264862a12028574a977e53')
        
        self.fc = Firecrawl(api_key=api_key)
        self.base_url = "https://www.leboncoin.fr/recherche"
        
    def build_url(self, model: str, year_min: int, year_max: int, page: int = 1) -> str:
        """
        Construit l'URL de recherche Leboncoin
        
        Args:
            model: Modèle de moto (ex: "triumph street triple 765 rs")
            year_min: Année minimum
            year_max: Année maximum
            page: Numéro de page
            
        Returns:
            URL complète
        """
        params = {
            'category': '3',  # Catégorie motos
            'text': model.strip(),
            'regdate': f"{year_min}-{year_max}",
            'moto_type': 'moto'
        }
        
        # Construction manuelle de l'URL pour plus de contrôle
        query_string = f"category={params['category']}"
        query_string += f"&text={quote_plus(params['text'])}"
        query_string += f"&regdate={params['regdate']}"
        query_string += f"&moto_type={params['moto_type']}"
        
        if page > 1:
            query_string += f"&page={page}"
        
        return f"{self.base_url}?{query_string}"
    
    def scrape_page(self, url: str) -> List[Dict]:
        """
        Scrape une page d'annonces
        
        Args:
            url: URL à scraper
            
        Returns:
            Liste de dictionnaires contenant les données d'annonces
        """
        try:
            print(f"📡 Scraping: {url}")
            
            # Schema enrichi pour extraire plus de données
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Le titre complet de l'annonce"
                        },
                        "price": {
                            "type": "string",
                            "description": "Le prix en euros (avec ou sans le symbole €)"
                        },
                        "mileage": {
                            "type": "string",
                            "description": "Le kilométrage (avec ou sans 'km')"
                        },
                        "year": {
                            "type": ["string", "integer"],
                            "description": "L'année de mise en circulation"
                        },
                        "location": {
                            "type": "string",
                            "description": "La ville ou localisation"
                        },
                        "link": {
                            "type": "string",
                            "description": "Le lien complet vers l'annonce"
                        },
                        "photo": {
                            "type": "string",
                            "description": "L'URL de la première photo de l'annonce"
                        }
                    },
                    "required": ["price", "mileage"]
                }
            }
            
            doc = self.fc.scrape(
                url,
                formats=[{
                    "type": "json",
                    "prompt": """Extract all motorcycle ads from this Leboncoin page.
                    For each ad, extract:
                    - title (full listing title)
                    - price (in euros)
                    - mileage (in km)
                    - year (registration year)
                    - location (city)
                    - link (full URL to the ad)
                    - photo (URL of the main image)
                    
                    Return an array of all ads found on the page.""",
                    "schema": schema
                }]
            )
            
            # Extraction des données
            if hasattr(doc, 'json') and doc.json:
                ads = doc.json if isinstance(doc.json, list) else [doc.json]
                print(f"✅ {len(ads)} annonces extraites")
                return ads
            else:
                print("⚠️ Aucune donnée JSON retournée par Firecrawl")
                return []
                
        except Exception as e:
            print(f"❌ Erreur lors du scraping: {str(e)}")
            return []
    
    def scrape(self, model: str, year_min: int, year_max: int, max_pages: int = 3) -> List[Dict]:
        """
        Scrape plusieurs pages d'annonces
        
        Args:
            model: Modèle de moto
            year_min: Année minimum
            year_max: Année maximum
            max_pages: Nombre maximum de pages à scraper
            
        Returns:
            Liste complète des annonces
        """
        all_ads = []
        
        print(f"\n🔍 Recherche: {model} ({year_min}-{year_max})")
        print(f"📄 Pages à scanner: {max_pages}\n")
        
        for page in range(1, max_pages + 1):
            url = self.build_url(model, year_min, year_max, page)
            ads = self.scrape_page(url)
            
            if ads:
                all_ads.extend(ads)
                print(f"📊 Total accumulé: {len(all_ads)} annonces\n")
            else:
                print(f"⚠️ Page {page} vide ou erreur, arrêt du scraping\n")
                break
            
            # Pause entre les pages pour éviter la surcharge
            if page < max_pages:
                time.sleep(1)
        
        print(f"✅ Scraping terminé: {len(all_ads)} annonces au total\n")
        return all_ads
    
    def scrape_with_retry(self, model: str, year_min: int, year_max: int, 
                          max_pages: int = 3, max_retries: int = 3) -> List[Dict]:
        """
        Scrape avec mécanisme de retry en cas d'échec
        
        Args:
            model: Modèle de moto
            year_min: Année minimum
            year_max: Année maximum
            max_pages: Nombre maximum de pages
            max_retries: Nombre maximum de tentatives
            
        Returns:
            Liste des annonces
        """
        for attempt in range(1, max_retries + 1):
            try:
                ads = self.scrape(model, year_min, year_max, max_pages)
                if ads:
                    return ads
                
                if attempt < max_retries:
                    print(f"⚠️ Tentative {attempt}/{max_retries} échouée, nouvelle tentative dans 3s...")
                    time.sleep(3)
                    
            except Exception as e:
                print(f"❌ Erreur tentative {attempt}/{max_retries}: {str(e)}")
                if attempt < max_retries:
                    time.sleep(3)
        
        print("❌ Échec après toutes les tentatives")
        return []


# Test du module
if __name__ == "__main__":
    scraper = LeboncoinScraper()
    
    # Test avec la Street Triple RS
    results = scraper.scrape(
        model="triumph street triple 765 rs",
        year_min=2017,
        year_max=2020,
        max_pages=2
    )
    
    print("\n📋 Résumé des résultats:")
    print(f"Nombre d'annonces: {len(results)}")
    
    if results:
        print("\n🔍 Première annonce:")
        for key, value in results[0].items():
            print(f"  {key}: {value}")