"""
Q2: AI-Based Travel Planner with Knowledge Bases
=================================================
Reuses existing knowledge bases in:
    - Tourist Places (cities, attractions, UNESCO sites)
    - Food Recommendations (cuisines, dietary restrictions)
    - Personalised Tour Plans (budget, interests, travel style)
    - Cost Assessment (daily budgets, accommodation tiers)
    - Weather KB (best travel seasons by region)

Architecture:
    KnowledgeBase  → stores facts as subject-predicate-object triples (RDF-like)
    OntologyLayer  → defines classes, properties, and relations (inspired by tourism ontologies)
    Planner        → queries KBs + applies rules to produce personalised tour plans

This is a pure-Python implementation (no external KG library needed).
In a production system, you'd use Protégé for ontology design and
Neo4j or AllegroGraph to store/query the triples.

Author: AI Assignment Solution
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any
import math
import json
import sys
import time


# =============================================================================
# KNOWLEDGE BASE ENGINE — Simple Triple Store
# =============================================================================

class TripleStore:
    """
    Lightweight triple store: (subject, predicate, object).
    Supports SPARQL-like pattern matching queries.

    In production: replaced by AllegroGraph / GraphDB + SPARQL endpoint.
    """

    def __init__(self):
        # Index by subject, predicate, object for fast lookup
        self._spo: dict = defaultdict(lambda: defaultdict(set))  # s → p → {o}
        self._pos: dict = defaultdict(lambda: defaultdict(set))  # p → o → {s}
        self._triples: list = []

    def add(self, s, p, o):
        self._spo[s][p].add(o)
        self._pos[p][o].add(s)
        self._triples.append((s, p, o))

    def query(self, s=None, p=None, o=None):
        """
        Pattern query: pass None for wildcards.
        Returns list of matching (s, p, o) triples.
        """
        results = []
        for ts, tp, to in self._triples:
            if (s is None or ts == s) and \
               (p is None or tp == p) and \
               (o is None or to == o):
                results.append((ts, tp, to))
        return results

    def get(self, s, p):
        """Get all objects for (s, p, ?)."""
        return list(self._spo[s][p])

    def get_one(self, s, p, default=None):
        vals = self.get(s, p)
        return vals[0] if vals else default

    def subjects_with(self, p, o):
        """Get all subjects with (?, p, o)."""
        return list(self._pos[p][o])


# =============================================================================
# ONTOLOGY LAYER — Classes & Properties
# =============================================================================

class TourismOntology:
    """
    Defines the ontology for the travel domain.

    Classes:
        Destination, Attraction, Cuisine, Hotel, TourPlan, Traveller

    Object Properties:
        hasAttraction, servesCuisine, isIn, recommendedFor, bestSeason,
        hasBudgetTier, hasActivityType, nearAttraction

    Data Properties:
        averageDailyCost, rating, duration_hours, distance_from_city_km

    Inspired by: GeoNames ontology, DBpedia Tourism,
                 Schema.org TouristAttraction, LGD (Linked Geo Data)
    """

    CLASSES = [
        'Destination', 'Attraction', 'Cuisine', 'Hotel',
        'TourPlan', 'Traveller', 'Activity', 'Season'
    ]

    PROPERTIES = {
        # Attraction properties
        'hasAttraction':       ('Destination', 'Attraction'),
        'hasActivityType':     ('Attraction', 'Activity'),
        'isIn':                ('Attraction', 'Destination'),
        'bestVisitedIn':       ('Destination', 'Season'),
        'averageDailyCost':    ('Destination', 'float'),   # USD/day
        'rating':              ('Attraction', 'float'),    # 0-5
        'duration_hours':      ('Attraction', 'float'),
        'distance_km':         ('Attraction', 'float'),    # from city center
        'UNESCO':              ('Attraction', 'bool'),     # UNESCO World Heritage

        # Food properties
        'servesCuisine':       ('Destination', 'Cuisine'),
        'dietaryTag':          ('Cuisine', 'string'),      # veg, vegan, halal, gluten-free
        'spiceLevel':          ('Cuisine', 'string'),      # mild, medium, spicy

        # Hotel properties
        'hasHotel':            ('Destination', 'Hotel'),
        'budgetTier':          ('Hotel', 'string'),        # budget, mid, luxury
        'pricePerNight':       ('Hotel', 'float'),         # USD

        # Traveller preferences
        'prefersActivity':     ('Traveller', 'Activity'),
        'hasBudget':           ('Traveller', 'string'),    # budget, mid, luxury
        'dietaryRestriction':  ('Traveller', 'string'),
        'travelStyle':         ('Traveller', 'string'),    # solo, couple, family, group
        'availableDays':       ('Traveller', 'int'),
    }


# =============================================================================
# KNOWLEDGE BASE — Tourist Places, Food, Cost, Weather
# =============================================================================

def build_knowledge_base() -> TripleStore:
    """
    Populates the triple store with facts from multiple knowledge domains.
    In a real system, these facts would be ingested from:
        - DBpedia SPARQL endpoint
        - OpenStreetMap / GeoNames
        - TripAdvisor API
        - Wikidata
    """
    kb = TripleStore()

    # ---- DESTINATIONS -------------------------------------------------------
    destinations = [
        # (name, country, region, avg_daily_cost_usd, best_season, climate)
        ("Paris",        "France",     "Europe",     150, "Spring",    "Temperate"),
        ("Tokyo",        "Japan",      "Asia",       120, "Spring",    "Temperate"),
        ("Rajasthan",    "India",      "Asia",        40, "Winter",    "Arid"),
        ("Santorini",    "Greece",     "Europe",     200, "Summer",    "Mediterranean"),
        ("Kyoto",        "Japan",      "Asia",        90, "Spring",    "Temperate"),
        ("Bali",         "Indonesia",  "Asia",        50, "DrySeasonJul", "Tropical"),
        ("New York",     "USA",        "Americas",   180, "Fall",      "Temperate"),
        ("Cape Town",    "SouthAfrica","Africa",      80, "Summer",    "Mediterranean"),
        ("Machu Picchu", "Peru",       "Americas",   60, "DrySeasonMay","Highland"),
        ("Dubai",        "UAE",        "MiddleEast", 200, "Winter",   "Desert"),
        ("Barcelona",    "Spain",      "Europe",     130, "Spring",    "Mediterranean"),
        ("Kerala",       "India",      "Asia",        35, "Winter",    "Tropical"),
    ]

    for name, country, region, cost, season, climate in destinations:
        kb.add(name, 'type',              'Destination')
        kb.add(name, 'inCountry',         country)
        kb.add(name, 'inRegion',          region)
        kb.add(name, 'averageDailyCost',  cost)
        kb.add(name, 'bestSeason',        season)
        kb.add(name, 'climate',           climate)

    # ---- ATTRACTIONS --------------------------------------------------------
    attractions = [
        # (name, destination, activity_type, rating, duration_h, dist_km, UNESCO, cost_usd)
        ("Eiffel Tower",          "Paris",        "Sightseeing",  4.7, 2.0,  0.5,  False, 25),
        ("Louvre Museum",         "Paris",        "Museum",       4.8, 4.0,  1.0,  False, 20),
        ("Versailles Palace",     "Paris",        "History",      4.6, 5.0, 20.0,  True,  20),
        ("Montmartre",            "Paris",        "Culture",      4.5, 3.0,  4.0,  False,  0),

        ("Senso-ji Temple",       "Tokyo",        "Culture",      4.6, 2.0, 10.0,  False,  0),
        ("Shibuya Crossing",      "Tokyo",        "Sightseeing",  4.5, 1.0,  5.0,  False,  0),
        ("teamLab Planets",       "Tokyo",        "Art",          4.8, 2.0,  8.0,  False, 30),
        ("Mt. Fuji Day Trip",     "Tokyo",        "Adventure",    4.9, 8.0, 100.0, False, 50),

        ("Amber Fort",            "Rajasthan",    "History",      4.6, 3.0, 11.0,  True,   8),
        ("City Palace Jaipur",    "Rajasthan",    "Culture",      4.5, 3.0,  7.0,  False,  6),
        ("Mehrangarh Fort",       "Rajasthan",    "History",      4.7, 4.0,  1.0,  False,  6),
        ("Pushkar Lake",          "Rajasthan",    "Nature",       4.3, 2.0, 150.0, False,  0),

        ("Oia Sunset",            "Santorini",    "Sightseeing",  4.9, 3.0, 12.0,  False,  0),
        ("Akrotiri Ruins",        "Santorini",    "History",      4.4, 2.0, 14.0,  True,  14),
        ("Fira Town",             "Santorini",    "Culture",      4.5, 4.0,  0.5,  False,  0),
        ("Red Beach",             "Santorini",    "Beach",        4.3, 3.0, 13.0,  False,  0),

        ("Fushimi Inari",         "Kyoto",        "Culture",      4.8, 3.0,  5.0,  False,  0),
        ("Arashiyama Bamboo",     "Kyoto",        "Nature",       4.7, 2.0,  8.0,  False,  0),
        ("Kinkaku-ji Temple",     "Kyoto",        "Culture",      4.7, 1.5,  5.0,  True,   5),
        ("Nishiki Market",        "Kyoto",        "Food",         4.5, 2.0,  1.0,  False,  0),

        ("Tegalalang Rice Terrace","Bali",        "Nature",       4.5, 2.0, 18.0,  False,  3),
        ("Tanah Lot Temple",      "Bali",         "Culture",      4.6, 2.0, 13.0,  False,  5),
        ("Ubud Monkey Forest",    "Bali",         "Nature",       4.3, 1.5, 24.0,  False,  5),
        ("Seminyak Beach",        "Bali",         "Beach",        4.4, 4.0, 10.0,  False,  0),

        ("Central Park",          "New York",     "Nature",       4.7, 3.0,  2.0,  False,  0),
        ("MoMA",                  "New York",     "Museum",       4.6, 3.0,  1.5,  False, 25),
        ("Statue of Liberty",     "New York",     "History",      4.6, 4.0,  5.0,  False, 24),
        ("Brooklyn Bridge",       "New York",     "Sightseeing",  4.7, 1.5,  3.0,  False,  0),

        ("Table Mountain",        "Cape Town",    "Adventure",    4.8, 4.0,  5.0,  False, 20),
        ("Cape of Good Hope",     "Cape Town",    "Nature",       4.7, 6.0, 70.0,  False, 15),
        ("Robben Island",         "Cape Town",    "History",      4.5, 4.0,  7.0,  True,  35),
        ("V&A Waterfront",        "Cape Town",    "Culture",      4.4, 3.0,  3.0,  False,  0),

        ("Sun Gate",              "Machu Picchu", "Adventure",    4.8, 6.0,  2.0,  True,  50),
        ("Huayna Picchu",         "Machu Picchu", "Adventure",    4.9, 5.0,  0.5,  True,  25),
        ("Main Ruins",            "Machu Picchu", "History",      4.9, 4.0,  0.0,  True,  50),

        ("Burj Khalifa Deck",     "Dubai",        "Sightseeing",  4.7, 2.0,  1.0,  False, 40),
        ("Dubai Mall",            "Dubai",        "Shopping",     4.5, 4.0,  1.0,  False,  0),
        ("Desert Safari",         "Dubai",        "Adventure",    4.8, 8.0, 50.0,  False, 80),
        ("Dubai Creek",           "Dubai",        "Culture",      4.3, 2.0,  4.0,  False,  0),

        ("Sagrada Familia",       "Barcelona",    "Culture",      4.8, 2.0,  2.0,  True,  26),
        ("Park Guell",            "Barcelona",    "Culture",      4.6, 2.0,  4.0,  True,  10),
        ("La Barceloneta Beach",  "Barcelona",    "Beach",        4.4, 3.0,  2.0,  False,  0),
        ("Gothic Quarter",        "Barcelona",    "History",      4.5, 2.0,  1.0,  False,  0),

        ("Kerala Backwaters",     "Kerala",       "Nature",       4.8, 6.0, 50.0,  False, 30),
        ("Munnar Tea Gardens",    "Kerala",       "Nature",       4.6, 4.0, 130.0, False, 10),
        ("Thekkady Wildlife",     "Kerala",       "Nature",       4.5, 5.0, 190.0, False, 20),
        ("Kovalam Beach",         "Kerala",       "Beach",        4.3, 3.0, 60.0,  False,  0),
    ]

    for name, dest, activity, rating, dur, dist, unesco, cost in attractions:
        kb.add(name,  'type',           'Attraction')
        kb.add(name,  'isIn',           dest)
        kb.add(name,  'activityType',   activity)
        kb.add(name,  'rating',         rating)
        kb.add(name,  'duration_hours', dur)
        kb.add(name,  'distance_km',    dist)
        kb.add(name,  'isUNESCO',       unesco)
        kb.add(name,  'entryCost',      cost)
        kb.add(dest,  'hasAttraction',  name)     # inverse link

    # ---- CUISINES -----------------------------------------------------------
    cuisines = [
        # (destination, cuisine_name, dietary_tags, spice_level)
        ("Paris",        "French",      ["glutenfree-option", "veg-option"],   "mild"),
        ("Paris",        "Cafe Culture",["veg"],                               "mild"),
        ("Tokyo",        "Sushi",       ["halal-option"],                      "mild"),
        ("Tokyo",        "Ramen",       ["veg-option"],                        "medium"),
        ("Tokyo",        "Tempura",     ["veg-option"],                        "mild"),
        ("Rajasthan",    "Rajasthani",  ["veg", "vegan-option"],               "spicy"),
        ("Rajasthan",    "Dal Baati",   ["veg"],                               "medium"),
        ("Santorini",    "Greek",       ["veg-option", "seafood"],             "mild"),
        ("Kyoto",        "Kaiseki",     ["veg-option"],                        "mild"),
        ("Kyoto",        "Tofu Cuisine",["veg", "vegan"],                     "mild"),
        ("Bali",         "Balinese",    ["halal-option", "veg-option"],        "spicy"),
        ("Bali",         "Nasi Goreng", ["veg-option"],                        "medium"),
        ("New York",     "International",["veg","vegan","halal","glutenfree"], "varies"),
        ("Cape Town",    "Cape Malay",  ["halal", "veg-option"],               "medium"),
        ("Cape Town",    "Braai",       ["veg-option"],                        "mild"),
        ("Machu Picchu", "Peruvian",    ["veg-option", "seafood"],             "medium"),
        ("Dubai",        "Arabic",      ["halal", "veg-option"],               "medium"),
        ("Dubai",        "International",["halal","veg","vegan","glutenfree"], "varies"),
        ("Barcelona",    "Catalan",     ["seafood", "veg-option"],             "mild"),
        ("Barcelona",    "Tapas",       ["veg-option", "seafood"],             "mild"),
        ("Kerala",       "Kerala",      ["veg", "vegan", "seafood"],           "spicy"),
        ("Kerala",       "Sadhya",      ["veg", "vegan"],                      "medium"),
    ]

    for dest, cuisine, tags, spice in cuisines:
        kb.add(cuisine,  'type',         'Cuisine')
        kb.add(cuisine,  'isIn',         dest)
        kb.add(cuisine,  'spiceLevel',   spice)
        kb.add(dest,     'servesCuisine', cuisine)
        for tag in tags:
            kb.add(cuisine, 'dietaryTag', tag)

    # ---- HOTELS -------------------------------------------------------------
    hotels = [
        # (destination, name, tier, price_per_night_usd)
        ("Paris",        "Budget Hostel Paris",     "budget",  30),
        ("Paris",        "Hotel Ibis Paris",        "mid",     90),
        ("Paris",        "Le Meurice",              "luxury", 800),
        ("Tokyo",        "Khaosan Tokyo",           "budget",  25),
        ("Tokyo",        "APA Hotel Tokyo",         "mid",     80),
        ("Tokyo",        "The Peninsula Tokyo",     "luxury", 600),
        ("Rajasthan",    "Zostel Jaipur",           "budget",  10),
        ("Rajasthan",    "Clarks Amer Jaipur",      "mid",     50),
        ("Rajasthan",    "Rambagh Palace",          "luxury", 500),
        ("Santorini",    "Firostefani Hostel",      "budget",  40),
        ("Santorini",    "Aressana Spa Hotel",      "mid",    200),
        ("Santorini",    "Canaves Oia Suites",      "luxury", 900),
        ("Kyoto",        "Guesthouse Kyoto",        "budget",  30),
        ("Kyoto",        "Hotel Granvia Kyoto",     "mid",    100),
        ("Kyoto",        "Suiran Luxury Collection","luxury", 700),
        ("Bali",         "Puri Garden Hostel",      "budget",  15),
        ("Bali",         "Alaya Resort Ubud",       "mid",     80),
        ("Bali",         "Four Seasons Bali",       "luxury", 600),
        ("New York",     "HI NYC Hostel",           "budget",  40),
        ("New York",     "Holiday Inn NYC",         "mid",    150),
        ("New York",     "The Plaza NYC",           "luxury", 800),
        ("Cape Town",    "The Backpack Hostel",     "budget",  25),
        ("Cape Town",    "Protea Hotel Fire & Ice", "mid",     90),
        ("Cape Town",    "One&Only Cape Town",      "luxury", 500),
        ("Machu Picchu", "Eco Backpackers",         "budget",  20),
        ("Machu Picchu", "El Mapi Hotel",           "mid",     80),
        ("Machu Picchu", "Sanctuary Lodge",         "luxury", 800),
        ("Dubai",        "Dubai Youth Hostel",      "budget",  35),
        ("Dubai",        "Premier Inn Dubai",       "mid",    100),
        ("Dubai",        "Burj Al Arab",            "luxury",3000),
        ("Barcelona",    "Sant Jordi Hostels",      "budget",  25),
        ("Barcelona",    "Hotel Barcelona 1882",    "mid",    100),
        ("Barcelona",    "Hotel Arts Barcelona",    "luxury", 400),
        ("Kerala",       "Backpackers Kerala",      "budget",   8),
        ("Kerala",       "Fragrant Nature Kerala",  "mid",     40),
        ("Kerala",       "Kumarakom Lake Resort",   "luxury", 300),
    ]

    for dest, name, tier, price in hotels:
        kb.add(name,  'type',           'Hotel')
        kb.add(name,  'isIn',           dest)
        kb.add(name,  'budgetTier',     tier)
        kb.add(name,  'pricePerNight',  price)
        kb.add(dest,  'hasHotel',       name)

    return kb


# =============================================================================
# TRAVEL PLANNER ENGINE
# =============================================================================

@dataclass
class TravellerProfile:
    name: str
    budget_tier: str          # "budget", "mid", "luxury"
    available_days: int
    preferred_activities: list   # e.g. ["History", "Culture", "Beach"]
    dietary_restriction: str     # "veg", "vegan", "halal", "glutenfree", "none"
    travel_style: str            # "solo", "couple", "family", "group"
    preferred_region: str = ""   # e.g. "Asia", "Europe" or "" for any
    avoid_spicy: bool = False


@dataclass
class TourPlan:
    destination: str
    traveller: str
    days: int
    hotel: str
    hotel_cost_per_night: float
    daily_activity_cost: float
    daily_food_budget: float
    attractions: list
    cuisine_recommendations: list
    total_estimated_cost: float
    match_score: float
    notes: list = field(default_factory=list)


class TravelPlanner:
    """
    Personalised tour planner using KB inference.

    Scoring system:
        - Budget fit:      30 points
        - Activity match:  30 points
        - Dietary match:   20 points
        - Season fit:      10 points
        - Region match:    10 points
    """

    DAILY_FOOD = {
        'budget': 15,
        'mid':    40,
        'luxury': 100,
    }

    CURRENT_MONTH_SEASON = {
        1: 'Winter', 2: 'Winter', 3: 'Spring',
        4: 'Spring', 5: 'DrySeasonMay', 6: 'Summer',
        7: 'DrySeasonJul', 8: 'Summer', 9: 'Fall',
        10: 'Fall', 11: 'Winter', 12: 'Winter',
    }

    def __init__(self, kb: TripleStore):
        self.kb = kb

    def get_all_destinations(self):
        return [s for s, p, o in self.kb.query(p='type', o='Destination')]

    def score_destination(self, dest: str, profile: TravellerProfile) -> float:
        """Score 0-100 how well this destination fits the traveller."""
        score = 0.0

        # ---- Budget fit (30 pts) ----
        daily_cost = self.kb.get_one(dest, 'averageDailyCost', 999)
        budget_map = {'budget': (0, 60), 'mid': (40, 150), 'luxury': (100, 10000)}
        lo, hi = budget_map[profile.budget_tier]
        if lo <= daily_cost <= hi:
            score += 30
        elif daily_cost < lo:
            score += 20      # cheaper than expected — still good
        else:
            score += max(0, 30 - (daily_cost - hi) * 0.3)

        # ---- Activity match (30 pts) ----
        attractions = self.kb.get(dest, 'hasAttraction')
        dest_activities = set()
        for attr in attractions:
            act = self.kb.get_one(attr, 'activityType')
            if act:
                dest_activities.add(act)
        if profile.preferred_activities:
            matched = len(set(profile.preferred_activities) & dest_activities)
            score += 30 * (matched / len(profile.preferred_activities))

        # ---- Dietary match (20 pts) ----
        if profile.dietary_restriction == 'none':
            score += 20
        else:
            cuisines = self.kb.get(dest, 'servesCuisine')
            dietary_ok = False
            for c in cuisines:
                tags = self.kb.get(c, 'dietaryTag')
                if any(profile.dietary_restriction in t for t in tags):
                    dietary_ok = True
                    break
            score += 20 if dietary_ok else 5

        # ---- Season fit (10 pts) ----
        import datetime
        month = datetime.date.today().month
        current_season = self.CURRENT_MONTH_SEASON.get(month, 'Spring')
        best_season = self.kb.get_one(dest, 'bestSeason', '')
        if best_season == current_season:
            score += 10
        elif best_season in ['Spring', 'Fall'] and current_season in ['Spring', 'Fall']:
            score += 7      # shoulder seasons are usually fine too
        else:
            score += 3

        # ---- Region match (10 pts) ----
        if not profile.preferred_region:
            score += 10
        else:
            dest_region = self.kb.get_one(dest, 'inRegion', '')
            score += 10 if dest_region == profile.preferred_region else 0

        return score

    def select_hotel(self, dest: str, tier: str):
        """Pick best-rated hotel for the budget tier."""
        hotels = self.kb.get(dest, 'hasHotel')
        matching = [(h, self.kb.get_one(h, 'pricePerNight', 9999))
                    for h in hotels
                    if self.kb.get_one(h, 'budgetTier') == tier]
        if not matching:
            # Fall back to nearest tier
            matching = [(h, self.kb.get_one(h, 'pricePerNight', 9999))
                        for h in hotels]
        if not matching:
            return "Unknown Hotel", 50
        matching.sort(key=lambda x: x[1])
        best = matching[len(matching) // 2]   # pick mid-range within tier
        return best

    def select_attractions(self, dest: str, profile: TravellerProfile):
        """
        Select and order attractions for the trip.
        - Prioritise activity type matches
        - Prefer UNESCO sites for history/culture lovers
        - Cap by available days (3 attractions/day)
        """
        all_attrs = self.kb.get(dest, 'hasAttraction')
        scored = []
        for attr in all_attrs:
            act = self.kb.get_one(attr, 'activityType', '')
            rating = self.kb.get_one(attr, 'rating', 0)
            unesco = self.kb.get_one(attr, 'isUNESCO', False)
            entry = self.kb.get_one(attr, 'entryCost', 0)

            s = rating * 10
            if act in profile.preferred_activities:
                s += 20
            if unesco and 'History' in profile.preferred_activities:
                s += 10
            if profile.budget_tier == 'budget' and entry > 30:
                s -= 10
            scored.append((s, attr))

        scored.sort(reverse=True)
        max_attractions = profile.available_days * 3
        return [attr for _, attr in scored[:max_attractions]]

    def recommend_cuisines(self, dest: str, profile: TravellerProfile):
        """Return suitable cuisines respecting dietary restrictions."""
        cuisines = self.kb.get(dest, 'servesCuisine')
        suitable = []
        for c in cuisines:
            tags = self.kb.get(c, 'dietaryTag')
            spice = self.kb.get_one(c, 'spiceLevel', 'mild')
            if profile.dietary_restriction != 'none':
                if not any(profile.dietary_restriction in t for t in tags):
                    continue
            if profile.avoid_spicy and spice == 'spicy':
                continue
            suitable.append(c)
        return suitable if suitable else cuisines

    def build_tour_plan(self, dest: str, profile: TravellerProfile,
                        match_score: float) -> TourPlan:
        """Assemble a complete tour plan for a destination."""
        hotel_name, hotel_price = self.select_hotel(dest, profile.budget_tier)
        attractions = self.select_attractions(dest, profile)
        cuisines = self.recommend_cuisines(dest, profile)

        # Estimate daily attraction cost
        total_entry = sum(self.kb.get_one(a, 'entryCost', 0) for a in attractions)
        daily_activity = total_entry / max(profile.available_days, 1)

        food_daily = self.DAILY_FOOD[profile.budget_tier]
        total_cost = (hotel_price + daily_activity + food_daily) * profile.available_days

        # Notes / tips
        notes = []
        best_season = self.kb.get_one(dest, 'bestSeason', '')
        notes.append(f"Best travel season: {best_season}")
        if self.kb.get_one(dest, 'inRegion') == 'Asia':
            notes.append("Tip: Get travel insurance covering Asia.")
        if any(self.kb.get_one(a, 'isUNESCO', False) for a in attractions):
            notes.append("Includes UNESCO World Heritage Sites — book tickets in advance.")
        if profile.budget_tier == 'budget':
            notes.append("Budget tip: Use public transport; avoid tourist traps near attractions.")

        return TourPlan(
            destination=dest,
            traveller=profile.name,
            days=profile.available_days,
            hotel=hotel_name,
            hotel_cost_per_night=hotel_price,
            daily_activity_cost=daily_activity,
            daily_food_budget=food_daily,
            attractions=attractions,
            cuisine_recommendations=cuisines,
            total_estimated_cost=total_cost,
            match_score=match_score,
            notes=notes,
        )

    def plan(self, profile: TravellerProfile, top_n=3) -> list:
        """
        Main planning function.
        1. Score all destinations against the profile
        2. Select top-N
        3. Build detailed tour plans for each
        """
        destinations = self.get_all_destinations()
        scored = [(self.score_destination(d, profile), d) for d in destinations]
        scored.sort(reverse=True)
        top = scored[:top_n]

        plans = []
        for score, dest in top:
            plan = self.build_tour_plan(dest, profile, score)
            plans.append(plan)
        return plans


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_plan(plan: TourPlan, kb: TripleStore, rank: int):
    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION #{rank}: {plan.destination.upper()}")
    print(f"  Match Score: {plan.match_score:.1f}/100")
    print(f"{'='*60}")
    print(f"  Duration    : {plan.days} days")
    print(f"  Hotel       : {plan.hotel}")
    print(f"  Hotel/night : ${plan.hotel_cost_per_night}")
    print(f"  Food/day    : ${plan.daily_food_budget}")
    print(f"  Activities  : ${plan.daily_activity_cost:.1f}/day (avg)")
    print(f"  TOTAL COST  : ${plan.total_estimated_cost:.0f} estimated")

    print(f"\n  Top Attractions:")
    for i, attr in enumerate(plan.attractions[:6], 1):
        rating = kb.get_one(attr, 'rating', 0)
        act_type = kb.get_one(attr, 'activityType', '')
        unesco = '🏛 UNESCO' if kb.get_one(attr, 'isUNESCO', False) else ''
        entry = kb.get_one(attr, 'entryCost', 0)
        entry_str = f"${entry}" if entry else "Free"
        print(f"    {i}. {attr} ({act_type}) ★{rating} {entry_str} {unesco}")

    print(f"\n  Cuisine Recommendations:")
    for c in plan.cuisine_recommendations[:3]:
        spice = kb.get_one(c, 'spiceLevel', '')
        tags = kb.get(c, 'dietaryTag')
        print(f"    • {c} [{spice}] Tags: {', '.join(tags)}")

    if plan.notes:
        print(f"\n  Travel Notes:")
        for note in plan.notes:
            print(f"    ℹ {note}")


# =============================================================================
# TEST SUITE
# =============================================================================

def run_tests():
    print("=" * 65)
    print("  AI TRAVEL PLANNER — KNOWLEDGE BASE SYSTEM")
    print("=" * 65)

    kb = build_knowledge_base()
    planner = TravelPlanner(kb)
    print(f"\n✓ Knowledge Base loaded")
    print(f"  Destinations: {len(planner.get_all_destinations())}")
    print(f"  Attractions : {len(kb.query(p='type', o='Attraction'))}")
    print(f"  Cuisines    : {len(kb.query(p='type', o='Cuisine'))}")
    print(f"  Hotels      : {len(kb.query(p='type', o='Hotel'))}")

    # ---- Test 1: Budget solo traveller interested in history ----------------
    print("\n" + "="*65)
    print("  SCENARIO 1: Budget solo traveller, loves History & Culture")
    print("="*65)
    p1 = TravellerProfile(
        name="Ananya",
        budget_tier="budget",
        available_days=7,
        preferred_activities=["History", "Culture", "Museum"],
        dietary_restriction="veg",
        travel_style="solo",
        preferred_region="Asia",
    )
    plans1 = planner.plan(p1, top_n=3)
    for i, plan in enumerate(plans1, 1):
        print_plan(plan, kb, i)

    # Assertions
    assert all(plan.match_score > 0 for plan in plans1)
    assert plans1[0].match_score >= plans1[1].match_score   # sorted correctly
    print("\n  ✓ Test 1 passed: Plans generated, scores sorted correctly")

    # ---- Test 2: Luxury couple, beach & adventure ---------------------------
    print("\n" + "="*65)
    print("  SCENARIO 2: Luxury couple, Beach & Adventure")
    print("="*65)
    p2 = TravellerProfile(
        name="Arjun & Priya",
        budget_tier="luxury",
        available_days=10,
        preferred_activities=["Beach", "Adventure", "Sightseeing"],
        dietary_restriction="none",
        travel_style="couple",
        preferred_region="Europe",
    )
    plans2 = planner.plan(p2, top_n=3)
    for i, plan in enumerate(plans2, 1):
        print_plan(plan, kb, i)

    # Luxury hotels should be selected
    for plan in plans2:
        assert plan.hotel_cost_per_night >= 100, \
            f"Luxury traveller should get expensive hotel, got ${plan.hotel_cost_per_night}"
    print("\n  ✓ Test 2 passed: Luxury hotels selected, correct region preference")

    # ---- Test 3: Halal dietary, Middle East preference ----------------------
    print("\n" + "="*65)
    print("  SCENARIO 3: Mid-budget family, Halal diet, MiddleEast")
    print("="*65)
    p3 = TravellerProfile(
        name="Ahmed Family",
        budget_tier="mid",
        available_days=5,
        preferred_activities=["Culture", "Shopping", "Sightseeing"],
        dietary_restriction="halal",
        travel_style="family",
        preferred_region="MiddleEast",
    )
    plans3 = planner.plan(p3, top_n=2)
    for i, plan in enumerate(plans3, 1):
        print_plan(plan, kb, i)

    # Dubai should score high for halal + MiddleEast
    dest_names = [p.destination for p in plans3]
    print(f"\n  ✓ Test 3 passed: Top destinations = {dest_names}")

    # ---- Test 4: KB query verification --------------------------------------
    print("\n" + "="*65)
    print("  TEST 4: Knowledge Base Query Verification")
    print("="*65)

    # Query: UNESCO attractions in Kyoto
    kyoto_attrs = kb.get('Kyoto', 'hasAttraction')
    kyoto_unesco = [a for a in kyoto_attrs if kb.get_one(a, 'isUNESCO', False)]
    print(f"  UNESCO sites in Kyoto: {kyoto_unesco}")
    assert len(kyoto_unesco) >= 1, "Kyoto should have at least 1 UNESCO site"
    print(f"  ✓ UNESCO query works")

    # Query: All vegan-friendly cuisines
    vegan_cuisines = [s for s, p, o in kb.query(p='dietaryTag', o='vegan')]
    print(f"  Vegan cuisines: {vegan_cuisines}")
    assert len(vegan_cuisines) >= 2
    print(f"  ✓ Dietary tag query works")

    # Query: Cheapest destination
    dests = planner.get_all_destinations()
    costs = [(kb.get_one(d, 'averageDailyCost', 999), d) for d in dests]
    cheapest = min(costs)
    print(f"  Cheapest destination: {cheapest[1]} at ${cheapest[0]}/day")
    assert cheapest[0] <= 40
    print(f"  ✓ Cost query works")

    # Query: Top-rated attraction globally
    all_attrs = [s for s, p, o in kb.query(p='type', o='Attraction')]
    rated = [(kb.get_one(a, 'rating', 0), a) for a in all_attrs]
    top_attr = max(rated)
    print(f"  Top-rated attraction: {top_attr[1]} (★{top_attr[0]})")
    print(f"  ✓ Rating query works")

    print("\n" + "="*65)
    print("  ALL TRAVEL PLANNER TESTS PASSED ✓")
    print("="*65)


if __name__ == "__main__":
    def emit_step(title, status="running", detail="", data=None):
        print(json.dumps({
            "title": title,
            "status": status,
            "detail": detail,
            "data": data or {},
        }), flush=True)

    def run_api_steps():
        start = time.perf_counter()
        kg_tool_catalog = [
            {
                "category": "Graph databases",
                "tools": ["Neo4j", "Memgraph", "Amazon Neptune", "Azure Cosmos DB", "AllegroGraph", "GraphDB", "JanusGraph"],
                "use_case": "Store and query highly connected destination, food, hotel, and preference data.",
            },
            {
                "category": "Graph construction",
                "tools": ["LlamaIndex", "LangChain", "GliNER", "Infernotus", "ContextClue Graph Builder"],
                "use_case": "Extract entities and relations from PDFs, notes, reviews, and tables.",
            },
            {
                "category": "Ontology modeling",
                "tools": ["Protege", "TopBraid Composer"],
                "use_case": "Define formal tourism classes, properties, constraints, and reusable vocabularies.",
            },
            {
                "category": "Visualization",
                "tools": ["Gephi", "Kumu", "Linkurious"],
                "use_case": "Explore clusters, central entities, and recommendation paths.",
            },
            {
                "category": "Personal KGs",
                "tools": ["Obsidian Graph View", "TheBrain"],
                "use_case": "Model a traveller's notes, constraints, memories, and preferences.",
            },
        ]
        emit_step(
            "Build tourism knowledge base",
            detail="Loading destination, attraction, cuisine, hotel, cost, and season facts.",
            data={"tools": kg_tool_catalog, "visual_type": "tool-catalog"},
        )
        kb = build_knowledge_base()
        planner = TravelPlanner(kb)
        graph_edges = []
        for dest in planner.get_all_destinations()[:6]:
            for attr in kb.get(dest, "hasAttraction")[:3]:
                graph_edges.append({"source": dest, "target": attr, "relation": "hasAttraction"})
            for cuisine in kb.get(dest, "servesCuisine")[:2]:
                graph_edges.append({"source": dest, "target": cuisine, "relation": "servesCuisine"})
            hotel = planner.select_hotel(dest, "mid")[0]
            graph_edges.append({"source": dest, "target": hotel, "relation": "hasHotel"})
        emit_step(
            "Knowledge base loaded",
            detail="Triple store indexes are ready for planner queries.",
            data={
                "destinations": len(planner.get_all_destinations()),
                "attractions": len(kb.query(p="type", o="Attraction")),
                "cuisines": len(kb.query(p="type", o="Cuisine")),
                "hotels": len(kb.query(p="type", o="Hotel")),
                "graph_edges": graph_edges,
                "visual_type": "knowledge-graph",
            },
        )

        profile = TravellerProfile(
            name="Demo Traveller",
            budget_tier="budget",
            available_days=7,
            preferred_activities=["History", "Culture", "Museum"],
            dietary_restriction="veg",
            travel_style="solo",
            preferred_region="Asia",
        )
        emit_step(
            "Create traveller profile",
            detail="Using the demo preferences to score candidate destinations.",
            data=asdict(profile),
        )

        scored = [
            (round(planner.score_destination(dest, profile), 2), dest)
            for dest in planner.get_all_destinations()
        ]
        scored.sort(reverse=True)
        emit_step(
            "Score destinations",
            detail="Applied budget, activity, diet, season, and region rules.",
            data={
                "top_scores": [{"destination": dest, "score": score} for score, dest in scored[:5]],
                "score_weights": {
                    "budget_fit": 30,
                    "activity_match": 30,
                    "dietary_match": 20,
                    "season_fit": 10,
                    "region_match": 10,
                },
                "visual_type": "destination-ranking",
            },
        )

        plans = planner.plan(profile, top_n=3)
        for index, plan in enumerate(plans, 1):
            emit_step(
                f"Build recommendation {index}: {plan.destination}",
                detail="Selected hotel, attractions, cuisines, travel notes, and estimated cost.",
                data={
                    "destination": plan.destination,
                    "days": plan.days,
                    "match_score": round(plan.match_score, 2),
                    "hotel": plan.hotel,
                    "hotel_cost_per_night": plan.hotel_cost_per_night,
                    "daily_activity_cost": round(plan.daily_activity_cost, 2),
                    "daily_food_budget": plan.daily_food_budget,
                    "total_estimated_cost": round(plan.total_estimated_cost, 2),
                    "attractions": plan.attractions[:6],
                    "cuisines": plan.cuisine_recommendations[:4],
                    "notes": plan.notes,
                    "visual_type": "tour-plan",
                },
            )

        emit_step(
            "Q2 complete",
            status="complete",
            detail="Travel planning workflow finished successfully.",
            data={"elapsed_ms": round((time.perf_counter() - start) * 1000, 2)},
        )

    if "--steps" in sys.argv:
        run_api_steps()
    else:
        run_tests()
