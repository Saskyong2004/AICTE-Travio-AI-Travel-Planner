"""
agent_tools.py
==============
Standalone tool functions used by the Travel Planner Agent.
Each function can be called independently or wired into a LangChain agent.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Weather Tool
# ---------------------------------------------------------------------------

def get_weather(city: str) -> dict:
    """
    Fetch current weather and a 5-day forecast for a given city.
    Uses OpenWeatherMap free-tier API.
    Falls back to mock data when no API key is configured.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key or api_key == "your_openweathermap_api_key_here":
        return _mock_weather(city)

    try:
        base = "https://api.openweathermap.org/data/2.5"
        current_url = f"{base}/weather?q={city}&appid={api_key}&units=metric"
        forecast_url = f"{base}/forecast?q={city}&appid={api_key}&units=metric&cnt=5"

        current_resp = requests.get(current_url, timeout=8)
        forecast_resp = requests.get(forecast_url, timeout=8)

        if current_resp.status_code != 200:
            return {"error": f"City '{city}' not found or API error.", "mock": False}

        cw = current_resp.json()
        fc = forecast_resp.json()

        forecast_days = []
        for item in fc.get("list", []):
            forecast_days.append({
                "date": item["dt_txt"],
                "temp_c": item["main"]["temp"],
                "description": item["weather"][0]["description"],
            })

        return {
            "city": cw["name"],
            "country": cw["sys"]["country"],
            "temperature_c": cw["main"]["temp"],
            "feels_like_c": cw["main"]["feels_like"],
            "humidity_pct": cw["main"]["humidity"],
            "description": cw["weather"][0]["description"],
            "wind_kmh": round(cw["wind"]["speed"] * 3.6, 1),
            "forecast": forecast_days,
            "mock": False,
        }
    except Exception as exc:
        return {"error": str(exc), "mock": False}


def _mock_weather(city: str) -> dict:
    """Return plausible mock weather so the UI works without an API key."""
    mock_map = {
        "paris": ("Paris", "FR", 18, 17, 65, "partly cloudy", 14.4),
        "tokyo": ("Tokyo", "JP", 22, 21, 70, "clear sky", 10.8),
        "new york": ("New York", "US", 15, 13, 60, "overcast clouds", 18.0),
        "dubai": ("Dubai", "AE", 38, 36, 30, "sunny", 20.0),
        "london": ("London", "GB", 12, 10, 80, "light rain", 22.0),
        "sydney": ("Sydney", "AU", 25, 24, 55, "clear sky", 16.2),
        "bali": ("Bali", "ID", 30, 29, 80, "humid and sunny", 12.0),
    }
    key = city.lower()
    name, country, temp, feels, hum, desc, wind = mock_map.get(
        key, (city.title(), "XX", 20, 19, 60, "fair weather", 15.0)
    )
    today = datetime.utcnow()
    forecast = [
        {
            "date": (today + timedelta(days=i)).strftime("%Y-%m-%d 12:00:00"),
            "temp_c": temp + (i % 3 - 1),
            "description": desc,
        }
        for i in range(1, 6)
    ]
    return {
        "city": name,
        "country": country,
        "temperature_c": temp,
        "feels_like_c": feels,
        "humidity_pct": hum,
        "description": desc,
        "wind_kmh": wind,
        "forecast": forecast,
        "mock": True,
    }


# ---------------------------------------------------------------------------
# Destination Suggestions Tool
# ---------------------------------------------------------------------------

DESTINATION_DB = {
    "beach": [
        {"name": "Bali, Indonesia", "highlights": "Temples, surfing, rice terraces", "budget": "Low-Medium", "best_months": "Apr–Oct"},
        {"name": "Santorini, Greece", "highlights": "Cliffs, sunsets, volcanic beaches", "budget": "High", "best_months": "May–Sep"},
        {"name": "Phuket, Thailand", "highlights": "Nightlife, clear waters, islands", "budget": "Low-Medium", "best_months": "Nov–Apr"},
        {"name": "Maldives", "highlights": "Overwater bungalows, snorkelling", "budget": "Very High", "best_months": "Nov–Apr"},
    ],
    "mountain": [
        {"name": "Himachal Pradesh, India", "highlights": "Trekking, snow peaks, monasteries", "budget": "Low", "best_months": "Mar–Jun"},
        {"name": "Swiss Alps, Switzerland", "highlights": "Skiing, scenic trains, fondue", "budget": "Very High", "best_months": "Dec–Mar"},
        {"name": "Queenstown, New Zealand", "highlights": "Adventure sports, fjords, wine", "budget": "High", "best_months": "Dec–Feb"},
    ],
    "cultural": [
        {"name": "Kyoto, Japan", "highlights": "Temples, geisha districts, cherry blossoms", "budget": "Medium-High", "best_months": "Mar–May"},
        {"name": "Rome, Italy", "highlights": "Colosseum, Vatican, pasta", "budget": "Medium-High", "best_months": "Apr–Jun"},
        {"name": "Machu Picchu, Peru", "highlights": "Inca ruins, Andes trekking", "budget": "Medium", "best_months": "May–Sep"},
    ],
    "adventure": [
        {"name": "Patagonia, Argentina/Chile", "highlights": "Glaciers, trekking, condors", "budget": "High", "best_months": "Nov–Mar"},
        {"name": "Iceland", "highlights": "Northern lights, geysers, waterfalls", "budget": "Very High", "best_months": "Jun–Aug"},
        {"name": "Costa Rica", "highlights": "Rainforests, zip-lining, volcanoes", "budget": "Medium", "best_months": "Dec–Apr"},
    ],
    "city": [
        {"name": "Tokyo, Japan", "highlights": "Tech, anime, cuisine, culture", "budget": "Medium-High", "best_months": "Mar–May"},
        {"name": "New York, USA", "highlights": "Broadway, museums, skyline", "budget": "High", "best_months": "Apr–Jun"},
        {"name": "Dubai, UAE", "highlights": "Luxury, desert safari, architecture", "budget": "High", "best_months": "Nov–Mar"},
        {"name": "Singapore", "highlights": "Food, gardens, shopping", "budget": "High", "best_months": "Feb–Apr"},
    ],
}


def get_destinations(travel_type: str = "beach", budget: Optional[str] = None) -> list:
    """
    Return destination suggestions filtered by travel type and optional budget tier.
    travel_type: beach | mountain | cultural | adventure | city
    budget: Low | Medium | High | Very High (optional substring match)
    """
    travel_type = travel_type.lower().strip()
    results = DESTINATION_DB.get(travel_type, DESTINATION_DB["beach"])

    if budget:
        budget_lower = budget.lower()
        filtered = [d for d in results if budget_lower in d["budget"].lower()]
        if filtered:
            results = filtered

    return results


# ---------------------------------------------------------------------------
# Itinerary Builder Tool
# ---------------------------------------------------------------------------

def build_itinerary_prompt(
    destination: str,
    days: int,
    travel_style: str,
    budget: str,
    interests: str,
) -> str:
    """
    Build a structured prompt to send to IBM Granite for itinerary generation.
    """
    return f"""You are an expert travel planner. Create a detailed {days}-day itinerary for a trip to {destination}.

Travel preferences:
- Travel style: {travel_style}
- Budget level: {budget}
- Interests: {interests}

For EACH day provide:
1. Morning activity with location name
2. Afternoon activity with location name
3. Evening activity / dinner recommendation
4. One practical travel tip for that day
5. Estimated daily spend in USD

Format your response as structured plain text with clear Day headers (Day 1:, Day 2:, etc.).
Be specific, realistic and practical. Include local transport tips where relevant.
End with a brief packing checklist suited to the destination and travel style."""


# ---------------------------------------------------------------------------
# Transport & Accommodation Helper
# ---------------------------------------------------------------------------

def get_transport_options(origin: str, destination: str) -> dict:
    """Return generic transport options between two places."""
    return {
        "origin": origin,
        "destination": destination,
        "options": [
            {
                "mode": "Flight",
                "typical_duration": "Varies",
                "cost_range_usd": "80–800",
                "tip": "Book 6–8 weeks ahead for best fares. Use Google Flights or Skyscanner.",
            },
            {
                "mode": "Train",
                "typical_duration": "Varies",
                "cost_range_usd": "20–200",
                "tip": "Rail passes (Eurail, JR Pass) offer great value for multi-city trips.",
            },
            {
                "mode": "Bus / Coach",
                "typical_duration": "Longer",
                "cost_range_usd": "5–60",
                "tip": "Budget-friendly; book FlixBus, Greyhound, or local operators.",
            },
            {
                "mode": "Car Rental",
                "typical_duration": "Flexible",
                "cost_range_usd": "30–120/day",
                "tip": "Best for rural or off-the-beaten-path destinations.",
            },
        ],
    }


def get_accommodation_options(destination: str, budget: str) -> list:
    """Return accommodation suggestions by budget tier."""
    budget_lower = budget.lower()
    if "low" in budget_lower:
        return [
            {"type": "Hostel / Dorm", "cost_usd_night": "8–25", "platforms": "Hostelworld, Booking.com"},
            {"type": "Budget Guesthouse", "cost_usd_night": "20–50", "platforms": "Booking.com, Agoda"},
            {"type": "Couchsurfing / Workaway", "cost_usd_night": "Free", "platforms": "Couchsurfing.com"},
        ]
    elif "very high" in budget_lower:
        return [
            {"type": "Luxury Resort / 5-Star Hotel", "cost_usd_night": "250–1000+", "platforms": "Leading Hotels, Aman"},
            {"type": "Boutique Private Villa", "cost_usd_night": "300–2000", "platforms": "Airbnb Luxe, OneFineStay"},
        ]
    elif "high" in budget_lower:
        return [
            {"type": "4-Star Hotel", "cost_usd_night": "120–250", "platforms": "Booking.com, Hotels.com"},
            {"type": "Premium Airbnb", "cost_usd_night": "100–300", "platforms": "Airbnb"},
        ]
    else:  # Medium
        return [
            {"type": "3-Star Hotel", "cost_usd_night": "50–120", "platforms": "Booking.com, Expedia"},
            {"type": "Serviced Apartment", "cost_usd_night": "60–150", "platforms": "Airbnb, Vrbo"},
            {"type": "Boutique Hostel (Private Room)", "cost_usd_night": "30–70", "platforms": "Hostelworld"},
        ]
