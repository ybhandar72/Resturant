"""Restaurant agent: tools + LangChain agent powered by Claude Haiku 5.5."""
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic

load_dotenv()  # loads ANTHROPIC_API_KEY from a local .env file if present

MODEL_NAME = "claude-haiku-5-5"

RESTAURANTS_BY_CUISINE = {
    "indian": [
        {"name": "Spice Palace", "rating": 4.5, "price": "₹₹"},
        {"name": "Curry House", "rating": 4.3, "price": "₹₹"},
        {"name": "Taj Flavors", "rating": 4.7, "price": "₹₹₹"},
    ],
    "italian": [
        {"name": "Pizza Bella", "rating": 4.4, "price": "₹₹"},
        {"name": "Pasta Dreams", "rating": 4.6, "price": "₹₹₹"},
        {"name": "Roma Kitchen", "rating": 4.2, "price": "₹₹"},
    ],
    "chinese": [
        {"name": "Dragon House", "rating": 4.5, "price": "₹₹"},
        {"name": "Ming's Kitchen", "rating": 4.3, "price": "₹₹"},
        {"name": "Golden Wok", "rating": 4.8, "price": "₹₹₹"},
    ],
    "mexican": [
        {"name": "Taco Fiesta", "rating": 4.4, "price": "₹₹"},
        {"name": "Salsa Nights", "rating": 4.6, "price": "₹₹₹"},
        {"name": "Burrito House", "rating": 4.2, "price": "₹₹"},
    ],
}

RESTAURANT_DETAILS = {
    "Spice Palace": {"cuisine": "Indian", "rating": 4.5, "address": "123 Food Street, City Center",
                     "phone": "9876543210", "hours": "11 AM - 11 PM", "speciality": "Butter Chicken, Biryani"},
    "Pizza Bella": {"cuisine": "Italian", "rating": 4.4, "address": "456 Dine Avenue, Downtown",
                    "phone": "9123456789", "hours": "12 PM - 10:30 PM", "speciality": "Margherita Pizza, Tiramisu"},
    "Dragon House": {"cuisine": "Chinese", "rating": 4.5, "address": "789 Noodle Lane, Market Area",
                     "phone": "9111222333", "hours": "11:30 AM - 11 PM", "speciality": "Hakka Noodles, Fried Rice"},
    "Taco Fiesta": {"cuisine": "Mexican", "rating": 4.4, "address": "321 Spice Row, Hip District",
                    "phone": "9222333444", "hours": "1 PM - 12 AM", "speciality": "Fish Tacos, Guacamole"},
    "Curry House": {"cuisine": "Indian", "rating": 4.3, "address": "555 Flavor Street, Old Town",
                    "phone": "9333444555", "hours": "11 AM - 10 PM", "speciality": "Paneer Tikka, Dal Makhani"},
    "Taj Flavors": {"cuisine": "Indian", "rating": 4.7, "address": "999 Taste Lane, Premium Zone",
                    "phone": "9444555666", "hours": "12 PM - 11 PM", "speciality": "Biryani, Tandoori Chicken"},
    "Golden Wok": {"cuisine": "Chinese", "rating": 4.8, "address": "777 Wok Street, Central Market",
                   "phone": "9555666777", "hours": "11 AM - 11 PM", "speciality": "Szechuan Noodles, Dim Sum"},
}

AVAILABLE_SLOTS = {
    "Spice Palace": {"2024-11-15": True, "2024-11-16": False, "2024-11-17": True},
    "Pizza Bella": {"2024-11-15": False, "2024-11-16": True, "2024-11-17": True},
    "Dragon House": {"2024-11-15": True, "2024-11-16": True, "2024-11-17": False},
    "Taco Fiesta": {"2024-11-15": True, "2024-11-16": True, "2024-11-17": True},
    "Curry House": {"2024-11-15": True, "2024-11-16": False, "2024-11-17": True},
    "Taj Flavors": {"2024-11-15": False, "2024-11-16": True, "2024-11-17": True},
}


@tool
def search_restaurants_by_cuisine(cuisine_type: str) -> list:
    """Search for restaurants by cuisine type.
    Supported cuisines: Indian, Italian, Chinese, Mexican."""
    return RESTAURANTS_BY_CUISINE.get(cuisine_type.lower().strip(), [])


@tool
def get_restaurant_details(restaurant_name: str) -> dict:
    """Get detailed information about a restaurant,
    including address, phone, hours and specialties."""
    return RESTAURANT_DETAILS.get(
        restaurant_name.strip(), {"error": f"Restaurant '{restaurant_name}' not found"}
    )


@tool
def check_reservation_availability(restaurant_name: str, date: str, party_size: int) -> str:
    """Check if a restaurant has reservation availability for a specific
    date (YYYY-MM-DD) and number of people."""
    name = restaurant_name.strip()
    if name not in AVAILABLE_SLOTS:
        return f"Restaurant '{restaurant_name}' not found."
    if AVAILABLE_SLOTS[name].get(date.strip(), False):
        return f"✓ {name} has availability for {party_size} people on {date}"
    return f"✗ {name} is fully booked (or has no slots listed) for {date}"


TOOLS = [search_restaurants_by_cuisine, get_restaurant_details, check_reservation_availability]

SYSTEM_PROMPT = """You are a friendly restaurant assistant.

You have exactly three tools: search restaurants by cuisine, get restaurant
details, and check reservation availability. Use them whenever relevant and
never invent restaurant information.

You ONLY help with restaurant search, restaurant details and reservation
availability. For anything else (general knowledge, math, coding, news,
opinions, etc.) politely decline in one or two sentences, say what you can
help with, and do not answer the off-topic question. You cannot make bookings,
only check availability. Be conversational and concise."""


def build_agent():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    model = ChatAnthropic(model=MODEL_NAME)
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


def message_text(content) -> str:
    """Claude may return a string or a list of content blocks."""
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def chat_with_agent(agent, user_message: str, history: list | None = None):
    messages = (history or []) + [{"role": "user", "content": user_message}]
    result = agent.invoke({"messages": messages})
    reply = message_text(result["messages"][-1].content)
    return reply, messages + [{"role": "assistant", "content": reply}]
