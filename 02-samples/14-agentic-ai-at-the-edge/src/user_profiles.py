"""
User profiles for personalized AI assistant responses.

This module contains predefined user profiles that can be loaded
based on the DRIVER_PROFILE environment variable.
"""

USER_PROFILES = {
    "sarah": {
        "name": "Sarah",
        "vehicle": {
            "make": "AutoDrive",
            "model": "CX-7",
            "year": "2023",
            "features": ["run-flat tires", "all-wheel drive", "parking assist", "heads-up display"],
        },
        "preferences": {
            "communication_style": "friendly and detailed",
            "response_length": "comprehensive with examples",
            "technical_level": "intermediate - explain technical terms",
        },
        "common_concerns": ["tire pressure", "maintenance schedules", "safety features"],
        "personalization_prompt": """
Remember: You're assisting Sarah who drives a 2023 AutoDrive CX-7.
- She appreciates detailed explanations with examples
- She's safety-conscious and likes to understand how things work
- Reference her AutoDrive CX-7's specific features when relevant
- Use a warm, friendly tone
""",
    },
    "bob": {
        "name": "Bob",
        "vehicle": {
            "make": "Tesla",
            "model": "Model 3",
            "year": "2022",
            "features": ["autopilot", "regenerative braking", "mobile app", "supercharging"],
        },
        "preferences": {
            "communication_style": "direct and concise",
            "response_length": "brief and to the point",
            "technical_level": "advanced - comfortable with technical details",
        },
        "common_concerns": ["battery efficiency", "software updates", "charging optimization"],
        "personalization_prompt": """
Remember: You're assisting Bob who drives a 2022 Tesla Model 3.
- He prefers brief, direct answers without unnecessary detail
- He's tech-savvy and understands EV technology
- Focus on data and facts rather than explanations
- Keep responses professional and efficient
""",
    },
    "guest": {
        "name": "Driver",
        "vehicle": {
            "make": "TechCar",
            "model": "Model X",
            "year": "2024",
            "features": [
                "standard safety features",
                "cruise control",
                "bluetooth",
                "backup camera",
            ],
        },
        "preferences": {
            "communication_style": "professional and helpful",
            "response_length": "moderate with key points",
            "technical_level": "general - avoid jargon",
        },
        "common_concerns": ["general maintenance", "warning lights", "basic features"],
        "personalization_prompt": """
Remember: You're assisting a driver with a standard vehicle.
- Provide clear, helpful information
- Don't assume specific vehicle features
- Use general automotive knowledge
- Maintain a professional, supportive tone
""",
    },
}


def get_user_profile(profile_name: str = None) -> dict:
    """
    Get user profile by name or return guest profile.

    Args:
        profile_name: Name of the profile (sarah, bob, or guest)

    Returns:
        Dictionary containing user profile information
    """
    if profile_name and profile_name.lower() in USER_PROFILES:
        return USER_PROFILES[profile_name.lower()]
    return USER_PROFILES["guest"]
