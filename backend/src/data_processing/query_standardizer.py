import re
from typing import Dict

from ..llm.factory import LLMFactory
from ..utils.logger import logger


def standardize_query(query: str, model_name: str = "gpt-3.5-turbo") -> Dict:
    """
    Standardize team names, venue, and identify home grounds from the query.

    Args:
        query: The user's query string
        model_name: The LLM model to use for standardization

    Returns:
        A dictionary with standardized team names, venue, and home ground information
    """
    try:
        # Get LLM instance
        llm_factory = LLMFactory()
        llm = llm_factory.create_llm(model_name)

        # Define the standardization prompt
        standardization_prompt = f"""
            You are an IPL cricket expert. Extract and standardize the following information from the user query:
            1. Team 1 name (standard IPL team name)
            2. Team 2 name (standard IPL team name)
            3. Venue name (standard IPL venue name)
            4. Home ground of Team 1 (Yes/No)
            5. Home ground of Team 2 (Yes/No)

            For team names, use these standard names: Mumbai_Indians, Chennai_Super_Kings, 
            Royal_Challengers_Bangalore, Kolkata_Knight_Riders, Rajasthan_Royals, 
            Delhi_Capitals, Sunrisers_Hyderabad, Punjab_Kings, Gujarat_Titans, 
            Lucknow_Super_Giants.

            For venues, use these standard names: Wankhede Stadium, M. A. Chidambaram Stadium, 
            M. Chinnaswamy Stadium, Eden Gardens, Sawai Mansingh Stadium, Arun Jaitley Stadium, 
            Rajiv Gandhi International Stadium, Punjab Cricket Association Stadium, Narendra Modi Stadium, 
            Ekana Cricket Stadium.

            Just return only the below information in this exact JSON format, no other text or comments before or after the JSON:
            {{
                "team1": "standard_team1_name",
                "team2": "standard_team2_name",
                "venue": "standard_venue_name",
                "is_team1_home": true/false,
                "is_team2_home": true/false
            }}

            User query: {query}
            """

        # Get response from LLM
        response = llm.generate([standardization_prompt]).generations[0][0].text
        match = re.search(r"{[\s\S]*}", response)
        json_text = match.group(0)

        # Parse the JSON response
        import json

        standardized_info = json.loads(json_text)

        logger.info(f"Standardized query: {standardized_info}")
        return standardized_info

    except Exception as e:
        logger.error(f"Error standardizing query: {e}")
        # Return a default structure in case of error
        return {
            "team1": "",
            "team2": "",
            "venue": "",
            "is_team1_home": False,
            "is_team2_home": False,
        }
