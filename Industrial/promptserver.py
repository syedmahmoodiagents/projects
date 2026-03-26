import os
from dotenv import load_dotenv
load_dotenv()
serp = os.getenv("SERP_API_KEY")

from fastmcp import FastMCP
from serpapi import GoogleSearch


mcp = FastMCP("Shopping Comparison Engine")

@mcp.tool()
def get_brand_prices(brand: str) -> str:
    """Fetches top 3 product prices for a brand using SerpApi."""
    params = {
        "engine": "google_shopping",
        "q": f"{brand} shoes",
        "api_key": serp,
        "num": 3
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("shopping_results", [])
    
    data_summary = ""
    for item in results:
        data_summary += f"- {item.get('title')}: {item.get('price')} (Rating: {item.get('rating')})\n"
    return data_summary

@mcp.prompt()
def compare_brands_template(brand_a: str, brand_b: str, data_a: str, data_b: str) -> str:
    """A template that instructs the AI how to compare the brands."""
    return f"""
    You are a professional market analyst. Use the real-time data below to compare {brand_a} and {brand_b}.
    
    --- DATA FOR {brand_a.upper()} ---
    {data_a}
    
    --- DATA FOR {brand_b.upper()} ---
    {data_b}
    
    TASK: 
    1. Which brand is generally more expensive?
    2. Which brand has better customer ratings based on these specific results?
    3. Give a final 'Value for Money' recommendation.
    """

if __name__ == "__main__":
    mcp.run()
