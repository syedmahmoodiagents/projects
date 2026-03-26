import asyncio
from fastmcp import Client

async def run_analysis():
    async with Client("promptserver.py") as client:
        
        print("Fetching Nike data...")
        nike_data = await client.call_tool("get_brand_prices", {"brand": "nike"})
        
        print("Fetching Puma data...")
        puma_data = await client.call_tool("get_brand_prices", {"brand": "puma"})

        final_prompt = await client.get_prompt(
            "compare_brands_template", 
            arguments={
                "brand_a": "Nike",
                "brand_b": "Puma",
                "data_a": nike_data,
                "data_b": puma_data
            }
        )

        print("\n--- GENERATED PROMPT FOR THE AI ---")
        print(final_prompt)

if __name__ == "__main__":
    asyncio.run(run_analysis())
