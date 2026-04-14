from serpapi import SerpApiClient
import os
from dotenv import load_dotenv

load_dotenv()

def search(query: str) -> str:
    """
    A Web search tool based on Serpapi.
    """
    print(f"Searching the web for: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SERPAPI_API_KEY not found in environment variables."
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn"
        }

        client = SerpApiClient(params)
        results = client.get_dict()
        
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # if no answer, return top 3 organic results
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return "Sorry. No relevant information found."
    
    except Exception as e:
        return f"Error occurred when calling Serpapi: {e}"
