import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class LLMClient:
    """
    A customized LLM client used to call any service
    compatible with the OpenAI api.
    """
    def __init__(self, model: str = None, apiKey: str = None, baseURL: str = None, timeout: int = None):
        """
        Initialize the client.
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("DEEPSEEK_API_KEY")
        baseURL = baseURL or os.getenv("LLM_BASE_URL")
        timeout = timeout
        
        if not all([self.model, apiKey, baseURL, timeout]):
            raise ValueError("Missing required parameters")
        
        self.client = OpenAI(api_key=apiKey, base_url=baseURL, timeout=timeout)
    
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        Call the LLM to think and return its response.
        """
        print(f"Calling {self.model} model...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            # Handle streaming response
            print("LLM response successful:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print() # Newline after streaming output ends
            return "".join(collected_content)
        
        except Exception as e:
            print(f"Error occurred when calling LLM API: {e}")
            return None

# Client Usage Example
if __name__ == "__main__":
    try:
        llmClient = LLMClient(timeout = 60)
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "Write a quick sort algorithm."}
        ]
            
        print("--- Calling LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- Complete Model Response ---")
            print(responseText)

    except ValueError as e:
        print(e)
                
