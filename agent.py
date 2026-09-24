import cohere
import os
from dotenv import load_dotenv
load_dotenv()

class Agent():
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    client = cohere.ClientV2(api_key=COHERE_API_KEY)

    def __init__(self, systemPrompt):
        self.messages = [
            {
                "role": "system", 
                "content": systemPrompt
            }
        ]

    def call(self, prompt: str, model: str = "command-a-03-2025"):
    
        self.messages.append(
            {"role": "user", "content": prompt}  # Add user message to existing list
        )
        try:
            response = self.client.chat(
                model=model,
                messages=self.messages
            )
            # print(f"Finish reason: {response.finish_reason}")

            msg = response.message
            self.messages.append(msg)
    
            if msg.content:
                for content in msg.content:
                    return(content.text)

        except Exception as e:
            print(type(e))
            self.messages.append({"role": "system", "content": f"{str(e)}"})