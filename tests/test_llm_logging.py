from depths.logger.llm import LoggedOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client=LoggedOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

response = client.chat.completions.create(
    model="gemini-2.5-flash-lite",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explain the importance of analytics for AI Agents"
        }
    ]
)

print(response.choices[0].message.content[:20])