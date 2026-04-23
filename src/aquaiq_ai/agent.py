import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from src.aquaiq_ai.retriever import retrieve
from src.aquaiq_ai.tools import get_air_quality
load_dotenv()

client = AzureOpenAI(
   api_version=os.getenv("API_VERSION"),
   azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
   api_key=os.getenv("AZUREOPENAIAPI_KEY"),
)
def extract_city(query: str) -> str:
   words = query.lower().split()
   if "in" in words:
       idx = words.index("in")
       if idx + 1 < len(words):
           return words[idx + 1].capitalize()
   return words[-1].capitalize()
def should_use_tool(query: str) -> bool:
   keywords = ["air quality", "aqi", "pollution"]
   return any(k in query.lower() for k in keywords)
def run_agent(messages):
   user_query = messages[-1]["content"]
   tool_result = ""
   if should_use_tool(user_query):
       city = extract_city(user_query)
       tool_result = get_air_quality(city)

   retrieved = retrieve(user_query)
   context = "\n\n".join([d["text"] for d in retrieved])
   sources = list({d["source"] for d in retrieved})

   system_prompt = f"""
You are an intelligent environmental assistant.
Use:
- Context for explanations (ONLY from given data)
- Tool data for real-time information
Context:
{context}
Tool Data:
{tool_result}
Rules:
- Use ONLY the provided context for facts
- DO NOT add external sources
- DO NOT hallucinate
- If tool data exists, ALWAYS use it
- If something is missing, say "not available in provided data"
- Answer clearly and in structured format
"""
   response = client.chat.completions.create(
       model="gpt-5.4-nano",
       messages=[{"role": "system", "content": system_prompt}] + messages,
   )
   answer = response.choices[0].message.content
   if sources:
       answer += "\n\nSources:\n"
       for s in sources:
           answer += f"- {s}\n"
   return answer
