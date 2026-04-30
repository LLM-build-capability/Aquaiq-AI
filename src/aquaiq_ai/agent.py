import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openai import AzureOpenAI
from dotenv import load_dotenv

from src.aquaiq_ai.retriever import WaterDocRetriever
from src.aquaiq_ai.embedding_helper import AzureEmbedder
from src.aquaiq_ai.tools import WATER_QUALITY_TOOL, execute_water_quality_tool

load_dotenv()


class WaterAgent:
    def __init__(self):
        # Set up Azure client
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.chat_model = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.max_tool_calls = int(os.getenv("MAX_TOOL_ITERATIONS", "3"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

        # Using the same embedder everywhere
        self.embedder = AzureEmbedder()
        self.retriever = WaterDocRetriever()
        self.tools = [WATER_QUALITY_TOOL]

        # Conversation history
        self.messages = []

        # Example questions for routing, I picked these randomly from the pdfs
        self.rag_examples = [
            "What is water treatment?",
            "How does chlorination work?",
            "Explain filtration methods",
            "What are EPA guidelines?",
            "Tell me about sedimentation",
            "How does coagulation work?",
            "What are drinking water standards?",
            "Explain membrane filtration",
            "What is disinfection?",
        ]

        self.tool_examples = [
            "Water quality in Travis County",
            "Nitrate levels in Williamson County",
            "pH data for Benton County",
            "Dissolved oxygen in Baxter County",
            "Water quality data for Prince George County",
            "Contamination in Oklahoma County",
        ]

        # Pre-calculate example embeddings
        print("Calculating example embeddings for routing...")
        self.rag_vecs = [self.embedder.embed(q) for q in self.rag_examples]
        self.tool_vecs = [self.embedder.embed(q) for q in self.tool_examples]
        print("Routing ready.")

    def _similarity(self, a, b):
        # We are using cosine similarity
        a_np = np.array(a)
        b_np = np.array(b)
        if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
            return 0
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

    def _classify(self, query):
        # checks what we need, rag or tool or both
        q_vec = self.embedder.embed(query)

        # Average similarity to RAG examples
        rag_scores = [self._similarity(q_vec, v) for v in self.rag_vecs]
        rag_avg = sum(rag_scores) / len(rag_scores)

        # Average similarity to tool examples
        tool_scores = [self._similarity(q_vec, v) for v in self.tool_vecs]
        tool_avg = sum(tool_scores) / len(tool_scores)

        print(f"  RAG similarity: {rag_avg:.3f}, Tool similarity: {tool_avg:.3f}")

        # If scores are close, use both
        if abs(rag_avg - tool_avg) < 0.08:
            return "both"
        elif rag_avg > tool_avg:
            return "rag"
        else:
            return "tool"

    def reset(self):
        # clear conversation history
        self.messages = []

    def chat(self, user_input):
        # this function processes user query and gives response

        # Add user message to history
        self.messages.append({"role": "user", "content": user_input})

        # Figure out what we need
        q_type = self._classify(user_input)
        print(f"Query type: {q_type}")

        # Add document context if needed
        if q_type in ["rag", "both"] and self.retriever.available:
            context = self.retriever.get_context(user_input)
            if context:
                self.messages.append({
                    "role": "system",
                    "content": f"Here's info from the documents:\n{context}\n\nUse this if it helps."
                })

        # Whether to allow tool calling
        use_tool = q_type in ["tool", "both"]

        # Loop for tool calls
        for _ in range(self.max_tool_calls):
            params = {
                "model": self.chat_model,
                "messages": self.messages,
                "temperature": self.temperature,
                "max_completion_tokens": 1000,
            }
            if use_tool:
                params["tools"] = self.tools
                params["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**params)
            msg = response.choices[0].message

            # Check if the model wants to call a tool
            if use_tool and msg.tool_calls:
                # Add assistant message
                self.messages.append(msg)

                # Execute each tool call
                for tc in msg.tool_calls:
                    if tc.function.name == "get_water_quality":
                        args = json.loads(tc.function.arguments)
                        result = execute_water_quality_tool(
                            county_name=args.get("county_name"),
                            characteristic=args.get("characteristic", "")
                        )
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result)
                        })
                # Loop again to get final answer
                continue
            else:
                # Final answer
                self.messages.append(msg)
                return msg.content

        return "Something went wrong - too many tool calls."