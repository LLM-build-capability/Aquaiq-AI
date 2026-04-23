from src.aquaiq_ai.agent import run_agent

def main():
   messages = []
   while True:
       user_input = input("You: ")
       messages.append({"role": "user", "content": user_input})
       response = run_agent(messages)
       print("\nAI:", response, "\n")
       messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
   main()