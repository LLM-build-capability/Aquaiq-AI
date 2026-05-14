# Exercise 2 – Notes
**Authors:**
- Prem Kumar Reddy K
- Deepak P
- Pavitra P

**Cohort:** LLM Capability

Honestly, We thought modelling my own RAG pipeline would be straightforward – we wrote the code, so we knew all the pieces. But drawing it in ArchiMate forced me to see things we had completely missed in the code. The biggest surprise was how many **data objects** we had to create. In our Python script, the conversation history is just `self.messages` – a list we pass around. we never thought of it as a separate thing that components access. But in the model, `Conversation History` became a full Data Object with read/write access from the `Agent Orchestrator`. The same happened with `Text Chunks`, `Chunk Embeddings`, and `Tool Schema`. 

The tool‑call loop was another thing we missed. In `agent.py`, it's just a `for` loop with a few lines of code. we almost modelled it as a simple edge between `Agent Orchestrator` and `Tool Execution Service`. But the assignment explicitly says the loop should be a named **Application Process**. Once we added `Tool‑Call Loop` as a separate element and connected it with a `Composition` from the `Agent Orchestrator`, we understood that the loop isn't just an implementation detail – it's a core part of the orchestration that could be extended or replaced.  

Finally, the **Technology Usage view** made me think about hosting. We had never considered that the `Python Runtime` is a Node that all my components run on. The assignment forced me to assign each component to that Node (or to `ChromaDB` for the vector store). That separation between logical components and physical infrastructure is something we always ignored in code, but ArchiMate makes it explicit.  

Overall, the exercise taught us that a good architecture model reveals couplings and responsibilities that are easy to hide in code. Now we can see the whole project differently - not just as a block of code, but as a system with clear layers, data flows, and external dependencies. It was very hard compared to writting code, mainly getting the relationship direction right.