This repository contains my submission for the AmbedkarGPT Internship Assignment.
The project implements a Q&A system using:
->HuggingFace Embeddings
->ChromaDB Vectorstore
->Chunk-based Retrieval
->Ollama Model for Response Generation (e.g., LLaMA/others)
->The system loads a provided speech.txt file, splits it into chunks, stores embeddings, retrieves answers based on user questions, and generates responses using the Ollama model.
STEPS:
->Load the entire speech.txt file
->Split the text into manageable chunks
->Generate embeddings for each chunk
->Store them in a ChromaDB vector database

For each user question:
->Retrieve the most relevant chunks using similarity search
->Build a prompt
->Send the prompt to the Ollama model
->Display the final answer
