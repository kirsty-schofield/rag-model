import os
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Setup and data
def initialise_vector_store(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    # Simplified: split_documents handles loading + splitting in one go
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    splits = loader.load_and_split(text_splitter)

    local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return Chroma.from_documents(
        documents=splits, 
        embedding=local_embeddings,
        collection_metadata={"hnsw:space": "cosine"} 
    )


# Quality checks and filtering
def log_and_filter_chunks(retrieved_docs_with_scores: list[tuple]):
    
    print("\n--- [Chunk Quality Scoring] ---")
    relevant_docs = []
    
    for doc, score in retrieved_docs_with_scores:
        is_relevant = score < 0.65  # relevance threshold
        status = "PASS" if is_relevant else "FILTERED"
        
        source_info = doc.metadata.get('source', 'Unknown')
        page_info = doc.metadata.get('page', 0) + 1
        
        print(f"Distance Score: {score:.3f} | Source: {source_info} (Pg {page_info})")
        print(f"Snippet: {doc.page_content[:75].strip()}...")
        print(f"Status: {status}\n")
        
        if is_relevant:
            relevant_docs.append(doc)
            
    return relevant_docs

# Citations to show where data comes from
def format_docs_with_sources(docs: list):
    if not docs:
        return "NO_CONTEXT"
    
    formatted_context = ""
    for i, doc in enumerate(docs):
        # Add metadata to the context so the LLM can reference it
        page = doc.metadata.get('page', 'unknown')
        formatted_context += f"[Doc {i+1} - Page {page}]: {doc.page_content}\n\n"
    
    return formatted_context


# Create RAG chain
def create_rag_chain(vectorstore):
    print("Loading local LLM (Qwen 0.5B)...")
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=False,
    )
    
    template = """You are a helpful assistant. Use the following context to answer the question.

    Context: {context}

    Question: {question}
    
    Answer:"""
    
    # Wrap it inside LangChain
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    llm_with_stop = llm.bind(stop=["Question:", "\n\n", "<|im_end|>", "<|endoftext|>"])
    
    prompt = PromptTemplate.from_template(template)

    # Custom Retrieval Step to handle scores and logging internally
    def custom_retriever(query):
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        filtered_docs = log_and_filter_chunks(docs_with_scores)
        return format_docs_with_sources(filtered_docs)

    # Guardrail Interceptor
    def guardrailed_response(inputs):
        if inputs["context"] == "NO_CONTEXT":
            return "I don't have enough information to answer this based on the uploaded documents."
        
        # Use the llm_with_stop here
        chain = prompt | llm_with_stop | StrOutputParser()
        return chain.invoke(inputs)

    return (
        {
            "context": RunnableLambda(lambda q: custom_retriever(q)),
            "question": RunnablePassthrough()
        }
        | RunnableLambda(guardrailed_response)
    )

# Execution
if __name__ == "__main__":
    # Point this to an accessible dummy/sample path for tests
    MY_PDF = "Data/LLM research paper 1.pdf" 
    
    # Simple check to help users set up smoothly
    if not os.path.exists(MY_PDF):
        print(f"Warning: Please place a sample PDF file at '{MY_PDF}' to test this execution pipeline natively.")
    else:
        try:
            print("Initialising vector store...")
            vstore = initialise_vector_store(MY_PDF)
            rag_system = create_rag_chain(vstore)
            
            # Test 1: Relevant Question Execution
            test_q1 = "What are the main findings?"
            print(f"\nUser: {test_q1}")
            print("Assistant:", rag_system.invoke(test_q1))
            
            # Test 2: Irrelevant Question Execution (Triggers Failure Handling Rule)
            test_q2 = "How do you make a pizza?"
            print(f"\nUser: {test_q2}")
            print("Assistant:", rag_system.invoke(test_q2))
            
        except Exception as e:
            print(f"\nExecution stopped with error: {e}")
