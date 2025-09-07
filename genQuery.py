import os
from typing import TypedDict, List, Any
from dotenv import load_dotenv
import logging
import psycopg2 # <-- Import the PostgreSQL driver
from pymongo import MongoClient

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define the Graph State ---
class GraphState(TypedDict):
    """Represents the state of our graph for this task."""
    query: str
    documents: List[Document]
    sql_query: str
    search_index_name: str
    sql_results: str
    sql_results_raw: List[dict[str, Any]] # <-- NEW: To store raw results for charting
    final_response: str # <-- NEW: To store the final natural language answer

# --- Graph Nodes (Functions) ---
def get_collection():
    """Initializes and returns a MongoDB collection object."""
    client = MongoClient(os.environ.get("MONGO_URI", "mongodb+srv://riya9742:riya62446@cluster0.uyxcsox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"))
    db = client[os.environ.get("MONGO_DB_NAME", "rag-db")]
    return db[os.environ.get("MONGO_COLLECTION_NAME", "rag-chunks")]

def set_embedding_model():
    """Initializes and returns the embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=os.environ.get("HF_MODEL_NAME", "all-MiniLM-L6-v2"))

def get_retriever(search_index_name: str, k: int = 10): # Increased k for better context
    """Initializes the retriever for MongoDB Atlas Vector Search."""
    embeddings = set_embedding_model()
    collection = get_collection()

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=search_index_name,
        relevance_score_fn="cosine",
        embedding_key="vector_embeddings"
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever

def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieves documents based on the user's query."""
    query = state.get("query", "")
    search_index_name = state.get("search_index_name", "vector_index")
    logging.info(f"Retrieving documents for query: '{query}'")

    try:
        retriever = get_retriever(search_index_name, k=10) # Fetch more docs for better context
        documents = retriever.invoke(query)
        
        # Log the retrieved documents' content for debugging
        logging.info("--- Retrieved Documents ---")
        for doc in documents:
            logging.info(f"Content: {doc.page_content}")
        logging.info("-------------------------")

        state["documents"] = documents
    except Exception as e:
        logging.error(f"Error during document retrieval: {e}")
        state["documents"] = []
    
    return state

def generate_sql_query(state: GraphState) -> GraphState:
    """Generates a PostgreSQL query from the retrieved schema and summaries."""
    query = state.get("query", "")
    documents = state.get("documents", [])
    
    # Combine the page_content of all documents into a single context string
    schema_context = "\n\n".join([doc.page_content for doc in documents])
    logging.info(f"Generating SQL query with context:\n{schema_context}")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
    
    sql_prompt = ChatPromptTemplate.from_template(
        """You are a PostgreSQL expert. Based on the user's query and the provided database context (which includes schema definitions and data summaries), generate a syntactically correct and efficient PostgreSQL query.

        **Instructions:**
        1.  The table name is `argo_measurements`.
        2.  Use quotes around all column names (e.g., "LATITUDE").
        3.  DO NOT assume the existence of any columns not mentioned in the context.
        4.  Use the data summaries to add specific `WHERE` clauses (e.g., `WHERE "PROJECT_NAME" = 'ARGO'`).
        5.  Always include a `LIMIT` clause (e.g., `LIMIT 20`) unless the user explicitly asks for all data. NEVER EXCEED LIMIT 1000 due to context issues. You might have to structure queries creatively or run multiple queries with one set of SQL commands while making sure the user gets all information asked for.
        6.  The location data is in "LATITUDE" and "LONGITUDE" columns.

        Column 'CYCLE_NUMBER': Represents 'Float cycle number'. Units are 'not specified'.
        Column 'DATA_CENTRE': Represents 'Data centre in charge of float data processing'. Units are 'not specified'.
        Column 'DATA_MODE': Represents 'Delayed mode or real time data'. Units are 'not specified'.
        Column 'DIRECTION': Represents 'Direction of the station profiles'. Units are 'not specified'.
        Column 'JULD_LOCATION': Represents 'Julian day (UTC) of the location relative to REFERENCE_DATE_TIME'. Units are 'not specified'.
        Column 'LATITUDE': Represents 'Latitude of the station, best estimate'. Units are 'degree_north'.
        Column 'LONGITUDE': Represents 'Longitude of the station, best estimate'. Units are 'degree_east'.
        Column 'PI_NAME': Represents 'Name of the principal investigator'. Units are 'not specified'.
        Column 'PLATFORM_NUMBER': Represents 'Float unique identifier'. Units are 'not specified'.
        Column 'PRES': Represents 'Sea water pressure, equals 0 at sea-level'. Units are 'decibar'.
        Column 'PRES_ADJUSTED': Represents 'Sea water pressure, equals 0 at sea-level'. Units are 'decibar'.
        Column 'PRES_ADJUSTED_ERROR': Represents 'Contains the error on the adjusted values as determined by the delayed mode QC process'. Units are 'decibar'.
        Column 'PRES_ADJUSTED_QC': Represents 'quality flag'. Units are 'not specified'.
        Column 'PROFILE_PRES_QC': Represents 'Global quality flag of PRES profile'. Units are 'not specified'.
        Column 'PROFILE_PSAL_QC': Represents 'Global quality flag of PSAL profile'. Units are 'not specified'.
        Column 'PROFILE_TEMP_QC': Represents 'Global quality flag of TEMP profile'. Units are 'not specified'.
        Column 'PROJECT_NAME': Represents 'Name of the project'. Units are 'not specified'. THIS DOES NOT REFER TO ARGO OR NOT ARGO, BUT THE SUBPROJECT COLLECTING THE DATA FOR ARGO
        Column 'PSAL': Represents 'Practical salinity'. Units are 'psu'.
        Column 'PSAL_ADJUSTED': Represents 'Practical salinity'. Units are 'psu'.
        Column 'PSAL_ADJUSTED_ERROR': Represents 'Contains the error on the adjusted values as determined by the delayed mode QC process'. Units are 'psu'.
        Column 'PSAL_ADJUSTED_QC': Represents 'quality flag'. Units are 'not specified'.
        Column 'TEMP': Represents 'Sea temperature in-situ ITS-90 scale'. Units are 'degree_Celsius'.
        Column 'TEMP_ADJUSTED': Represents 'Sea temperature in-situ ITS-90 scale'. Units are 'degree_Celsius'.
        Column 'TEMP_ADJUSTED_ERROR': Represents 'Contains the error on the adjusted values as determined by the delayed mode QC process'. Units are 'degree_Celsius'.
        Column 'TEMP_ADJUSTED_QC': Represents 'quality flag'. Units are 'not          specified'.

        Note: When the user request nearest loaction, strict bounds must not be placed on latitude and longitude. Instead, use order by.

        **Database Context:**
        {schema_context}
        
        **User Query:** {query}
        
        **SQL Query:**"""
    )
    
    sql_chain = (sql_prompt | llm | StrOutputParser())
    
    try:
        raw_output = sql_chain.invoke({"schema_context": schema_context, "query": query})
        
        # --- MODIFIED: Parse potential markdown code block ---
        # The LLM might wrap the SQL query in a markdown block. We need to extract just the query.
        if "```sql" in raw_output:
            # Find the start of the SQL query after the opening fence
            sql_query = raw_output.split("```sql\n")[1].strip()
            # Find the end of the SQL query by removing the closing fence
            if "```" in sql_query:
                sql_query = sql_query.split("```")[0].strip()
        else:
            # If no markdown block is found, assume the output is the raw query
            sql_query = raw_output.strip()

        state["sql_query"] = sql_query
        logging.info(f"Cleaned SQL Query:\n{sql_query}")
    except Exception as e:
        logging.error(f"Error during SQL query generation: {e}")
        state["sql_query"] = "SELECT 'Error generating query';"

    return state

# --- NEW: Node to Execute the SQL Query ---
def execute_sql(state: GraphState) -> GraphState:
    """Connects to PostgreSQL and executes the generated SQL query."""
    sql_query = state.get("sql_query", "")
    logging.info(f"Executing SQL query:\n{sql_query}")
    
    if not sql_query or "Error" in sql_query:
        state["sql_results"] = "Invalid SQL query provided. Cannot execute."
        return state

    conn = None
    try:
        conn = psycopg2.connect(
            dbname=os.environ["POSTGRES_DBNAME"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=os.environ["POSTGRES_PORT"]
        )
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        column_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        # --- NEW: Store raw results as a list of dictionaries ---
        raw_results_list = [dict(zip(column_names, row)) for row in rows]
        state["sql_results_raw"] = raw_results_list
        
        # Format results into a string for the LLM
        if not rows:
            formatted_results = "The query returned no results."
        else:
            header = " | ".join(column_names)
            separator = "-" * len(header)
            data_rows = [" | ".join(map(str, row)) for row in rows]
            formatted_results = "\n".join([header, separator] + data_rows)
            
        state["sql_results"] = formatted_results
        logging.info(f"SQL execution successful. Results:\n{formatted_results}")
        
    except Exception as e:
        error_message = f"Error executing SQL query: {e}"
        logging.error(error_message)
        state["sql_results"] = error_message
        state["sql_results_raw"] = [] # Ensure it's an empty list on error
    finally:
        if conn:
            conn.close()

    return state

# --- NEW: Node to Generate the Final Natural Language Response ---
def generate_final_response(state: GraphState) -> GraphState:
    """Uses an LLM to generate a final response based on the SQL results."""
    query = state.get("query", "")
    sql_results = state.get("sql_results", "")
    logging.info("Generating final natural language response.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    
    response_prompt = ChatPromptTemplate.from_template(
        """Based on the user's original question and the data retrieved from the database, provide a clear, concise, and friendly answer.

        **Original Question:**
        {query}

        **Database Results:**
        {sql_results}

        **Final Answer:**"""
    )

    response_chain = (response_prompt | llm | StrOutputParser())
    
    try:
        final_response = response_chain.invoke({"query": query, "sql_results": sql_results})
        state["final_response"] = final_response
        logging.info(f"Final response generated:\n{final_response}")
    except Exception as e:
        logging.error(f"Error generating final response: {e}")
        state["final_response"] = "I was unable to generate a final response due to an error."

    return state

def build_rag_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Add nodes to the graph
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate_sql", generate_sql_query)
    workflow.add_node("execute_sql", execute_sql) # <-- NEW
    workflow.add_node("generate_final_response", generate_final_response) # <-- NEW

    # Define the flow of the graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql") # <-- NEW
    workflow.add_edge("execute_sql", "generate_final_response") # <-- NEW
    workflow.add_edge("generate_final_response", END) # <-- NEW

    app = workflow.compile()
    logging.info("LangGraph workflow compiled successfully.")
    return app

# --- How to use the graph ---
if __name__ == "__main__":
    app = build_rag_graph()
    
    # Example usage:
    user_query = "What are the floats closest to the Andaman Islands with salinity above 4 psu and temperature below 24 degrees Celsius?"
    
    print(f"User Query: {user_query}\n")
    
    inputs = {
        "query": user_query, 
        "search_index_name": "vector_index" 
    }
    
    # The invoke method will run the full graph
    final_state = app.invoke(inputs)
    
    print("\n--- Final Output ---")
    print(f"Generated SQL Query:\n{final_state.get('sql_query')}\n")
    print(f"SQL Execution Results:\n{final_state.get('sql_results')}\n")
    print(f"Final Answer:\n{final_state.get('final_response')}")
