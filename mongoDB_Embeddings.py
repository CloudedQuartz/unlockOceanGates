import os
import glob
import xarray as xr
import logging
import psycopg2 # <-- NEW: Import the PostgreSQL driver
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
# Suppress the default INFO logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB Connection Details
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = "rag-db"
COLLECTION_NAME = "rag-chunks"

# --- NEW: PostgreSQL Connection Details ---
# Replace with your actual PostgreSQL credentials
POSTGRES_DBNAME = "argodb"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "passwd"
POSTGRES_HOST = "localhost" # or your DB host
POSTGRES_PORT = "5432"
POSTGRES_TABLE_NAME = "argo_measurements" # The table you populated

# Directory containing your ARGO NetCDF files
ARGO_DATA_DIR = "../ssh25/2024/01/"

# Embedding Model
HF_MODEL_NAME = "all-MiniLM-L6-v2"

# The exact list of parameters to include (for schema)
PARAMS_TO_KEEP = [
    'PLATFORM_NUMBER', 'PROJECT_NAME', 'PI_NAME', 'CYCLE_NUMBER',
    'DIRECTION', 'DATA_CENTRE', 'DATA_MODE', 'JULD_LOCATION',
    'LATITUDE', 'LONGITUDE', 'PROFILE_PRES_QC', 'PROFILE_TEMP_QC',
    'PROFILE_PSAL_QC', 'PRES', 'PRES_ADJUSTED', 'PRES_ADJUSTED_QC',
    'PRES_ADJUSTED_ERROR', 'TEMP', 'TEMP_ADJUSTED', 'TEMP_ADJUSTED_QC',
    'TEMP_ADJUSTED_ERROR', 'PSAL', 'PSAL_ADJUSTED', 'PSAL_ADJUSTED_QC',
    'PSAL_ADJUSTED_ERROR'
]

# --- NEW: Columns for Statistical Summaries ---
CATEGORICAL_COLS = ['PROJECT_NAME', 'PI_NAME', 'DATA_CENTRE', 'DATA_MODE']
NUMERICAL_COLS = ['LATITUDE', 'LONGITUDE', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED', 'PRES_ADJUSTED']


def create_schema_documents(file_path, params_list):
    """
    Reads a single NetCDF file and creates schema documents (without embeddings).
    This function is slightly modified to separate document creation from embedding.
    """
    documents_to_embed = []
    
    try:
        with xr.open_dataset(file_path) as ds:
            print(f"\n[DEBUG] --- Processing file for schema: {file_path} ---")

            for param_name in params_list:
                if param_name in ds.variables:
                    var = ds[param_name]
                    long_name = var.attrs.get('long_name', f"NOT_FOUND: Data for {param_name}")
                    units = var.attrs.get('units', 'NOT_FOUND: not specified')
                    
                    text_to_embed = f"Column '{param_name}': Represents '{long_name}'. Units are '{units}'."

                    mongo_doc = {
                        "type": "schema", # Added a type for clarity
                        "column_name": param_name,
                        "long_name": long_name,
                        "units": units,
                        "text": text_to_embed,
                        "file_source": os.path.basename(file_path)
                    }
                    documents_to_embed.append(mongo_doc)
    
    except Exception as e:
        print(f"[ERROR] Could not process file {file_path} for schema: {e}")
    
    return documents_to_embed

# --- NEW: Function to get a PostgreSQL connection ---
def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=POSTGRES_DBNAME,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        print("[INFO] PostgreSQL connection successful.")
        return conn
    except Exception as e:
        print(f"❌ [ERROR] Could not connect to PostgreSQL: {e}")
        return None

# --- NEW: Function to create statistical summary documents ---
def create_statistical_summary_documents(conn):
    """
    Queries the PostgreSQL database to generate statistical summaries for specified columns.
    """
    if not conn:
        print("❌ [DEBUG] No PostgreSQL connection provided. Skipping summary generation.")
        return []
        
    documents_to_embed = []
    print("\n[STEP] Generating statistical summaries from PostgreSQL...")

    try:
        cursor = conn.cursor()

        # 1. Process Categorical Columns
        print("\n--- [DEBUG] Querying CATEGORICAL columns ---")
        for col in CATEGORICAL_COLS:
            try:
                query = f'SELECT DISTINCT "{col}" FROM {POSTGRES_TABLE_NAME} WHERE "{col}" IS NOT NULL LIMIT 10;'
                print(f"[DEBUG] Executing query: {query}") # <-- ADDED
                cursor.execute(query)
                results = cursor.fetchall()
                print(f"[DEBUG] Results for '{col}': {results}") # <-- ADDED
                
                if results:
                    unique_values = [str(row[0]) for row in results]
                    text_to_embed = f"The column '{col}' contains categorical data. A sample of its unique values includes: {', '.join(unique_values)}."
                    doc = {"type": "summary_categorical", "column_name": col, "text": text_to_embed}
                    documents_to_embed.append(doc)

            except Exception as e:
                print(f"❌ [ERROR] Could not generate summary for categorical column '{col}': {e}")

        # 2. Process Numerical Columns
        print("\n--- [DEBUG] Querying NUMERICAL columns ---")
        for col in NUMERICAL_COLS:
            try:
                query = f'SELECT MIN("{col}"), MAX("{col}"), AVG("{col}") FROM {POSTGRES_TABLE_NAME};'
                print(f"[DEBUG] Executing query: {query}") # <-- ADDED
                cursor.execute(query)
                result = cursor.fetchone()
                print(f"[DEBUG] Result for '{col}': {result}") # <-- ADDED
                
                if result and all(val is not None for val in result):
                    min_val, max_val, avg_val = result
                    text_to_embed = f"The numerical column '{col}' has a value range from {min_val:.2f} to {max_val:.2f}, with an average of {avg_val:.2f}."
                    doc = {"type": "summary_numerical", "column_name": col, "text": text_to_embed}
                    documents_to_embed.append(doc)
            
            except Exception as e:
                print(f"❌ [ERROR] Could not generate summary for numerical column '{col}': {e}")
        
        cursor.close()
    except Exception as e:
        print(f"❌ [FATAL ERROR] An error occurred with the database cursor: {e}")

    print(f"\n[INFO] Generated {len(documents_to_embed)} statistical summary documents.")
    print(f"[DEBUG] Final summary docs list: {documents_to_embed}") # <-- ADDED
    return documents_to_embed

def main():
    """Main function to process files, generate summaries, create embeddings, and upload to MongoDB."""
    print("🚀 [START] Beginning data upload process with schema and summaries.")
    
    # --- STEP 1: Initialize Embedding Model ---
    print(f"\n[STEP 1] Initializing Hugging Face embedding model '{HF_MODEL_NAME}'...")
    embeddings_model = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
    print("[INFO] Embedding model loaded successfully.")

    # --- STEP 2: Generate Schema Documents from NetCDF file ---
    print(f"\n[STEP 2] Searching for NetCDF files in '{ARGO_DATA_DIR}'...")
    nc_files = glob.glob(os.path.join(ARGO_DATA_DIR, "*.nc"))
    if not nc_files:
        print(f"❌ [ERROR] No NetCDF (.nc) files found. Halting execution.")
        return
    first_file_path = nc_files[0]
    schema_documents = create_schema_documents(first_file_path, PARAMS_TO_KEEP)
    
    # --- STEP 3: Generate Statistical Summary Documents from PostgreSQL ---
    pg_conn = get_db_connection()
    summary_documents = create_statistical_summary_documents(pg_conn)
    if pg_conn:
        pg_conn.close()
        print("[INFO] PostgreSQL connection closed.")
        
    # --- STEP 4: Combine and Embed all documents ---
    all_documents_to_embed = schema_documents + summary_documents
    if not all_documents_to_embed:
        print("❌ [ERROR] No documents (schema or summary) were generated. Cannot proceed.")
        return
        
    print(f"\n[STEP 4] Generating embeddings for {len(all_documents_to_embed)} documents...")
    final_mongo_documents = []
    for doc in all_documents_to_embed:
        embedding = embeddings_model.embed_query(doc["text"])
        doc["vector_embeddings"] = embedding
        final_mongo_documents.append(doc)
    
    # --- STEP 5: MongoDB Upload ---
    client = None
    try:
        print("\n[STEP 5] Connecting to MongoDB and uploading documents...")
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        collection.delete_many({}) # Clear existing data
        result = collection.insert_many(final_mongo_documents)
        
        print(f"✅ Successfully inserted {len(result.inserted_ids)} documents into MongoDB.")
        
    except Exception as e:
        print(f"❌ [ERROR] Could not connect to or write to MongoDB: {e}")
    finally:
        if client:
            client.close()
            print("[INFO] MongoDB connection closed.")

if __name__ == "__main__":
    main()