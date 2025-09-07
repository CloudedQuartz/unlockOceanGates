import os
import glob
import xarray as xr
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Suppress the default INFO logging to make our custom prints clearer
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory containing your ARGO NetCDF files
ARGO_DATA_DIR = "./2024/01" 
# Directory where the vector database will be stored
CHROMA_PERSIST_DIR = "./chroma_db_metadata" 
# Name for the collection within ChromaDB
CHROMA_COLLECTION_NAME = "argo_sql_schema"
# Open-source embedding model
HF_MODEL_NAME = "all-MiniLM-L6-v2" 

# The exact list of parameters to include from your SQL schema
PARAMS_TO_KEEP = [
    'PLATFORM_NUMBER', 'PROJECT_NAME', 'PI_NAME', 'CYCLE_NUMBER',
    'DIRECTION', 'DATA_CENTRE', 'DATA_MODE', 'JULD_LOCATION',
    'LATITUDE', 'LONGITUDE', 'PROFILE_PRES_QC', 'PROFILE_TEMP_QC',
    'PROFILE_PSAL_QC', 'PRES', 'PRES_ADJUSTED', 'PRES_ADJUSTED_QC',
    'PRES_ADJUSTED_ERROR', 'TEMP', 'TEMP_ADJUSTED', 'TEMP_ADJUSTED_QC',
    'TEMP_ADJUSTED_ERROR', 'PSAL', 'PSAL_ADJUSTED', 'PSAL_ADJUSTED_QC',
    'PSAL_ADJUSTED_ERROR'
]

def create_documents_from_netcdf(file_path, params_list):
    """
    Reads a single NetCDF file and creates descriptive text documents
    for the specified list of parameters, with verbose printing.
    """
    documents = []
    print(f"\n[DEBUG] --- Entering function `create_documents_from_netcdf` for file: {file_path} ---")
    
    try:
        with xr.open_dataset(file_path) as ds:
            print(f"[DEBUG] Successfully opened dataset from {file_path}")
            
            for param_name in params_list:
                print(f"[DEBUG]   Processing parameter: '{param_name}'")
                
                if param_name in ds.variables:
                    var = ds[param_name]
                    
                    long_name = var.attrs.get('long_name', f"NOT_FOUND: Data for {param_name}")
                    units = var.attrs.get('units', 'NOT_FOUND: not specified')
                    
                    print(f"[DEBUG]     -> Found long_name: '{long_name}'")
                    print(f"[DEBUG]     -> Found units: '{units}'")

                    doc_string = f"Column '{param_name}': Represents '{long_name}'. Units are '{units}'."
                    print(f"[DEBUG]     -> Constructed document: \"{doc_string}\"")
                    documents.append(doc_string)
                else:
                    print(f"[DEBUG]     -> Parameter '{param_name}' not found in this file's variables. Skipping.")

    except Exception as e:
        print(f"[ERROR] Could not process file {file_path}: {e}")
    
    print(f"[DEBUG] --- Exiting function. Generated {len(documents)} documents from this file. ---")
    return documents

def main():
    """Main function with extensive debugging printouts."""
    print("🚀 [START] Beginning vector database generation.")
    print(f"[INFO]  Configuration:")
    print(f"[INFO]    - ARGO Data Directory: {ARGO_DATA_DIR}")
    print(f"[INFO]    - Chroma DB Directory: {CHROMA_PERSIST_DIR}")
    print(f"[INFO]    - Embedding Model: {HF_MODEL_NAME}")
    
    # Find all NetCDF files
    print(f"\n[STEP 1] Searching for NetCDF files in '{ARGO_DATA_DIR}'...")
    nc_files = glob.glob(os.path.join(ARGO_DATA_DIR, "*.nc"))
    
    if not nc_files:
        print(f"❌ [ERROR] No NetCDF (.nc) files found. Halting execution.")
        return
    
    print(f"[INFO]  Found {len(nc_files)} file(s):")
    for f in nc_files:
        print(f"          - {f}")

    # Process the first file to extract metadata
    print("\n[STEP 2] Processing the first file to extract schema metadata...")
    first_file = nc_files[0]
    documents = create_documents_from_netcdf(first_file, PARAMS_TO_KEEP)

    if not documents:
        print("❌ [ERROR] No documents were generated from the file. Cannot create vector database.")
        return

    # De-duplicate documents
    print("\n[STEP 3] Making documents unique...")
    print(f"[INFO]  Number of documents before de-duplication: {len(documents)}")
    unique_documents = sorted(list(set(documents)))
    print(f"[INFO]  Number of documents after de-duplication: {len(unique_documents)}")

    print("\n[INFO]  --- Final list of documents to be embedded: ---")
    for i, doc in enumerate(unique_documents):
        print(f"  {i+1:02d}: \"{doc}\"")
    print("--------------------------------------------------")

    # Initialize the embedding model
    print(f"\n[STEP 4] Initializing Hugging Face embedding model '{HF_MODEL_NAME}'...")
    print("[INFO]  (This may trigger a one-time download of the model if not cached)")
    embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
    print("[INFO]  Embedding model loaded successfully.")

    # Create the Chroma vector store
    print("\n[STEP 5] Creating and persisting the Chroma vector store...")
    Chroma.from_texts(
        texts=unique_documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    print(f"\n✅ [COMPLETE] Vector database successfully created and saved to '{CHROMA_PERSIST_DIR}'")

if __name__ == "__main__":
    main()