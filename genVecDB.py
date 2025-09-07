import os
import glob
import xarray as xr
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory containing your ARGO NetCDF files
ARGO_DATA_DIR = "./2024/01" 
# Directory where the vector database will be stored
CHROMA_PERSIST_DIR = "./chroma_db_metadata" 
# Name for the collection within ChromaDB
CHROMA_COLLECTION_NAME = "argo_sql_schema"
# Recommended open-source embedding model
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
    for the specified list of parameters.
    """
    documents = []
    logging.info(f"Processing file: {file_path}")
    try:
        with xr.open_dataset(file_path) as ds:
            for param_name in params_list:
                # Check if the parameter exists as a variable in the NetCDF file
                if param_name in ds.variables:
                    var = ds[param_name]
                    
                    # Extract attributes, providing sensible defaults if they don't exist
                    long_name = var.attrs.get('long_name', f"Data for {param_name}")
                    units = var.attrs.get('units', 'not specified')

                    # Construct the descriptive document for the LLM
                    # This format clearly links the column name to its meaning.
                    doc_string = f"Column '{param_name}': Represents '{long_name}'. Units are '{units}'."
                    documents.append(doc_string)
    except Exception as e:
        logging.error(f"Could not process file {file_path}: {e}")
    
    return documents

def main():
    """Main function to find files, generate documents, and create the vector store."""
    logging.info("🚀 Starting vector database generation from NetCDF metadata...")
    
    nc_files = glob.glob(os.path.join(ARGO_DATA_DIR, "*.nc"))
    if not nc_files:
        logging.error(f"No NetCDF (.nc) files found in '{ARGO_DATA_DIR}'.")
        return

    # Use the first valid NetCDF file to extract the metadata.
    # We only need to do this once, as the metadata is consistent.
    if nc_files:
        documents = create_documents_from_netcdf(nc_files[0], PARAMS_TO_KEEP)
    else:
        documents = []

    # Ensure documents are unique (though they should be already)
    unique_documents = sorted(list(set(documents)))

    if not unique_documents:
        logging.error("No documents were generated. Cannot create vector database. Check if the variables in PARAMS_TO_KEEP exist in your NetCDF files.")
        return

    logging.info(f"Generated {len(unique_documents)} unique documents for embedding.")
    for doc in unique_documents: # Print first 5 as a sample
        print(doc)
    logging.info("-------------------------")

    # Initialize the Hugging Face embedding model
    logging.info(f"Loading Hugging Face model: {HF_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
    
    # Create the Chroma vector store from the documents
    logging.info("Creating and persisting vector store...")
    Chroma.from_texts(
        texts=unique_documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    logging.info(f"✅ Vector database successfully created and saved to '{CHROMA_PERSIST_DIR}'")

if __name__ == "__main__":
    main()