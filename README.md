# unlockOceanGates

unlockOceanGates — SIH Problem Statement #25040

An AI-powered platform to simplify access, exploration and visualization of ARGO ocean datasets. This repository contains a proof-of-concept pipeline, ingestion utilities and a lightweight dashboard that together enable non-experts (students, educators, policy makers) to query and explore oceanographic data without needing domain-specific tooling.

## Proposed solution (high level)
- End-to-end pipeline to process ARGO NetCDF data into queryable stores.
- Linked storage:
  - Vector DB (MongoDB with vector-capable index) for RAG( embeddings / semantic search).
  - LangGraph and LangChain components for RAG pipeline.
  - Relational DB (PostgreSQL) for normalized tabular records and analytics.
- Backend LLM + RAG: natural-language queries → SQL (and semantic retrieval) to answer user questions and generate visualizations.
- Streamlit-based dashboard for interactive visualization and conversational exploration.
- Conversational interface that guides users through data exploration (follow-up questions, clarifications).
- Proof-of-Concept: Indian Ocean ARGO NetCDF → PostgreSQL (scalable to BGC, gliders, buoys, satellites).

Benefits
- Accessibility: lowers barrier so students and laypeople can explore ocean data.
- Social: encourages free distribution and awareness of climate data.
- Environmental: helps improve public understanding of climate change and ocean cycles.
- Education: simplifies scientific data for outreach and teaching.

## Architecture (concise)
1. NetCDF processor: parse ARGO NetCDF, extract observations and metadata.
2. Tabular ingest: write cleaned/normalized rows to PostgreSQL.
3. Semantic ingest: create document chunks + embeddings, store in MongoDB vector index.
4. RAG layer: retrieve relevant documents + tabular rows, pass to LLM to synthesize answers and generate SQL when needed.
5. UI: Streamlit dashboard + conversational interface that issues queries to the backend and renders charts/maps.


## Input / Output (overview)
- Input: ARGO NetCDF files (NetCDF4/CF-compliant), optional metadata CSVs.
- Intermediate: cleaned parquet/CSV, Relational DB, embeddings (vectors), DB tables.
- Output: SQL-queryable tables, semantic search results, dashboards and visualizations (maps, timeseries), conversational QA responses.
