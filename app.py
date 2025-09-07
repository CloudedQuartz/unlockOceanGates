import streamlit as st
import pandas as pd
import altair as alt
from genQuery import build_rag_graph # Import the graph builder from your existing script

# --- Page Configuration ---
st.set_page_config(
    page_title="Ocean Data Query Assistant",
    page_icon="🌊",
    layout="wide"
)

# --- Title and Introduction ---
st.title("🌊 Ocean Data Query Assistant")
st.caption("Ask me anything about the ARGO float data.")

# --- Charting Functions ---
def get_chart(df, chart_type, x_col, y_col=None, color_col=None):
    """Generates an Altair chart."""
    if chart_type == "bar":
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(x_col, sort='-y'),
            y=y_col
        )
    elif chart_type == "line":
        chart = alt.Chart(df).mark_line().encode(
            x=x_col,
            y=y_col
        )
    elif chart_type == "scatter":
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=x_col,
            y=y_col,
            tooltip=df.columns.tolist()
        ).interactive()
    if color_col:
        chart = chart.encode(color=alt.Color(color_col, legend=alt.Legend(title="Color")))
    return chart

def display_charts(df):
    """Dynamically selects and displays charts based on dataframe columns."""
    st.sidebar.header("Query Visualizations")
    cols = df.columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    if "LATITUDE" in cols and "LONGITUDE" in cols:
        st.sidebar.subheader("Float Locations")
        st.sidebar.map(df)

    if len(numeric_cols) >= 2:
        st.sidebar.subheader("Scatter Plots")
        # Prioritize common relationships
        if "TEMP_ADJUSTED" in numeric_cols and "PSAL_ADJUSTED" in numeric_cols:
            st.altair_chart(get_chart(df, "scatter", "TEMP_ADJUSTED", "PSAL_ADJUSTED"), use_container_width=True)
        if "PRES_ADJUSTED" in numeric_cols and "TEMP_ADJUSTED" in numeric_cols:
            st.altair_chart(get_chart(df, "scatter", "PRES_ADJUSTED", "TEMP_ADJUSTED"), use_container_width=True)
        
    if len(numeric_cols) >= 1:
        st.sidebar.subheader("Histograms")
        for col in numeric_cols:
            # Avoid plotting histograms for lat/lon
            if col not in ["LATITUDE", "LONGITUDE", "PLATFORM_NUMBER", "CYCLE_NUMBER"]:
                 chart = alt.Chart(df).mark_bar().encode(
                    alt.X(col, bin=True),
                    y='count()',
                 ).properties(title=f"Distribution of {col}")
                 st.altair_chart(chart, use_container_width=True)


# --- Caching the RAG Graph ---
# This decorator ensures the complex LangGraph object is built only once,
# making the app much faster on subsequent interactions.
@st.cache_resource
def load_rag_app():
    """Builds and returns the compiled RAG LangGraph application."""
    try:
        app = build_rag_graph()
        return app
    except Exception as e:
        st.error(f"Failed to load the RAG application. Please check your configurations. Error: {e}")
        return None

# Load the app
rag_app = load_rag_app()

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sql_query" in message and message["sql_query"]:
            with st.expander("View Generated SQL Query"):
                st.code(message["sql_query"], language="sql")

# --- Main Chat Interface ---
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... 🧠")

        if rag_app:
            try:
                # Prepare the inputs for the RAG graph
                inputs = {
                    "query": prompt,
                    "search_index_name": "vector_index"
                }
                
                # Invoke the RAG pipeline
                final_state = rag_app.invoke(inputs)
                
                # Extract the results
                response = final_state.get('final_response', "Sorry, I couldn't generate a response.")
                sql_query = final_state.get('sql_query', "")
                sql_results_raw = final_state.get('sql_results_raw', [])


                # Display the final answer
                message_placeholder.markdown(response)
                
                # Display the generated SQL query in an expander for transparency
                if sql_query:
                    with st.expander("View Generated SQL Query"):
                        st.code(sql_query, language="sql")

                # --- NEW: Display charts in the sidebar ---
                if sql_results_raw:
                    df = pd.DataFrame(sql_results_raw)
                    display_charts(df)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "sql_query": sql_query,
                    "sql_results_raw": sql_results_raw
                })
            
            except Exception as e:
                error_msg = f"An error occurred while processing your request: {e}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg, "sql_query": "", "sql_results_raw": []})
        else:
            message_placeholder.error("The RAG application is not available. Please check the server logs.")
            st.session_state.messages.append({"role": "assistant", "content": "RAG application is not available.", "sql_query": "", "sql_results_raw": []})
