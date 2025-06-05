import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from langchain.document_loaders import PyPDFLoader
import tempfile
import os
import requests
import json

# Streamlit app title
st.title("📚 PDF Context-Knowledge Graph Generator")

# Define Cohere API URL and headers
url = "https://api.cohere.ai/v1/chat"  # Update if using a custom endpoint
headers = {
    "Authorization": f"Bearer {st.secrets['auth_token']}",
    "Content-Type": "application/json"
}

# Chat function
def chat(message, preamble=""):
    data = {
        "stream": True,
        "message": message,
        "preamble": preamble
    }

    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    final_response = ""

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            json_data = json.loads(decoded_line)
            if json_data.get("event_type") == "text-generation":
                final_response += json_data.get("text", "")
            elif json_data.get("event_type") == "stream-end":
                break

    return final_response

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# User inputs
context = st.text_input("Context (e.g., poet, lawyer, doctor)", value="lawyer")
pages_from = st.number_input("From Page", min_value=0, value=0)
pages_to = st.number_input("To Page", min_value=1, value=5)

# Generate Button
if uploaded_file and st.button("Generate Knowledge Graph"):
    with st.spinner("Extracting and analyzing..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            # Extract text
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            os.unlink(pdf_path)  # delete temp file

            pages_from = max(0, pages_from)
            pages_to = min(len(docs), pages_to)
            content = "\n".join([doc.page_content for doc in docs[pages_from:pages_to]])

            # Prompt
            prompt = f"""
You are an expert {context}. I need to design a knowledge graph from the following text. 
Please analyze the content, extract the key entities (such as characters, themes, symbols, emotions, etc.), 
and identify the relationships between them. 

For each relationship, specify the type (e.g., ‘is associated with’, ‘is influenced by’, ‘symbolizes’, etc.), 
and link the related entities.

The output should be structured with:

Entities: [list of key elements identified in the content]
Relationships: [list of relationships, each described with the entities involved and the type of relationship, 
e.g., ‘Entity A - (symbolizes) -> Entity B’]

Here is the content to analyze: {content}
"""

            result_text = chat(prompt)

        except Exception as e:
            st.error("❌ Failed to get a response from Cohere API.")
            st.exception(e)
            st.stop()

    # Display raw output
    st.subheader("🔍 Raw Output")
    st.text_area("LLM Output", value=result_text, height=200)

    # Parse and build graph
    try:
        entities = []
        relationships = []

        lines = result_text.splitlines()
        in_entities = False
        in_relationships = False

        for line in lines:
            line = line.strip()
            if line.lower().startswith("entities:"):
                in_entities = True
                in_relationships = False
                continue
            if line.lower().startswith("relationships:"):
                in_entities = False
                in_relationships = True
                continue

            if in_entities and line:
                cleaned = line.strip("[],\" ")
                if cleaned:
                    entities.append(cleaned)

            elif in_relationships and "->" in line:
                parts = line.split("->")
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    if "-" in left:
                        src, relation = left.split("-", 1)
                        src = src.strip()
                        relation = relation.strip("() ")
                        relationships.append((src, relation, right))

        if not entities or not relationships:
            st.warning("No entities or relationships found in the output.")
        else:
            # Create graph
            G = nx.DiGraph()
            for e in entities:
                G.add_node(e)
            for src, rel, dst in relationships:
                G.add_edge(src, dst, label=rel)

            # Visualize using PyVis
            net = Network(height="600px", width="100%", directed=True)
            for node in G.nodes:
                net.add_node(node, label=node)
            for src, dst, data in G.edges(data=True):
                net.add_edge(src, dst, label=data.get("label", ""))

            net.save_graph("graph.html")
            with open("graph.html", 'r', encoding='utf-8') as f:
                html_content = f.read()
                components.html(html_content, height=600)

    except Exception as e:
        st.error("❌ Failed to parse and visualize the graph.")
        st.exception(e)
