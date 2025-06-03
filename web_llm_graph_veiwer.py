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
st.title("📚 PDF Knowledge Graph Generator")

# Secret headers
headers = {
    "Authorization": st.secrets["auth_token"],
    "Content-Type": "application/json"
}

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# User inputs
context = st.text_input("Context (e.g., poet, lawyer, doctor)", value="lawyer")
pages_from = st.number_input("From Page", min_value=0, value=0)
pages_to = st.number_input("To Page", min_value=1, value=5)

# Generate Button
if uploaded_file and st.button("Generate Knowledge Graph"):
    with st.spinner("Extracting and analyzing..."):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Extract text from PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        os.unlink(pdf_path)  # delete temp file

        # Clip page range
        pages_from = max(0, pages_from)
        pages_to = min(len(docs), pages_to)

        content = "\n".join([doc.page_content for doc in docs[pages_from:pages_to]])

        # Prompt construction
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

        # Make Cohere call (if using HTTP request)
        try:
            payload = {
                "message": prompt,
                "model": "command-xlarge-nightly",
                "temperature": 0
            }
            response = requests.post(
                "https://api.cohere.ai/v1/chat",
                headers=headers,
                json=payload
            )
            result_text = response.json()['text']

        except Exception as e:
            st.error("Failed to get a response from Cohere API.")
            st.exception(e)
            st.stop()

    # Display raw output
    st.subheader("🔍 Raw Output")
    st.text_area("LLM Output", value=result_text, height=200)

    # Parse Entities and Relationships
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

            elif in_relationships and line and "->" in line:
                parts = line.split("->")
                left = parts[0].strip()
                right = parts[1].strip()
                if "-" in left:
                    src, relation = left.split("-", 1)
                    src = src.strip()
                    relation = relation.strip("() ")
                    relationships.append((src, relation, right))

        # Create a directed graph
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
        HtmlFile = open("graph.html", 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=600)

    except Exception as e:
        st.error("❌ Failed to parse and visualize graph.")
        st.exception(e)
