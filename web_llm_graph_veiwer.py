import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from langchain.document_loaders import PyPDFLoader
import cohere
import tempfile
import os
import requests

headers = {
    "authorization" : st.secrets["auth_token"],
    "content-typer" : "application/json"
}

# Initialize Cohere client
co = cohere.Client("YOUR_COHERE_API_KEY")  # Replace with your real API key

st.title("📚 PDF Knowledge Graph Generator")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# User inputs
context = st.text_input("Context (e.g., poet, lawyer, doctor)", value="lawyer")
pages_from = st.number_input("From Page", min_value=0, value=0)
pages_to = st.number_input("To Page", min_value=1, value=5)

if uploaded_file and st.button("Generate Knowledge Graph"):
    with st.spinner("Extracting and analyzing..."):

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Extract text
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        os.unlink(pdf_path)  # delete temp file

        # Page range safety
        pages_from = max(0, pages_from)
        pages_to = min(len(docs), pages_to)

        content = "\n".join([doc.page_content for doc in docs[pages_from:pages_to]])

        # Prompt
        prompt = f"""You are an expert {context}. I need to design a knowledge graph from the following text. 
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

        # Cohere call
        response = co.chat(
            model='command-xlarge-nightly',
            message=prompt,
            temperature=0
        )
        result_text = response.text

    st.subheader("🔍 Raw Output")
    st.text_area("LLM Output", value=result_text, height=200)

    # Parse entities & relationships
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
                entities.append(line.strip("[],\" "))
            elif in_relationships and line and "->" in line:
                parts = line.split("->")
                left = parts[0].strip()
                right = parts[1].strip()
                if "-" in left:
                    src, relation = left.split("-", 1)
                    src = src.strip()
                    relation = relation.strip("() ")
                    relationships.append((src, relation, right))

        # Create Graph
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
        st.error("Failed to parse and generate graph.")
        st.exception(e)
