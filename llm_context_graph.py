import networkx as nx
import cohere
from langchain.document_loaders import WebBaseLoader, PyPDFLoader


co = cohere.Client("API_KEYS")
pages_from = 23 
pages_to = 100 #input
context = "LAWYER" #input


def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def generate_graph(prompt, temp = 0):
  response = co.chat(
      model = 'command-xlarge-nightly',
      message = prompt,
      temperature = temp
  )
  return response.text

content = "\n".join([doc.page_content for doc in docs[pages_from:pages_to]])

prompt = f"""You are an expert {context}. I need to design a knowledge graph from the following text. Please analyze the content, extract the key entities (such as characters, themes, symbols, emotions, etc.), and identify the relationships between them. For each relationship, specify the type (e.g., ‘is associated with’, ‘is influenced by’, ‘symbolizes’, etc.), and link the related entities.

The output should be structured with:

Entities: [list of key elements identified in the content, such as characters, emotions, themes, etc.]
Relationships: [list of relationships, each described with the entities involved and the type of relationship, e.g., ‘Entity A - (symbolizes) -> Entity B’]
Here is the content to analyze: {content}



"""

text = generate_graph(prompt)
