import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import sys

# Download necessary NLTK data
nltk.download('punkt')

def tokenize_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def construct_graph(sentences, threshold):
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    
    # Compute cosine similarity between sentences
    cosine_matrix = cosine_similarity(vectors)
    
    graph = nx.Graph()
    
    for i, sentence in enumerate(sentences):
        graph.add_node(i, text=sentence)
    
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            if cosine_matrix[i][j] > threshold:
                graph.add_edge(i, j, weight=cosine_matrix[i][j])
    
    return graph

def main():
    text = input("Enter the text: ")
    threshold = float(input("Enter the similarity threshold (0 to 1): "))
    
    sentences = tokenize_sentences(text)
    graph = construct_graph(sentences, threshold)
    
    print("\nGraph constructed. Nodes and edges are as follows:\n")
    
    for node, data in graph.nodes(data=True):
        print(f"Node {node}: {data['text']}")
    
    print("\nEdges:")
    for u, v, weight in graph.edges(data='weight'):
        print(f"Edge between Node {u} and Node {v} with weight {weight:.2f}")

if __name__ == "__main__":
    main()
