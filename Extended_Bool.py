import os
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import sys

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocess text (Tokenization, case folding, stop word removal, stemming)
def preprocess(text, stem=True):
    # Case folding
    text = text.lower()
    
    # Tokenization and removing non-alphanumeric characters
    tokens = word_tokenize(re.sub(r'\W+', ' ', text))
    
    # Remove stop words and apply stemming (optional based on flag)
    if stem:
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    else:
        tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Create Inverted Index
def build_inverted_index(documents):
    inverted_index = defaultdict(set)
    
    for doc_id, text in documents.items():
        tokens = preprocess(text)
        for token in tokens:
            inverted_index[token].add(doc_id)
    
    return inverted_index

# Create Biword Index (for phrase queries)
def build_biword_index(documents):
    biword_index = defaultdict(set)
    
    for doc_id, text in documents.items():
        # Preprocess the text with stemming to ensure consistency
        tokens = preprocess(text)
        
        # Create biwords
        for i in range(len(tokens) - 1):
            biword = f"{tokens[i]} {tokens[i + 1]}"
            biword_index[biword].add(doc_id)
    
    return biword_index

# Create Positional Index
def build_positional_index(documents):
    positional_index = defaultdict(lambda: defaultdict(list))
    
    for doc_id, text in documents.items():
        tokens = preprocess(text)
        
        # Store the positions of each token in the document
        for position, token in enumerate(tokens):
            positional_index[token][doc_id].append(position)
    
    return positional_index

# Process Boolean Query
def process_query(query, inverted_index):
    query = query.lower()
    terms = re.split(r'\s+(and|or|not)\s+', query)
    
    result = set()
    operator = None
    
    for term in terms:
        if term in ['and', 'or', 'not']:
            operator = term
        else:
            # Process individual term (stemming)
            term_tokens = preprocess(term)
            if term_tokens:
                term_results = inverted_index.get(term_tokens[0], set())
                if operator == 'not':
                    result -= term_results
                elif operator == 'or':
                    result |= term_results
                else:  # Default to 'and'
                    if result:
                        result &= term_results
                    else:
                        result = term_results
    
    return result

# Process Phrase Query (using biword index)
def process_phrase_query(phrase, biword_index):
    # Tokenize and build biwords from the query phrase, apply stemming to be consistent
    tokens = preprocess(phrase)
    
    # Generate biwords
    biwords = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    
    # Get documents containing all biwords
    if not biwords:
        return set()  # Return empty if no valid biwords
    
    result = biword_index.get(biwords[0], set())  # Start with the first biword
    for biword in biwords[1:]:
        result &= biword_index.get(biword, set())  # Intersect with other biword results
    
    return result

# Process Proximity Query 
def process_proximity_query(terms, proximity, positional_index):
    # Preprocess the terms
    terms = preprocess(terms)
    
    if len(terms) < 2:
        return set()  # A proximity query must have at least two terms
    
    # Get the positional data for the first term
    result_docs = positional_index.get(terms[0], {})
    
    for term in terms[1:]:
        next_term_docs = positional_index.get(term, {})
        matching_docs = set()
        
        # Find documents where both terms exist
        for doc_id, positions in result_docs.items():
            if doc_id in next_term_docs:
                term2_positions = next_term_docs[doc_id]
                
                # Check for proximity match
                for pos1 in positions:
                    for pos2 in term2_positions:
                        if abs(pos1 - pos2) <= proximity:
                            matching_docs.add(doc_id)
        
        result_docs = {doc_id: result_docs[doc_id] for doc_id in matching_docs}
    
    return set(result_docs.keys())

# Load documents from 'Corpus' 
def load_documents(corpus_dir='corpus'):
    documents = {}
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

# Main program
if __name__ == "__main__":
    # Load all documents from the Corpus 
    corpus_dir = 'Corpus'  
    documents = load_documents(corpus_dir)

    # Build the indexes
    inverted_index = build_inverted_index(documents)
    biword_index = build_biword_index(documents)
    positional_index = build_positional_index(documents)

    while True:
        print("Select query type:")
        print("1. Boolean Query")
        print("2. Phrase Query")
        print("3. Proximity Query")
        print("4. Exit")
        
        choice = input("Enter choice (1/2/3/4): ").strip()
        
        if choice == '1':
            query = input("Enter Boolean query: ").strip()
            result = process_query(query, inverted_index)
            print(f"Documents matching Boolean query '{query}': {result}")
            sys.stdout.flush()
            
        elif choice == '2':
            phrase = input("Enter Phrase query: ").strip()
            result = process_phrase_query(phrase, biword_index)
            print(f"Documents matching phrase query '{phrase}': {result}")
            sys.stdout.flush()
            
        elif choice == '3':
            terms = input("Enter Proximity query terms: ").strip()
            proximity_distance = int(input("Enter proximity distance: ").strip())
            result = process_proximity_query(terms, proximity_distance, positional_index)
            print(f"Documents matching proximity query '{terms}' within {proximity_distance} words: {result}")
            sys.stdout.flush()
            
        elif choice == '4':
            print("Exiting...")
            sys.stdout.flush()
            break
        
        else:
            print("Invalid choice. Please try again.")
            sys.stdout.flush()
