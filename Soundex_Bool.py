import os
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize stop words
stop_words = set(stopwords.words('english'))

# Soundex Generator Function
def soundex_generator(token):
    token = token.upper()
    soundex_result = token[0]  # Retain the first letter

    # Remove h's and w's
    token = re.sub('[hw]', '', token, flags=re.I)

    # Replace consonants with values
    token = re.sub('[bfpv]+', '1', token, flags=re.I)
    token = re.sub('[cgjkqsxz]+', '2', token, flags=re.I)
    token = re.sub('[dt]+', '3', token, flags=re.I)
    token = re.sub('l+', '4', token, flags=re.I)
    token = re.sub('[mn]+', '5', token, flags=re.I)
    token = re.sub('r+', '6', token, flags=re.I)

    # Remove vowels and y's
    token = re.sub('[aeiouhy]', '', token, flags=re.I)

    # Take the first 4 digits
    soundex_result += token[:4].ljust(4, '0')

    # Pad with zeros if needed
    soundex_result = soundex_result.ljust(4, '0')

    return soundex_result

# Preprocess text (Tokenization, stop word removal, and Soundex)
def preprocess(text):
    text = text.lower()

    # Tokenize text into words (basic tokenization)
    tokens = word_tokenize(re.sub(r'\W+', ' ', text))

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]

    # Generate Soundex codes for tokens
    soundex_tokens = [soundex_generator(word) for word in tokens]

    return tokens, soundex_tokens

# Create Inverted Index with Soundex
def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: {'tokens': set(), 'soundex': set()})
    soundex_mapping = defaultdict(set)
    
    for doc_id, text in documents.items():
        # Preprocess each document's text using Soundex and stop word removal
        tokens, soundex_tokens = preprocess(text)

        # Update the inverted index for both tokens and Soundex tokens
        for token in tokens:
            inverted_index[token]['tokens'].add(doc_id)
        for soundex_token in soundex_tokens:
            inverted_index[soundex_token]['soundex'].add(doc_id)
            soundex_mapping[soundex_token].update(tokens)
    
    return inverted_index, soundex_mapping

# Process Boolean Query with Soundex
def process_query(query, inverted_index):
    query = query.lower()
    terms = re.split(r'\s+(and|or|not)\s+', query)
    
    result = set()
    operator = None
    
    matched_tokens = defaultdict(set)
    
    for term in terms:
        if term in ['and', 'or', 'not']:
            operator = term
        else:
            term_tokens, soundex_tokens = preprocess(term)
            
            term_results = set()
            if term_tokens:
                term_results.update(inverted_index.get(term_tokens[0], {}).get('tokens', set()))
                for token in term_tokens:
                    matched_tokens[token].update(inverted_index.get(token, {}).get('tokens', set()))
            if soundex_tokens:
                term_results.update(inverted_index.get(soundex_tokens[0], {}).get('soundex', set()))
                for soundex_token in soundex_tokens:
                    matched_tokens[soundex_token].update(inverted_index.get(soundex_token, {}).get('tokens', set()))
            
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

# Load documents from 'corpus' directory
def load_documents(corpus_dir='corpus'):
    documents = {}
    for filename in os.listdir(corpus_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as file:
                documents[filename] = file.read()
    return documents

# Save Inverted Index
def save_inverted_index(inverted_index, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for key, value in inverted_index.items():
            file.write(f"{key} {len(value['tokens'])} {len(value['soundex'])}\n")
            file.write('tokens: ')
            for doc_id in value['tokens']:
                file.write(f"{doc_id} ")
            file.write('\n')
            file.write('soundex: ')
            for doc_id in value['soundex']:
                file.write(f"{doc_id} ")
            file.write('\n\n')

# Save Soundex Inverted Index
def save_soundex_index(inverted_index, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for key, value in inverted_index.items():
            file.write(f"{key} {len(value['soundex'])}\n")
            file.write('soundex: ')
            for doc_id in value['soundex']:
                file.write(f"{doc_id} ")
            file.write('\n\n')

# Save Soundex Mapping
def save_soundex_mapping(soundex_mapping, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for key, value in soundex_mapping.items():
            file.write(f"{key} {' '.join(value)}\n")

def main():
    corpus_dir = 'Corpus' 
    documents = load_documents(corpus_dir)

    inverted_index, soundex_mapping = build_inverted_index(documents)

    save_inverted_index(inverted_index, 'inv.txt')
    save_soundex_index(inverted_index, 'son.txt')
    save_soundex_mapping(soundex_mapping, 'soundex_mapping.txt')

    while True:
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        result = process_query(query, inverted_index)
        print(f"Documents matching '{query}': {result}")

if __name__ == "__main__":
    main()
