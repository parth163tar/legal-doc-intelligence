import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_cuad(filepath: str) -> list[dict]:
    """Load contracts from CUAD JSON file"""
    with open(filepath) as f:
        data = json.load(f)
    contracts = []
    for item in data['data']:
        text = item['paragraphs'][0]['context']
        contracts.append({
            'title': item['title'],
            'text': text
        })
    print(f"✅ Loaded {len(contracts)} contracts")
    return contracts

def chunk_document(title: str, text: str) -> list[dict]:
    """Split a contract into overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=['\n\n', '\n', '. ', ' ']
    )
    chunks = splitter.split_text(text)
    return [
        {
            'id': f"{title}_chunk_{i}",
            'text': chunk,
            'doc_id': title,
            'chunk_index': i
        }
        for i, chunk in enumerate(chunks)
    ]

def chunk_all_contracts(filepath: str) -> list[dict]:
    """Load and chunk all contracts"""
    contracts = load_cuad(filepath)
    all_chunks = []
    for contract in contracts:
        chunks = chunk_document(contract['title'], contract['text'])
        all_chunks.extend(chunks)
    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = chunk_all_contracts('data/CUADv1.json')
    print(f"\n📄 Sample chunk:")
    print(f"  ID: {chunks[0]['id']}")
    print(f"  Text: {chunks[0]['text'][:200]}")
    print(f"  Chunk index: {chunks[0]['chunk_index']}")