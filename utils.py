import os
import csv
from chunking_strategies import ChunkingStrategies
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def read_document(file_path):
    ''' 
    Returns the input document's content,
    and booleans indicating if it is a markdown or python file.
    '''
    _, file_extension = os.path.splitext(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read(), file_extension == '.md', file_extension == '.py'

def save_chunks_csv(filename, chunks):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for chunk in chunks:
            writer.writerow([chunk])

def log_chunk_lengths(chunks):
    print(f"Total number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} length: {len(chunk)}")
   
def process_document(file_path, strategy, chunk_size, chunk_overlap, separator, separators):
    '''
    Processes the document and splits it into chunks using the specified strategy.
    Returns a list of text chunks.
    '''
    document_content, is_md, is_py = read_document(file_path)
    chunker = ChunkingStrategies(separator, separators, chunk_size, chunk_overlap)

    if strategy == 'fixed':
        return chunker.fixed_splitter(document_content)
    elif strategy == 'recursive':
        return chunker.recursive_splitter(document_content)
    elif strategy == 'markdown':
        if not is_md:
            raise ValueError(f"Unsupported file format for strategy: {strategy}")
        documents = chunker.markdown_splitter(document_content)
        return [doc.page_content for doc in documents]
    elif strategy == 'markdown-recursive':
        if not is_md:
            raise ValueError(f"Unsupported file format for strategy: {strategy}")
        return chunker.markdown_recursive_splitter(document_content)
    elif strategy == 'pythoncode':
        if not is_py:
            raise ValueError(f"Unsupported file format for strategy: {strategy}")
        return chunker.python_splitter(document_content)
    elif strategy == 'semantic':
        return chunker.semantic_splitter()
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")
