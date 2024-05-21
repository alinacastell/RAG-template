# imports
import os
import csv
import argparse
from langchain_text_splitters import MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter, PythonCodeTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# Set up environmnet API key
with open('APIopenAI.txt') as f:
    os.environ['OPENAI_API_KEY'] = f.read()

# Set up Global parameters
CHUNK_SIZE = 300
CHUNK_OVERLAP = 20
SEPARATOR = "\n\n" # Can be any given character such as "." or "the"
SEPARATORS = ["\n\n", "\n", " ", ""]

# Step 1: Function to read input documents
def read_document(file_path):
    ''' 
    Reads a document based on its extension.
    Returns its content and a boolean indicating if it is a markdown file.
    '''
    _, file_extension = os.path.splitext(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read(), file_extension == '.md', file_extension == '.py'
    
# Step 2: Different chunking strategies
class ChunkingStrategies:
    def __init__(self, separator, separators, chunk_size, chunk_overlap):
        self.separator = separator
        self.separators = separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def markdown_splitter(self, text):
        headers_to_split_on = [("#", "Header 1"),("##", "Header 2"),('###', 'Header 3')]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, 
            strip_headers=False)
        return md_splitter.split_text(text)
        
    
    def fixed_splitter(self, text):
        splitter = CharacterTextSplitter(
            separator=self.separator, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap)
        return splitter.split_text(text)
    
    def recursive_splitter(self, text):
        splitter = RecursiveCharacterTextSplitter(
            separators=self.separators, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap)
        return splitter.split_text(text)

    def markdown_recursive_splitter(self, text):
        text_md = self.markdown_splitter(text)
        splitter = RecursiveCharacterTextSplitter(
            separators=self.separators, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap)
        documents = splitter.split_documents(text_md)
        return [doc.page_content for doc in documents]

    def python_splitter(self, text):
        splitter = PythonCodeTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap)
        return splitter.split_text(text)

def process_document(file_path, strategy):
    '''
    Processes the document and splits it into chunks using the specified strategy.
    Returns a list of text chunks.
    '''
    document_content, is_md, is_py = read_document(file_path)
    chunker = ChunkingStrategies(SEPARATOR, SEPARATORS, CHUNK_SIZE, CHUNK_OVERLAP)

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
        return SemanticChunker(OpenAIEmbeddings())
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

def save_chunks_csv(filename, chunks):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for sentence in chunks:
            writer.writerow([sentence])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a document with a specified chunking strategy.')
    parser.add_argument('file_path', type=str, help='The path to the document file')
    parser.add_argument('strategy', type=str, help='The chunking strategy to use',
                        choices=['fixed', 'recursive', 'markdown', 'markdown-recursive', 'pythoncode', 'semantic'])
    parser.add_argument('save_chunks', type=bool, help=['Save the chunks to a csv file'])
    args = parser.parse_args()

    chunks = process_document(args.file_path, args.strategy)
    if args.save_chunks:
        save_chunks_csv(args.strategy+'_chunks.csv', chunks)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}...")  # Print the first 100 characters of each chunk for brevity