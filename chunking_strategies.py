'''
Module with chunking strategies class.
Returns Python list with all chunks for a given input text.
'''

from langchain_text_splitters import MarkdownHeaderTextSplitter, CharacterTextSplitter,\
                                    RecursiveCharacterTextSplitter, PythonCodeTextSplitter

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