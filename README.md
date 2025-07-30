# RAG template
Template files for a RAG implementation. This project contains the implementation of the first step of a RAG, chunking the documentation, using LangChain. It implements different strategies to chunk text based on the following characteristics:

- `markdown splitter` based on markdown headers.
- `fixed splitter` based on a fixed number of characters.
- `recursive splitter` based on a fixed number of characters but recursively finds one that works.
- `python splitter` based on a fixed number of characters splits Python code.

The method is executed in the `main.py` file which runs the `chunking_strategies.py` module.

Explore the code to discover the effect of different chunking strategies combined with parameters like the chunk size, overlap, or the type of separators.

## Usage

```
python main.py sample.txt fixed --chunk_size 300 --chunk_overlap 90 --separator "\n\n" --separators "\n\n" "\n" " " "" --save_chunks
```
