import os
import argparse
from utils import save_chunks_csv, log_chunk_lengths, process_document
from chunking_strategies import ChunkingStrategies
from config import set_openai_api_key

# Set up the OpenAI API key
set_openai_api_key()

def process_folder(folder_path, strategy, save_chunks):
    '''Process all files in the folder with the given strategy.'''
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(('.txt', '.pdf', '.md', '.py')):
            print(f"Processing file: {filename}")
            chunks = process_document(file_path, strategy)
            log_chunk_lengths(chunks)
            if save_chunks:
                save_chunks_csv('./data_chunks/' + filename + strategy + '_chunks.csv', chunks)
def main():
    parser = argparse.ArgumentParser(description='Process a document with a specified chunking strategy.')
    parser.add_argument('input_path', type=str, help='The path to the document file or folder')
    parser.add_argument('strategy', type=str, help='The chunking strategy to use',
                        choices=['fixed', 'recursive', 'markdown', 'markdown-recursive', 'pythoncode', 'semantic'])
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=150, help='Chunk overlap')
    parser.add_argument('--separator', type=str, default="\n\n", help='Separator')
    parser.add_argument('--separators', nargs='*', default=["\n\n", "\n", " ", ""], help='List of separators')
    parser.add_argument('--save_chunks', action='store_true', help='Save the chunks to a CSV file')

    args = parser.parse_args()

    # Check if the input path is a file or folder
    if os.path.isfile(args.input_path):
        chunks = process_document(args.input_path, args.strategy, 
                                  args.chunk_size, args.chunk_overlap, 
                                  args.separator, args.separators)
        log_chunk_lengths(chunks)
        if args.save_chunks:
            save_chunks_csv('./data_chunks/' + args.input_path + args.strategy + '_chunks.csv', chunks)
    elif os.path.isdir(args.input_path):
        process_folder(args.input_path, args.strategy, args.save_chunks)
    else:
        raise ValueError(f"Invalid input path: {args.input_path}")

if __name__ == "__main__":
    main()

# Sample execution usage on terminal
# python main.py sample.txt fixed --chunk_size 300 --chunk_overlap 90 --separator "\n\n" --separators "\n\n" "\n" " " "" --save_chunks
