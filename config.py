import os

def set_openai_api_key(file_path='APIopenAI.txt'):
    """
    Reads the OpenAI API key from the specified file and sets it as an environment variable.
    """
    with open(file_path) as f:
        os.environ['OPENAI_API_KEY'] = f.read().strip()
