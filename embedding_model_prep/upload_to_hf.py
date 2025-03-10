import os
from huggingface_hub import HfApi, create_repo

def upload_to_huggingface(model_dir, repo_name, hf_token=None):
    """
    Upload the embedding model to Hugging Face Hub
    
    Args:
        model_dir: Local directory containing the model files
        repo_name: Name for the repository on Hugging Face (e.g., 'username/model-name')
        hf_token: Hugging Face API token. If None, will look for the HUGGINGFACE_TOKEN environment variable
    """
    # Use token from environment variable if not provided
    if hf_token is None:
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token is None:
            raise ValueError(
                "No Hugging Face token provided. Either pass it as an argument or "
                "set the HUGGINGFACE_TOKEN environment variable."
            )
    
    # Initialize the Hugging Face API
    api = HfApi(token=hf_token)
    
    # Create the repository if it doesn't exist
    try:
        create_repo(repo_name, token=hf_token, private=False, exist_ok=True)
        print(f"Repository {repo_name} is ready")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    
    # Upload all files from the model directory
    file_paths = []
    for root, _, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    
    # Upload the files
    for file_path in file_paths:
        # Get the path relative to the model directory
        relative_path = os.path.relpath(file_path, model_dir)
        
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=relative_path,
                repo_id=repo_name,
                commit_message=f"Upload {relative_path}"
            )
            print(f"Uploaded {relative_path}")
        except Exception as e:
            print(f"Error uploading {relative_path}: {e}")
    
    print(f"Model successfully uploaded to https://huggingface.co/{repo_name}")
    print(f"You can now use it with: `model = WordEmbeddingModel.from_pretrained('{repo_name}')`")

if __name__ == "__main__":
    model_dir = "models/word_embedding_384"  # Directory containing saved model
    repo_name = "tx-smitht/word-embeddings-1830"
    
    # Get the Hugging Face token from environment for security
    # You can set this with: export HUGGINGFACE_TOKEN=your_token
    
    upload_to_huggingface(model_dir, repo_name)