import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

HF_AVAILABLE = None

def _check_hf_availability():
    global HF_AVAILABLE
    if HF_AVAILABLE is not None:
        return HF_AVAILABLE
    try:
        import huggingface_hub
        HF_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Hugging Face Hub not available: {e}")
        HF_AVAILABLE = False
    return HF_AVAILABLE

class HuggingFaceService:
    def __init__(self, token: Optional[str] = None):
        if not _check_hf_availability():
            self.api = None
            self.token = None
            return

        from huggingface_hub import HfApi
        self.token = token or os.environ.get("HUGGINGFACE_TOKEN")
        self.api = HfApi(token=self.token) if self.token else HfApi()
        
    def authenticate(self, token: str):
        """Authenticates with the Hugging Face Hub."""
        if not _check_hf_availability():
            raise ImportError("Hugging Face Hub library not found.")
            
        from huggingface_hub import login, HfApi
        self.token = token
        self.api = HfApi(token=token)
        login(token=token)
        logger.info("Authenticated with Hugging Face Hub.")

    def list_models(self, query: str = None, author: str = None) -> List[Dict[str, Any]]:
        """Lists models on the Hub based on search query or author."""
        if not self.api:
            return []
        models = self.api.list_models(search=query, author=author, limit=10)
        return [{"id": m.id, "author": m.author, "lastModified": m.lastModified} for m in models]

    def upload_model(self, model_path: str, repo_id: str, commit_message: str = "Upload AutoML model", private: bool = True):
        """Uploads a model file or directory to a HF repository."""
        if not self.api:
            raise ImportError("Hugging Face Hub library not found.")
        if not self.token:
            raise ValueError("Authentication token is required for upload.")
            
        repo_url = self.api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        logger.info(f"Hub repository ready: {repo_url}")
        
        if os.path.isdir(model_path):
            self.api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                commit_message=commit_message
            )
        else:
            self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=os.path.basename(model_path),
                repo_id=repo_id,
                commit_message=commit_message
            )
        logger.info(f"Successfully uploaded {model_path} to {repo_id}")

    def download_model(self, repo_id: str, filename: str, local_dir: str = "models/hf_downloads") -> str:
        """Downloads a specific file from a HF repository."""
        if not _check_hf_availability():
            raise ImportError("Hugging Face Hub library not found.")
            
        from huggingface_hub import hf_hub_download
        os.makedirs(local_dir, exist_ok=True)
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        logger.info(f"Downloaded {filename} from {repo_id} to {path}")
        return path

    def consult_model_info(self, repo_id: str) -> Dict[str, Any]:
        """Gets metadata about a model on the Hub."""
        if not self.api:
            return {}
        info = self.api.model_info(repo_id=repo_id)
        return {
            "id": info.id,
            "tags": info.tags,
            "pipeline_tag": info.pipeline_tag,
            "downloads": info.downloads
        }
