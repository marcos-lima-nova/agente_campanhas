import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ManifestManager:
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def is_already_ingested(self, file_hash: str, filename: str) -> bool:
        """Checks if a file with the same hash and filename has been ingested."""
        if filename in self.data:
            return self.data[filename].get('hash') == file_hash
        return False

    def get_entry(self, filename: str) -> Optional[Dict[str, Any]]:
        return self.data.get(filename)

    def update(self, filename: str, file_hash: str, file_type: str, source_path: str, campaign: str = "General"):
        self.data[filename] = {
            "hash": file_hash,
            "filename": filename,
            "file_type": file_type,
            "source_path": source_path,
            "campaign": campaign,
            "modified_time": os.path.getmtime(source_path),
            "ingested_at": datetime.now().isoformat()
        }
        self._save()
