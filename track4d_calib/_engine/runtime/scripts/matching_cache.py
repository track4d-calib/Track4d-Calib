"""Reuse full-image query correspondences between stages of one calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np


def file_identity(value):
    path = Path(value).resolve()
    stat = path.stat()
    return [str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns]


class QueryMatchCache:
    """Store lossless arrays with image, model and query identities in the key."""

    def __init__(self, directory, *, weights, source):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "schema": 1,
            "weights": file_identity(weights),
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        }
        self.hits = 0
        self.misses = 0

    def match(self, function, *, retain=True, **arguments):
        model = arguments["model"]
        queries = np.ascontiguousarray(arguments["queries"])
        identity = {
            **self.identity,
            "model": {name: getattr(model, name) for name in
                      ("H_lr", "W_lr", "H_hr", "W_hr", "bidirectional", "threshold")},
            "images": [file_identity(arguments[name]) for name in ("static_image", "target_image")],
            "queries": [str(queries.dtype), list(queries.shape),
                        hashlib.sha256(queries.tobytes()).hexdigest()],
            "parameters": {key: value for key, value in arguments.items()
                           if key not in {"model", "queries", "static_image", "target_image"}},
        }
        encoded = json.dumps(identity, sort_keys=True)
        path = self.directory / (hashlib.sha256(encoded.encode()).hexdigest() + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"]))
                if metadata["identity"] != identity:
                    raise ValueError(f"Query cache identity mismatch: {path}")
                result = {key: saved[key] for key in saved.files if key != "metadata"}
            result.update(metadata["values"], inference_seconds=0.0)
            self.hits += 1
            return result
        self.misses += 1
        result = function(**arguments)
        if retain:
            arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
            values = {key: value for key, value in result.items() if key not in arrays}
            metadata = json.dumps({"identity": identity, "values": values})
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez(stream, metadata=np.asarray(metadata), **arrays)
            temporary.replace(path)
        return result
