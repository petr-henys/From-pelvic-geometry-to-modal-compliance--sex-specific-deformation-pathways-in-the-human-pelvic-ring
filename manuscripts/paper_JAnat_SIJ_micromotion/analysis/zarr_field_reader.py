"""Synchronous read-only access to the archived Zarr-v3 vector field chunks.

The archive has float64 bytes/zstd codecs and complete spatial dimensions per
chunk. Explicit checks prevent silently applying this reader to other layouts.
Avoids an asynchronous LocalStore hang in the current postprocessing runtime.
"""
from pathlib import Path
import json
import numpy as np
import zstandard


class ArchivedVectorFields:
    def __init__(self, directory):
        self.directory = Path(directory) / "data"
        m = json.loads((self.directory / "zarr.json").read_text())
        self.shape = tuple(m["shape"])
        self.chunks = tuple(m["chunk_grid"]["configuration"]["chunk_shape"])
        assert m["zarr_format"] == 3 and m["node_type"] == "array"
        assert m["data_type"] == "float64"
        assert m["chunk_grid"]["name"] == "regular"
        assert len(self.shape) == 3 and self.shape[-1] == 3
        assert self.chunks[1:] == self.shape[1:]
        assert m["chunk_key_encoding"] == {"name": "default", "configuration": {"separator": "/"}}
        assert [c["name"] for c in m["codecs"]] == ["bytes", "zstd"]
        assert m["codecs"][0]["configuration"]["endian"] == "little"
        assert not m.get("storage_transformers")
        self.codec = zstandard.ZstdDecompressor()
        self.cached_chunk = None
        self.cached_index = None

    def __getitem__(self, index):
        if not 0 <= index < self.shape[0]:
            raise IndexError(index)
        chunk, offset = divmod(index, self.chunks[0])
        if chunk != self.cached_index:
            path = self.directory / "c" / str(chunk) / "0" / "0"
            if not path.exists():
                raise FileNotFoundError(f"Required scientific field chunk missing: {path}")
            expected = int(np.prod(self.chunks))*8
            decoded = self.codec.decompress(path.read_bytes(), max_output_size=expected)
            if len(decoded) != expected:
                raise ValueError(f"Unexpected decoded chunk length: {path}")
            self.cached_chunk = np.frombuffer(decoded, dtype="<f8").reshape(self.chunks)
            self.cached_index = chunk
        result = self.cached_chunk[offset]
        if not np.isfinite(result).all():
            raise ValueError(f"Non-finite displacement for subject {index}")
        return result
