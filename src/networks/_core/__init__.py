"""Machinery shared by every data source.

Caching, downloading, the :class:`Network` handle and the cross-source domain
vocabulary live here so that adding a source means writing a catalog and a
metadata record, not another downloader.
"""

from __future__ import annotations

from .cache import (
    CACHE_ENV_VAR,
    CachedDataset,
    CacheInfo,
    cache_dir,
    cache_info,
    clear_cache,
    dataset_dir,
    file_lock,
    read_sidecar,
    set_cache_dir,
    source_dir,
    write_atomic,
    write_sidecar,
)
from .domains import DOMAINS, domains_for_netzschleuder_tags, domains_for_paths, domains_for_snap
from .download import DownloadError, extract_gzip, fetch_text, head, verify
from .models import DataFile, DatasetLike
from .network import Network
from .state_network import StateNetwork

__all__ = [
    "CACHE_ENV_VAR",
    "DOMAINS",
    "CacheInfo",
    "CachedDataset",
    "DataFile",
    "DatasetLike",
    "DownloadError",
    "Network",
    "StateNetwork",
    "cache_dir",
    "cache_info",
    "clear_cache",
    "dataset_dir",
    "domains_for_netzschleuder_tags",
    "domains_for_paths",
    "domains_for_snap",
    "extract_gzip",
    "fetch_text",
    "file_lock",
    "head",
    "read_sidecar",
    "set_cache_dir",
    "source_dir",
    "verify",
    "write_atomic",
    "write_sidecar",
]
