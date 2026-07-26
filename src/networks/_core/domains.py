"""A single domain vocabulary across every source.

Sources label their data differently: SNAP groups by provenance ("Autonomous
systems graphs", "Product co-purchasing networks"), Netzschleuder tags by
domain. Comparing methods *across kinds of system* -- the thing a paper needs --
requires one vocabulary, so this maps both onto the six domains conventionally
used in the network-science literature.

The mapping is editorial and deliberately visible here rather than buried in
each source, so it can be argued with and corrected in one place.
"""

from __future__ import annotations

__all__ = ["DOMAINS", "domains_for_netzschleuder_tags", "domains_for_snap"]

#: The shared vocabulary, in the order used for reporting.
DOMAINS: tuple[str, ...] = (
    "Social",
    "Informational",
    "Biological",
    "Technological",
    "Transportation",
    "Economic",
)

# SNAP category anchor -> domain. Categories that mix domains (its temporal and
# graph-classification sections cut across everything) map to nothing and let
# the dataset's other categories decide.
_SNAP_CATEGORY_DOMAINS: dict[str, tuple[str, ...]] = {
    "socnets": ("Social",),
    "email": ("Social",),
    "canets": ("Social",),
    "signnets": ("Social",),
    "locnet": ("Social",),
    "actions": ("Social",),
    "twitter": ("Social",),
    "onlinecoms": ("Social",),
    "face2face": ("Social",),
    "citnets": ("Informational",),
    "web": ("Informational",),
    "wikipedia": ("Informational",),
    "amazon": ("Economic",),
    "reviews": ("Economic",),
    "cct": ("Economic",),
    "p2p": ("Technological",),
    "as": ("Technological",),
    "ccn": ("Technological",),
    "tcn": ("Technological",),
    "road": ("Transportation",),
    # Deliberately unmapped: these group by a property, not a domain.
    "temporal": (),
    "disjointgraphs": (),
    "communities": (),
}

# Datasets whose category is unmapped or misleading. SNAP files every com-*
# network under "Networks with ground-truth communities", which says nothing
# about what the network is.
_SNAP_DATASET_DOMAINS: dict[str, tuple[str, ...]] = {
    "com-amazon": ("Economic",),
    "com-dblp": ("Social",),
    "com-livejournal": ("Social",),
    "com-orkut": ("Social",),
    "com-friendster": ("Social",),
    "com-youtube": ("Social",),
    "email-eu-core": ("Social",),
    "cisco-secure-workload": ("Technological",),
    "amazon-meta": ("Economic",),
    "wikispeedia": ("Informational",),
    # Filed only under "Temporal networks", which describes the edges rather
    # than the system.
    "sx-stackoverflow": ("Social",),
    "sx-mathoverflow": ("Social",),
    "sx-superuser": ("Social",),
    "sx-askubuntu": ("Social",),
    "collegemsg": ("Social",),
    "email-eu-core-temporal": ("Social",),
    "wiki-talk-temporal": ("Social", "Informational"),
    "wiki-topcats": ("Informational",),
    # Collections of many small graphs rather than one network.
    "deezer ego-nets": ("Social",),
    "github stargazers": ("Social",),
    "reddit threads": ("Social",),
    "ego-nets": ("Social",),
}

# Netzschleuder tags that are domains. Its other tags describe properties
# (Weighted, Multilayer) or subtypes (Food web, Airport) and are kept as-is on
# the dataset rather than folded into the domain.
_NETZSCHLEUDER_DOMAIN_TAGS = frozenset(DOMAINS)


def _ordered(found: set[str]) -> tuple[str, ...]:
    return tuple(domain for domain in DOMAINS if domain in found)


def domains_for_snap(name: str, categories: tuple[str, ...]) -> tuple[str, ...]:
    """Normalised domains for a SNAP dataset.

    A per-dataset override wins; otherwise every category the dataset is filed
    under contributes, which is why cross-listed datasets can span domains.
    """
    override = _SNAP_DATASET_DOMAINS.get(name.casefold())
    if override is not None:
        return override

    found: set[str] = set()
    for category in categories:
        found.update(_SNAP_CATEGORY_DOMAINS.get(category, ()))
    return _ordered(found)


def domains_for_netzschleuder_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    """Normalised domains for a Netzschleuder network.

    Its tags already use this vocabulary, so this filters rather than
    translates. ``Online``/``Offline`` are social qualifiers, not domains, and
    are excluded.
    """
    return _ordered({tag for tag in tags if tag in _NETZSCHLEUDER_DOMAIN_TAGS})
