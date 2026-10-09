"""Memory networks with planted overlapping communities: Lancichinetti's trigram sampler.

Physical nodes belong to several communities at once, and the overlap is resolved in
state space: every second-order state ``{i}_j`` belongs to exactly one community, so a
node shared by ``k`` communities is split into states that each belong to one of them.
That is the regime the memory map equation exists for -- a first-order partition cannot
represent the planted structure, a state partition can.

The sampler draws trigrams ``a -> b -> c``:

* ``M = n_nodes * memberships / community_size`` communities of ``community_size``
  physical nodes, every node in ``memberships`` of them (the published parameter sets
  all have ``n_nodes = 4 * community_size``, so ``M = 4 * memberships``);
* ``(1 - mu) * n_trigrams`` trigrams drawn inside one community each, every ordered pair
  ``(a, b)`` owned by the first community that drew it and never reused by another;
* ``mu * n_trigrams`` cross-community trigrams stitched from two existing pairs owned by
  different communities, so mixing adds links but no state nodes.

A trigram becomes the state link ``{a}_b -> {b}_c``, and repeated trigrams add weight.

Ported from ``build_syn_network`` in the higher-order regularization paper project, with
the RNG seeded by ``seed`` so a network is reproducible. Every random draw is the
original's, in the original's order, on a :class:`random.Random`, and state ids are
assigned in the original's order -- so a seeded network has the same states, links and
planted partition as the paper project's script run with the same seed. Needs only the
standard library.
"""

from __future__ import annotations

import random
import warnings

from .._core.state_network import StateNetwork
from ._naming import run_name, settings

#: Rounds of the membership assignment before giving up. The original loops until every
#: membership is placed; a cap only turns a would-be endless loop into an error.
_MAX_ASSIGNMENT_ROUNDS = 10_000


def overlapping_memory_benchmark(
    memberships: int,
    n_trigrams: int,
    *,
    n_nodes: int = 256,
    community_size: int = 64,
    mu: float = 0.1,
    seed: int | None = None,
) -> StateNetwork:
    """Sample a second-order memory network with planted overlapping communities.

    Args:
        memberships: Communities per physical node (``om`` in the published file names).
        n_trigrams: Trigrams to draw (``E``); the link weights sum to this unless the
            cross-community draw runs out of candidates (a warning says so).
        n_nodes: Physical nodes (``N``).
        community_size: Physical nodes per community (``nc``).
        mu: Fraction of the trigrams that cross communities.
        seed: Seeds a :class:`random.Random`; the published files use the sample id.
            ``None`` draws an unreproducible network.

    Returns:
        A second-order :class:`~networks.StateNetwork` over physical nodes ``1..n_nodes``.
        State ids follow the order pairs were planted in; ``partition`` maps each state id
        to its planted community (from 0), and ``info`` holds the parameters and each
        community's physical nodes. Run it directed (``infomap -d``).

    Raises:
        ValueError: The parameters admit no such network.
    """
    if community_size < 3:
        raise ValueError(
            f"community_size must be at least 3 to hold a trigram, got {community_size}."
        )
    if community_size > n_nodes:
        raise ValueError(f"community_size {community_size} exceeds n_nodes {n_nodes}.")
    if memberships < 1:
        raise ValueError(f"memberships must be at least 1, got {memberships}.")
    if (n_nodes * memberships) % community_size:
        raise ValueError(
            f"n_nodes * memberships ({n_nodes * memberships}) must be a multiple of "
            f"community_size ({community_size}) for every node to have {memberships} memberships."
        )
    if not 0.0 <= mu < 1.0:
        raise ValueError(f"mu must be in [0, 1), got {mu}.")
    if n_trigrams < 1:
        raise ValueError(f"n_trigrams must be positive, got {n_trigrams}.")

    rng = random.Random(seed)
    num_communities = n_nodes * memberships // community_size
    communities = _assign_memberships(rng, n_nodes, num_communities, community_size)
    inside, pair_community = _trigrams_inside(rng, communities, int((1 - mu) * n_trigrams))
    across = _trigrams_across(rng, pair_community, int(mu * n_trigrams))

    # States in planting order: the first community to draw a pair owns it. A crossing
    # trigram reuses two planted pairs, so every link lands on a planted state.
    state_of = {pair: state for state, pair in enumerate(pair_community, start=1)}
    links: dict[tuple[int, int], int] = {}
    for a, b, c in inside + across:
        key = (state_of[(a, b)], state_of[(b, c)])
        links[key] = links.get(key, 0) + 1

    parameters = {
        "memberships": memberships,
        "n_trigrams": n_trigrams,
        "n_nodes": n_nodes,
        "community_size": community_size,
        "mu": mu,
        "seed": seed,
    }
    return StateNetwork(
        order=2,
        physical_nodes={node: str(node) for node in range(1, n_nodes + 1)},
        states={state: (a + 1, b + 1) for (a, b), state in state_of.items()},
        links={key: float(weight) for key, weight in links.items()},
        num_paths=len(inside) + len(across),
        name=run_name(
            "overlapping-memory",
            n=n_nodes,
            om=memberships,
            nc=community_size,
            E=n_trigrams,
            mu=mu,
            seed=seed,
        ),
        partition={state_of[pair]: c for pair, c in pair_community.items()},
        info={
            "generator": "overlapping_memory_benchmark",
            "parameters": settings(parameters),
            "communities": {
                c: tuple(node + 1 for node in nodes) for c, nodes in communities.items()
            },
            "cross_trigrams": len(across),
        },
    )


def _assign_memberships(
    rng: random.Random, n_nodes: int, num_communities: int, community_size: int
) -> dict[int, list[int]]:
    """Deal ``community_size`` slots of every community round-robin over shuffled nodes.

    A slot that would give a node the same community twice is dealt again in the next
    round, over a fresh shuffle. Returns community -> nodes (from 0), communities in
    order of their first member.
    """
    slots = [c for c in range(num_communities) for _ in range(community_size)]
    slots = rng.sample(slots, len(slots))
    order = rng.sample(range(n_nodes), n_nodes)
    memberships: list[list[int]] = [[] for _ in range(n_nodes)]
    for _ in range(_MAX_ASSIGNMENT_ROUNDS):
        redealt = []
        for counter, community in enumerate(slots):
            node = order[counter % len(order)]
            if community not in memberships[node]:
                memberships[node].append(community)
            else:
                redealt.append(community)
        slots = redealt
        order = rng.sample(range(n_nodes), n_nodes)
        if not slots:
            break
    else:
        raise RuntimeError(
            f"could not place {len(slots)} memberships in {_MAX_ASSIGNMENT_ROUNDS} rounds."
        )

    communities: dict[int, list[int]] = {}
    for node, mine in enumerate(memberships):
        for community in mine:
            communities.setdefault(community, []).append(node)
    return communities


def _owned_by(
    pair: tuple[int, int], community: int, pair_community: dict[tuple[int, int], int]
) -> bool:
    return pair not in pair_community or pair_community[pair] == community


def _trigrams_inside(
    rng: random.Random, communities: dict[int, list[int]], count: int
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int], int]]:
    """Draw *count* trigrams, each inside one uniformly drawn community.

    A trigram is redrawn while either of its pairs belongs to another community. Returns
    the trigrams and pair -> owning community, in order of first draw.
    """
    pair_community: dict[tuple[int, int], int] = {}
    trigrams = []
    community_ids = list(communities)
    drawn = [rng.sample(community_ids, 1)[0] for _ in range(count)]
    for community in drawn:
        while True:
            a, b, c = rng.sample(communities[community], 3)
            if _owned_by((a, b), community, pair_community) and _owned_by(
                (b, c), community, pair_community
            ):
                trigrams.append((a, b, c))
                pair_community[(a, b)] = community
                pair_community[(b, c)] = community
                break
    return trigrams, pair_community


def _trigrams_across(
    rng: random.Random, pair_community: dict[tuple[int, int], int], count: int
) -> list[tuple[int, int, int]]:
    """Draw *count* trigrams ``a -> b -> c`` whose two pairs belong to different communities.

    Both pairs are drawn from the planted ones, so no state node is added. Gives up after
    ``10 * count`` draws and warns with the shortfall.
    """
    pairs_from: dict[int, list[tuple[int, int]]] = {}
    for pair in pair_community:
        pairs_from.setdefault(pair[0], []).append(pair)
    pairs = list(pair_community)
    trigrams: list[tuple[int, int, int]] = []
    draws = 0
    while len(trigrams) < count:
        draws += 1
        if draws > count * 10:
            warnings.warn(
                f"drew {len(trigrams)} of {count} cross-community trigrams before giving up",
                stacklevel=3,
            )
            break
        first = rng.sample(pairs, 1)[0]
        candidates = [
            p for p in pairs_from.get(first[1], []) if pair_community[p] != pair_community[first]
        ]
        if not candidates:
            continue
        second = rng.sample(candidates, 1)[0]
        trigrams.append((first[0], first[1], second[1]))
    return trigrams
