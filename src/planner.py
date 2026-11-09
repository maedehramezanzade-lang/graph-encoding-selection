"""Query-aware planner: a short prompt that picks one of the nine encodings for a question.
The planner sees only the question (in neutral, integer form) and the graph SIZE, never the graph itself,
so its own token cost stays small. Training-free: it is just a prompt."""
import re, random
from encoders import ENCODINGS, question_text

PREFIX_BALANCED = """You choose how to write a graph as text for another AI that must answer a question about it. Pick the encoding that lets it answer correctly with as few tokens as possible.
Encodings:
adjacency: edge list with node numbers, like (0,1)
incident: one line per node listing its neighbours (numbers)
friendship: sentences "X and Y are friends" (first names)
coauthorship: sentences "X and Y wrote a paper together" (first names)
sp: friend list with South Park names
got: friend list with Game of Thrones names
social: sentences "X and Y are connected" (first names)
politician: sentences "X and Y are connected" (politician names)
expert: arrows "A -> B" with letters
"""
PREFIX_COST = PREFIX_BALANCED + "Shorter encodings are cheaper. Choose the cheapest one unless the question really needs a longer one.\n"
PREFIXES = {"balanced": PREFIX_BALANCED, "cost": PREFIX_COST}
HEAD = PREFIX_BALANCED.split("Encodings:\n")[0]
ENC_LINES = PREFIX_BALANCED.split("Encodings:\n")[1].strip().split("\n")
SUFFIX_FMT = "Graph: {n} nodes, {m} edges.\nQuestion: {q}\nReply with exactly one encoding name.\nEncoding:"


def planner_parts(rec, variant="balanced", hints=""):
    """variants: balanced | cost | shuffled (encoding list in a per-question random order, tests position bias)
    | hinted (adds practice-set results per question type, made by make_hints.py from the HELD-OUT run)."""
    q = question_text("adjacency", rec["task"], rec["query"])   # neutral integer wording
    suffix = SUFFIX_FMT.format(n=rec["n_nodes"], m=rec["n_edges"], q=q)
    if variant in PREFIXES: return PREFIXES[variant], suffix
    lines = list(ENC_LINES)
    if variant == "shuffled":
        random.Random(rec["id"]).shuffle(lines)
        return HEAD + "Encodings:\n" + "\n".join(lines) + "\n", suffix
    if variant == "hinted":
        return HEAD + "Encodings:\n" + "\n".join(lines) + "\n" + hints.strip() + "\n", suffix
    raise ValueError(variant)


def parse_choice(text, default="incident"):
    """First encoding name that appears in the planner output; (choice, parsed_ok)."""
    t = text.lower().replace("co-authorship", "coauthorship").replace("social network", "social")
    best, pos = None, 10**9
    for e in ENCODINGS:
        # tolerate obvious misspellings the models make: "adjacent", "incidence"
        pat = {"adjacency": r"\badjacen\w*", "incident": r"\bincid\w*"}.get(e, rf"\b{e}\b")
        m = re.search(pat, t)
        if m and m.start() < pos: best, pos = e, m.start()
    return (best, True) if best else (default, False)
