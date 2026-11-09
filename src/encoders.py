"""Nine graph-to-text encoders following Fatemi et al. (Talk like a Graph), Appendix A.1.
Integer encoders: adjacency, incident. Name-based: friendship, coauthorship, sp, got, social, politician, expert.
NOTE: the paper lists 9 names per set; our graphs have up to 20 nodes, so each list is extended
to 20 distinct names (documented assumption). The first 9 names are the paper's."""

NAMES = {
 "friendship":   ["James","Robert","John","Michael","David","Mary","Patricia","Jennifer","Linda","William","Richard","Joseph","Thomas","Charles","Barbara","Susan","Jessica","Sarah","Karen","Lisa"],
 "sp":           ["Eric","Kenny","Kyle","Stan","Tolkien","Heidi","Bebe","Liane","Sharon","Butters","Token","Craig","Clyde","Wendy","Randy","Timmy","Jimmy","Gerald","Sheila","Linda"],
 "got":          ["Ned","Cat","Daenerys","Jon","Bran","Sansa","Arya","Cersei","Jaime","Tyrion","Robb","Joffrey","Theon","Sam","Brienne","Hodor","Ygritte","Stannis","Varys","Bronn"],
 "politician":   ["Barack","Jimmy","Arnold","Bernie","Bill","Kamala","Hillary","Elizabeth","John","Joe","Donald","George","Ronald","Nancy","Mitt","Mike","Ted","Al","Gerald","Richard"],
 "expert":       list("ABCDEFGHIJKLMNOPQRST"),
}
NAMES["coauthorship"] = NAMES["friendship"]
NAMES["social"] = NAMES["friendship"]

ENCODINGS = ["adjacency","incident","friendship","coauthorship","sp","got","social","politician","expert"]


def _label(enc, i):
    return str(i) if enc in ("adjacency", "incident") else NAMES[enc][i]


def label(enc, i):
    return _label(enc, i)


def _join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + ", and " + xs[-1]


def encode(enc, n, edges):
    L = lambda i: _label(enc, i)
    edges = sorted(tuple(sorted(e)) for e in edges)
    if enc == "adjacency":
        return ("In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.\n"
                f"G describes a graph among nodes {_join(map(str, range(n)))}.\nThe edges in G are: "
                + " ".join(f"({u},{v})" for u, v in edges) + ".")
    if enc == "incident":
        nb = {i: [] for i in range(n)}
        for u, v in edges: nb[u].append(v); nb[v].append(u)
        lines = [f"G describes a graph among nodes {_join(map(str, range(n)))}.\nIn this graph:"]
        for i in range(n):
            if nb[i]:
                lines.append(f"Node {i} is connected to nodes {', '.join(map(str, sorted(nb[i])))}.")
        return "\n".join(lines)
    nodes = _join(L(i) for i in range(n))
    if enc in ("friendship", "coauthorship", "social", "politician"):
        verb = {"friendship": "{a} and {b} are friends.", "coauthorship": "{a} and {b} wrote a paper together.",
                "social": "{a} and {b} are connected.", "politician": "{a} and {b} are connected."}[enc]
        kind = {"friendship": "friendship", "coauthorship": "co-authorship", "social": "social network", "politician": "politician"}[enc]
        body = "\n".join(verb.format(a=L(u), b=L(v)) for u, v in edges)
        return f"G describes a {kind} graph among {nodes}.\nIn this graph:\n{body}"
    if enc in ("sp", "got"):
        kind = "friendship graph"
        body = ", ".join(f"{L(u)} and {L(v)} are friends" for u, v in edges)
        return f"G describes a {kind} among {nodes}.\nIn this graph: {body}."
    if enc == "expert":
        body = "\n".join(f"{L(u)} -> {L(v)}" for u, v in edges)
        return (f"You are a graph analyst and you have been given a graph G among {nodes}.\n"
                f"G has the following undirected edges:\n{body}")
    raise ValueError(enc)


def question_text(enc, task, query):
    L = lambda i: _label(enc, i)
    word = "node" if enc in ("adjacency", "incident") else ("node" if enc == "expert" else "person")
    if task == "edge_existence":
        return f"Is there an edge between {word} {L(query[0])} and {word} {L(query[1])}? Answer Yes or No."
    if task == "node_degree":
        return f"What is the degree of {word} {L(query[0])}? Answer with a single integer."
    if task == "node_count":
        return "How many nodes are there in this graph? Answer with a single integer."
    if task == "connected_nodes":
        return (f"List all the {word}s that are directly connected to {word} {L(query[0])}. "
                "Answer as a list in square brackets, e.g. [a, b].")
    if task == "cycle_check":
        return "Is there a cycle in this graph? Answer Yes or No."
    raise ValueError(task)


def build_prompt(enc, rec):
    return f"{encode(enc, rec['n_nodes'], rec['edges'])}\nQ: {question_text(enc, rec['task'], rec['query'])}\nA:"
