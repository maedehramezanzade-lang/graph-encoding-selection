import re
from encoders import NAMES

def _decode_set(enc, text):
    if enc in ("adjacency", "incident"):
        return set(int(x) for x in re.findall(r"\d+", text))
    names = NAMES[enc]
    found = set()
    for i, nm in enumerate(names):
        if re.search(rf"\b{re.escape(nm)}\b", text):
            found.add(i)
    return found

def score(enc, rec, output):
    """Return 1 if the model output matches the ground truth else 0. Lenient to extra words."""
    out = output.strip()
    t, ans = rec["task"], rec["answer"]
    if t in ("edge_existence", "cycle_check"):
        m = re.search(r"\b(yes|no)\b", out.lower())
        return int(bool(m) and m.group(1) == ans)
    if t in ("node_degree", "node_count"):
        m = re.search(r"-?\d+", out)
        return int(bool(m) and int(m.group(0)) == ans)
    if t == "connected_nodes":
        m = re.search(r"\[(.*?)\]", out, re.S)
        s = _decode_set(enc, m.group(1) if m else out)
        return int(s == set(ans))
    raise ValueError(t)

def gold_text(enc, rec):
    """Perfect answer string for a record (used by the self-test)."""
    from encoders import label
    t, a = rec["task"], rec["answer"]
    if t == "connected_nodes":
        return "[" + ", ".join(label(enc, i) for i in a) + "]"
    return str(a)
