"""IT5005 Assignment 1: student implementation file.

Implement the functions marked below. Do not modify utils.py or logic_.py.
"""

from utils import *
from logic_ import *






# Do not change this function; it is used to create atomic propositions.
def atom(prefix, r, c, v):
    """prefix is 'Is' or 'Not'. Returns the Expr for e.g. Is3_2_4."""
    return expr(f'{prefix}{r}_{c}_{v}')


def build_general_kb(n, box_h, box_w, givens):
    kb = PropKB()

    # 1. Each cell must contain at least one value.
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            choices = []

            for v in range(1, n + 1):
                choices.append(atom('Is', r, c, v))

            kb.tell(associate('|', choices))

    # 2. Each cell can contain at most one value.
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v1 in range(1, n + 1):
                for v2 in range(v1 + 1, n + 1):
                    kb.tell(
                        ~atom('Is', r, c, v1)
                        | ~atom('Is', r, c, v2)
                    )

    # 3. No two cells in the same row can contain the same value.
    for r in range(1, n + 1):
        for v in range(1, n + 1):
            for c1 in range(1, n + 1):
                for c2 in range(c1 + 1, n + 1):
                    kb.tell(
                        ~atom('Is', r, c1, v)
                        | ~atom('Is', r, c2, v)
                    )

    # 4. No two cells in the same column can contain the same value.
    for c in range(1, n + 1):
        for v in range(1, n + 1):
            for r1 in range(1, n + 1):
                for r2 in range(r1 + 1, n + 1):
                    kb.tell(
                        ~atom('Is', r1, c, v)
                        | ~atom('Is', r2, c, v)
                    )

    # 5. No two cells in the same box can contain the same value.
    for box_r in range(1, n + 1, box_h):
        for box_c in range(1, n + 1, box_w):
            cells = []

            # Collect the coordinates of all cells in this box.
            for r in range(box_r, box_r + box_h):
                for c in range(box_c, box_c + box_w):
                    cells.append((r, c))

            # Consider every pair of distinct cells in this box.
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    r1, c1 = cells[i]
                    r2, c2 = cells[j]

                    for v in range(1, n + 1):
                        kb.tell(
                            ~atom('Is', r1, c1, v)
                            | ~atom('Is', r2, c2, v)
                        )

    # 6. Preserve all given values.
    for (r, c), v in givens.items():
        kb.tell(atom('Is', r, c, v))

    return kb


def build_definite_kb(n, box_h, box_w, givens):
    kb = PropDefiniteKB()

    # 1. Add all given values as facts.
    for (r, c), v in givens.items():
        kb.tell(atom('Is', r, c, v))

    # 2–5. If a cell contains v, eliminate all conflicting possibilities.
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                premise = atom('Is', r, c, v)

                # 2. This cell cannot contain any other value.
                for other_v in range(1, n + 1):
                    if other_v != v:
                        conclusion = atom('Not', r, c, other_v)
                        kb.tell(Expr('==>', premise, conclusion))

                # 3. Other cells in the same row cannot contain v.
                for other_c in range(1, n + 1):
                    if other_c != c:
                        conclusion = atom('Not', r, other_c, v)
                        kb.tell(Expr('==>', premise, conclusion))

                # 4. Other cells in the same column cannot contain v.
                for other_r in range(1, n + 1):
                    if other_r != r:
                        conclusion = atom('Not', other_r, c, v)
                        kb.tell(Expr('==>', premise, conclusion))

                # 5. Other cells in the same box cannot contain v.
                box_r = ((r - 1) // box_h) * box_h + 1
                box_c = ((c - 1) // box_w) * box_w + 1

                for other_r in range(box_r, box_r + box_h):
                    for other_c in range(box_c, box_c + box_w):
                        if (other_r, other_c) != (r, c):
                            conclusion = atom('Not', other_r, other_c, v)
                            kb.tell(Expr('==>', premise, conclusion))

    # 6. If all other values are excluded, infer the remaining value.
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                excluded = []

                for other_v in range(1, n + 1):
                    if other_v != v:
                        excluded.append(atom('Not', r, c, other_v))

                premise = associate('&', excluded)
                conclusion = atom('Is', r, c, v)
                kb.tell(Expr('==>', premise, conclusion))

    return kb


# This helper prepares the knowledge base for inference.
# It builds rule indexes and caches inference results to speed up queries.
# For example, backward chaining can look up rules for a conclusion directly,
# instead of scanning more than 20,000 clauses on every lookup.
def _prepare_inference(kb):
    """Build rule indexes and rebuild the cache when the knowledge base changes."""
    snapshot = tuple(kb.clauses)
    cached = getattr(kb, '_inference_cache', None)
    if cached is not None and cached['snapshot'] == snapshot:
        return cached

    facts = set()
    by_premise = {}
    by_conclusion = {}
    clauses = list(dict.fromkeys(snapshot))
    for clause in clauses:
        premises, conclusion = parse_definite_clause(clause)
        if not premises:
            facts.add(conclusion)
        else:
            by_conclusion.setdefault(conclusion, []).append(tuple(premises))
            for premise in premises:
                by_premise.setdefault(premise, []).append(clause)

    cached = {
        'snapshot': snapshot,
        'clauses': clauses,
        'by_premise': by_premise,
        'by_conclusion': by_conclusion,
        'proven': facts.copy(),
        'unprovable': set(),
        'proofs': {fact: () for fact in facts},
    }
    kb._inference_cache = cached
    return cached


class _IndexedDefiniteKB(PropDefiniteKB):
    """Speed up rule lookup while retaining the provided pl_fc_entails."""

    def __init__(self, kb):
        state = _prepare_inference(kb)
        self.clauses = list(state['clauses'])
        self._by_premise = state['by_premise']

    def clauses_with_premise(self, premise):
        return self._by_premise.get(premise, [])


def solve_full_grid_fc(n, box_h, box_w, givens):
    """Solve each cell using the provided forward-chaining function."""
    kb = _IndexedDefiniteKB(build_definite_kb(n, box_h, box_w, givens))
    solved = dict(givens)
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            if (r, c) in solved:
                continue
            for v in range(1, n + 1):
                query = atom('Is', r, c, v)
                if pl_fc_entails(kb, query):
                    solved[(r, c)] = v
                    # Use the proven conclusion as a fact in subsequent queries.
                    kb.tell(query)
                    break
            else:
                raise ValueError(
                    f'The rules cannot determine the value at row {r}, column {c}.'
                )
    return solved


def pl_bc_entails(kb, query):
    """Use recursive backward chaining with AND premises and OR rule choices."""
    state = _prepare_inference(kb)
    proven = state['proven']
    unprovable = state['unprovable']
    rules = state['by_conclusion']

    if query in proven:
        return True
    if query in unprovable:
        return False

    while True:
        before = len(proven)
        active = set()
        attempted = set()

        def prove(goal):
            if goal in proven:
                return True
            if goal in unprovable or goal in active or goal in attempted:
                return False

            active.add(goal)
            # Try rules with fewer unproven premises first.
            choices = sorted(
                rules.get(goal, []),
                key=lambda premises: sum(p not in proven for p in premises),
            )
            for premises in choices:
                success = True
                for premise in premises:
                    if not prove(premise):
                        success = False
                        break
                if success:
                    active.remove(goal)
                    proven.add(goal)
                    state['proofs'][goal] = premises
                    return True

            active.remove(goal)
            attempted.add(goal)
            return False

        if prove(query):
            return True
        if len(proven) == before:
            # Cache failed goals only when a full pass derives no new facts.
            unprovable.update(attempted)
            return False
        # New facts were derived; reset temporary failure records and retry.


def solve_full_grid_bc(n, box_h, box_w, givens):
    """Try candidate values for each cell using our backward-chaining function."""
    kb = build_definite_kb(n, box_h, box_w, givens)
    solved = dict(givens)
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            if (r, c) in solved:
                continue
            for v in range(1, n + 1):
                if pl_bc_entails(kb, atom('Is', r, c, v)):
                    solved[(r, c)] = v
                    break
            else:
                raise ValueError(
                    f'The rules cannot determine the value at row {r}, column {c}.'
                )
    return solved