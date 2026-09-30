"""Part C: Streamlit interface for the shared Sudoku solver."""

import json
from pathlib import Path
import time

import streamlit as st

from sudoku_solver import (
    atom,
    build_definite_kb,
    build_general_kb,
    solve_full_grid_fc,
    solve_full_grid_bc,
    pl_bc_entails,
)


ALGORITHMS = {
    'Backward chaining': solve_full_grid_bc,
    'Forward chaining': solve_full_grid_fc,
}


@st.cache_data
def load_puzzles(path):
    """Read puzzle inputs; the app never uses the supplied solutions."""
    with open(path, encoding='utf-8') as file:
        data = json.load(file)
    puzzles = []
    for entry in data['puzzles']:
        givens = {
            tuple(map(int, position.split('_'))): value
            for position, value in entry['givens'].items()
        }
        puzzles.append(givens)
    return data['n'], data['box_h'], data['box_w'], puzzles


def board_html(n, box_h, box_w, givens, values, label):
    """Format a board; givens and inferred values have distinct appearances."""
    rows = []
    for r in range(1, n + 1):
        cells = []
        for c in range(1, n + 1):
            value = values.get((r, c))
            kind = 'given' if (r, c) in givens else 'inferred' if value else 'empty'
            classes = [kind]
            if c % box_w == 0 and c != n:
                classes.append('box-right')
            if r % box_h == 0 and r != n:
                classes.append('box-bottom')
            description = f'Row {r}, column {c}: {value if value else "empty"}'
            cells.append(
                f'<td class="{" ".join(classes)}" aria-label="{description}">'
                f'{value if value else "&nbsp;"}</td>'
            )
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return (
        f'<table class="sudoku-board" aria-label="{label}">'
        '<tbody>' + ''.join(rows) + '</tbody></table>'
    )


def decode_atom(symbol):
    """Read coordinates from the solver's Is/Not proposition names."""
    name = str(symbol)
    prefix = 'Not' if name.startswith('Not') else 'Is'
    r, c, v = map(int, name[len(prefix):].split('_'))
    return prefix, r, c, v


def describe_step(conclusion, premises, n, box_h, box_w, givens):
    """Translate a recorded rule application, without performing inference."""
    prefix, r, c, v = decode_atom(conclusion)
    cell = f'row {r}, column {c}'
    if not premises:
        if prefix != 'Is' or givens.get((r, c)) != v:
            raise ValueError('The proof contains an unexplained starting fact.')
        return 'Given', f'The puzzle gives {v} at {cell}.'

    if prefix == 'Not' and len(premises) == 1:
        source_prefix, sr, sc, sv = decode_atom(premises[0])
        if source_prefix != 'Is':
            raise ValueError('An elimination must be supported by a placed value.')
        source = f'row {sr}, column {sc}'
        if (sr, sc) == (r, c) and sv != v:
            return 'One value per cell', (
                f'{cell.capitalize()} is already {sv}, so it cannot also be {v}.'
            )
        if sv == v and sr == r and sc != c:
            return 'Row elimination', (
                f'{source.capitalize()} is {v}. A row cannot repeat a value, '
                f'so {cell} cannot be {v}.'
            )
        if sv == v and sc == c and sr != r:
            return 'Column elimination', (
                f'{source.capitalize()} is {v}. A column cannot repeat a value, '
                f'so {cell} cannot be {v}.'
            )
        same_box = (
            (sr - 1) // box_h == (r - 1) // box_h
            and (sc - 1) // box_w == (c - 1) // box_w
        )
        if sv == v and same_box and (sr, sc) != (r, c):
            return 'Box elimination', (
                f'{source.capitalize()} is {v}. These cells share a '
                f'{box_h} × {box_w} box, so {cell} cannot be {v}.'
            )

    expected = {('Not', r, c, other) for other in range(1, n + 1) if other != v}
    if prefix == 'Is' and {decode_atom(p) for p in premises} == expected:
        excluded = ', '.join(str(other) for other in range(1, n + 1) if other != v)
        return 'Last remaining candidate', (
            f'At {cell}, the values {excluded} have all been excluded. '
            f'The cell must contain a value from 1 to {n}, so it must be {v}.'
        )
    raise ValueError('The recorded proof contains an unrecognised Sudoku rule.')


def recorded_trace(kb, target, n, box_h, box_w, givens):
    """Keep the actual proof records supporting target, in recorded order."""
    proofs = kb._inference_cache['proofs']
    required = set()
    pending = [target]
    while pending:
        goal = pending.pop()
        if goal in required:
            continue
        required.add(goal)
        pending.extend(proofs[goal])

    steps = []
    step_numbers = {}
    # The solver adds each conclusion only after proving all its premises.
    # Dictionary insertion order preserves the successful proof order.
    for conclusion, premises in proofs.items():
        if conclusion not in required:
            continue
        kind, explanation = describe_step(
            conclusion, premises, n, box_h, box_w, givens
        )
        number = len(steps) + 1
        steps.append({
            'number': number,
            'kind': kind,
            'explanation': explanation,
            'premise_steps': [step_numbers[p] for p in premises],
            'conclusion': str(conclusion),
        })
        step_numbers[conclusion] = number
    return steps


def check_query(n, box_h, box_w, givens, r, c, v):
    """Call the shared BC implementation, then read its recorded proof."""
    start = time.perf_counter()
    kb = build_definite_kb(n, box_h, box_w, givens)
    query = atom('Is', r, c, v)
    verdict = bool(pl_bc_entails(kb, query))
    elapsed = time.perf_counter() - start

    target = query if verdict else None
    excluded = False
    explanation_elapsed = 0.0
    if not verdict:
        # Failure to prove Is is not, by itself, a proof of Not.
        # Ask the same inference engine a separate exclusion query.
        explanation_start = time.perf_counter()
        exclusion = atom('Not', r, c, v)
        excluded = bool(pl_bc_entails(kb, exclusion))
        if excluded:
            target = exclusion
        explanation_elapsed = time.perf_counter() - explanation_start

    return {
        'cell': (r, c, v),
        'verdict': verdict,
        'excluded': excluded,
        'elapsed': elapsed,
        'explanation_elapsed': explanation_elapsed,
        'steps': recorded_trace(kb, target, n, box_h, box_w, givens) if target else [],
    }


def reset_puzzle_results():
    """A new puzzle must not inherit another puzzle's results or timings."""
    for key in ('solve_results', 'last_algorithm', 'query_result', 'trace_page'):
        st.session_state.pop(key, None)


def show_query_result(result):
    r, c, v = result['cell']
    if result['verdict']:
        st.success(f'True — row {r}, column {c} is {v}.')
    else:
        st.warning(f'False — the knowledge base does not entail {v} at row {r}, column {c}.')
        if result['excluded']:
            st.info(
                f'A separate backward-chaining query also proved that this cell '
                f'cannot be {v}. Its supporting proof is shown below.'
            )
        else:
            st.info(
                'These rules proved neither this value nor its exclusion. '
                'False here means “not proved”; it does not mean “proved impossible”.'
            )
    st.caption(
        f'BC query, including knowledge-base construction: {result["elapsed"]:.4f} s. '
        + (
            f'Additional exclusion check: {result["explanation_elapsed"]:.4f} s.'
            if not result['verdict'] else ''
        )
    )

    steps = result['steps']
    if not steps:
        return
    st.subheader('Why this result?')
    st.caption(
        f'{len(steps)} supporting steps, taken from the BC solver’s proof records. '
        'Givens come first; successful deductions follow in recorded order. '
        'Unsuccessful branches and unrelated deductions are omitted.'
    )
    page_size = 12
    page_count = (len(steps) + page_size - 1) // page_size
    page = 0
    if page_count > 1:
        page = st.selectbox(
            'Trace page', range(page_count), key='trace_page',
            format_func=lambda p: (
                f'Steps {p * page_size + 1}–{min((p + 1) * page_size, len(steps))}'
            ),
        )
    for step in steps[page * page_size:(page + 1) * page_size]:
        with st.expander(
            f'Step {step["number"]} · {step["kind"]}',
            expanded=step['number'] == len(steps),
        ):
            st.write(step['explanation'])
            if step['premise_steps']:
                st.caption('Based on step(s): ' + ', '.join(map(str, step['premise_steps'])))
    trace_text = '\n\n'.join(
        f'{step["number"]}. {step["explanation"]}'
        + ('\nBased on step(s): ' + ', '.join(map(str, step['premise_steps']))
           if step['premise_steps'] else '')
        for step in steps
    )
    st.download_button(
        'Download the full explanation', trace_text,
        file_name=f'proof_r{r}_c{c}_v{v}.txt', mime='text/plain',
    )


def main():
    st.set_page_config(page_title='Sudoku · Solve & Understand', page_icon='🧩', layout='wide')
    st.markdown('''
        <style>
        .block-container { max-width: 1120px; padding-top: 2rem; }
        .sudoku-board {
            border-collapse: collapse; table-layout: fixed; width: 100%;
            max-width: 440px; border: 3px solid #334155;
            margin: 0.5rem 0 0.75rem; background: #fff;
        }
        .sudoku-board td {
            text-align: center; vertical-align: middle; width: 11.11%;
            height: 43px; padding: 0; border: 1px solid #cbd5e1;
            font-size: 1.25rem; font-variant-numeric: tabular-nums;
        }
        .sudoku-board .given { background: #e8eef5; color: #172b45; font-weight: 750; }
        .sudoku-board .inferred { color: #087566; font-weight: 500; }
        .sudoku-board .empty { background: #fff; }
        .sudoku-board .box-right { border-right: 3px solid #334155; }
        .sudoku-board .box-bottom { border-bottom: 3px solid #334155; }
        @media (max-width: 600px) {
            .sudoku-board td { height: 34px; font-size: 1.05rem; }
        }
        </style>
    ''', unsafe_allow_html=True)

    st.title('Sudoku: solve & understand')
    st.write('Choose a puzzle, compare two inference methods, and follow the logic behind a cell.')
    try:
        n, box_h, box_w, puzzles = load_puzzles(str(Path(__file__).with_name('puzzles.json')))
    except (OSError, ValueError, KeyError) as error:
        st.error(f'Could not load puzzles.json: {error}')
        st.stop()

    selection, method = st.columns([1, 1])
    with selection:
        selected = st.selectbox(
            'Puzzle', range(len(puzzles)), key='puzzle_index',
            format_func=lambda i: f'Puzzle {i + 1} · {len(puzzles[i])} givens',
            on_change=reset_puzzle_results,
        )
    with method:
        algorithm = st.radio('Solving method', list(ALGORITHMS), horizontal=True)
    givens = puzzles[selected]

    if st.button('Solve full grid', type='primary', key='solve_button'):
        try:
            with st.spinner(f'Solving with {algorithm.lower()}…'):
                start = time.perf_counter()
                solved = ALGORITHMS[algorithm](n, box_h, box_w, givens)
                elapsed = time.perf_counter() - start
            results = st.session_state.setdefault('solve_results', {})
            results[algorithm] = {'grid': solved, 'elapsed': elapsed}
            st.session_state['last_algorithm'] = algorithm
        except (ValueError, RecursionError) as error:
            st.error(f'The solver could not complete this puzzle: {error}')

    results = st.session_state.get('solve_results', {})
    last_algorithm = st.session_state.get('last_algorithm')
    puzzle_column, answer_column = st.columns(2, gap='large')
    with puzzle_column:
        st.subheader('Starting puzzle')
        st.markdown(board_html(n, box_h, box_w, givens, givens, 'Starting puzzle'), unsafe_allow_html=True)
        st.caption(f'{len(givens)} givens · {n * n - len(givens)} empty cells. Shaded, bold cells are givens.')
    with answer_column:
        st.subheader('Solved grid')
        if last_algorithm:
            result = results[last_algorithm]
            st.markdown(board_html(n, box_h, box_w, givens, result['grid'], 'Solved grid'), unsafe_allow_html=True)
            st.caption(f'{last_algorithm} · {result["elapsed"]:.4f} s. Green values were inferred.')
        else:
            st.info('Choose a method and click “Solve full grid” to display its result here.')

    if results:
        fc_column, bc_column = st.columns(2)
        for column, name in ((fc_column, 'Forward chaining'), (bc_column, 'Backward chaining')):
            with column:
                timing = results.get(name)
                st.metric(name, f'{timing["elapsed"]:.4f} s' if timing else 'Not run yet')
        st.caption(
            'Each click runs the selected solver afresh. Times include building the knowledge base '
            'and solving the grid; they exclude rendering. Run both methods to compare this puzzle.'
        )
        if len(results) == 2:
            ratio = results['Forward chaining']['elapsed'] / results['Backward chaining']['elapsed']
            st.write(f'Forward time ÷ backward time: **{ratio:.2f}×** on the latest runs.')

    st.divider()
    st.subheader('Ask about one cell')
    st.write('Can the givens and rules prove this value? Row and column numbers start at 1.')
    with st.form('cell_query'):
        row_column, col_column, value_column = st.columns(3)
        with row_column:
            r = st.number_input('Row', min_value=1, max_value=n, value=1, step=1, key='query_row')
        with col_column:
            c = st.number_input('Column', min_value=1, max_value=n, value=1, step=1, key='query_col')
        with value_column:
            v = st.number_input('Value', min_value=1, max_value=n, value=1, step=1, key='query_value')
        submitted = st.form_submit_button('Check with backward chaining')

    if submitted:
        st.session_state.pop('query_result', None)
        st.session_state.pop('trace_page', None)
        try:
            with st.spinner('Checking the query and collecting its proof…'):
                st.session_state['query_result'] = check_query(n, box_h, box_w, givens, r, c, v)
        except (ValueError, RecursionError, KeyError) as error:
            st.error(f'Could not complete this query: {error}')
    if 'query_result' in st.session_state:
        show_query_result(st.session_state['query_result'])


if __name__ == '__main__':
    main()
