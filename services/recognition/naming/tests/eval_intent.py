"""Evaluate the naming listener (vlm.listen) against a private case file on a live VLM endpoint.

  python3 tests/eval_intent.py --cases DATA_ROOT/naming-eval/intent_cases.jsonl \
      --known DATA_ROOT/naming-eval/known.json --url http://127.0.0.1:22410/v1 --model qwen3.6-35b-a3b [--json out.json]

The cases are real speech from the owner's recordings, so they live outside the repository (on Spark 2 under
DATA_ROOT/naming-eval, mode 0600). Each line is one case: {"text", "context": [[speaker, text], ...], "intent",
"name", optional "accept", "kind", "known", "people", "expect_known", "allow_known"}.

Standard library only. vlm.py is imported by file path (default: ../vlm.py next to this script), so the pair can be
copied anywhere. "Acted" metrics treat any prediction with confidence < --min-confidence as "none", mirroring the
service, which only acts on non-none intents at or above that confidence. A case may list other acceptable intents in
"accept" (e.g. a repeat of a known name is fine as name or same_as: the service acts the same); they count as correct.
Expected "name" values are informational: the names the model returned are listed for review, not scored.

The listener also sees the objects and people Eleagent knows; "kind" (default "object") is checked for acted cases. A
case may give them ("known": [[id, name], ...], "people") and the one it means ("expect_known"); otherwise it gets the
--known file's lists (memory names as sleep writes them) plus, for same_as cases, an object carrying the expected name,
which it should pick. Picks are scored separately.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
INTENTS = ('name', 'rename', 'same_as', 'none')


def load_vlm(path):
    spec = importlib.util.spec_from_file_location('naming_vlm', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cases(path):
    cases = []
    with open(path, encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                case = json.loads(line)
                assert case['intent'] in INTENTS, f'line {number}: bad intent {case["intent"]!r}'
                assert all(a in INTENTS for a in case.get('accept') or ()), f'line {number}: bad accept'
                case['line'] = number
                cases.append(case)
    return cases


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))] if ordered else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default='http://127.0.0.1:22410/v1')
    parser.add_argument('--model', default='qwen3.6-35b-a3b')
    parser.add_argument('--cases', required=True, help='private JSONL case file')
    parser.add_argument('--known', help='private JSON {"known": [[id, name], ...], "people": [[id, name], ...]}')
    parser.add_argument('--vlm-module', default=str(HERE.parent / 'vlm.py'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--min-confidence', type=float, default=0.6)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--json', help='write per-case results and summary here')
    args = parser.parse_args()

    vlm = load_vlm(args.vlm_module)
    cases = load_cases(args.cases)
    lists = json.loads(pathlib.Path(args.known).read_text(encoding='utf-8')) if args.known else {}
    known_default, people_default = lists.get('known', []), lists.get('people', [])

    def known_of(case):
        if 'known' in case:
            return [{'id': i, 'name': n} for i, n in case['known']], case.get('expect_known')
        known = [{'id': i, 'name': n, 'name_source': 'eleagent'} for i, n in known_default]   # sleep's descriptive names
        if case['intent'] == 'same_as' and case.get('name') and case.get('kind', 'object') == 'object':
            return known + [{'id': 'object:said', 'name': case['name']}], 'object:said'
        return known, case.get('expect_known')

    def run(case):
        context = [tuple(turn) for turn in case.get('context') or ()]
        known, expect_known = known_of(case)
        case = {**case, 'expect_known': expect_known}
        try:
            people = [{'id': i, 'name': n} for i, n in case['people']] if 'people' in case else \
                     [{'id': i, 'name': n, 'name_source': 'eleagent'} for i, n in people_default]
            got, seconds = vlm.listen(args.url, args.model, case['text'], context, known, timeout=args.timeout, people=people)
            error = None
        except Exception as exc:  # count as a "none" prediction, report separately
            got, seconds, error = {'intent': 'none', 'name': None, 'known_id': None, 'referent': None, 'confidence': 0.}, None, repr(exc)[:200]
        acted = got['intent'] if got['intent'] != 'none' and got['confidence'] >= args.min_confidence else 'none'
        if acted != case['intent'] and acted in (case.get('accept') or ()):
            acted = case['intent']  # an accepted alternative
        if got['intent'] != case['intent'] and got['intent'] in (case.get('accept') or ()):
            got = {**got, 'intent': case['intent'], 'accepted_as': got['intent']}
        return {**case, 'got': got, 'acted': acted, 'seconds': seconds, 'error': error}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(run, cases))
    wall = time.perf_counter() - started

    print(f'== misses (vlm={args.vlm_module}, model={args.model}, min_confidence={args.min_confidence})')
    for r in results:
        raw_ok, acted_ok = r['got']['intent'] == r['intent'], r['acted'] == r['intent']
        if raw_ok and acted_ok and not r['error']:
            continue
        tag = 'MISS' if not acted_ok else 'raw-miss'
        if r['intent'] == 'none' and r['acted'] != 'none':
            tag = 'FALSE-ACTION'
        context = ' | '.join(f'[{s}] {t}' for s, t, *_ in r.get('context') or ())
        print(f"{tag:<12} L{r['line']:<3} want={r['intent']:<7} acted={r['acted']:<7} "
              f"got={json.dumps(r['got'], ensure_ascii=False)} :: {r['text']}" + (f'   (ctx: {context})' if context else '')
              + (f"   ERROR {r['error']}" if r['error'] else ''))

    named = [r for r in results if r['acted'] != 'none' and r.get('name')]
    if named:
        print('\n== names (expected -> returned)')
        for r in named:
            mark = '  ' if r['got'].get('name') == r['name'] else '~ '
            print(f"{mark}L{r['line']:<3} {r['name']} -> {r['got'].get('name')}   :: {r['text']}")

    kinds = [r for r in results if r['acted'] != 'none' and r['got'].get('kind', 'object') != r.get('kind', 'object')]
    print(f"\n== kind (person/object) of acted cases: {len(kinds)} wrong")
    for r in kinds:
        print(f"  L{r['line']:<3} want {r.get('kind', 'object')} got {r['got'].get('kind')} :: {r['text']}")
    picks = [r for r in results if r.get('expect_known') and r['acted'] != 'none']
    wrong_picks = [r for r in results if r['acted'] != 'none' and r['got'].get('known_id')
                   and r['got']['known_id'] not in (r.get('expect_known'), *(r.get('allow_known') or ()))]
    print(f"\n== known-object picks: {sum(r['got'].get('known_id') == r['expect_known'] for r in picks)}/{len(picks)} expected picks right; "
          f"{len(wrong_picks)} picks of an object the case did not mean")
    for r in wrong_picks + [r for r in picks if r['got'].get('known_id') != r['expect_known']]:
        print(f"  L{r['line']:<3} want {r.get('expect_known')} got {r['got'].get('known_id')} :: {r['text']}")

    n = len(results)
    raw_correct = sum(r['got']['intent'] == r['intent'] for r in results)
    acted_correct = sum(r['acted'] == r['intent'] for r in results)
    matrix = {want: {got: 0 for got in INTENTS} for want in INTENTS}
    for r in results:
        matrix[r['intent']][r['acted']] += 1
    expected_none = [r for r in results if r['intent'] == 'none']
    false_actions = sum(r['acted'] != 'none' for r in expected_none)
    per_class = {}
    for intent in INTENTS:
        support = sum(matrix[intent].values())
        predicted = sum(matrix[w][intent] for w in INTENTS)
        hit = matrix[intent][intent]
        per_class[intent] = {'support': support, 'recall': hit / support if support else None,
                             'precision': hit / predicted if predicted else None}
    latencies = [r['seconds'] for r in results if r['seconds'] is not None]
    errors = sum(bool(r['error']) for r in results)

    print('\n== summary')
    print(f'cases {n}  errors {errors}  wall {wall:.1f}s  workers {args.workers}')
    print(f'accuracy raw   {raw_correct}/{n} = {raw_correct / n:.1%}')
    print(f'accuracy acted {acted_correct}/{n} = {acted_correct / n:.1%}')
    print(f'false actions on expected-none: {false_actions}/{len(expected_none)}')
    name_recall = per_class['name']['recall']
    print(f"name recall (acted): {matrix['name']['name']}/{per_class['name']['support']} = "
          f"{name_recall:.1%}" if name_recall is not None else 'name recall: n/a')
    print('\nconfusion (acted; rows=expected, cols=predicted)')
    print(f"{'':>9}" + ''.join(f'{g:>9}' for g in INTENTS))
    for want in INTENTS:
        print(f'{want:>9}' + ''.join(f'{matrix[want][g]:>9}' for g in INTENTS))
    print('\nper class (acted)')
    for intent, stats in per_class.items():
        fmt = lambda v: f'{v:.1%}' if v is not None else '  n/a'
        print(f"{intent:>9}  support {stats['support']:>3}  recall {fmt(stats['recall']):>6}  precision {fmt(stats['precision']):>6}")
    if latencies:
        print(f'\nlatency median {statistics.median(latencies):.2f}s  p90 {percentile(latencies, 0.9):.2f}s  '
              f'max {max(latencies):.2f}s  (n={len(latencies)}, concurrent workers={args.workers})')

    if args.json:
        summary = {'cases': n, 'errors': errors, 'accuracy_raw': raw_correct / n, 'accuracy_acted': acted_correct / n,
                   'false_actions_on_none': false_actions, 'expected_none': len(expected_none), 'name_recall': name_recall,
                   'confusion_acted': matrix, 'per_class': per_class, 'min_confidence': args.min_confidence,
                   'latency_median': statistics.median(latencies) if latencies else None,
                   'latency_p90': percentile(latencies, 0.9) if latencies else None, 'model': args.model}
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump({'summary': summary, 'results': results}, handle, ensure_ascii=False, indent=1)
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
