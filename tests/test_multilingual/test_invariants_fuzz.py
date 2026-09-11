# Copyright Sierra
"""Property-based fuzzing of the localization invariants.

Hypothesis properties against ``tau2.multilingual.invariants``:

1. ``extract_values`` finds every generated concrete value embedded in
   surrounding prose (emails, ``[a-z]+_[a-z]+_\\d+`` user ids, uppercase
   alphanumeric codes, numeric runs);
2. any value-preserving mangling of the prose passes the value-survival
   check, while deleting or altering any single value is caught;
3. substituting a Latin digit with a localized digit form is caught for
   every script in ``LOCALIZED_DIGIT_RES``;
4. mutating any ``evaluation_criteria`` leaf is caught by
   ``check_task_localization``.

The base tasks are the toy 'toyair' domain and its valid pre-localized
Toylang variant from ``factory_testing.toy_language``.
"""

import copy
import string

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from tau2.multilingual.invariants import (
    LOCALIZED_DIGIT_RES,
    check_task_localization,
    extract_values,
    instruction_text,
    script_regex,
)
from test_multilingual.factory_testing.toy_language import (
    toy_db,
    toy_tasks,
    toy_tasks_localized,
)

# Fast profile: bounded examples, no per-example deadline (CI machines vary).
FAST = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

LATIN_SCRIPT_RE = script_regex("latn")

# Latin digit -> localized digit form, one table per script that has a
# LOCALIZED_DIGIT_RES entry.
LOCALIZED_DIGIT_FORMS = {
    "deva": {str(d): chr(0x0966 + d) for d in range(10)},  # ०-९
    "arab": {str(d): chr(0x0660 + d) for d in range(10)},  # ٠-٩
    "hans": {str(d): chr(0xFF10 + d) for d in range(10)},  # fullwidth
    "hant": {str(d): chr(0xFF10 + d) for d in range(10)},
}


# ---------------------------------------------------------------------------
# Generated-value strategies (each matches exactly one VALUE_PATTERNS shape)
# ---------------------------------------------------------------------------

_lower_word = st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=8)

emails = st.builds(
    "{}{}@{}.{}".format,
    _lower_word,
    st.integers(0, 9999).map(str),
    _lower_word,
    st.sampled_from(["com", "org", "net", "co.in"]),
)

user_ids = st.builds(
    "{}_{}_{}".format, _lower_word, _lower_word, st.integers(0, 99999).map(str)
)

codes = st.builds(
    "{}{}{}".format,
    st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=3),
    st.text(alphabet=string.digits, min_size=2, max_size=4),
    st.text(alphabet=string.ascii_uppercase, min_size=0, max_size=3),
)

numbers = st.integers(0, 10**9).map(str)

values_lists = st.lists(
    st.one_of(emails, user_ids, codes, numbers),
    min_size=1,
    max_size=8,
    unique=True,
)


def _sanity_check_digit_tables():
    for script, table in LOCALIZED_DIGIT_FORMS.items():
        digits_re = LOCALIZED_DIGIT_RES[script]
        for localized in table.values():
            assert digits_re.search(localized), (script, localized)


_sanity_check_digit_tables()


# ---------------------------------------------------------------------------
# 1. Extraction completeness
# ---------------------------------------------------------------------------


@FAST
@given(values=values_lists)
def test_extract_values_finds_all_embedded_values(values):
    text = (
        "The caller says " + " and then mentions ".join(values) + " before hanging up."
    )
    extracted = extract_values(text)
    for value in values:
        assert value in extracted, f"'{value}' not extracted from: {text!r}"


# ---------------------------------------------------------------------------
# 2. Value survival: preserving passes, deleting/altering is caught
# ---------------------------------------------------------------------------


def _toy_pairs():
    """(localized, source) pairs from the valid toy fixtures."""
    sources = {t["id"]: t for t in toy_tasks()}
    return [
        (localized, sources[localized["id"].removesuffix("_tl")])
        for localized in toy_tasks_localized()
    ]


TOY_PAIRS = _toy_pairs()
TOY_DB = toy_db()


def _with_instruction_text(localized, text):
    """A copy of `localized` whose task_instructions carry `text` verbatim."""
    out = copy.deepcopy(localized)
    out["user_scenario"]["instructions"]["task_instructions"] = text
    return out


@FAST
@given(
    pair_index=st.integers(0, len(TOY_PAIRS) - 1),
    fillers=st.lists(
        st.sampled_from(["zorg", "blat", "vu", "wil", "een", "naar", "danke"]),
        min_size=1,
        max_size=6,
    ),
    data=st.data(),
)
def test_value_preserving_mangle_passes_value_survival(pair_index, fillers, data):
    localized, source = TOY_PAIRS[pair_index]
    values = sorted(extract_values(instruction_text(source)))
    shuffled = data.draw(st.permutations(values))
    # A mangled "translation": arbitrary prose, but every value kept verbatim.
    mangled = " ".join(fillers) + " " + " , ".join(shuffled) + " ."
    candidate = _with_instruction_text(localized, mangled)
    problems = check_task_localization(
        candidate, source, LATIN_SCRIPT_RE, domain_db=TOY_DB
    )
    survival_problems = [p for p in problems if "concrete value" in p]
    assert not survival_problems, "\n".join(survival_problems)


@FAST
@given(
    pair_index=st.integers(0, len(TOY_PAIRS) - 1),
    alter=st.booleans(),
    data=st.data(),
)
def test_deleting_or_altering_any_single_value_is_caught(pair_index, alter, data):
    localized, source = TOY_PAIRS[pair_index]
    values = sorted(extract_values(instruction_text(source)))
    assume(values)
    victim = data.draw(st.sampled_from(values))

    survivors = [v for v in values if v != victim]
    if alter:
        # Bump every digit (mod 10) so the value is present-but-wrong.
        altered = victim.translate(str.maketrans(string.digits, "1234567890"))
        assume(altered != victim)
        survivors.append(altered)
    mangled = "zorg blat " + " , ".join(survivors) + " ."
    # The victim must not survive accidentally as a substring of another
    # value (e.g. '25' inside '250').
    assume(victim not in mangled)

    candidate = _with_instruction_text(localized, mangled)
    # Wipe the other instruction prose so the victim cannot hide there;
    # keep some Latin text so script checks still pass.
    instructions = candidate["user_scenario"]["instructions"]
    for field in ("reason_for_call", "known_info", "unknown_info"):
        if instructions.get(field):
            instructions[field] = "zorg blat"

    problems = check_task_localization(
        candidate, source, LATIN_SCRIPT_RE, domain_db=TOY_DB
    )
    assert any("concrete value" in p and f"'{victim}'" in p for p in problems), (
        f"deletion of '{victim}' not caught; problems: {problems}"
    )


# ---------------------------------------------------------------------------
# 3. Localized digit forms are caught per script
# ---------------------------------------------------------------------------


@FAST
@given(
    script=st.sampled_from(sorted(LOCALIZED_DIGIT_RES)),
    digit=st.integers(0, 9),
    pair_index=st.integers(0, len(TOY_PAIRS) - 1),
)
def test_localized_digit_substitution_is_caught(script, digit, pair_index):
    localized, source = TOY_PAIRS[pair_index]
    localized_digit = LOCALIZED_DIGIT_FORMS[script][str(digit)]
    candidate = copy.deepcopy(localized)
    instructions = candidate["user_scenario"]["instructions"]
    instructions["task_instructions"] += f" Wacht {localized_digit} minuten."

    problems = check_task_localization(
        candidate,
        source,
        script_regex(script),
        domain_db=TOY_DB,
        localized_digits_re=LOCALIZED_DIGIT_RES[script],
    )
    assert any("localized digit forms" in p for p in problems), problems


def test_latin_digits_pass_localized_digit_check():
    """Control: the valid Latin-digit fixtures trip no digit problems."""
    for script in sorted(LOCALIZED_DIGIT_RES):
        for localized, source in TOY_PAIRS:
            problems = check_task_localization(
                localized,
                source,
                LATIN_SCRIPT_RE,
                domain_db=TOY_DB,
                localized_digits_re=LOCALIZED_DIGIT_RES[script],
            )
            assert not any("localized digit forms" in p for p in problems)


# ---------------------------------------------------------------------------
# 4. Any evaluation_criteria leaf mutation is caught
# ---------------------------------------------------------------------------


def _leaf_paths(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_paths(value, prefix + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _leaf_paths(value, prefix + (index,))
    else:
        yield prefix


def _mutate_leaf(criteria, path):
    node = criteria
    for step in path[:-1]:
        node = node[step]
    leaf = node[path[-1]]
    if isinstance(leaf, bool):
        node[path[-1]] = not leaf
    elif isinstance(leaf, (int, float)):
        node[path[-1]] = leaf + 1
    elif isinstance(leaf, str):
        node[path[-1]] = leaf + " MUTATED"
    else:  # None
        node[path[-1]] = "MUTATED"


@FAST
@given(pair_index=st.integers(0, len(TOY_PAIRS) - 1), data=st.data())
def test_any_eval_criteria_leaf_mutation_is_caught(pair_index, data):
    localized, source = TOY_PAIRS[pair_index]
    paths = list(_leaf_paths(localized["evaluation_criteria"]))
    assume(paths)
    path = data.draw(st.sampled_from(paths))

    candidate = copy.deepcopy(localized)
    _mutate_leaf(candidate["evaluation_criteria"], path)
    assert candidate["evaluation_criteria"] != localized["evaluation_criteria"]

    problems = check_task_localization(
        candidate, source, LATIN_SCRIPT_RE, domain_db=TOY_DB
    )
    assert any("evaluation_criteria differs" in p for p in problems), (
        f"mutation at {path} not caught; problems: {problems}"
    )


def test_unmutated_fixture_pairs_pass_everything():
    """Control for all properties above: the valid pairs are clean."""
    for localized, source in TOY_PAIRS:
        problems = check_task_localization(
            localized, source, LATIN_SCRIPT_RE, domain_db=TOY_DB
        )
        assert not problems, "\n".join(problems)
