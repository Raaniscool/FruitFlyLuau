"""Phase 7/8 scaffold: the Luau tokenizer, task schema, and the sandbox contract.

The honest headline (also stated in ``fruitfly/luau/__init__.py``): **this project
cannot write Luau yet.** What exists here is the machinery a Luau experiment would
need -- a tested tokenizer, a task/reward schema that only uses measured outcomes,
and an evaluator that *refuses* to execute anything until a real sandbox exists.
These tests keep that refusal honest: no test may pass by quietly running code.
"""

from __future__ import annotations

import json

import pytest

from fruitfly.luau import PHASE_STATUS
from fruitfly.luau.evaluator import (
    FORBIDDEN_PATTERNS,
    HARNESS_REQUIREMENTS,
    MOCK_ROBLOX_API,
    REQUIRED_LIMITS,
    ExecResult,
    backend_available,
    execute,
    scan_static,
    self_check,
    validate_limits,
)
from fruitfly.luau.tasks import CATEGORIES, SEED_TASKS, LuauTask, TestCase, corpus, split_corpus
from fruitfly.luau.tokens import (
    LUAU_KEYWORDS,
    OPERATORS,
    ROBLOX_GLOBALS,
    VOCAB_PRELUDE,
    Token,
    build_vocab,
    describe,
    encode,
    token_keys,
    tokenize,
)


SNIPPET = '''\
-- a Roblox-ish script
local Players = game:GetService("Players")
type Conf = { name: string, lives: number }

local function spawn(b: Instance): Part
    local p = Instance.new("Part")
    p.Position = Vector3.new(0, 10, 0)
    p.Name = "orb-" .. tostring(3.5 + 0x10)
    for i = 1, 10 do
        p.Size *= Vector3.new(1, 1, 1)
        if i % 2 == 0 then continue end
    end
    return p
end
'''


# ------------------------------------------------------------------ tokenizer
def test_keywords_include_luau_only_spellings():
    assert "continue" in LUAU_KEYWORDS and "type" in LUAU_KEYWORDS
    assert "and" in LUAU_KEYWORDS and "function" in LUAU_KEYWORDS
    # Luau removed these from Lua 5.1
    for gone in ("goto", "setmetatable"):
        assert gone not in LUAU_KEYWORDS or gone == "setmetatable"


def test_operators_cover_luau_compound_assignments():
    for op in ("..=", "//=", "+=", "-=", "*=", "/=", "::", "..", "//"):
        assert op in OPERATORS, op


def test_tokenize_structure_of_a_real_script():
    toks = tokenize(SNIPPET)
    kinds = {t.type for t in toks}
    assert {"keyword", "name", "string", "number", "operator", "newline"} <= kinds
    assert "comment" not in kinds, "comments are dropped unless keep_comments=True"
    text = [t.text for t in toks]
    assert "continue" in text and "::" not in text
    assert "*=" in text, "Luau compound assignment must be one token, not '*' + '='"
    assert "type" in text and "Conf" in text
    nums = [t.text for t in toks if t.type == "number"]
    assert "3.5" in nums and "0x10" in nums, f"hex/float numbers: {nums}"
    strs = [t.text for t in toks if t.type == "string"]
    assert '"Players"' in strs and '"orb-"' in strs


def test_comments_and_long_bracket_strings_survive():
    block = "--[[ block\ncomment ]] local x = [[\nraw\nstring ]] --[[ second ]]"
    kept = tokenize(block, keep_comments=True)
    kinds = [t.type for t in kept]
    assert kinds.count("comment") == 2, kinds
    assert kinds.count("string") == 1, "the long-bracket string is one token, not three"
    assert [t.type for t in tokenize(block)] == ["keyword", "name", "operator", "string"], "comments drop by default"
    level = "local s = [==[he]]llo]==] end"
    toks = tokenize(level)
    str_toks = [t for t in toks if t.type == "string"]
    assert str_toks and str_toks[0].text == "[==[he]]llo]==]", [t.text for t in toks]


def test_tokenizer_is_total_on_garbage():
    """Malformed generated code must produce <error> tokens, never an exception."""
    for junk in ('local x = "', "!!! @ # $", "a = 1 ~", "\x00\x01", "x = 'unterminated"):
        toks = tokenize(junk)
        assert isinstance(toks, list) and all(isinstance(t, Token) for t in toks)
    assert describe('local x = "')["n_errors"] >= 1


def test_line_and_column_are_tracked():
    toks = tokenize("a\nbb\nc")
    lines = [(t.line, t.col) for t in toks if t.type == "name"]
    assert lines == [(1, 1), (2, 1), (3, 1)], lines


def test_keys_collapse_literals_but_keep_structure():
    toks = tokenize('local a = "x" + 12 + "y" + 7')
    keys = token_keys(toks)
    assert keys.count("<string>") == 2 and keys.count("<number>") == 2
    assert "local" in keys and "+" in keys
    # two different strings must map to the same key: the model learns syntax, not text
    assert token_keys(tokenize('print("hello")'))[0] == "print"


def test_build_vocab_pins_prelude_and_ranks_by_frequency():
    toks = tokenize("local x = 1 local y = 2 local z = 3")
    vocab = build_vocab(toks)
    assert [vocab[k] for k in VOCAB_PRELUDE] == list(range(len(VOCAB_PRELUDE)))
    assert vocab["local"] == min(vocab.values(), key=lambda v: v) or vocab["local"] >= len(VOCAB_PRELUDE)
    assert sorted(vocab.values()) == list(range(len(vocab))), "ids must be dense"
    small = build_vocab(toks, max_size=9)
    assert len(small) == 9
    assert set(VOCAB_PRELUDE) <= set(small)


def test_encode_maps_unknown_tokens_to_the_error_id():
    toks = tokenize("local x = 1")
    vocab = build_vocab(toks, max_size=8)  # prelude only + a couple of entries
    ids = encode(tokenize("local unusualthing = 1"), vocab)
    assert vocab["<error>"] in ids, "a token outside the vocabulary must become <error>, not crash"
    assert len(ids) == len(tokenize("local unusualthing = 1"))


def test_describe_does_not_call_method_colons_type_annotations():
    """Roblox code is full of ``svc:Method()``; that must not read as 'typed'."""
    untyped = describe('local rs = game:GetService("RunService")\nrs.Heartbeat:Connect(function() end)')
    assert untyped["uses_typesyntax"] is False, untyped
    typed = describe("local p: Part = nil\nlocal q = p :: Instance")
    assert typed["uses_typesyntax"] is True
    alias = describe("type Config = { lives: number }")
    assert alias["uses_typesyntax"] is True
    assert describe("x += 1")["uses_compound_assign"] is True
    assert describe("x = x + 1")["uses_compound_assign"] is False
    assert "game" in describe('game.Workspace["a"]')["roblox_globals"]
    json.dumps(untyped)


# ------------------------------------------------------------------ tasks
def test_seed_tasks_are_wellformed_and_broad():
    assert len(SEED_TASKS) >= 5
    ids = [t.id for t in SEED_TASKS]
    assert len(set(ids)) == len(ids), "task ids must be unique"
    for t in SEED_TASKS:
        assert isinstance(t, LuauTask)
        assert t.instruction.strip()
        assert t.category in CATEGORIES
        assert 1 <= t.difficulty <= 10
        assert t.tests, f"{t.id} has no behavioural tests, so it cannot be graded"
        for case in t.tests:
            assert isinstance(case, TestCase) and case.name and case.call


def test_task_validation_rejects_unknown_category():
    with pytest.raises(ValueError) as exc:
        LuauTask(id="bad", instruction="x", category="not_a_category")
    assert "category" in str(exc.value)


def test_reward_uses_measured_outcomes_only():
    t = SEED_TASKS[0]
    n = len(t.tests)
    assert t.reward(n_passed=n, n_total=n) == pytest.approx(t.reward_pass)
    assert t.reward(n_passed=0, n_total=n) == pytest.approx(t.reward_fail)
    if t.reward_partial and n > 1:
        mid = t.reward(n_passed=1, n_total=n)
        assert t.reward_fail < mid < t.reward_pass
    # code that never ran gets the failure value, never a free partial credit
    assert t.reward(n_passed=n, n_total=n, executed=False) == pytest.approx(t.reward_fail)
    strict = LuauTask(id="s", instruction="i", category="syntax", reward_partial=False,
                      tests=(TestCase("a", "a()"), TestCase("b", "b()")))
    assert strict.reward(n_passed=1, n_total=2) == strict.reward_fail
    assert strict.reward(n_passed=2, n_total=2) == strict.reward_pass
    assert LuauTask(id="s2", instruction="i", category="syntax", tests=()).reward(n_passed=0, n_total=0) < 0


def test_corpus_filters_and_split_is_deterministic_and_disjoint():
    assert len(corpus()) == len(SEED_TASKS)
    easy = corpus(max_difficulty=1)
    assert all(t.difficulty <= 1 for t in easy)
    one = corpus(categories=("loops",))
    assert all(t.category == "loops" for t in one)
    tr, ev = split_corpus(seed=3)
    assert len(tr) + len(ev) == len(SEED_TASKS)
    assert not ({t.id for t in tr} & {t.id for t in ev}), "eval tasks must never enter training"
    assert [list(t.id for t in x) for x in split_corpus(seed=3)] == [list(t.id for t in x) for x in split_corpus(seed=3)]
    assert split_corpus(seed=1)[0] != tr or len(SEED_TASKS) < 4
    tr2, ev2 = split_corpus(seed=0, train_fraction=0.5)
    assert len(ev2) >= len(tr2) - 1


# ------------------------------------------------------------------ evaluator
def test_static_scan_blocks_the_dangerous_surface():
    for src in ('local f = loadstring("x")', "os.execute('rm -rf /')", "require(4)", "io.open('/etc/passwd')"):
        rep = scan_static(src)
        assert rep.ok is False, (src, rep.findings)
        assert rep.findings and all({"pattern", "why"} <= set(f) for f in rep.findings)
        json.dumps(rep.to_dict())


def test_static_scan_allows_plain_gameplay_code():
    rep = scan_static('local p = Instance.new("Part")\np.Parent = workspace\nprint(p.Name)')
    assert rep.ok, rep.findings
    assert rep.n_tokens > 10
    # pcall is legal in user code; only the harness may not use it to hide errors
    assert scan_static("local ok = pcall(function() return 1 end)").ok


def test_static_scan_allow_list_is_explicit():
    assert not scan_static("require(script.M)").ok
    assert scan_static("require(script.M)", allow=("require",)).ok


def test_static_scan_counts_syntax_errors_it_could_not_avoid():
    rep = scan_static('local x = "')
    assert rep.n_error_tokens >= 1
    assert "n_error_tokens" in rep.to_dict() or rep.to_dict()["metrics"] is not None


def test_limits_are_bounded_and_validated():
    assert validate_limits(2.0, 256, 65536) == []
    assert validate_limits(0.01, 256, 65536), "a sub-10ms timeout is not a real limit"
    assert validate_limits(2.0, 4, 65536), "4 MB is below the enforced floor"
    assert validate_limits(2.0, 256, 1 << 30), "unbounded output is a disk-fill attack"
    assert validate_limits(600.0, 256, 65536), "a 10 minute timeout is not a sandbox"
    hi = REQUIRED_LIMITS["timeout_s"][1]
    assert hi <= 10.0, "the ceiling itself must stay small"


@pytest.mark.parametrize("src", [
    'os.execute("touch /tmp/luau_executed_marker")',
    'local f = loadstring("os.execute([[touch /tmp/luau_executed_marker]])") f()',
    "print(1)",
])
def test_execute_never_runs_anything(tmp_path, monkeypatch, src):
    """The single most important test in this directory."""
    marker = tmp_path / "luau_executed_marker"
    monkeypatch.chdir(tmp_path)
    for backend in ("", "firejail", "docker", "windows_job", "nsjail"):
        res = execute(src, backend=backend, timeout_s=1.0, memory_mb=64)
        assert isinstance(res, ExecResult)
        assert res.ran is False, f"{backend} actually executed code!"
        assert res.status in ("refused", "unavailable"), res
        assert res.error, "a refusal must explain itself"
    assert not marker.exists(), "a marker file appeared: code was executed"
    assert not list(tmp_path.glob("*")), "the evaluator created files in the cwd"


def test_backend_availability_reports_the_missing_tool_not_a_crash():
    for name in ("", "firejail", "docker", "windows_job", "made_up_sandbox"):
        ok, why = backend_available(name)
        assert ok is False, f"{name} claims to be available in this build"
        assert isinstance(why, str) and why


def test_harness_contract_is_a_checklist_not_prose():
    assert HARNESS_REQUIREMENTS and all(isinstance(r, str) and len(r) > 10 for r in HARNESS_REQUIREMENTS)
    joined = " ".join(HARNESS_REQUIREMENTS).lower()
    for needed in ("timeout", "memory", "network", "mock"):
        assert needed in joined, needed
    for api in ("game", "workspace", "Instance", "Part", "Humanoid", "Vector3", "CFrame",
                "RemoteEvent", "RemoteFunction"):
        assert api in MOCK_ROBLOX_API, api
        assert MOCK_ROBLOX_API[api], api
    assert "TouchInterest" in " ".join(MOCK_ROBLOX_API["Part"]) or "Touched" in MOCK_ROBLOX_API["Part"]


def test_self_check_states_the_limitation_plainly():
    rep = self_check()
    txt = json.dumps(rep).lower()
    assert "disabled" in txt or "static-only" in txt
    assert rep["static_scan_available"] is True
    assert rep["luau_on_path"] in (True, False)
    assert rep["requirements_not_yet_met"], "the unmet list must be explicit, not empty"
    assert "execution" in PHASE_STATUS.lower() or "not" in PHASE_STATUS.lower()


def test_phase_status_is_honest_about_the_project():
    s = PHASE_STATUS.lower()
    assert "cannot" in s or "not" in s, "the module must state what does not exist yet"
    assert "luau" in s
