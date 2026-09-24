"""A small Luau tokenizer (phase-7 groundwork, tested).

Luau, not Lua: the keyword set includes Luau-only spellings (``continue``,
``type``, exported type syntax) and the operator set includes Luau's compound
assignments (``+= -= *= /= ..=``) and ``//`` floor division, which plain Lua 5.x
does not have. Roblox globals are kept in a separate list because the project's
target is Roblox Luau.

This tokenizer is deliberately simple and total: it never raises on unknown
input, it emits an ``error`` token type instead. That matters because generated
code under evaluation is frequently malformed, and a tokenizer that crashes on a
stray quote would make reward computation unreliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Luau keywords (``type`` and ``continue`` are Luau-only in this list; Roblox
#: additionally allows ``self``-style soft keywords which we treat as names).
LUAU_KEYWORDS = frozenset(
    """and break do else elseif end false for function if in local nil not or
    repeat return then true until while continue type export""".split()
)

#: Roblox/Luau compound and arithmetic operators, longest-first for greedy matching.
OPERATORS = (
    "..=", "//=", "+=", "-=", "*=", "/=", "%=", "^=",
    "...", "..", "==", "~=", "<=", ">=", "//", "::",
    "+", "-", "*", "/", "%", "^", "#", "<", ">", "=", "(", ")", "{", "}", "[", "]",
    ".", ",", ";", ":", "|", "&", "~",
)

#: Frequently used Roblox globals/API names. Kept separate from keywords so a
#: future phase can weigh them differently (they are the point of the project).
ROBLOX_GLOBALS = frozenset(
    """game workspace script task Instance Vector3 Vector2 CFrame Color3 UDim2 UDim
    Enum Ray Quaternion Numbers Humanoid Part Folder Model RemoteEvent RemoteFunction
    BindableEvent BindableFunction ModuleScript Players ReplicatedStorage ServerScriptService
    RunService TweenService HttpService Lighting Debris SoundListener Camera GuiService
    ContextActionService UserInputService PathfindingService PhysicsService CollectionService
    ProximityPromptService StarterGui StarterPlayer ServerStorage Attributes ChangeHistoryService""".split()
)

#: Fixed vocabulary head used by early experiments: structural tokens first.
VOCAB_PRELUDE = ("<pad>", "<eos>", "<start>", "<newline>", "<indent>", "<dedent>", "<error>")


@dataclass(frozen=True)
class Token:
    type: str  # keyword | name | number | string | operator | comment | error | newline
    text: str
    line: int
    col: int

    def key(self) -> str:
        """Canonical vocabulary key: strings/numbers are collapsed to their type."""
        if self.type in {"string", "number", "error"}:
            return f"<{self.type}>"
        return self.text


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("comment", re.compile(r"--\[\[[\s\S]*?\]\]|--[^\n]*")),
    ("string", re.compile(r'\[(=*)\[[\s\S]*?\]\1\]|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')),
    ("number", re.compile(r"0[xX][0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?(?:[pP][+-]?\d+)?|\d+\.?\d*(?:[eE][+-]?\d+)?")),
    ("name", re.compile(r"[A-Za-z_][A-Za-z0-9_]*")),
    ("newline", re.compile(r"\r?\n")),
]
_OP_ALT = "|".join(re.escape(op) for op in sorted(OPERATORS, key=len, reverse=True))
_OPERATORS = re.compile(_OP_ALT)
_WS = re.compile(r"[ \t]+")


def tokenize(source: str, *, keep_comments: bool = False) -> list[Token]:
    """Split Luau source into tokens. Total function: junk becomes ``<error>``."""
    out: list[Token] = []
    i, line, col = 0, 1, 1
    n = len(source)
    while i < n:
        ch = source[i]
        if ch == "\n":
            out.append(Token("newline", "\n", line, col))
            i, line, col = i + 1, line + 1, 1
            continue
        if ch in " \t\r":
            step = 1 if ch != "\t" else 4
            i, col = i + 1, col + step
            continue
        rest = source[i : i + 4096]
        matched = False
        for kind, pat in _PATTERNS:
            m = pat.match(rest)
            if not m:
                continue
            text = m.group(0)
            if kind == "comment" and not keep_comments:
                i += len(text)
                col += len(text)
                matched = True
                break
            if kind == "newline":
                i += len(text)
                out.append(Token("newline", text, line, col))
                line, col = line + 1, 1
                matched = True
                break
            type_ = kind
            if kind == "name" and text in LUAU_KEYWORDS:
                type_ = "keyword"
            out.append(Token(type_, text, line, col))
            i += len(text)
            col += len(text)
            matched = True
            break
        if matched:
            continue
        m = _OPERATORS.match(rest)
        if m:
            out.append(Token("operator", m.group(0), line, col))
            i += len(m.group(0))
            col += len(m.group(0))
            continue
        out.append(Token("error", ch, line, col))
        i, col = i + 1, col + 1
    return out


def token_keys(tokens: list[Token]) -> list[str]:
    return [t.key() for t in tokens]


def build_vocab(tokens: list[Token], *, max_size: int = 4000) -> dict[str, int]:
    """Frequency-ranked vocabulary with the structural prelude pinned at the front."""
    from collections import Counter

    counts = Counter(token_keys(tokens))
    ranked = [t for t, _ in counts.most_common()]
    vocab = {tok: idx for idx, tok in enumerate(VOCAB_PRELUDE)}
    for tok in ranked:
        if len(vocab) >= max_size:
            break
        vocab.setdefault(tok, len(vocab))
    return vocab


def encode(tokens: list[Token], vocab: dict[str, int], *, unk: int = 6) -> list[int]:
    unk_id = vocab.get("<error>", unk)
    return [vocab.get(k, unk_id) for k in token_keys(tokens)]


#: Type names that appear after ``:`` in an annotation (Luau's builtins plus the
#: Roblox classes people annotate with). ``:`` alone is *not* evidence of a type:
#: method calls (``game:GetService(...)``) use it constantly, and Roblox code is
#: mostly method calls, so a naive check would mark every script "typed".
TYPE_NAMES = frozenset(
    """number string boolean table thread buffer nil any unknown none
    Instance Part Folder Model Vector3 Vector2 CFrame Color3 UDim2 UDim Enum
    Ray Quaternion NumberRange Axes typeof""".split()
)


def _uses_type_syntax(tokens: list[Token]) -> bool:
    toks = [t for t in tokens if t.type != "newline"]
    for i, t in enumerate(toks):
        if t.text == "::":  # type assertion: `x :: number`
            return True
        if t.text == "type" and i + 2 < len(toks) and toks[i + 2].text == "=":  # `type Foo = ...`
            return True
        if t.text == ":" and i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt.type == "name" and (nxt.text in TYPE_NAMES or nxt.text == "typeof"):
                return True
    return False


def describe(source: str) -> dict:
    """Quick structural summary of a Luau snippet (used by task generators)."""
    toks = tokenize(source)
    return {
        "n_tokens": len(toks),
        "n_keywords": sum(1 for t in toks if t.type == "keyword"),
        "n_errors": sum(1 for t in toks if t.type == "error"),
        "uses_compound_assign": any(t.text in {"+=", "-=", "*=", "/=", "..="} for t in toks),
        "uses_typesyntax": _uses_type_syntax(toks),
        "roblox_globals": sorted({t.text for t in toks if t.text in ROBLOX_GLOBALS}),
    }
