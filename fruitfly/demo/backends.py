"""Where the Luau in the demo comes from -- and what is honestly claimed about it.

The point of this module is the boundary. There are two ways the demo can produce
code, they are wired to the same interface, and **every response carries a
provenance label that the user interface is required to display**:

``template``
    A hand-written snippet library with slot filling. It is a lookup, it does no
    learning, and it has nothing to do with the connectome. It exists so the demo
    shell can be built and shown before the science is finished.

``connectome``
    Text produced by decoding the state of the simulated FAFB network. **This does
    not exist yet.** Asking for it returns an unavailable response explaining what
    is missing, rather than silently falling back to the template backend -- a
    silent fallback is precisely how a demo starts lying.

The project's rule (README, LIMITATIONS.md): learning must change simulated neural
state. A snippet library is the canonical example of what does NOT count, so it is
labelled as a placeholder everywhere it appears.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

__all__ = ["Response", "TemplateBackend", "ConnectomeBackend", "get_backend", "BACKENDS"]

#: Shown in the UI banner and attached to every response. Deliberately blunt.
PLACEHOLDER_NOTICE = (
    "Placeholder backend: this Luau comes from a hand-written snippet library, "
    "NOT from the simulated fly brain. No neuron influenced this text."
)

CONNECTOME_NOTICE = (
    "Connectome backend: text decoded from simulated neural activity."
)


@dataclass
class Response:
    """One reply from a backend. ``provenance`` is not optional and not cosmetic."""

    code: str
    provenance: str            # "template" | "connectome" | "unavailable"
    notice: str                # human-readable honesty line, displayed in the UI
    is_placeholder: bool
    matched: str = ""          # which snippet/rule fired, for auditability
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- template
#: (name, keywords, code). Keywords are matched case-insensitively against the prompt.
SNIPPETS: list[tuple[str, tuple[str, ...], str]] = [
    ("part", ("part", "brick", "block", "cube"), '''local part = Instance.new("Part")
part.Size = Vector3.new(4, 1, 2)
part.Position = Vector3.new(0, 10, 0)
part.Anchored = true
part.BrickColor = BrickColor.new("Bright blue")
part.Parent = workspace'''),
    ("kill brick", ("kill", "damage", "lava", "hurt"), '''local killBrick = script.Parent

killBrick.Touched:Connect(function(hit)
    local humanoid = hit.Parent:FindFirstChildWhichIsA("Humanoid")
    if humanoid then
        humanoid.Health = 0
    end
end)'''),
    ("leaderstats", ("leaderstat", "coin", "points", "score", "cash", "money"), '''local Players = game:GetService("Players")

Players.PlayerAdded:Connect(function(player)
    local stats = Instance.new("Folder")
    stats.Name = "leaderstats"
    stats.Parent = player

    local coins = Instance.new("IntValue")
    coins.Name = "Coins"
    coins.Value = 0
    coins.Parent = stats
end)'''),
    ("teleport", ("teleport", "tp", "spawn point", "move player"), '''local function teleport(player, position)
    local character = player.Character
    if not character then
        return false
    end
    local root = character:FindFirstChild("HumanoidRootPart")
    if not root then
        return false
    end
    root.CFrame = CFrame.new(position)
    return true
end

return teleport'''),
    ("tween", ("tween", "animate", "smooth", "move part"), '''local TweenService = game:GetService("TweenService")

local part = workspace:WaitForChild("Part")
local info = TweenInfo.new(2, Enum.EasingStyle.Quad, Enum.EasingDirection.Out)
local goal = { Position = part.Position + Vector3.new(0, 10, 0) }

TweenService:Create(part, info, goal):Play()'''),
    ("remote event", ("remote", "server", "client", "fire"), '''local ReplicatedStorage = game:GetService("ReplicatedStorage")

local event = ReplicatedStorage:FindFirstChild("MyEvent")
if not event then
    event = Instance.new("RemoteEvent")
    event.Name = "MyEvent"
    event.Parent = ReplicatedStorage
end

event.OnServerEvent:Connect(function(player, message)
    print(player.Name, "sent", message)
end)'''),
    ("loop", ("loop", "repeat", "every second", "while"), '''local RunService = game:GetService("RunService")

local elapsed = 0
RunService.Heartbeat:Connect(function(delta)
    elapsed += delta
    if elapsed >= 1 then
        elapsed = 0
        print("one second passed")
    end
end)'''),
    ("gui", ("gui", "button", "ui", "screen", "text label"), '''local player = game:GetService("Players").LocalPlayer

local gui = Instance.new("ScreenGui")
gui.Parent = player:WaitForChild("PlayerGui")

local button = Instance.new("TextButton")
button.Size = UDim2.new(0, 200, 0, 50)
button.Position = UDim2.new(0.5, -100, 0.5, -25)
button.Text = "Click me"
button.Parent = gui

button.Activated:Connect(function()
    button.Text = "Clicked"
end)'''),
]

FALLBACK = '''-- The snippet library has no entry for that request.
-- This is a placeholder backend, so its coverage is exactly the list it ships with.
print("hello from a placeholder, not from a fly brain")'''


class TemplateBackend:
    """Keyword lookup over :data:`SNIPPETS`. Deterministic, no learning, no network."""

    name = "template"

    def available(self) -> tuple[bool, str]:
        return True, "snippet library loaded"

    def generate(self, prompt: str) -> Response:
        text = (prompt or "").lower()
        words = set(re.findall(r"[a-z]+", text))
        best_name, best_code = "", FALLBACK
        best_key: tuple[int, int] = (0, 0)
        for name, keywords, code in SNIPPETS:
            score = 0
            for rank, kw in enumerate(keywords):
                # the first keyword is the defining one. Without this weighting
                # "make a kill brick" scores equally for 'part' (via "brick") and for
                # 'kill brick', and the wrong snippet wins on list order.
                hit = 2 if kw in words else (1 if kw in text else 0)
                score += hit * (3 if rank == 0 else 1)
            # tiebreak: when two snippets score equally ("tween a part upward" matches
            # both 'tween' and 'part'), prefer the one the user mentioned first
            pos = text.find(keywords[0])
            rank_key = (score, -pos if pos >= 0 else -10 ** 6)
            if rank_key > best_key:
                best_name, best_code, best_key = name, code, rank_key
        return Response(
            code=best_code,
            provenance="template",
            notice=PLACEHOLDER_NOTICE,
            is_placeholder=True,
            matched=best_name or "none (fallback)",
            detail={"score": best_key[0], "library_size": len(SNIPPETS)},
        )


# -------------------------------------------------------------------- connectome
class ConnectomeBackend:
    """Decode Luau from simulated neural state. Not implemented -- and says so.

    It must never fall back to the template backend. A demo that quietly substitutes
    a lookup table when the science is not ready is the exact failure this project
    is set up to avoid.
    """

    name = "connectome"

    def available(self) -> tuple[bool, str]:
        return False, (
            "Not implemented. The simulated network has not been shown to learn any task "
            "above chance (EXPERIMENTS.md: 0/18 configurations, twice), and the first "
            "real-connectome measurement found no decodable advantage over a shuffled "
            "control. There is no trained decoder from neural state to Luau tokens, so "
            "there is nothing to call."
        )

    def generate(self, prompt: str) -> Response:
        ok, why = self.available()
        assert not ok
        return Response(
            code="",
            provenance="unavailable",
            notice=why,
            is_placeholder=False,
            matched="",
            detail={"requested_prompt_chars": len(prompt or "")},
        )


BACKENDS = {b.name: b for b in (TemplateBackend(), ConnectomeBackend())}


def get_backend(name: str):
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r} (have: {sorted(BACKENDS)})")
    return BACKENDS[name]
