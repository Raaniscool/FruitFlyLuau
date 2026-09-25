// Browser-side only. Every request is a relative URL: the page must work behind a
// proxy, so it never names a host.
const $ = (id) => document.getElementById(id);
const out = $("out"), banner = $("banner"), bannerText = $("banner-text");
const provenance = $("provenance"), flywrap = $("flywrap");

let typing = null;

/** Type `text` into the code pane, animating the fly while it runs. */
function typeOut(text) {
  if (typing) { clearInterval(typing); typing = null; }
  out.classList.remove("error");
  out.textContent = "";
  flywrap.classList.add("typing");

  const cursor = document.createElement("span");
  cursor.className = "cursor";
  cursor.textContent = "\u00a0";
  out.appendChild(cursor);

  // ~1.6 chars/frame at 16ms: fast enough not to bore, slow enough to read as typing
  let i = 0;
  const chunk = Math.max(1, Math.round(text.length / 320));
  typing = setInterval(() => {
    i = Math.min(text.length, i + chunk);
    cursor.insertAdjacentText("beforebegin", text.slice(i - chunk, i));
    if (i >= text.length) {
      clearInterval(typing); typing = null;
      flywrap.classList.remove("typing");
      cursor.remove();
    }
  }, 16);
}

function showUnavailable(message) {
  if (typing) { clearInterval(typing); typing = null; }
  flywrap.classList.remove("typing");
  out.classList.add("error");
  out.textContent = message;
}

async function ask(prompt) {
  const backend = $("backend").value;
  $("go").disabled = true;
  provenance.textContent = "";
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, backend }),
    });
    const data = await res.json();

    const placeholder = data.provenance === "template";
    banner.classList.toggle("unavailable", data.provenance === "unavailable");
    banner.querySelector("strong").textContent =
      data.provenance === "unavailable" ? "Backend unavailable." : "Placeholder backend.";
    bannerText.textContent = data.notice || "";

    if (data.provenance === "unavailable" || !data.code) {
      showUnavailable(data.notice || "No output.");
    } else {
      typeOut(data.code);
      provenance.textContent =
        `source: ${data.provenance} backend · matched snippet: ${data.matched}` +
        (placeholder ? " · not produced by the simulated connectome" : "");
    }
  } catch (err) {
    showUnavailable("Request failed: " + err);
  } finally {
    $("go").disabled = false;
  }
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const v = $("prompt").value.trim();
  if (v) ask(v);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("prompt").value = chip.textContent;
    ask(chip.textContent);
  });
});

fetch("/api/status").then((r) => r.json()).then((s) => {
  const m = s.measured;
  $("status").innerHTML = `
    <b>Project status</b><br>
    ${s.headline}<br><br>
    <b>Real connectome loaded</b><br>${m.connectome}<br><br>
    <b>Subgraph under test</b><br>${m.subgraph}<br><br>
    <b>Frozen-reservoir measurement</b><br>
    real wiring: <code>${m.reservoir_real}</code><br>
    shuffled: <code>${m.reservoir_shuffled}</code><br>
    ${m.verdict}.`;
}).catch(() => { $("status").textContent = "status unavailable"; });
