function backendHttpBase() {
  const queryBase = new URLSearchParams(location.search).get("backend") || "";
  const savedBase = localStorage.getItem("cardsense_backend_base") || "";
  const configuredBase = (queryBase || savedBase).trim();
  if (configuredBase) {
    const normalized = configuredBase.replace(/\/$/, "");
    localStorage.setItem("cardsense_backend_base", normalized);
    return normalized;
  }
  const protocol = location.protocol || "http:";
  const host = location.hostname || "localhost";
  if (String(location.port || "") === "8080") {
    return `${protocol}//${host}:8000`;
  }
  return `${protocol}//${location.host}`;
}

function backendUrl(path) {
  if (!path) return backendHttpBase();
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${backendHttpBase()}${suffix}`;
}

const form = document.getElementById("signup-form");
const statusEl = document.getElementById("status");

function setStatus(text, ok) {
  statusEl.textContent = text;
  statusEl.className = `status ${ok ? "ok" : "err"}`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const displayName = document.getElementById("display-name").value.trim();
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;

  try {
    const response = await fetch(backendUrl("/api/auth/register"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email,
        password,
        display_name: displayName,
      }),
    });

    if (!response.ok) {
      throw new Error(await response.text());
    }

    localStorage.setItem("cardsense_email", email);
    setStatus("Signup successful. Go back and login with this account.", true);
    form.reset();
  } catch (error) {
    setStatus(`Signup failed: ${error}`, false);
  }
});
