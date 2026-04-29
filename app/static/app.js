const chatForm = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const askButton = document.querySelector("#ask-button");
const ingestButton = document.querySelector("#ingest-button");
const conversationLog = document.querySelector("#conversation-log");
const sourcesList = document.querySelector("#sources-list");
const operationStatus = document.querySelector("#operation-status");
const filesCount = document.querySelector("#files-count");
const chunksCount = document.querySelector("#chunks-count");

const statusElements = {
  api: document.querySelector("#api-status"),
  ollama: document.querySelector("#ollama-status"),
  chromadb: document.querySelector("#chroma-status"),
};

function setBusy(isBusy) {
  askButton.disabled = isBusy;
  questionInput.disabled = isBusy;
}

function addMessage(role, text, isError = false) {
  const message = document.createElement("article");
  message.className = `message ${role}${isError ? " error" : ""}`;

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Agent";

  const body = document.createElement("p");
  body.textContent = text;

  message.append(label, body);
  conversationLog.append(message);
  conversationLog.scrollTop = conversationLog.scrollHeight;
}

function renderSources(sources) {
  sourcesList.replaceChildren();
  if (!sources.length) {
    const item = document.createElement("li");
    item.textContent = "No sources";
    sourcesList.append(item);
    return;
  }

  for (const source of sources) {
    const item = document.createElement("li");
    item.textContent = source;
    sourcesList.append(item);
  }
}

function updateStatusPill(name, value) {
  const element = statusElements[name];
  if (!element) {
    return;
  }

  element.classList.remove("ok", "degraded", "unavailable");
  element.classList.add(value);
  element.textContent = `${element.dataset.label || element.textContent.split(" ")[0]} ${value}`;
}

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    for (const [name, value] of Object.entries(data.services)) {
      updateStatusPill(name, value);
    }
  } catch {
    for (const name of Object.keys(statusElements)) {
      updateStatusPill(name, "unavailable");
    }
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  addMessage("user", question);
  questionInput.value = "";
  setBusy(true);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Chat request failed.");
    }

    addMessage("assistant", data.answer);
    renderSources(data.sources || []);
  } catch (error) {
    addMessage("assistant", error.message, true);
  } finally {
    setBusy(false);
    questionInput.focus();
    refreshHealth();
  }
});

ingestButton.addEventListener("click", async () => {
  ingestButton.disabled = true;
  operationStatus.textContent = "Ingesting";

  try {
    const response = await fetch("/ingest", { method: "POST" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Ingestion failed.");
    }

    filesCount.textContent = String(data.ingested_files.length);
    chunksCount.textContent = String(data.chunks_indexed);
    operationStatus.textContent = data.collection;
  } catch (error) {
    operationStatus.textContent = error.message;
  } finally {
    ingestButton.disabled = false;
    refreshHealth();
  }
});

for (const [name, element] of Object.entries(statusElements)) {
  element.dataset.label = element.textContent;
  updateStatusPill(name, "degraded");
}

refreshHealth();
setInterval(refreshHealth, 30000);
