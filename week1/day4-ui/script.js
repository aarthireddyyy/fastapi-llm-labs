const API_URL = "http://127.0.0.1:8000/chat";

const chatBox = document.getElementById("chat");
const messageInput = document.getElementById("message");
const sendBtn = document.getElementById("sendBtn");

function appendLine(text, className) {
  const div = document.createElement("div");
  if (className) div.classList.add(className);
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message) return;

  messageInput.value = "";
  appendLine("You: " + message, "user");

  sendBtn.disabled = true;
  appendLine("Assistant: …thinking…", "assistant");
  const loadingNode = chatBox.lastChild;

  try {
    const payload = {
      message: message,
      system_prompt: "You are a helpful assistant.",
      max_tokens: 256,
      temperature: 0.7,
      model: "qwen2.5:1.5b"
    };

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error("HTTP " + res.status);
    }

    const text = await res.text(); // backend returns plain text

    loadingNode.textContent = "Assistant: " + text;
  } catch (err) {
    loadingNode.textContent = "Assistant: [error]";
    appendLine("Error: " + err.message, "error");
  } finally {
    sendBtn.disabled = false;
    messageInput.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
