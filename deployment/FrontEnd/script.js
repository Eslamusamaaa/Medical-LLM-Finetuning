// FrontEnd/script.js
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('user-input');
const sendBtn    = document.getElementById('send-btn');

function toggleTheme() {
  const body = document.body;
  const icon = document.querySelector('.theme-icon');
  if (body.classList.contains('dark')) {
    body.classList.replace('dark', 'light');
    icon.textContent = '☀️';
  } else {
    body.classList.replace('light', 'dark');
    icon.textContent = '🌙';
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 130) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function sendSuggestion(btn) {
  inputEl.value = btn.innerText;
  sendMessage();
}

function clearChat() {
  messagesEl.innerHTML = `
    <div class="welcome-msg">
      <div class="welcome-icon">🩺</div>
      <h2>Welcome to Shifa</h2>
      <p>Ask me any medical question in Arabic and I'll answer with accuracy and professionalism.<br>
      Remember: This assistant is for information only — always consult a doctor.</p>
    </div>`;
}

function appendMessage(role, content, meta = '') {
  const wc = messagesEl.querySelector('.welcome-msg');
  if (wc) wc.remove();

  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '👤' : '☤';

  const bubbleWrap = document.createElement('div');
  bubbleWrap.className = 'bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;
  bubbleWrap.appendChild(bubble);

  if (meta) {
    const metaEl = document.createElement('div');
    metaEl.className = 'meta';
    metaEl.textContent = meta;
    bubbleWrap.appendChild(metaEl);
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubbleWrap);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showTyping() {
  const wrapper = document.createElement('div');
  wrapper.className = 'message bot typing';
  wrapper.id = 'typing';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = '☤';

  const bubbleWrap = document.createElement('div');
  bubbleWrap.className = 'bubble-wrap';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement('div');
    dot.className = 'dot';
    bubble.appendChild(dot);
  }
  bubbleWrap.appendChild(bubble);
  wrapper.appendChild(avatar);
  wrapper.appendChild(bubbleWrap);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideTyping() {
  const t = document.getElementById('typing');
  if (t) t.remove();
}

async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message || sendBtn.disabled) return;

  sendBtn.disabled   = true;
  inputEl.disabled   = true;
  inputEl.value      = '';
  inputEl.style.height = 'auto';

  appendMessage('user', message);
  showTyping();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, max_tokens: 512, temperature: 0.7 }),
    });

    hideTyping();

    if (!res.ok) {
      const err = await res.json();
      appendMessage('bot', `Error: ${err.detail || 'An unexpected error occurred'}`);
    } else {
      const data = await res.json();
      const device = data.device === 'cuda' ? 'GPU' : 'CPU';
      appendMessage('bot', data.response, `⏱ ${data.elapsed_seconds}s · ${device}`);
    }
  } catch (err) {
    hideTyping();
    appendMessage('bot', 'Could not connect to server. Make sure the backend is running.');
  } finally {
    sendBtn.disabled   = false;
    inputEl.disabled   = false;
    inputEl.focus();
  }
}