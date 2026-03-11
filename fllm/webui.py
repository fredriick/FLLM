"""
webui.py — Browser-based ChatGPT-like chat interface for FLLM.

Serves a single-page HTML app on GET / that connects to the
existing OpenAI-compatible /v1/chat/completions endpoint.
"""

from __future__ import annotations


def get_html(model_name: str) -> str:
    """Return a complete HTML document with embedded CSS + JS."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FLLM — {model_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #1a1a2e;
  color: #e0e0e0;
  height: 100vh;
  display: flex;
  flex-direction: column;
}}
#header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  background: #16213e;
  border-bottom: 1px solid #0f3460;
}}
#header h1 {{
  font-size: 16px;
  font-weight: 600;
  color: #e94560;
}}
#header .model-name {{
  font-size: 13px;
  color: #8899aa;
}}
#new-chat {{
  background: #0f3460;
  color: #e0e0e0;
  border: 1px solid #e94560;
  border-radius: 6px;
  padding: 6px 14px;
  cursor: pointer;
  font-size: 13px;
}}
#new-chat:hover {{ background: #e94560; color: #fff; }}
#messages {{
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}}
.msg {{
  max-width: 780px;
  width: 100%;
  margin: 0 auto;
  display: flex;
  gap: 12px;
}}
.msg .avatar {{
  width: 32px;
  height: 32px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  flex-shrink: 0;
}}
.msg.user .avatar {{ background: #0f3460; }}
.msg.assistant .avatar {{ background: #e94560; }}
.msg .bubble {{
  background: #16213e;
  border-radius: 12px;
  padding: 12px 16px;
  line-height: 1.6;
  flex: 1;
  min-width: 0;
  overflow-wrap: break-word;
}}
.msg.user .bubble {{ background: #0f3460; }}
.msg .bubble p {{ margin-bottom: 8px; }}
.msg .bubble p:last-child {{ margin-bottom: 0; }}
.msg .bubble pre {{
  background: #0d1117;
  border-radius: 8px;
  padding: 12px;
  overflow-x: auto;
  margin: 8px 0;
  font-size: 13px;
  line-height: 1.5;
}}
.msg .bubble code {{
  font-family: 'SF Mono', Monaco, Consolas, monospace;
  font-size: 13px;
}}
.msg .bubble :not(pre) > code {{
  background: #0d1117;
  padding: 2px 6px;
  border-radius: 4px;
}}
.msg .bubble strong {{ color: #fff; }}
#input-area {{
  padding: 16px 20px;
  background: #16213e;
  border-top: 1px solid #0f3460;
}}
#input-row {{
  max-width: 780px;
  margin: 0 auto;
  display: flex;
  gap: 10px;
  align-items: flex-end;
}}
#user-input {{
  flex: 1;
  background: #1a1a2e;
  border: 1px solid #0f3460;
  border-radius: 10px;
  padding: 10px 14px;
  color: #e0e0e0;
  font-size: 15px;
  font-family: inherit;
  resize: none;
  max-height: 200px;
  line-height: 1.5;
  outline: none;
}}
#user-input:focus {{ border-color: #e94560; }}
#send-btn, #stop-btn {{
  padding: 10px 18px;
  border: none;
  border-radius: 10px;
  font-size: 14px;
  cursor: pointer;
  font-weight: 600;
}}
#send-btn {{
  background: #e94560;
  color: #fff;
}}
#send-btn:hover {{ background: #c73e54; }}
#send-btn:disabled {{ background: #444; cursor: not-allowed; }}
#stop-btn {{
  background: #ff6b6b;
  color: #fff;
  display: none;
}}
#stop-btn:hover {{ background: #ee5a5a; }}
.typing-indicator {{
  display: inline-block;
  width: 8px; height: 8px;
  background: #e94560;
  border-radius: 50%;
  animation: pulse 1s infinite;
  margin-left: 4px;
}}
@keyframes pulse {{
  0%, 100% {{ opacity: 0.3; }}
  50% {{ opacity: 1; }}
}}
</style>
</head>
<body>
<div id="header">
  <div>
    <h1>FLLM</h1>
    <span class="model-name">{model_name}</span>
  </div>
  <button id="new-chat" onclick="newChat()">New Chat</button>
</div>
<div id="messages"></div>
<div id="input-area">
  <div id="input-row">
    <textarea id="user-input" rows="1" placeholder="Type a message..."
      onkeydown="handleKey(event)"></textarea>
    <button id="send-btn" onclick="sendMessage()">Send</button>
    <button id="stop-btn" onclick="stopGeneration()">Stop</button>
  </div>
</div>

<script>
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');

let history = [];
let abortController = null;
let generating = false;

function handleKey(e) {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendMessage();
  }}
}}

// Auto-resize textarea
inputEl.addEventListener('input', () => {{
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
}});

function scrollToBottom() {{
  messagesEl.scrollTop = messagesEl.scrollHeight;
}}

function renderMarkdown(text) {{
  // Code blocks
  text = text.replace(/```(\\w*)\\n([\\s\\S]*?)```/g, (_, lang, code) => {{
    return '<pre><code>' + escapeHtml(code.trim()) + '</code></pre>';
  }});
  // Inline code
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  text = text.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
  // Paragraphs
  text = text.split('\\n\\n').map(p => {{
    if (p.startsWith('<pre>')) return p;
    return '<p>' + p.replace(/\\n/g, '<br>') + '</p>';
  }}).join('');
  return text;
}}

function escapeHtml(s) {{
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

function addMessage(role, content) {{
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const avatar = role === 'user' ? 'U' : 'AI';
  div.innerHTML = `<div class="avatar">${{avatar}}</div><div class="bubble">${{
    role === 'user' ? '<p>' + escapeHtml(content).replace(/\\n/g, '<br>') + '</p>'
                    : renderMarkdown(content)
  }}</div>`;
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}}

function setGenerating(val) {{
  generating = val;
  sendBtn.style.display = val ? 'none' : 'block';
  stopBtn.style.display = val ? 'block' : 'none';
  sendBtn.disabled = val;
  inputEl.disabled = val;
  if (!val) inputEl.focus();
}}

function newChat() {{
  history = [];
  messagesEl.innerHTML = '';
  if (generating) stopGeneration();
  inputEl.focus();
}}

function stopGeneration() {{
  if (abortController) {{
    abortController.abort();
    abortController = null;
  }}
  setGenerating(false);
}}

async function sendMessage() {{
  const text = inputEl.value.trim();
  if (!text || generating) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  addMessage('user', text);
  history.push({{ role: 'user', content: text }});

  setGenerating(true);
  abortController = new AbortController();

  const assistantDiv = addMessage('assistant', '');
  const bubbleEl = assistantDiv.querySelector('.bubble');
  bubbleEl.innerHTML = '<span class="typing-indicator"></span>';

  let fullText = '';

  try {{
    const resp = await fetch('/v1/chat/completions', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        messages: history,
        stream: true,
      }}),
      signal: abortController.signal,
    }});

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {{
      const {{ done, value }} = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, {{ stream: true }});
      const lines = buffer.split('\\n');
      buffer = lines.pop();

      for (const line of lines) {{
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') continue;
        try {{
          const json = JSON.parse(data);
          const delta = json.choices?.[0]?.delta?.content;
          if (delta) {{
            fullText += delta;
            bubbleEl.innerHTML = renderMarkdown(fullText);
            scrollToBottom();
          }}
        }} catch (e) {{}}
      }}
    }}
  }} catch (e) {{
    if (e.name !== 'AbortError') {{
      fullText += '\\n\\n[Error: ' + e.message + ']';
      bubbleEl.innerHTML = renderMarkdown(fullText);
    }}
  }}

  if (fullText) {{
    history.push({{ role: 'assistant', content: fullText }});
  }}
  abortController = null;
  setGenerating(false);
}}

inputEl.focus();
</script>
</body>
</html>"""


def mount_webui(app, model_name: str) -> None:
    """Add GET / (HTML page) and GET /health routes to a FastAPI app."""
    from starlette.responses import HTMLResponse, JSONResponse

    html = get_html(model_name)

    @app.get("/", response_class=HTMLResponse)
    async def web_root():
        return html

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok", "model": model_name})
