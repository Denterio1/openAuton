"""
web/app.py
==========
Advanced Web UI for openAuton with chat, file upload, memory, DNA viewer, and streaming logs.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import aiofiles

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.agent import PrimeAgent, AgentConfig
from src.experience.episodes import EpisodeStore
from src.llm.provider import ModelProvider
from src.genome.dna import CognitiveDNA

# ---------- FastAPI app ----------
app = FastAPI(title="openAuton", description="Self‑improving AI engineer")

# Static files (optional, for custom CSS/JS)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates (optional – we'll embed HTML directly for simplicity)
# ---------- Agent initialization (lazy) ----------
agent = None
episode_store = None

def get_agent():
    global agent
    if agent is None:
        agent = PrimeAgent(config=AgentConfig(verbose=False))
    return agent

def get_episode_store():
    global episode_store
    if episode_store is None:
        episode_store = EpisodeStore(Path("experiments/episodes"))
    return episode_store

# For streaming training logs
training_streams: Dict[str, asyncio.Queue] = {}

# ---------- Data models ----------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    episode_id: Optional[str] = None

class TrainFileRequest(BaseModel):
    file_path: str
    epochs: int = 10
    batch_size: int = 8

# ---------- HTML (embedded) ----------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>openAuton – Self‑improving AI Engineer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #1e293b; }
        ::-webkit-scrollbar-thumb { background: #475569; border-radius: 3px; }
        .message-user { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; }
        .message-agent { background: #1e293b; color: #e2e8f0; border-left: 4px solid #3b82f6; }
        .progress-bar { transition: width 0.3s ease; }
    </style>
</head>
<body class="bg-gray-900 text-gray-200">
    <div class="flex h-screen">
        <!-- Sidebar -->
        <div class="w-80 bg-gray-800 border-r border-gray-700 flex flex-col">
            <div class="p-4 border-b border-gray-700">
                <h1 class="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">openAuton</h1>
                <p class="text-xs text-gray-400 mt-1">Self‑improving AI engineer</p>
            </div>
            <div class="flex-1 overflow-y-auto p-3 space-y-4">
                <!-- DNA Panel -->
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-700">
                    <h2 class="text-sm font-semibold text-blue-400 mb-2">🧬 Cognitive DNA</h2>
                    <div id="dna-list" class="space-y-2 text-xs">
                        <div class="animate-pulse">Loading genes...</div>
                    </div>
                </div>
                <!-- Memory Stats -->
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-700">
                    <h2 class="text-sm font-semibold text-purple-400 mb-2">📚 Memory</h2>
                    <div id="memory-stats" class="text-xs space-y-1">
                        <div>Episodes: <span id="episode-count">-</span></div>
                        <div>Success rate: <span id="success-rate">-</span>%</div>
                        <div>Avg accuracy: <span id="avg-accuracy">-</span></div>
                    </div>
                </div>
                <!-- Upload -->
                <div class="bg-gray-900 rounded-lg p-3 border border-gray-700">
                    <h2 class="text-sm font-semibold text-green-400 mb-2">📤 Upload & Train</h2>
                    <input type="file" id="file-input" class="text-xs w-full text-gray-300 file:mr-2 file:py-1 file:px-2 file:rounded file:bg-gray-700 file:text-white file:border-0">
                    <div class="flex gap-2 mt-2">
                        <input type="number" id="epochs" placeholder="Epochs" value="10" class="w-20 text-xs bg-gray-700 rounded px-2 py-1">
                        <input type="number" id="batch-size" placeholder="Batch" value="8" class="w-20 text-xs bg-gray-700 rounded px-2 py-1">
                        <button id="upload-btn" class="bg-green-600 hover:bg-green-700 text-xs px-3 py-1 rounded">Train</button>
                    </div>
                    <div id="upload-progress" class="text-xs text-gray-400 mt-2 hidden">Processing...</div>
                </div>
            </div>
            <div class="p-3 border-t border-gray-700 text-xs text-gray-500">
                <button id="dark-mode-toggle" class="w-full text-left">🌓 Dark mode</button>
            </div>
        </div>

        <!-- Main chat area -->
        <div class="flex-1 flex flex-col">
            <div class="flex-1 overflow-y-auto p-4 space-y-3" id="chat-messages">
                <div class="message-agent p-3 rounded-lg max-w-3xl">👋 Hello! I'm openAuton. Describe a task (e.g., "Train a sentiment classifier on reviews.csv") or upload a file.</div>
            </div>
            <div class="p-4 border-t border-gray-700 bg-gray-800">
                <div class="flex gap-2">
                    <input type="text" id="chat-input" placeholder="Type your task..." class="flex-1 bg-gray-900 rounded-lg px-4 py-2 border border-gray-700 focus:outline-none focus:border-blue-500">
                    <button id="send-btn" class="bg-blue-600 hover:bg-blue-700 px-5 py-2 rounded-lg">Send</button>
                </div>
                <div class="text-xs text-gray-500 mt-2">Supports file references, e.g. "train-file data.csv"</div>
            </div>
        </div>
    </div>

    <script>
        // DOM elements
        const chatMessages = document.getElementById('chat-messages');
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const dnaList = document.getElementById('dna-list');
        const episodeCountSpan = document.getElementById('episode-count');
        const successRateSpan = document.getElementById('success-rate');
        const avgAccuracySpan = document.getElementById('avg-accuracy');
        const fileInput = document.getElementById('file-input');
        const uploadBtn = document.getElementById('upload-btn');
        const uploadProgress = document.getElementById('upload-progress');
        const epochsInput = document.getElementById('epochs');
        const batchSizeInput = document.getElementById('batch-size');

        // Helper: add message to chat
        function addMessage(text, isUser) {
            const div = document.createElement('div');
            div.className = `p-3 rounded-lg max-w-3xl ${isUser ? 'message-user ml-auto' : 'message-agent'}`;
            div.textContent = text;
            chatMessages.appendChild(div);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Send chat message
        async function sendMessage() {
            const msg = chatInput.value.trim();
            if (!msg) return;
            addMessage(msg, true);
            chatInput.value = '';
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: msg })
                });
                const data = await res.json();
                addMessage(data.reply, false);
                refreshMemory();
                refreshDNA();
            } catch (err) {
                addMessage('❌ Error: ' + err.message, false);
            }
        }

        // Upload file and train
        async function uploadAndTrain() {
            const file = fileInput.files[0];
            if (!file) {
                addMessage('Please select a file first.', false);
                return;
            }
            const epochs = epochsInput.value || 10;
            const batchSize = batchSizeInput.value || 8;
            const formData = new FormData();
            formData.append('file', file);
            formData.append('epochs', epochs);
            formData.append('batch_size', batchSize);
            uploadProgress.classList.remove('hidden');
            uploadProgress.textContent = 'Uploading and training...';
            try {
                const res = await fetch('/api/train-file', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                uploadProgress.classList.add('hidden');
                if (data.status === 'success') {
                    addMessage(`✅ Training completed on ${data.filename}. Accuracy: ${data.accuracy?.toFixed(3) || 'N/A'}. Model saved.`, false);
                    refreshMemory();
                    refreshDNA();
                } else {
                    addMessage(`❌ Training failed: ${data.error}`, false);
                }
            } catch (err) {
                uploadProgress.classList.add('hidden');
                addMessage(`❌ Upload error: ${err.message}`, false);
            }
        }

        // Refresh memory stats
        async function refreshMemory() {
            try {
                const res = await fetch('/api/memory');
                const data = await res.json();
                episodeCountSpan.textContent = data.total_episodes;
                successRateSpan.textContent = (data.success_rate * 100).toFixed(0);
                avgAccuracySpan.textContent = data.avg_accuracy?.toFixed(3) || 'N/A';
            } catch (err) { console.error(err); }
        }

        // Refresh DNA display
        async function refreshDNA() {
            try {
                const res = await fetch('/api/dna');
                const data = await res.json();
                if (!data.genes || data.genes.length === 0) {
                    dnaList.innerHTML = '<div class="text-gray-500">No genes yet</div>';
                    return;
                }
                dnaList.innerHTML = data.genes.map(g => `
                    <div class="border-l-2 border-blue-500 pl-2">
                        <div class="font-mono text-blue-300">${g.name}</div>
                        <div class="text-gray-400 text-xs">${g.type} | conf: ${g.confidence.toFixed(2)}</div>
                        <div class="w-full bg-gray-700 h-1 rounded mt-1"><div class="bg-blue-500 h-1 rounded" style="width: ${g.confidence*100}%"></div></div>
                    </div>
                `).join('');
            } catch (err) { console.error(err); }
        }

        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
        uploadBtn.addEventListener('click', uploadAndTrain);

        // Initial load
        refreshMemory();
        refreshDNA();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(HTML_PAGE)

# ---------- API endpoints ----------
@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a task description to the agent (runs full loop)."""
    result = get_agent().run(req.message)
    reply = f"**Status:** {result.get('status')}\n**Reflection:** {result.get('reflection', '')}\n**Next step:** {result.get('next_step', '')}"
    return ChatResponse(reply=reply, episode_id=result.get('plan_id'))

@app.post("/api/train-file")
async def train_file(file: UploadFile, epochs: int = Form(10), batch_size: int = Form(8)):
    """Upload a file and run autonomous training."""
    # Save uploaded file
    upload_dir = Path("experiments/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    # Run training
    try:
        result = get_agent().run_on_file(file_path)
        return {
            "status": "success",
            "filename": file.filename,
            "accuracy": result.get("final_accuracy"),
            "episode_id": result.get("episode_id"),
            "gene_hints": result.get("gene_hints", [])
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/memory")
async def get_memory():
    """Return episode statistics and recent episodes."""
    stats = get_episode_store().get_statistics()
    return {
        "total_episodes": stats.get("total", 0),
        "success_rate": stats.get("success_rate", 0),
        "avg_accuracy": stats.get("avg_accuracy", 0),
        "avg_reasoning": stats.get("avg_reasoning_score", 0),
        "recent": [
            {
                "id": ep.episode_id[:8],
                "task": ep.task_description[:60],
                "status": ep.status.value,
                "accuracy": ep.evaluation.accuracy if ep.evaluation else None,
                "timestamp": ep.timestamp.isoformat()
            }
            for ep in get_episode_store().search(limit=10)
        ]
    }

@app.get("/api/dna")
async def get_dna():
    """Return current DNA genes."""
    genes = []
    for g in get_agent().dna.genes:
        genes.append({
            "name": g.name,
            "type": g.gene_type.value,
            "confidence": g.confidence,
            "value": str(g.value)[:100]
        })
    return {"genes": genes}