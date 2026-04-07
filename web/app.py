"""
web/app.py
==========
Advanced Web UI for openAuton with user accounts, chat, file upload, and evolution tracking.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
import secrets

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Cookie
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

# ---------- User database (simple file-based) ----------
USERS_DB_PATH = Path("experiments/users.json")
USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_users():
    if USERS_DB_PATH.exists():
        with open(USERS_DB_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_DB_PATH, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed

# ---------- Agent initialization (lazy) ----------
agent = None
episode_store = None
sessions = {}  # session_id -> {username, user_config}

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

def get_user_config(username: str) -> Dict:
    users = load_users()
    if username in users:
        return users[username].get('config', {
            'provider': 'groq',
            'model': 'llama-3.3-70b-versatile',
            'api_key': ''
        })
    return {'provider': 'groq', 'model': 'llama-3.3-70b-versatile', 'api_key': ''}

def save_user_config(username: str, config: Dict):
    users = load_users()
    if username not in users:
        users[username] = {'password': '', 'config': {}}
    users[username]['config'] = config
    save_users(users)

# ---------- Data models ----------
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    password_confirm: str

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    episode_id: Optional[str] = None

class SaveConfigRequest(BaseModel):
    provider: str
    model: str
    api_key: str

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
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root { --primary: #3b82f6; --gray: #64748b; }
        * { transition: background-color 0.2s, border-color 0.2s; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1e293b; }
        ::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #64748b; }
        
        body { background: #0f172a; }
        .gradient-text { color: #3b82f6; }
        .message-user { background: #3b82f6; color: white; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2); }
        .message-agent { background: #1e293b; color: #e2e8f0; border-left: 4px solid #3b82f6; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3); }
        .stat-card { background: #1e293b; border: 1px solid #334155; }
        .stat-card:hover { border-color: #64748b; box-shadow: 0 8px 16px rgba(59, 130, 246, 0.08); }
        .btn-primary { background: #3b82f6; color: white; }
        .btn-primary:hover { background: #2563eb; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
        .btn-secondary { background: #64748b; color: white; }
        .btn-secondary:hover { background: #475569; box-shadow: 0 4px 12px rgba(100, 116, 139, 0.3); }
        .tab-active { border-bottom-color: #3b82f6; color: #3b82f6; }
        .gene-bar { background: #3b82f6; }
        .loading { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes slideIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message-new { animation: slideIn 0.3s ease; }
        .chart-container { position: relative; height: 250px; }
    </style>
</head>
<body class="text-gray-100">
    <div class="flex h-screen flex-col md:flex-row">
        <!-- Sidebar -->
        <div class="w-full md:w-96 bg-gray-900 border-b md:border-b-0 md:border-r border-gray-800 flex flex-col">
            <!-- Header -->
            <div class="p-5 border-b border-gray-800 bg-gray-900">
                <h1 class="text-3xl font-bold text-blue-500">openAuton</h1>
                <p class="text-xs text-gray-400 mt-1">v1.0 - Self-improving AI Engineer</p>
            </div>

            <!-- Tabs -->
            <div class="flex border-b border-gray-800 px-4 pt-4 gap-4">
                <button class="tab-btn tab-active text-sm font-medium pb-3 border-b-2" data-tab="dashboard">Dashboard</button>
                <button class="tab-btn text-sm font-medium pb-3 text-gray-400 hover:text-gray-200" data-tab="training">Training</button>
                <button class="tab-btn text-sm font-medium pb-3 text-gray-400 hover:text-gray-200" data-tab="episodes">History</button>
                <button class="tab-btn text-sm font-medium pb-3 text-gray-400 hover:text-gray-200" data-tab="settings">Settings</button>
            </div>

            <!-- Tab Content -->
            <div class="flex-1 overflow-y-auto p-4">
                <!-- Dashboard Tab -->
                <div class="tab-content space-y-4" id="dashboard-tab">
                    <!-- Stats Grid -->
                    <div class="grid grid-cols-2 gap-3">
                        <div class="stat-card p-4 rounded-lg">
                            <div class="text-2xl font-bold text-blue-400" id="episode-count">0</div>
                            <div class="text-xs text-gray-400 mt-1">Episodes</div>
                        </div>
                        <div class="stat-card p-4 rounded-lg">
                            <div class="text-2xl font-bold text-blue-400" id="success-rate">0%</div>
                            <div class="text-xs text-gray-400 mt-1">Success Rate</div>
                        </div>
                        <div class="stat-card p-4 rounded-lg">
                            <div class="text-2xl font-bold text-blue-400" id="avg-accuracy">0.0</div>
                            <div class="text-xs text-gray-400 mt-1">Avg Accuracy</div>
                        </div>
                        <div class="stat-card p-4 rounded-lg">
                            <div class="text-2xl font-bold text-blue-400" id="avg-reasoning">0.0</div>
                            <div class="text-xs text-gray-400 mt-1">Score</div>
                        </div>
                    </div>

                    <!-- DNA Panel -->
                    <div class="stat-card p-4 rounded-lg border border-gray-700">
                        <h3 class="text-sm font-semibold text-blue-400 mb-3">
                            Cognitive DNA
                        </h3>
                        <div id="dna-list" class="space-y-3 text-xs">
                            <div class="loading text-gray-500">Loading genes...</div>
                        </div>
                    </div>

                    <!-- Chart -->
                    <div class="stat-card p-4 rounded-lg border border-gray-700">
                        <h3 class="text-sm font-semibold text-blue-400 mb-3">Accuracy Trend</h3>
                        <div class="chart-container">
                            <canvas id="accuracyChart"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Training Tab -->
                <div class="tab-content hidden space-y-4" id="training-tab">
                    <div class="stat-card p-4 rounded-lg border border-gray-700">
                        <h3 class="text-sm font-semibold text-blue-400 mb-3">Train on File</h3>
                        <div class="space-y-3">
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">Select CSV File</label>
                                <input type="file" id="file-input" accept=".csv" class="w-full text-xs text-gray-300 file:mr-2 file:py-2 file:px-3 file:rounded file:bg-blue-600 file:text-white file:border-0 file:cursor-pointer file:hover:bg-blue-700">
                            </div>
                            <div class="grid grid-cols-2 gap-2">
                                <div>
                                    <label class="block text-xs font-medium text-gray-300 mb-1">Epochs</label>
                                    <input type="number" id="epochs" value="10" min="1" max="100" class="w-full text-xs bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                                </div>
                                <div>
                                    <label class="block text-xs font-medium text-gray-300 mb-1">Batch Size</label>
                                    <input type="number" id="batch-size" value="8" min="1" max="256" class="w-full text-xs bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                                </div>
                            </div>
                            <button id="upload-btn" class="w-full btn-primary text-white text-xs font-medium py-2 rounded-lg hover:shadow-lg">
                                Start Training
                            </button>
                            <div id="upload-progress" class="text-xs text-gray-400 hidden text-center">
                                Training in progress...
                            </div>
                            <div id="training-result" class="text-xs p-2 rounded hidden"></div>
                        </div>
                    </div>
                </div>

                <!-- Episodes Tab -->
                <div class="tab-content hidden space-y-3" id="episodes-tab">
                    <h3 class="text-sm font-semibold text-blue-400 mb-2">Recent Episodes</h3>
                    <div id="episodes-list" class="space-y-2">
                        <div class="loading text-gray-500 text-center text-xs py-4">Loading episodes...</div>
                    </div>
                </div>

                <!-- Settings Tab -->
                <div class="tab-content hidden space-y-4" id="settings-tab">
                    <div class="stat-card p-4 rounded-lg border border-gray-700">
                        <h3 class="text-sm font-semibold text-blue-400 mb-3">LLM Provider Settings</h3>
                        <div class="space-y-3">
                            <!-- Provider Selection -->
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">Select Provider</label>
                                <select id="provider-select" class="w-full text-xs bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                                    <option value="groq">Groq (Fast & Free tier available)</option>
                                    <option value="openai">OpenAI (GPT-4, GPT-4o)</option>
                                    <option value="anthropic">Anthropic (Claude)</option>
                                    <option value="ollama">Ollama (Local)</option>
                                </select>
                            </div>

                            <!-- Model Selection -->
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">Model</label>
                                <input type="text" id="model-input" placeholder="e.g., llama-3.3-70b-versatile" class="w-full text-xs bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                                <div class="text-xs text-gray-500 mt-1" id="model-hint">Groq: llama-3.3-70b-versatile, mixtral-8x7b-32768</div>
                            </div>

                            <!-- API Key Input -->
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">API Key</label>
                                <input type="password" id="api-key-input" placeholder="Enter your API key..." class="w-full text-xs bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none font-mono">
                                <button id="toggle-api-visibility" class="text-xs text-blue-400 mt-1 hover:text-blue-300">Show API Key</button>
                            </div>

                            <!-- Status -->
                            <div id="settings-status" class="text-xs p-2 rounded hidden"></div>

                            <!-- Save Button -->
                            <button id="save-settings-btn" class="w-full btn-primary text-white text-xs font-medium py-2 rounded-lg hover:shadow-lg">
                                Save Settings
                            </button>

                            <!-- Info -->
                            <div class="bg-gray-800 border border-gray-700 rounded p-3 mt-3">
                                <p class="text-xs text-gray-400 mb-2"><strong>Tips:</strong></p>
                                <ul class="text-xs text-gray-500 space-y-1 ml-3">
                                    <li>- Get free Groq API: <a href="https://console.groq.com/keys" target="_blank" class="text-blue-400 hover:text-blue-300">console.groq.com/keys</a></li>
                                    <li>- Settings saved to your account</li>
                                    <li>- API key is never logged</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Footer -->
            <div class="p-4 border-t border-gray-800 text-xs text-gray-500">
                <div class="flex justify-between items-center">
                    <span class="text-green-400">Online</span>
                    <div class="flex items-center gap-2">
                        <span class="inline-flex gap-1">
                            <span class="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse"></span>
                            <span class="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style="animation-delay: 0.2s;"></span>
                            <span class="w-1.5 h-1.5 bg-gray-600 rounded-full animate-pulse" style="animation-delay: 0.4s;"></span>
                        </span>
                        <button id="refresh-btn" class="hover:text-gray-300 ml-2">Refresh</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Chat Area -->
        <div class="flex-1 flex flex-col bg-gray-950">
            <!-- Chat Header -->
            <div class="p-4 border-b border-gray-800 bg-gray-900 flex justify-between items-center">
                <h2 class="text-lg font-semibold text-blue-400">
                    Assistant
                </h2>
                <div class="flex items-center gap-3">
                    <span id="user-display" class="text-xs text-gray-400 hidden"><span id="username-display"></span></span>
                    <button id="logout-btn" class="text-xs text-gray-400 hover:text-gray-200 hidden">Logout</button>
                </div>
            </div>

            <!-- Login/Register Panel (shown before auth) -->
            <div id="auth-panel" class="flex-1 flex items-center justify-center p-4 bg-gray-950">
                <div class="w-full max-w-md">
                    <!-- Login Form -->
                    <div id="login-form" class="stat-card p-6 rounded-lg border border-gray-700">
                        <h3 class="text-lg font-bold text-blue-400 mb-4 text-center">Welcome to openAuton</h3>
                        <div class="space-y-3">
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">Username</label>
                                <input type="text" id="login-username" placeholder="Enter username" class="w-full text-sm bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                            </div>
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">Password</label>
                                <input type="password" id="login-password" placeholder="Enter password" class="w-full text-sm bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                            </div>
                            <div id="login-error" class="text-xs text-red-400 hidden px-2 py-1"></div>
                            <button id="login-btn" class="w-full btn-primary text-white text-sm font-medium py-2 rounded-lg hover:shadow-lg">
                                Login
                            </button>
                            <div class="text-center">
                                <span class="text-xs text-gray-400">Don't have account? </span>
                                <button id="switch-to-register" class="text-xs text-blue-400 hover:text-blue-300 font-medium">Register here</button>
                            </div>
                        </div>
                    </div>

                    <!-- Register Form (hidden by default) -->
                    <div id="register-form" class="stat-card p-6 rounded-lg border border-gray-700 hidden">
                        <h3 class="text-lg font-bold text-blue-400 mb-4 text-center">Create Account</h3>
                        
                        <!-- Social Login Options -->
                        <div class="space-y-2 mb-4">
                            <button onclick="window.open('https://github.com/login/oauth/authorize?client_id=YOUR_GITHUB_CLIENT_ID&redirect_uri=http://127.0.0.1:8000/api/auth/github/callback', '_blank')" class="w-full flex items-center justify-center gap-2 p-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-sm text-gray-300">
                                <i class="fab fa-github"></i> Sign up with GitHub
                            </button>
                            <button onclick="alert('Google OAuth coming soon!')" class="w-full flex items-center justify-center gap-2 p-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-sm text-gray-300">
                                <i class="fab fa-google"></i> Sign up with Google
                            </button>
                            <button onclick="alert('Discord OAuth coming soon!')" class="w-full flex items-center justify-center gap-2 p-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-sm text-gray-300">
                                <i class="fab fa-discord"></i> Sign up with Discord
                            </button>
                        </div>

                        <div class="relative mb-4">
                            <div class="absolute inset-0 flex items-center">
                                <div class="w-full border-t border-gray-700"></div>
                            </div>
                            <div class="relative flex justify-center text-xs">
                                <span class="px-2 bg-gray-700 text-gray-400">or sign up with email</span>
                            </div>
                        </div>

                        <div class="space-y-3">
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">Email Address</label>
                                <input type="email" id="register-email" placeholder="your@email.com" class="w-full text-sm bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                            </div>
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">Username (optional)</label>
                                <input type="text" id="register-username-email" placeholder="Choose a username (3+ chars)" class="w-full text-sm bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                            </div>
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">Password</label>
                                <input type="password" id="register-password" placeholder="Enter password (6+ chars)" class="w-full text-sm bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                            </div>
                            <div>
                                <label class="block text-xs font-medium text-gray-300 mb-1">Confirm Password</label>
                                <input type="password" id="register-password-confirm" placeholder="Confirm password" class="w-full text-sm bg-gray-800 rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none">
                            </div>
                            <div class="flex items-start gap-2">
                                <input type="checkbox" id="terms-checkbox" class="mt-1 rounded">
                                <label class="text-xs text-gray-400">I agree to the <span class="text-blue-400 hover:underline cursor-pointer">Terms of Service</span> and <span class="text-blue-400 hover:underline cursor-pointer">Privacy Policy</span></label>
                            </div>
                            <div id="register-error" class="text-xs text-red-400 hidden px-2 py-1"></div>
                            <button id="register-btn" class="w-full btn-primary text-white text-sm font-medium py-2 rounded-lg hover:shadow-lg">
                                Create Account
                            </button>
                            <div class="text-center">
                                <span class="text-xs text-gray-400">Already have account? </span>
                                <button id="switch-to-login" class="text-xs text-blue-400 hover:text-blue-300 font-medium">Login here</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Chat Messages (shown after auth) -->
            <div id="chat-panel" class="hidden flex-1 flex flex-col">
                <div class="flex-1 overflow-y-auto p-4 space-y-4" id="chat-messages">
                    <div class="message-agent p-4 rounded-lg max-w-2xl">
                        <div class="flex items-start gap-3">
                            <div>
                                <div class="font-semibold text-sm text-blue-300 mb-1">openAuton</div>
                                <p class="text-sm">Welcome! I'm openAuton, your autonomous AI engineer. You can:</p>
                                <ul class="text-sm mt-2 ml-4 space-y-1 text-gray-300">
                                    <li>- Chat tasks: "Analyze this data" or "Build a classifier"</li>
                                    <li>- Train models: Upload CSV files and let me train on them</li>
                                    <li>- Monitor progress: Watch my evolution in real-time</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Input Area -->
                <div class="p-4 border-t border-gray-800 bg-gray-900">
                    <div class="flex gap-3">
                        <input type="text" id="chat-input" placeholder="Describe your task..." class="flex-1 bg-gray-800 rounded-lg px-4 py-3 border border-gray-700 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500/50 text-sm">
                        <button id="send-btn" class="btn-primary text-white px-6 py-3 rounded-lg font-medium hover:shadow-lg">
                            Send
                        </button>
                    </div>
                    <p class="text-xs text-gray-500 mt-2 ml-2">Tip: Configure your API in Settings first</p>
                </div>
            </div>
        </div>
    </div>


    <script>
        // Chart instances
        let accuracyChart = null;
        
        // Session management
        let currentSession = localStorage.getItem('session_id');
        let currentUsername = localStorage.getItem('username');

        // DOM elements
        const authPanel = document.getElementById('auth-panel');
        const chatPanel = document.getElementById('chat-panel');
        const loginForm = document.getElementById('login-form');
        const registerForm = document.getElementById('register-form');
        const loginBtn = document.getElementById('login-btn');
        const registerBtn = document.getElementById('register-btn');
        const switchToRegister = document.getElementById('switch-to-register');
        const switchToLogin = document.getElementById('switch-to-login');
        const loginUsername = document.getElementById('login-username');
        const loginPassword = document.getElementById('login-password');
        const registerUsername = document.getElementById('register-username');
        const registerPassword = document.getElementById('register-password');
        const registerPasswordConfirm = document.getElementById('register-password-confirm');
        const loginError = document.getElementById('login-error');
        const registerError = document.getElementById('register-error');
        const logoutBtn = document.getElementById('logout-btn');
        const userDisplay = document.getElementById('user-display');
        const usernameDisplay = document.getElementById('username-display');

        const chatMessages = document.getElementById('chat-messages');
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const dnaList = document.getElementById('dna-list');
        const episodeCountSpan = document.getElementById('episode-count');
        const successRateSpan = document.getElementById('success-rate');
        const avgAccuracySpan = document.getElementById('avg-accuracy');
        const avgReasoningSpan = document.getElementById('avg-reasoning');
        const fileInput = document.getElementById('file-input');
        const uploadBtn = document.getElementById('upload-btn');
        const uploadProgress = document.getElementById('upload-progress');
        const trainingResult = document.getElementById('training-result');
        const epochsInput = document.getElementById('epochs');
        const batchSizeInput = document.getElementById('batch-size');
        const refreshBtn = document.getElementById('refresh-btn');
        const episodesList = document.getElementById('episodes-list');

        // Settings elements
        const providerSelect = document.getElementById('provider-select');
        const modelInput = document.getElementById('model-input');
        const apiKeyInput = document.getElementById('api-key-input');
        const toggleApiVisibility = document.getElementById('toggle-api-visibility');
        const saveSettingsBtn = document.getElementById('save-settings-btn');
        const settingsStatus = document.getElementById('settings-status');
        const modelHint = document.getElementById('model-hint');

        // LLM Config
        let llmConfig = {
            provider: 'groq',
            model: 'llama-3.3-70b-versatile',
            api_key: ''
        };

        // Auth handlers
        async function handleLogin() {
            const username = loginUsername.value.trim();
            const password = loginPassword.value;
            
            if (!username || !password) {
                showError(loginError, 'Please enter username and password');
                return;
            }
            
            try {
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                
                if (data.success) {
                    currentSession = data.session_id;
                    currentUsername = username;
                    llmConfig = data.config;
                    
                    localStorage.setItem('session_id', currentSession);
                    localStorage.setItem('username', currentUsername);
                    
                    showChatPanel();
                } else {
                    showError(loginError, data.error);
                }
            } catch (err) {
                showError(loginError, 'Login error: ' + err.message);
            }
        }

        async function handleRegister() {
            const username = registerUsername.value.trim();
            const password = registerPassword.value;
            const passwordConfirm = registerPasswordConfirm.value;
            
            if (!username || !password || !passwordConfirm) {
                showError(registerError, 'Please fill all fields');
                return;
            }
            
            try {
                const res = await fetch('/api/auth/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password, password_confirm: passwordConfirm })
                });
                const data = await res.json();
                
                if (data.success) {
                    showError(registerError, data.message, 'success');
                    setTimeout(() => switchToLogin.click(), 2000);
                } else {
                    showError(registerError, data.error);
                }
            } catch (err) {
                showError(registerError, 'Registration error: ' + err.message);
            }
        }

        function showError(element, message, type = 'error') {
            element.textContent = message;
            element.className = type === 'success' ? 'text-xs text-green-400 hidden px-2 py-1' : 'text-xs text-red-400 px-2 py-1';
            element.classList.remove('hidden');
        }

        function showChatPanel() {
            authPanel.classList.add('hidden');
            chatPanel.classList.remove('hidden');
            userDisplay.classList.remove('hidden');
            logoutBtn.classList.remove('hidden');
            usernameDisplay.textContent = currentUsername;
            loadSettings();
            refreshMemory();
            refreshDNA();
            initChart();
            updateChart();
            
            // Auto-refresh every 10 seconds
            setInterval(() => {
                refreshMemory();
                refreshDNA();
            }, 10000);
        }

        function showAuthPanel() {
            authPanel.classList.remove('hidden');
            chatPanel.classList.add('hidden');
            userDisplay.classList.add('hidden');
            logoutBtn.classList.add('hidden');
            currentSession = null;
            currentUsername = null;
            localStorage.removeItem('session_id');
            localStorage.removeItem('username');
        }

        async function handleLogout() {
            await fetch('/api/auth/logout', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSession })
            });
            showAuthPanel();
            loginForm.classList.remove('hidden');
            registerForm.classList.add('hidden');
        }

        // Tab management
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tabName = e.currentTarget.dataset.tab;
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('tab-active', 'border-b-4', 'border-blue-500'));
                e.currentTarget.classList.add('tab-active', 'border-b-4', 'border-blue-500');
                document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
                document.getElementById(tabName + '-tab').classList.remove('hidden');
            });
        });

        // Add message to chat with animation
        function addMessage(text, isUser) {
            const div = document.createElement('div');
            div.className = `message-new p-4 rounded-lg max-w-2xl ${isUser ? 'message-user ml-auto' : 'message-agent'} flex items-start gap-3`;
            
            if (!isUser) {
                div.innerHTML = `
                    <div>
                        <div class="font-semibold text-sm text-blue-300 mb-1">openAuton</div>
                        <div class="text-sm break-words">${text}</div>
                    </div>
                `;
            } else {
                div.innerHTML = `<div class="text-sm break-words flex-1">${text}</div>`;
                div.style.marginLeft = 'auto';
                div.style.alignItems = 'flex-start';
            }
            
            chatMessages.appendChild(div);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Send chat message
        async function sendMessage() {
            const msg = chatInput.value.trim();
            if (!msg) return;
            
            if (!llmConfig.api_key) {
                addMessage('Please configure your API key in Settings first!', false);
                return;
            }
            
            addMessage(msg, true);
            chatInput.value = '';
            sendBtn.disabled = true;
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: msg,
                        api_key: llmConfig.api_key,
                        provider: llmConfig.provider,
                        model: llmConfig.model
                    })
                });
                const data = await res.json();
                addMessage(data.reply, false);
                refreshMemory();
                refreshDNA();
                updateChart();
            } catch (err) {
                addMessage('Error: ' + err.message, false);
            } finally {
                sendBtn.disabled = false;
            }
        }

        // Upload file and train
        async function uploadAndTrain() {
            const file = fileInput.files[0];
            if (!file) {
                trainingResult.innerHTML = '<div class="bg-red-900 text-red-100 p-3 rounded">Please select a file first.</div>';
                trainingResult.classList.remove('hidden');
                return;
            }
            const epochs = epochsInput.value || 10;
            const batchSize = batchSizeInput.value || 8;
            const formData = new FormData();
            formData.append('file', file);
            formData.append('epochs', epochs);
            formData.append('batch_size', batchSize);
            uploadProgress.classList.remove('hidden');
            trainingResult.classList.add('hidden');
            uploadBtn.disabled = true;
            try {
                const res = await fetch('/api/train-file', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                uploadProgress.classList.add('hidden');
                if (data.status === 'success') {
                    trainingResult.innerHTML = `<div class="bg-blue-900 text-blue-100 p-3 rounded">
                        <strong>Training Complete</strong><br>
                        File: ${data.filename}<br>
                        Accuracy: ${(data.accuracy * 100).toFixed(2)}%<br>
                        Episode: ${data.episode_id?.substring(0, 8) || 'N/A'}
                    </div>`;
                    trainingResult.classList.remove('hidden');
                    addMessage(`Successfully trained on ${data.filename} with ${(data.accuracy * 100).toFixed(2)}% accuracy`, false);
                    fileInput.value = '';
                    refreshMemory();
                    refreshDNA();
                    updateChart();
                } else {
                    trainingResult.innerHTML = `<div class="bg-slate-700 text-slate-100 p-3 rounded">Error: ${data.error}</div>`;
                    trainingResult.classList.remove('hidden');
                }
            } catch (err) {
                uploadProgress.classList.add('hidden');
                trainingResult.innerHTML = `<div class="bg-slate-700 text-slate-100 p-3 rounded">Upload error: ${err.message}</div>`;
                trainingResult.classList.remove('hidden');
            } finally {
                uploadBtn.disabled = false;
            }
        }

        // Refresh memory stats
        async function refreshMemory() {
            try {
                const res = await fetch('/api/memory');
                const data = await res.json();
                episodeCountSpan.textContent = data.total_episodes;
                successRateSpan.textContent = (data.success_rate * 100).toFixed(0);
                avgAccuracySpan.textContent = data.avg_accuracy?.toFixed(3) || '0.000';
                avgReasoningSpan.textContent = data.avg_reasoning?.toFixed(2) || '0.00';
                
                // Update episodes list
                episodesList.innerHTML = (data.recent || []).map(ep => `
                    <div class="stat-card p-3 rounded-lg border border-gray-700 text-xs">
                        <div class="flex justify-between items-start mb-2">
                            <span class="font-mono text-blue-300">${ep.id}</span>
                            <span class="text-${ep.status === 'success' ? 'green' : 'red'}-400 text-xs font-medium">${ep.status}</span>
                        </div>
                        <p class="text-gray-400 mb-2">${ep.task}</p>
                        <div class="flex justify-between text-gray-500">
                            <span>Acc: ${ep.accuracy ? (ep.accuracy * 100).toFixed(1) + '%' : 'N/A'}</span>
                            <span>${new Date(ep.timestamp).toLocaleDateString()}</span>
                        </div>
                    </div>
                `).join('') || '<p class="text-center text-gray-500 py-4">No episodes yet</p>';
            } catch (err) { console.error(err); }
        }

        // Refresh DNA display
        async function refreshDNA() {
            try {
                const res = await fetch('/api/dna');
                const data = await res.json();
                if (!data.genes || data.genes.length === 0) {
                    dnaList.innerHTML = '<div class="text-gray-500 text-center py-4">No genes evolved yet</div>';
                    return;
                }
                dnaList.innerHTML = data.genes.map(g => `
                    <div class="border-l-2 border-blue-500 pl-3 py-2">
                        <div class="font-semibold text-sm text-blue-300 mb-1">${g.name}</div>
                        <div class="text-xs text-gray-400 mb-2">${g.type}</div>
                        <div class="w-full bg-gray-700 h-2 rounded">
                            <div class="gene-bar h-2 rounded" style="width: ${Math.min(g.confidence * 100, 100)}%"></div>
                        </div>
                        <div class="text-xs text-gray-500 mt-1">Confidence: ${(g.confidence * 100).toFixed(0)}%</div>
                    </div>
                `).join('');
            } catch (err) { console.error(err); }
        }

        // Initialize accuracy chart
        function initChart() {
            const ctx = document.getElementById('accuracyChart');
            if (!ctx) return;
            accuracyChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Accuracy',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 5,
                        pointBackgroundColor: '#3b82f6',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        filler: { propagate: true }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ticks: { color: '#9ca3af', callback: v => (v * 100).toFixed(0) + '%' },
                            grid: { color: '#374151' }
                        },
                        x: { ticks: { color: '#9ca3af' }, grid: { display: false } }
                    }
                }
            });
        }

        // Update accuracy chart with recent data
        async function updateChart() {
            try {
                const res = await fetch('/api/memory');
                const data = await res.json();
                if (accuracyChart && data.recent) {
                    const recentAccuracies = data.recent
                        .filter(ep => ep.accuracy !== null)
                        .slice(0, 8)
                        .reverse();
                    accuracyChart.data.labels = recentAccuracies.map((_, i) => `Episode ${i + 1}`);
                    accuracyChart.data.datasets[0].data = recentAccuracies.map(ep => ep.accuracy);
                    accuracyChart.update();
                }
            } catch (err) { console.error(err); }
        }

        // Auth Event listeners
        loginBtn.addEventListener('click', handleLogin);
        registerBtn.addEventListener('click', handleRegister);
        switchToRegister.addEventListener('click', () => {
            loginForm.classList.add('hidden');
            registerForm.classList.remove('hidden');
            loginError.classList.add('hidden');
            registerError.classList.add('hidden');
        });
        switchToLogin.addEventListener('click', () => {
            loginForm.classList.remove('hidden');
            registerForm.classList.add('hidden');
            loginError.classList.add('hidden');
            registerError.classList.add('hidden');
        });
        logoutBtn.addEventListener('click', handleLogout);

        // Allow Enter key for login/register
        loginPassword.addEventListener('keypress', (e) => { if (e.key === 'Enter') loginBtn.click(); });
        registerPasswordConfirm.addEventListener('keypress', (e) => { if (e.key === 'Enter') registerBtn.click(); });

        // Chat Event listeners
        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
        uploadBtn.addEventListener('click', uploadAndTrain);
        refreshBtn.addEventListener('click', () => {
            refreshMemory();
            refreshDNA();
            updateChart();
        });

        // Settings handlers
        const modelsByProvider = {
            'groq': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
            'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
            'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'],
            'ollama': ['llama2', 'mistral', 'neural-chat']
        };

        // Update model hint when provider changes
        providerSelect.addEventListener('change', (e) => {
            const provider = e.target.value;
            const models = modelsByProvider[provider];
            modelHint.textContent = `${provider.charAt(0).toUpperCase() + provider.slice(1)}: ${models.join(', ')}`;
            if (models.length > 0) {
                modelInput.placeholder = models[0];
            }
        });

        // Toggle API visibility
        toggleApiVisibility.addEventListener('click', () => {
            const isPassword = apiKeyInput.type === 'password';
            apiKeyInput.type = isPassword ? 'text' : 'password';
            toggleApiVisibility.textContent = isPassword ? 'Hide API Key' : 'Show API Key';
        });

        // Save settings to server
        saveSettingsBtn.addEventListener('click', async () => {
            const provider = providerSelect.value;
            const model = modelInput.value || modelsByProvider[provider][0];
            const apiKey = apiKeyInput.value.trim();

            if (!apiKey) {
                showSettingsStatus('Please enter an API key', 'error');
                return;
            }

            try {
                const res = await fetch('/api/config/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: currentSession,
                        provider,
                        model,
                        api_key: apiKey
                    })
                });
                const data = await res.json();
                if (data.success) {
                    llmConfig.provider = provider;
                    llmConfig.model = model;
                    llmConfig.api_key = apiKey;
                    showSettingsStatus('Settings saved successfully', 'success');
                    setTimeout(() => settingsStatus.classList.add('hidden'), 3000);
                } else {
                    showSettingsStatus('Error saving settings', 'error');
                }
            } catch (err) {
                showSettingsStatus('Error: ' + err.message, 'error');
            }
        });

        function showSettingsStatus(message, type) {
            settingsStatus.textContent = message;
            settingsStatus.className = `text-xs p-2 rounded ${type === 'success' ? 'bg-green-900 text-green-100' : 'bg-red-900 text-red-100'}`;
            settingsStatus.classList.remove('hidden');
        }

        // Load settings into form
        function loadSettings() {
            providerSelect.value = llmConfig.provider;
            modelInput.value = llmConfig.model;
            apiKeyInput.value = llmConfig.api_key || '';
            updateModelHint();
        }

        function updateModelHint() {
            const provider = providerSelect.value;
            const models = modelsByProvider[provider];
            modelHint.textContent = `${provider.charAt(0).toUpperCase() + provider.slice(1)}: ${models.join(', ')}`;
        }

        // Initial load
        if (currentSession && currentUsername) {
            showChatPanel();
        } else {
            showAuthPanel();
        }
    </script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(HTML_PAGE)

# ---------- Auth endpoints ----------
@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    """Register a new user."""
    if len(req.username) < 3:
        return {"success": False, "error": "Username must be at least 3 characters"}
    if len(req.password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters"}
    if req.password != req.password_confirm:
        return {"success": False, "error": "Passwords don't match"}
    
    users = load_users()
    if req.username in users:
        return {"success": False, "error": "Username already exists"}
    
    users[req.username] = {
        'password': hash_password(req.password),
        'config': {'provider': 'groq', 'model': 'llama-3.3-70b-versatile', 'api_key': ''}
    }
    save_users(users)
    return {"success": True, "message": "Registration successful! Please login."}

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    """Login user."""
    users = load_users()
    if req.username not in users:
        return {"success": False, "error": "Username not found"}
    
    if not verify_password(req.password, users[req.username]['password']):
        return {"success": False, "error": "Invalid password"}
    
    # Create session
    session_id = secrets.token_hex(16)
    sessions[session_id] = {
        'username': req.username,
        'config': users[req.username].get('config', {})
    }
    
    return {
        "success": True,
        "session_id": session_id,
        "username": req.username,
        "config": users[req.username].get('config', {})
    }

@app.post("/api/auth/logout")
async def logout(session_id: str = None):
    """Logout user."""
    if session_id and session_id in sessions:
        del sessions[session_id]
    return {"success": True}

@app.post("/api/config/save")
async def save_config(session_id: str, req: SaveConfigRequest):
    """Save user's LLM configuration."""
    if session_id not in sessions:
        return {"success": False, "error": "Invalid session"}
    
    username = sessions[session_id]['username']
    config = {
        'provider': req.provider,
        'model': req.model,
        'api_key': req.api_key
    }
    save_user_config(username, config)
    sessions[session_id]['config'] = config
    return {"success": True, "message": "Settings saved"}

@app.get("/api/config/get")
async def get_config(session_id: str = None):
    """Get user's current configuration."""
    if session_id and session_id in sessions:
        return sessions[session_id]['config']
    return {'provider': 'groq', 'model': 'llama-3.3-70b-versatile', 'api_key': ''}

# ---------- API endpoints ----------
@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a task description to the agent (runs full loop)."""
    # Get or use provided LLM config
    api_key = req.api_key
    provider = req.provider or "groq"
    model = req.model or "llama-3.3-70b-versatile"
    
    if not api_key:
        return ChatResponse(reply="Please configure API key in Settings first")
    
    agent = get_agent()
    
    # Set up LLM provider with user's config
    try:
        from src.llm.provider import ModelProvider
        agent.llm = ModelProvider(provider=provider, model=model, api_key=api_key)
    except Exception as e:
        return ChatResponse(reply=f"Error setting up LLM: {str(e)}")
    
    try:
        result = agent.run(req.message)
        reply = f"**Status:** {result.get('status')}\n**Reflection:** {result.get('reflection', '')}\n**Next step:** {result.get('next_step', '')}"
        return ChatResponse(reply=reply, episode_id=result.get('plan_id'))
    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}")

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
