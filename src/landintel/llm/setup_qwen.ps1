# Local Qwen setup for the LandIntel LLM brain (Windows / PowerShell).
#
# Installs Ollama and pulls Qwen so the LLM brain runs LOCALLY (free/offline). The pipeline
# auto-detects it via landintel.llm.providers.local_llm_status() -- no code change needed once
# this completes. Run:  powershell -ExecutionPolicy Bypass -File setup_qwen.ps1
#
# Model: qwen2.5:7b (~4.7 GB, good default). For a stronger model set $Model = "qwen2.5:14b".

param([string]$Model = "qwen2.5:7b")

Write-Host "== LandIntel local-LLM setup (Qwen via Ollama) =="

# 1) Ensure Ollama is installed
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama not found -> installing..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    } else {
        Write-Host "winget unavailable. Download the installer from https://ollama.com/download and re-run."
        exit 1
    }
} else {
    Write-Host "Ollama already installed: $((ollama --version) 2>$null)"
}

# 2) Start the Ollama service (no-op if already running)
Start-Process -WindowStyle Hidden ollama -ArgumentList "serve" -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 3) Pull the model
Write-Host "Pulling $Model (this downloads a few GB the first time)..."
ollama pull $Model

# 4) Point the brain at this model (process env; persist in your .env for the worker)
Write-Host "`nSet these for the pipeline:"
Write-Host "  `$env:LANDINTEL_LOCAL_LLM = '$Model'"
Write-Host "  `$env:LANDINTEL_LLM_ORDER = 'local,claude,manus'   # Qwen-first"
Write-Host "`nVerify with:  python -c `"from landintel.llm.providers import local_llm_status; print(local_llm_status())`""
