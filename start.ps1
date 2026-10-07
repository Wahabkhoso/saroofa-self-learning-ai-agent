Set-Location -Path $PSScriptRoot
Start-Process "http://127.0.0.1:8000"
& ".\venv\Scripts\uvicorn.exe" app.main:app --reload