# DeepSeek-OCR server

## Setup Instructions

```bash
# Create a Python virtual environment
python -m venv venv

# Activate the environment
. venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Server

### Run with nohup
```bash
nohup python server.py > server.log 2>&1 &
```

## Monitor the Server

```bash
# View server logs in real-time
tail -f server.log

# Check if server is running
ps aux | grep server.py

# Check if port 9999 is listening
lsof -i :9999
```

## Monitor GPU Usage

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# One-time GPU status
nvidia-smi

# Detailed GPU memory usage
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
```

## Stop the Server

```bash
# Find and kill the process
pkill -f server.py
```


---
Updated at 14 Jan, 2026