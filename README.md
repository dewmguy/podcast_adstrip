# Podcast AdStrip CLI

**Podcast AdStrip CLI** is command-line tool for automatically removing ads from podcast audio files. It uses Whisper for transcription and either a local LLM or OpenAI's GPT model to identify and strip ad segments, delivering a clean version of the audio. It is a fork of https://github.com/jdrbc/podly_pure_podcasts

---

## Features

- Transcribes podcast audio using Whisper (local)
- Identifies ad segments using a local LLM (via HTTP API) or OpenAI's GPT (e.g., GPT-4o)
- Supports two context modes: `stateless` (system prompt + transcript) and `assistant` (OpenAI assistant thread)
- Clips out ad segments with optional audio fade in/out transitions (default: 500 ms)
- Caches transcript data for reuse and debugging
- CLI interface for processing single files or batches from a directory
- Allows selection of LLM backend and model dynamically via CLI flags

---

## Installation

1. **Install FFmpeg:**
```bash
sudo apt install ffmpeg
```

2. **Set up Python environment (with pipenv):**
```bash
pip install pipenv
pipenv --python 3.11
pipenv install
pipenv shell
```

3. **Add API keys (if using OpenAI GPT):**
Create a `.env` file or export your OpenAI credentials:
```bash
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_ASSISTANT_ID=your_openai_assistant_id   # Only needed for assistant mode
```

---

## Usage

### Basic CLI
```bash
python strip_ads.py path/to/audio.mp3 --output path/to/output/
```

### CLI Arguments
- `input` (positional): Path to input `.mp3` file or directory
- `--output`: Output directory (default: `output`)
- `--llm`: LLM backend to use (`local` or `chatgpt`, default: `chatgpt`)
- `--model`: Model to use (e.g., `gpt-4o`, `mixtral`, default: `gpt-4o`)
- `--context`: Context mode (`stateless` or `assistant`, default: `stateless`)

### Example: OpenAI stateless mode
```bash
python strip_ads.py podcast.mp3 \
  --llm chatgpt \
  --model gpt-4o \
  --context stateless
```

### Example: Local model via HTTP API (Ollama)
```bash
python strip_ads.py podcast.mp3 \
  --llm local \
  --model mixtral \
  --context stateless
```

### Example: Batch process a folder
```bash
python strip_ads.py /path/to/audio/folder/ --output /path/to/output/
```

---

## Configuration

### System Prompt
When using `stateless` context mode, you must include a system prompt at:
```
config/system_prompt.txt
```
This prompt guides the model in identifying ad segments.

---

## Notes

- Transcripts are cached in `.cache_segments/` for speed and debugging
- Segment classification uses 35-line transcript chunks
- Minimum ad segment duration: 2.0 seconds
- Minimum separation between ads: 1.0 second
- Fade duration between audio segments is set to 500 milliseconds by default
- Final output file includes a timestamp and model label in the filename
- Local model must support the Ollama-compatible API at `http://localhost:11434/api/generate`

---

## Credits

Developed as a lightweight CLI tool for automated podcast ad stripping using Whisper and LLM inference.

