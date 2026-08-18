# AI-system

Personal training and testing repo for AI/ML experiments.

| Project | What it is | Status |
|---|---|---|
| [`toolbox/`](toolbox/) | FastAPI service that proxies prompts to a local llama.cpp server | Working — early scratch service, two endpoints |

---

# toolbox

A thin FastAPI wrapper around a **local** llama.cpp server, talking to it over
its OpenAI-compatible API. There are no external API calls — the model runs on
your own machine, and the app is just a client in front of it.

Two endpoints so far: a health check and a raw chat passthrough.

## Requirements

- Python 3.12+ (or Docker)
- A llama.cpp build with `llama-server` on your `PATH`
- A GGUF model file

## Quick start

Everything runs through `make`. Run `make` on its own to list all commands.

### 1. Start the model server

The app is only a client — **it cannot work without this running.**

```bash
make llama
```

Leave that terminal open. Wait for `main: server is listening on http://0.0.0.0:8085`.

Check it from a second terminal:

```bash
make check-llama      # prints JSON listing the loaded model
```

> `make llama` assumes the model lives at `$HOME/llama-cpp-model/models/qwen3-1.7b.gguf`.
> Override with `make llama LLAMA_DIR=... LLAMA_MODEL_FILE=...`, or start
> `llama-server` yourself — the app only cares that it answers on port 8085.

### 2. Start the app

Pick **one** of these. They both bind port 8001, so they cannot run at once.

**Local:**

```bash
make install          # first time only
make run              # occupies this terminal
```

**Docker:**

```bash
make up               # builds and starts detached
make ps               # confirm "Up (healthy)"
make logs             # follow logs; Ctrl-C exits the view, not the app
make down             # stop
```

### 3. Try it

```bash
make health           # {"status":"ok"}
make chat             # {"answer":"Hello"}
make docs             # prints the Swagger UI URL
```

## Configuration

All settings live in `.env`, which is **not** committed. Copy the template:

```bash
cp toolbox/.env.example toolbox/.env
```

| Variable | Default | Used by |
|---|---|---|
| `LLAMA_BASE_URL` | `http://localhost:8085/v1` | app + Makefile (host) |
| `LLAMA_BASE_URL_DOCKER` | `http://host.docker.internal:8085/v1` | docker-compose |
| `LLAMA_MODEL` | `qwen3` | app |
| `LLAMA_TIMEOUT` | `120` | app — local generation is slow; httpx's 5s default is far too short |
| `APP_HOST` / `APP_PORT` | `127.0.0.1` / `8001` | Makefile + docker-compose |

`make config` prints the effective values — run it first whenever something
looks wrong.

Two things that trip people up:

- **`.env` beats the shell environment** in make. For a one-off override use
  make's own syntax: `make health APP_PORT=9999`, *not*
  `APP_PORT=9999 make health` (silently ignored).
- **Port 8001, not 8000.** Port 8000 is commonly taken by ChromaDB.

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /chat/raw`

```bash
curl -X POST http://127.0.0.1:8001/chat/raw \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Say hello in one word"}'
```

```json
{"answer": "Hello"}
```

The prompt travels in the JSON body, not the query string, so prompts stay out
of URLs and access logs. If the model returns a reasoning trace, only the final
content is returned.

| Status | Meaning |
|---|---|
| `200` | OK |
| `422` | Missing or empty `prompt` |
| `502` | Model server unreachable or returned something unusable |

## Layout

```
toolbox/
├── app/
│   ├── main.py                  FastAPI routes
│   ├── config.py                pydantic-settings, reads .env
│   ├── dependencies.py          cached client factory
│   └── adapters/llama_client.py async httpx client for the OpenAI-compatible API
├── Dockerfile                   runs non-root
├── docker-compose.yml           dev stack
└── Makefile                     single human interface
```

## Docker notes

Inside a container, `localhost` is the *container*, not your host — so the
model server is reached via `host.docker.internal`, wired up by the
`extra_hosts: host-gateway` entry in `docker-compose.yml`.

This requires `llama-server` to bind `0.0.0.0` (as `make llama` does). That
also exposes it to your local network. If you only ever use `make run`, prefer
`make llama LLAMA_BIND=127.0.0.1` — but note the Docker path will then lose
access, and you'd need `network_mode: host` instead.

The image never contains `.env`; config is injected by compose at runtime.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `App is not running at ...` | App not started, or you hit it during a `--reload` restart. Wait for `Application startup complete.` |
| `Model server is not running at ...` | Run `make llama` |
| `address already in use` | `make run` and `make up` both want 8001 — stop one. Port 8000 is ChromaDB. |
| `502` from `/chat/raw` | App is up but can't reach the model. Check `make config`, then `make check-llama`. |
