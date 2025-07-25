# Print Guardian
[![CI](https://github.com/nlaha/print-guardian/actions/workflows/ci.yaml/badge.svg)](https://github.com/nlaha/print-guardian/actions/workflows/ci.yaml)

AI-powered 3D print failure detection system that monitors printer cameras in real-time and automatically responds to detected failures.

Supported alerting methods:
- Discord webhook

Utilizes the Obico Darknet model for object detection, integrated with Discord for alerts and Moonraker for printer control.

<img width="522" height="461" alt="image" src="https://github.com/user-attachments/assets/e10f3c66-688f-4da0-9925-99f5f6b50576" />
<img width="454" height="358" alt="image" src="https://github.com/user-attachments/assets/2ada9d3a-dda0-4e75-9557-ac9aea56f2c9" />


## Installation

### Building

```bash
git clone https://github.com/yourusername/print-guardian
cd print-guardian
cargo build --release
```

## Configuration

### Environment Variables

All configuration is handled through environment variables to support containerized deployments. Create a `.env` file or set these environment variables:

#### Required Variables

```bash
export IMAGE_URL="http://camera.local/image.jpg"
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
export MOONRAKER_API_URL="http://your-printer.local:7125"
```

#### Optional Variables

```bash
export LABEL_FILE="./labels.txt"
export MODEL_CFG="./model.cfg"
export WEIGHTS_FILE="./model/model-weights.darknet"
export OBJECTNESS_THRESHOLD="0.08"
export CLASS_PROB_THRESHOLD="0.4"
```

### Using .env File

Copy the example environment file and customize it:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Basic Usage

`IMAGE_URL` can be a single URL or a comma-separated list. These URLs will be cycled through sequentially for detections.
```bash
# Set environment variables
export IMAGE_URL="http://camera.local/image.jpg"
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
export MOONRAKER_API_URL="http://printer.local:7125"

# Run with default settings
./print-guardian

# Or use a .env file
cp .env.example .env
# Edit .env with your configuration
./print-guardian
```

### Docker Usage

Using environment variables:

```bash
docker build -t print-guardian .
docker run \
  -e IMAGE_URL="http://camera.local/image.jpg" \
  -e DISCORD_WEBHOOK="https://discord.com/api/webhooks/..." \
  -e MOONRAKER_API_URL="http://printer.local:7125" \
  print-guardian
```

Using .env file:

```bash
docker run --env-file .env print-guardian
```

### Docker Compose

```yaml
services:
  worker:
    image: printguardian
    restart: unless-stopped
    build:
      context: .
      dockerfile: ./Dockerfile
    env_file: .env
    volumes:
      - ./model:/usr/src/app/model
    healthcheck:
      test: ["CMD-SHELL", "test -f .ready || exit 1"]
      interval: 1m30s
      timeout: 30s
      retries: 5
      start_period: 30s
    devices:
      # VAAPI Devices
      - /dev/dri:/dev/dri
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 1024M
        reservations:
          cpus: '4'
          memory: 512M
```

## Model Files

The application requires these files:

1. **model.cfg** - YOLO configuration file
2. **model-weights.darknet** - Trained model weights (auto-downloaded if missing)
3. **labels.txt** - Class labels file
