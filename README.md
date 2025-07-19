# Print Guardian

AI-powered 3D print failure detection system that monitors printer cameras in real-time and automatically responds to detected failures.

Supported alerting methods:
- Discord webhook

Utilizes the Obico Darknet model for object detection, integrated with Discord for alerts and Moonraker for printer control.

<img width="1043" height="922" alt="image" src="https://github.com/user-attachments/assets/e10f3c66-688f-4da0-9925-99f5f6b50576" />


## Installation

### Building

```bash
git clone https://github.com/yourusername/print-guardian
cd print-guardian
cargo build --release
```

## Configuration

### Environment Variables

All configuration is now handled through environment variables to support containerized deployments. Create a `.env` file or set these environment variables:

#### Required Variables

```bash
export IMAGE_URL="http://camera.local/image.jpg"
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
export MOONRAKER_API_URL="http://your-printer.local:7125"
```

#### Optional Variables (with defaults)

```bash
export LABEL_FILE="./labels.txt"
export MODEL_CFG="./model.cfg"
export WEIGHTS_FILE="./model/model-weights.darknet"
export OUTPUT_DIR="./output"
export OBJECTNESS_THRESHOLD="0.5"
export CLASS_PROB_THRESHOLD="0.5"
```

### Using .env File

Copy the example environment file and customize it:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Basic Usage

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
version: "3.8"
services:
  print-guardian:
    build: .
    env_file: .env
    # Or use environment variables directly:
    # environment:
    #   - IMAGE_URL=http://camera.local/image.jpg
    #   - DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
    #   - MOONRAKER_API_URL=http://printer.local:7125
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

## Model Files

The application requires these files:

1. **model.cfg** - YOLO configuration file
2. **model-weights.darknet** - Trained model weights (auto-downloaded if missing)
3. **labels.txt** - Class labels file
