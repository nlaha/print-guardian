# Migration Guide: CLI Arguments to Environment Variables

## Summary of Changes

The Print Guardian application has been refactored to use environment variables instead of CLI arguments, making it more suitable for containerized deployments.

## Key Changes

### 1. Configuration System (`config.rs`)

- **Removed**: `argh` dependency and CLI argument parsing
- **Added**: Environment variable loading with defaults
- **Merged**: `Config` and `EnvConfig` into a single `Config` struct
- **Enhanced**: Error handling for missing or invalid environment variables

### 2. Main Application (`main.rs`)

- **Removed**: CLI argument parsing with `argh::from_env()`
- **Updated**: Configuration loading to use `Config::load()`
- **Simplified**: Service initialization with single config struct

### 3. Dependencies (`Cargo.toml`)

- **Removed**: `argh = "0.1.13"` dependency

### 4. Environment Configuration

- **Added**: `.env.example` file with all configuration options
- **Updated**: Docker Compose configuration examples
- **Enhanced**: README documentation

## Environment Variables

### Required

- `IMAGE_URL`: Camera image URL for monitoring
- `DISCORD_WEBHOOK`: Discord webhook URL for alerts
- `MOONRAKER_API_URL`: Moonraker API endpoint for printer control

### Optional (with defaults)

- `LABEL_FILE`: Path to label file (default: "./labels.txt")
- `MODEL_CFG`: Path to model config file (default: "./model.cfg")
- `WEIGHTS_FILE`: Path to weights file (default: "./model-weights.darknet")
- `OUTPUT_DIR`: Output directory (default: "./output")
- `OBJECTNESS_THRESHOLD`: Objectness threshold (default: "0.5")
- `CLASS_PROB_THRESHOLD`: Class probability threshold (default: "0.5")

## Migration Steps

1. **Copy environment template**:

   ```bash
   cp .env.example .env
   ```

2. **Set required variables**:

   ```bash
   export IMAGE_URL="http://your-camera.local/image.jpg"
   export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
   export MOONRAKER_API_URL="http://your-printer.local:7125"
   ```

3. **Run without CLI arguments**:
   ```bash
   ./print-guardian
   ```

## Docker Usage

The application is now fully compatible with containerized environments:

```bash
# Using environment variables
docker run --env-file .env print-guardian

# Using docker-compose
docker-compose up
```

## Benefits

- ✅ Better container support
- ✅ Simplified deployment
- ✅ Environment-specific configuration
- ✅ Reduced CLI complexity
- ✅ Follows 12-factor app principles
