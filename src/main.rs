use anyhow::Result;
use log::{debug, error, info, warn};
use std::{fs, thread, time::Duration};

// Module declarations
mod alerts;
mod config;
mod detector;
mod error;
mod image_fetcher;
mod printer;

// Import our modules
use alerts::AlertService;
use config::{Config, constants};
use detector::FailureDetector;
use image_fetcher::{AlertType, ImageFetcher};
use printer::PrinterService;

/// Print Guardian - AI-powered 3D print failure detection system.
///
/// This application monitors 3D printer cameras in real-time using machine learning
/// to detect print failures such as spaghetti, layer shifts, warping, and other issues.
/// When failures are detected, it can automatically pause the printer and send alerts
/// via Discord webhooks.
///
/// # Features
///
/// * Real-time camera monitoring with retry logic
/// * AI-powered print failure detection using YOLO/Darknet
/// * Automatic printer pause on multiple failures
/// * Discord webhook notifications with rich embeds
/// * Configurable detection thresholds
/// * Robust error handling and recovery
///
/// # Environment Variables
///
/// Required:
/// * `IMAGE_URL` - Camera image URL for monitoring
/// * `DISCORD_WEBHOOK` - Discord webhook URL for alerts
/// * `MOONRAKER_API_URL` - Moonraker API endpoint for printer control
///
/// Optional (with defaults):
/// * `LABEL_FILE` - Path to label file (default: "./labels.txt")
/// * `MODEL_CFG` - Path to model config file (default: "./model.cfg")
/// * `WEIGHTS_FILE` - Path to weights file (default: "./model/model-weights.darknet")
/// * `OUTPUT_DIR` - Output directory (default: "./output")
/// * `OBJECTNESS_THRESHOLD` - Objectness threshold (default: "0.5")
/// * `CLASS_PROB_THRESHOLD` - Class probability threshold (default: "0.5")
/// * `FLIP_IMAGE` - Flip images vertically (default: "false")
///
/// # Usage
///
/// ```bash
/// export IMAGE_URL="http://camera.local/image.jpg"
/// export DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
/// export MOONRAKER_API_URL="http://printer.local:7125"
/// export FLIP_IMAGE="true"  # Optional: flip images if camera is mounted upside-down
/// ./print-guardian
/// ```
fn main() -> Result<()> {
    // Initialize logger to output to stdout, using RUST_LOG env var or info level by default
    env_logger::Builder::from_default_env()
        .target(env_logger::Target::Stdout)
        .filter_level(
            std::env::var("RUST_LOG")
                .ok()
                .and_then(|level| level.parse().ok())
                .unwrap_or(log::LevelFilter::Info),
        )
        .init();

    // Load configuration from environment variables
    let config = Config::load().expect(
        "Failed to load configuration. Please ensure all required environment variables are set.",
    );

    info!("Print Guardian starting...");
    info!("Using Moonraker API URL: {}", config.moonraker_api_url);
    info!("Monitoring camera: {}", config.image_url);

    // Initialize services
    let alert_service = AlertService::new(config.discord_webhook.clone());
    let printer_service = PrinterService::new(config.moonraker_api_url.clone());

    // Download model weights if needed
    FailureDetector::ensure_weights_downloaded(&config.weights, constants::MODEL_WEIGHTS_URL)?;

    // Initialize failure detector
    let mut detector = FailureDetector::new(
        config.model_cfg.clone(),
        config.weights.clone(),
        config.label_file.clone(),
        config.objectness_threshold,
        config.class_prob_threshold,
    )?;

    info!("Failure detector initialized");

    // Initialize image fetcher
    let mut image_fetcher = ImageFetcher::new(
        config.image_url.clone(),
        constants::MAX_RETRIES,
        constants::RETRY_DELAY_SECONDS,
    );

    info!("Image fetcher initialized with URL: {}", config.image_url);

    // Create output directory
    fs::create_dir_all(&config.output_dir)?;

    info!(
        "Output directory created at: {}",
        config.output_dir.display()
    );

    // Create .ready file to indicate the application is fully initialized
    fs::write(".ready", "ready")?;
    info!("Application ready - created .ready file for healthcheck");

    // Main monitoring loop state
    let mut print_failures = 0;
    let mut last_status_update = String::new();

    info!("Print Guardian initialized successfully. Starting monitoring loop...");

    loop {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        debug!("{}: Starting new monitoring iteration", timestamp);

        // Check printer status and send alert with image if status changed
        let res = printer_service.get_printer_status();
        match res {
            Ok(data) => {
                let state = data["result"]["status"]["print_stats"]["state"]
                    .as_str()
                    .unwrap_or("unknown");

                if last_status_update != state {
                    // send the status of the print to discord with the current processed image
                    if let Err(e) = alert_service.send_printer_status_alert(&data, None) {
                        error!("Failed to send printer status alert: {}", e);
                        thread::sleep(Duration::from_secs(constants::RETRY_DELAY_SECONDS));
                        continue;
                    } else {
                        info!("Printer status alert sent successfully with image");
                        last_status_update = state.to_string();
                    }
                }

                if state != "printing" {
                    warn!("Printer is not currently printing. Skipping detection.");
                    thread::sleep(Duration::from_secs(constants::RETRY_DELAY_SECONDS));
                    continue;
                }
            }
            Err(e) => {
                warn!("Failed to get printer status: {}", e);
                thread::sleep(Duration::from_secs(constants::RETRY_DELAY_SECONDS));
                continue;
            }
        }

        // Fetch image with retry logic first (we need it for both status updates and detection)
        let image_url = image_fetcher.get_image_url().to_string();
        let max_retries = image_fetcher.get_max_retries();

        let image_data = match image_fetcher.fetch_with_retry(|alert_type| match alert_type {
            AlertType::SystemOffline => {
                alert_service.send_system_offline_alert(&image_url, max_retries)
            }
            AlertType::SystemRecovery => alert_service.send_system_recovery_alert(),
        }) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to fetch image: {}", e);
                thread::sleep(Duration::from_secs(constants::RETRY_DELAY_SECONDS));
                continue;
            }
        };

        info!(
            "{}: Fetched image from {} successfully",
            timestamp, image_url
        );

        // Apply image transformations if configured
        let processed_image_data =
            match ImageFetcher::apply_image_transformations(&image_data, config.flip_image) {
                Ok(data) => data,
                Err(e) => {
                    error!(
                        "{}: Failed to apply image transformations: {}",
                        timestamp, e
                    );
                    image_data // Fall back to original image
                }
            };

        // Convert image bytes directly to darknet Image (no disk I/O)
        let darknet_image = match ImageFetcher::bytes_to_darknet_image(&processed_image_data) {
            Ok(image) => image,
            Err(e) => {
                error!("{}: Failed to parse image from memory: {}", timestamp, e);
                continue;
            }
        };

        // Run failure detection directly on the image
        let detections = match detector.detect_failures_from_image(&darknet_image) {
            Ok(detections) => detections,
            Err(e) => {
                error!("{}: Detection failed: {}", timestamp, e);
                continue;
            }
        };

        // get detection with max confidence
        let max_detection_prob = detections
            .iter()
            .map(|d| d.confidence_percent())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        info!(
            "{}: Detected {} failures with max confidence {:.2}%",
            timestamp,
            detections.len(),
            max_detection_prob
        );

        // Process detections
        let significant_detections: Vec<_> = detections
            .into_iter()
            .filter(|d| d.exceeds_threshold(constants::ALERT_PROBABILITY_THRESHOLD))
            .collect();

        if !significant_detections.is_empty() {
            // Log all significant detections
            for detection in &significant_detections {
                warn!(
                    "{}: Detected {} print failure with {:.2}% confidence at x: {:.1}, y: {:.1}, w: {:.1}, h: {:.1}",
                    timestamp,
                    detection.label,
                    detection.confidence_percent(),
                    detection.center_x(),
                    detection.center_y(),
                    detection.width(),
                    detection.height()
                );
            }

            // Annotate image with all detections
            let annotated_image = match ImageFetcher::annotate_image_with_detections(
                &processed_image_data,
                &significant_detections,
                darknet_image.width() as u32,
                darknet_image.height() as u32,
            ) {
                Ok(annotated) => Some(annotated),
                Err(e) => {
                    error!("{}: Failed to annotate image: {}", timestamp, e);
                    None
                }
            };

            // Process each detection for alerts and printer control
            for detection in significant_detections {
                print_failures += 1;

                // Check if we should pause the printer
                if print_failures > constants::PRINT_FAILURE_THRESHOLD {
                    match printer_service.pause_print() {
                        Ok(()) => {
                            let pause_image = annotated_image.as_deref();
                            if let Err(e) =
                                alert_service.send_print_pause_alert(print_failures, pause_image)
                            {
                                error!("Failed to send pause alert: {}", e);
                            } else {
                                info!(
                                    "Print paused due to multiple failures. Alert sent to Discord with image."
                                );
                            }
                            print_failures = 0; // Reset after pausing
                        }
                        Err(e) => {
                            error!("Failed to pause print: {}", e);
                        }
                    }
                }

                // Send failure alert with annotated image
                let image_data_ref = annotated_image.as_deref();
                if let Err(e) = alert_service.send_print_failure_alert(
                    &detection.label,
                    detection.confidence_percent(),
                    detection.center_x(),
                    detection.center_y(),
                    detection.width(),
                    detection.height(),
                    image_data_ref,
                ) {
                    error!("Failed to send Discord print failure alert: {}", e);
                } else {
                    info!("Sent Discord print failure alert with annotated image");
                }
            }
        } else {
            debug!("{}: No significant print failure detected.", timestamp);
        }

        // Small delay before next iteration
        thread::sleep(Duration::from_secs(1));
    }
}
