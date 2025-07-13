use anyhow::Result;
use darknet::{BBox, Image, Network};
use log::{debug, info};
use std::{fs, path::PathBuf};

/// Print failure detection service using YOLO/Darknet neural networks.
///
/// This service handles loading and running object detection models to identify
/// print failures in real-time camera images. It can detect various types of
/// print failures such as spaghetti, layer shifts, warping, and other issues.
pub struct FailureDetector {
    network: Network,
    labels: Vec<String>,
    objectness_threshold: f32,
    class_prob_threshold: f32,
}

impl FailureDetector {
    /// Create a new FailureDetector with the specified model and configuration.
    ///
    /// # Arguments
    ///
    /// * `model_cfg` - Path to the YOLO/Darknet configuration file
    /// * `weights_path` - Path to the trained model weights file
    /// * `labels_path` - Path to the file containing class labels
    /// * `objectness_threshold` - Minimum objectness score for detections
    /// * `class_prob_threshold` - Minimum class probability for valid detections
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model files cannot be loaded
    /// - Labels file cannot be read
    /// - Model configuration is invalid
    ///
    /// # Examples
    ///
    /// ```rust
    /// let detector = FailureDetector::new(
    ///     PathBuf::from("model.cfg"),
    ///     PathBuf::from("weights.darknet"),
    ///     PathBuf::from("labels.txt"),
    ///     0.5,
    ///     0.5
    /// )?;
    /// ```
    pub fn new(
        model_cfg: PathBuf,
        weights_path: PathBuf,
        labels_path: PathBuf,
        objectness_threshold: f32,
        class_prob_threshold: f32,
    ) -> Result<Self> {
        // Load class labels
        let labels = fs::read_to_string(labels_path)?
            .lines()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        // Load the neural network
        let network = Network::load(model_cfg, Some(weights_path), false)?;

        Ok(Self {
            network,
            labels,
            objectness_threshold,
            class_prob_threshold,
        })
    }

    /// Download model weights if they don't exist locally.
    ///
    /// Automatically downloads the trained model weights from a remote URL
    /// if the weights file doesn't exist on the local filesystem.
    ///
    /// # Arguments
    ///
    /// * `weights_path` - Local path where weights should be stored
    /// * `download_url` - URL to download weights from
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Download fails
    /// - File cannot be written to disk
    /// - Remote server returns error status
    pub fn ensure_weights_downloaded(weights_path: &PathBuf, download_url: &str) -> Result<()> {
        if !weights_path.exists() {
            info!(
                "Model weights not found, downloading from: {}",
                download_url
            );

            let response = reqwest::blocking::get(download_url)?;
            if !response.status().is_success() {
                return Err(anyhow::anyhow!(
                    "Failed to download model weights from {}: {}",
                    download_url,
                    response.status()
                ));
            }

            let model_weights_data = response.bytes()?;
            fs::write(weights_path, model_weights_data)?;

            info!("Model weights downloaded successfully");
        }

        Ok(())
    }

    /// Detect print failures in the provided image.
    ///
    /// Runs object detection on the input image and returns a list of detected
    /// print failures with their confidence scores and bounding box locations.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image file to analyze
    ///
    /// # Returns
    ///
    /// Returns a vector of `Detection` objects containing information about
    /// each detected print failure.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image file cannot be loaded
    /// - Neural network inference fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// let detections = detector.detect_failures(&PathBuf::from("image.jpg"))?;
    /// for detection in detections {
    ///     info!("Found {}: {:.2}% confidence", detection.label, detection.confidence * 100.0);
    /// }
    /// ```
    pub fn detect_failures(&mut self, image_path: &PathBuf) -> Result<Vec<Detection>> {
        let image = Image::open(image_path)?;

        // Run object detection with NMS parameters
        let detections = self.network.predict(&image, 0.25, 0.5, 0.45, true);

        debug!("Raw detections count: {}", detections.len());

        let mut results = Vec::new();
        let mut filtered_by_objectness = 0;
        let mut filtered_by_class_prob = 0;

        // Process detections and filter by thresholds
        for det in detections.iter() {
            debug!(
                "Detection objectness: {:.4} (threshold: {:.4})",
                det.objectness(),
                self.objectness_threshold
            );

            if det.objectness() > self.objectness_threshold {
                if let Some((class_index, prob)) = det.best_class(Some(self.class_prob_threshold)) {
                    let bbox = *det.bbox();
                    let label = self
                        .labels
                        .get(class_index)
                        .unwrap_or(&"unknown".to_string())
                        .clone();

                    debug!(
                        "Valid detection - Class: {} (index: {}), Probability: {:.4} (threshold: {:.4})",
                        label, class_index, prob, self.class_prob_threshold
                    );

                    results.push(Detection {
                        label,
                        confidence: prob,
                        bbox,
                    });
                } else {
                    filtered_by_class_prob += 1;
                }
            } else {
                filtered_by_objectness += 1;
            }
        }

        debug!(
            "Filtered by objectness threshold: {}",
            filtered_by_objectness
        );
        debug!(
            "Filtered by class probability threshold: {}",
            filtered_by_class_prob
        );
        debug!("Final valid detections: {}", results.len());

        Ok(results)
    }

    /// Get the list of class labels.
    pub fn get_labels(&self) -> &[String] {
        &self.labels
    }

    /// Get the current objectness threshold.
    pub fn get_objectness_threshold(&self) -> f32 {
        self.objectness_threshold
    }

    /// Get the current class probability threshold.
    pub fn get_class_prob_threshold(&self) -> f32 {
        self.class_prob_threshold
    }

    /// Update the objectness threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - New objectness threshold (0.0 to 1.0)
    pub fn set_objectness_threshold(&mut self, threshold: f32) {
        self.objectness_threshold = threshold.clamp(0.0, 1.0);
    }

    /// Update the class probability threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - New class probability threshold (0.0 to 1.0)
    pub fn set_class_prob_threshold(&mut self, threshold: f32) {
        self.class_prob_threshold = threshold.clamp(0.0, 1.0);
    }
}

/// Represents a single print failure detection result.
///
/// Contains all the information about a detected print failure, including
/// the type of failure, confidence level, and location in the image.
#[derive(Debug, Clone)]
pub struct Detection {
    /// The type of print failure detected (e.g., "spaghetti", "layer_shift").
    pub label: String,

    /// Confidence score from 0.0 to 1.0.
    pub confidence: f32,

    /// Bounding box coordinates and dimensions.
    pub bbox: BBox,
}

impl Detection {
    /// Get the bounding box center X coordinate.
    pub fn center_x(&self) -> f32 {
        self.bbox.x
    }

    /// Get the bounding box center Y coordinate.
    pub fn center_y(&self) -> f32 {
        self.bbox.y
    }

    /// Get the bounding box width.
    pub fn width(&self) -> f32 {
        self.bbox.w
    }

    /// Get the bounding box height.
    pub fn height(&self) -> f32 {
        self.bbox.h
    }

    /// Get the confidence as a percentage.
    pub fn confidence_percent(&self) -> f32 {
        self.confidence * 100.0
    }

    /// Check if this detection exceeds the specified confidence threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Confidence threshold to check against (0.0 to 1.0)
    pub fn exceeds_threshold(&self, threshold: f32) -> bool {
        self.confidence > threshold
    }
}
