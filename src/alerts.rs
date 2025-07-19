use anyhow::Result;
use reqwest::blocking::multipart;
use serde_json::json;

/// Discord alert service for sending rich embed notifications.
///
/// This module handles sending formatted Discord alerts about print failures,
/// system status changes, and other important events in the print monitoring system.
pub struct AlertService {
    webhook_url: String,
}

impl AlertService {
    /// Create a new AlertService with the provided Discord webhook URL.
    ///
    /// # Arguments
    ///
    /// * `webhook_url` - A valid Discord webhook URL
    ///
    pub fn new(webhook_url: String) -> Self {
        Self { webhook_url }
    }

    /// Send a Discord alert with rich embed formatting.
    ///
    /// Creates a rich embed message with the specified title, description, color,
    /// and emoji, then sends it to the configured Discord webhook.
    ///
    /// # Arguments
    ///
    /// * `title` - The title of the alert
    /// * `description` - The main content of the alert (supports Markdown)
    /// * `color` - The color of the embed sidebar (as a hex value, e.g., 0xFF0000 for red)
    /// * `emoji` - An emoji to display with the title
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The Discord API returns an error status
    /// - JSON serialization fails
    ///
    pub fn send_alert(
        &self,
        title: &str,
        description: &str,
        color: u32,
        emoji: &str,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now().to_rfc3339();

        let embed = json!({
            "embeds": [{
                "title": format!("{} {}", emoji, title),
                "description": description,
                "color": color,
                "timestamp": timestamp,
                "footer": {
                    "text": "Print Guardian"
                }
            }]
        });

        let client = reqwest::blocking::Client::new();
        let response = client
            .post(&self.webhook_url)
            .header("Content-Type", "application/json")
            .json(&embed)
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to send Discord alert: HTTP {}",
                response.status()
            ));
        }

        Ok(())
    }

    /// Send a print failure alert with standardized formatting.
    ///
    /// Convenience method for sending print failure notifications with
    /// consistent formatting and color scheme. Optionally includes an
    /// annotated image showing the detected failure area.
    ///
    /// # Arguments
    ///
    /// * `label` - The type of failure detected (e.g., "spaghetti", "layer_shift")
    /// * `confidence` - Confidence percentage (0.0 to 100.0)
    /// * `x`, `y`, `w`, `h` - Bounding box coordinates and dimensions
    /// * `annotated_image` - Optional annotated image data (JPEG format)
    ///
    pub fn send_print_failure_alert(
        &self,
        label: &str,
        confidence: f32,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        annotated_image: Option<&[u8]>,
    ) -> Result<()> {
        let description = format!(
            "Detected **{}** print failure with **{:.2}%** confidence\n\n**Location:**\n• X: {:.1}\n• Y: {:.1}\n• Width: {:.1}\n• Height: {:.1}",
            label, confidence, x, y, w, h
        );

        // If we have an annotated image, send it with the alert
        if let Some(image_data) = annotated_image {
            let filename = format!("failure_detection_{}.jpg", chrono::Utc::now().timestamp());
            self.send_alert_with_image(
                "Print Failure Detected",
                &description,
                0xFFA500, // Orange color
                "⚠️",
                image_data,
                &filename,
            )
        } else {
            // Send regular alert without image
            self.send_alert(
                "Print Failure Detected",
                &description,
                0xFFA500, // Orange color
                "⚠️",
            )
        }
    }

    pub fn send_printer_status_alert(
        &self,
        status: &serde_json::Value,
        image_data: Option<&[u8]>,
    ) -> Result<()> {
        let stats = &status["result"]["status"]["print_stats"];

        let description = format!(
            "
            Current file: **{}**
            Current printer state: **{}**
            **Print Stats:**
            • Filament Used: {:.2}m
            • Print Duration: {}
            ",
            stats["filename"].as_str().unwrap_or("Unknown"),
            stats["state"].as_str().unwrap_or("unknown"),
            stats["filament_used"].as_f64().unwrap_or(0.0) / 1000.0, // convert mm to meters
            // convert seconds to a human-readable format with hours and minutes
            format!(
                "{}h {}m {}s",
                f64::floor(stats["print_duration"].as_f64().unwrap_or(0.0) / 3600.0),
                f64::floor(stats["print_duration"].as_f64().unwrap_or(0.0) % 3600.0 / 60.0),
                f64::floor(stats["print_duration"].as_f64().unwrap_or(0.0) % 60.0)
            )
        );

        let title = status["result"]["status"]["webhooks"]["state_message"]
            .as_str()
            .unwrap_or("Printer Status Update");

        // If we have an image, send it with the alert
        if let Some(image_bytes) = image_data {
            let filename = format!("printer_status_{}.jpg", chrono::Utc::now().timestamp());
            self.send_alert_with_image(
                title,
                &description,
                0x0099FF, // Blue color
                "ℹ️",
                image_bytes,
                &filename,
            )
        } else {
            // Send regular alert without image
            self.send_alert(
                title,
                &description,
                0x0099FF, // Blue color
                "ℹ️",
            )
        }
    }

    /// Send a critical system offline alert.
    ///
    /// Used when the image fetching system fails repeatedly and monitoring
    /// is temporarily offline.
    ///
    /// # Arguments
    ///
    /// * `image_url` - The URL that failed to respond
    /// * `max_retries` - Number of retry attempts that were made
    pub fn send_system_offline_alert(&self, image_url: &str, max_retries: u32) -> Result<()> {
        let description = format!(
            "Failed to fetch image from {} after {} attempts. Print monitoring is offline!",
            image_url, max_retries
        );

        self.send_alert(
            "CRITICAL: Print Monitoring Offline",
            &description,
            0xFF0000, // Red color
            "🚨",
        )
    }

    /// Send a system recovery alert.
    ///
    /// Used when the system comes back online after a period of being disconnected.
    pub fn send_system_recovery_alert(&self) -> Result<()> {
        self.send_alert(
            "RECOVERY: Print Monitoring Back Online",
            "Image fetch successful after connection issues.",
            0x00FF00, // Green color
            "✅",
        )
    }

    /// Send a print pause alert.
    ///
    /// Used when the system automatically pauses the printer due to multiple
    /// detected failures.
    ///
    /// # Arguments
    ///
    /// * `failure_count` - Number of failures that triggered the pause
    /// * `annotated_image` - Optional annotated image showing the failures
    pub fn send_print_pause_alert(
        &self,
        failure_count: u32,
        annotated_image: Option<&[u8]>,
    ) -> Result<()> {
        let description = format!(
            "Print has been paused after detecting {} print failures. Please check the printer.",
            failure_count
        );

        // If we have an annotated image, send it with the alert
        if let Some(image_data) = annotated_image {
            let filename = format!("print_pause_{}.jpg", chrono::Utc::now().timestamp());
            self.send_alert_with_image(
                "Print Paused Due to Multiple Failures",
                &description,
                0xFF0000, // Red color
                "🚨",
                image_data,
                &filename,
            )
        } else {
            // Send regular alert without image
            self.send_alert(
                "Print Paused Due to Multiple Failures",
                &description,
                0xFF0000, // Red color
                "🚨",
            )
        }
    }

    /// Send a Discord alert with an attached image.
    ///
    /// Creates a rich embed message with an attached image file.
    ///
    /// # Arguments
    ///
    /// * `title` - The title of the alert
    /// * `description` - The main content of the alert (supports Markdown)
    /// * `color` - The color of the embed sidebar (as a hex value)
    /// * `emoji` - An emoji to display with the title
    /// * `image_data` - JPEG image data to attach
    /// * `filename` - Name for the attached image file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The Discord API returns an error status
    /// - JSON serialization fails
    pub fn send_alert_with_image(
        &self,
        title: &str,
        description: &str,
        color: u32,
        emoji: &str,
        image_data: &[u8],
        filename: &str,
    ) -> Result<()> {
        let timestamp = chrono::Utc::now().to_rfc3339();

        let embed = json!({
            "embeds": [{
                "title": format!("{} {}", emoji, title),
                "description": description,
                "color": color,
                "timestamp": timestamp,
                "footer": {
                    "text": "Print Guardian"
                },
                "image": {
                    "url": format!("attachment://{}", filename)
                }
            }]
        });

        let client = reqwest::blocking::Client::new();

        // Create multipart form with JSON payload and image
        let form = multipart::Form::new()
            .text("payload_json", embed.to_string())
            .part(
                "files[0]",
                multipart::Part::bytes(image_data.to_vec())
                    .file_name(filename.to_string())
                    .mime_str("image/jpeg")?,
            );

        let response = client.post(&self.webhook_url).multipart(form).send()?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to send Discord alert with image: HTTP {}",
                response.status()
            ));
        }

        Ok(())
    }
}
