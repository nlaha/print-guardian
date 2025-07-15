use anyhow::Result;
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
    /// # Examples
    ///
    /// ```rust
    /// let alert_service = AlertService::new("https://discord.com/api/webhooks/...".to_string());
    /// ```
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
    /// # Examples
    ///
    /// ```rust
    /// alert_service.send_alert(
    ///     "Print Failure Detected",
    ///     "Detected spaghetti with 85% confidence",
    ///     0xFFA500, // Orange
    ///     "⚠️"
    /// )?;
    /// ```
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
    /// consistent formatting and color scheme.
    ///
    /// # Arguments
    ///
    /// * `label` - The type of failure detected (e.g., "spaghetti", "layer_shift")
    /// * `confidence` - Confidence percentage (0.0 to 100.0)
    /// * `x`, `y`, `w`, `h` - Bounding box coordinates and dimensions
    ///
    /// # Examples
    ///
    /// ```rust
    /// alert_service.send_print_failure_alert("spaghetti", 85.5, 120.0, 80.0, 50.0, 30.0)?;
    /// ```
    pub fn send_print_failure_alert(
        &self,
        label: &str,
        confidence: f32,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
    ) -> Result<()> {
        let description = format!(
            "Detected **{}** print failure with **{:.2}%** confidence\n\n**Location:**\n• X: {:.1}\n• Y: {:.1}\n• Width: {:.1}\n• Height: {:.1}",
            label, confidence, x, y, w, h
        );

        self.send_alert(
            "Print Failure Detected",
            &description,
            0xFFA500, // Orange color
            "⚠️",
        )
    }

    pub fn send_printer_status_alert(&self, status: &serde_json::Value) -> Result<()> {
        let stats = &status["result"]["status"]["print_stats"];

        let description = format!(
            "
            Current file: **{}**
            Current printer state: **{}**
            **Print Stats:**
            • Filament Used: {}mm
            • Print Duration: {}
            ",
            stats["file_name"].as_str().unwrap_or("Unknown"),
            stats["state"].as_str().unwrap_or("unknown"),
            stats["filament_used"].as_f64().unwrap_or(0.0),
            // convert seconds to a human-readable format with hours and minutes
            format!(
                "{}h {}m {}s",
                stats["print_duration"].as_u64().unwrap_or(0) / 3600,
                (stats["print_duration"].as_u64().unwrap_or(0) % 3600) / 60,
                stats["print_duration"].as_u64().unwrap_or(0) % 60
            )
        );

        self.send_alert(
            status["result"]["status"]["webhooks"]["state_message"]
                .as_str()
                .unwrap_or("Printer Status Update"),
            &description,
            0x0099FF, // Blue color
            "ℹ️",
        )
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
    pub fn send_print_pause_alert(&self, failure_count: u32) -> Result<()> {
        let description = format!(
            "Print has been paused after detecting {} print failures. Please check the printer.",
            failure_count
        );

        self.send_alert(
            "Print Paused Due to Multiple Failures",
            &description,
            0xFF0000, // Red color
            "🚨",
        )
    }
}

/// Alert color constants for consistent theming.
pub mod colors {
    /// Red color for critical alerts and errors.
    pub const RED: u32 = 0xFF0000;

    /// Orange color for warnings and print failures.
    pub const ORANGE: u32 = 0xFFA500;

    /// Green color for recovery and success messages.
    pub const GREEN: u32 = 0x00FF00;

    /// Blue color for informational messages.
    pub const BLUE: u32 = 0x0099FF;
}
