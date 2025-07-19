#![allow(dead_code)]
use anyhow::Result;

/// Printer control service for interacting with Moonraker API.
///
/// This service provides methods to control 3D printer operations through
/// the Moonraker API, such as pausing prints when failures are detected.
pub struct PrinterService {
    pub api_url: String,
}

impl PrinterService {
    /// Create a new PrinterService with the provided Moonraker API URL.
    ///
    /// # Arguments
    ///
    /// * `api_url` - Base URL for the Moonraker API (e.g., "http://printer.local:7125")
    ///
    pub fn new(api_url: String) -> Self {
        Self { api_url }
    }

    /// Pause the current print job.
    ///
    /// Sends a pause command to the Moonraker API to immediately pause
    /// the current print job. This is typically called when multiple
    /// print failures are detected.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The Moonraker API returns an error status
    /// - The printer is not currently printing
    ///
    pub fn pause_print(&self) -> Result<()> {
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(format!("{}/printer/print/pause", self.api_url))
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to pause print: HTTP {}",
                response.status()
            ));
        }

        Ok(())
    }

    /// Resume the current print job.
    ///
    /// Sends a resume command to the Moonraker API to resume a paused print job.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The Moonraker API returns an error status
    /// - The printer is not currently paused
    pub fn resume_print(&self) -> Result<()> {
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(format!("{}/printer/print/resume", self.api_url))
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to resume print: HTTP {}",
                response.status()
            ));
        }

        Ok(())
    }

    /// Cancel the current print job.
    ///
    /// Sends a cancel command to the Moonraker API to completely cancel
    /// the current print job.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The Moonraker API returns an error status
    /// - There is no active print job to cancel
    pub fn cancel_print(&self) -> Result<()> {
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(format!("{}/printer/print/cancel", self.api_url))
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to cancel print: HTTP {}",
                response.status()
            ));
        }

        Ok(())
    }

    /// Get the current printer status.
    ///
    /// Retrieves the current status of the printer including print state,
    /// temperatures, and other relevant information.
    ///
    /// # Returns
    ///
    /// Returns a JSON value containing the printer status information.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The HTTP request fails
    /// - The Moonraker API returns an error status
    /// - JSON parsing fails
    pub fn get_printer_status(&self) -> Result<serde_json::Value> {
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(format!(
                "{}/printer/objects/query?webhooks&print_stats",
                self.api_url
            ))
            .send()?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to get printer status: HTTP {}",
                response.status()
            ));
        }

        let status: serde_json::Value = response.json()?;
        Ok(status)
    }
}
