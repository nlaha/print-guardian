//! Print Guardian - AI-powered 3D print failure detection system.
//! 
//! This library provides components for monitoring 3D printer cameras in real-time
//! using machine learning to detect print failures and automatically respond to them.
//! 
//! # Core Components
//! 
//! * [`config`] - Configuration management and command-line arguments
//! * [`alerts`] - Discord webhook notification system
//! * [`printer`] - Moonraker API integration for printer control
//! * [`image_fetcher`] - Robust image downloading with retry logic
//! * [`detector`] - AI-powered print failure detection using YOLO/Darknet
//! * [`error`] - Comprehensive error types and handling
//! 
//! # Quick Start
//! 
//! ```rust
//! use print_guardian::*;
//! 
//! // Set up configuration
//! let config = config::Config::from_env();
//! let env_config = config::EnvConfig::load()?;
//! 
//! // Initialize services
//! let alert_service = alerts::AlertService::new(env_config.discord_webhook);
//! let printer_service = printer::PrinterService::new(env_config.moonraker_api_url);
//! 
//! // Start monitoring...
//! ```

pub mod config;
pub mod alerts;
pub mod printer;
pub mod image_fetcher;
pub mod detector;
pub mod error;

// Re-export commonly used types for convenience
pub use config::{Config, EnvConfig};
pub use alerts::AlertService;
pub use printer::PrinterService;
pub use image_fetcher::ImageFetcher;
pub use detector::{FailureDetector, Detection};
pub use error::PrintGuardianError;
