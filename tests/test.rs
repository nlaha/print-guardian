use print_guardian::ImageFetcher;

#[test]
fn test_image_fetcher_basic_methods() {
    let urls = vec!["https://fastly.picsum.photos/id/866/200/300.jpg?hmac=rcadCENKh4rD6MAp6V_ma-AyWv641M4iiOpe1RyFHeI".to_string()];
    let mut fetcher = ImageFetcher::new(urls.clone(), 3, 1);
    assert_eq!(fetcher.get_image_urls(), &urls);
    assert_eq!(fetcher.get_max_retries(), 3);
    assert_eq!(fetcher.get_retry_count(), 0);
    fetcher.reset_state();
    assert_eq!(fetcher.get_retry_count(), 0);
    assert!(!fetcher.is_disconnect_alert_sent());
}

#[test]
fn test_image_fetcher_transformations() {
    // Use a small valid JPEG image for this test (or skip if not available)
    let image_bytes = std::fs::read("tests/test_failure.jpg").expect("Test image missing");
    let flipped =
        ImageFetcher::apply_image_transformations(&image_bytes, true).expect("Flip failed");
    assert!(!flipped.is_empty());
    let not_flipped =
        ImageFetcher::apply_image_transformations(&image_bytes, false).expect("No flip failed");
    assert_eq!(not_flipped, image_bytes);
}

#[test]
fn test_printer_service_url_and_mock() {
    // This test only checks struct creation and URL formatting, not real HTTP
    let url = "http://localhost:7125".to_string();
    let printer = print_guardian::PrinterService::new(url.clone());
    // The struct should store the URL
    let status_url = format!("{}/printer/objects/query?webhooks&print_stats", url);
    // We can't call get_printer_status() without a real server, but we can check the type
    assert_eq!(printer.api_url, url);
    // Optionally, test that the pause/resume/cancel methods return errors if the server is unreachable
    let pause_result = printer.pause_print();
    assert!(pause_result.is_err());
    let resume_result = printer.resume_print();
    assert!(resume_result.is_err());
    let cancel_result = printer.cancel_print();
    assert!(cancel_result.is_err());
}
