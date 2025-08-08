use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use futures::AsyncReadExt;
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use serde::{Deserialize, Serialize};

const ANTHROPIC_OAUTH_TOKEN_URL: &str = "https://console.anthropic.com/v1/oauth/token";
const CLAUDE_CODE_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

#[derive(Debug, Serialize)]
struct RefreshTokenRequest<'a> {
    grant_type: &'a str,
    refresh_token: &'a str,
    client_id: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    #[allow(dead_code)]
    pub token_type: String,
    pub expires_in: i64,
    pub refresh_token: String,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: String,
    error_description: Option<String>,
}

/// Refreshes an access token using a refresh token.
pub async fn refresh_access_token(
    client: &dyn HttpClient,
    refresh_token: &str,
) -> Result<TokenResponse> {
    let request_body = RefreshTokenRequest {
        grant_type: "refresh_token",
        refresh_token,
        client_id: CLAUDE_CODE_CLIENT_ID,
    };

    let serialized_request = serde_json::to_string(&request_body)
        .context("failed to serialize refresh token request")?;

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(ANTHROPIC_OAUTH_TOKEN_URL)
        .header("Content-Type", "application/json")
        .body(AsyncBody::from(serialized_request))
        .context("failed to build refresh token request")?;

    let mut response = client
        .send(request)
        .await
        .context("failed to send refresh token request")?;

    let status = response.status();
    let mut body = String::new();
    response
        .body_mut()
        .read_to_string(&mut body)
        .await
        .context("failed to read response body")?;

    if status.is_success() {
        serde_json::from_str::<TokenResponse>(&body).context("failed to parse token response")
    } else {
        // Try to parse as error response first
        if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&body) {
            anyhow::bail!(
                "Failed to refresh token: {} - {}",
                error_response.error,
                error_response.error_description.unwrap_or_default()
            );
        } else {
            anyhow::bail!("Failed to refresh token: HTTP {} - {}", status, body);
        }
    }
}

/// Determines if an access token is expired based on its expiration time.
pub fn is_token_expired(expires_at: DateTime<Utc>) -> bool {
    let now = Utc::now();
    // Add a small buffer (5 minutes) to refresh slightly before expiration
    let buffer = chrono::Duration::minutes(5);
    expires_at - buffer <= now
}

/// Calculates the expiration time for a token given its lifetime in seconds.
pub fn calculate_token_expiration(expires_in_seconds: i64) -> DateTime<Utc> {
    Utc::now() + chrono::Duration::seconds(expires_in_seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_token_expired() {
        // Token that expired 10 minutes ago
        let expired = Utc::now() - chrono::Duration::minutes(10);
        assert!(is_token_expired(expired));

        // Token that expires in 4 minutes (within buffer)
        let expiring_soon = Utc::now() + chrono::Duration::minutes(4);
        assert!(is_token_expired(expiring_soon));

        // Token that expires in 10 minutes (outside buffer)
        let valid = Utc::now() + chrono::Duration::minutes(10);
        assert!(!is_token_expired(valid));
    }

    #[test]
    fn test_calculate_token_expiration() {
        let expires_in = 3600; // 1 hour
        let expiration = calculate_token_expiration(expires_in);
        let expected = Utc::now() + chrono::Duration::seconds(expires_in);

        // Allow for small time difference due to execution time
        let diff = (expiration - expected).num_seconds().abs();
        assert!(diff < 2);
    }
}
