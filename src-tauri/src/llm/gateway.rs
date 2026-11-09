//! Enterprise LLM gateway support.
//!
//! Many organisations route every model call through an internal gateway that
//! speaks one of the public wire protocols (OpenAI chat completions, Anthropic
//! Messages or Gemini `generateContent`) but authenticates differently from the
//! public APIs: a custom auth header, extra routing/tenant headers, and keys
//! that are short-lived and minted by a local helper binary (SSO broker, `az`,
//! `gcloud`, a corporate token tool, ...).
//!
//! This module holds that configuration and resolves the key at request time.
//! The helper binary is executed directly (never through a shell), its stdout
//! is the key (optionally extracted from JSON), and the result is cached in
//! memory only for a bounded time. Keys from the helper are never written to
//! disk or logged.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

pub const GATEWAY_PROVIDER: &str = "gateway";

pub const PROTOCOL_OPENAI: &str = "openai";
pub const PROTOCOL_ANTHROPIC: &str = "anthropic";
pub const PROTOCOL_GEMINI: &str = "gemini";

pub const AUTH_STATIC: &str = "static";
pub const AUTH_COMMAND: &str = "command";
pub const AUTH_NONE: &str = "none";

const DEFAULT_KEY_CACHE_SECONDS: u32 = 300;
const MAX_KEY_CACHE_SECONDS: u32 = 24 * 60 * 60;
const DEFAULT_COMMAND_TIMEOUT_SECONDS: u32 = 15;
const MAX_COMMAND_TIMEOUT_SECONDS: u32 = 120;
const MAX_EXTRA_HEADERS: usize = 32;
const MAX_COMMAND_OUTPUT_BYTES: usize = 64 * 1024;

/// Headers the transport owns; letting users set them breaks framing.
const RESERVED_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "upgrade",
];

pub fn is_gateway_provider(provider: &str) -> bool {
    provider.trim().eq_ignore_ascii_case(GATEWAY_PROVIDER)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct GatewayHeader {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct LlmGatewayConfig {
    /// Wire protocol the gateway exposes: `openai`, `anthropic` or `gemini`.
    pub protocol: String,
    /// Where the key comes from: `static` (credential vault), `command`
    /// (external helper binary) or `none` (gateway authenticates otherwise,
    /// e.g. by network location or a static header below).
    pub auth_mode: String,
    /// Header that carries the key. Empty = protocol default
    /// (`Authorization` / `x-api-key` / `x-goog-api-key`).
    pub auth_header: String,
    /// Text placed before the key in `auth_header`. Only used when
    /// `auth_header` is set; empty means the raw key.
    pub auth_prefix: String,
    /// Absolute path (or PATH-resolvable name) of the key helper binary.
    /// `%VAR%` and `${VAR}` are expanded from the environment.
    pub key_command: String,
    /// Arguments for the helper, one per entry; no shell parsing is applied.
    pub key_command_args: Vec<String>,
    /// Optional dotted path (e.g. `accessToken` or `data.token`) when the
    /// helper prints JSON. Empty = the trimmed stdout is the key.
    pub key_command_json_field: String,
    /// How long a helper-issued key is reused. 0 = run the helper per request.
    pub key_cache_seconds: u32,
    /// Kill the helper if it has not exited after this many seconds.
    pub key_command_timeout_seconds: u32,
    /// Additional static headers sent with every gateway request.
    pub extra_headers: Vec<GatewayHeader>,
}

impl Default for LlmGatewayConfig {
    fn default() -> Self {
        Self {
            protocol: PROTOCOL_OPENAI.to_string(),
            auth_mode: AUTH_STATIC.to_string(),
            auth_header: String::new(),
            auth_prefix: String::new(),
            key_command: String::new(),
            key_command_args: Vec::new(),
            key_command_json_field: String::new(),
            key_cache_seconds: DEFAULT_KEY_CACHE_SECONDS,
            key_command_timeout_seconds: DEFAULT_COMMAND_TIMEOUT_SECONDS,
            extra_headers: Vec::new(),
        }
    }
}

impl LlmGatewayConfig {
    pub fn protocol(&self) -> &'static str {
        match self.protocol.trim().to_ascii_lowercase().as_str() {
            PROTOCOL_ANTHROPIC | "claude" => PROTOCOL_ANTHROPIC,
            PROTOCOL_GEMINI | "google" => PROTOCOL_GEMINI,
            _ => PROTOCOL_OPENAI,
        }
    }

    pub fn auth_mode(&self) -> &'static str {
        match self.auth_mode.trim().to_ascii_lowercase().as_str() {
            AUTH_COMMAND => AUTH_COMMAND,
            AUTH_NONE => AUTH_NONE,
            _ => AUTH_STATIC,
        }
    }

    pub fn uses_key_command(&self) -> bool {
        self.auth_mode() == AUTH_COMMAND
    }

    /// Header name and prefix used to send the key.
    pub fn auth_header_and_prefix(&self) -> (String, String) {
        let header = self.auth_header.trim();
        if !header.is_empty() {
            return (header.to_string(), self.auth_prefix.clone());
        }
        match self.protocol() {
            PROTOCOL_ANTHROPIC => ("x-api-key".to_string(), String::new()),
            PROTOCOL_GEMINI => ("x-goog-api-key".to_string(), String::new()),
            _ => ("Authorization".to_string(), "Bearer ".to_string()),
        }
    }

    pub fn normalize(&mut self) {
        self.protocol = self.protocol().to_string();
        self.auth_mode = self.auth_mode().to_string();
        self.auth_header = self.auth_header.trim().to_string();
        self.key_command = self.key_command.trim().to_string();
        self.key_command_json_field = self.key_command_json_field.trim().to_string();
        self.key_cache_seconds = self.key_cache_seconds.min(MAX_KEY_CACHE_SECONDS);
        if self.key_command_timeout_seconds == 0 {
            self.key_command_timeout_seconds = DEFAULT_COMMAND_TIMEOUT_SECONDS;
        }
        self.key_command_timeout_seconds = self
            .key_command_timeout_seconds
            .min(MAX_COMMAND_TIMEOUT_SECONDS);
        for header in &mut self.extra_headers {
            header.name = header.name.trim().to_string();
        }
        self.extra_headers
            .retain(|header| !header.name.is_empty() || !header.value.trim().is_empty());
    }

    /// Checks everything that can be checked without contacting the gateway.
    pub fn validate(&self) -> Result<(), String> {
        if self.uses_key_command() && self.key_command.trim().is_empty() {
            return Err("Gateway key command is empty.".to_string());
        }
        let (auth_header, auth_prefix) = self.auth_header_and_prefix();
        if self.auth_mode() != AUTH_NONE {
            parse_header_name(&auth_header)?;
            if auth_prefix.contains(['\r', '\n']) {
                return Err("Gateway auth prefix must not contain line breaks.".to_string());
            }
        }
        if self.extra_headers.len() > MAX_EXTRA_HEADERS {
            return Err(format!(
                "Too many gateway headers (maximum {MAX_EXTRA_HEADERS})."
            ));
        }
        for header in &self.extra_headers {
            parse_header_name(&header.name)?;
            HeaderValue::from_str(&header.value)
                .map_err(|_| format!("Gateway header '{}' has an invalid value.", header.name))?;
        }
        Ok(())
    }

    /// Builds the auth + extra headers for one request. Extra headers win over
    /// protocol defaults so a gateway can override e.g. `anthropic-version`.
    pub fn request_headers(&self, api_key: &str) -> Result<HeaderMap, String> {
        let mut headers = HeaderMap::new();
        if self.protocol() == PROTOCOL_ANTHROPIC {
            headers.insert(
                HeaderName::from_static("anthropic-version"),
                HeaderValue::from_static(super::protocol::ANTHROPIC_VERSION),
            );
        }
        let api_key = api_key.trim();
        if self.auth_mode() != AUTH_NONE && !api_key.is_empty() {
            let (name, prefix) = self.auth_header_and_prefix();
            let mut value = HeaderValue::from_str(&format!("{prefix}{api_key}"))
                .map_err(|_| "Gateway key contains characters not allowed in a header.")?;
            value.set_sensitive(true);
            headers.insert(parse_header_name(&name)?, value);
        }
        for header in &self.extra_headers {
            let value = HeaderValue::from_str(&header.value)
                .map_err(|_| format!("Gateway header '{}' has an invalid value.", header.name))?;
            headers.insert(parse_header_name(&header.name)?, value);
        }
        Ok(headers)
    }
}

fn parse_header_name(name: &str) -> Result<HeaderName, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("Gateway header name is empty.".to_string());
    }
    let parsed = HeaderName::from_bytes(trimmed.as_bytes())
        .map_err(|_| format!("Invalid gateway header name '{trimmed}'."))?;
    if RESERVED_HEADERS.contains(&parsed.as_str()) {
        return Err(format!("Gateway header '{trimmed}' cannot be overridden."));
    }
    Ok(parsed)
}

// ---------------------------------------------------------------------------
// Key resolution
// ---------------------------------------------------------------------------

struct CachedKey {
    value: String,
    expires_at: Instant,
}

static KEY_CACHE: LazyLock<Mutex<HashMap<u64, CachedKey>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
/// Serialises helper invocations so concurrent requests (pipeline pre-warm,
/// polish, Ask) do not spawn a storm of SSO prompts.
static COMMAND_LOCK: LazyLock<tokio::sync::Mutex<()>> =
    LazyLock::new(|| tokio::sync::Mutex::new(()));

fn command_cache_key(config: &LlmGatewayConfig) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.key_command.trim().hash(&mut hasher);
    config.key_command_args.hash(&mut hasher);
    config.key_command_json_field.trim().hash(&mut hasher);
    hasher.finish()
}

fn cached_key(cache_key: u64) -> Option<String> {
    let cache = KEY_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache
        .get(&cache_key)
        .filter(|entry| entry.expires_at > Instant::now())
        .map(|entry| entry.value.clone())
}

/// Drops the cached helper key, e.g. after the gateway answered 401/403.
pub fn invalidate_cached_key(config: &LlmGatewayConfig) {
    KEY_CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&command_cache_key(config));
}

/// Resolves the key to send to the gateway.
///
/// `static_key` is the vault-stored key and is only used in `static` mode.
/// `force_refresh` bypasses the cache (used by "Test" and after an auth
/// failure).
pub async fn resolve_api_key(
    config: &LlmGatewayConfig,
    static_key: &str,
    force_refresh: bool,
) -> Result<String, String> {
    config.validate()?;
    match config.auth_mode() {
        AUTH_NONE => Ok(String::new()),
        AUTH_COMMAND => {
            let cache_key = command_cache_key(config);
            if !force_refresh {
                if let Some(value) = cached_key(cache_key) {
                    return Ok(value);
                }
            }
            let _guard = COMMAND_LOCK.lock().await;
            // Another request may have refreshed the key while we waited.
            if !force_refresh {
                if let Some(value) = cached_key(cache_key) {
                    return Ok(value);
                }
            }
            let value = run_key_command(config).await?;
            if config.key_cache_seconds > 0 {
                KEY_CACHE.lock().unwrap_or_else(|e| e.into_inner()).insert(
                    cache_key,
                    CachedKey {
                        value: value.clone(),
                        expires_at: Instant::now()
                            + Duration::from_secs(u64::from(config.key_cache_seconds)),
                    },
                );
            }
            Ok(value)
        }
        _ => {
            let key = static_key.trim();
            if key.is_empty() {
                Err("Gateway API key is empty.".to_string())
            } else {
                Ok(key.to_string())
            }
        }
    }
}

/// Expands `%VAR%` (Windows style) and `${VAR}` from the process environment.
/// Unknown variables are left untouched so the error message shows them.
pub fn expand_env_vars(input: &str) -> String {
    expand_env_vars_with(input, |name| std::env::var(name).ok())
}

fn expand_env_vars_with(input: &str, lookup: impl Fn(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix("${") {
            if let Some(end) = after.find('}') {
                let name = &after[..end];
                if is_env_name(name) {
                    if let Some(value) = lookup(name) {
                        out.push_str(&value);
                        rest = &after[end + 1..];
                        continue;
                    }
                }
            }
        } else if let Some(after) = rest.strip_prefix('%') {
            if let Some(end) = after.find('%') {
                let name = &after[..end];
                if is_env_name(name) {
                    if let Some(value) = lookup(name) {
                        out.push_str(&value);
                        rest = &after[end + 1..];
                        continue;
                    }
                }
            }
        }
        let ch = rest.chars().next().unwrap();
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

fn is_env_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '(' || c == ')')
}

fn truncate_for_message(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    let mut out: String = trimmed.chars().take(max_chars).collect();
    if trimmed.chars().count() > max_chars {
        out.push('…');
    }
    out
}

/// Extracts the key from the helper's stdout.
pub fn extract_key_from_output(stdout: &str, json_field: &str) -> Result<String, String> {
    let stdout = stdout.trim_start_matches('\u{feff}').trim();
    if stdout.is_empty() {
        return Err("Gateway key command printed nothing on stdout.".to_string());
    }

    let json_field = json_field.trim();
    let key = if json_field.is_empty() {
        if stdout.contains(['\r', '\n']) {
            return Err(
                "Gateway key command printed multiple lines. Print only the key, or set a JSON field."
                    .to_string(),
            );
        }
        stdout.to_string()
    } else {
        let value: serde_json::Value = serde_json::from_str(stdout)
            .map_err(|_| "Gateway key command did not print valid JSON.".to_string())?;
        let mut current = &value;
        for segment in json_field.split('.').filter(|s| !s.is_empty()) {
            current = match current {
                serde_json::Value::Array(items) => segment
                    .parse::<usize>()
                    .ok()
                    .and_then(|index| items.get(index)),
                serde_json::Value::Object(map) => map.get(segment),
                _ => None,
            }
            .ok_or_else(|| format!("JSON field '{json_field}' not found in key command output."))?;
        }
        match current {
            serde_json::Value::String(s) => s.trim().to_string(),
            serde_json::Value::Number(n) => n.to_string(),
            _ => {
                return Err(format!(
                    "JSON field '{json_field}' in key command output is not a string."
                ))
            }
        }
    };

    if key.is_empty() {
        return Err("Gateway key command returned an empty key.".to_string());
    }
    if HeaderValue::from_str(&key).is_err() {
        return Err("Gateway key contains characters not allowed in a header.".to_string());
    }
    Ok(key)
}

async fn run_key_command(config: &LlmGatewayConfig) -> Result<String, String> {
    let program = expand_env_vars(config.key_command.trim());
    if program.is_empty() {
        return Err("Gateway key command is empty.".to_string());
    }
    let args: Vec<String> = config
        .key_command_args
        .iter()
        .map(|arg| expand_env_vars(arg))
        .collect();

    let mut command = tokio::process::Command::new(&program);
    command
        .args(&args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    #[cfg(windows)]
    {
        // CREATE_NO_WINDOW: do not flash a console window for console helpers.
        command.creation_flags(0x0800_0000);
    }

    let timeout = Duration::from_secs(u64::from(
        config
            .key_command_timeout_seconds
            .clamp(1, MAX_COMMAND_TIMEOUT_SECONDS),
    ));

    let child = command
        .spawn()
        .map_err(|error| format!("Failed to start gateway key command '{program}': {error}"))?;
    let output = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| {
            format!(
                "Gateway key command timed out after {}s.",
                timeout.as_secs()
            )
        })?
        .map_err(|error| format!("Gateway key command failed: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output
            .status
            .code()
            .map(|code| code.to_string())
            .unwrap_or_else(|| "signal".to_string());
        let details = truncate_for_message(&stderr, 200);
        tracing::warn!("Gateway key command exited with status {code}");
        return Err(if details.is_empty() {
            format!("Gateway key command exited with status {code}.")
        } else {
            format!("Gateway key command exited with status {code}: {details}")
        });
    }

    if output.stdout.len() > MAX_COMMAND_OUTPUT_BYTES {
        return Err("Gateway key command output is too large.".to_string());
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| "Gateway key command output is not valid UTF-8.".to_string())?;
    extract_key_from_output(&stdout, &config.key_command_json_field)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command_config(program: &str, args: &[&str]) -> LlmGatewayConfig {
        LlmGatewayConfig {
            auth_mode: AUTH_COMMAND.to_string(),
            key_command: program.to_string(),
            key_command_args: args.iter().map(|s| s.to_string()).collect(),
            ..LlmGatewayConfig::default()
        }
    }

    #[test]
    fn defaults_follow_protocol_conventions() {
        let mut config = LlmGatewayConfig::default();
        assert_eq!(
            config.auth_header_and_prefix(),
            ("Authorization".to_string(), "Bearer ".to_string())
        );
        config.protocol = "anthropic".to_string();
        assert_eq!(config.auth_header_and_prefix().0, "x-api-key");
        config.protocol = "gemini".to_string();
        assert_eq!(config.auth_header_and_prefix().0, "x-goog-api-key");
        config.protocol = "bogus".to_string();
        assert_eq!(config.protocol(), PROTOCOL_OPENAI);
    }

    #[test]
    fn request_headers_apply_custom_auth_and_extra_headers() {
        let config = LlmGatewayConfig {
            protocol: PROTOCOL_ANTHROPIC.to_string(),
            auth_header: "api-key".to_string(),
            auth_prefix: "Token ".to_string(),
            extra_headers: vec![
                GatewayHeader {
                    name: "X-Tenant".to_string(),
                    value: "finance".to_string(),
                },
                GatewayHeader {
                    name: "anthropic-version".to_string(),
                    value: "2024-01-01".to_string(),
                },
            ],
            ..LlmGatewayConfig::default()
        };
        let headers = config.request_headers("secret").unwrap();
        assert_eq!(headers["api-key"], "Token secret");
        assert_eq!(headers["x-tenant"], "finance");
        assert_eq!(headers["anthropic-version"], "2024-01-01");
        assert!(headers.get("x-api-key").is_none());
    }

    #[test]
    fn auth_mode_none_sends_no_key_header() {
        let config = LlmGatewayConfig {
            auth_mode: AUTH_NONE.to_string(),
            ..LlmGatewayConfig::default()
        };
        let headers = config.request_headers("ignored").unwrap();
        assert!(headers.get("authorization").is_none());
    }

    #[test]
    fn validate_rejects_bad_and_reserved_headers() {
        let mut config = LlmGatewayConfig {
            extra_headers: vec![GatewayHeader {
                name: "Bad Header".to_string(),
                value: "x".to_string(),
            }],
            ..LlmGatewayConfig::default()
        };
        assert!(config.validate().is_err());
        config.extra_headers[0].name = "Host".to_string();
        assert!(config.validate().is_err());
        config.extra_headers[0].name = "X-Ok".to_string();
        config.extra_headers[0].value = "line\nbreak".to_string();
        assert!(config.validate().is_err());
        config.extra_headers[0].value = "fine".to_string();
        assert!(config.validate().is_ok());

        let empty_command = command_config("  ", &[]);
        assert!(empty_command.validate().is_err());
    }

    #[test]
    fn normalize_clamps_and_drops_blank_headers() {
        let mut config = LlmGatewayConfig {
            protocol: " Gemini ".to_string(),
            auth_mode: "COMMAND".to_string(),
            key_cache_seconds: u32::MAX,
            key_command_timeout_seconds: 0,
            extra_headers: vec![GatewayHeader::default()],
            ..LlmGatewayConfig::default()
        };
        config.normalize();
        assert_eq!(config.protocol, PROTOCOL_GEMINI);
        assert_eq!(config.auth_mode, AUTH_COMMAND);
        assert_eq!(config.key_cache_seconds, MAX_KEY_CACHE_SECONDS);
        assert_eq!(
            config.key_command_timeout_seconds,
            DEFAULT_COMMAND_TIMEOUT_SECONDS
        );
        assert!(config.extra_headers.is_empty());
    }

    #[test]
    fn extract_key_handles_plain_and_json_output() {
        assert_eq!(
            extract_key_from_output("  tok-123\n", "").unwrap(),
            "tok-123"
        );
        assert!(extract_key_from_output("banner\ntok", "").is_err());
        assert!(extract_key_from_output("   ", "").is_err());
        let json = r#"{"accessToken":"abc","data":{"items":[{"token":"deep"}]}}"#;
        assert_eq!(extract_key_from_output(json, "accessToken").unwrap(), "abc");
        assert_eq!(
            extract_key_from_output(json, "data.items.0.token").unwrap(),
            "deep"
        );
        assert!(extract_key_from_output(json, "missing").is_err());
        assert!(extract_key_from_output("not json", "token").is_err());
    }

    #[test]
    fn env_expansion_supports_windows_and_shell_forms() {
        let lookup = |name: &str| match name {
            "LOCALAPPDATA" => Some(r"C:\Users\me\AppData\Local".to_string()),
            "HOME" => Some("/home/me".to_string()),
            _ => None,
        };
        assert_eq!(
            expand_env_vars_with(r"%LOCALAPPDATA%\corp\token.exe", lookup),
            r"C:\Users\me\AppData\Local\corp\token.exe"
        );
        assert_eq!(
            expand_env_vars_with("${HOME}/bin/token", lookup),
            "/home/me/bin/token"
        );
        assert_eq!(
            expand_env_vars_with("100% %UNKNOWN%", lookup),
            "100% %UNKNOWN%"
        );
    }

    #[tokio::test]
    async fn static_mode_requires_a_key() {
        let config = LlmGatewayConfig::default();
        assert!(resolve_api_key(&config, "  ", false).await.is_err());
        assert_eq!(resolve_api_key(&config, " k ", false).await.unwrap(), "k");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn command_mode_runs_helper_and_caches_result() {
        let dir = std::env::temp_dir().join(format!(
            "otl-gateway-test-{}-{}",
            std::process::id(),
            command_cache_key(&command_config("x", &["y"]))
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let counter = dir.join("count");
        let _ = std::fs::remove_file(&counter);
        // Each run appends a line, so the key changes on every invocation.
        let script = format!(
            "echo x >> '{}'; printf 'key-%s' \"$(wc -l < '{}' | tr -d ' ')\"",
            counter.display(),
            counter.display()
        );
        let config = command_config("/bin/sh", &["-c", &script]);

        let first = resolve_api_key(&config, "", false).await.unwrap();
        assert_eq!(first, "key-1");
        let cached = resolve_api_key(&config, "", false).await.unwrap();
        assert_eq!(cached, "key-1");
        let refreshed = resolve_api_key(&config, "", true).await.unwrap();
        assert_eq!(refreshed, "key-2");
        invalidate_cached_key(&config);
        let after_invalidate = resolve_api_key(&config, "", false).await.unwrap();
        assert_eq!(after_invalidate, "key-3");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn command_mode_reports_failures_and_timeouts() {
        let failing = command_config("/bin/sh", &["-c", "echo denied >&2; exit 3"]);
        let error = resolve_api_key(&failing, "", true).await.unwrap_err();
        assert!(error.contains("status 3"), "{error}");
        assert!(error.contains("denied"), "{error}");

        let mut slow = command_config("/bin/sh", &["-c", "sleep 5; echo late"]);
        slow.key_command_timeout_seconds = 1;
        let error = resolve_api_key(&slow, "", true).await.unwrap_err();
        assert!(error.contains("timed out"), "{error}");

        let missing = command_config("/definitely/not/a/binary", &[]);
        let error = resolve_api_key(&missing, "", true).await.unwrap_err();
        assert!(error.contains("Failed to start"), "{error}");
    }
}
