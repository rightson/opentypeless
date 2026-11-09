use reqwest::RequestBuilder;
use serde_json::{json, Value};
use std::time::Duration;

const ANTHROPIC_API_HOST: &str = "api.anthropic.com";
const OPENAI_API_HOST: &str = "api.openai.com";
pub(crate) const ANTHROPIC_VERSION: &str = "2023-06-01";

use super::gateway::{self, LlmGatewayConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmApiKind {
    OpenAiCompatible,
    AnthropicMessages,
    GeminiGenerateContent,
}

/// Everything needed to talk to a configured LLM endpoint. `gateway` is only
/// set when the provider is the enterprise gateway; all other providers keep
/// their host-based protocol detection.
#[derive(Debug, Clone, Copy)]
pub struct LlmTarget<'a> {
    pub provider: &'a str,
    pub base_url: &'a str,
    pub gateway: Option<&'a LlmGatewayConfig>,
}

impl<'a> LlmTarget<'a> {
    pub fn new(provider: &'a str, base_url: &'a str, gateway: &'a LlmGatewayConfig) -> Self {
        Self {
            provider,
            base_url,
            gateway: gateway::is_gateway_provider(provider).then_some(gateway),
        }
    }

    pub fn plain(provider: &'a str, base_url: &'a str) -> Self {
        Self {
            provider,
            base_url,
            gateway: None,
        }
    }

    pub fn from_config(config: &'a crate::storage::AppConfig) -> Self {
        Self::new(
            &config.llm_provider,
            &config.llm_base_url,
            &config.llm_gateway,
        )
    }

    pub fn api_kind(&self) -> LlmApiKind {
        match self.gateway {
            Some(gateway) => match gateway.protocol() {
                gateway::PROTOCOL_ANTHROPIC => LlmApiKind::AnthropicMessages,
                gateway::PROTOCOL_GEMINI => LlmApiKind::GeminiGenerateContent,
                _ => LlmApiKind::OpenAiCompatible,
            },
            None => detect_api_kind(self.provider, self.base_url),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct StreamEvent {
    pub text: Option<String>,
    pub reasoning: Option<String>,
    pub error: Option<String>,
    pub done: bool,
}

pub fn detect_api_kind(provider: &str, base_url: &str) -> LlmApiKind {
    let provider = provider.trim().to_ascii_lowercase();
    let host = url::Url::parse(base_url.trim())
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase));

    if matches!(provider.as_str(), "claude" | "anthropic")
        && host.as_deref() == Some(ANTHROPIC_API_HOST)
    {
        LlmApiKind::AnthropicMessages
    } else {
        LlmApiKind::OpenAiCompatible
    }
}

fn parse_http_url(base_url: &str) -> Result<url::Url, String> {
    let mut url = url::Url::parse(base_url.trim())
        .map_err(|error| format!("Invalid LLM base URL: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("LLM base URL must use http or https scheme".to_string());
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("LLM base URL must not include credentials".to_string());
    }
    if url.fragment().is_some() {
        return Err("LLM base URL must not include a fragment".to_string());
    }
    url.set_fragment(None);
    Ok(url)
}

fn replace_or_append_path(url: &mut url::Url, current_suffix: &str, target_suffix: &str) {
    let path = url.path().trim_end_matches('/');
    let root = path.strip_suffix(current_suffix).unwrap_or(path);
    url.set_path(&format!("{root}{target_suffix}"));
}

fn anthropic_api_root(path: &str) -> String {
    let path = path.trim_end_matches('/');
    if let Some(root) = path.strip_suffix("/messages") {
        return root.to_string();
    }
    if let Some(root) = path.strip_suffix("/models") {
        return root.to_string();
    }
    if path.is_empty() {
        "/v1".to_string()
    } else {
        path.to_string()
    }
}

/// Root of a Gemini-style API (`.../v1beta`), tolerating a pasted `/models`
/// or full `/models/<model>:generateContent` URL.
fn gemini_api_root(path: &str) -> String {
    let path = path.trim_end_matches('/');
    if let Some(index) = path.find("/models") {
        let rest = &path[index..];
        if rest == "/models" || rest.starts_with("/models/") {
            return path[..index].to_string();
        }
    }
    path.to_string()
}

fn gemini_model_segment(model: &str) -> Result<String, String> {
    let model = model.trim();
    let model = model.strip_prefix("models/").unwrap_or(model);
    if model.is_empty() {
        return Err("A model is required for the Gemini protocol".to_string());
    }
    if model.contains(['/', '?', '#']) {
        return Err("Gemini model name must not contain '/', '?' or '#'".to_string());
    }
    Ok(model.to_string())
}

pub fn chat_endpoint(provider: &str, base_url: &str) -> Result<String, String> {
    chat_endpoint_for(&LlmTarget::plain(provider, base_url), "", false)
}

/// Chat endpoint for a target. `model` and `stream` only matter for Gemini,
/// whose URL encodes both.
pub fn chat_endpoint_for(target: &LlmTarget, model: &str, stream: bool) -> Result<String, String> {
    let kind = target.api_kind();
    let mut url = parse_http_url(target.base_url)?;
    match kind {
        LlmApiKind::GeminiGenerateContent => {
            let root = gemini_api_root(url.path());
            let model = gemini_model_segment(model)?;
            let method = if stream {
                "streamGenerateContent"
            } else {
                "generateContent"
            };
            url.set_path(&format!("{root}/models/{model}:{method}"));
            url.set_query(stream.then_some("alt=sse"));
        }
        LlmApiKind::AnthropicMessages => {
            let root = anthropic_api_root(url.path());
            url.set_path(&format!("{root}/messages"));
        }
        LlmApiKind::OpenAiCompatible => {
            let path = url.path().trim_end_matches('/');
            if !path.ends_with("/chat/completions") {
                url.set_path(&format!("{path}/chat/completions"));
            }
        }
    }
    Ok(url.to_string())
}

pub fn models_endpoint(provider: &str, base_url: &str) -> Result<String, String> {
    models_endpoint_for(&LlmTarget::plain(provider, base_url))
}

pub fn models_endpoint_for(target: &LlmTarget) -> Result<String, String> {
    let kind = target.api_kind();
    let mut url = parse_http_url(target.base_url)?;
    match kind {
        LlmApiKind::GeminiGenerateContent => {
            let root = gemini_api_root(url.path());
            url.set_path(&format!("{root}/models"));
            url.set_query(None);
        }
        LlmApiKind::AnthropicMessages => {
            let root = anthropic_api_root(url.path());
            url.set_path(&format!("{root}/models"));
        }
        LlmApiKind::OpenAiCompatible => {
            replace_or_append_path(&mut url, "/chat/completions", "/models");
        }
    }
    Ok(url.to_string())
}

fn is_direct_openai(provider: &str, base_url: &str) -> bool {
    provider.trim().eq_ignore_ascii_case("openai")
        && url::Url::parse(base_url.trim())
            .ok()
            .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
            .as_deref()
            == Some(OPENAI_API_HOST)
}

fn is_reasoning_model_without_sampling_controls(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    model.starts_with("gpt-5")
        || model == "o1"
        || model.starts_with("o1-")
        || model == "o3"
        || model.starts_with("o3-")
        || model == "o4"
        || model.starts_with("o4-")
}

pub fn request_timeout(provider: &str, base_url: &str, model: &str) -> Duration {
    request_timeout_for(&LlmTarget::plain(provider, base_url), model)
}

pub fn request_timeout_for(target: &LlmTarget, model: &str) -> Duration {
    // Corporate gateways add policy checks and logging hops; give them the
    // longer budget regardless of protocol.
    if target.gateway.is_some()
        || target.api_kind() == LlmApiKind::AnthropicMessages
        || is_reasoning_model_without_sampling_controls(model)
    {
        Duration::from_secs(60)
    } else {
        Duration::from_secs(30)
    }
}

fn normalize_anthropic_model(model: &str) -> String {
    let model = model
        .trim()
        .strip_prefix("anthropic/")
        .unwrap_or(model.trim());
    match model {
        "claude-sonnet-4" => "claude-sonnet-4-0".to_string(),
        "claude-opus-4" => "claude-opus-4-0".to_string(),
        _ => model.to_string(),
    }
}

pub fn build_chat_body(
    provider: &str,
    base_url: &str,
    model: &str,
    messages: Vec<Value>,
    max_tokens: u32,
    temperature: f64,
    stream: bool,
) -> Value {
    build_chat_body_for(
        &LlmTarget::plain(provider, base_url),
        model,
        messages,
        max_tokens,
        temperature,
        stream,
    )
}

fn build_gemini_body(messages: Vec<Value>, max_tokens: u32, temperature: f64) -> Value {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    for message in messages {
        let text = message["content"].as_str().unwrap_or_default().to_string();
        match message["role"].as_str() {
            Some("system") => {
                if !text.trim().is_empty() {
                    system_parts.push(json!({ "text": text }));
                }
            }
            Some("assistant") | Some("model") => {
                contents.push(json!({ "role": "model", "parts": [{ "text": text }] }));
            }
            _ => contents.push(json!({ "role": "user", "parts": [{ "text": text }] })),
        }
    }
    let mut body = json!({
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature.clamp(0.0, 2.0)
        }
    });
    if !system_parts.is_empty() {
        body.as_object_mut().unwrap().insert(
            "systemInstruction".to_string(),
            json!({ "parts": system_parts }),
        );
    }
    body
}

pub fn build_chat_body_for(
    target: &LlmTarget,
    model: &str,
    messages: Vec<Value>,
    max_tokens: u32,
    temperature: f64,
    stream: bool,
) -> Value {
    let provider = target.provider;
    let base_url = target.base_url;
    match target.api_kind() {
        LlmApiKind::GeminiGenerateContent => build_gemini_body(messages, max_tokens, temperature),
        LlmApiKind::AnthropicMessages => {
            let mut system_parts = Vec::new();
            let mut anthropic_messages = Vec::new();
            for message in messages {
                if message["role"].as_str() == Some("system") {
                    if let Some(content) = message["content"].as_str() {
                        if !content.trim().is_empty() {
                            system_parts.push(content.to_string());
                        }
                    }
                } else {
                    anthropic_messages.push(message);
                }
            }

            let mut body = json!({
                "model": normalize_anthropic_model(model),
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
                "temperature": temperature.clamp(0.0, 1.0),
                "stream": stream
            });
            if !system_parts.is_empty() {
                body.as_object_mut().unwrap().insert(
                    "system".to_string(),
                    Value::String(system_parts.join("\n\n")),
                );
            }
            body
        }
        LlmApiKind::OpenAiCompatible => {
            let mut body = json!({
                "model": model,
                "messages": messages,
                "stream": stream
            });
            let object = body.as_object_mut().unwrap();
            // Gateways usually front Azure/OpenAI deployments, which reject
            // sampling controls on reasoning models just like the direct API.
            let strict_openai_fields = is_direct_openai(provider, base_url)
                || (target.gateway.is_some()
                    && is_reasoning_model_without_sampling_controls(model));
            if strict_openai_fields {
                object.insert("max_completion_tokens".to_string(), json!(max_tokens));
                if !is_reasoning_model_without_sampling_controls(model) {
                    object.insert("temperature".to_string(), json!(temperature));
                }
            } else {
                object.insert("max_tokens".to_string(), json!(max_tokens));
                object.insert("temperature".to_string(), json!(temperature));
            }
            body
        }
    }
}

pub fn apply_auth_headers(
    request: RequestBuilder,
    provider: &str,
    base_url: &str,
    api_key: &str,
) -> RequestBuilder {
    apply_auth_headers_for(request, &LlmTarget::plain(provider, base_url), api_key)
}

/// Applies auth (and, for gateways, the configured extra headers). Invalid
/// gateway headers are rejected earlier by `LlmGatewayConfig::validate`; if
/// one slips through it is skipped with a warning rather than sent malformed.
pub fn apply_auth_headers_for(
    request: RequestBuilder,
    target: &LlmTarget,
    api_key: &str,
) -> RequestBuilder {
    if let Some(gateway) = target.gateway {
        return match gateway.request_headers(api_key) {
            Ok(headers) => request.headers(headers),
            Err(error) => {
                tracing::warn!("Skipping invalid gateway headers: {error}");
                request
            }
        };
    }
    let provider = target.provider;
    let api_key = api_key.trim();
    match target.api_kind() {
        LlmApiKind::GeminiGenerateContent => {
            if api_key.is_empty() {
                request
            } else {
                request.header("x-goog-api-key", api_key)
            }
        }
        LlmApiKind::AnthropicMessages => request
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION),
        LlmApiKind::OpenAiCompatible => {
            if super::provider_requires_api_key(provider) || !api_key.is_empty() {
                request.header("Authorization", format!("Bearer {api_key}"))
            } else {
                request
            }
        }
    }
}

fn gemini_candidate_text(body: &Value) -> String {
    body["candidates"][0]["content"]["parts"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|part| !part["thought"].as_bool().unwrap_or(false))
        .filter_map(|part| part["text"].as_str())
        .collect::<Vec<_>>()
        .join("")
}

pub fn response_text(kind: LlmApiKind, body: &Value) -> String {
    match kind {
        LlmApiKind::GeminiGenerateContent => gemini_candidate_text(body),
        LlmApiKind::AnthropicMessages => body["content"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|block| block["type"].as_str() == Some("text"))
            .filter_map(|block| block["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        LlmApiKind::OpenAiCompatible => {
            let message = &body["choices"][0]["message"];
            message["content"]
                .as_str()
                .filter(|content| !content.is_empty())
                .or_else(|| message["reasoning_content"].as_str())
                .unwrap_or("")
                .to_string()
        }
    }
}

pub fn parse_stream_event(kind: LlmApiKind, body: &Value) -> StreamEvent {
    match kind {
        LlmApiKind::GeminiGenerateContent => {
            if let Some(message) = body["error"]["message"].as_str() {
                return StreamEvent {
                    error: Some(message.to_string()),
                    ..StreamEvent::default()
                };
            }
            let text = gemini_candidate_text(body);
            StreamEvent {
                text: (!text.is_empty()).then_some(text),
                done: body["candidates"][0]["finishReason"].is_string(),
                ..StreamEvent::default()
            }
        }
        LlmApiKind::AnthropicMessages => {
            if body["type"].as_str() == Some("error") {
                return StreamEvent {
                    error: body["error"]["message"].as_str().map(str::to_string),
                    ..StreamEvent::default()
                };
            }
            if body["type"].as_str() == Some("message_stop") {
                return StreamEvent {
                    done: true,
                    ..StreamEvent::default()
                };
            }
            if body["type"].as_str() == Some("content_block_delta")
                && body["delta"]["type"].as_str() == Some("text_delta")
            {
                return StreamEvent {
                    text: body["delta"]["text"].as_str().map(str::to_string),
                    ..StreamEvent::default()
                };
            }
            StreamEvent::default()
        }
        LlmApiKind::OpenAiCompatible => {
            let delta = &body["choices"][0]["delta"];
            StreamEvent {
                text: delta["content"].as_str().map(str::to_string),
                reasoning: delta["reasoning_content"].as_str().map(str::to_string),
                ..StreamEvent::default()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn messages() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "Be concise."}),
            json!({"role": "user", "content": "Hello"}),
        ]
    }

    #[test]
    fn native_anthropic_uses_messages_endpoint_without_double_appending() {
        for base_url in [
            "https://api.anthropic.com",
            "https://api.anthropic.com/v1",
            "https://api.anthropic.com/v1/messages",
        ] {
            assert_eq!(
                chat_endpoint("claude", base_url).unwrap(),
                "https://api.anthropic.com/v1/messages"
            );
        }
        assert_eq!(
            models_endpoint("claude", "https://api.anthropic.com/v1/messages").unwrap(),
            "https://api.anthropic.com/v1/models"
        );
    }

    #[test]
    fn claude_on_openrouter_stays_openai_compatible() {
        assert_eq!(
            detect_api_kind("claude", "https://openrouter.ai/api/v1"),
            LlmApiKind::OpenAiCompatible
        );
        assert_eq!(
            chat_endpoint("claude", "https://openrouter.ai/api/v1").unwrap(),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[test]
    fn native_anthropic_body_moves_system_prompt_and_normalizes_old_default_model() {
        let body = build_chat_body(
            "claude",
            "https://api.anthropic.com/v1/messages",
            "anthropic/claude-sonnet-4",
            messages(),
            256,
            0.3,
            true,
        );

        assert_eq!(body["model"], "claude-sonnet-4-0");
        assert_eq!(body["system"], "Be concise.");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["max_tokens"], 256);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn direct_openai_gpt5_uses_compatible_token_field_and_omits_temperature() {
        let body = build_chat_body(
            "openai",
            "https://api.openai.com/v1",
            "gpt-5",
            messages(),
            4096,
            0.3,
            false,
        );

        assert_eq!(body["max_completion_tokens"], 4096);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn openai_compatible_proxies_keep_legacy_fields_for_compatibility() {
        let body = build_chat_body(
            "openrouter",
            "https://openrouter.ai/api/v1",
            "openai/gpt-5",
            messages(),
            4096,
            0.3,
            false,
        );

        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["temperature"], 0.3);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn slow_reasoning_apis_get_a_longer_timeout_without_a_new_setting() {
        assert_eq!(
            request_timeout("openai", "https://api.openai.com/v1", "gpt-5"),
            Duration::from_secs(60)
        );
        assert_eq!(
            request_timeout(
                "claude",
                "https://api.anthropic.com/v1",
                "claude-sonnet-4-0"
            ),
            Duration::from_secs(60)
        );
        assert_eq!(
            request_timeout(
                "openrouter",
                "https://openrouter.ai/api/v1",
                "gemini-2.5-flash"
            ),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn auth_headers_match_native_anthropic_and_openai_protocols() {
        let anthropic = apply_auth_headers(
            reqwest::Client::new().post("https://api.anthropic.com/v1/messages"),
            "claude",
            "https://api.anthropic.com/v1",
            "anthropic-key",
        )
        .build()
        .unwrap();
        assert_eq!(anthropic.headers()["x-api-key"], "anthropic-key");
        assert_eq!(anthropic.headers()["anthropic-version"], ANTHROPIC_VERSION);
        assert!(anthropic.headers().get("Authorization").is_none());

        let openai = apply_auth_headers(
            reqwest::Client::new().post("https://api.openai.com/v1/chat/completions"),
            "openai",
            "https://api.openai.com/v1",
            "openai-key",
        )
        .build()
        .unwrap();
        assert_eq!(openai.headers()["Authorization"], "Bearer openai-key");
    }

    #[test]
    fn response_parsers_support_anthropic_json_and_streaming_events() {
        let response = json!({
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " world"}
            ]
        });
        assert_eq!(
            response_text(LlmApiKind::AnthropicMessages, &response),
            "Hello world"
        );

        let event = parse_stream_event(
            LlmApiKind::AnthropicMessages,
            &json!({
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"}
            }),
        );
        assert_eq!(event.text.as_deref(), Some("Hello"));
        assert!(!event.done);
    }

    fn gateway(protocol: &str) -> LlmGatewayConfig {
        LlmGatewayConfig {
            protocol: protocol.to_string(),
            ..LlmGatewayConfig::default()
        }
    }

    #[test]
    fn gateway_target_only_applies_to_gateway_provider() {
        let config = gateway("anthropic");
        let openai = LlmTarget::new("openai", "https://api.openai.com/v1", &config);
        assert!(openai.gateway.is_none());
        assert_eq!(openai.api_kind(), LlmApiKind::OpenAiCompatible);

        let gw = LlmTarget::new("gateway", "https://llm.corp.example/v1", &config);
        assert_eq!(gw.api_kind(), LlmApiKind::AnthropicMessages);
    }

    #[test]
    fn gateway_endpoints_follow_the_selected_protocol() {
        let openai = gateway("openai");
        let target = LlmTarget::new("gateway", "https://llm.corp.example/openai/v1", &openai);
        assert_eq!(
            chat_endpoint_for(&target, "gpt-4o", true).unwrap(),
            "https://llm.corp.example/openai/v1/chat/completions"
        );
        assert_eq!(
            models_endpoint_for(&target).unwrap(),
            "https://llm.corp.example/openai/v1/models"
        );

        let anthropic = gateway("anthropic");
        let target = LlmTarget::new(
            "gateway",
            "https://llm.corp.example/anthropic/v1",
            &anthropic,
        );
        assert_eq!(
            chat_endpoint_for(&target, "claude-sonnet-4-5", true).unwrap(),
            "https://llm.corp.example/anthropic/v1/messages"
        );

        let gemini = gateway("gemini");
        let target = LlmTarget::new("gateway", "https://llm.corp.example/gemini/v1beta", &gemini);
        assert_eq!(
            chat_endpoint_for(&target, "models/gemini-2.5-flash", true).unwrap(),
            "https://llm.corp.example/gemini/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            chat_endpoint_for(&target, "gemini-2.5-flash", false).unwrap(),
            "https://llm.corp.example/gemini/v1beta/models/gemini-2.5-flash:generateContent"
        );
        assert_eq!(
            models_endpoint_for(&target).unwrap(),
            "https://llm.corp.example/gemini/v1beta/models"
        );
        let pasted = LlmTarget::new(
            "gateway",
            "https://llm.corp.example/v1beta/models/gemini-2.5-flash:generateContent",
            &gemini,
        );
        assert_eq!(
            chat_endpoint_for(&pasted, "gemini-2.5-pro", false).unwrap(),
            "https://llm.corp.example/v1beta/models/gemini-2.5-pro:generateContent"
        );
        assert!(chat_endpoint_for(&target, " ", false).is_err());
    }

    #[test]
    fn gemini_body_moves_system_prompt_and_maps_roles() {
        let config = gateway("gemini");
        let target = LlmTarget::new("gateway", "https://llm.corp.example/v1beta", &config);
        let mut msgs = messages();
        msgs.push(json!({"role": "assistant", "content": "Hi"}));
        let body = build_chat_body_for(&target, "gemini-2.5-flash", msgs, 128, 0.3, true);
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Be concise.");
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "Hello");
        assert_eq!(body["contents"][1]["role"], "model");
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 128);
        assert!(body.get("model").is_none());
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn gateway_openai_reasoning_models_use_strict_fields() {
        let config = gateway("openai");
        let target = LlmTarget::new("gateway", "https://llm.corp.example/v1", &config);
        let body = build_chat_body_for(&target, "gpt-5", messages(), 64, 0.3, false);
        assert_eq!(body["max_completion_tokens"], 64);
        assert!(body.get("temperature").is_none());
        let body = build_chat_body_for(&target, "gpt-4o", messages(), 64, 0.3, false);
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["temperature"], 0.3);
    }

    #[test]
    fn gateway_auth_headers_use_custom_header_and_extras() {
        let config = LlmGatewayConfig {
            protocol: "openai".to_string(),
            auth_header: "api-key".to_string(),
            extra_headers: vec![gateway::GatewayHeader {
                name: "X-Department".to_string(),
                value: "rd".to_string(),
            }],
            ..LlmGatewayConfig::default()
        };
        let target = LlmTarget::new("gateway", "https://llm.corp.example/v1", &config);
        let request = apply_auth_headers_for(
            reqwest::Client::new().post("https://llm.corp.example/v1/chat/completions"),
            &target,
            "corp-key",
        )
        .build()
        .unwrap();
        assert_eq!(request.headers()["api-key"], "corp-key");
        assert_eq!(request.headers()["x-department"], "rd");
        assert!(request.headers().get("Authorization").is_none());

        let anthropic = gateway("anthropic");
        let target = LlmTarget::new("gateway", "https://llm.corp.example/v1", &anthropic);
        let request = apply_auth_headers_for(
            reqwest::Client::new().post("https://llm.corp.example/v1/messages"),
            &target,
            "corp-key",
        )
        .build()
        .unwrap();
        assert_eq!(request.headers()["x-api-key"], "corp-key");
        assert_eq!(request.headers()["anthropic-version"], ANTHROPIC_VERSION);
    }

    #[test]
    fn gemini_parsers_skip_thoughts_and_detect_finish() {
        let body = json!({
            "candidates": [{
                "content": {"parts": [
                    {"text": "thinking...", "thought": true},
                    {"text": "Hello"},
                    {"text": " world"}
                ]},
                "finishReason": "STOP"
            }]
        });
        assert_eq!(
            response_text(LlmApiKind::GeminiGenerateContent, &body),
            "Hello world"
        );
        let event = parse_stream_event(LlmApiKind::GeminiGenerateContent, &body);
        assert_eq!(event.text.as_deref(), Some("Hello world"));
        assert!(event.done);

        let partial = json!({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]});
        let event = parse_stream_event(LlmApiKind::GeminiGenerateContent, &partial);
        assert_eq!(event.text.as_deref(), Some("Hel"));
        assert!(!event.done);

        let error = json!({"error": {"message": "quota"}});
        assert_eq!(
            parse_stream_event(LlmApiKind::GeminiGenerateContent, &error)
                .error
                .as_deref(),
            Some("quota")
        );
    }
}
