use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;

use crate::error::AppError;

use super::{
    prompt, protocol, ChunkCallback, LlmConfig, LlmProvider, PolishRequest, PolishResponse,
};

pub struct OpenAiProvider {
    client: Client,
}

impl Default for OpenAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub fn with_client(client: Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn polish(
        &self,
        config: &LlmConfig,
        req: &PolishRequest,
        on_chunk: Option<&ChunkCallback>,
    ) -> Result<PolishResponse, AppError> {
        let has_selected_text = req
            .selected_text
            .as_ref()
            .is_some_and(|s| !s.trim().is_empty());

        let system_prompt = prompt::build_context_system_prompt(prompt::ContextPromptOptions {
            context: &req.context,
            dictionary: &req.dictionary,
            correction_rules: &req.correction_rules,
            polish_style: &req.polish_style,
            personal_style_prompt: "",
            mapped_scene_prompt: &req.mapped_scene_prompt,
            active_scene_prompt: &req.active_scene_prompt,
            polish_custom_prompt: &req.polish_custom_prompt,
            translate_enabled: req.translate_enabled,
            target_lang: &req.target_lang,
            has_selected_text,
            voice_intent: Some(&req.voice_intent),
        });

        let mut messages = vec![serde_json::json!({ "role": "system", "content": system_prompt })];
        if has_selected_text {
            messages.push(serde_json::json!({
                "role": "user",
                "content": format!("<selected_text>\n{}\n</selected_text>", req.selected_text.as_ref().unwrap())
            }));
        }
        messages.push(serde_json::json!({
            "role": "user",
            "content": format!("<transcription>\n{}\n</transcription>", req.raw_text)
        }));

        let target = config.target();
        let api_kind = target.api_kind();
        let stream = on_chunk.is_some();
        let endpoint = protocol::chat_endpoint_for(&target, &config.model, stream)
            .map_err(AppError::Config)?;
        let mut body = protocol::build_chat_body_for(
            &target,
            &config.model,
            messages,
            config.max_tokens,
            config.temperature,
            stream,
        );

        // GLM-4.7/4.5/5 default to thinking mode, but without explicitly enabling it
        // the API may return content in reasoning_content only, leaving content empty.
        // Explicitly enable thinking so both fields are properly populated.
        // Thinking mode also requires temperature >= 0.6 (recommended 1.0).
        if api_kind == protocol::LlmApiKind::OpenAiCompatible && config.model.starts_with("glm-") {
            if let Some(obj) = body.as_object_mut() {
                obj.insert(
                    "thinking".to_string(),
                    serde_json::json!({"type": "enabled"}),
                );
                obj.insert("temperature".to_string(), serde_json::json!(1.0));
                obj.insert("top_p".to_string(), serde_json::json!(0.95));
            }
        }

        // Retry the initial connection (not once streaming starts)
        #[allow(unused_assignments)]
        let mut response = None;
        let mut last_error: Option<AppError> = None;
        let mut attempt = 0u32;
        let mut api_key = config.api_key.clone();
        // Helper-issued gateway keys can expire before the cache TTL; refresh
        // once on an auth rejection.
        let mut can_refresh_gateway_key = target.gateway.is_some_and(|g| g.uses_key_command());

        loop {
            let request = self
                .client
                .post(&endpoint)
                .header("Content-Type", "application/json");
            match protocol::apply_auth_headers_for(request, &target, &api_key)
                .json(&body)
                .timeout(protocol::request_timeout_for(&target, &config.model))
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        response = Some(resp);
                        break;
                    } else if matches!(status.as_u16(), 401 | 403) && can_refresh_gateway_key {
                        can_refresh_gateway_key = false;
                        tracing::warn!("LLM gateway rejected the key ({status}), refreshing it");
                        super::gateway::invalidate_cached_key(&config.gateway);
                        api_key = super::gateway::resolve_api_key(&config.gateway, "", true)
                            .await
                            .map_err(AppError::Config)?;
                        continue;
                    } else if status.as_u16() >= 500 && attempt < 2 {
                        let body_text = resp.text().await.unwrap_or_default();
                        tracing::warn!(
                            "LLM server error {} (attempt {}/3), retrying",
                            status,
                            attempt + 1
                        );
                        last_error = Some(AppError::Api {
                            status: status.as_u16(),
                            body: body_text,
                        });
                        attempt += 1;
                        tokio::time::sleep(std::time::Duration::from_millis(
                            1000 * 2u64.pow(attempt - 1),
                        ))
                        .await;
                        continue;
                    } else {
                        let status = resp.status();
                        let text = resp.text().await.unwrap_or_default();
                        // Truncate at a valid UTF-8 char boundary to avoid panic on multi-byte chars
                        let truncate_at = text
                            .char_indices()
                            .take_while(|&(i, _)| i < 200)
                            .last()
                            .map(|(i, c)| i + c.len_utf8())
                            .unwrap_or(text.len());
                        let sanitized = &text[..truncate_at];
                        return Err(AppError::Api {
                            status: status.as_u16(),
                            body: sanitized.to_string(),
                        });
                    }
                }
                Err(e) if e.is_timeout() && attempt < 2 => {
                    tracing::warn!(
                        "LLM connection timeout (attempt {}/3), retrying",
                        attempt + 1
                    );
                    last_error = Some(e.into());
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(
                        1000 * 2u64.pow(attempt - 1),
                    ))
                    .await;
                    continue;
                }
                Err(e) if e.is_connect() && attempt < 2 => {
                    tracing::warn!(
                        "LLM connection failed (attempt {}/3), retrying",
                        attempt + 1
                    );
                    last_error = Some(e.into());
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(
                        1000 * 2u64.pow(attempt - 1),
                    ))
                    .await;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }

        let response = response.ok_or_else(|| last_error.unwrap())?;

        if let Some(callback) = on_chunk {
            // Streaming mode
            let mut full_text = String::new();
            let mut reasoning_text = String::new();
            let mut stream = response.bytes_stream();

            let mut buffer = String::new();
            let mut stream_done = false;
            while !stream_done {
                let Some(chunk) = stream.next().await else {
                    break;
                };
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process SSE lines
                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if let Some(data) = line.strip_prefix("data:").map(str::trim_start) {
                        if data == "[DONE]" {
                            stream_done = true;
                            break;
                        }
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            let event = protocol::parse_stream_event(api_kind, &v);
                            if let Some(error) = event.error {
                                return Err(AppError::Config(error));
                            }
                            if let Some(content) = event.text {
                                if !content.is_empty() {
                                    full_text.push_str(&content);
                                    callback(&content);
                                }
                            }

                            // Collect reasoning_content as fallback for thinking-mode models
                            // where all output may land in this field instead of content
                            if let Some(rc) = event.reasoning {
                                if !rc.is_empty() {
                                    reasoning_text.push_str(&rc);
                                }
                            }
                            stream_done = event.done;
                        }
                    }
                }
            }

            // If content was empty but reasoning_content had text, use it as output.
            // This handles GLM thinking-mode where the API puts all output in reasoning_content.
            if full_text.is_empty() && !reasoning_text.is_empty() {
                tracing::warn!(
                    "LLM content empty, using reasoning_content ({} chars) as output",
                    reasoning_text.len()
                );
                callback(&reasoning_text);
                full_text = reasoning_text;
            } else if full_text.is_empty() {
                tracing::error!("LLM streaming returned no content and no reasoning_content");
            }

            Ok(PolishResponse {
                polished_text: full_text,
            })
        } else {
            // Non-streaming mode
            let v: serde_json::Value = response.json().await?;
            let text = protocol::response_text(api_kind, &v);

            if text.is_empty() {
                tracing::warn!(
                    "LLM non-streaming returned empty content, full response: {}",
                    v
                );
            }

            Ok(PolishResponse {
                polished_text: text,
            })
        }
    }

    fn name(&self) -> &str {
        "OpenAI"
    }
}

#[cfg(all(test, unix))]
mod gateway_tests {
    use super::*;
    use crate::app_detector::types::{BrowserAccessStatus, ContextFamily, ContextProfileSummary};
    use crate::llm::gateway::{GatewayHeader, LlmGatewayConfig};
    use crate::voice_intent::{VoiceIntent, VoiceIntentKind, VoiceOutputPlacement};
    use std::io::{Read, Write};
    use std::sync::{Arc, Mutex};

    fn read_request(socket: &mut std::net::TcpStream) -> String {
        socket
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .unwrap();
        let mut data = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = socket.read(&mut buf).unwrap();
            data.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&data).to_string();
            if let Some(header_end) = text.find("\r\n\r\n") {
                let content_length = text[..header_end]
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                if data.len() >= header_end + 4 + content_length {
                    return text;
                }
            }
            if n == 0 {
                return text;
            }
        }
    }

    #[tokio::test]
    async fn gemini_gateway_refreshes_rejected_helper_key_and_streams() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::<String>::new()));
        let server_requests = requests.clone();
        let server = std::thread::spawn(move || {
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().unwrap();
                let request = read_request(&mut socket);
                server_requests.lock().unwrap().push(request);
                if attempt == 0 {
                    socket
                        .write_all(
                            b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 7\r\nConnection: close\r\n\r\nexpired",
                        )
                        .unwrap();
                } else {
                    let body = concat!(
                        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"plan\",\"thought\":true}]}}]}\r\n\r\n",
                        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\r\n\r\n",
                        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world\"}]},\"finishReason\":\"STOP\"}]}\r\n\r\n",
                    );
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    socket.write_all(response.as_bytes()).unwrap();
                }
            }
        });

        // The helper prints a different key on each run; the first one is
        // "stale" and must be replaced after the 401.
        let dir = std::env::temp_dir().join(format!("otl-gw-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let counter = dir.join("count");
        let _ = std::fs::remove_file(&counter);
        let script = format!(
            "echo x >> '{0}'; printf '{{\"token\":\"tok-%s\"}}' \"$(wc -l < '{0}' | tr -d ' ')\"",
            counter.display()
        );
        let gateway = LlmGatewayConfig {
            protocol: "gemini".to_string(),
            auth_mode: "command".to_string(),
            auth_header: "X-Corp-Auth".to_string(),
            auth_prefix: "Token ".to_string(),
            key_command: "/bin/sh".to_string(),
            key_command_args: vec!["-c".to_string(), script],
            key_command_json_field: "token".to_string(),
            extra_headers: vec![GatewayHeader {
                name: "X-Tenant".to_string(),
                value: "rd".to_string(),
            }],
            ..LlmGatewayConfig::default()
        };
        let stale_key = crate::llm::gateway::resolve_api_key(&gateway, "", false)
            .await
            .unwrap();
        assert_eq!(stale_key, "tok-1");

        let config = LlmConfig {
            provider: "gateway".to_string(),
            api_key: stale_key,
            model: "gemini-2.5-flash".to_string(),
            base_url: format!("http://{address}/gemini/v1beta"),
            max_tokens: 64,
            temperature: 0.3,
            gateway,
        };
        let request = PolishRequest {
            raw_text: "hello world".to_string(),
            context: ContextProfileSummary {
                profile_id: "general.native".to_string(),
                family: ContextFamily::General,
                app_label: "Test".to_string(),
                icon_key: "general".to_string(),
                override_id: None,
                browser_access_status: BrowserAccessStatus::NotApplicable,
                browser_target: None,
            },
            dictionary: vec![],
            correction_rules: vec![],
            polish_style: "clean".to_string(),
            mapped_scene_prompt: String::new(),
            active_scene_prompt: String::new(),
            polish_custom_prompt: String::new(),
            translate_enabled: false,
            target_lang: "en".to_string(),
            selected_text: None,
            operation_id: None,
            voice_intent: VoiceIntent::from_parts(
                VoiceIntentKind::DictateInsert,
                VoiceOutputPlacement::InsertAtCursor,
                1.0,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
        };

        let chunks = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = chunks.clone();
        let on_chunk: ChunkCallback =
            Box::new(move |chunk| sink.lock().unwrap().push(chunk.into()));
        let response = OpenAiProvider::new()
            .polish(&config, &request, Some(&on_chunk))
            .await
            .unwrap();
        server.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(response.polished_text, "Hello world");
        assert_eq!(*chunks.lock().unwrap(), vec!["Hello", " world"]);

        let requests = requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        let first = requests[0].to_ascii_lowercase();
        let second = requests[1].to_ascii_lowercase();
        assert!(first.starts_with(
            "post /gemini/v1beta/models/gemini-2.5-flash:streamgeneratecontent?alt=sse "
        ));
        assert!(first.contains("x-corp-auth: token tok-1"), "{first}");
        assert!(second.contains("x-corp-auth: token tok-2"), "{second}");
        assert!(second.contains("x-tenant: rd"));
        assert!(!second.contains("authorization:"));
        assert!(requests[1].contains("\"systemInstruction\""));
        assert!(requests[1].contains("<transcription>"));
    }
}
