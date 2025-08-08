use crate::api_key::ApiKeyState;
use crate::ui::InstructionListItem;
use anthropic::{
    ANTHROPIC_API_URL, AnthropicError, AnthropicModelMode, ContentDelta, Event, ResponseContent,
    ToolResultContent, ToolResultPart, Usage, oauth,
};
use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use collections::{BTreeMap, HashMap};
use editor::{Editor, EditorElement, EditorStyle};
use futures::{FutureExt, Stream, StreamExt, future::BoxFuture, stream::BoxStream};
use gpui::{AnyView, App, AsyncApp, Context, Entity, FontStyle, Task, TextStyle, WhiteSpace};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, ConfigurationViewTargetAgent, LanguageModel,
    LanguageModelCacheConfiguration, LanguageModelCompletionError, LanguageModelId,
    LanguageModelName, LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, LanguageModelRequest, LanguageModelToolChoice,
    LanguageModelToolResultContent, MessageContent, RateLimiter, Role,
};
use language_model::{LanguageModelCompletionEvent, LanguageModelToolUse, StopReason};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore};
use std::pin::Pin;
use std::str::FromStr;
use std::sync::{Arc, LazyLock};
use strum::IntoEnumIterator;
use theme::ThemeSettings;
use ui::{
    Icon, IconName, List, ToggleButtonGroup, ToggleButtonGroupStyle, ToggleButtonSimple, Tooltip,
    prelude::*,
};
use util::{ResultExt, truncate_and_trailoff};
use zed_env_vars::{EnvVar, env_var};

const PROVIDER_ID: LanguageModelProviderId = language_model::ANTHROPIC_PROVIDER_ID;
const PROVIDER_NAME: LanguageModelProviderName = language_model::ANTHROPIC_PROVIDER_NAME;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct AnthropicSettings {
    pub api_url: String,
    /// Extend Zed's list of Anthropic models.
    pub available_models: Vec<AvailableModel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AvailableModel {
    /// The model's name in the Anthropic API. e.g. claude-3-5-sonnet-latest, claude-3-opus-20240229, etc
    pub name: String,
    /// The model's name in Zed's UI, such as in the model selector dropdown menu in the assistant panel.
    pub display_name: Option<String>,
    /// The model's context window size.
    pub max_tokens: u64,
    /// A model `name` to substitute when calling tools, in case the primary model doesn't support tool calling.
    pub tool_override: Option<String>,
    /// Configuration of Anthropic's caching API.
    pub cache_configuration: Option<LanguageModelCacheConfiguration>,
    pub max_output_tokens: Option<u64>,
    pub default_temperature: Option<f32>,
    #[serde(default)]
    pub extra_beta_headers: Vec<String>,
    /// The model's mode (e.g. thinking)
    pub mode: Option<ModelMode>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelMode {
    #[default]
    Default,
    Thinking {
        /// The maximum number of tokens to use for reasoning. Must be lower than the model's `max_output_tokens`.
        budget_tokens: Option<u32>,
    },
}

impl From<ModelMode> for AnthropicModelMode {
    fn from(value: ModelMode) -> Self {
        match value {
            ModelMode::Default => AnthropicModelMode::Default,
            ModelMode::Thinking { budget_tokens } => AnthropicModelMode::Thinking { budget_tokens },
        }
    }
}

impl From<AnthropicModelMode> for ModelMode {
    fn from(value: AnthropicModelMode) -> Self {
        match value {
            AnthropicModelMode::Default => ModelMode::Default,
            AnthropicModelMode::Thinking { budget_tokens } => ModelMode::Thinking { budget_tokens },
        }
    }
}

pub struct AnthropicLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: gpui::Entity<State>,
}

const API_KEY_ENV_VAR_NAME: &str = "ANTHROPIC_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);

#[derive(Debug, Clone)]
enum AuthMethod {
    ApiKey,
    OAuth {
        refresh_token: String,
        access_token: Option<CachedAccessToken>,
    },
}

#[derive(Debug, Clone)]
struct CachedAccessToken {
    token: String,
    expires_at: DateTime<Utc>,
}

pub struct State {
    api_key_state: ApiKeyState,
    auth_method: Option<AuthMethod>,
}

impl State {
    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let api_url = AnthropicLanguageModelProvider::api_url(cx);
        self.auth_method = api_key.as_ref().map(|_| AuthMethod::ApiKey);
        self.api_key_state
            .store(api_url, api_key, |this| &mut this.api_key_state, cx)
    }

    fn set_refresh_token(
        &mut self,
        refresh_token: String,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let api_url = AnthropicLanguageModelProvider::api_url(cx);
        let credentials_provider = <dyn credentials_provider::CredentialsProvider>::global(cx);

        self.auth_method = Some(AuthMethod::OAuth {
            refresh_token: refresh_token.clone(),
            access_token: None,
        });

        cx.spawn(async move |this, cx| {
            credentials_provider
                .write_credentials(&api_url, "OAuth", refresh_token.as_bytes(), cx)
                .await
                .ok();

            this.update(cx, |_, cx| {
                cx.notify();
            })
        })
    }

    fn is_authenticated(&self) -> bool {
        self.api_key_state.has_key() || self.auth_method.is_some()
    }

    fn is_using_api_key(&self) -> bool {
        matches!(self.auth_method, Some(AuthMethod::ApiKey))
            || (self.auth_method.is_none() && self.api_key_state.has_key())
    }

    fn is_api_key_from_env(&self) -> bool {
        self.api_key_state.is_from_env_var()
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let api_url = AnthropicLanguageModelProvider::api_url(cx);
        let credentials_provider = <dyn credentials_provider::CredentialsProvider>::global(cx);
        
        cx.spawn(async move |this, cx| {
            // First check if we have OAuth credentials stored
            let oauth_check = credentials_provider.read_credentials(&api_url, &cx).await;
            
            if let Ok(Some((auth_type, credentials))) = oauth_check {
                if auth_type == "OAuth" {
                    let refresh_token = String::from_utf8(credentials)
                        .map_err(|_| AuthenticateError::CredentialsNotFound)?;
                    this.update(cx, |this, cx| {
                        this.auth_method = Some(AuthMethod::OAuth {
                            refresh_token,
                            access_token: None,
                        });
                        cx.notify();
                    })?;
                    return Ok(());
                }
            }

            // Fall back to API key authentication
            this.update(cx, |this, cx| {
                this.api_key_state.load_if_needed(
                    api_url,
                    &API_KEY_ENV_VAR,
                    |this| &mut this.api_key_state,
                    cx,
                )
            })?
            .await
        })
    }

    fn get_cached_token(&self, api_url: &SharedString) -> Option<String> {
        match &self.auth_method {
            Some(AuthMethod::ApiKey) | None => {
                self.api_key_state.key(api_url).map(|k| k.to_string())
            }
            Some(AuthMethod::OAuth { access_token, .. }) => {
                access_token.as_ref().and_then(|cached| {
                    if !oauth::is_token_expired(cached.expires_at) {
                        Some(cached.token.clone())
                    } else {
                        None
                    }
                })
            }
        }
    }

    fn refresh_oauth_token(
        &mut self,
        http_client: Arc<dyn HttpClient>,
        cx: &mut Context<Self>,
    ) -> Task<Result<String>> {
        let refresh_token = match &self.auth_method {
            Some(AuthMethod::OAuth { refresh_token, .. }) => refresh_token.clone(),
            _ => return Task::ready(Err(anyhow!("No OAuth refresh token available"))),
        };

        let api_url = AnthropicLanguageModelProvider::api_url(cx);
        let credentials_provider = <dyn credentials_provider::CredentialsProvider>::global(cx);

        cx.spawn(async move |this, cx| {
            let token_response =
                oauth::refresh_access_token(http_client.as_ref(), &refresh_token).await?;
            let expires_at = oauth::calculate_token_expiration(token_response.expires_in);

            let new_cached = CachedAccessToken {
                token: token_response.access_token.clone(),
                expires_at,
            };

            // Persist the new refresh token to credentials storage
            credentials_provider
                .write_credentials(
                    &api_url,
                    "OAuth",
                    token_response.refresh_token.as_bytes(),
                    cx,
                )
                .await
                .ok();

            this.update(cx, |this, cx| {
                this.auth_method = Some(AuthMethod::OAuth {
                    refresh_token: token_response.refresh_token.clone(),
                    access_token: Some(new_cached),
                });
                cx.notify();
            })?;

            Ok(token_response.access_token)
        })
    }
}

impl AnthropicLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                let api_url = Self::api_url(cx);
                this.api_key_state.handle_url_change(
                    api_url,
                    &API_KEY_ENV_VAR,
                    |this| &mut this.api_key_state,
                    cx,
                );
                cx.notify();
            })
            .detach();
            State {
                api_key_state: ApiKeyState::new(Self::api_url(cx)),
                auth_method: None,
            }
        });

        Self { http_client, state }
    }

    fn create_language_model(&self, model: anthropic::Model) -> Arc<dyn LanguageModel> {
        Arc::new(AnthropicModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn settings(cx: &App) -> &AnthropicSettings {
        &crate::AllLanguageModelSettings::get_global(cx).anthropic
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            ANTHROPIC_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }
}

impl LanguageModelProviderState for AnthropicLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<gpui::Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for AnthropicLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconName {
        IconName::AiAnthropic
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(anthropic::Model::default()))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(anthropic::Model::default_fast()))
    }

    fn recommended_models(&self, _cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        [
            anthropic::Model::ClaudeSonnet4,
            anthropic::Model::ClaudeSonnet4Thinking,
        ]
        .into_iter()
        .map(|model| self.create_language_model(model))
        .collect()
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = BTreeMap::default();

        // Add base models from anthropic::Model::iter()
        for model in anthropic::Model::iter() {
            if !matches!(model, anthropic::Model::Custom { .. }) {
                models.insert(model.id().to_string(), model);
            }
        }

        // Override with available models from settings
        for model in &AnthropicLanguageModelProvider::settings(cx).available_models {
            models.insert(
                model.name.clone(),
                anthropic::Model::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    tool_override: model.tool_override.clone(),
                    cache_configuration: model.cache_configuration.as_ref().map(|config| {
                        anthropic::AnthropicModelCacheConfiguration {
                            max_cache_anchors: config.max_cache_anchors,
                            should_speculate: config.should_speculate,
                            min_total_token: config.min_total_token,
                        }
                    }),
                    max_output_tokens: model.max_output_tokens,
                    default_temperature: model.default_temperature,
                    extra_beta_headers: model.extra_beta_headers.clone(),
                    mode: model.mode.clone().unwrap_or_default().into(),
                },
            );
        }

        models
            .into_values()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(
        &self,
        target_agent: ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), target_agent, window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| {
            state.auth_method = None;
            state.set_api_key(None, cx)
        })
    }
}

pub struct AnthropicModel {
    id: LanguageModelId,
    model: anthropic::Model,
    state: gpui::Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

pub fn count_anthropic_tokens(
    request: LanguageModelRequest,
    cx: &App,
) -> BoxFuture<'static, Result<u64>> {
    cx.background_spawn(async move {
        let messages = request.messages;
        let mut tokens_from_images = 0;
        let mut string_messages = Vec::with_capacity(messages.len());

        for message in messages {
            use language_model::MessageContent;

            let mut string_contents = String::new();

            for content in message.content {
                match content {
                    MessageContent::Text(text) => {
                        string_contents.push_str(&text);
                    }
                    MessageContent::Thinking { .. } => {
                        // Thinking blocks are not included in the input token count.
                    }
                    MessageContent::RedactedThinking(_) => {
                        // Thinking blocks are not included in the input token count.
                    }
                    MessageContent::Image(image) => {
                        tokens_from_images += image.estimate_tokens();
                    }
                    MessageContent::ToolUse(_tool_use) => {
                        // TODO: Estimate token usage from tool uses.
                    }
                    MessageContent::ToolResult(tool_result) => match &tool_result.content {
                        LanguageModelToolResultContent::Text(text) => {
                            string_contents.push_str(text);
                        }
                        LanguageModelToolResultContent::Image(image) => {
                            tokens_from_images += image.estimate_tokens();
                        }
                    },
                }
            }

            if !string_contents.is_empty() {
                string_messages.push(tiktoken_rs::ChatCompletionRequestMessage {
                    role: match message.role {
                        Role::User => "user".into(),
                        Role::Assistant => "assistant".into(),
                        Role::System => "system".into(),
                    },
                    content: Some(string_contents),
                    name: None,
                    function_call: None,
                });
            }
        }

        // Tiktoken doesn't yet support these models, so we manually use the
        // same tokenizer as GPT-4.
        tiktoken_rs::num_tokens_from_messages("gpt-4", &string_messages)
            .map(|tokens| (tokens + tokens_from_images) as u64)
    })
    .boxed()
}

impl AnthropicModel {
    fn stream_completion(
        &self,
        request: anthropic::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<anthropic::Event, AnthropicError>>,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let state = self.state.clone();

        // Extract values from cx before async move
        let result = cx.update(|cx| {
            let api_url = AnthropicLanguageModelProvider::api_url(cx);
            let (cached_token, is_using_api_key) = cx.read_entity(&state, |state, _| {
                (state.get_cached_token(&api_url), state.is_using_api_key())
            });
            let needs_oauth_refresh = if cached_token.is_none() {
                cx.read_entity(&state, |state, _| {
                    matches!(state.auth_method, Some(AuthMethod::OAuth { .. }))
                })
            } else {
                false
            };
            let refresh_task = if cached_token.is_none() && needs_oauth_refresh {
                Some(state.update(cx, |state, cx| {
                    state.refresh_oauth_token(http_client.clone(), cx)
                }))
            } else {
                None
            };
            (api_url, cached_token, is_using_api_key, refresh_task)
        });

        let beta_headers = self.model.beta_headers();

        async move {
            let (api_url, cached_token, is_using_api_key, refresh_task) =
                result.map_err(|_| anyhow!("App state dropped"))?;

            let (auth_token, is_oauth) = if let Some(token) = cached_token {
                (token, !is_using_api_key)
            } else if let Some(refresh_task) = refresh_task {
                let token = refresh_task.await?;
                (token, true)
            } else {
                return Err(LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                });
            };

            // Prepend Claude Code system prompt for OAuth authentication
            let request = if is_oauth {
                let mut modified_request = request;
                let claude_code_prompt =
                    "You are Claude Code, Anthropic's official CLI for Claude.";

                // Prepend the Claude Code prompt to the system message
                modified_request.system = match modified_request.system {
                    Some(anthropic::StringOrContents::String(existing)) => {
                        Some(anthropic::StringOrContents::Content(vec![
                            anthropic::RequestContent::Text {
                                text: claude_code_prompt.to_string(),
                                cache_control: None,
                            },
                            anthropic::RequestContent::Text {
                                text: existing,
                                cache_control: None,
                            },
                        ]))
                    }
                    Some(anthropic::StringOrContents::Content(mut contents)) => {
                        // Insert Claude Code prompt at the beginning
                        contents.insert(
                            0,
                            anthropic::RequestContent::Text {
                                text: claude_code_prompt.to_string(),
                                cache_control: None,
                            },
                        );
                        Some(anthropic::StringOrContents::Content(contents))
                    }
                    None => Some(anthropic::StringOrContents::String(
                        claude_code_prompt.to_string(),
                    )),
                };

                modified_request
            } else {
                request
            };

            let request = if is_oauth {
                // Use OAuth endpoint with Bearer token (request already has Claude Code prompt)
                anthropic::stream_completion_with_oauth(
                    http_client.as_ref(),
                    &api_url,
                    &auth_token,
                    request,
                )
                .await
            } else {
                // Use regular API key endpoint with beta headers
                anthropic::stream_completion(
                    http_client.as_ref(),
                    &api_url,
                    &auth_token,
                    request,
                    beta_headers,
                )
                .await
            };

            request.map_err(Into::into)
        }
        .boxed()
    }
}

impl LanguageModel for AnthropicModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto
            | LanguageModelToolChoice::Any
            | LanguageModelToolChoice::None => true,
        }
    }

    fn telemetry_id(&self) -> String {
        format!("anthropic/{}", self.model.id())
    }

    fn api_key(&self, cx: &App) -> Option<String> {
        self.state.read_with(cx, |state, cx| {
            let api_url = AnthropicLanguageModelProvider::api_url(cx);
            state.api_key_state.key(&api_url).map(|key| key.to_string())
        })
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        Some(self.model.max_output_tokens())
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        count_anthropic_tokens(request, cx)
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            BoxStream<'static, Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>,
            LanguageModelCompletionError,
        >,
    > {
        let request = into_anthropic(
            request,
            self.model.request_id().into(),
            self.model.default_temperature(),
            self.model.max_output_tokens(),
            self.model.mode(),
        );
        let request = self.stream_completion(request, cx);
        let future = self.request_limiter.stream(async move {
            let response = request.await?;
            Ok(AnthropicEventMapper::new().map_stream(response))
        });
        async move { Ok(future.await?.boxed()) }.boxed()
    }

    fn cache_configuration(&self) -> Option<LanguageModelCacheConfiguration> {
        self.model
            .cache_configuration()
            .map(|config| LanguageModelCacheConfiguration {
                max_cache_anchors: config.max_cache_anchors,
                should_speculate: config.should_speculate,
                min_total_token: config.min_total_token,
            })
    }
}

pub fn into_anthropic(
    request: LanguageModelRequest,
    model: String,
    default_temperature: f32,
    max_output_tokens: u64,
    mode: AnthropicModelMode,
) -> anthropic::Request {
    let mut new_messages: Vec<anthropic::Message> = Vec::new();
    let mut system_message = String::new();

    for message in request.messages {
        if message.contents_empty() {
            continue;
        }

        match message.role {
            Role::User | Role::Assistant => {
                let mut anthropic_message_content: Vec<anthropic::RequestContent> = message
                    .content
                    .into_iter()
                    .filter_map(|content| match content {
                        MessageContent::Text(text) => {
                            let text = if text.chars().last().is_some_and(|c| c.is_whitespace()) {
                                text.trim_end().to_string()
                            } else {
                                text
                            };
                            if !text.is_empty() {
                                Some(anthropic::RequestContent::Text {
                                    text,
                                    cache_control: None,
                                })
                            } else {
                                None
                            }
                        }
                        MessageContent::Thinking {
                            text: thinking,
                            signature,
                        } => {
                            if !thinking.is_empty() {
                                Some(anthropic::RequestContent::Thinking {
                                    thinking,
                                    signature: signature.unwrap_or_default(),
                                    cache_control: None,
                                })
                            } else {
                                None
                            }
                        }
                        MessageContent::RedactedThinking(data) => {
                            if !data.is_empty() {
                                Some(anthropic::RequestContent::RedactedThinking { data })
                            } else {
                                None
                            }
                        }
                        MessageContent::Image(image) => Some(anthropic::RequestContent::Image {
                            source: anthropic::ImageSource {
                                source_type: "base64".to_string(),
                                media_type: "image/png".to_string(),
                                data: image.source.to_string(),
                            },
                            cache_control: None,
                        }),
                        MessageContent::ToolUse(tool_use) => {
                            Some(anthropic::RequestContent::ToolUse {
                                id: tool_use.id.to_string(),
                                name: tool_use.name.to_string(),
                                input: tool_use.input,
                                cache_control: None,
                            })
                        }
                        MessageContent::ToolResult(tool_result) => {
                            Some(anthropic::RequestContent::ToolResult {
                                tool_use_id: tool_result.tool_use_id.to_string(),
                                is_error: tool_result.is_error,
                                content: match tool_result.content {
                                    LanguageModelToolResultContent::Text(text) => {
                                        ToolResultContent::Plain(text.to_string())
                                    }
                                    LanguageModelToolResultContent::Image(image) => {
                                        ToolResultContent::Multipart(vec![ToolResultPart::Image {
                                            source: anthropic::ImageSource {
                                                source_type: "base64".to_string(),
                                                media_type: "image/png".to_string(),
                                                data: image.source.to_string(),
                                            },
                                        }])
                                    }
                                },
                                cache_control: None,
                            })
                        }
                    })
                    .collect();
                let anthropic_role = match message.role {
                    Role::User => anthropic::Role::User,
                    Role::Assistant => anthropic::Role::Assistant,
                    Role::System => unreachable!("System role should never occur here"),
                };
                if let Some(last_message) = new_messages.last_mut()
                    && last_message.role == anthropic_role
                {
                    last_message.content.extend(anthropic_message_content);
                    continue;
                }

                // Mark the last segment of the message as cached
                if message.cache {
                    let cache_control_value = Some(anthropic::CacheControl {
                        cache_type: anthropic::CacheControlType::Ephemeral,
                    });
                    for message_content in anthropic_message_content.iter_mut().rev() {
                        match message_content {
                            anthropic::RequestContent::RedactedThinking { .. } => {
                                // Caching is not possible, fallback to next message
                            }
                            anthropic::RequestContent::Text { cache_control, .. }
                            | anthropic::RequestContent::Thinking { cache_control, .. }
                            | anthropic::RequestContent::Image { cache_control, .. }
                            | anthropic::RequestContent::ToolUse { cache_control, .. }
                            | anthropic::RequestContent::ToolResult { cache_control, .. } => {
                                *cache_control = cache_control_value;
                                break;
                            }
                        }
                    }
                }

                new_messages.push(anthropic::Message {
                    role: anthropic_role,
                    content: anthropic_message_content,
                });
            }
            Role::System => {
                if !system_message.is_empty() {
                    system_message.push_str("\n\n");
                }
                system_message.push_str(&message.string_contents());
            }
        }
    }

    anthropic::Request {
        model,
        messages: new_messages,
        max_tokens: max_output_tokens,
        system: if system_message.is_empty() {
            None
        } else {
            Some(anthropic::StringOrContents::String(system_message))
        },
        thinking: if request.thinking_allowed
            && let AnthropicModelMode::Thinking { budget_tokens } = mode
        {
            Some(anthropic::Thinking::Enabled { budget_tokens })
        } else {
            None
        },
        tools: request
            .tools
            .into_iter()
            .map(|tool| anthropic::Tool {
                name: tool.name,
                description: tool.description,
                input_schema: tool.input_schema,
            })
            .collect(),
        tool_choice: request.tool_choice.map(|choice| match choice {
            LanguageModelToolChoice::Auto => anthropic::ToolChoice::Auto,
            LanguageModelToolChoice::Any => anthropic::ToolChoice::Any,
            LanguageModelToolChoice::None => anthropic::ToolChoice::None,
        }),
        metadata: None,
        stop_sequences: Vec::new(),
        temperature: request.temperature.or(Some(default_temperature)),
        top_k: None,
        top_p: None,
    }
}

pub struct AnthropicEventMapper {
    tool_uses_by_index: HashMap<usize, RawToolUse>,
    usage: Usage,
    stop_reason: StopReason,
}

impl AnthropicEventMapper {
    pub fn new() -> Self {
        Self {
            tool_uses_by_index: HashMap::default(),
            usage: Usage::default(),
            stop_reason: StopReason::EndTurn,
        }
    }

    pub fn map_stream(
        mut self,
        events: Pin<Box<dyn Send + Stream<Item = Result<Event, AnthropicError>>>>,
    ) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>>
    {
        events.flat_map(move |event| {
            futures::stream::iter(match event {
                Ok(event) => self.map_event(event),
                Err(error) => vec![Err(error.into())],
            })
        })
    }

    pub fn map_event(
        &mut self,
        event: Event,
    ) -> Vec<Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
        match event {
            Event::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                ResponseContent::Text { text } => {
                    vec![Ok(LanguageModelCompletionEvent::Text(text))]
                }
                ResponseContent::Thinking { thinking } => {
                    vec![Ok(LanguageModelCompletionEvent::Thinking {
                        text: thinking,
                        signature: None,
                    })]
                }
                ResponseContent::RedactedThinking { data } => {
                    vec![Ok(LanguageModelCompletionEvent::RedactedThinking { data })]
                }
                ResponseContent::ToolUse { id, name, .. } => {
                    self.tool_uses_by_index.insert(
                        index,
                        RawToolUse {
                            id,
                            name,
                            input_json: String::new(),
                        },
                    );
                    Vec::new()
                }
            },
            Event::ContentBlockDelta { index, delta } => match delta {
                ContentDelta::TextDelta { text } => {
                    vec![Ok(LanguageModelCompletionEvent::Text(text))]
                }
                ContentDelta::ThinkingDelta { thinking } => {
                    vec![Ok(LanguageModelCompletionEvent::Thinking {
                        text: thinking,
                        signature: None,
                    })]
                }
                ContentDelta::SignatureDelta { signature } => {
                    vec![Ok(LanguageModelCompletionEvent::Thinking {
                        text: "".to_string(),
                        signature: Some(signature),
                    })]
                }
                ContentDelta::InputJsonDelta { partial_json } => {
                    if let Some(tool_use) = self.tool_uses_by_index.get_mut(&index) {
                        tool_use.input_json.push_str(&partial_json);

                        // Try to convert invalid (incomplete) JSON into
                        // valid JSON that serde can accept, e.g. by closing
                        // unclosed delimiters. This way, we can update the
                        // UI with whatever has been streamed back so far.
                        if let Ok(input) = serde_json::Value::from_str(
                            &partial_json_fixer::fix_json(&tool_use.input_json),
                        ) {
                            return vec![Ok(LanguageModelCompletionEvent::ToolUse(
                                LanguageModelToolUse {
                                    id: tool_use.id.clone().into(),
                                    name: tool_use.name.clone().into(),
                                    is_input_complete: false,
                                    raw_input: tool_use.input_json.clone(),
                                    input,
                                },
                            ))];
                        }
                    }
                    vec![]
                }
            },
            Event::ContentBlockStop { index } => {
                if let Some(tool_use) = self.tool_uses_by_index.remove(&index) {
                    let input_json = tool_use.input_json.trim();
                    let input_value = if input_json.is_empty() {
                        Ok(serde_json::Value::Object(serde_json::Map::default()))
                    } else {
                        serde_json::Value::from_str(input_json)
                    };
                    let event_result = match input_value {
                        Ok(input) => Ok(LanguageModelCompletionEvent::ToolUse(
                            LanguageModelToolUse {
                                id: tool_use.id.into(),
                                name: tool_use.name.into(),
                                is_input_complete: true,
                                input,
                                raw_input: tool_use.input_json.clone(),
                            },
                        )),
                        Err(json_parse_err) => {
                            Ok(LanguageModelCompletionEvent::ToolUseJsonParseError {
                                id: tool_use.id.into(),
                                tool_name: tool_use.name.into(),
                                raw_input: input_json.into(),
                                json_parse_error: json_parse_err.to_string(),
                            })
                        }
                    };

                    vec![event_result]
                } else {
                    Vec::new()
                }
            }
            Event::MessageStart { message } => {
                update_usage(&mut self.usage, &message.usage);
                vec![
                    Ok(LanguageModelCompletionEvent::UsageUpdate(convert_usage(
                        &self.usage,
                    ))),
                    Ok(LanguageModelCompletionEvent::StartMessage {
                        message_id: message.id,
                    }),
                ]
            }
            Event::MessageDelta { delta, usage } => {
                update_usage(&mut self.usage, &usage);
                if let Some(stop_reason) = delta.stop_reason.as_deref() {
                    self.stop_reason = match stop_reason {
                        "end_turn" => StopReason::EndTurn,
                        "max_tokens" => StopReason::MaxTokens,
                        "tool_use" => StopReason::ToolUse,
                        "refusal" => StopReason::Refusal,
                        _ => {
                            log::error!("Unexpected anthropic stop_reason: {stop_reason}");
                            StopReason::EndTurn
                        }
                    };
                }
                vec![Ok(LanguageModelCompletionEvent::UsageUpdate(
                    convert_usage(&self.usage),
                ))]
            }
            Event::MessageStop => {
                vec![Ok(LanguageModelCompletionEvent::Stop(self.stop_reason))]
            }
            Event::Error { error } => {
                vec![Err(error.into())]
            }
            _ => Vec::new(),
        }
    }
}

struct RawToolUse {
    id: String,
    name: String,
    input_json: String,
}

/// Updates usage data by preferring counts from `new`.
fn update_usage(usage: &mut Usage, new: &Usage) {
    if let Some(input_tokens) = new.input_tokens {
        usage.input_tokens = Some(input_tokens);
    }
    if let Some(output_tokens) = new.output_tokens {
        usage.output_tokens = Some(output_tokens);
    }
    if let Some(cache_creation_input_tokens) = new.cache_creation_input_tokens {
        usage.cache_creation_input_tokens = Some(cache_creation_input_tokens);
    }
    if let Some(cache_read_input_tokens) = new.cache_read_input_tokens {
        usage.cache_read_input_tokens = Some(cache_read_input_tokens);
    }
}

fn convert_usage(usage: &Usage) -> language_model::TokenUsage {
    language_model::TokenUsage {
        input_tokens: usage.input_tokens.unwrap_or(0),
        output_tokens: usage.output_tokens.unwrap_or(0),
        cache_creation_input_tokens: usage.cache_creation_input_tokens.unwrap_or(0),
        cache_read_input_tokens: usage.cache_read_input_tokens.unwrap_or(0),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum AuthMethodSelection {
    ApiKey,
    OAuth,
}

struct ConfigurationView {
    api_key_editor: Entity<Editor>,
    state: gpui::Entity<State>,
    load_credentials_task: Option<Task<()>>,
    auth_method: AuthMethodSelection,
    target_agent: ConfigurationViewTargetAgent,
}

impl ConfigurationView {
    const API_KEY_PLACEHOLDER: &'static str = "sk-ant-xxxxx...";
    const OAUTH_PLACEHOLDER: &'static str = "Refresh token from Claude subscription";

    fn new(
        state: gpui::Entity<State>,
        target_agent: ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn({
            let state = state.clone();
            async move |this, cx| {
                if let Some(task) = state
                    .update(cx, |state, cx| state.authenticate(cx))
                    .log_err()
                {
                    // We don't log an error, because "not signed in" is also an error.
                    let _ = task.await;
                }
                this.update(cx, |this, cx| {
                    // Set the auth_method based on what was loaded
                    if let Some(auth_method) = this.state.read(cx).auth_method.as_ref() {
                        this.auth_method = match auth_method {
                            AuthMethod::ApiKey => AuthMethodSelection::ApiKey,
                            AuthMethod::OAuth { .. } => AuthMethodSelection::OAuth,
                        };
                    }
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor: cx.new(|cx| {
                let mut editor = Editor::single_line(window, cx);
                editor.set_placeholder_text(Self::API_KEY_PLACEHOLDER, window, cx);
                editor
            }),
            state,
            load_credentials_task,
            auth_method: AuthMethodSelection::ApiKey,
            target_agent,
        }
    }

    fn save_credentials(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let input = self.api_key_editor.read(cx).text(cx).trim().to_string();
        if input.is_empty() {
            return;
        }

        // url changes can cause the editor to be displayed again
        self.api_key_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        let state = self.state.clone();
        let auth_method = self.auth_method;
        cx.spawn_in(window, async move |_, cx| match auth_method {
            AuthMethodSelection::ApiKey => {
                state
                    .update(cx, |state, cx| state.set_api_key(Some(input), cx))?
                    .await
            }
            AuthMethodSelection::OAuth => {
                state
                    .update(cx, |state, cx| state.set_refresh_token(input, cx))?
                    .await
            }
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn set_auth_method(
        &mut self,
        method: AuthMethodSelection,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.auth_method != method {
            self.auth_method = method;
            self.api_key_editor.update(cx, |editor, cx| {
                editor.set_text("", window, cx);
                editor.set_placeholder_text(
                    match method {
                        AuthMethodSelection::ApiKey => Self::API_KEY_PLACEHOLDER,
                        AuthMethodSelection::OAuth => Self::OAUTH_PLACEHOLDER,
                    },
                    window,
                    cx,
                );
            });
            cx.notify();
        }
    }

    fn reset_api_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |editor, cx| editor.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| {
                    state.auth_method = None;
                    state.set_api_key(None, cx)
                })?
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn render_credentials_editor(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let settings = ThemeSettings::get_global(cx);
        let text_style = TextStyle {
            color: cx.theme().colors().text,
            font_family: settings.ui_font.family.clone(),
            font_features: settings.ui_font.features.clone(),
            font_fallbacks: settings.ui_font.fallbacks.clone(),
            font_size: rems(0.875).into(),
            font_weight: settings.ui_font.weight,
            font_style: FontStyle::Normal,
            line_height: relative(1.3),
            white_space: WhiteSpace::Normal,
            ..Default::default()
        };
        EditorElement::new(
            &self.api_key_editor,
            EditorStyle {
                background: cx.theme().colors().editor_background,
                local_player: cx.theme().players().local(),
                text: text_style,
                ..Default::default()
            },
        )
    }

    fn should_render_editor(&self, cx: &mut Context<Self>) -> bool {
        !self.state.read(cx).is_authenticated()
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let env_var_set = self.state.read(cx).is_api_key_from_env();

        let state = self.state.read(cx);
        let is_using_oauth = !state.is_using_api_key() && state.is_authenticated();

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials...")).into_any()
        } else if self.should_render_editor(cx) {
            v_flex()
                .size_full()
                .on_action(cx.listener(Self::save_credentials))
                .child(Label::new(format!("To use {}, you need to authenticate.", match &self.target_agent {
                    ConfigurationViewTargetAgent::ZedAgent => "Zed's agent with Anthropic".into(),
                    ConfigurationViewTargetAgent::Other(agent) => agent.clone(),
                })))
                .child(
                    h_flex()
                        .w_full()
                        .my_2()
                        .child(
                            ToggleButtonGroup::single_row(
                                "auth-method-selector",
                                [
                                    ToggleButtonSimple::new("API Key", {
                                        let this = cx.entity().downgrade();
                                        move |_, window, cx| {
                                            if let Some(this) = this.upgrade() {
                                                this.update(cx, |this, cx| {
                                                    this.set_auth_method(AuthMethodSelection::ApiKey, window, cx);
                                                });
                                            }
                                        }
                                    }),
                                    ToggleButtonSimple::new("Claude Subscription", {
                                        let this = cx.entity().downgrade();
                                        move |_, window, cx| {
                                            if let Some(this) = this.upgrade() {
                                                this.update(cx, |this, cx| {
                                                    this.set_auth_method(AuthMethodSelection::OAuth, window, cx);
                                                });
                                            }
                                        }
                                    }),
                                ],
                            )
                            .selected_index(match self.auth_method {
                                AuthMethodSelection::ApiKey => 0,
                                AuthMethodSelection::OAuth => 1,
                            })
                            .style(ToggleButtonGroupStyle::Outlined)
                            .width(rems_from_px(180.))
                        )
                )
                .child(
                    v_flex()
                        .gap_1()
                        .child(
                            match self.auth_method {
                                AuthMethodSelection::ApiKey => {
                                    List::new()
                                        .child(
                                            InstructionListItem::new(
                                                "Create an API key by visiting",
                                                Some("Anthropic's settings"),
                                                Some("https://console.anthropic.com/settings/keys")
                                            )
                                        )
                                        .child(
                                            InstructionListItem::text_only("Paste your API key below (starts with 'sk-ant-')")
                                        )
                                }
                                AuthMethodSelection::OAuth => {
                                    List::new()
                                        .child(
                                            InstructionListItem::text_only("Use your Claude Pro or Max subscription")
                                        )
                                        .child(
                                            InstructionListItem::text_only("Paste your refresh token below")
                                        )
                                        .child(
                                            InstructionListItem::text_only("This allows usage through your subscription without separate API billing")
                                        )
                                }
                            }
                        )
                )
                .child(
                    h_flex()
                        .w_full()
                        .my_2()
                        .px_2()
                        .py_1()
                        .bg(cx.theme().colors().editor_background)
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .rounded_sm()
                        .child(self.render_credentials_editor(cx)),
                )
                .child(
                    Label::new(
                        format!("You can also assign the {API_KEY_ENV_VAR_NAME} environment variable and restart Zed."),
                    )
                    .size(LabelSize::Small)
                    .color(Color::Muted),
                )
                .into_any()
        } else {
            h_flex()
                .mt_1()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border)
                .bg(cx.theme().colors().background)
                .child(
                    h_flex()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(Label::new(if env_var_set {
                            format!("API key set in {API_KEY_ENV_VAR_NAME} environment variable")
                        } else if is_using_oauth {
                            "Authenticated with Claude Pro/Max subscription.".to_string()
                        } else {
                            let api_url = AnthropicLanguageModelProvider::api_url(cx);
                            if api_url == ANTHROPIC_API_URL {
                                "API key configured".to_string()
                            } else {
                                format!("API key configured for {}", truncate_and_trailoff(&api_url, 32))
                            }
                        })),
                )
                .child(
                    Button::new("reset-key", "Reset Key")
                        .label_size(LabelSize::Small)
                        .icon(Some(IconName::Trash))
                        .icon_size(IconSize::Small)
                        .icon_position(IconPosition::Start)
                        .disabled(env_var_set)
                        .when(env_var_set, |this| {
                            this.tooltip(Tooltip::text(format!("To reset your API key, unset the {API_KEY_ENV_VAR_NAME} environment variable.")))
                        })
                        .on_click(cx.listener(|this, _, window, cx| this.reset_api_key(window, cx))),
                )
                .into_any()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anthropic::AnthropicModelMode;
    use language_model::{LanguageModelRequestMessage, MessageContent};

    #[test]
    fn test_cache_control_only_on_last_segment() {
        let request = LanguageModelRequest {
            messages: vec![LanguageModelRequestMessage {
                role: Role::User,
                content: vec![
                    MessageContent::Text("Some prompt".to_string()),
                    MessageContent::Image(language_model::LanguageModelImage::empty()),
                    MessageContent::Image(language_model::LanguageModelImage::empty()),
                    MessageContent::Image(language_model::LanguageModelImage::empty()),
                    MessageContent::Image(language_model::LanguageModelImage::empty()),
                ],
                cache: true,
            }],
            thread_id: None,
            prompt_id: None,
            intent: None,
            mode: None,
            stop: vec![],
            temperature: None,
            tools: vec![],
            tool_choice: None,
            thinking_allowed: true,
        };

        let anthropic_request = into_anthropic(
            request,
            "claude-3-5-sonnet".to_string(),
            0.7,
            4096,
            AnthropicModelMode::Default,
        );

        assert_eq!(anthropic_request.messages.len(), 1);

        let message = &anthropic_request.messages[0];
        assert_eq!(message.content.len(), 5);

        assert!(matches!(
            message.content[0],
            anthropic::RequestContent::Text {
                cache_control: None,
                ..
            }
        ));
        for i in 1..3 {
            assert!(matches!(
                message.content[i],
                anthropic::RequestContent::Image {
                    cache_control: None,
                    ..
                }
            ));
        }

        assert!(matches!(
            message.content[4],
            anthropic::RequestContent::Image {
                cache_control: Some(anthropic::CacheControl {
                    cache_type: anthropic::CacheControlType::Ephemeral,
                }),
                ..
            }
        ));
    }
}
