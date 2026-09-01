/**
 * Public module surface for programmatic embedding: the cordis services, the
 * tool-namespace plugins, the assembly, and the plain broker primitives the
 * CLI subcommands use.
 */
export { createApp, featurePlugins, SERVER_INSTRUCTIONS } from './app.js';
export { Broker } from './services/broker.js';
export { Mcp, type McpConfig, type ToolConfig } from './services/mcp.js';
export * as workspaceTools from './plugins/workspace.js';
export * as sceneTools from './plugins/scene.js';
export * as agentTools from './plugins/agent.js';
export * as exportTools from './plugins/export.js';
export * as sceneResources from './plugins/resources.js';
export { loadConfig, DEFAULT_BROKER, DEFAULT_LOGIN_URL, type Config } from './config.js';
export { Session } from './session.js';
export { ApiClient, ApiError } from './http.js';
