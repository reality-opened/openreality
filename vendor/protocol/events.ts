import type {
  AgentAction,
  AgentFinding,
  AgentJobEvent,
  AgentState,
  AgentTaskEvent,
  AgentThought,
  AgentToolEvent,
  AgentUICommand,
  AgentUIResult,
  DebugDetectRequest,
  DebugDetectResults,
  DetectionDescriptionRequest,
  DetectionDescriptionResult,
  DetectionPartialResult,
  DetectionPreview,
  SceneReport,
  SLAMUpdate,
} from './types';

export const CLIENT_EVENTS = {
  FRAME: 'frame',
  /**
   * @deprecated No server-side handler exists — SLAM auto-starts on the first `frame` event
   * (server/app.py:4057-4066). Kept for wire compatibility; do not gate behavior on it.
   */
  START_SLAM: 'start_slam',
  STOP_SLAM: 'stop_slam',
  PLACE_BEACON: 'place_beacon',
  CLEAR_BEACONS: 'clear_beacons',
  SET_DETECTION_QUERIES: 'set_detection_queries',
  GET_DETECTION_PREVIEW: 'get_detection_preview',
  /** Fine-grained "click-in" description of one detection (server/server/app.py:4213). */
  DESCRIBE_DETECTION: 'describe_detection',
  /** Production detection pipeline + per-frame diagnostics (server/server/app.py:4299). */
  DEBUG_DETECT: 'debug_detect',
  GET_GLOBAL_MAP: 'get_global_map',
  GET_SCENE_REPORT: 'get_scene_report',
  AGENT_CHAT: 'agent_chat',
  AGENT_SET_GOAL: 'agent_set_goal',
  AGENT_TOGGLE: 'agent_toggle',
  SET_RECONSTRUCTION_ONLY: 'set_reconstruction_only',
  GET_AGENT_STATE: 'get_agent_state',
  AGENT_UI_RESULT: 'agent_ui_result',
} as const;

export const SERVER_EVENTS = {
  CONNECTED: 'connected',
  SLAM_UPDATE: 'slam_update',
  GLOBAL_MAP: 'global_map',
  SLAM_RESET: 'slam_reset',
  SLAM_STOPPED: 'slam_stopped',
  SESSION_TIMEOUT: 'session_timeout',
  FRAME_RECEIVED: 'frame_received',
  BEACON_QUEUED: 'beacon_queued',
  DETECTION_PREVIEW: 'detection_preview',
  DETECTION_PARTIAL: 'detection_partial',
  /** Answer to `describe_detection` (server/server/app.py:4225-4275). */
  DETECTION_DESCRIPTION: 'detection_description',
  /** Answer to `debug_detect` (server/server/app.py:4302-4324). */
  DEBUG_DETECT_RESULTS: 'debug_detect_results',
  /** Generic server-side error envelope — see {@link ErrorPayload}. */
  ERROR: 'error',
  SCENE_REPORT_UPDATE: 'scene_report_update',
  SCENE_REPORT_READY: 'scene_report_ready',
  AGENT_THOUGHT: 'agent_thought',
  AGENT_ACTION: 'agent_action',
  AGENT_FINDING: 'agent_finding',
  AGENT_STATE: 'agent_state',
  AGENT_UI_COMMAND: 'agent_ui_command',
  AGENT_TOOL_EVENT: 'agent_tool_event',
  AGENT_TASK_EVENT: 'agent_task_event',
  AGENT_JOB_EVENT: 'agent_job_event',
} as const;

/**
 * Where a `frame` event's pixels came from. Optional provenance only — the server's
 * `handle_frame` reads `image` (+ `timestamp`) and ignores unknown keys, so senders that
 * never set it stay wire-identical.
 *
 * `glasses` = a paired wearable relayed through a companion app (e.g. Meta Ray-Ban via the
 * Wearables Device Access Toolkit); the phone is the transport, not the camera.
 */
export type CaptureSource = 'browser' | 'phone' | 'glasses';

export interface FramePayload {
  image: string;
  timestamp: number;
  source?: CaptureSource;
  /** Free-form hardware label for dataset provenance, e.g. `ray-ban-meta-gen2`. */
  device_model?: string;
}

export interface PlaceBeaconPayload {
  beacon_id: number;
  frame_number: number;
}

export interface DetectionQueriesPayload {
  queries: string[];
}

export interface DetectionPreviewRequest {
  submap_id: number;
  frame_idx: number;
  query: string;
}

export interface AgentChatPayload {
  message: string;
}

export interface AgentSetGoalPayload {
  goal: string;
  initial_queries: string[];
}

export interface AgentTogglePayload {
  enabled: boolean;
}

export interface ReconstructionOnlyPayload {
  enabled: boolean;
}

export interface ConnectedPayload {
  status?: string;
  [key: string]: unknown;
}

export interface FrameReceivedPayload {
  status: string;
}

export interface BeaconQueuedPayload {
  beacon_id?: number;
  [key: string]: unknown;
}

/**
 * Generic server-side error envelope, emitted on the `error` event.
 *
 * `'frame_too_large'` is the ONLY code emitted today: `handle_frame` rejects a base64 frame
 * longer than `MAX_FRAME_B64_LEN` and drops it (server/server/app.py:4050-4052). Typed as an
 * open string so a future server-side code does not silently break this contract.
 */
export interface ErrorPayload {
  error: string;
}

export interface ClientToServerEvents {
  [CLIENT_EVENTS.FRAME]: FramePayload;
  /**
   * @deprecated No server-side handler exists — SLAM auto-starts on the first `frame` event
   * (server/app.py:4057-4066). Kept for wire compatibility; do not gate behavior on it.
   */
  [CLIENT_EVENTS.START_SLAM]: undefined;
  [CLIENT_EVENTS.STOP_SLAM]: undefined;
  [CLIENT_EVENTS.PLACE_BEACON]: PlaceBeaconPayload;
  [CLIENT_EVENTS.CLEAR_BEACONS]: undefined;
  [CLIENT_EVENTS.SET_DETECTION_QUERIES]: DetectionQueriesPayload;
  [CLIENT_EVENTS.GET_DETECTION_PREVIEW]: DetectionPreviewRequest;
  [CLIENT_EVENTS.DESCRIBE_DETECTION]: DetectionDescriptionRequest;
  [CLIENT_EVENTS.DEBUG_DETECT]: DebugDetectRequest;
  [CLIENT_EVENTS.GET_GLOBAL_MAP]: undefined;
  [CLIENT_EVENTS.GET_SCENE_REPORT]: undefined;
  [CLIENT_EVENTS.AGENT_CHAT]: AgentChatPayload;
  [CLIENT_EVENTS.AGENT_SET_GOAL]: AgentSetGoalPayload;
  [CLIENT_EVENTS.AGENT_TOGGLE]: AgentTogglePayload;
  [CLIENT_EVENTS.SET_RECONSTRUCTION_ONLY]: ReconstructionOnlyPayload;
  [CLIENT_EVENTS.GET_AGENT_STATE]: undefined;
  [CLIENT_EVENTS.AGENT_UI_RESULT]: AgentUIResult;
}

export interface ServerToClientEvents {
  [SERVER_EVENTS.CONNECTED]: ConnectedPayload;
  [SERVER_EVENTS.SLAM_UPDATE]: SLAMUpdate;
  [SERVER_EVENTS.GLOBAL_MAP]: SLAMUpdate;
  [SERVER_EVENTS.SLAM_RESET]: undefined;
  [SERVER_EVENTS.SLAM_STOPPED]: undefined;
  [SERVER_EVENTS.SESSION_TIMEOUT]: undefined;
  [SERVER_EVENTS.FRAME_RECEIVED]: FrameReceivedPayload;
  [SERVER_EVENTS.BEACON_QUEUED]: BeaconQueuedPayload;
  [SERVER_EVENTS.DETECTION_PREVIEW]: DetectionPreview;
  [SERVER_EVENTS.DETECTION_PARTIAL]: DetectionPartialResult;
  [SERVER_EVENTS.DETECTION_DESCRIPTION]: DetectionDescriptionResult;
  [SERVER_EVENTS.DEBUG_DETECT_RESULTS]: DebugDetectResults;
  [SERVER_EVENTS.ERROR]: ErrorPayload;
  [SERVER_EVENTS.SCENE_REPORT_UPDATE]: SceneReport;
  [SERVER_EVENTS.SCENE_REPORT_READY]: SceneReport;
  [SERVER_EVENTS.AGENT_THOUGHT]: AgentThought;
  [SERVER_EVENTS.AGENT_ACTION]: AgentAction;
  [SERVER_EVENTS.AGENT_FINDING]: AgentFinding;
  [SERVER_EVENTS.AGENT_STATE]: AgentState;
  [SERVER_EVENTS.AGENT_UI_COMMAND]: AgentUICommand;
  [SERVER_EVENTS.AGENT_TOOL_EVENT]: AgentToolEvent;
  [SERVER_EVENTS.AGENT_TASK_EVENT]: AgentTaskEvent;
  [SERVER_EVENTS.AGENT_JOB_EVENT]: AgentJobEvent;
}

export type ClientEventName = typeof CLIENT_EVENTS[keyof typeof CLIENT_EVENTS];
export type ServerEventName = typeof SERVER_EVENTS[keyof typeof SERVER_EVENTS];
