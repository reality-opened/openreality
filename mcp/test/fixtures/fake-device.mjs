// A stand-in hardware device for test/hardware*.test.ts: an MCP server over stdio that serves
// the tool list and instructions captured from the real SO-101 arm server
// (artist-arm-tools.json) and answers calls with scripted behaviour, so the bridge is tested
// against the device's real schemas without Python or hardware.
//   move_joints: reports progress "moving", then waits for stop and answers status "halted"
//   stop, arm_status, release_torque (refuses without confirm), look (an image)
//   any other tool: echoes { status: "reached", tool, received: <arguments> }
// Env: FAKE_DEVICE_PIDFILE (write the pid there), FAKE_DEVICE_TAG (echoed by arm_status).
import { readFileSync, writeFileSync } from 'node:fs';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';

const captured = JSON.parse(readFileSync(new URL('./artist-arm-tools.json', import.meta.url), 'utf8'));
if (process.env.FAKE_DEVICE_PIDFILE) writeFileSync(process.env.FAKE_DEVICE_PIDFILE, String(process.pid));

const server = new Server(
  { name: captured.server.name, version: '0.0.0-fake' },
  { capabilities: { tools: {} }, instructions: captured.instructions },
);

let releaseMove;
const structured = (value) => ({ content: [{ type: 'text', text: JSON.stringify(value) }], structuredContent: value });

server.setRequestHandler(ListToolsRequestSchema, () => ({ tools: captured.tools }));
server.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
  const { name, arguments: args = {} } = request.params;
  switch (name) {
    case 'arm_status':
      return structured({ busy: releaseMove !== undefined, tag: process.env.FAKE_DEVICE_TAG ?? null });
    case 'move_joints': {
      const stopped = new Promise((resolve) => {
        releaseMove = resolve;
      });
      const token = request.params._meta?.progressToken;
      if (token !== undefined) {
        await extra.sendNotification({
          method: 'notifications/progress',
          params: { progressToken: token, progress: 0, message: 'moving' },
        });
      }
      await stopped;
      releaseMove = undefined;
      return structured({ status: 'halted', received: args });
    }
    case 'stop': {
      const wasMoving = releaseMove !== undefined;
      releaseMove?.();
      return structured({ stopped: String(wasMoving) });
    }
    case 'release_torque':
      return args.confirm === true
        ? structured({ released: args.which ?? 'gripper' })
        : { content: [{ type: 'text', text: captured.replies.release_torque_unconfirmed }], isError: true };
    case 'look':
      return { content: [{ type: 'image', data: captured.replies.look_png_base64, mimeType: 'image/png' }] };
    default:
      return structured({ status: 'reached', tool: name, received: args });
  }
});

await server.connect(new StdioServerTransport());
