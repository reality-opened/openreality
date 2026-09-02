# CLI demo recording

`demo.tape` renders the README's terminal demo with
[VHS](https://github.com/charmbracelet/vhs): a real Claude Code session driving
the openreality MCP tools against the **offline simulator** (deterministic
backend data, no account, no GPU). The beats:

1. list the scans in the account
2. measure desk to chair: the result is honestly labeled **relative units**
3. give one real-world distance, calibrate, measure again: now **metres**
4. export robot-training data and list the zip contents on disk

## Re-render

```bash
# prerequisites: vhs, ttyd, ffmpeg on PATH; a logged-in `claude` CLI;
# `npm run build` done in mcp/ (falls back to the published npm package)
cd docs/demo
vhs demo.tape        # writes demo.gif and demo.mp4
```

Notes:

- `setup.sh` (sourced by the tape's hidden section) creates a throwaway Claude
  project in `/tmp/openreality-vhs-demo`, starts the simulator on port 8973,
  and pre-approves the openreality MCP server plus `ls`/`unzip` so the session
  runs prompt-free. It edits `~/.claude.json` only to mark that one directory
  trusted.
- On headless servers Chromium may need extra shared libraries
  (`ldd ~/.cache/rod/browser/*/chrome | grep "not found"`, fetch the debs, set
  `LD_LIBRARY_PATH`) and `VHS_NO_SANDBOX=true` where unprivileged user
  namespaces are restricted (Ubuntu 24.04 default).
- Claude's wording varies between renders; the backend data does not. Re-render
  until you like the take. Waits key on stable strings (scan id, "relative",
  tool names, zip entries), so a take either completes or fails loudly.
- The rendered gif is published as a GitHub release asset and referenced from
  the root README; it is not committed to the repo.
