# Why our 3D tools refuse to say "metres" until you prove it

*The one design decision in Open Reality that people ask about most. September 2026.*

Point a phone at a room, walk around, and a reconstruction pipeline can give
you back a 3D scene: camera path, dense geometry, detected objects. Ask an AI
assistant "how long is the couch?" and it will look at the scene, find two
points, compute a distance of 1.72, and answer "about 1.7 metres."

Nobody told it metres. The number is in an arbitrary unit, and the assistant
filled in the most plausible word. That is the failure mode this post is about,
and the reason the Open Reality MCP tools carry a small, stubborn rule: **a
length is metres only when the server says so, and the server only says so
after you have proven it.**

## Where the unit goes missing

Reconstruction from a single ordinary camera (no depth sensor, no LiDAR) is a
SLAM problem: simultaneous localization and mapping, meaning the software works
out where the camera was and what the world looks like from the video alone. A
plain camera cannot tell a large room filmed from far away from a small room
filmed up close. The geometry it produces is internally consistent, so ratios
are right, but the absolute scale is not observable. Everything comes out in
"relative units": the scene is correct up to one unknown multiplier.

Every monocular pipeline has this property. What differs is whether the
software admits it. A number without a unit is an invitation for the next
system in the chain, human or model, to invent one.

## The four places we enforce it

**1. The response envelope.** Every measurement route on the server returns
the number together with `units`, `scale_source`, `provenance`, and `degraded`.
For a fresh scan, `units` is `"relative"` and `scale_source` says why. The
MCP server (the small process that exposes the tools to Claude, Codex, or
Cursor) passes these fields through verbatim. There is a rule in the code
review checklist, not just the docs: never soften, never convert, never
summarize a refusal.

**2. The anchor.** Getting to metres takes one fact the model cannot invent: a
real distance. The `scene_anchor` tool takes two picked points in the scene and
the true distance between them (the width of a door, the length of a counter)
and computes the scene's scale factor. One measurement you trust rescales the
whole scene. From then on the same routes answer with `units: "m"` and
`scale_source` names the anchor that made it so.

**3. The gate.** Some outputs are metric by definition. A robot simulator scene
(Isaac Sim, in USD format) is one: a scene at the wrong scale is worse than no
scene, because a policy trained in it learns the wrong reach. So the Isaac
export is refused when there is no anchor. The refusal is a typed error that
reaches the model unedited, with the anchor flow as the suggested next step.
The agent's job is to relay it and offer the calibration, not to route around
it.

**4. The depth model that is not allowed to claim metres.** Inside the SLAM we
do run a monocular metric-depth network (Depth Anything V2, metric head). It
would be tempting to let it set absolute scale. We measured it: on real indoor
footage against motion-capture ground truth it is very consistent from frame to
frame, but carries a constant absolute bias of roughly sixteen percent. So it
is used only as a ratio between consecutive map segments, which cancels a
constant bias and removes scale drift along the walk. It corrects the shape of
the scene; it never gets to name the unit.

## Teaching the model the same rule

An MCP tool can label its output honestly and the model can still paraphrase
it away. So the package ships a skill file (a short instruction document the
assistant loads alongside the tools) that restates the doctrine in the model's
own terms:

> A length is metres ONLY when the response says `units: "m"`. `units:
> "relative"` means relative SLAM units: never call them metres, never convert.
> To unlock metres, run `scene_anchor` with two picked points and their real
> distance.

The same file carries the neighbouring rules that follow from the same idea.
The objects in a scene are a closed world: the list the server returns is the
only list, so the model does not invent a lamp because rooms usually have one.
Synthetic views are renders of the reconstruction, not photos, and are labelled
that way. Generated object meshes carry a `generated: true` flag. A degraded
scene is a state to report, not a failure to hide.

## What it looks like in practice

The demo in the repository README runs through it in four beats. Upload a
video. Ask for the desk-to-chair distance: the answer comes back labelled as
relative units, and the assistant says so. Give one real distance, run the
anchor. Ask again: now it is metres, and the assistant says where the scale
came from. Then export robot-training data, which is only possible because the
anchor exists.

None of this makes the reconstruction more accurate. It makes the system tell
the truth about how accurate it is, which is the property you need before you
would let an assistant write a measurement into an inspection report, or a
robotics team train on a scene it did not survey by hand.

## Try it

```bash
claude mcp add openreality -- npx -y openreality-mcp serve
npx -y openreality-mcp login
```

Then upload a clip of your own room and ask how long something is. It will
tell you it does not know the unit yet. That is the feature.

Source, self-hosting instructions, and the skill file:
[github.com/reality-opened/openreality](https://github.com/reality-opened/openreality).
