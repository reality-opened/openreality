# Open Reality core

The Open Reality SLAM library: dense feed-forward monocular SLAM (the VGGT-SLAM
2.0 line, MIT-SPARK lineage) with metric anchoring, open-set object detection,
splat export, and the `vggt_slam` package the [../server](../server) broker
imports. BSD-2-Clause.

This is the `core/` component of the public `reality-opened/openreality`
monorepo, a curated mirror of the internal core repo, published for
self-hosting the Open Reality stack (see
[../server/docs/self-hosting.md](../server/docs/self-hosting.md)). The hosted product at
[open-reality.io](https://open-reality.io) runs the same library on a
commercially licensed backbone.

## The backbone is fetched, not bundled

`vggt_slam` consumes the `vggt` package namespace. This repo does not ship it:

```bash
git clone https://github.com/MIT-SPARK/VGGT_SPARK.git third_party/vggt
git -C third_party/vggt checkout 6e6e16107b88e8e76c751826af10d4295d87ecd2
pip install -e third_party/vggt
```

**Licensing:** the VGGT_SPARK code and the `facebook/VGGT-1B` weights (which
download automatically on first run) are **CC BY-NC 4.0: non-commercial use
only**, and the metric-anchor depth model (Depth-Anything-V2 Large) is also
CC BY-NC 4.0. You accept those terms from their owners directly; this
repository is BSD-2-Clause and redistributes none of them. For commercial use,
use the hosted service or obtain your own model licenses.

## Install

```bash
pip install torch==2.3.1 torchvision==0.18.1
pip install -r requirements.txt
pip install -e .
# plus the backbone clone above, and for loop detection:
git clone https://github.com/Dominic101/salad.git third_party/salad
pip install -e third_party/salad
```

Standalone SLAM over a video or image directory:

```bash
python main.py --help
```

Docs: [docs/overview.md](docs/overview.md), [docs/setup-and-running.md](docs/setup-and-running.md),
[docs/code-structure.md](docs/code-structure.md), [docs/conventions.md](docs/conventions.md).
Some docs reference the internal repo names; `reality-opened/server` corresponds
to the sibling [../server](../server) mirror (see [MIRROR.md](MIRROR.md)).
