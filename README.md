# Sparse-Radiance-Cascades-Experiment-webGPU

A WebGPU prototype for experimenting with **DDGI-like probes** plus **radiance cascades** for approximate global illumination.

## What this prototype does

- Renders a simple analytic scene (ground + two spheres + back wall) via GPU ray intersection.
- Builds a G-buffer with albedo/normal/depth.
- Runs two compute-driven probe cascades:
  - Cascade 0: coarse grid (global context)
  - Cascade 1: finer grid (local detail, seeded by cascade 0)
- Shades the final image by combining direct emissive lighting with probe-based indirect bounce.

This is intentionally compact and educational rather than physically exact.

## Run locally

Because browsers restrict module loading from `file://`, use a local server:

```bash
python3 -m http.server 4173
```

Then open <http://localhost:4173> in a WebGPU-capable browser (Chrome/Edge recent versions).
