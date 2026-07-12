# AeroML Frontend Blueprint & Implementation Plan

This document serves as the implementation plan for the AeroML web dashboard replacement. We are moving away from Streamlit and building a custom, aesthetic, high-fidelity frontend with a Bun-based backend.

---

## 🎨 Visual & Aesthetic Foundations

### 1. Color Palette
We will implement the specified color palette using custom CSS variables:
*   **Cotton (`#EDEBDD`)**: Primary typography, gridlines, container borders, and high-contrast vector lines.
*   **Cherry Red (`#810100`)**: Primary interactive accents, active states, and suction pressure curves.
*   **Maroon (`#630000`)**: Secondary accents, container headers, and compression pressure curves.
*   **Noir Black (`#1B1717`)**: Base canvas background, panel backdrops, and solid frames.

### 2. Typography
*   **Headings / Accent Text**: *Cinzel Decorative* / *Syne* for dramatic, premium industrial labels.
*   **Core UI / Numbers**: *Space Grotesk* for clean, legible numeric metrics.
*   **Terminal / Console Logs**: *JetBrains Mono* for retro, dithered ASCII and code blocks.

### 3. Layout Styles
We will support two distinct design modes matching the user's provided references:
*   **Landing Page (Slack/Pixel-Art Aesthetic)**: Monochrome dithered container backdrops, thick retro borders, pixelated dithered graphics, and terminal elements.
*   **Workbench Page (Industrial CAD Aesthetic)**: Solid Noir background, thin Cotton gridlines, neon-like accent line plots, technical readout dials, and simulated analog gauge bars.

---

## 🏗️ Architecture & Directories

We will follow the modular root directory structure:
```
AeroML/
├── frontend/
│   ├── index.html           # Landing page
│   ├── workbench.html       # Predictor/Optimizer control center
│   ├── style.css            # Common CSS system, grid, tokens
│   ├── landing.js           # ASCII wind-tunnel loop, landing logic
│   ├── workbench.js         # Interactive Canvas plotter, api integrations
│   └── assets/              # Static SVG designs/icons
├── backend/
│   ├── server.ts            # Hono-based static server + API gateway
│   ├── package.json         # Bun dependencies (Hono, Zod, Hono-Rate-Limit)
│   └── tsconfig.json        # TypeScript configuration
├── context.md               # Unified development context
└── FRONTEND_BLUEPRINT.md    # This plan
```

---

## 🏁 Phase-by-Phase Plan

We will implement the frontend incrementally, validating each phase before moving to the next.

### Phase 1: CSS Design Tokens & Backend Framework
*   **Goal**: Establish the structural grid, styling tokens, and the API gateway.
*   **Tasks**:
    1. Create `frontend/style.css` defining the CSS variables, custom typography imports, grid layouts, and custom buttons/borders.
    2. Set up `backend/package.json` for Bun using Hono and Zod.
    3. Implement `backend/server.ts` with static file serving, rate limiting, and parameter validation.

### Phase 2: Landing Page & ASCII Wind-Tunnel Loop
*   **Goal**: Create a high-motion landing experience.
*   **Tasks**:
    1. Create `frontend/index.html` structure with sections: Hero, Project Architecture (Pixel-Artsy), and Achieved Performance Stats.
    2. Write a JS canvas-rendered or text-rendered **ASCII wind-tunnel loop** in `frontend/landing.js` that programmatically bends airflow lines around an airfoil profile.
    3. Render real-time fluctuating parameters (flow velocity, Reynolds number, transient drag coefficients) in the corner of the hero section.

### Phase 3: Project Architecture Render (Pixel-Artsy)
*   **Goal**: Add the Slack/Jeep-style pixel-art section.
*   **Tasks**:
    1. Create CSS classes for 1-bit style dithering and retro-windows borders.
    2. Build the visual representation of our modular package architecture (`src/aeroml/` pipeline) using a dithered visual card layout.

### Phase 4: Workbench Layout & Interactive Canvas Plotter
*   **Goal**: Build the primary industrial workspace.
*   **Tasks**:
    1. Create `frontend/workbench.html` with dual predictor/optimizer panels, configuration sliders, and logs.
    2. Implement `frontend/workbench.js` to render the coordinate drafting board using `<canvas>` with grid overlays.
    3. Implement suction/compression pressure curve visualization using smooth cubic splines.

### Phase 5: API Integration & System Verification
*   **Goal**: Connect the frontend UI controls to the python inference runtimes.
*   **Tasks**:
    1. Expose endpoints in `backend/server.ts` that trigger prediction (`ForwardV3Predictor`) and optimization (`ReverseV3Designer`) via `Bun.spawn` sub-processes.
    2. Integrate the prediction panels in `workbench.js` with the API.
    3. Verify that uploading a custom `.dat` file or setting target inputs plots geometries and runs optimization loops successfully.
