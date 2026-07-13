// -*- coding: utf-8 -*-
/**
 * @file server.ts
 * @description Bun Hono static server and API gateway with validation & rate limiting
 * @module backend
 */

import { Hono } from "hono";
import { serveStatic } from "hono/bun";
import { z } from "zod";
import { existsSync } from "fs";
import { join } from "path";

const PROJECT_ROOT = join(import.meta.dir, "..");

function getPythonExecutable(): string {
  if (process.env.PYTHON_BIN) {
    return process.env.PYTHON_BIN;
  }
  const condaPaths = [
    "C:\\Users\\sanid\\.conda\\envs\\aeroml\\python.exe",
    "C:\\ProgramData\\miniconda3\\envs\\aeroml\\python.exe",
    "C:\\Users\\sanid\\AppData\\Local\\miniconda3\\envs\\aeroml\\python.exe",
    "C:\\Users\\sanid\\anaconda3\\envs\\aeroml\\python.exe"
  ];
  for (const path of condaPaths) {
    if (existsSync(path)) {
      return path;
    }
  }
  return "python3";
}

const PYTHON_BIN = getPythonExecutable();

/* ==========================================================================
   Persistent Python Inference Server Management & Watchdog
   ========================================================================== */
const INFERENCE_PORT = Number(process.env.INFERENCE_PORT) || 8500;
let pythonProcess: any = null;
let restartAttempts = 0;
const MAX_RESTART_ATTEMPTS = 5;
const RESTART_COOLDOWN_MS = 5000;

function startInferenceServer() {
  console.log(`[Hono Gateway] Launching persistent Python inference server on port ${INFERENCE_PORT}...`);
  const cmd = [
    PYTHON_BIN,
    "./scripts/inference_server.py"
  ];
  
  pythonProcess = Bun.spawn(cmd, {
    cwd: PROJECT_ROOT,
    env: {
      ...process.env,
      INFERENCE_PORT: String(INFERENCE_PORT),
      TF_CPP_MIN_LOG_LEVEL: "3",
      absl_minloglevel: "3",
      TF_ENABLE_ONEDNN_OPTS: "0"
    },
    stdout: "inherit",
    stderr: "inherit"
  });

  pythonProcess.exited.then((code: number) => {
    console.error(`[Hono Gateway] Watchdog: Python process exited with code ${code}.`);
    pythonProcess = null;
    
    if (restartAttempts < MAX_RESTART_ATTEMPTS) {
      restartAttempts++;
      const nextDelay = RESTART_COOLDOWN_MS * Math.pow(1.5, restartAttempts - 1);
      console.log(`[Hono Gateway] Watchdog: Scheduling restart attempt ${restartAttempts}/${MAX_RESTART_ATTEMPTS} in ${(nextDelay / 1000).toFixed(1)}s...`);
      setTimeout(() => {
        if (!pythonProcess) {
          startInferenceServer();
        }
      }, nextDelay);
    } else {
      console.error("[Hono Gateway] Watchdog: Max python subprocess restart attempts exceeded. Server will remain offline.");
    }
  });
}

// Reset restart counters on a successful connection to prevent permanent failure
async function verifyServerHeartbeat() {
  try {
    const res = await fetch(`http://127.0.0.1:${INFERENCE_PORT}/health`, {
      signal: AbortSignal.timeout(800)
    });
    if (res.status === 200 || res.status === 503) {
      if (restartAttempts > 0) {
        console.log("[Hono Gateway] Heartbeat: Server alive, resetting watchdog counter.");
        restartAttempts = 0;
      }
    }
  } catch (e) {
    // Ignore error
  }
}
setInterval(verifyServerHeartbeat, 10000);

async function checkHealth(): Promise<{ status: "ready" | "starting" | "error" | "offline"; message?: string }> {
  try {
    const res = await fetch(`http://127.0.0.1:${INFERENCE_PORT}/health`, {
      signal: AbortSignal.timeout(1000)
    });
    if (res.status === 200) {
      return { status: "ready" };
    }
    if (res.status === 503) {
      return { status: "starting" };
    }
    const data: any = await res.json().catch(() => ({ message: "Unknown inference server status." }));
    return { status: "error", message: data.message || "Failed to load models." };
  } catch (err) {
    return { status: "offline" };
  }
}

// Initialize persistent inference backend process
startInferenceServer();

process.on("exit", () => {
  if (pythonProcess) {
    console.log("[Hono Gateway] Terminating persistent Python server...");
    pythonProcess.kill();
  }
});

const app = new Hono();

/* ==========================================================================
   Rate Limiting Middleware (IP & User Dual-Layer)
   ========================================================================== */
interface RateLimitRecord {
  count: number;
  resetTime: number;
}

class MemoryRateLimiter {
  private store = new Map<string, RateLimitRecord>();

  constructor(private limit: number, private windowMs: number) {}

  public check(key: string): { allowed: boolean; limit: number; remaining: number; reset: number } {
    const now = Date.now();
    let record = this.store.get(key);

    if (!record || now > record.resetTime) {
      record = {
        count: 0,
        resetTime: now + this.windowMs,
      };
    }

    record.count++;
    this.store.set(key, record);

    const remaining = Math.max(0, this.limit - record.count);
    const allowed = record.count <= this.limit;
    const resetSeconds = Math.ceil((record.resetTime - now) / 1000);

    return {
      allowed,
      limit: this.limit,
      remaining,
      reset: resetSeconds,
    };
  }
}

// IP layer: 100 requests per 15 minutes
const ipLimiter = new MemoryRateLimiter(100, 15 * 60 * 1000);
// User layer: 1000 requests per hour
const userLimiter = new MemoryRateLimiter(1000, 60 * 60 * 1000);

app.use("/api/*", async (c, next) => {
  const ip = c.req.header("x-forwarded-for") || "127.0.0.1";
  const userId = c.req.header("x-user-id") || ip;

  const ipResult = ipLimiter.check(`ip:${ip}`);
  const userResult = userLimiter.check(`user:${userId}`);

  c.header("X-RateLimit-Limit", String(ipResult.limit));
  c.header("X-RateLimit-Remaining", String(Math.min(ipResult.remaining, userResult.remaining)));
  c.header("X-RateLimit-Reset", String(Math.max(ipResult.reset, userResult.reset)));

  if (!ipResult.allowed || !userResult.allowed) {
    c.status(429);
    c.header("Retry-After", String(Math.max(ipResult.reset, userResult.reset)));
    return c.json({
      error: "rate_limit_exceeded",
      message: "Too many requests. Please try again later.",
    });
  }

  await next();
});

/* ==========================================================================
   Validation Schemas (Zod Strict)
   ========================================================================== */
const predictSchema = z.object({
  re: z.number().min(1e4).max(1e8),
  mach: z.number().min(0.0).max(1.0),
}).strict();

const optimizeSchema = z.object({
  ldmax: z.number().min(0),
  clmax: z.number().min(0),
  cdmin: z.number().min(0),
  re: z.number().min(1e4).max(1e8),
  mach: z.number().min(0.0).max(1.0),
  n_restarts: z.number().int().min(1).max(32).optional(),
  opt_maxiter: z.number().int().min(1).max(100).optional(),
}).strict();

/* ==========================================================================
   API Routes
   ========================================================================== */

/**
 * POST /api/predict
 * Uploads an airfoil coordinates file and queries predictions from forward model.
 */
app.post("/api/predict", async (c) => {
  const health = await checkHealth();
  if (health.status !== "ready") {
    c.status(503);
    return c.json({
      error: "service_unavailable",
      message: health.status === "starting"
        ? "Surrogate model server is still starting up, please try again shortly."
        : health.status === "error"
        ? `Model server failed to load: ${health.message}`
        : "Model server is currently offline. Attempting watchdog auto-restart."
    });
  }

  const body = await c.req.parseBody();
  const file = body["file"] as File;
  const reStr = body["re"];
  const machStr = body["mach"];

  if (!file || !reStr || !machStr) {
    return c.json(
      { error: "missing_fields", message: "File, re, and mach parameters are required." },
      400
    );
  }

  const re = parseFloat(String(reStr));
  const mach = parseFloat(String(machStr));

  // Schema check
  const validation = predictSchema.safeParse({ re, mach });
  if (!validation.success) {
    return c.json({ error: "validation_failed", details: validation.error.flatten() }, 400);
  }

  // Save the uploaded dat file to a temporary location using absolute path
  const tempPath = join(PROJECT_ROOT, `temp_${Date.now()}_predict.dat`);
  const fileBytes = await file.arrayBuffer();
  await Bun.write(tempPath, fileBytes);

  try {
    const res = await fetch(`http://127.0.0.1:${INFERENCE_PORT}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_path: tempPath,
        re,
        mach
      })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: "Unknown inference server error." }));
      throw new Error(err.message || "CFD surrogate solver engine runtime error.");
    }

    const payload = await res.json();
    return c.json(payload);
  } catch (err: any) {
    console.error(err);
    return c.json({ error: "prediction_failed", message: err.message }, 500);
  } finally {
    // Cleanup temporary file
    const f = Bun.file(tempPath);
    if (await f.exists()) {
      await f.delete();
    }
  }
});

/**
 * POST /api/optimize
 * Accepts design goals and runs latent-space optimizer algorithm.
 */
app.post("/api/optimize", async (c) => {
  const health = await checkHealth();
  if (health.status !== "ready") {
    c.status(503);
    return c.json({
      error: "service_unavailable",
      message: health.status === "starting"
        ? "Surrogate model server is still starting up, please try again shortly."
        : health.status === "error"
        ? `Model server failed to load: ${health.message}`
        : "Model server is currently offline. Attempting watchdog auto-restart."
    });
  }

  let rawBody: any;
  try {
    rawBody = await c.req.json();
  } catch (e) {
    return c.json({ error: "invalid_json", message: "Body must be valid JSON." }, 400);
  }

  // Schema check
  const validation = optimizeSchema.safeParse(rawBody);
  if (!validation.success) {
    return c.json({ error: "validation_failed", details: validation.error.flatten() }, 400);
  }

  const { ldmax, clmax, cdmin, re, mach, n_restarts = 8, opt_maxiter = 35 } = validation.data;

  try {
    const res = await fetch(`http://127.0.0.1:${INFERENCE_PORT}/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ldmax,
        clmax,
        cdmin,
        re,
        mach,
        restarts: n_restarts,
        maxiter: opt_maxiter
      })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ message: "Unknown inference server error." }));
      throw new Error(err.message || "Optimization solver runtime execution error.");
    }

    const payload = await res.json();
    return c.json(payload);
  } catch (err: any) {
    console.error(err);
    return c.json({ error: "optimization_failed", message: err.message }, 500);
  }
});

/* ==========================================================================
   Static Front-End Serving Routing
   ========================================================================== */
app.get("/", serveStatic({ path: "../frontend/index.html" }));
app.get("/workbench", serveStatic({ path: "../frontend/workbench.html" }));
app.get("/*", serveStatic({ root: "../frontend" }));

// Launch Hono server
export default {
  port: Number(process.env.PORT) || 8080,
  fetch: app.fetch,
};
