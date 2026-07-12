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
  return "python";
}

const PYTHON_BIN = getPythonExecutable();

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
    // Spawn python subprocess bridging to our neural network predictor with root CWD
    const proc = Bun.spawn([
      PYTHON_BIN,
      "./scripts/api_bridge.py",
      "predict",
      "--file", tempPath,
      "--re", String(re),
      "--mach", String(mach),
    ], {
      cwd: PROJECT_ROOT
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      console.error("Predict Subprocess stderr:", stderr);
      return c.json(
        { error: "prediction_failed", message: "Prediction calculation error.", details: stderr },
        500
      );
    }

    const payload = JSON.parse(stdout);
    if (payload.error) {
      return c.json({ error: "prediction_failed", message: payload.error }, 400);
    }

    return c.json(payload);
  } catch (err: any) {
    console.error(err);
    return c.json({ error: "server_error", message: err.message }, 500);
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
    // Spawn python subprocess bridging to our reverse optimizer with root CWD
    const proc = Bun.spawn([
      PYTHON_BIN,
      "./scripts/api_bridge.py",
      "optimize",
      "--ldmax", String(ldmax),
      "--clmax", String(clmax),
      "--cdmin", String(cdmin),
      "--re", String(re),
      "--mach", String(mach),
      "--restarts", String(n_restarts),
      "--maxiter", String(opt_maxiter),
    ], {
      cwd: PROJECT_ROOT
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;

    if (exitCode !== 0) {
      console.error("Optimize Subprocess stderr:", stderr);
      return c.json(
        { error: "optimization_failed", message: "Optimization calculation error.", details: stderr },
        500
      );
    }

    const payload = JSON.parse(stdout);
    return c.json(payload);
  } catch (err: any) {
    console.error(err);
    return c.json({ error: "server_error", message: err.message }, 500);
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
  port: 8080,
  fetch: app.fetch,
};
