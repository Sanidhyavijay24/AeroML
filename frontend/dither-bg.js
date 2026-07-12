// -*- coding: utf-8 -*-
/**
 * @file dither-bg.js
 * @description Zero-dependency native WebGL background dithered wave simulation
 * @module frontend
 */

const ditherVertexShader = `
attribute vec2 a_position;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const ditherFragmentShader = `
precision highp float;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_waveSpeed;
uniform float u_waveFrequency;
uniform float u_waveAmplitude;
uniform vec3 u_waveColor;
uniform vec2 u_mousePos;
uniform float u_enableMouse;
uniform float u_mouseRadius;
uniform float u_colorNum;
uniform float u_pixelSize;

vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return vec4(1.79284291400159) - vec4(0.85373472095314) * r; }
vec2 fade(vec2 t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

float cnoise(vec2 P) {
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod289(Pi);
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;
  vec4 i = permute(permute(ix) + iy);
  vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0;
  vec4 gy = abs(gx) - 0.5;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;
  vec2 g00 = vec2(gx.x, gy.x);
  vec2 g10 = vec2(gx.y, gy.y);
  vec2 g01 = vec2(gx.z, gy.z);
  vec2 g11 = vec2(gx.w, gy.w);
  vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;
  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));
  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  return 2.3 * mix(n_x.x, n_x.y, fade_xy.y);
}

float fbm(vec2 p) {
  float value = 0.0;
  float amp = 1.0;
  float freq = u_waveFrequency;
  for (int i = 0; i < 4; i++) {
    value += amp * abs(cnoise(p));
    p *= freq;
    amp *= u_waveAmplitude;
  }
  return value;
}

float pattern(vec2 p) {
  vec2 p2 = p - u_time * u_waveSpeed;
  return fbm(p + fbm(p2));
}

float getBayerThreshold(vec2 coord) {
  float x = mod(coord.x, 8.0);
  float y = mod(coord.y, 8.0);
  float val = 0.0;
  float bx = floor(x / 4.0);
  float by = floor(y / 4.0);
  val += 1.0 * ((1.0 - bx) * (2.0 * by) + bx * (3.0 - 2.0 * by));
  bx = floor(mod(x, 4.0) / 2.0);
  by = floor(mod(y, 4.0) / 2.0);
  val += 4.0 * ((1.0 - bx) * (2.0 * by) + bx * (3.0 - 2.0 * by));
  bx = mod(x, 2.0);
  by = mod(y, 2.0);
  val += 16.0 * ((1.0 - bx) * (2.0 * by) + bx * (3.0 - 2.0 * by));
  return val / 64.0;
}

vec3 dither(vec2 uv, vec3 color) {
  vec2 scaledCoord = floor(uv * u_resolution / u_pixelSize);
  float threshold = getBayerThreshold(scaledCoord) - 0.25;
  float ditherStep = 1.0 / (u_colorNum - 1.0);
  color += threshold * ditherStep;
  color = clamp(color - 0.15, 0.0, 1.0);
  return floor(color * (u_colorNum - 1.0) + 0.5) / (u_colorNum - 1.0);
}

void main() {
  vec2 normalizedPixelSize = u_pixelSize / u_resolution;
  vec2 uvPixel = normalizedPixelSize * (floor(gl_FragCoord.xy / u_pixelSize) + 0.5);
  vec2 uv = uvPixel - 0.5;
  uv.x *= u_resolution.x / u_resolution.y;

  float f = pattern(uv);

  if (u_enableMouse > 0.5) {
    vec2 mouseNDC = (u_mousePos / u_resolution - 0.5) * vec2(1.0, -1.0);
    mouseNDC.x *= u_resolution.x / u_resolution.y;
    float dist = length(uv - mouseNDC);
    float effect = 1.0 - smoothstep(0.0, u_mouseRadius, dist);
    f -= 0.5 * effect;
  }

  vec3 col = mix(vec3(0.0), u_waveColor, f);
  vec3 ditheredCol = dither(gl_FragCoord.xy / u_resolution, col);
  gl_FragColor = vec4(ditheredCol, 1.0);
}
`;

(function () {
  "use strict";

  function createShader(gl, type, source) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      var log = gl.getShaderInfoLog(shader);
      console.error("Shader compile error:", log);
      showError("Shader compile error:\n" + log);
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  function showError(msg) {
    var div = document.createElement("div");
    div.style.cssText =
      "position:fixed;top:10px;left:10px;right:10px;background:rgba(201,16,16,0.95);" +
      "color:#EDEBDD;border:2px solid #EDEBDD;padding:15px;font-family:monospace;" +
      "font-size:12px;white-space:pre-wrap;z-index:99999;";
    div.textContent = "AeroML Shader Diagnostic:\n\n" + msg;
    document.body.appendChild(div);
  }

  function init() {
    var canvas = document.createElement("canvas");
    canvas.id = "dither-canvas";
    canvas.style.cssText =
      "position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:-1;pointer-events:none;";
    document.body.insertBefore(canvas, document.body.firstChild);

    var gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    if (!gl) {
      console.warn("WebGL not available.");
      return;
    }

    var vs = createShader(gl, gl.VERTEX_SHADER, ditherVertexShader);
    var fs = createShader(gl, gl.FRAGMENT_SHADER, ditherFragmentShader);
    if (!vs || !fs) return;

    var program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      showError("Program link error:\n" + gl.getProgramInfoLog(program));
      return;
    }

    // Attribute and uniform locations
    var posLoc = gl.getAttribLocation(program, "a_position");
    var uRes = gl.getUniformLocation(program, "u_resolution");
    var uTime = gl.getUniformLocation(program, "u_time");
    var uSpeed = gl.getUniformLocation(program, "u_waveSpeed");
    var uFreq = gl.getUniformLocation(program, "u_waveFrequency");
    var uAmp = gl.getUniformLocation(program, "u_waveAmplitude");
    var uColor = gl.getUniformLocation(program, "u_waveColor");
    var uMouse = gl.getUniformLocation(program, "u_mousePos");
    var uMouseEn = gl.getUniformLocation(program, "u_enableMouse");
    var uMouseR = gl.getUniformLocation(program, "u_mouseRadius");
    var uColorNum = gl.getUniformLocation(program, "u_colorNum");
    var uPixelSize = gl.getUniformLocation(program, "u_pixelSize");

    // Fullscreen quad buffer
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );

    var mouseX = 0, mouseY = 0;
    document.addEventListener("mousemove", function (e) {
      mouseX = e.clientX;
      mouseY = e.clientY;
    });

    var startTime = performance.now();
    var width = 0, height = 0;

    function resize() {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
    window.addEventListener("resize", resize);
    resize();

    function frame() {
      requestAnimationFrame(frame);
      var t = (performance.now() - startTime) / 1000.0;

      gl.clearColor(0.106, 0.09, 0.09, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(program);

      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(uRes, width, height);
      gl.uniform1f(uTime, t);
      gl.uniform1f(uSpeed, 0.05);
      gl.uniform1f(uFreq, 3.0);
      gl.uniform1f(uAmp, 0.3);
      gl.uniform3f(uColor, 0.5, 0.18, 0.18);
      gl.uniform2f(uMouse, mouseX, mouseY);
      gl.uniform1f(uMouseEn, 1.0);
      gl.uniform1f(uMouseR, 0.3);
      gl.uniform1f(uColorNum, 4.0);
      gl.uniform1f(uPixelSize, 2.0);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    frame();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
