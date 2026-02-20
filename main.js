const canvas = document.querySelector('#gpu-canvas');
const statusEl = document.querySelector('#status');

const PROBE_CASCADES = [
  { size: 16, spacing: 1.5 },
  { size: 32, spacing: 0.75 },
];

const COMMON_WGSL = /* wgsl */ `
struct Camera {
  viewProjInv : mat4x4f,
  cameraPos : vec4f,
  timeResolution : vec4f, // x=time, y=width, z=height
};

@group(0) @binding(0) var<uniform> camera : Camera;

fn hash12(p: vec2f) -> f32 {
  let h = dot(p, vec2f(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}

fn sceneAlbedo(p: vec3f) -> vec3f {
  var c = vec3f(0.7, 0.72, 0.76);
  if (length(p - vec3f(-1.2, 0.55, 0.4)) < 0.58) {
    c = vec3f(0.85, 0.25, 0.2);
  }
  if (length(p - vec3f(1.1, 0.45, -0.4)) < 0.45) {
    c = vec3f(0.25, 0.65, 0.95);
  }
  if (p.z > 2.9) {
    c = vec3f(0.35, 0.4, 0.5);
  }
  return c;
}

fn emissiveLight(p: vec3f) -> vec3f {
  let pulse = 0.5 + 0.5 * sin(camera.timeResolution.x * 1.2);
  let l0 = vec3f(0.0, 2.3, -0.6 + pulse * 0.4);
  let l1 = vec3f(-2.3, 1.8, 1.6);
  let d0 = max(length(p - l0), 0.25);
  let d1 = max(length(p - l1), 0.25);
  let c0 = vec3f(11.0, 8.5, 5.4) / (d0 * d0);
  let c1 = vec3f(3.8, 8.2, 11.4) / (d1 * d1);
  return c0 + c1;
}

fn sceneIntersect(ro: vec3f, rd: vec3f) -> vec4f {
  var hitT = 1e9;
  var normal = vec3f(0.0);

  if (rd.y < -0.001) {
    let tPlane = (-ro.y) / rd.y;
    if (tPlane > 0.0 && tPlane < hitT) {
      hitT = tPlane;
      normal = vec3f(0.0, 1.0, 0.0);
    }
  }

  let s0 = vec3f(-1.2, 0.55, 0.4);
  let s1 = vec3f(1.1, 0.45, -0.4);
  let r0 = 0.58;
  let r1 = 0.45;

  let oc0 = ro - s0;
  let b0 = dot(oc0, rd);
  let c0 = dot(oc0, oc0) - r0 * r0;
  let h0 = b0 * b0 - c0;
  if (h0 > 0.0) {
    let t = -b0 - sqrt(h0);
    if (t > 0.0 && t < hitT) {
      hitT = t;
      normal = normalize((ro + rd * t) - s0);
    }
  }

  let oc1 = ro - s1;
  let b1 = dot(oc1, rd);
  let c1 = dot(oc1, oc1) - r1 * r1;
  let h1 = b1 * b1 - c1;
  if (h1 > 0.0) {
    let t = -b1 - sqrt(h1);
    if (t > 0.0 && t < hitT) {
      hitT = t;
      normal = normalize((ro + rd * t) - s1);
    }
  }

  if (abs(rd.z) > 0.001) {
    let tWall = (3.0 - ro.z) / rd.z;
    let p = ro + rd * tWall;
    if (tWall > 0.0 && tWall < hitT && p.y > 0.0 && abs(p.x) < 3.2) {
      hitT = tWall;
      normal = vec3f(0.0, 0.0, -1.0);
    }
  }

  if (hitT < 1e8) {
    return vec4f(normal, hitT);
  }
  return vec4f(0.0, 0.0, 0.0, -1.0);
}

fn worldRay(uv: vec2f) -> vec3f {
  let ndc = vec4f(uv * 2.0 - 1.0, 1.0, 1.0);
  let world = camera.viewProjInv * ndc;
  return normalize(world.xyz / world.w - camera.cameraPos.xyz);
}
`;

const GBUFFER_SHADER = /* wgsl */ `
${COMMON_WGSL}

struct VSOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var out : VSOut;
  let p = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  out.position = vec4f(p[vi], 0.0, 1.0);
  out.uv = p[vi] * 0.5 + 0.5;
  return out;
}

struct GOut {
  @location(0) albedoDepth : vec4f,
  @location(1) normalHit : vec4f,
};

@fragment
fn fsMain(in: VSOut) -> GOut {
  let ro = camera.cameraPos.xyz;
  let rd = worldRay(in.uv);
  let hit = sceneIntersect(ro, rd);
  var out : GOut;
  if (hit.w > 0.0) {
    let pos = ro + rd * hit.w;
    out.albedoDepth = vec4f(sceneAlbedo(pos), hit.w);
    out.normalHit = vec4f(hit.xyz * 0.5 + 0.5, 1.0);
  } else {
    out.albedoDepth = vec4f(0.0, 0.0, 0.0, -1.0);
    out.normalHit = vec4f(0.0, 0.0, 0.0, 0.0);
  }
  return out;
}
`;

const PROBE_SHADER = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(1) var coarseIn : texture_2d<f32>;
@group(0) @binding(2) var probeOut : texture_storage_2d<rgba16float, write>;

struct ProbeParams {
  size : u32,
  spacing : f32,
  useCoarse : u32,
  pad : u32,
};

@group(0) @binding(3) var<uniform> probeParams : ProbeParams;

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid : vec3u) {
  if (gid.x >= probeParams.size || gid.y >= probeParams.size) {
    return;
  }

  let coord = vec2u(gid.xy);
  let dim = f32(probeParams.size);
  let uv = (vec2f(coord) + 0.5) / dim;
  let worldXZ = (uv - 0.5) * (probeParams.spacing * dim * 0.85);
  let probePos = vec3f(worldXZ.x, 0.8, worldXZ.y);

  var accum = vec3f(0.0);
  let sampleCount = 12u;
  for (var i = 0u; i < sampleCount; i = i + 1u) {
    let fi = f32(i);
    let a = fi / f32(sampleCount);
    let ang = 6.2831853 * (a + hash12(vec2f(fi, uv.x + uv.y)) * 0.11);
    let lift = mix(0.2, 0.95, fract(a * 2.31 + uv.x));
    let dir = normalize(vec3f(cos(ang), lift, sin(ang)));

    let hit = sceneIntersect(probePos, dir);
    if (hit.w > 0.0) {
      let hp = probePos + dir * hit.w;
      let n = hit.xyz;
      let bounce = max(dot(n, normalize(vec3f(0.5, 1.0, -0.2))), 0.0);
      var incoming = sceneAlbedo(hp) * emissiveLight(hp) * (0.22 + 0.78 * bounce);

      if (probeParams.useCoarse == 1u) {
        let cDim = vec2u(textureDimensions(coarseIn));
        let coarseUV = clamp(vec2i(vec2f(cDim) * (uv * 0.96 + 0.02)), vec2i(0), vec2i(cDim) - vec2i(1));
        incoming += textureLoad(coarseIn, coarseUV, 0).rgb * 0.65;
      }
      accum += incoming;
    } else {
      accum += vec3f(0.02, 0.03, 0.06);
    }
  }

  let irradiance = accum / f32(sampleCount);
  textureStore(probeOut, vec2i(coord), vec4f(irradiance, 1.0));
}
`;

const LIGHTING_SHADER = /* wgsl */ `
${COMMON_WGSL}

@group(0) @binding(1) var albedoDepthTex : texture_2d<f32>;
@group(0) @binding(2) var normalHitTex : texture_2d<f32>;
@group(0) @binding(3) var coarseProbes : texture_2d<f32>;
@group(0) @binding(4) var fineProbes : texture_2d<f32>;

struct VSOut {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
};

@vertex
fn vsMain(@builtin(vertex_index) vi: u32) -> VSOut {
  var out : VSOut;
  let p = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  out.position = vec4f(p[vi], 0.0, 1.0);
  out.uv = p[vi] * 0.5 + 0.5;
  return out;
}

fn sampleProbe(tex: texture_2d<f32>, p: vec3f) -> vec3f {
  let d = vec2f(textureDimensions(tex));
  let uv = clamp(p.xz / 8.0 + 0.5, vec2f(0.0), vec2f(0.999));
  let c = vec2i(uv * d);
  return textureLoad(tex, c, 0).rgb;
}

@fragment
fn fsMain(in: VSOut) -> @location(0) vec4f {
  let texDim = vec2u(textureDimensions(albedoDepthTex));
  let coord = vec2i(clamp(in.uv * vec2f(texDim), vec2f(0.0), vec2f(vec2u(texDim - 1u))));
  let ad = textureLoad(albedoDepthTex, coord, 0);
  if (ad.a < 0.0) {
    let sky = vec3f(0.02, 0.04, 0.08) + pow(1.0 - in.uv.y, 2.0) * vec3f(0.04, 0.1, 0.2);
    return vec4f(sky, 1.0);
  }

  let ro = camera.cameraPos.xyz;
  let rd = worldRay(in.uv);
  let pos = ro + rd * ad.a;
  let n = normalize(textureLoad(normalHitTex, coord, 0).xyz * 2.0 - 1.0);

  let direct = emissiveLight(pos) * max(dot(n, normalize(vec3f(0.35, 1.0, -0.45))), 0.1);

  let fine = sampleProbe(fineProbes, pos + n * 0.2);
  let coarse = sampleProbe(coarseProbes, pos + n * 0.25);

  let depthBlend = smoothstep(1.2, 8.5, ad.a);
  let indirect = mix(fine, coarse, depthBlend) * ad.rgb * (0.45 + 0.55 * max(n.y, 0.0));

  let color = ad.rgb * direct + indirect;
  let mapped = color / (color + 1.0);
  return vec4f(pow(mapped, vec3f(0.4545)), 1.0);
}
`;

async function init() {
  if (!navigator.gpu) {
    statusEl.textContent = 'WebGPU not available in this browser.';
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter?.requestDevice();

  if (!device) {
    statusEl.textContent = 'Unable to acquire a WebGPU device.';
    return;
  }

  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();

  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(640, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(360, Math.floor((canvas.clientWidth * 9 * dpr) / 16));
    canvas.width = width;
    canvas.height = height;
  }

  resizeCanvas();

  context.configure({
    device,
    format,
    alphaMode: 'opaque',
  });

  const cameraBuffer = device.createBuffer({
    size: 16 * 4 + 4 * 4 + 4 * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const probeParamsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  function createRenderTexture(label, formatTex) {
    return device.createTexture({
      label,
      size: [canvas.width, canvas.height],
      format: formatTex,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  let albedoDepthTex = createRenderTexture('gAlbedoDepth', 'rgba16float');
  let normalTex = createRenderTexture('gNormal', 'rgba16float');

  const probeTextures = PROBE_CASCADES.map((cascade, i) =>
    device.createTexture({
      label: `probeCascade${i}`,
      size: [cascade.size, cascade.size],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
  );

  const gbufferPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({ code: GBUFFER_SHADER }),
      entryPoint: 'vsMain',
    },
    fragment: {
      module: device.createShaderModule({ code: GBUFFER_SHADER }),
      entryPoint: 'fsMain',
      targets: [{ format: 'rgba16float' }, { format: 'rgba16float' }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const probePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: PROBE_SHADER }),
      entryPoint: 'csMain',
    },
  });

  const lightingPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({ code: LIGHTING_SHADER }),
      entryPoint: 'vsMain',
    },
    fragment: {
      module: device.createShaderModule({ code: LIGHTING_SHADER }),
      entryPoint: 'fsMain',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
  });

  function cameraData(timeMs) {
    const t = timeMs * 0.001;
    const aspect = canvas.width / canvas.height;
    const eye = [Math.cos(t * 0.25) * 4.6, 2.4, Math.sin(t * 0.25) * 4.6 - 0.5];
    const target = [0, 0.7, 0.6];

    const view = mat4LookAt(eye, target, [0, 1, 0]);
    const proj = mat4Perspective((55 * Math.PI) / 180, aspect, 0.1, 40);
    const vp = mat4Multiply(proj, view);
    const inv = mat4Inverse(vp);

    return new Float32Array([...inv, ...eye, 0, t, canvas.width, canvas.height, 0]);
  }

  function rebuildRenderTargets() {
    albedoDepthTex.destroy();
    normalTex.destroy();
    albedoDepthTex = createRenderTexture('gAlbedoDepth', 'rgba16float');
    normalTex = createRenderTexture('gNormal', 'rgba16float');
  }

  window.addEventListener('resize', () => {
    const oldW = canvas.width;
    resizeCanvas();
    if (oldW !== canvas.width) {
      rebuildRenderTargets();
    }
  });

  function createProbeBindGroup(cascadeIndex) {
    const coarse = cascadeIndex === 0 ? probeTextures[0] : probeTextures[cascadeIndex - 1];
    return device.createBindGroup({
      layout: probePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: coarse.createView() },
        { binding: 2, resource: probeTextures[cascadeIndex].createView() },
        { binding: 3, resource: { buffer: probeParamsBuffer } },
      ],
    });
  }

  let probeBindGroups = PROBE_CASCADES.map((_, i) => createProbeBindGroup(i));

  function draw(timeMs) {
    device.queue.writeBuffer(cameraBuffer, 0, cameraData(timeMs));

    const encoder = device.createCommandEncoder();

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: albedoDepthTex.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: -1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
          {
            view: normalTex.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(gbufferPipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: gbufferPipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: cameraBuffer } }],
        })
      );
      pass.draw(3);
      pass.end();
    }

    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(probePipeline);
      PROBE_CASCADES.forEach((cascade, i) => {
        const params = new Uint32Array([
          cascade.size,
          floatToUint(cascade.spacing),
          i === 0 ? 0 : 1,
          0,
        ]);
        device.queue.writeBuffer(probeParamsBuffer, 0, params);
        pass.setBindGroup(0, probeBindGroups[i]);
        pass.dispatchWorkgroups(Math.ceil(cascade.size / 8), Math.ceil(cascade.size / 8));
      });
      pass.end();
    }

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0.01, g: 0.01, b: 0.015, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });

      pass.setPipeline(lightingPipeline);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: lightingPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: cameraBuffer } },
            { binding: 1, resource: albedoDepthTex.createView() },
            { binding: 2, resource: normalTex.createView() },
            { binding: 3, resource: probeTextures[0].createView() },
            { binding: 4, resource: probeTextures[1].createView() },
          ],
        })
      );
      pass.draw(3);
      pass.end();
    }

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(draw);
  }

  requestAnimationFrame(draw);
  statusEl.textContent = 'Running on WebGPU with 2 radiance cascades and dynamic DDGI probes.';

  if (device.lost) {
    device.lost.then((info) => {
      statusEl.textContent = `WebGPU device was lost: ${info.message}`;
    });
  }

  // Rebuild bind groups if textures are recreated in the future.
  probeBindGroups = PROBE_CASCADES.map((_, i) => createProbeBindGroup(i));
}

function floatToUint(value) {
  const f = new Float32Array([value]);
  return new Uint32Array(f.buffer)[0];
}

function mat4LookAt(eye, target, up) {
  const z = normalize(sub(eye, target));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x[0],
    y[0],
    z[0],
    0,
    x[1],
    y[1],
    z[1],
    0,
    x[2],
    y[2],
    z[2],
    0,
    -dot(x, eye),
    -dot(y, eye),
    -dot(z, eye),
    1,
  ];
}

function mat4Perspective(fov, aspect, near, far) {
  const f = 1.0 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  return [
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) * nf,
    -1,
    0,
    0,
    2 * far * near * nf,
    0,
  ];
}

function mat4Multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let r = 0; r < 4; r += 1) {
    for (let c = 0; c < 4; c += 1) {
      for (let k = 0; k < 4; k += 1) {
        out[c + r * 4] += a[k + r * 4] * b[c + k * 4];
      }
    }
  }
  return out;
}

function mat4Inverse(m) {
  const inv = new Array(16);
  inv[0] =
    m[5] * m[10] * m[15] -
    m[5] * m[11] * m[14] -
    m[9] * m[6] * m[15] +
    m[9] * m[7] * m[14] +
    m[13] * m[6] * m[11] -
    m[13] * m[7] * m[10];
  inv[4] =
    -m[4] * m[10] * m[15] +
    m[4] * m[11] * m[14] +
    m[8] * m[6] * m[15] -
    m[8] * m[7] * m[14] -
    m[12] * m[6] * m[11] +
    m[12] * m[7] * m[10];
  inv[8] =
    m[4] * m[9] * m[15] -
    m[4] * m[11] * m[13] -
    m[8] * m[5] * m[15] +
    m[8] * m[7] * m[13] +
    m[12] * m[5] * m[11] -
    m[12] * m[7] * m[9];
  inv[12] =
    -m[4] * m[9] * m[14] +
    m[4] * m[10] * m[13] +
    m[8] * m[5] * m[14] -
    m[8] * m[6] * m[13] -
    m[12] * m[5] * m[10] +
    m[12] * m[6] * m[9];
  inv[1] =
    -m[1] * m[10] * m[15] +
    m[1] * m[11] * m[14] +
    m[9] * m[2] * m[15] -
    m[9] * m[3] * m[14] -
    m[13] * m[2] * m[11] +
    m[13] * m[3] * m[10];
  inv[5] =
    m[0] * m[10] * m[15] -
    m[0] * m[11] * m[14] -
    m[8] * m[2] * m[15] +
    m[8] * m[3] * m[14] +
    m[12] * m[2] * m[11] -
    m[12] * m[3] * m[10];
  inv[9] =
    -m[0] * m[9] * m[15] +
    m[0] * m[11] * m[13] +
    m[8] * m[1] * m[15] -
    m[8] * m[3] * m[13] -
    m[12] * m[1] * m[11] +
    m[12] * m[3] * m[9];
  inv[13] =
    m[0] * m[9] * m[14] -
    m[0] * m[10] * m[13] -
    m[8] * m[1] * m[14] +
    m[8] * m[2] * m[13] +
    m[12] * m[1] * m[10] -
    m[12] * m[2] * m[9];
  inv[2] =
    m[1] * m[6] * m[15] -
    m[1] * m[7] * m[14] -
    m[5] * m[2] * m[15] +
    m[5] * m[3] * m[14] +
    m[13] * m[2] * m[7] -
    m[13] * m[3] * m[6];
  inv[6] =
    -m[0] * m[6] * m[15] +
    m[0] * m[7] * m[14] +
    m[4] * m[2] * m[15] -
    m[4] * m[3] * m[14] -
    m[12] * m[2] * m[7] +
    m[12] * m[3] * m[6];
  inv[10] =
    m[0] * m[5] * m[15] -
    m[0] * m[7] * m[13] -
    m[4] * m[1] * m[15] +
    m[4] * m[3] * m[13] +
    m[12] * m[1] * m[7] -
    m[12] * m[3] * m[5];
  inv[14] =
    -m[0] * m[5] * m[14] +
    m[0] * m[6] * m[13] +
    m[4] * m[1] * m[14] -
    m[4] * m[2] * m[13] -
    m[12] * m[1] * m[6] +
    m[12] * m[2] * m[5];
  inv[3] =
    -m[1] * m[6] * m[11] +
    m[1] * m[7] * m[10] +
    m[5] * m[2] * m[11] -
    m[5] * m[3] * m[10] -
    m[9] * m[2] * m[7] +
    m[9] * m[3] * m[6];
  inv[7] =
    m[0] * m[6] * m[11] -
    m[0] * m[7] * m[10] -
    m[4] * m[2] * m[11] +
    m[4] * m[3] * m[10] +
    m[8] * m[2] * m[7] -
    m[8] * m[3] * m[6];
  inv[11] =
    -m[0] * m[5] * m[11] +
    m[0] * m[7] * m[9] +
    m[4] * m[1] * m[11] -
    m[4] * m[3] * m[9] -
    m[8] * m[1] * m[7] +
    m[8] * m[3] * m[5];
  inv[15] =
    m[0] * m[5] * m[10] -
    m[0] * m[6] * m[9] -
    m[4] * m[1] * m[10] +
    m[4] * m[2] * m[9] +
    m[8] * m[1] * m[6] -
    m[8] * m[2] * m[5];

  let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  det = 1.0 / det;
  return inv.map((x) => x * det);
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

init();
