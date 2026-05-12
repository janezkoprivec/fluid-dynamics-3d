struct Particle {
  position: vec3<f32>,
  velocity: vec3<f32>,
};

struct Camera {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  eye: vec3<f32>,
  _pad0: f32,
};

struct RenderParams {
  viewport: vec2<f32>,
  pointSize: f32,
  _pad: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> camera: Camera;
@group(0) @binding(2) var<uniform> rparams: RenderParams;

struct VsOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) viewZ: f32,
};

const CORNERS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2<f32>(-1.0, -1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>(-1.0,  1.0),
  vec2<f32>( 1.0, -1.0),
  vec2<f32>( 1.0,  1.0),
);

fn ramp(t: f32) -> vec3<f32> {
  let cool = vec3<f32>(0.18, 0.55, 0.95);
  let warm = vec3<f32>(0.95, 0.55, 0.18);
  let k = clamp(t, 0.0, 1.0);
  return mix(cool, warm, k);
}

@vertex
fn vs_main(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VsOut {
  let p = particles[ii];
  let corner = CORNERS[vi];

  let viewPos = camera.view * vec4<f32>(p.position, 1.0);
  let clipCenter = camera.proj * viewPos;

  let pxToClipX = 2.0 / rparams.viewport.x;
  let pxToClipY = 2.0 / rparams.viewport.y;
  let halfPx = rparams.pointSize * 0.5;

  var clip = clipCenter;
  clip.x = clip.x + corner.x * halfPx * pxToClipX * clipCenter.w;
  clip.y = clip.y + corner.y * halfPx * pxToClipY * clipCenter.w;

  let speed = length(p.velocity);
  let t = clamp(speed / 6.0, 0.0, 1.0);

  var out: VsOut;
  out.clip = clip;
  out.uv = corner;
  out.color = ramp(t);
  out.viewZ = -viewPos.z;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let z = sqrt(max(0.0, 1.0 - r2));
  let normal = vec3<f32>(in.uv.x, in.uv.y, z);
  let lightDir = normalize(vec3<f32>(0.2, 0.85, 0.4));
  let ndotl = clamp(dot(normal, lightDir), 0.0, 1.0);
  let ambient = 0.25;
  let lit = in.color * (ambient + (1.0 - ambient) * ndotl);
  return vec4<f32>(lit, 1.0);
}
