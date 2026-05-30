// Renders the GPU-emitted marching-cubes mesh. Vertices and normals are
// padded vec4 arrays — positions in `.xyz`, normals in `.xyz`.
//
// Shading model: Blinn-Phong diffuse/spec for the directional light, plus a
// Fresnel-Schlick term that mixes a sky/horizon "reflection" in at grazing
// angles. Two-sided: the front-facing test flips the normal so backfaces of
// an open puddle still light correctly.

struct Camera {
  viewProj: mat4x4<f32>,
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  eye: vec3<f32>,
  _pad0: f32,
};

struct SurfaceRenderParams {
  baseColor: vec3<f32>,
  ambient: f32,
  lightDir: vec3<f32>,
  shininess: f32,
  specularStrength: f32,
  fresnelStrength: f32,
  transmission: f32,
  _pad0: f32,
};

@group(0) @binding(0) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<uniform> rparams: SurfaceRenderParams;

struct VsOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
  let p = vertices[vi].xyz;
  let n = normals[vi].xyz;
  var out: VsOut;
  out.clip = camera.viewProj * vec4<f32>(p, 1.0);
  out.worldPos = p;
  out.normal = n;
  return out;
}

const SKY_TOP: vec3<f32> = vec3<f32>(0.55, 0.70, 0.92);
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.18, 0.30, 0.45);
const F0_WATER: f32 = 0.02;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  var nrm = in.normal;
  let nlen = length(nrm);
  if (nlen < 1e-5) {
    nrm = vec3<f32>(0.0, 1.0, 0.0);
  } else {
    nrm = nrm / nlen;
  }

  let L = normalize(rparams.lightDir);
  let V = normalize(camera.eye - in.worldPos);

  // Re-orient the normal to face the viewer. We can't trust front_facing for
  // this — Twinklebear's MC table winds CCW from the *inside*, so the outside
  // camera view is geometrically back-facing. Aligning with V handles both
  // sides correctly and is independent of triangulation conventions.
  if (dot(nrm, V) < 0.0) { nrm = -nrm; }

  let H = normalize(L + V);
  let NdotL = max(0.0, dot(nrm, L));
  let NdotH = max(0.0, dot(nrm, H));
  let NdotV = clamp(dot(nrm, V), 0.0, 1.0);

  // Sky lookup along the reflection ray (cheap two-color sky).
  let R = reflect(-V, nrm);
  let skyT = clamp(R.y * 0.5 + 0.5, 0.0, 1.0);
  let sky = mix(SKY_HORIZON, SKY_TOP, skyT);

  // Fresnel-Schlick — strong rim brightening at grazing angles.
  let fres = F0_WATER + (1.0 - F0_WATER) * pow(1.0 - NdotV, 5.0);
  let reflectance = clamp(fres * rparams.fresnelStrength, 0.0, 1.0);

  let deep = rparams.baseColor;
  // Fake a thickness cue: light from above transmits more through thin
  // upward-facing sheets (top of splash) and less through downward-facing
  // pockets, giving a darker interior look without real refraction.
  let thicknessFake = clamp(0.5 - 0.5 * nrm.y, 0.0, 1.0);
  let deepTinted = mix(deep, deep * 0.4, thicknessFake);

  let ambient = mix(deepTinted, sky, 0.08) * rparams.ambient;
  let diffuse = deepTinted * NdotL * (1.0 - reflectance);
  let reflected = sky * reflectance;
  let specular = pow(NdotH, max(1.0, rparams.shininess)) * rparams.specularStrength;

  let lit = ambient + diffuse + reflected + vec3<f32>(specular);

  // Fresnel-weighted opacity: head-on regions show through (low alpha),
  // grazing edges and specular hot-spots stay opaque (mirror-like). This is
  // the cheap approximation of refraction — no scene resampling required.
  let alpha = clamp(mix(rparams.transmission, 1.0, reflectance), 0.0, 1.0);
  return vec4<f32>(lit, alpha);
}
