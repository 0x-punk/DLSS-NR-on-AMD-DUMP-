# HIP Processing Path: Complete Data-Flow Report (dlssnr_on_amd_setup.exe)

> Binary: `dlssnr_on_amd_setup.exe` (4,689,622 B, PE32+ x64, no exports)
> Method: static string/import/symbol extraction from the binary (no execution)
> Scope: ONLY the HIP processing section - from data entry to processing exit,
> every function called, every DLL invoked, every precision used.

---

## 1. Entry: How Data Enters HIP

### 1.1 Source surfaces (D3D12 game side)

The engine hooks the game's D3D12 swapchain and reads four inputs per frame
(from the log string `"staging ready: colour %ux%u dxgi %d ... motion %dx%d ... depth %dx%d ..."`):

| Input | Format | Resolution | Role |
|---|---|---|---|
| Colour | DXGI RGBA (game backbuffer) | Full (e.g. 2560x1440) | Main image |
| Motion (MVec) | DXGI RG | Half (1280x720) | Temporal reprojection |
| Depth | DXGI R32F | Half (1280x720) | Disocclusion + guides |
| Exposure | scalar (auto-exposure) | 1x1 | `auto-exposure: encoded mean %.3f -> exposure %.4f` |

### 1.2 Zero-copy bridge D3D12 -> HIP (the ONLY cross-API call)

```
D3D12 resource
  -> CreateSharedHandle (D3D12)            // "interop: CreateSharedHandle failed 0x%08lx"
  -> hipImportExternalMemory (amdhip64)    // "interop: hipImportExternalMemory: %s"
  -> hipExternalMemoryGetMappedBuffer      // "interop: hipExternalMemoryGetMappedBuffer: %s"
  -> HIP device pointer (zero-copy, no memcpy)
```

Fallback if interop fails: `"inline mode needs zero-copy interop; running asynchronously"`
and `"residual textures failed; falling back to frame replacement"`.

### 1.3 Staging log line (proves the data contract)

```
staging ready: colour %ux%u dxgi %d (pix %d, tonemap %d);
               motion %dx%d dxgi %d;
               depth %dx%d dxgi %d (inverted %d);
               exposure %s; residual %s
```

---

## 2. HIP Runtime Calls (in order, from import table + log strings)

The binary imports HIP dynamically (late-load via `amdhip64_7.dll`; no static HIP import
table — all `hip*` names appear as runtime strings). Observed call sequence per frame:

| # | HIP call | Purpose | Evidence |
|---|---|---|---|
| 1 | `hipGetDeviceCount` | Enumerate GPUs | import string |
| 2 | `hipGetDevicePropertiesR0600` | Read arch/CUs/VRAM | log `"env: HIP device 0: %s, arch %s, %d CUs, %.0f MB, driver %d, runtime %d"` |
| 3 | `hipDriverGetVersion` / `hipRuntimeGetVersion` | Version check | import strings |
| 4 | `hipImportExternalMemory` | Import D3D12 shared handle | log `"interop: hipImportExternalMemory: %s"` |
| 5 | `hipExternalMemoryGetMappedBuffer` | Map to device pointer | log `"interop: hipExternalMemoryGetMappedBuffer: %s"` |
| 6 | `hipMalloc` | Allocate internal tensors (activations, history) | import string |
| 7 | `hipMemset` / `hipMemsetAsync` | Clear / reset (`DLSSNR.Reset`) | import strings |
| 8 | `hipMemcpy` / `hipMemcpyAsync` | Upload weights (once), download debug | import strings |
| 9 | `hipMemcpyToSymbol` | Upload constants/scales | import string |
| 10 | `hipEventCreate` / `hipEventRecord` | Per-job timing | `"network job %d done in %llu ms"` |
| 11 | `hipLaunchKernel` (x N) | Launch each kernel below | import string |
| 12 | `hipEventElapsedTime` / `hipEventSynchronize` | Measure + sync | import strings |
| 13 | `hipDeviceSynchronize` | Frame barrier (inline mode) | `"inline (same-frame, game waits for the network)"` |
| 14 | `hipGetErrorString` / `hipGetLastError` | Error path | `"job %d GPU errors: %s %s"` |
| 15 | `hipFree` / `hipDestroyExternalMemory` | Teardown | import strings |

Synchronization model: `hipEventRecord` per job + spin-wait flag pipeline
(`"inline: spin used %u iterations for a %llu ms job"`, HLSL flag shader below).

---

## 3. Kernel Functions (30 unique, from mangled `_Z` symbols)

Naming: `k_<stage><variant>(<ParamsStruct>)`. Each kernel carries `.kd` (kernel descriptor),
`.num_vgpr/.num_agpr`, `.private_seg_size`, `.uses_vcc` metadata (standard AMDGPU code-object fields).

### 3.1 Data ingress (D3D12 pixels -> model tensors)

| Kernel | Params struct | Role | Precision |
|---|---|---|---|
| `k_import` | `ImportParams` | Copy staged colour/motion/depth into HIP tensors | fp32 (pixels) |
| `k_repack` | `RepackParams` | Repack FP8 weights to WMMA layout (once per weight load) | **FP8 E4M3** -> WMMA tile |
| `k_mean` | `MeanParams` | Exposure metering (`encoded mean`) | fp32 reduce |
| `k_reproject` | `ReprojParams` | Motion reprojection + depth linearize | fp32 |

### 3.2 Pre-block (encoder stem, 1 head / 32ch, FP8)

| Kernel | Params struct | Role | Precision |
|---|---|---|---|
| `k_pre_block_1h_32_fp8` | `PreParams` | Stem conv + norm, 1 head x 32ch | **FP8 E4M3** (name-embedded) |
| `k_swin_1h_32_fp8` | `SwinParams` | Swin encoder block, 1 head x 32ch | **FP8 E4M3** |
| `k_post_block_1h_32_fp8` | `PostParams` | Post stem | **FP8 E4M3** |

The `_fp8` suffix is embedded in the symbol name: these three kernels operate
natively in FP8 E4M3 (confirmed by the `g_e4m3_lut` lookup table also present in the binary).

### 3.3 Core transformer (QKV -> attention -> FFN)

| Kernel | Params struct | Role | Precision |
|---|---|---|---|
| `k_qkv` | `QkvParams` | Fused QKV projection (small) | fp16/fp32 accum |
| `k_qkv2` | `QkvParams` | Fused QKV projection (large / ViT) | fp16/fp32 accum |
| `k_qkv_attn` | `AttnParams` | Fused QKV + attention (small) | fp16/fp32 |
| `k_qkv_attn2` | `AttnParams` | Fused QKV + attention (large) | fp16/fp32 |
| `k_attention` | `AttnParams1d` | Standalone attention (1-D launch) | fp32 softmax |
| `k_attention2` | `AttnParams1d` | Standalone attention, variant 2 | fp32 softmax |
| `k_swin_var<32,64,128,256>` | `VarParams` | Swin blocks at 32/64/128/256 channels (`ILi32/64/128/256`, `Lb0/Lb1` = with/without shift) | fp16/fp32 |
| `k_ffwd` | `FfwdParams` | Feed-forward (small) | fp16/fp32 |
| `k_ffwd2` | `Ffwd2Params` | Feed-forward (large) | fp16/fp32 |
| `k_ffwd_inpview` | `FfwdPlParams` | FFN with in-place views (fused act) | fp16/fp32 |
| `k_expand` / `k_expand2` | `ExpandParams` | FFN up-projection | fp16 |
| `k_contract2` | `ConvParams1d` | FFN down-projection | fp16 |

### 3.4 Convolution / residual branches

| Kernel | Params struct | Role | Precision |
|---|---|---|---|
| `k_conv_res` | `ConvParams` | Residual conv | fp16/fp32 |
| `k_conv_res2` | `Conv2Params` | Residual conv, variant 2 | fp16/fp32 |
| `k_conv_res_views` | `ConvPlParams` | Residual conv with views | fp16/fp32 |
| `k_conv_splitk` | `ConvParams1d` | Split-K convolution | fp16/fp32 |

### 3.5 Decoder / output

| Kernel | Params struct | Role | Precision |
|---|---|---|---|
| `k_dec_upsample` | `DecUpParams` | Decoder upsampling | fp16/fp32 |
| `k_final_head` | `HeadParams` | Final projection to residual | fp32 out |
| `k_export` | `ExportParams` | Write residual to shared output | fp32 |

### 3.6 Synchronization helpers (NOT neural)

| Kernel | Role |
|---|---|
| `k_flag_set` | Set flag buffer (job done) |
| `k_flag_wait` | Spin on flag buffer (inline mode) |

Plus two D3D12-side HLSL helpers embedded as source strings:

1. **Flag pipeline** (`[numthreads(1,1,1)]`, `RWByteAddressBuffer flags`): spin-wait
   with timeout counter (`"inline: spin used %u iterations ... cap %u"`).
2. **Residual apply + tonemap** (`Texture2D res`, `RWTexture2D outp`): `tmf`/`tmi`
   sRGB transfer functions, `expo` scaling (`"residual apply pipeline ready"`).

---

## 4. Precision Map (from symbol names + `g_e4m3_lut`)

| Stage | Storage | Compute | Accumulate | Evidence |
|---|---|---|---|---|
| Weights (153 tensors) | **FP8 E4M3** (1 byte/elem) | — | — | WEIGHTS_HT blob; `g_e4m3_lut` decode table |
| Stem / pre-post blocks | FP8 E4M3 | **FP8 WMMA** (`v_wmma_f32_16x16x16_fp8_fp8`) | fp32 | `_fp8` in kernel names |
| QKV / FFN GEMMs | FP8 E4M3 | FP8 WMMA or fp16 | fp32 | `k_qkv*`, `k_ffwd*`, `k_expand*` |
| Attention softmax | — | **fp32** (exp/sum) | fp32 | `k_attention*` (numerically required) |
| Residual / export | fp32 | fp32 | fp32 | `k_final_head`, `k_export`, HLSL tonemap |
| Exposure / mean | fp32 | fp32 reduce | fp32 | `k_mean`, `"auto-exposure"` |

**No E5M2, no BF16, no INT8 anywhere in the binary.** The entire model is E4M3 + fp32.

---

## 5. Target GPUs (from fatbin triples)

```
hipv4-amdgcn-amd-amdhsa--gfx1100
hipv4-amdgcn-amd-amdhsa--gfx1101
hipv4-amdgcn-amd-amdhsa--gfx1102
hipv4-amdgcn-amd-amdhsa--gfx1201
```

- gfx1100/1101/1102 = RDNA3 (RX 7000). Log line: `"env: RDNA3 (%s) detected:
  the RDNA3 kernels are untested on hardware"`.
- gfx1201 = RDNA4 (RX 9000). Primary target.
- Anything else: `"env: UNSUPPORTED GPU ARCHITECTURE %s ... the pass stays off"`.

HIP runtime resolved at load from `amdhip64_7.dll` (`__hipRegisterFatBinary`,
`__hipRegisterFunction`, `__hipPushCallConfiguration`, `hipLaunchKernel`, ...).

---

## 6. Per-Frame Call Graph (HIP section only)

```
Present hook
  -> overlayTick
  -> staging (D3D12 read)
  -> hipImportExternalMemory + hipExternalMemoryGetMappedBuffer   [once per resource]
  -> k_import                                                        [pixels -> tensors]
  -> k_mean                                                          [exposure]
  -> k_reproject                                                    [motion/depth]
  -> k_pre_block_1h_32_fp8                                          [stem, FP8]
  -> k_swin_1h_32_fp8 (x N)                                         [encoder, FP8]
  -> k_qkv / k_qkv_attn / k_attention                               [core attn]
  -> k_ffwd / k_expand / k_contract2                               [core FFN]
  -> k_swin_var<32/64/128/256>                                     [multi-scale]
  -> k_conv_res*                                                   [residual branch]
  -> k_dec_upsample                                                 [decoder]
  -> k_final_head                                                   [residual out, fp32]
  -> k_export                                                       [to shared buffer]
  -> k_flag_set  (+ HLSL flag shader spin on game side)             [sync]
  -> hipEventRecord / hipEventElapsedTime                           ["network job %d done in %llu ms"]
  -> HLSL residual-apply shader (D3D12)                             [compose + tonemap]
  -> Present original
```

Observed timing log: `"network job %d done in %llu ms (history %s, %s)"`,
`"frames %d dispatches %d submitted %d ready %d route %s"`.

---

## 7. Exit: How Data Leaves HIP

```
HIP residual tensor (fp32)
  -> k_export -> shared D3D12 buffer (zero-copy, same allocation)
  -> HLSL residual-apply shader on the game queue:
       tone path:  c.rgb * expo -> tmf (sRGB encode) -> + d -> tmi (sRGB decode) / expo
       plain path: saturate(c.rgb + d)
  -> game backbuffer -> Present
```

Fallbacks (from strings): `"output has no UAV flag; residual apply unavailable,
using frame replacement"`, `"async (residual from an earlier frame)"` vs
`"inline (same-frame, game waits for the network)"`.

Debug path: `"dump %d: exposure %.5f ..."` writes
`dlssnr_in/mv/depth/out<job>.raw` next to the DLL.

---

## 8. DLL / Module Map (HIP section)

| Module | Loaded by | Role |
|---|---|---|
| `amdhip64_7.dll` | exe at startup | HIP runtime (all `hip*` calls) |
| `nvngx_dlssnr.dll` | setup tool | Weight source (WEIGHTS_HT resource, 153 tensors FP8) |
| `dxgi.dll` (`CreateDXGIFactory1`) | engine | Swapchain interception point |
| `version.dll` (proxy) | game process | Injection point (forwards real version.dll) |
| exe's own fatbin (`__hipFatB`, `.hip_fatx`) | `__hipRegisterFatBinary` | 30 kernels x 4 arch targets |

DirectX DLLs (`d3d12.dll`, `d3d11.dll`) are consumed but never patched;
only DXGI factory/swapchain/Present entry points are hooked.

---

*Source of every claim above: static strings, import table, and kernel symbols
read directly from `dlssnr_on_amd_setup.exe`. No execution was used.*
