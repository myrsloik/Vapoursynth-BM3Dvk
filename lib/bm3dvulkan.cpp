/**
 * Copyright (C) 2021 WolframRhodium
 * Copyright (C) 2025 Sunflower Dolls
 *
 * BM3D for the VapourSynth Vulkan GPU API (API 4.3), built on the core's declaration
 * driver. Ported from this project's earlier Metal implementation, itself a port of
 * VapourSynth-BM3DCUDA; gpufilter.h is copied beside this file, as its header instructs.
 * GPL-3.0-or-later, like both.
 *
 * What the driver replaces from the Metal implementation: the resource pool and ticket
 * semaphore (exec contexts), the per frame waitUntilCompleted (producer pairs; nothing
 * here ever waits on the host), the copy engines and shared-storage-mode host wrapping
 * (frames are already resident), and the eight precompiled pipeline states (one GLSL
 * source specialized by TEMPORAL/CHROMA/FINAL constants). VAggregate, CPU vDSP code in
 * the original, is a GPU pass here, so a temporal chain stays resident end to end.
 *
 * The Metal plugin's device_id, fast and zero_init parameters do not exist here: the core
 * owns device selection, pipelining depth comes from the exec pool, and the unprocessed
 * planes of the tall temporal output are always zeroed.
 *
 * Built by CMake (see ../CMakeLists.txt), or by hand with a compiler that has C23 #embed:
 *   clang-cl /LD /MD /O2 /EHsc /std:c++20 /bigobj /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS ^
 *     bm3dvulkan.cpp /I<this dir> /I<vapoursynth include> /I<vulkan sdk include> ^
 *     /Fe:bm3dvulkan.dll
 */

#define VS_USE_API_43
#include "VapourSynth4.h"
#include "VSHelper4.h"
#include "VSVulkan4.h"

#include "gpufilter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace std::string_literals;

namespace {

VSPlugin *myself = nullptr;

/* Must mirror the push_constant block in shader.comp field for field. */
struct KernelParams {
    int32_t width;
    int32_t height;
    int32_t stride;
    float sigma;
    int32_t block_step;
    int32_t bm_range;
    int32_t radius;
    int32_t ps_num;
    int32_t ps_range;
    float sigma_u;
    float sigma_v;
    float extractor;
};

struct CopyPush { uint32_t srcStride, dstStride, width, height, srcOffset, dstOffset; };
struct ZeroPush { uint32_t words, offset; };
struct AggPush { uint32_t srcStride, dstStride, width, height, srcOffset, weightRel; };
struct VAggPush { uint32_t srcStride, dstStride, width, height; int32_t lo, hi; };

const char copyGlsl[] = R"(#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer S { uint s[]; };
layout(std430, binding = 1) writeonly buffer D { uint d[]; };
layout(push_constant) uniform PC { uint srcStride, dstStride, width, height, srcOffset, dstOffset; } pc;
void main() {
    uint x = gl_GlobalInvocationID.x, y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;
    d[pc.dstOffset + y * pc.dstStride + x] = s[pc.srcOffset + y * pc.srcStride + x];
}
)";

const char zeroGlsl[] = R"(#version 460
layout(local_size_x = 256) in;
layout(std430, binding = 0) writeonly buffer D { uint d[]; };
layout(push_constant) uniform PC { uint words, offset; } pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.words) d[pc.offset + i] = 0u;
}
)";

const char aggGlsl[] = R"(#version 460
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer R { uint r[]; };
layout(std430, binding = 1) writeonly buffer D { float d[]; };
layout(push_constant) uniform PC { uint srcStride, dstStride, width, height, srcOffset, weightRel; } pc;
void main() {
    uint x = gl_GlobalInvocationID.x, y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;
    float wdst = uintBitsToFloat(r[pc.srcOffset + y * pc.srcStride + x]);
    float w = uintBitsToFloat(r[pc.srcOffset + pc.weightRel + y * pc.srcStride + x]);
    d[y * pc.dstStride + x] = wdst / (w > 0.0 ? w : 1.0);
}
)";

/* The kernel source ships inside the binary. #embed is the direct route and needs no build
   step, but it is C23 (clang 19+, gcc 15+, not MSVC), so CMake also generates
   shader_comp.h -- a raw string literal -- and that is used wherever #embed is missing.
   Define BM3DVK_NO_EMBED to force the generated header even where #embed exists, which is
   how the fallback gets exercised on a compiler that would not otherwise take it. */
#if defined(__has_embed) && !defined(BM3DVK_NO_EMBED)
#  if __has_embed("shader.comp")
#    define BM3DVK_HAVE_EMBED 1
#  endif
#endif

#ifdef BM3DVK_HAVE_EMBED
const char bm3dGlsl[] = {
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wc23-extensions"
#embed "shader.comp"
#  pragma clang diagnostic pop
    , '\0'
};
#else
#  include "shader_comp.h"
#endif

/* The kernel runs on any subgroup width that is a multiple of its 8-lane clusters and that
   the workgroup can be sized to match; 32 and 64 cover real hardware. Prefer 32 (better
   register occupancy for a kernel this heavy), fall back to 64 for wave64-only devices.
   subgroupSize receives the chosen width, pinnedSize the value to pin (0 when the device
   already runs that width unpinned). BM3DVK_FORCE_SUBGROUP=32|64 overrides for testing. */
bool chooseSubgroup(VSCore *core, const VSAPI *vsapi, uint32_t &subgroupSize, uint32_t &pinnedSize,
    std::string &error) {
    const VSVULKANAPI *vkapi = vsapi->getVulkanAPI(VSVULKAN_API_VERSION);
    if (!vkapi) {
        error = "the Vulkan API is not available";
        return false;
    }
    char err[512] = {};
    VSVulkanCoreHandles handles;
    if (vkapi->getVulkanHandles(core, &handles, err, sizeof(err))) {
        error = err;
        return false;
    }
    const VSVulkanFunctions *vk = vkapi->getVulkanFunctions(core, err, sizeof(err));
    if (!vk) {
        error = err;
        return false;
    }
    VkPhysicalDeviceVulkan13Properties props13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES };
    VkPhysicalDeviceVulkan11Properties props11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES };
    props11.pNext = &props13;
    VkPhysicalDeviceProperties2 props2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    props2.pNext = &props11;
    vk->vkGetPhysicalDeviceProperties2(handles.physicalDevice, &props2);

    const bool canPin = (props13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    uint32_t candidates[2] = { 32, 64 };
    int numCandidates = 2;
    if (const char *forced = std::getenv("BM3DVK_FORCE_SUBGROUP")) {
        candidates[0] = static_cast<uint32_t>(std::atoi(forced));
        numCandidates = 1;
        if (candidates[0] != 32 && candidates[0] != 64) {
            error = "BM3DVK_FORCE_SUBGROUP must be 32 or 64";
            return false;
        }
    }
    for (int i = 0; i < numCandidates; ++i) {
        const uint32_t c = candidates[i];
        if (canPin && props13.minSubgroupSize <= c && c <= props13.maxSubgroupSize) {
            subgroupSize = c;
            pinnedSize = c;
            return true;
        }
        if (props11.subgroupSize == c) {
            subgroupSize = c;
            pinnedSize = 0;
            return true;
        }
    }
    error = "this kernel needs 32- or 64-wide subgroups, which this device cannot provide (reports " +
        std::to_string(props11.subgroupSize) + ")";
    return false;
}

struct BM3DParams {
    std::array<float, 3> sigma;
    std::array<int, 3> block_step;
    std::array<int, 3> bm_range;
    std::array<int, 3> ps_num;
    std::array<int, 3> ps_range;
    std::array<bool, 3> process;
    float extractor;
    int radius;
    bool chroma;
    bool final_;
};

/* One entry per pass, consulted by fillPush and the reshape lambdas. */
struct PMeta {
    enum Kind { ZeroRes, ZeroOut, Copy, BM3D, Agg, Unpack } kind;
    int slotIndex = 0; /* Copy: position in the packed stack; Agg/Unpack: packed plane */
};

static void VS_CC BM3DCreate(const VSMap *in, VSMap *out, void *, VSCore *core, const VSAPI *vsapi) noexcept {
    VSNode *node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    const VSVideoInfo *vi = vsapi->getVideoInfo(node);
    VSNode *refNode = nullptr;
    int err = 0;

    auto fail = [&](const std::string &msg) {
        vsapi->mapSetError(out, ("BM3D: " + msg).c_str());
        vsapi->freeNode(node);
        if (refNode)
            vsapi->freeNode(refNode);
    };

    if (vi->format.sampleType != stFloat || vi->format.bitsPerSample != 32 ||
        !vsh::isConstantVideoFormat(vi))
        return fail("only constant format 32bit float input supported");

    refNode = vsapi->mapGetNode(in, "ref", 0, &err);
    BM3DParams p = {};
    p.final_ = (err == 0);
    if (p.final_) {
        const VSVideoInfo *refVi = vsapi->getVideoInfo(refNode);
        if (!vsh::isSameVideoFormat(&refVi->format, &vi->format) || refVi->width != vi->width ||
            refVi->height != vi->height || refVi->numFrames != vi->numFrames)
            return fail("ref clip properties must match input clip");
    }

    for (int i = 0; i < 3; ++i) {
        p.sigma[i] = static_cast<float>(vsapi->mapGetFloat(in, "sigma", i, &err));
        if (err)
            p.sigma[i] = (i == 0) ? 3.0f : p.sigma[i - 1];
        if (p.sigma[i] < 0.0f)
            return fail("sigma must be non-negative");
        p.process[i] = p.sigma[i] >= std::numeric_limits<float>::epsilon();
        p.sigma[i] *= (3.0f / 4.0f) / 255.0f * 64.0f * (p.final_ ? 1.0f : 2.7f);
    }
    for (int i = 0; i < 3; ++i) {
        p.block_step[i] = static_cast<int>(vsapi->mapGetInt(in, "block_step", i, &err));
        if (err)
            p.block_step[i] = (i == 0) ? 8 : p.block_step[i - 1];
        if (p.block_step[i] <= 0 || p.block_step[i] > 8)
            return fail("block_step must be in [1, 8]");
    }
    for (int i = 0; i < 3; ++i) {
        p.bm_range[i] = static_cast<int>(vsapi->mapGetInt(in, "bm_range", i, &err));
        if (err)
            p.bm_range[i] = (i == 0) ? 9 : p.bm_range[i - 1];
        if (p.bm_range[i] <= 0)
            return fail("bm_range must be positive");
    }
    p.radius = static_cast<int>(vsapi->mapGetInt(in, "radius", 0, &err));
    if (err)
        p.radius = 0;
    if (p.radius < 0)
        return fail("radius must be non-negative");
    /* VAggregate binds one tall frame per tap plus the output, and the driver allows 32
       bindings per pass. */
    if (p.radius > 15)
        return fail("radius must be at most 15");
    for (int i = 0; i < 3; ++i) {
        p.ps_num[i] = static_cast<int>(vsapi->mapGetInt(in, "ps_num", i, &err));
        if (err)
            p.ps_num[i] = (i == 0) ? 2 : p.ps_num[i - 1];
        if (p.ps_num[i] <= 0 || p.ps_num[i] > 8)
            return fail("ps_num must be in [1, 8]");
    }
    for (int i = 0; i < 3; ++i) {
        p.ps_range[i] = static_cast<int>(vsapi->mapGetInt(in, "ps_range", i, &err));
        if (err)
            p.ps_range[i] = (i == 0) ? 4 : p.ps_range[i - 1];
        if (p.ps_range[i] <= 0)
            return fail("ps_range must be positive");
    }
    p.chroma = vsapi->mapGetInt(in, "chroma", 0, &err) != 0;
    if (p.chroma && !vsh::isSameVideoPresetFormat(pfYUV444PS, &vi->format, core, vsapi))
        return fail("chroma=true requires YUV444PS");
    int extractorExp = static_cast<int>(vsapi->mapGetInt(in, "extractor_exp", 0, &err));
    p.extractor = (!err && extractorExp) ? std::ldexp(1.0f, extractorExp) : 0.0f;

    const int numPlanes = vi->format.numPlanes;
    int heavyGate = -1;
    for (int i = 0; i < numPlanes; ++i) {
        if (p.process[i]) {
            heavyGate = i;
            break;
        }
    }
    if (heavyGate < 0)
        return fail("at least one plane needs a positive sigma (BM3Dv2 handles the all-zero case)");

    uint32_t sgSize = 0, pinnedSize = 0;
    std::string sgError;
    if (!chooseSubgroup(core, vsapi, sgSize, pinnedSize, sgError))
        return fail(sgError);

    const int T = 2 * p.radius + 1;
    const int clips = p.final_ ? 2 : 1;
    const int numPack = p.chroma ? 3 : 1;
    const bool temporal = p.radius > 0;
    const bool resIsOutput = temporal && !p.chroma;

    vsgpu::FilterDesc desc;
    desc.vi = *vi;
    if (temporal)
        desc.vi.height *= 2 * T;
    desc.nodes.push_back(node);
    if (p.final_)
        desc.nodes.push_back(refNode);
    node = nullptr;
    refNode = nullptr; /* owned by desc from here, consumed by createFilter either way */

    for (int i = 0; i < 3; ++i)
        desc.process[i] = temporal ? true : p.process[i];

    /* Programs: 0 the BM3D kernel, 1 copy, 2 zero, 3 aggregate (spatial only). */
    {
        vsgpu::Program prog;
        prog.glsl = bm3dGlsl;
        prog.storageBufferCount = 2;
        prog.pushConstantBytes = sizeof(KernelParams);
        prog.localSizeX = sgSize;
        prog.localSizeY = 1;
        prog.requiredSubgroupSize = pinnedSize;
        prog.requireFullSubgroups = true;
        const uint32_t spec[4] = { temporal, p.chroma, p.final_, sgSize };
        prog.specData.assign(reinterpret_cast<const uint8_t *>(spec),
            reinterpret_cast<const uint8_t *>(spec) + sizeof(spec));
        for (uint32_t i = 0; i < 4; ++i)
            prog.specEntries.push_back({ i, i * 4, 4 });
        desc.programs.push_back(std::move(prog));
    }
    {
        vsgpu::Program prog;
        prog.glsl = copyGlsl;
        prog.storageBufferCount = 2;
        prog.pushConstantBytes = sizeof(CopyPush);
        desc.programs.push_back(std::move(prog));
    }
    {
        vsgpu::Program prog;
        prog.glsl = zeroGlsl;
        prog.storageBufferCount = 1;
        prog.pushConstantBytes = sizeof(ZeroPush);
        prog.localSizeX = 256;
        prog.localSizeY = 1;
        desc.programs.push_back(std::move(prog));
    }
    if (!temporal) {
        vsgpu::Program prog;
        prog.glsl = aggGlsl;
        prog.storageBufferCount = 2;
        prog.pushConstantBytes = sizeof(AggPush);
        desc.programs.push_back(std::move(prog));
    }

    /* Scratch 0 is the packed source stack; scratch 1 the accumulator, except for
       non-chroma temporal where the tall output plane itself accumulates. Sizes use a
       256-byte stride bound, which the core's plane stride never exceeds; runtime offsets
       use the actual stride carried in frameParams. */
    auto strideBound = [](int width) {
        return (static_cast<VkDeviceSize>(width) * 4 + 255) / 256 * 256;
    };
    VkDeviceSize srcBytes = 0, resBytes = 0;
    for (int pl = 0; pl < numPlanes; ++pl) {
        const bool heavyHere = p.chroma ? (pl == heavyGate) : p.process[pl];
        if (!heavyHere)
            continue;
        const int w = pl ? vi->width >> vi->format.subSamplingW : vi->width;
        const int h = pl ? vi->height >> vi->format.subSamplingH : vi->height;
        srcBytes = std::max(srcBytes, static_cast<VkDeviceSize>(clips) * numPack * T * h * strideBound(w));
        resBytes = std::max(resBytes, static_cast<VkDeviceSize>(numPack) * T * 2 * h * strideBound(w));
    }
    desc.scratchCount = resIsOutput ? 1 : 2;
    desc.scratchDefs.push_back({ srcBytes, 0 });
    if (!resIsOutput)
        desc.scratchDefs.push_back({ resBytes, 0 });

    /* The actual per-plane strides only exist once frames do; prepareFrame publishes them
       for fillPush and the reshape lambdas, since all-scratch passes have no plane binding
       to read a stride from. */
    desc.frameParamCount = 3;
    desc.prepareFrame = [numPlanes](int, const VSFrame *const *sources, int, const VSAPI *api,
        uint32_t *params, std::string &) {
        for (int pl = 0; pl < numPlanes && pl < 3; ++pl)
            params[pl] = static_cast<uint32_t>(api->getStride(sources[0], pl) / 4);
        return true;
    };

    auto gateHeavy = [&](vsgpu::Pass &pass) {
        for (int i = 0; i < 3; ++i)
            pass.planes[i] = p.chroma ? (i == heavyGate) : p.process[i];
    };
    /* Chroma packs use plane 0's stride slot (4:4:4, all equal); non-chroma each plane its
       own. */
    const bool chroma = p.chroma;
    auto strideSlot = [chroma](const vsgpu::PassInfo &info) { return chroma ? 0 : info.plane; };

    std::vector<PMeta> meta;

    /* Zero the accumulator: the whole res scratch (one pass, all packed planes at once),
       or the tall output plane per plane. */
    if (!resIsOutput) {
        vsgpu::Pass pass;
        pass.program = 2;
        pass.bindings.push_back(vsgpu::Operand::scratch(1));
        gateHeavy(pass);
        const int packF = numPack * T * 2;
        pass.reshape = [packF, strideSlot](vsgpu::PassInfo &info) {
            info.width = packF * info.srcHeight * info.frameParams[strideSlot(info)];
            info.height = 1;
        };
        desc.passes.push_back(std::move(pass));
        meta.push_back({ PMeta::ZeroRes, packF });
    } else {
        vsgpu::Pass pass;
        pass.program = 2;
        pass.bindings.push_back(vsgpu::Operand::output());
        const int packF = T * 2;
        pass.reshape = [packF, strideSlot](vsgpu::PassInfo &info) {
            info.width = packF * info.srcHeight * info.frameParams[strideSlot(info)];
            info.height = 1;
        };
        desc.passes.push_back(std::move(pass));
        meta.push_back({ PMeta::ZeroRes, packF });
    }
    /* Chroma temporal: sigma-zero planes of the tall output get only a zero pass. */
    if (temporal && p.chroma) {
        for (int pl = 0; pl < numPlanes; ++pl) {
            if (p.process[pl])
                continue;
            vsgpu::Pass pass;
            pass.program = 2;
            pass.bindings.push_back(vsgpu::Operand::output());
            for (int i = 0; i < 3; ++i)
                pass.planes[i] = (i == pl);
            const int packF = T * 2;
            pass.reshape = [packF, strideSlot](vsgpu::PassInfo &info) {
                info.width = packF * info.srcHeight * info.frameParams[strideSlot(info)];
                info.height = 1;
            };
            desc.passes.push_back(std::move(pass));
            meta.push_back({ PMeta::ZeroOut, packF });
        }
    }

    /* Pack the source stack: ref clip first in final mode, then per packed plane, per tap,
       exactly the Metal d_src layout. Slots advance positionally even for planes the
       kernel will skip. */
    for (int outer = 0; outer < clips; ++outer) {
        const int clipIdx = p.final_ ? (outer == 0 ? 1 : 0) : 0;
        for (int pack = 0; pack < numPack; ++pack) {
            for (int tap = 0; tap < T; ++tap) {
                vsgpu::Pass pass;
                pass.program = 1;
                pass.bindings.push_back(p.chroma
                    ? vsgpu::Operand::sourcePlane(pack, clipIdx, tap - p.radius)
                    : vsgpu::Operand::source(clipIdx, tap - p.radius));
                pass.bindings.push_back(vsgpu::Operand::scratch(0));
                pass.geometryFromBinding = 0;
                gateHeavy(pass);
                desc.passes.push_back(std::move(pass));
                meta.push_back({ PMeta::Copy, (outer * numPack + pack) * T + tap });
            }
        }
    }

    /* The kernel itself. */
    {
        vsgpu::Pass pass;
        pass.program = 0;
        pass.bindings.push_back(resIsOutput ? vsgpu::Operand::output() : vsgpu::Operand::scratch(1));
        pass.bindings.push_back(vsgpu::Operand::scratch(0));
        gateHeavy(pass);
        const std::array<int, 3> steps = p.block_step;
        const uint32_t clusters = sgSize / 8;
        pass.reshape = [steps, chroma, clusters, sgSize](vsgpu::PassInfo &info) {
            const uint32_t bs = static_cast<uint32_t>(steps[chroma ? 0 : info.plane]);
            const uint32_t gx = (info.srcWidth + clusters * bs - 1) / (clusters * bs);
            const uint32_t gy = (info.srcHeight + bs - 1) / bs;
            info.width = gx * sgSize;
            info.height = gy;
        };
        desc.passes.push_back(std::move(pass));
        meta.push_back({ PMeta::BM3D, 0 });
    }

    /* Out of the accumulator: divide for spatial output, raw slice copy for chroma
       temporal (VAggregate divides later). Non-chroma temporal accumulated in place. */
    if (!temporal) {
        vsgpu::Pass pass;
        pass.program = 3;
        pass.bindings.push_back(vsgpu::Operand::scratch(1));
        pass.bindings.push_back(vsgpu::Operand::output());
        desc.passes.push_back(std::move(pass));
        meta.push_back({ PMeta::Agg, 0 });
    } else if (p.chroma) {
        for (int pl = 0; pl < numPlanes; ++pl) {
            if (!p.process[pl])
                continue;
            vsgpu::Pass pass;
            pass.program = 1;
            pass.bindings.push_back(vsgpu::Operand::scratch(1));
            pass.bindings.push_back(vsgpu::Operand::output());
            for (int i = 0; i < 3; ++i)
                pass.planes[i] = (i == pl);
            desc.passes.push_back(std::move(pass));
            meta.push_back({ PMeta::Unpack, pl });
        }
    }

    const BM3DParams pv = p;
    const int Tv = T;
    desc.fillPush = [meta, pv, Tv, strideSlot, chroma](const vsgpu::PassInfo &info, void *pushData) {
        const PMeta &m = meta[info.pass];
        const uint32_t stride = info.frameParams[strideSlot(info)];
        const uint32_t H = info.srcHeight;
        switch (m.kind) {
        case PMeta::ZeroRes:
        case PMeta::ZeroOut: {
            ZeroPush push = { static_cast<uint32_t>(m.slotIndex) * H * stride, 0 };
            std::memcpy(pushData, &push, sizeof(push));
            break;
        }
        case PMeta::Copy: {
            CopyPush push = {};
            push.srcStride = info.strideElements[0];
            push.dstStride = stride;
            push.width = info.width;
            push.height = info.height;
            push.srcOffset = 0;
            push.dstOffset = static_cast<uint32_t>(m.slotIndex) * info.height * stride;
            std::memcpy(pushData, &push, sizeof(push));
            break;
        }
        case PMeta::BM3D: {
            const int pl = chroma ? 0 : info.plane;
            KernelParams push = {};
            push.width = static_cast<int32_t>(info.srcWidth);
            push.height = static_cast<int32_t>(H);
            push.stride = static_cast<int32_t>(stride);
            push.sigma = pv.sigma[pl];
            push.block_step = pv.block_step[pl];
            push.bm_range = pv.bm_range[pl];
            push.radius = pv.radius;
            push.ps_num = pv.ps_num[pl];
            push.ps_range = pv.ps_range[pl];
            push.sigma_u = chroma ? pv.sigma[1] : 0.0f;
            push.sigma_v = chroma ? pv.sigma[2] : 0.0f;
            push.extractor = pv.extractor;
            std::memcpy(pushData, &push, sizeof(push));
            break;
        }
        case PMeta::Agg: {
            const uint32_t packIdx = chroma ? static_cast<uint32_t>(info.plane) : 0;
            AggPush push = {};
            push.srcStride = stride;
            push.dstStride = info.dstStrideElements();
            push.width = info.width;
            push.height = info.height;
            push.srcOffset = packIdx * static_cast<uint32_t>(Tv) * 2 * H * stride;
            push.weightRel = H * stride;
            std::memcpy(pushData, &push, sizeof(push));
            break;
        }
        case PMeta::Unpack: {
            CopyPush push = {};
            push.srcStride = stride;
            push.dstStride = info.dstStrideElements();
            push.width = info.width;   /* output tall plane dims */
            push.height = info.height;
            push.srcOffset = static_cast<uint32_t>(m.slotIndex) * static_cast<uint32_t>(Tv) * 2 * H * stride;
            push.dstOffset = 0;
            std::memcpy(pushData, &push, sizeof(push));
            break;
        }
        }
    };

    const int numFrames = vi->numFrames;
    desc.mapFrame = [numFrames](int n, int, int frameOffset) {
        return std::clamp(n + frameOffset, 0, numFrames - 1);
    };

    if (temporal) {
        const int radius = p.radius;
        const std::array<bool, 3> proc = p.process;
        desc.finishFrame = [radius, proc](int, VSFrame *dst, const VSFrame *const *, int,
            const uint32_t *, VSCore *, const VSAPI *api) {
            VSMap *props = api->getFramePropertiesRW(dst);
            api->mapSetInt(props, "BM3D_V_radius", radius, maReplace);
            for (int i = 0; i < 3; ++i)
                api->mapSetInt(props, "BM3D_V_process", proc[i] ? 1 : 0, i == 0 ? maReplace : maAppend);
        };
    }

    std::vector<VSFilterDependency> deps;
    const int rp = temporal ? rpGeneral : rpStrictSpatial;
    deps.push_back({ desc.nodes[0], rp });
    if (p.final_)
        deps.push_back({ desc.nodes[1], rp });

    std::string createError;
    VSNode *result = vsgpu::createFilter("BM3D", desc, deps.data(), static_cast<int>(deps.size()),
        core, vsapi, createError);
    if (result)
        vsapi->mapConsumeNode(out, "clip", result, maAppend);
    else
        vsapi->mapSetError(out, ("BM3D: " + createError).c_str());
}

/* Temporal aggregation of BM3D's stacked output, the CPU vDSP half of the original moved
   onto the device: per output pixel, sum the wdst and weight rows of every neighbouring
   tall frame's slot for this frame and divide. */
static void VS_CC VAggregateCreate(const VSMap *in, VSMap *out, void *, VSCore *core, const VSAPI *vsapi) noexcept {
    VSNode *node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    VSNode *srcNode = vsapi->mapGetNode(in, "src", 0, nullptr);
    const VSVideoInfo *vi = vsapi->getVideoInfo(node);
    const VSVideoInfo *srcVi = vsapi->getVideoInfo(srcNode);

    const int radius = (vi->height / srcVi->height - 2) / 4;
    const int T = 2 * radius + 1;

    std::array<bool, 3> process = { false, false, false };
    for (int i = 0; i < vsapi->mapNumElements(in, "planes"); ++i) {
        const int pl = static_cast<int>(vsapi->mapGetInt(in, "planes", i, nullptr));
        if (pl >= 0 && pl < 3)
            process[pl] = true;
    }

    vsgpu::FilterDesc desc;
    desc.vi = *srcVi;
    desc.nodes.push_back(node);
    desc.nodes.push_back(srcNode);
    for (int i = 0; i < 3; ++i) {
        desc.process[i] = process[i];
        desc.shareClip[i] = 1; /* unprocessed planes come from src */
    }

    std::string glsl =
        "#version 460\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n";
    for (int i = 0; i < T; ++i)
        glsl += "layout(std430, binding = " + std::to_string(i) + ") readonly buffer S" +
            std::to_string(i) + " { float s" + std::to_string(i) + "[]; };\n";
    glsl += "layout(std430, binding = " + std::to_string(T) + ") writeonly buffer D { float d[]; };\n"
        "layout(push_constant) uniform PC { uint srcStride, dstStride, width, height; int lo, hi; } pc;\n"
        "void main() {\n"
        "    uint x = gl_GlobalInvocationID.x, y = gl_GlobalInvocationID.y;\n"
        "    if (x >= pc.width || y >= pc.height) return;\n"
        "    float sum = 0.0, wsum = 0.0;\n"
        "    int z; uint o;\n";
    for (int i = 0; i < T; ++i) {
        const std::string s = "s" + std::to_string(i);
        glsl += "    z = clamp(" + std::to_string(2 * radius - i) + ", pc.lo, pc.hi);\n"
            "    o = (uint(z) * 2u * pc.height + y) * pc.srcStride + x;\n"
            "    sum += " + s + "[o];\n"
            "    wsum += " + s + "[o + pc.height * pc.srcStride];\n";
    }
    glsl += "    d[y * pc.dstStride + x] = sum / (wsum > 0.0 ? wsum : 1.0);\n"
        "}\n";

    vsgpu::Program prog;
    prog.glsl = std::move(glsl);
    prog.storageBufferCount = T + 1;
    prog.pushConstantBytes = sizeof(VAggPush);
    desc.programs.push_back(std::move(prog));

    vsgpu::Pass pass;
    for (int i = 0; i < T; ++i)
        pass.bindings.push_back(vsgpu::Operand::source(0, i - radius));
    pass.bindings.push_back(vsgpu::Operand::output());
    desc.passes.push_back(std::move(pass));

    const int numFrames = vi->numFrames;
    desc.mapFrame = [numFrames](int n, int clip, int frameOffset) {
        return clip == 0 ? std::clamp(n + frameOffset, 0, numFrames - 1) : n;
    };

    desc.frameParamCount = 2;
    const int nf = numFrames;
    const int rad = radius;
    desc.prepareFrame = [nf, rad](int n, const VSFrame *const *, int, const VSAPI *,
        uint32_t *params, std::string &) {
        params[0] = static_cast<uint32_t>(n - nf + 1 + rad);
        params[1] = static_cast<uint32_t>(n + rad);
        return true;
    };

    desc.fillPush = [](const vsgpu::PassInfo &info, void *pushData) {
        VAggPush push = {};
        push.srcStride = info.strideElements[0];
        push.dstStride = info.dstStrideElements();
        push.width = info.width;
        push.height = info.height;
        push.lo = static_cast<int32_t>(info.frameParams[0]);
        push.hi = static_cast<int32_t>(info.frameParams[1]);
        std::memcpy(pushData, &push, sizeof(push));
    };

    VSFilterDependency deps[] = {
        { desc.nodes[0], rpGeneral },
        { desc.nodes[1], rpStrictSpatial },
    };
    std::string createError;
    VSNode *result = vsgpu::createFilter("VAggregate", desc, deps, 2, core, vsapi, createError);
    if (result)
        vsapi->mapConsumeNode(out, "clip", result, maAppend);
    else
        vsapi->mapSetError(out, ("VAggregate: " + createError).c_str());
}

/* Unchanged invoke composition from the original: BM3D, then VAggregate when temporal. */
static void VS_CC BM3Dv2Create(const VSMap *in, VSMap *out, void *, VSCore *, const VSAPI *vsapi) noexcept {
    std::array<bool, 3> process;
    process.fill(true);
    const int numSigma = vsapi->mapNumElements(in, "sigma");
    for (int i = 0; i < std::min(3, numSigma); ++i) {
        if (vsapi->mapGetFloat(in, "sigma", i, nullptr) < std::numeric_limits<float>::epsilon())
            process[i] = false;
    }
    if (numSigma > 0) {
        for (int i = numSigma; i < 3; ++i)
            process[i] = process[i - 1];
    }

    VSNode *src = vsapi->mapGetNode(in, "clip", 0, nullptr);
    const VSVideoInfo *srcVi = vsapi->getVideoInfo(src);
    bool skip = true;
    for (int i = 0; i < srcVi->format.numPlanes; ++i)
        skip &= !process[i];
    if (skip) {
        vsapi->mapConsumeNode(out, "clip", src, maReplace);
        return;
    }

    VSMap *map = vsapi->invoke(myself, "BM3D", in);
    if (vsapi->mapGetError(map)) {
        vsapi->mapSetError(out, vsapi->mapGetError(map));
        vsapi->freeMap(map);
        vsapi->freeNode(src);
        return;
    }

    int err = 0;
    int radius = static_cast<int>(vsapi->mapGetInt(in, "radius", 0, &err));
    if (err)
        radius = 0;
    if (radius == 0) {
        VSNode *node = vsapi->mapGetNode(map, "clip", 0, nullptr);
        vsapi->freeMap(map);
        vsapi->mapConsumeNode(out, "clip", node, maReplace);
        vsapi->freeNode(src);
        return;
    }

    vsapi->mapConsumeNode(map, "src", src, maReplace);
    for (int i = 0; i < 3; ++i) {
        if (process[i])
            vsapi->mapSetInt(map, "planes", i, maAppend);
    }

    VSMap *map2 = vsapi->invoke(myself, "VAggregate", map);
    vsapi->freeMap(map);
    if (vsapi->mapGetError(map2)) {
        vsapi->mapSetError(out, vsapi->mapGetError(map2));
        vsapi->freeMap(map2);
        return;
    }
    VSNode *node = vsapi->mapGetNode(map2, "clip", 0, nullptr);
    vsapi->freeMap(map2);
    vsapi->mapConsumeNode(out, "clip", node, maReplace);
}

} // namespace

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    myself = plugin;
    vspapi->configPlugin("com.sunflower-dolls.bm3dvulkan", "bm3dvk",
        "BM3D algorithm on the VapourSynth Vulkan device",
        VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    const char *bm3dArgs =
        "clip:vnode:gpu;ref:vnode:gpu:opt;sigma:float[]:opt;block_step:int[]:opt;"
        "bm_range:int[]:opt;radius:int:opt;ps_num:int[]:opt;ps_range:int[]:opt;"
        "chroma:int:opt;extractor_exp:int:opt;";
    vspapi->registerFunction("BM3D", bm3dArgs, "clip:vnode:gpu;", BM3DCreate, nullptr, plugin);
    vspapi->registerFunction("VAggregate", "clip:vnode:gpu;src:vnode:gpu;planes:int[];",
        "clip:vnode:gpu;", VAggregateCreate, nullptr, plugin);
    vspapi->registerFunction("BM3Dv2", bm3dArgs, "clip:vnode:gpu;", BM3Dv2Create, nullptr, plugin);
}
