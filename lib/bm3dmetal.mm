/**
 * Copyright (C) 2021 WolframRhodium
 * Copyright (C) 2025 Sunflower Dolls
 *
 * This file is part of Vapoursynth-BM3DMETAL.
 *
 * Vapoursynth-BM3DMETAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Vapoursynth-BM3DMETAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Vapoursynth-BM3DMETAL.  If not, see <https://www.gnu.org/licenses/>.
 */

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <VSHelper4.h>
#include <VapourSynth4.h>

using namespace std::string_literals;

static inline void encode_copy_from_vs(id<MTLComputeCommandEncoder> encoder,
                                       id<MTLDevice> device,
                                       const void* src_ptr, size_t src_len,
                                       uint src_stride_bytes,
                                       id<MTLBuffer> dst_buf, size_t dst_offset,
                                       uint dst_stride_bytes, uint width_floats,
                                       uint height) {
    id<MTLBuffer> src_buf =
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        [device newBufferWithBytesNoCopy:const_cast<void*>(src_ptr)
                                  length:src_len
                                 options:MTLResourceStorageModeShared
                             deallocator:nil];

    [encoder setBuffer:src_buf offset:0 atIndex:0];
    [encoder setBuffer:dst_buf offset:dst_offset atIndex:1];
    uint src_stride = src_stride_bytes / sizeof(float);
    uint dst_stride = dst_stride_bytes / sizeof(float);
    [encoder setBytes:&src_stride length:sizeof(uint) atIndex:2];
    [encoder setBytes:&dst_stride length:sizeof(uint) atIndex:3];
    [encoder setBytes:&width_floats length:sizeof(uint) atIndex:4];

    MTLSize grid = MTLSizeMake(width_floats, height, 1);
    MTLSize group = MTLSizeMake(std::min((int)width_floats, 32),
                                std::min((int)height, 32), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
}

static inline void encode_copy_to_vs(id<MTLComputeCommandEncoder> encoder,
                                     id<MTLDevice> device,
                                     id<MTLBuffer> src_buf, size_t src_offset,
                                     uint src_stride_bytes, void* dst_ptr,
                                     size_t dst_len, uint dst_stride,
                                     uint width_floats, uint height) {
    id<MTLBuffer> dst_buf =
        [device newBufferWithBytesNoCopy:dst_ptr
                                  length:dst_len
                                 options:MTLResourceStorageModeShared
                             deallocator:nil];

    [encoder setBuffer:src_buf offset:src_offset atIndex:0];
    [encoder setBuffer:dst_buf offset:0 atIndex:1];
    uint src_stride = src_stride_bytes / sizeof(float);
    uint dst_stride_floats = dst_stride / sizeof(float);
    [encoder setBytes:&src_stride length:sizeof(uint) atIndex:2];
    [encoder setBytes:&dst_stride_floats length:sizeof(uint) atIndex:3];
    [encoder setBytes:&width_floats length:sizeof(uint) atIndex:4];

    MTLSize grid = MTLSizeMake(width_floats, height, 1);
    MTLSize group = MTLSizeMake(std::min((int)width_floats, 32),
                                std::min((int)height, 32), 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static VSPlugin* myself = nullptr;

struct ticket_semaphore {
    std::atomic<intptr_t> ticket;
    std::atomic<intptr_t> current;

    void acquire() noexcept {
        intptr_t tk = ticket.fetch_add(1, std::memory_order::acquire);
        while (true) {
            intptr_t curr = current.load(std::memory_order::acquire);
            if (tk <= curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

struct KernelParams {
    int width;
    int height;
    int stride;
    float sigma;
    int block_step;
    int bm_range;
    int radius;
    int ps_num;
    int ps_range;
    float sigma_u;
    float sigma_v;
    float extractor;
};

struct Metal_Resource {
    id<MTLBuffer> d_src = nil;
    id<MTLBuffer> d_res = nil;
    id<MTLCommandQueue> commandQueue = nil;
    std::array<id<MTLComputePipelineState>, 8> psos = {};
    id<MTLComputePipelineState> copy_pso = nil;
    id<MTLBuffer> params = nil;

    ~Metal_Resource() {
        d_src = nil;
        d_res = nil;
        commandQueue = nil;
        params = nil;
        psos.fill(nil);
        copy_pso = nil;
    }

    Metal_Resource() = default;

    Metal_Resource(Metal_Resource&& other) noexcept
        : d_src(std::move(other.d_src)), d_res(std::move(other.d_res)),
          commandQueue(std::move(other.commandQueue)),
          psos(std::move(other.psos)), copy_pso(std::move(other.copy_pso)),
          params(std::move(other.params)) {
        other.d_src = nil;
        other.d_res = nil;
        other.commandQueue = nil;
        other.psos.fill(nil);
        other.copy_pso = nil;
        other.params = nil;
    }

    Metal_Resource& operator=(Metal_Resource&& other) noexcept {
        if (this != &other) {
            d_src = std::move(other.d_src);
            d_res = std::move(other.d_res);
            commandQueue = std::move(other.commandQueue);
            psos = std::move(other.psos);
            copy_pso = std::move(other.copy_pso);
            params = std::move(other.params);
            other.d_src = nil;
            other.d_res = nil;
            other.commandQueue = nil;
            other.psos.fill(nil);
            other.copy_pso = nil;
            other.params = nil;
        }
        return *this;
    }

    Metal_Resource(const Metal_Resource&) = delete;
    Metal_Resource& operator=(const Metal_Resource&) = delete;
};

struct MetalCaptureScope {
    MTLCaptureManager* manager = nil;
    bool active = false;

    MetalCaptureScope(id<MTLDevice> device, int n, bool final_) {
        if (std::getenv("bm3d_METAL_CAPTURE") != nullptr) {
            manager = [MTLCaptureManager sharedCaptureManager];
            if ((manager != nullptr) && !manager.isCapturing) {
                MTLCaptureDescriptor* desc =
                    [[MTLCaptureDescriptor alloc] init];
                desc.captureObject = device;
                desc.destination = MTLCaptureDestinationGPUTraceDocument;

                std::string filename =
                    "bm3dmetal_capture_" + std::to_string(n) +
                    std::to_string(static_cast<int>(final_)) + ".gputrace";
                desc.outputURL = [NSURL
                    fileURLWithPath:[NSString
                                        stringWithUTF8String:filename.c_str()]];

                NSError* error = nil;
                if ([manager startCaptureWithDescriptor:desc error:&error]) {
                    active = true;
                    std::cout << "Started Metal Capture for frame " << n
                              << "\n";
                } else {
                    std::cerr << "Failed to start Metal Capture: " <<
                        [[error localizedDescription] UTF8String] << "\n";
                    std::cerr << "Please run with environment variable "
                              << "METAL_CAPTURE_ENABLED=1" << "\n";
                }
            }
        }
    }

    ~MetalCaptureScope() {
        if (active && (manager != nullptr)) {
            [manager stopCapture];
            std::cout << "Stopped Metal Capture" << "\n";
        }
    }

    MetalCaptureScope(const MetalCaptureScope&) = delete;
    MetalCaptureScope& operator=(const MetalCaptureScope&) = delete;
    MetalCaptureScope(MetalCaptureScope&&) = delete;
    MetalCaptureScope& operator=(MetalCaptureScope&&) = delete;
};

struct BM3DData {
    VSNode* node = nullptr;
    VSNode* ref_node = nullptr;
    const VSVideoInfo* vi = nullptr;

    std::array<float, 3> sigma;
    std::array<int, 3> block_step;
    std::array<int, 3> bm_range;
    std::array<int, 3> ps_num;
    std::array<int, 3> ps_range;
    float extractor;

    int radius;
    int num_copy_engines;
    bool chroma;
    std::array<bool, 3> process;
    bool final_;
    bool zero_init;

    size_t d_pitch;
    id<MTLDevice> device = nil;

    ticket_semaphore semaphore;
    std::vector<Metal_Resource> resources;
    std::mutex resources_lock;
};

static inline void Aggregation(float* __restrict dstp, int dst_stride,
                               const float* __restrict srcp, int src_stride,
                               int width, int height) noexcept {
    const float* wdst = srcp;
    const float* weight = &srcp[(size_t)height * src_stride];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] = wdst[x] / (weight[x] > 0 ? weight[x] : 1.F);
        }
        dstp += dst_stride;
        wdst += src_stride;
        weight += src_stride;
    }
}

static const VSFrame* VS_CC BM3DGetFrame(int n, int activationReason,
                                         void* instanceData,
                                         [[maybe_unused]] void** frameData,
                                         VSFrameContext* frameCtx,
                                         VSCore* core,
                                         const VSAPI* vsapi) noexcept {
    auto* d = static_cast<BM3DData*>(instanceData);

    if (activationReason == arInitial) {
        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, d->vi->numFrames - 1);
        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        if (d->final_) {
            for (int i = start_frame; i <= end_frame; ++i) {
                vsapi->requestFrameFilter(i, d->ref_node, frameCtx);
            }
        }
    } else if (activationReason == arAllFramesReady) {

        int radius = d->radius;
        int temporal_width = (2 * radius) + 1;
        bool final_ = d->final_;
        int num_input_frames = temporal_width * (final_ ? 2 : 1);
        MetalCaptureScope capture_scope(d->device, n, final_);
        using freeFrame_t = decltype(vsapi->freeFrame);
        const auto srcs = [&]() {
            std::vector<std::unique_ptr<const VSFrame, const freeFrame_t&>>
                temp;
            temp.reserve(num_input_frames);
            if (final_) {
                for (int i = -radius; i <= radius; ++i) {
                    temp.emplace_back(
                        vsapi->getFrameFilter(
                            std::clamp(n + i, 0, d->vi->numFrames - 1),
                            d->ref_node, frameCtx),
                        vsapi->freeFrame);
                }
            }
            for (int i = -radius; i <= radius; ++i) {
                temp.emplace_back(
                    vsapi->getFrameFilter(
                        std::clamp(n + i, 0, d->vi->numFrames - 1), d->node,
                        frameCtx),
                    vsapi->freeFrame);
            }
            return temp;
        }();

        const VSFrame* src =
            srcs[radius + (final_ ? temporal_width : 0)].get();
        std::unique_ptr<VSFrame, const freeFrame_t&> dst{nullptr,
                                                         vsapi->freeFrame};

        if (radius != 0) {
            dst.reset(vsapi->newVideoFrame(&d->vi->format, d->vi->width,
                                           d->vi->height * 2 * temporal_width,
                                           src, core));
            for (int i = 0; i < d->vi->format.numPlanes; ++i) {
                if (d->zero_init && !d->process.at(i)) {
                    memset(vsapi->getWritePtr(dst.get(), i), 0,
                           (size_t)vsapi->getFrameHeight(dst.get(), i) *
                               vsapi->getStride(dst.get(), i));
                }
            }
        } else {
            std::array<const VSFrame*, 3> fr = {
                d->process[0] ? nullptr : src, d->process[1] ? nullptr : src,
                d->process[2] ? nullptr : src};
            std::array<int, 3> plane_indices = {0, 1, 2};
            dst.reset(vsapi->newVideoFrame2(&d->vi->format, d->vi->width,
                                            d->vi->height, fr.data(),
                                            plane_indices.data(), src, core));
        }

        d->semaphore.acquire();
        d->resources_lock.lock();
        auto resource = std::move(d->resources.back());
        d->resources.pop_back();
        d->resources_lock.unlock();

        @autoreleasepool {
            if (resource.d_src == nil || resource.d_res == nil ||
                resource.d_src.contents == nullptr ||
                resource.d_res.contents == nullptr) {
                d->resources_lock.lock();
                d->resources.push_back(std::move(resource));
                d->resources_lock.unlock();
                d->semaphore.release();
                vsapi->setFilterError("BM3DMetal: Invalid resource detected. "
                                      "Buffer allocation may have failed.",
                                      frameCtx);
                return nullptr;
            }

            int temporal_width = (2 * d->radius) + 1;
            int num_planes = d->chroma ? 3 : 1;
            int max_height = d->process[0] ? vsapi->getFrameHeight(src, 0)
                                           : vsapi->getFrameHeight(src, 1);
            size_t res_buffer_size = (size_t)num_planes * temporal_width * 2 *
                                     max_height * d->d_pitch;

            if (res_buffer_size > resource.d_res.length) {
                d->resources_lock.lock();
                d->resources.push_back(std::move(resource));
                d->resources_lock.unlock();
                d->semaphore.release();
                vsapi->setFilterError("BM3DMetal: Buffer size mismatch. "
                                      "Required size exceeds allocated buffer.",
                                      frameCtx);
                return nullptr;
            }

            if (d->chroma) {
                id<MTLCommandBuffer> commandBuffer =
                    [resource.commandQueue commandBuffer];
                id<MTLBlitCommandEncoder> blit =
                    [commandBuffer blitCommandEncoder];
                [blit fillBuffer:resource.d_res
                           range:NSMakeRange(0, res_buffer_size)
                           value:0];
                [blit endEncoding];
                id<MTLComputeCommandEncoder> encoder =
                    [commandBuffer computeCommandEncoder];

                [encoder setComputePipelineState:resource.copy_pso];

                size_t d_src_offset = 0;

                for (int outer = 0; outer < (final_ ? 2 : 1); ++outer) {
                    for (int i = 0; i < 3; ++i) {
                        int height = vsapi->getFrameHeight(src, i);
                        for (int j = 0; j < temporal_width; ++j) {
                            if (i == 0 || d->process.at(i)) {
                                const VSFrame* frame =
                                    srcs[j + (outer * temporal_width)].get();
                                int s_pitch = vsapi->getStride(frame, i);
                                int width =
                                    vsapi->getFrameWidth(frame, i);
                                const auto* p_read =
                                    vsapi->getReadPtr(frame, i);

                                encode_copy_from_vs(encoder, d->device, p_read,
                                                    (size_t)height * s_pitch,
                                                    s_pitch,
                                                    resource.d_src, d_src_offset,
                                                    d->d_pitch,
                                                    static_cast<uint>(width),
                                                    height);
                            }
                            d_src_offset += d->d_pitch * height;
                        }
                    }
                }

                int width = vsapi->getFrameWidth(src, 0);
                int height = vsapi->getFrameHeight(src, 0);
                int d_stride = static_cast<int>(d->d_pitch / sizeof(float));

                int pso_idx = (static_cast<int>(d->radius > 0) * 4) + (1 * 2) +
                              static_cast<int>(d->final_);
                [encoder setComputePipelineState:resource.psos.at(pso_idx)];

                [encoder setBuffer:resource.d_res offset:0 atIndex:0];
                [encoder setBuffer:resource.d_src offset:0 atIndex:1];

                auto* params =
                    static_cast<KernelParams*>([resource.params contents]);
                params->width = width;
                params->height = height;
                params->stride = d_stride;
                params->sigma = d->sigma[0];
                params->block_step = d->block_step[0];
                params->bm_range = d->bm_range[0];
                params->radius = d->radius;
                params->ps_num = d->ps_num[0];
                params->ps_range = d->ps_range[0];
                params->sigma_u = d->sigma[1];
                params->sigma_v = d->sigma[2];
                params->extractor = d->extractor;

                [encoder setBuffer:resource.params offset:0 atIndex:2];
                [encoder setThreadgroupMemoryLength:sizeof(float) * 8 * (32 + 1)
                                            atIndex:0];

                MTLSize gridDim = MTLSizeMake(
                    (width + (4 * d->block_step[0] - 1)) /
                        (4 * d->block_step[0]),
                    (height + (d->block_step[0] - 1)) / d->block_step[0], 1);
                [encoder dispatchThreadgroups:gridDim
                        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [encoder endEncoding];

                if (radius != 0) {
                    id<MTLComputeCommandEncoder> outEnc =
                        [commandBuffer computeCommandEncoder];
                    [outEnc setComputePipelineState:resource.copy_pso];

                    size_t h_dst_offset_bytes = 0;
                    const size_t plane_slice_bytes =
                        (size_t)d_stride * height * 2 * temporal_width *
                        sizeof(float);
                    for (int plane = 0; plane < 3; ++plane) {
                        if (!d->process.at(plane)) {
                            h_dst_offset_bytes += plane_slice_bytes;
                            continue;
                        }
                        const int dst_pitch = vsapi->getStride(dst.get(), plane);
                        const int out_height =
                            vsapi->getFrameHeight(src, plane) * 2 *
                            temporal_width;
                        const int width =
                            vsapi->getFrameWidth(src, plane);

                        encode_copy_to_vs(
                            outEnc, d->device, resource.d_res,
                            h_dst_offset_bytes, d->d_pitch,
                            vsapi->getWritePtr(dst.get(), plane),
                            (size_t)out_height * dst_pitch, dst_pitch,
                            static_cast<uint>(width), out_height);
                        h_dst_offset_bytes += plane_slice_bytes;
                    }
                    [outEnc endEncoding];
                }

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                if (radius == 0) {
                    auto* h_dst = static_cast<float*>(resource.d_res.contents);
                    for (int plane = 0; plane < 3; ++plane) {
                        if (!d->process.at(plane)) {
                            h_dst +=
                                (size_t)d_stride * height * 2 * temporal_width;
                            continue;
                        }
                        const int dst_pitch =
                            vsapi->getStride(dst.get(), plane);
                        Aggregation(reinterpret_cast<float*>(
                                        vsapi->getWritePtr(dst.get(), plane)),
                                    static_cast<int>(dst_pitch / sizeof(float)),
                                    h_dst, d_stride, width, height);
                        h_dst +=
                            (size_t)d_stride * height * 2 * temporal_width;
                    }
                }
            } else {
                for (int plane = 0; plane < d->vi->format.numPlanes; ++plane) {
                    if (!d->process.at(plane)) {
                        continue;
                    }

                    int width = vsapi->getFrameWidth(src, plane);
                    int height = vsapi->getFrameHeight(src, plane);
                    int s_pitch = vsapi->getStride(src, plane);
                    int dst_pitch = vsapi->getStride(dst.get(), plane);
                    int d_stride = static_cast<int>(d->d_pitch / sizeof(float));

                    size_t res_size =
                        (size_t)temporal_width * 2 * height * d->d_pitch;

                    if (res_size > resource.d_res.length) {
                        d->resources_lock.lock();
                        d->resources.push_back(std::move(resource));
                        d->resources_lock.unlock();
                        d->semaphore.release();
                        vsapi->setFilterError(
                            "BM3DMetal: Buffer size mismatch for plane. "
                            "Required size exceeds allocated buffer.",
                            frameCtx);
                        return nullptr;
                    }

                    id<MTLCommandBuffer> commandBuffer =
                        [resource.commandQueue commandBuffer];
                    id<MTLBlitCommandEncoder> blit =
                        [commandBuffer blitCommandEncoder];
                    [blit fillBuffer:resource.d_res
                               range:NSMakeRange(0, res_size)
                               value:0];
                    [blit endEncoding];
                    id<MTLComputeCommandEncoder> encoder =
                        [commandBuffer computeCommandEncoder];

                    [encoder setComputePipelineState:resource.copy_pso];

                    size_t d_src_offset = 0;
                    for (int i = 0; i < num_input_frames; ++i) {
                        const VSFrame* frame = srcs[i].get();
                        encode_copy_from_vs(
                            encoder, d->device, vsapi->getReadPtr(frame, plane),
                            (size_t)height * s_pitch, s_pitch, resource.d_src,
                            d_src_offset, d->d_pitch, static_cast<uint>(width),
                            height);
                        d_src_offset += d->d_pitch * height;
                    }

                    int pso_idx = (static_cast<int>(d->radius > 0) * 4) +
                                  (0 * 2) + static_cast<int>(d->final_);
                    [encoder setComputePipelineState:resource.psos.at(pso_idx)];

                    [encoder setBuffer:resource.d_res offset:0 atIndex:0];
                    [encoder setBuffer:resource.d_src offset:0 atIndex:1];

                    auto* params =
                        static_cast<KernelParams*>([resource.params contents]);
                    params->width = width;
                    params->height = height;
                    params->stride = d_stride;
                    params->sigma = d->sigma.at(plane);
                    params->block_step = d->block_step.at(plane);
                    params->bm_range = d->bm_range.at(plane);
                    params->radius = d->radius;
                    params->ps_num = d->ps_num.at(plane);
                    params->ps_range = d->ps_range.at(plane);
                    params->sigma_u = 0.F;
                    params->sigma_v = 0.F;
                    params->extractor = d->extractor;

                    [encoder setBuffer:resource.params offset:0 atIndex:2];
                    [encoder
                        setThreadgroupMemoryLength:sizeof(float) * 8 * (32 + 1)
                                           atIndex:0];

                    MTLSize gridDim = MTLSizeMake(
                        (width + (4 * d->block_step.at(plane) - 1)) /
                            (4 * d->block_step.at(plane)),
                        (height + (d->block_step.at(plane) - 1)) /
                            d->block_step.at(plane),
                        1);
                    [encoder dispatchThreadgroups:gridDim
                            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    [encoder endEncoding];

                    if (radius != 0) {
                        id<MTLComputeCommandEncoder> outEnc =
                            [commandBuffer computeCommandEncoder];
                        [outEnc setComputePipelineState:resource.copy_pso];

                        int out_height = vsapi->getFrameHeight(src, plane) * 2 *
                                         temporal_width;
                        int out_width = vsapi->getFrameWidth(src, plane);

                        encode_copy_to_vs(outEnc, d->device, resource.d_res, 0,
                                          d->d_pitch,
                                          vsapi->getWritePtr(dst.get(), plane),
                                          (size_t)out_height * dst_pitch,
                                          dst_pitch,
                                          static_cast<uint>(out_width),
                                          out_height);
                        [outEnc endEncoding];
                    }

                    [commandBuffer commit];
                    [commandBuffer waitUntilCompleted];

                    if (radius == 0) {
                        auto* h_dst =
                            static_cast<float*>(resource.d_res.contents);
                        auto* dstp = reinterpret_cast<float*>(
                            vsapi->getWritePtr(dst.get(), plane));
                        Aggregation(
                            dstp, static_cast<int>(dst_pitch / sizeof(float)),
                            h_dst, d_stride, width, height);
                    }
                }
            }
        }

        d->resources_lock.lock();
        d->resources.push_back(std::move(resource));
        d->resources_lock.unlock();
        d->semaphore.release();

        if (radius != 0) {
            VSMap* dst_prop = vsapi->getFramePropertiesRW(dst.get());
            vsapi->mapSetInt(dst_prop, "BM3D_V_radius", d->radius, maReplace);
            std::array<int64_t, 3> process = {
                static_cast<int64_t>(d->process[0]),
                static_cast<int64_t>(d->process[1]),
                static_cast<int64_t>(d->process[2])};
            vsapi->mapSetIntArray(dst_prop, "BM3D_V_process", process.data(),
                                  3);
        }

        return dst.release();
    }

    return nullptr;
}

static void VS_CC BM3DFree(void* instanceData, [[maybe_unused]] VSCore* core,
                           const VSAPI* vsapi) noexcept {
    auto d = std::unique_ptr<BM3DData>(static_cast<BM3DData*>(instanceData));

    @autoreleasepool {
        d->resources.clear();
        d->device = nil;
    }

    vsapi->freeNode(d->node);
    if (d->ref_node != nullptr) {
        vsapi->freeNode(d->ref_node);
    }
}

static void VS_CC BM3DCreate(const VSMap* in, VSMap* out,
                             [[maybe_unused]] void* userData, VSCore* core,
                             const VSAPI* vsapi) noexcept {
    auto d = std::make_unique<BM3DData>();
    int err = 0;

    auto set_error = [&](const std::string& errorMessage) {
        vsapi->mapSetError(out, ("BM3DMetal: " + errorMessage).c_str());
        d->resources.clear();
        d->device = nil;
        if (d->node) {
            vsapi->freeNode(d->node);
        }
        if (d->ref_node) {
            vsapi->freeNode(d->ref_node);
        }
    };

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    if (d->vi->format.sampleType != stFloat ||
        d->vi->format.bitsPerSample != 32 ||
        !vsh::isConstantVideoFormat(d->vi)) {
        set_error("only constant format 32bit float input supported");
        return;
    }

    d->ref_node = vsapi->mapGetNode(in, "ref", 0, &err);
    d->final_ = (err == 0);
    if (d->final_) {
        const VSVideoInfo* ref_vi = vsapi->getVideoInfo(d->ref_node);
        if (!vsh::isSameVideoFormat(&ref_vi->format, &d->vi->format) ||
            ref_vi->width != d->vi->width || ref_vi->height != d->vi->height ||
            ref_vi->numFrames != d->vi->numFrames) {
            set_error("ref clip properties must match input clip");
            return;
        }
    }

    for (int i = 0; i < 3; ++i) {
        d->sigma.at(i) =
            static_cast<float>(vsapi->mapGetFloat(in, "sigma", i, &err));
        if (err != 0) {
            d->sigma.at(i) = (i == 0) ? 3.0F : d->sigma.at(i - 1);
        }
        if (d->sigma.at(i) < 0.0F) {
            set_error("sigma must be non-negative");
            return;
        }
        d->process.at(i) =
            d->sigma.at(i) >= std::numeric_limits<float>::epsilon();
        d->sigma.at(i) *=
            (3.0F / 4.0F) / 255.0F * 64.0F * (d->final_ ? 1.0F : 2.7F);
    }

    for (int i = 0; i < 3; ++i) {
        d->block_step.at(i) =
            static_cast<int>(vsapi->mapGetInt(in, "block_step", i, &err));
        if (err != 0) {
            d->block_step.at(i) = (i == 0) ? 8 : d->block_step.at(i - 1);
        }
        if (d->block_step.at(i) <= 0 || d->block_step.at(i) > 8) {
            set_error("block_step must be in [1, 8]");
            return;
        }
    }

    for (int i = 0; i < 3; ++i) {
        d->bm_range.at(i) =
            static_cast<int>(vsapi->mapGetInt(in, "bm_range", i, &err));
        if (err != 0) {
            d->bm_range.at(i) = (i == 0) ? 9 : d->bm_range.at(i - 1);
        }
        if (d->bm_range.at(i) <= 0) {
            set_error("bm_range must be positive");
            return;
        }
    }

    d->radius = static_cast<int>(vsapi->mapGetInt(in, "radius", 0, &err));
    if (err != 0) {
        d->radius = 0;
    }
    if (d->radius < 0) {
        set_error("radius must be non-negative");
        return;
    }

    for (int i = 0; i < 3; ++i) {
        d->ps_num.at(i) =
            static_cast<int>(vsapi->mapGetInt(in, "ps_num", i, &err));
        if (err != 0) {
            d->ps_num.at(i) = (i == 0) ? 2 : d->ps_num.at(i - 1);
        }
        if (d->ps_num.at(i) <= 0 || d->ps_num.at(i) > 8) {
            set_error("ps_num must be in [1, 8]");
            return;
        }
    }

    for (int i = 0; i < 3; ++i) {
        d->ps_range.at(i) =
            static_cast<int>(vsapi->mapGetInt(in, "ps_range", i, &err));
        if (err != 0) {
            d->ps_range.at(i) = (i == 0) ? 4 : d->ps_range.at(i - 1);
        }
        if (d->ps_range.at(i) <= 0) {
            set_error("ps_range must be positive");
            return;
        }
    }

    d->chroma = (vsapi->mapGetInt(in, "chroma", 0, &err) != 0);
    if (d->chroma &&
        !vsh::isSameVideoPresetFormat(pfYUV444PS, &d->vi->format, core,
                                      vsapi)) {
        set_error("chroma=true requires YUV444PS");
        return;
    }

    int device_id =
        static_cast<int>(vsapi->mapGetInt(in, "device_id", 0, &err));
    if (err != 0) {
        device_id = 0;
    }

    int extractor_exp =
        static_cast<int>(vsapi->mapGetInt(in, "extractor_exp", 0, &err));
    if (err == 0) {
        d->extractor =
            (extractor_exp != 0) ? std::ldexp(1.0F, extractor_exp) : 0.0F;
    } else {
        d->extractor = 0.0F;
    }

    d->zero_init = (vsapi->mapGetInt(in, "zero_init", 0, &err) != 0);
    if (err != 0) {
        d->zero_init = true;
    }

    std::string error_message;
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (device_id >= static_cast<int>(devices.count)) {
            set_error("invalid device_id");
            return;
        }
        d->device = devices[device_id];
        // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
        const char metal_source_chars[] = {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc23-extensions"
#embed "shader.metal"
#pragma clang diagnostic pop
        };
        NSString* metalSource = [[NSString alloc]
            initWithBytes:static_cast<const void*>(metal_source_chars)
                   length:sizeof(metal_source_chars)
                 encoding:NSUTF8StringEncoding];
        NSError* error = nil;
        id<MTLLibrary> library = [d->device newLibraryWithSource:metalSource
                                                         options:nil
                                                           error:&error];
        if (library == nullptr) {
            set_error("Failed to compile metal library: "s +
                      [error.localizedDescription UTF8String]);
            return;
        }

        bool fast = (vsapi->mapGetInt(in, "fast", 0, &err) != 0);
        if (err != 0) {
            fast = false;
        }
        d->num_copy_engines = fast ? 4 : 1;
        d->semaphore.current.store(d->num_copy_engines - 1,
                                   std::memory_order::relaxed);
        d->resources.reserve(d->num_copy_engines);

        int max_width = d->process[0]
                            ? d->vi->width
                            : d->vi->width >> d->vi->format.subSamplingW;
        int max_height = d->process[0]
                             ? d->vi->height
                             : d->vi->height >> d->vi->format.subSamplingH;
        int num_planes = d->chroma ? 3 : 1;
        int temporal_width = (2 * d->radius) + 1;

        size_t min_pitch = max_width * sizeof(float);
        d->d_pitch = ((min_pitch + 255) / 256) * 256;

        size_t src_buffer_height =
            (d->final_ ? 2 : 1) * num_planes * temporal_width * max_height;
        size_t res_buffer_height = num_planes * temporal_width * 2 * max_height;

        for (int i = 0; i < d->num_copy_engines; ++i) {
            Metal_Resource res;
            size_t src_alloc_size = src_buffer_height * d->d_pitch;
            res.d_src =
                [d->device newBufferWithLength:src_alloc_size
                                       options:MTLResourceStorageModeShared];
            if (res.d_src == nil) {
                error_message = "Failed to allocate source buffer (size: "s +
                                std::to_string(src_alloc_size) +
                                " bytes). Out of memory?";
                break;
            }

            size_t res_alloc_size = res_buffer_height * d->d_pitch;
            res.d_res =
                [d->device newBufferWithLength:res_alloc_size
                                       options:MTLResourceStorageModeShared];
            if (res.d_res == nil) {
                error_message = "Failed to allocate result buffer (size: "s +
                                std::to_string(res_alloc_size) +
                                " bytes). Out of memory?";
                break;
            }
            res.commandQueue = [d->device newCommandQueue];
            if (res.commandQueue == nil) {
                set_error("Failed to create command queue");
                return;
            }

            res.params =
                [d->device newBufferWithLength:sizeof(KernelParams)
                                       options:MTLResourceStorageModeShared];
            if (res.params == nil) {
                set_error("Failed to allocate parameters buffer");
                return;
            }

            std::array<const char*, 8> kernel_names = {
                "bm3d_false_false_false", "bm3d_false_false_true",
                "bm3d_false_true_false",  "bm3d_false_true_true",
                "bm3d_true_false_false",  "bm3d_true_false_true",
                "bm3d_true_true_false",   "bm3d_true_true_true"};

            for (int k = 0; k < 8; ++k) {
                id<MTLFunction> func = [library
                    newFunctionWithName:[NSString
                                            stringWithUTF8String:kernel_names
                                                                     .at(k)]];
                if (func == nil) {
                    set_error("Failed to get kernel function "s +
                              kernel_names.at(k));
                    return;
                }
                res.psos.at(k) =
                    [d->device newComputePipelineStateWithFunction:func
                                                             error:&error];
                if (error != nil) {
                    set_error("Failed to create PSO: "s +
                              [error.localizedDescription UTF8String]);
                    return;
                }
            }

            id<MTLFunction> copy_func = [library
                newFunctionWithName:[NSString
                                        stringWithUTF8String:"copy_kernel"]];
            if (copy_func == nil) {
                set_error("Failed to get kernel function copy_kernel");
                return;
            }
            res.copy_pso =
                [d->device newComputePipelineStateWithFunction:copy_func
                                                         error:&error];
            if (error != nil) {
                set_error("Failed to create PSO for copy_kernel: "s +
                          [error.localizedDescription UTF8String]);
                return;
            }

            d->resources.push_back(std::move(res));
        }
    }

    if (!error_message.empty()) {
        set_error(error_message);
        return;
    }

    VSVideoInfo out_vi = *d->vi;
    if (d->radius != 0) {
        out_vi.height *= 2 * (2 * d->radius + 1);
    }
    std::array<VSFilterDependency, 2> deps = {{
        {d->node, rpGeneral},
        {d->ref_node, rpGeneral},
    }};
    vsapi->createVideoFilter(out, "BM3D", &out_vi, BM3DGetFrame, BM3DFree,
                             fmParallel, deps.data(), d->final_ ? 2 : 1,
                             d.release(), core);
}

struct VAggregateData {
    VSNode* node = nullptr;
    VSNode* src_node = nullptr;
    const VSVideoInfo* src_vi = nullptr;
    std::array<bool, 3> process;
    int radius;
    std::unordered_map<std::thread::id, std::vector<float>> buffer;
    std::shared_mutex buffer_lock;
};

static const VSFrame* VS_CC
VAggregateGetFrame(int n, int activationReason, void* instanceData,
                   [[maybe_unused]] void** frameData, VSFrameContext* frameCtx,
                   VSCore* core, const VSAPI* vsapi) {
    auto* d = static_cast<VAggregateData*>(instanceData);

    if (activationReason == arInitial) {
        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, d->src_vi->numFrames - 1);
        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        vsapi->requestFrameFilter(n, d->src_node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame* src_frame =
            vsapi->getFrameFilter(n, d->src_node, frameCtx);
        std::vector<const VSFrame*> vbm3d_frames;
        vbm3d_frames.reserve((2 * d->radius) + 1);
        for (int i = n - d->radius; i <= n + d->radius; ++i) {
            vbm3d_frames.emplace_back(vsapi->getFrameFilter(
                std::clamp(i, 0, d->src_vi->numFrames - 1), d->node, frameCtx));
        }

        float* buffer{};
        {
            const auto thread_id = std::this_thread::get_id();
            bool init = true;
            {
                std::shared_lock _{d->buffer_lock};
                if (!d->buffer.contains(thread_id)) {
                    init = false;
                } else {
                    buffer = d->buffer.at(thread_id).data();
                }
            }
            if (!init) {
                const int max_width = d->process[0]
                                          ? vsapi->getFrameWidth(src_frame, 0)
                                          : vsapi->getFrameWidth(src_frame, 1);
                std::vector<float> owned_buffer(2 * max_width);
                buffer = owned_buffer.data();
                std::lock_guard _{d->buffer_lock};
                d->buffer.emplace(thread_id, std::move(owned_buffer));
            }
        }

        std::array<const VSFrame*, 3> fr = {
            d->process[0] ? nullptr : src_frame,
            d->process[1] ? nullptr : src_frame,
            d->process[2] ? nullptr : src_frame};
        std::array<int, 3> plane_indices = {0, 1, 2};
        auto* dst_frame = vsapi->newVideoFrame2(
            &d->src_vi->format, d->src_vi->width, d->src_vi->height,
            fr.data(), plane_indices.data(), src_frame, core);

        for (int plane = 0; plane < d->src_vi->format.numPlanes; ++plane) {
            if (d->process.at(plane)) {
                int plane_width = vsapi->getFrameWidth(src_frame, plane);
                int plane_height = vsapi->getFrameHeight(src_frame, plane);
                int plane_stride = static_cast<int>(
                    vsapi->getStride(src_frame, plane) / sizeof(float));

                std::vector<const float*> srcps;
                srcps.reserve((2 * d->radius) + 1);
                for (int i = 0; i < (2 * d->radius) + 1; ++i) {
                    srcps.emplace_back(reinterpret_cast<const float*>(
                        vsapi->getReadPtr(vbm3d_frames[i], plane)));
                }

                auto* dstp = reinterpret_cast<float*>(
                    vsapi->getWritePtr(dst_frame, plane));

                float* sum_buffer = buffer;
                float* weight_buffer = buffer + plane_width;

                for (int y = 0; y < plane_height; ++y) {
                    {
                        const float* agg_src =
                            srcps[0] +
                            ((size_t)((std::clamp((2 * d->radius),
                                                  n - d->src_vi->numFrames + 1 +
                                                      d->radius,
                                                  n + d->radius) *
                                       2 * plane_height) +
                                      y) *
                             plane_stride);
                        vDSP_mmov(agg_src, sum_buffer,
                                  static_cast<vDSP_Length>(plane_width), 1, 1,
                                  1);
                        agg_src += (size_t)plane_height * plane_stride;
                        vDSP_mmov(agg_src, weight_buffer,
                                  static_cast<vDSP_Length>(plane_width), 1, 1,
                                  1);
                    }

                    for (int i = 1; i < (2 * d->radius) + 1; ++i) {
                        const float* agg_src =
                            srcps[i] +
                            ((size_t)((std::clamp((2 * d->radius) - i,
                                                  n - d->src_vi->numFrames + 1 +
                                                      d->radius,
                                                  n + d->radius) *
                                       2 * plane_height) +
                                      y) *
                             plane_stride);
                        vDSP_vadd(sum_buffer, 1, agg_src, 1, sum_buffer, 1,
                                  static_cast<vDSP_Length>(plane_width));
                        agg_src += (size_t)plane_height * plane_stride;
                        vDSP_vadd(weight_buffer, 1, agg_src, 1, weight_buffer,
                                  1, static_cast<vDSP_Length>(plane_width));
                    }

                    for (int x = 0; x < plane_width; ++x) {
                        dstp[x] =
                            sum_buffer[x] /
                            (weight_buffer[x] > 0 ? weight_buffer[x] : 1.F);
                    }
                    dstp += plane_stride;
                }
            }
        }

        for (const auto& frame : vbm3d_frames) {
            vsapi->freeFrame(frame);
        }
        vsapi->freeFrame(src_frame);
        return dst_frame;
    }
    return nullptr;
}

static void VS_CC VAggregateFree(void* instanceData,
                                 [[maybe_unused]] VSCore* core,
                                 const VSAPI* vsapi) noexcept {
    auto d = std::unique_ptr<VAggregateData>(
        static_cast<VAggregateData*>(instanceData));
    vsapi->freeNode(d->src_node);
    vsapi->freeNode(d->node);
}

static void VS_CC VAggregateCreate(const VSMap* in, VSMap* out,
                                   [[maybe_unused]] void* userData,
                                   VSCore* core, const VSAPI* vsapi) {
    auto d = std::make_unique<VAggregateData>();
    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    const auto* vi = vsapi->getVideoInfo(d->node);
    d->src_node = vsapi->mapGetNode(in, "src", 0, nullptr);
    d->src_vi = vsapi->getVideoInfo(d->src_node);
    d->radius = (vi->height / d->src_vi->height - 2) / 4;
    d->process.fill(false);
    for (int i = 0; i < vsapi->mapNumElements(in, "planes"); ++i) {
        int plane =
            static_cast<int>(vsapi->mapGetInt(in, "planes", i, nullptr));
        if (plane >= 0 && plane < static_cast<int>(d->process.size())) {
            d->process.at(plane) = true;
        }
    }
    VSCoreInfo core_info;
    vsapi->getCoreInfo(core, &core_info);
    d->buffer.reserve(core_info.numThreads);
    std::array<VSFilterDependency, 2> deps = {{
        {d->node, rpGeneral},
        {d->src_node, rpStrictSpatial},
    }};
    vsapi->createVideoFilter(out, "VAggregate", d->src_vi, VAggregateGetFrame,
                             VAggregateFree, fmParallel, deps.data(),
                             static_cast<int>(deps.size()), d.release(), core);
}

static void VS_CC BM3Dv2Create(const VSMap* in, VSMap* out,
                               [[maybe_unused]] void* userData,
                               [[maybe_unused]] VSCore* core,
                               const VSAPI* vsapi) {
    std::array<bool, 3> process;
    process.fill(true);
    int num_sigma_args = vsapi->mapNumElements(in, "sigma");
    for (int i = 0; i < std::min(3, num_sigma_args); ++i) {
        if (vsapi->mapGetFloat(in, "sigma", i, nullptr) <
            std::numeric_limits<float>::epsilon()) {
            process.at(i) = false;
        }
    }
    if (num_sigma_args > 0) {
        for (int i = num_sigma_args; i < 3; ++i) {
            process.at(i) = process.at(i - 1);
        }
    }

    bool skip = true;
    auto* src = vsapi->mapGetNode(in, "clip", 0, nullptr);
    const auto* src_vi = vsapi->getVideoInfo(src);
    for (int i = 0; i < src_vi->format.numPlanes; ++i) {
        skip &= !process.at(i);
    }
    if (skip) {
        vsapi->mapSetNode(out, "clip", src, maReplace);
        vsapi->freeNode(src);
        return;
    }

    auto* map = vsapi->invoke(myself, "BM3D", in);
    if (vsapi->mapGetError(map) != nullptr) {
        vsapi->mapSetError(out, vsapi->mapGetError(map));
        vsapi->freeMap(map);
        vsapi->freeNode(src);
        return;
    }

    int err = 0;
    int radius = static_cast<int>(vsapi->mapGetInt(in, "radius", 0, &err));
    if (err != 0) {
        radius = 0;
    }
    if (radius == 0) {
        auto* node = vsapi->mapGetNode(map, "clip", 0, nullptr);
        vsapi->freeMap(map);
        vsapi->mapSetNode(out, "clip", node, maReplace);
        vsapi->freeNode(node);
        vsapi->freeNode(src);
        return;
    }

    vsapi->mapSetNode(map, "src", src, maReplace);
    vsapi->freeNode(src);
    for (int i = 0; i < 3; ++i) {
        if (process.at(i)) {
            vsapi->mapSetInt(map, "planes", i, maAppend);
        }
    }

    auto* map2 = vsapi->invoke(myself, "VAggregate", map);
    vsapi->freeMap(map);
    if (vsapi->mapGetError(map2) != nullptr) {
        vsapi->mapSetError(out, vsapi->mapGetError(map2));
        vsapi->freeMap(map2);
        return;
    }

    auto* node = vsapi->mapGetNode(map2, "clip", 0, nullptr);
    vsapi->freeMap(map2);
    vsapi->mapSetNode(out, "clip", node, maReplace);
    vsapi->freeNode(node);
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    myself = plugin;
    vspapi->configPlugin("com.Sunflower-Dolls.bm3dmetal", "bm3dmetal",
                         "BM3D algorithm implemented in Metal",
                         VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0,
                         plugin);
    const char* bm3d_args = "clip:vnode;ref:vnode:opt;sigma:float[]:opt;block_"
                            "step:int[]:opt;bm_range:"
                            "int[]:opt;radius:int:opt;ps_num:int[]:opt;ps_"
                            "range:int[]:opt;chroma:int:"
                            "opt;device_id:int:opt;fast:int:opt;extractor_exp:"
                            "int:opt;zero_init:int:"
                            "opt;";
    vspapi->registerFunction("BM3D", bm3d_args, "clip:vnode;", BM3DCreate,
                             nullptr, plugin);
    vspapi->registerFunction("VAggregate",
                             "clip:vnode;src:vnode;planes:int[];",
                             "clip:vnode;", VAggregateCreate, nullptr, plugin);
    vspapi->registerFunction("BM3Dv2", bm3d_args, "clip:vnode;", BM3Dv2Create,
                             nullptr, plugin);
}
