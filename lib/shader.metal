/**
 * Copyright (C) 2003, 2007-14 Matteo Frigo
 * Copyright (C) 2003, 2007-14 Massachusetts Institute of Technology
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

#include <metal_stdlib>

using namespace metal;

template <typename T>
METAL_FUNC T metal_shfl_xor_sync(T value, int laneMask, int width,
                                 ushort tid_in_simdgroup) {
    ushort source_lane = tid_in_simdgroup ^ ushort(laneMask);
    if ((tid_in_simdgroup / width) == (source_lane / width)) {
        return simd_shuffle_xor(value, ushort(laneMask));
    }
    return value;
}

template <typename T>
METAL_FUNC T metal_shfl_up_sync(T value, uint delta, int width,
                                ushort tid_in_simdgroup) {
    if ((tid_in_simdgroup % width) < delta) {
        return value;
    }
    return simd_shuffle_up(value, delta);
}

template <>
METAL_FUNC bool metal_shfl_up_sync<bool>(bool value, uint delta, int width,
                                         ushort tid_in_simdgroup) {
    if ((tid_in_simdgroup % width) < delta) {
        return value;
    }
    int int_value = value ? 1 : 0;
    int result = simd_shuffle_up(int_value, delta);
    return result != 0;
}

template <typename T>
METAL_FUNC T metal_shfl_sync(T value, int srcLane, int width,
                             ushort tid_in_simdgroup) {
    ushort partition_start = (tid_in_simdgroup / width) * width;
    ushort absolute_src_lane = partition_start + ushort(srcLane);
    return simd_broadcast(value, absolute_src_lane);
}

#define FMA(a, b, c) fma(a, b, c)
#define FMS(a, b, c) fma(a, b, -c)
#define FNMS(a, b, c) fma(-a, b, c)

constant int smem_stride = 32 + 1;

METAL_FUNC void atomic_add_float(device float *addr, float val) {
    device atomic_uint *atomic_addr = (device atomic_uint *)addr;
    uint expected = atomic_load_explicit(atomic_addr, memory_order_relaxed);
    uint desired;
    float current;

    do {
        current = as_type<float>(expected);
        desired = as_type<uint>(current + val);
    } while (!atomic_compare_exchange_weak_explicit(
        atomic_addr, &expected, desired, memory_order_relaxed,
        memory_order_relaxed));
}

// Forward declaration for DCT functions
template <bool forward> void dct(thread float v[8]);

template <bool forward, int stride = 1, int howmany = 8, int howmany_stride = 8>
METAL_FUNC void transform_pack8_interleave4(thread float *data,
                                            threadgroup float *buffer,
                                            ushort tid_in_simdgroup) {

    for (int iter = 0; iter < howmany; ++iter, data += howmany_stride) {
        float v[8];

        for (int i = 0; i < 8; ++i) {
            v[i] = data[i * stride];
        }

        dct<forward>(v);

        for (int i = 0; i < 8; ++i) {
            data[i * stride] = v[i];
        }
    }
}

template <bool forward> METAL_FUNC void dct(thread float v[8]) {
    if constexpr (forward) {
        constexpr float KP414213562 =
            +0.414213562373095048801688724209698078569671875f;
        constexpr float KP1_847759065 =
            +1.847759065022573512256366378793576573644833252f;
        constexpr float KP198912367 =
            +0.198912367379658006911597622644676228597850501f;
        constexpr float KP1_961570560 =
            +1.961570560806460898252364472268478073947867462f;
        constexpr float KP1_414213562 =
            +1.414213562373095048801688724209698078569671875f;
        constexpr float KP668178637 =
            +0.668178637919298919997757686523080761552472251f;
        constexpr float KP1_662939224 =
            +1.662939224605090474157576755235811513477121624f;
        constexpr float KP707106781 =
            +0.707106781186547524400844362104849039284835938f;

        auto T1 = v[0];
        auto T2 = v[7];
        auto T3 = T1 - T2;
        auto Tj = T1 + T2;
        auto Tc = v[4];
        auto Td = v[3];
        auto Te = Tc - Td;
        auto Tk = Tc + Td;
        auto T4 = v[2];
        auto T5 = v[5];
        auto T6 = T4 - T5;
        auto T7 = v[1];
        auto T8 = v[6];
        auto T9 = T7 - T8;
        auto Ta = T6 + T9;
        auto Tn = T7 + T8;
        auto Tf = T6 - T9;
        auto Tm = T4 + T5;
        auto Tb = FNMS(KP707106781, Ta, T3);
        auto Tg = FNMS(KP707106781, Tf, Te);
        v[3] = KP1_662939224 * (FMA(KP668178637, Tg, Tb));
        v[5] = -(KP1_662939224 * (FNMS(KP668178637, Tb, Tg)));
        auto Tp = Tj + Tk;
        auto Tq = Tm + Tn;
        v[4] = KP1_414213562 * (Tp - Tq);
        v[0] = KP1_414213562 * (Tp + Tq);
        auto Th = FMA(KP707106781, Ta, T3);
        auto Ti = FMA(KP707106781, Tf, Te);
        v[1] = KP1_961570560 * (FNMS(KP198912367, Ti, Th));
        v[7] = KP1_961570560 * (FMA(KP198912367, Th, Ti));
        auto Tl = Tj - Tk;
        auto To = Tm - Tn;
        v[2] = KP1_847759065 * (FNMS(KP414213562, To, Tl));
        v[6] = KP1_847759065 * (FMA(KP414213562, Tl, To));
    } else {
        constexpr float KP1_662939224 =
            +1.662939224605090474157576755235811513477121624f;
        constexpr float KP668178637 =
            +0.668178637919298919997757686523080761552472251f;
        constexpr float KP1_961570560 =
            +1.961570560806460898252364472268478073947867462f;
        constexpr float KP198912367 =
            +0.198912367379658006911597622644676228597850501f;
        constexpr float KP1_847759065 =
            +1.847759065022573512256366378793576573644833252f;
        constexpr float KP707106781 =
            +0.707106781186547524400844362104849039284835938f;
        constexpr float KP414213562 =
            +0.414213562373095048801688724209698078569671875f;
        constexpr float KP1_414213562 =
            +1.414213562373095048801688724209698078569671875f;

        auto T1 = v[0] * KP1_414213562;
        auto T2 = v[4];
        auto T3 = FMA(KP1_414213562, T2, T1);
        auto Tj = FNMS(KP1_414213562, T2, T1);
        auto T4 = v[2];
        auto T5 = v[6];
        auto T6 = FMA(KP414213562, T5, T4);
        auto Tk = FMS(KP414213562, T4, T5);
        auto T8 = v[1];
        auto Td = v[7];
        auto T9 = v[5];
        auto Ta = v[3];
        auto Tb = T9 + Ta;
        auto Te = Ta - T9;
        auto Tc = FMA(KP707106781, Tb, T8);
        auto Tn = FNMS(KP707106781, Te, Td);
        auto Tf = FMA(KP707106781, Te, Td);
        auto Tm = FNMS(KP707106781, Tb, T8);
        auto T7 = FMA(KP1_847759065, T6, T3);
        auto Tg = FMA(KP198912367, Tf, Tc);
        v[7] = FNMS(KP1_961570560, Tg, T7);
        v[0] = FMA(KP1_961570560, Tg, T7);
        auto Tp = FNMS(KP1_847759065, Tk, Tj);
        auto Tq = FMA(KP668178637, Tm, Tn);
        v[5] = FNMS(KP1_662939224, Tq, Tp);
        v[2] = FMA(KP1_662939224, Tq, Tp);
        auto Th = FNMS(KP1_847759065, T6, T3);
        auto Ti = FNMS(KP198912367, Tc, Tf);
        v[3] = FNMS(KP1_961570560, Ti, Th);
        v[4] = FMA(KP1_961570560, Ti, Th);
        auto Tl = FMA(KP1_847759065, Tk, Tj);
        auto To = FNMS(KP668178637, Tn, Tm);
        v[6] = FNMS(KP1_662939224, To, Tl);
        v[1] = FMA(KP1_662939224, To, Tl);
    }
}

template <int stride = 1, int howmany = 8, int howmany_stride = 8>
METAL_FUNC void transpose_pack8_interleave4(thread float *data,
                                            threadgroup float *buffer,
                                            ushort tid_in_simdgroup) {

    for (int iter = 0; iter < howmany; ++iter, data += howmany_stride) {
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 8; ++i) {
            buffer[i * smem_stride + tid_in_simdgroup] = data[i * stride];
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 8; ++i) {
            data[i * stride] = buffer[(tid_in_simdgroup % 8) * smem_stride +
                                      (tid_in_simdgroup & ~7) + i];
        }
    }
}

template <int stride = 1>
METAL_FUNC float hard_thresholding(thread float *data, float sigma,
                                   ushort tid_in_simdgroup) {
    constexpr float inv_norm = 1.0f / 4096.0f;

    float ks[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int i = 0; i < 64; ++i) {
        auto val = data[i * stride];

        float thr;
        if (i == 0) {
            thr =
                (tid_in_simdgroup % 8) ? sigma : 0.0f; // protects DC component
        } else {
            thr = sigma;
        }

        float flag = fabs(val) >= thr;

        ks[i % 4] += flag;
        data[i * stride] = flag ? (val * inv_norm) : 0.0f;
    }

    float k = (ks[0] + ks[1]) + (ks[2] + ks[3]);

    for (int i = 4; i >= 1; i /= 2) {
        k += metal_shfl_xor_sync(k, i, 8, tid_in_simdgroup);
    }

    return 1.0f / k;
}

METAL_FUNC float collaborative_hard(thread float *denoising_patch, float sigma,
                                    threadgroup float *buffer,
                                    ushort tid_in_simdgroup) {
    constexpr int stride1 = 1;
    constexpr int stride2 = 8;

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<true, stride1, 8, stride2>(
            denoising_patch, buffer, tid_in_simdgroup);
        transpose_pack8_interleave4<stride1, 8, stride2>(
            denoising_patch, buffer, tid_in_simdgroup);
    }
    transform_pack8_interleave4<true, stride2, 8, stride1>(
        denoising_patch, buffer, tid_in_simdgroup);

    float adaptive_weight =
        hard_thresholding<stride1>(denoising_patch, sigma, tid_in_simdgroup);

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<false, stride1, 8, stride2>(
            denoising_patch, buffer, tid_in_simdgroup);
        transpose_pack8_interleave4<stride1, 8, stride2>(
            denoising_patch, buffer, tid_in_simdgroup);
    }
    transform_pack8_interleave4<false, stride2, 8, stride1>(
        denoising_patch, buffer, tid_in_simdgroup);

    return adaptive_weight;
}

template <int stride = 1>
METAL_FUNC float wiener_filtering(thread float *data, thread float *ref,
                                  float sigma, ushort tid_in_simdgroup) {
    constexpr float inv_norm = 1.0f / 4096.0f;

    float ks[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    float sigma_sq = sigma * sigma;

    for (int i = 0; i < 64; ++i) {
        auto val = data[i * stride];
        auto ref_val = ref[i * stride];
        float ref_val_sq = ref_val * ref_val;
        float coeff = ref_val_sq / (ref_val_sq + sigma_sq);
        if (i == 0 && (tid_in_simdgroup % 8) == 0) {
            coeff = 1.0f; // protects DC component
        }
        val *= coeff;

        ks[i % 4] += coeff * coeff;

        data[i * stride] = val * inv_norm;
    }

    float k = (ks[0] + ks[1]) + (ks[2] + ks[3]);

    for (int i = 4; i >= 1; i /= 2) {
        k += metal_shfl_xor_sync(k, i, 8, tid_in_simdgroup);
    }

    return 1.0f / k;
}

METAL_FUNC float collaborative_wiener(thread float *denoising_patch,
                                      thread float *ref_patch, float sigma,
                                      threadgroup float *buffer,
                                      ushort tid_in_simdgroup) {
    constexpr int stride1 = 1;
    constexpr int stride2 = 8;

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<true, stride1, 8, stride2>(
            denoising_patch, buffer, tid_in_simdgroup);
        transpose_pack8_interleave4<stride1, 8, stride2>(
            denoising_patch, buffer, tid_in_simdgroup);
    }
    transform_pack8_interleave4<true, stride2, 8, stride1>(
        denoising_patch, buffer, tid_in_simdgroup);

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<true, stride1, 8, stride2>(
            ref_patch, buffer, tid_in_simdgroup);
        transpose_pack8_interleave4<stride1, 8, stride2>(ref_patch, buffer,
                                                         tid_in_simdgroup);
    }
    transform_pack8_interleave4<true, stride2, 8, stride1>(ref_patch, buffer,
                                                           tid_in_simdgroup);

    float adaptive_weight = wiener_filtering<stride1>(
        denoising_patch, ref_patch, sigma, tid_in_simdgroup);

    for (int ndim = 0; ndim < 2; ++ndim) {
        transform_pack8_interleave4<false, stride1, 8, stride2>(
            denoising_patch, buffer, tid_in_simdgroup);
        transpose_pack8_interleave4<stride1, 8, stride2>(
            denoising_patch, buffer, tid_in_simdgroup);
    }
    transform_pack8_interleave4<false, stride2, 8, stride1>(
        denoising_patch, buffer, tid_in_simdgroup);

    return adaptive_weight;
}

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

// Common BM3D kernel logic
template <bool temporal, bool chroma, bool final_>
METAL_FUNC void bm3d_kernel_logic(device float *res, device const float *src,
                                  constant KernelParams &params, uint2 gid,
                                  ushort tid_in_simdgroup,
                                  threadgroup float *buffer) {
    float sigma = params.sigma;
    const int sub_lane_id = tid_in_simdgroup % 8;
    int x = (4 * gid.x + tid_in_simdgroup / 8) * params.block_step;
    int y = params.block_step * gid.y;
    if (x >= params.width - 8 + params.block_step ||
        y >= params.height - 8 + params.block_step) {
        return;
    }

    x = min(x, params.width - 8);
    y = min(y, params.height - 8);

    int radius = 0;
    if constexpr (temporal) {
        radius = params.radius;
    }

    int temporal_stride = params.height * params.stride;
    int temporal_width = 2 * radius + 1;
    int plane_stride = temporal_width * temporal_stride;
    int clip_stride = (chroma ? 3 : 1) * temporal_width * temporal_stride;

    float current_patch[8];
    device const float *const srcpc =
        &src[radius * temporal_stride + sub_lane_id];

    {
        device const float *srcp = &srcpc[y * params.stride + x];

        for (int i = 0; i < 8; ++i) {
            current_patch[i] = srcp[i * params.stride];
        }
    }

    constexpr float initial_error = __FLT_MAX__;
    float errors8 = initial_error;
    int index8_x = 0;
    int index8_y = 0;

    {
        int left = max(x - params.bm_range, 0);
        int right = min(x + params.bm_range, params.width - 8);
        int top = max(y - params.bm_range, 0);
        int bottom = min(y + params.bm_range, params.height - 8);

        device const float *srcp_row = &srcpc[top * params.stride + left];
        for (int row_i = top; row_i <= bottom; ++row_i) {
            device const float *srcp_col = srcp_row;
            for (int col_i = left; col_i <= right; ++col_i) {
                float errors[2] = {0.0f, 0.0f};
                device const float *srcp = srcp_col;

                simdgroup_barrier(mem_flags::mem_none);

                for (int i = 0; i < 8; ++i) {
                    float val = current_patch[i] - srcp[i * params.stride];
                    errors[i % 2] += val * val;
                }

                float error = errors[0] + errors[1];

                for (int i = 4; i >= 1; i /= 2) {
                    error += metal_shfl_xor_sync(error, i, 8, tid_in_simdgroup);
                }

                auto pre_error =
                    metal_shfl_up_sync(errors8, 1, 8, tid_in_simdgroup);
                int pre_index_x =
                    metal_shfl_up_sync(index8_x, 1, 8, tid_in_simdgroup);
                int pre_index_y =
                    metal_shfl_up_sync(index8_y, 1, 8, tid_in_simdgroup);

                bool flag = error < errors8;
                bool pre_flag =
                    metal_shfl_up_sync(flag, 1, 8, tid_in_simdgroup);

                if (flag) {
                    bool first = (sub_lane_id == 0) || (!pre_flag);
                    errors8 = first ? error : pre_error;
                    index8_x = first ? col_i : pre_index_x;
                    index8_y = first ? row_i : pre_index_y;
                }
                ++srcp_col;
            }
            srcp_row += params.stride;
        }
    }
    int index8_z = radius;

    if constexpr (temporal) {
        int center_index8_x = index8_x;
        int center_index8_y = index8_y;

        for (int direction = -1; direction <= 1; direction += 2) {
            int last_index8_x = center_index8_x;
            int last_index8_y = center_index8_y;

            for (int t = 1; t <= radius; ++t) {
                int temporal_index = radius + direction * t;
                float frame_errors8 = initial_error;
                int frame_index8_x = 0;
                int frame_index8_y = 0;

                device const float *temporal_srcpc =
                    &src[temporal_index * temporal_stride + sub_lane_id];

                for (int i = 0; i < params.ps_num; ++i) {
                    int xx =
                        metal_shfl_sync(last_index8_x, i, 8, tid_in_simdgroup);
                    int yy =
                        metal_shfl_sync(last_index8_y, i, 8, tid_in_simdgroup);

                    int left = max(xx - params.ps_range, 0);
                    int right = min(xx + params.ps_range, params.width - 8);
                    int top = max(yy - params.ps_range, 0);
                    int bottom = min(yy + params.ps_range, params.height - 8);

                    device const float *srcp_row =
                        &temporal_srcpc[top * params.stride + left];
                    for (int row_i = top; row_i <= bottom; ++row_i) {
                        device const float *srcp_col = srcp_row;
                        for (int col_i = left; col_i <= right; ++col_i) {
                            float errors[2] = {0.0f, 0.0f};
                            device const float *srcp = srcp_col;

                            simdgroup_barrier(mem_flags::mem_none);

                            for (int j = 0; j < 8; ++j) {
                                float val =
                                    current_patch[j] - srcp[j * params.stride];
                                errors[j % 2] += val * val;
                            }

                            float error = errors[0] + errors[1];

                            for (int i = 4; i >= 1; i /= 2) {
                                error += metal_shfl_xor_sync(error, i, 8,
                                                             tid_in_simdgroup);
                            }

                            float pre_error = metal_shfl_up_sync(
                                frame_errors8, 1, 8, tid_in_simdgroup);
                            int pre_index_x = metal_shfl_up_sync(
                                frame_index8_x, 1, 8, tid_in_simdgroup);
                            int pre_index_y = metal_shfl_up_sync(
                                frame_index8_y, 1, 8, tid_in_simdgroup);

                            bool flag = error < frame_errors8;
                            bool pre_flag = metal_shfl_up_sync(
                                flag, 1, 8, tid_in_simdgroup);

                            if (flag) {
                                bool first = (sub_lane_id == 0) || (!pre_flag);
                                frame_errors8 = first ? error : pre_error;
                                frame_index8_x = first ? col_i : pre_index_x;
                                frame_index8_y = first ? row_i : pre_index_y;
                            }
                            ++srcp_col;
                        }
                        srcp_row += params.stride;
                    }
                }

                for (int i = 0; i < params.ps_num; ++i) {
                    float tmp_error =
                        metal_shfl_sync(frame_errors8, i, 8, tid_in_simdgroup);
                    int tmp_x =
                        metal_shfl_sync(frame_index8_x, i, 8, tid_in_simdgroup);
                    int tmp_y =
                        metal_shfl_sync(frame_index8_y, i, 8, tid_in_simdgroup);

                    bool flag = tmp_error < errors8;
                    bool pre_flag =
                        metal_shfl_up_sync(flag, 1, 8, tid_in_simdgroup);
                    float pre_error =
                        metal_shfl_up_sync(errors8, 1, 8, tid_in_simdgroup);
                    int pre_index_x =
                        metal_shfl_up_sync(index8_x, 1, 8, tid_in_simdgroup);
                    int pre_index_y =
                        metal_shfl_up_sync(index8_y, 1, 8, tid_in_simdgroup);
                    int pre_index_z =
                        metal_shfl_up_sync(index8_z, 1, 8, tid_in_simdgroup);

                    if (flag) {
                        bool first = (sub_lane_id == 0) || (!pre_flag);
                        errors8 = first ? tmp_error : pre_error;
                        index8_x = first ? tmp_x : pre_index_x;
                        index8_y = first ? tmp_y : pre_index_y;
                        index8_z = first ? temporal_index : pre_index_z;
                    }
                }
                last_index8_x = frame_index8_x;
                last_index8_y = frame_index8_y;
            }
        }
    }

    {
        int flag_val;
        if constexpr (temporal) {
            flag_val = (index8_x == x && index8_y == y && index8_z == radius);
        } else {
            flag_val = (index8_x == x && index8_y == y);
        }
        float flag = flag_val;

        for (int i = 4; i >= 1; i /= 2) {
            flag += metal_shfl_xor_sync(flag, i, 8, tid_in_simdgroup);
        }

        float pre_error = metal_shfl_up_sync(errors8, 1, 8, tid_in_simdgroup);
        int pre_index_x = metal_shfl_up_sync(index8_x, 1, 8, tid_in_simdgroup);
        int pre_index_y = metal_shfl_up_sync(index8_y, 1, 8, tid_in_simdgroup);
        if (flag == 0.0f) {
            bool first = (sub_lane_id == 0);
            errors8 = first ? 0.0f : pre_error;
            index8_x = first ? x : pre_index_x;
            index8_y = first ? y : pre_index_y;
            if constexpr (temporal) {
                int pre_index_z =
                    metal_shfl_up_sync(index8_z, 1, 8, tid_in_simdgroup);
                index8_z = first ? radius : pre_index_z;
            }
        }
    }

    thread float denoising_patch[64];
    thread float ref_patch[64];

    int num_planes = chroma ? 3 : 1;
    for (int plane = 0; plane < num_planes; ++plane) {
        if (plane == 1) {
            sigma = params.sigma_u;
        } else if (plane == 2) {
            sigma = params.sigma_v;
        }

        if constexpr (chroma) {
            constexpr float epsilon = 1.19209290e-07F; // FLT_EPSILON
            if (sigma < epsilon) {
                src += plane_stride;
                res += plane_stride * 2;
                continue;
            }
        }

        float adaptive_weight;
        if constexpr (final_) {

            for (int i = 0; i < 8; ++i) {
                int tmp_x = metal_shfl_sync(index8_x, i, 8, tid_in_simdgroup);
                int tmp_y = metal_shfl_sync(index8_y, i, 8, tid_in_simdgroup);
                device const float *refp;
                if constexpr (temporal) {
                    int tmp_z =
                        metal_shfl_sync(index8_z, i, 8, tid_in_simdgroup);
                    refp = &src[tmp_z * temporal_stride +
                                tmp_y * params.stride + tmp_x + sub_lane_id];
                } else {
                    refp = &src[tmp_y * params.stride + tmp_x + sub_lane_id];
                }
                device const float *srcp = &refp[clip_stride];

                for (int j = 0; j < 8; ++j) {
                    ref_patch[i * 8 + j] = refp[j * params.stride];
                    denoising_patch[i * 8 + j] = srcp[j * params.stride];
                }
            }
            adaptive_weight = collaborative_wiener(
                denoising_patch, ref_patch, sigma, buffer, tid_in_simdgroup);
        } else {

            for (int i = 0; i < 8; ++i) {
                int tmp_x = metal_shfl_sync(index8_x, i, 8, tid_in_simdgroup);
                int tmp_y = metal_shfl_sync(index8_y, i, 8, tid_in_simdgroup);
                device const float *srcp;
                if constexpr (temporal) {
                    int tmp_z =
                        metal_shfl_sync(index8_z, i, 8, tid_in_simdgroup);
                    srcp = &src[tmp_z * temporal_stride +
                                tmp_y * params.stride + tmp_x + sub_lane_id];
                } else {
                    srcp = &src[tmp_y * params.stride + tmp_x + sub_lane_id];
                }

                for (int j = 0; j < 8; ++j) {
                    denoising_patch[i * 8 + j] = srcp[j * params.stride];
                }
            }
            adaptive_weight = collaborative_hard(denoising_patch, sigma, buffer,
                                                 tid_in_simdgroup);
        }

        device float *const wdstpc = &res[sub_lane_id];
        device float *const weightpc = &res[temporal_stride + sub_lane_id];

        for (int i = 0; i < 8; ++i) {
            int tmp_x = metal_shfl_sync(index8_x, i, 8, tid_in_simdgroup);
            int tmp_y = metal_shfl_sync(index8_y, i, 8, tid_in_simdgroup);
            int offset;
            if constexpr (temporal) {
                int tmp_z = metal_shfl_sync(index8_z, i, 8, tid_in_simdgroup);
                offset =
                    tmp_z * 2 * temporal_stride + tmp_y * params.stride + tmp_x;
            } else {
                offset = tmp_y * params.stride + tmp_x;
            }

            device float *wdstp = &wdstpc[offset];
            device float *weightp = &weightpc[offset];

            for (int j = 0; j < 8; ++j) {
                float wdst_val = adaptive_weight * denoising_patch[i * 8 + j];
                float weight_val = adaptive_weight;

                wdst_val = (wdst_val + params.extractor) - params.extractor;
                weight_val = (weight_val + params.extractor) - params.extractor;

                atomic_add_float(&wdstp[j * params.stride], wdst_val);
                atomic_add_float(&weightp[j * params.stride], weight_val);
            }
        }

        src += plane_stride;
        res += plane_stride * 2;
    }
}

#define DECLARE_BM3D_KERNEL(name, temporal_val, chroma_val, final_val)         \
    kernel void name(device float *res [[buffer(0)]],                          \
                     device const float *src [[buffer(1)]],                    \
                     constant KernelParams &params [[buffer(2)]],              \
                     uint2 gid [[threadgroup_position_in_grid]],               \
                     ushort tid_in_simdgroup [[thread_index_in_simdgroup]],    \
                     threadgroup float *buffer [[threadgroup(0)]]) {           \
        bm3d_kernel_logic<temporal_val, chroma_val, final_val>(                \
            res, src, params, gid, tid_in_simdgroup, buffer);                  \
    }

// Spatial kernels
DECLARE_BM3D_KERNEL(bm3d_false_false_false, false, false, false)
DECLARE_BM3D_KERNEL(bm3d_false_false_true, false, false, true)
DECLARE_BM3D_KERNEL(bm3d_false_true_false, false, true, false)
DECLARE_BM3D_KERNEL(bm3d_false_true_true, false, true, true)

// Temporal kernels
DECLARE_BM3D_KERNEL(bm3d_true_false_false, true, false, false)
DECLARE_BM3D_KERNEL(bm3d_true_false_true, true, false, true)
DECLARE_BM3D_KERNEL(bm3d_true_true_false, true, true, false)
DECLARE_BM3D_KERNEL(bm3d_true_true_true, true, true, true)

kernel void copy_kernel(device const float *src [[buffer(0)]],
                        device float *dst [[buffer(1)]],
                        constant uint &src_stride [[buffer(2)]],
                        constant uint &dst_stride [[buffer(3)]],
                        constant uint &width [[buffer(4)]],
                        uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= width) {
        return;
    }

    ulong src_offset = (ulong)tid.y * src_stride + tid.x;
    ulong dst_offset = (ulong)tid.y * dst_stride + tid.x;

    dst[dst_offset] = src[src_offset];
}
