// Copyright 2026 The Android Open Source Project
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "host/vulkan/testing/VulkanTestHelper.h"
#include "host/vulkan/vk_format_utils.h"

namespace gfxstream::host::vk::testing {
namespace {
constexpr VkFormat kFormats[] = {
    VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
    VK_FORMAT_ASTC_5x5_UNORM_BLOCK, VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
    VK_FORMAT_ASTC_6x6_UNORM_BLOCK, VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
    VK_FORMAT_ASTC_8x6_UNORM_BLOCK, VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
    VK_FORMAT_ASTC_10x5_UNORM_BLOCK, VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
    VK_FORMAT_ASTC_10x8_UNORM_BLOCK, VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
    VK_FORMAT_ASTC_12x10_UNORM_BLOCK, VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
};

class MesaAstcTest : public ::testing::Test {
protected:
    void SetUp() override {
        host = vkDispatch(false);
        helper = std::make_unique<VulkanTestHelper>();
        helper->failOnValidationErrors(false);
        helper->initialize({.astcLdrEmulationMode = AstcEmulationMode::Gpu,
                            .appInfo = VkApplicationInfo{
                                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                .pApplicationName = "mesa_astc_test",
                                .apiVersion = VK_API_VERSION_1_3},
                            .deviceFeatures = {.textureCompressionASTC_LDR = VK_TRUE},
                            .astcHdr = true});
        device = helper->device();
    }
    void TearDown() override {
        helper->destroy();
        EXPECT_FALSE(helper->hasValidationErrors());
    }

    // Void-extent blocks carry exact constants, so readback needs no CPU decoder.
    static std::array<uint16_t, 4> color(bool hdr, uint32_t layer, uint32_t mip) {
        if (hdr) return {uint16_t(0x4000 + ((layer + mip) % 5) * 0x400), 0x3c00, 0x3800, 0x3c00};
        return {uint16_t((23 + layer * 17 + mip * 7) * 257),
                uint16_t(97 * 257), uint16_t(173 * 257), 65535};
    }

    void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                 VkImageLayout newLayout, uint32_t mip, uint32_t firstLayer, uint32_t layers) {
        const VkImageMemoryBarrier transition = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ? 0u : VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = unbox_VkImage(image),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, firstLayer, layers},
        };
        helper->vk().vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                                          0, nullptr, 1, &transition);
    }

    void check(VkFormat format, VkImageType type, uint32_t layers, bool cube,
               int endpointMode = -1, bool gradient = false) {
        SCOPED_TRACE(::testing::Message() << "format=" << format << " type=" << type
                                         << " layers=" << layers << " cube=" << cube);
        auto& vk = helper->vk();
        const bool hdr = isAstcHdr(format);
        const auto block = astcBlockSize(format);
        const uint32_t width = 37, height = cube ? 37 : 29;
        const uint32_t depth = type == VK_IMAGE_TYPE_3D ? 5 : 1;
        constexpr uint32_t levels = 3;
        const VkImageFormatListCreateInfo formatList = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
            .viewFormatCount = 1,
            .pViewFormats = &format,
        };
        const VkImageCreateInfo imageInfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &formatList,
            .flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u,
            .imageType = type, .format = format, .extent = {width, height, depth},
            .mipLevels = levels, .arrayLayers = layers, .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VkImage image = VK_NULL_HANDLE;
        ASSERT_EQ(vk.vkCreateImage(device, &imageInfo, nullptr, &image), VK_SUCCESS);
        VkMemoryRequirements requirements;
        vk.vkGetImageMemoryRequirements(device, image, &requirements);
        const VkMemoryAllocateInfo allocation = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = helper->findMemoryType(requirements.memoryTypeBits,
                                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        };
        VkDeviceMemory imageMemory;
        ASSERT_EQ(vk.vkAllocateMemory(device, &allocation, nullptr, &imageMemory), VK_SUCCESS);
        ASSERT_EQ(vk.vkBindImageMemory(device, image, imageMemory, 0), VK_SUCCESS);

        std::vector<uint8_t> compressed;
        std::vector<VkBufferImageCopy> uploads;
        std::vector<VkBufferImageCopy> readbacks;
        size_t readbackSize = 0;
        for (uint32_t mip = 0; mip < levels; ++mip) {
            const uint32_t w = std::max(width >> mip, 1u);
            const uint32_t h = std::max(height >> mip, 1u);
            const uint32_t d = std::max(depth >> mip, 1u);
            const uint32_t blocks = ((w + block.width - 1) / block.width) *
                                    ((h + block.height - 1) / block.height);
            uploads.push_back({compressed.size(), 0, 0,
                               {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, layers}, {0, 0, 0}, {w, h, d}});
            for (uint32_t slice = 0; slice < layers * d; ++slice) {
                const auto values = color(hdr, slice, mip);
                std::array<uint8_t, 16> data;
                data.fill(0xff);
                data[0] = 0xfc;
                data[1] = hdr ? 0xff : 0xfd;
                std::memcpy(data.data() + 8, values.data(), 8);
                if (endpointMode >= 0) {
                    // One partition, a 4x4 two-bit weight grid, and zero endpoints/weights.
                    // All endpoint values fit the eight-bit quantizer in this encoding.
                    data.fill(0);
                    const uint32_t header = 0x42u | (uint32_t(endpointMode) << 13);
                    std::memcpy(data.data(), &header, sizeof(header));
                    if (gradient) {
                        auto putBits = [&data](uint32_t offset, uint32_t value, uint32_t count) {
                            for (uint32_t bit = 0; bit < count; ++bit) {
                                data[(offset + bit) / 8] |=
                                    ((value >> bit) & 1u) << ((offset + bit) % 8);
                            }
                        };
                        // Direct luminance endpoints: HDR 1 and 16, LDR 0 and 255.
                        putBits(17, hdr ? 120 : 0, 8);
                        putBits(25, hdr ? 152 : 255, 8);
                        for (uint32_t weight = 0; weight < 16; ++weight) {
                            for (uint32_t bit = 0; bit < 2; ++bit) {
                                putBits(127 - weight * 2 - bit, (weight % 4) >> bit, 1);
                            }
                        }
                    }
                }
                for (uint32_t b = 0; b < blocks; ++b) {
                    compressed.insert(compressed.end(), data.begin(), data.end());
                }
            }
            readbacks.push_back({readbackSize, 0, 0,
                                 {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, layers}, {0, 0, 0}, {w, h, d}});
            readbackSize += size_t(w) * h * d * layers * (hdr ? 8 : 4);
        }
        VkBuffer upload, readback;
        VkDeviceMemory uploadMemory, readbackMemory;
        helper->createBuffer(compressed.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              upload, uploadMemory);
        helper->createBuffer(readbackSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              readback, readbackMemory);
        void* mapped;
        ASSERT_EQ(vk.vkMapMemory(device, uploadMemory, 0, VK_WHOLE_SIZE, 0, &mapped), VK_SUCCESS);
        std::memcpy(mapped, compressed.data(), compressed.size());
        vk.vkUnmapMemory(device, uploadMemory);

        const auto cmd = helper->beginCommandBuffer();
        for (uint32_t mip = 0; mip < levels; ++mip) {
            barrier(cmd, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    mip, 0, layers);
        }
        vk.vkCmdCopyBufferToImage(cmd, upload, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  static_cast<uint32_t>(uploads.size()), uploads.data());
        for (uint32_t mip = 0; mip < levels; ++mip) {
            // Separate layer transitions exercise nonzero baseArrayLayer dispatches.
            for (uint32_t layer = 0; layer < layers; ++layer) {
                barrier(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, mip, layer, 1);
            }
        }
        // Read the host backing image to inspect decoded pixels rather than compressed blocks.
        host->vkCmdCopyImageToBuffer(unbox_VkCommandBuffer(cmd), unbox_VkImage(image),
                                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, unbox_VkBuffer(readback),
                                     static_cast<uint32_t>(readbacks.size()), readbacks.data());
        helper->submitCommandBuffer(cmd);
        ASSERT_EQ(vk.vkMapMemory(device, readbackMemory, 0, VK_WHOLE_SIZE, 0, &mapped), VK_SUCCESS);
        const auto* bytes = static_cast<const uint8_t*>(mapped);
        size_t mismatches = 0;
        for (uint32_t mip = 0; mip < levels; ++mip) {
            const auto& region = readbacks[mip];
            const size_t pixels = size_t(region.imageExtent.width) * region.imageExtent.height;
            for (uint32_t slice = 0; slice < layers * region.imageExtent.depth; ++slice) {
                auto values = color(hdr, slice, mip);
                if (endpointMode >= 0) {
                    const bool opaque = hdr ? endpointMode < 14 :
                        (endpointMode == 0 || endpointMode == 1 || endpointMode == 6 ||
                         endpointMode == 8 || endpointMode == 9);
                    values = {0, 0, 0, uint16_t(opaque ? (hdr ? 0x3c00 : 0xffff) : 0)};
                }
                for (size_t p = 0; p < pixels; ++p) {
                    if (gradient) {
                        // The 4x4 weight grid maps directly to the four pixel columns.
                        constexpr uint16_t hdrRamp[] = {0x3c00, 0x4100, 0x4680, 0x4c00};
                        constexpr uint16_t ldrRamp[] = {0, 84 * 257, 171 * 257, 65535};
                        const uint32_t column = (p % region.imageExtent.width) % 4;
                        const auto value = hdr ? hdrRamp[column] : ldrRamp[column];
                        values = {value, value, value, uint16_t(hdr ? 0x3c00 : 0xffff)};
                    }
                    const auto* pixel = bytes + region.bufferOffset +
                                        (slice * pixels + p) * (hdr ? 8 : 4);
                    for (uint32_t channel = 0; channel < 4; ++channel) {
                        uint16_t actual = pixel[channel];
                        if (hdr) std::memcpy(&actual, pixel + channel * 2, 2);
                        const uint16_t expected = hdr ? values[channel] : values[channel] >> 8;
                        if (actual != expected) ++mismatches;
                    }
                }
            }
        }
        EXPECT_EQ(mismatches, 0u);
        vk.vkUnmapMemory(device, readbackMemory);
        vk.vkDestroyBuffer(device, upload, nullptr);
        vk.vkDestroyBuffer(device, readback, nullptr);
        vk.vkFreeMemory(device, uploadMemory, nullptr);
        vk.vkFreeMemory(device, readbackMemory, nullptr);
        vk.vkDestroyImage(device, image, nullptr);
        vk.vkFreeMemory(device, imageMemory, nullptr);
    }

    std::unique_ptr<VulkanTestHelper> helper;
    VulkanDispatch* host = nullptr;
    VkDevice device = VK_NULL_HANDLE;
};

TEST_F(MesaAstcTest, AllFormatsArrayMipmaps) {
    for (uint32_t i = 0; i < std::size(kFormats); ++i) {
        check(kFormats[i], VK_IMAGE_TYPE_2D, 3, false);
        check(static_cast<VkFormat>(kFormats[i] + 1), VK_IMAGE_TYPE_2D, 3, false);
        check(static_cast<VkFormat>(VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK + i), VK_IMAGE_TYPE_2D, 3, false);
    }
}

TEST_F(MesaAstcTest, AdvertisesHdrExtensionAndFeatures) {
    gfxstream::base::BumpPool pool;
    auto* state = VkDecoderGlobalState::get();
    VkPhysicalDeviceTextureCompressionASTCHDRFeatures hdr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TEXTURE_COMPRESSION_ASTC_HDR_FEATURES,
    };
    VkPhysicalDeviceFeatures2 features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &hdr,
    };
    state->on_vkGetPhysicalDeviceFeatures2(&pool, kInvalidSnapshotApiCallHandle,
                                          helper->physDev(), &features);
    EXPECT_EQ(features.features.textureCompressionASTC_LDR, VK_TRUE);
    EXPECT_EQ(hdr.textureCompressionASTC_HDR, VK_TRUE);
    VkPhysicalDeviceVulkan13Features features13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    };
    features.pNext = &features13;
    state->on_vkGetPhysicalDeviceFeatures2(&pool, kInvalidSnapshotApiCallHandle,
                                          helper->physDev(), &features);
    EXPECT_EQ(features13.textureCompressionASTC_HDR, VK_TRUE);
    uint32_t count = 0;
    ASSERT_EQ(state->on_vkEnumerateDeviceExtensionProperties(
                  &pool, kInvalidSnapshotApiCallHandle, helper->physDev(), nullptr, &count, nullptr),
              VK_SUCCESS);
    std::vector<VkExtensionProperties> extensions(count);
    ASSERT_EQ(state->on_vkEnumerateDeviceExtensionProperties(
                  &pool, kInvalidSnapshotApiCallHandle, helper->physDev(), nullptr,
                  &count, extensions.data()), VK_SUCCESS);
    EXPECT_TRUE(std::any_of(extensions.begin(), extensions.end(), [](const auto& extension) {
        return std::strcmp(extension.extensionName,
                           VK_EXT_TEXTURE_COMPRESSION_ASTC_HDR_EXTENSION_NAME) == 0;
    }));
}

TEST_F(MesaAstcTest, CubeArrayAndVolumeMipmaps) {
    check(VK_FORMAT_ASTC_5x4_UNORM_BLOCK, VK_IMAGE_TYPE_2D, 12, true);
    check(VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK, VK_IMAGE_TYPE_2D, 12, true);
    check(VK_FORMAT_ASTC_5x4_UNORM_BLOCK, VK_IMAGE_TYPE_3D, 1, false);
    check(VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK, VK_IMAGE_TYPE_3D, 1, false);
}

TEST_F(MesaAstcTest, AllEndpointModes) {
    for (int mode : {2, 3, 7, 11, 14, 15}) {
        SCOPED_TRACE(mode);
        check(VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK, VK_IMAGE_TYPE_2D, 2, false, mode);
    }
    for (int mode : {0, 1, 4, 5, 6, 8, 9, 10, 12, 13}) {
        SCOPED_TRACE(mode);
        check(VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_IMAGE_TYPE_2D, 2, false, mode);
    }
}

TEST_F(MesaAstcTest, WeightedLuminanceAboveOne) {
    check(VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_IMAGE_TYPE_2D, 3, false, 0, true);
    check(VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK, VK_IMAGE_TYPE_2D, 3, false, 2, true);
}
}  // namespace
}  // namespace gfxstream::host::vk::testing
