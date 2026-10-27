// Copyright 2022 The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "compressed_image_info.h"

#include "emulated_textures/shaders/decompression_shaders.h"
#include "gfxstream/ArraySize.h"
#include "gfxstream/common/logging.h"
#include "goldfish_vk_deepcopy.h"
#include "vk_format_utils.h"
#include "vk_utils.h"

namespace gfxstream {
namespace host {
namespace vk {
namespace {

// Returns x / y, rounded up. E.g. ceil_div(7, 2) == 4
// Note the potential integer overflow for large numbers.
inline constexpr uint32_t ceil_div(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

VkImageView createDefaultImageView(VulkanDispatch* vk, VkDevice device, VkImage image,
                                   VkFormat format, VkImageType imageType, uint32_t mipLevel,
                                   uint32_t layerCount, uint32_t baseLayer = 0) {
    VkImageViewCreateInfo imageViewInfo = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .format = format,
        .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY},
        .subresourceRange =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = mipLevel,
                .levelCount = 1,
                .baseArrayLayer = baseLayer,
                .layerCount = layerCount,
            },
    };

    switch (imageType) {
        case VK_IMAGE_TYPE_1D:
            imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
            break;
        case VK_IMAGE_TYPE_2D:
            imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            break;
        case VK_IMAGE_TYPE_3D:
            imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            break;
        default:
            imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            break;
    }
    VkImageView imageView;
    VkResult result = vk->vkCreateImageView(device, &imageViewInfo, nullptr, &imageView);
    if (result != VK_SUCCESS) {
        GFXSTREAM_WARNING("GPU decompression: createDefaultImageView failed: %d", result);
        return VK_NULL_HANDLE;
    }
    return imageView;
}

VkExtent2D getBlockSize(VkFormat format) {
    if (isAstc(format)) return astcBlockSize(format);
    switch (format) {
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
        case VK_FORMAT_EAC_R11_UNORM_BLOCK:
        case VK_FORMAT_EAC_R11_SNORM_BLOCK:
        case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
        case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
            return {4, 4};
        default:
            return {1, 1};
    }
}

bool isReadableImageLayout(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_GENERAL:
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR:
            return true;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            // Transfer reads use the original compressed blocks. A transfer-only queue
            // cannot execute the compute shader used to produce decoded texels.
            return false;
        default:
            return false;
    }
}

bool imageWillBecomeReadable(const VkImageMemoryBarrier& barrier,
                             VkPipelineStageFlags dstStageMask) {
    // If we're transitioning from UNDEFINED, the image content is undefined, so don't try to
    // decompress it.
    if (barrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) return false;

    if (barrier.newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        // GENERAL also permits transfer reads and writes, which use compressed blocks.
        constexpr VkAccessFlags shaderReads = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
        constexpr VkPipelineStageFlags shaderStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
            VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
            VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT |
            VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT |
            VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR |
            VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        return (barrier.dstAccessMask & shaderReads) && (dstStageMask & shaderStages);
    }
    return isReadableImageLayout(barrier.newLayout);
}

bool isCompressedFormat(VkFormat format) {
    return isAstc(format) || isEtc2(format) || isBc(format);
}

// Returns the format that the shader uses to write the output image
VkFormat getShaderFormat(VkFormat outputFormat) {
    if (outputFormat == VK_FORMAT_R16G16B16A16_SFLOAT) return VK_FORMAT_R16G16B16A16_UINT;
    switch (outputFormat) {
        case VK_FORMAT_R16_UNORM:
        case VK_FORMAT_R16_SNORM:
        case VK_FORMAT_R16G16_UNORM:
        case VK_FORMAT_R16G16_SNORM:
            return outputFormat;
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
            return VK_FORMAT_R32G32B32A32_UINT;
        default:
            return VK_FORMAT_R8G8B8A8_UINT;
    }
}

// Returns the next memory offset on a given alignment.
// Will divide by zero if alignment is zero.
VkDeviceSize nextAlignedOffset(VkDeviceSize offset, VkDeviceSize alignment) {
    return ceil_div(offset, alignment) * alignment;
}

// Check that the alignment is valid:
// - sets the alignment to 1 if it's 0
// - aborts if it's not a power of 2
void checkValidAlignment(VkDeviceSize& n) {
    if (n == 0) {
        n = 1;
        return;
    }

    // Check that the alignment is a power of 2
    // http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    if ((n & (n - 1))) {
        GFXSTREAM_FATAL("vkGetImageMemoryRequirements returned non-power-of-two alignment: %d", n);
    }
}

}  // namespace

CompressedImageInfo::CompressedImageInfo(VkDevice device) : mDevice(device) {}

CompressedImageInfo::CompressedImageInfo(VkDevice device, const VkImageCreateInfo& createInfo,
                                         GpuDecompressionPipelineManager* pipelineManager)
    : mCompressedFormat(createInfo.format),
      mOutputFormat(getOutputFormat(mCompressedFormat)),
      mCompressedMipmapsFormat(getCompressedMipmapsFormat(mCompressedFormat)),
      mImageType(createInfo.imageType),
      mMipLevels(createInfo.mipLevels),
      mExtent(createInfo.extent),
      mBlock(getBlockSize(mCompressedFormat)),
      mLayerCount(createInfo.arrayLayers),
      mDevice(device),
      mPipelineManager(pipelineManager) {}

// static
VkFormat CompressedImageInfo::getOutputFormat(VkFormat compFmt) {
    if (isAstcHdr(compFmt)) return VK_FORMAT_R16G16B16A16_SFLOAT;
    switch (compFmt) {
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case VK_FORMAT_EAC_R11_UNORM_BLOCK:
            return VK_FORMAT_R16_UNORM;
        case VK_FORMAT_EAC_R11_SNORM_BLOCK:
            return VK_FORMAT_R16_SNORM;
        case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
            return VK_FORMAT_R16G16_UNORM;
        case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
            return VK_FORMAT_R16G16_SNORM;
        case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
        case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
        case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
        case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
        case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
            return GpuDecompressionPipelineManager::astcDecoder() == AstcDecoder::NewBc3
                       ? VK_FORMAT_BC3_UNORM_BLOCK
                       : VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
        case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
        case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:
        case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:
        case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
        case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:
        case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
            return GpuDecompressionPipelineManager::astcDecoder() == AstcDecoder::NewBc3
                       ? VK_FORMAT_BC3_SRGB_BLOCK
                       : VK_FORMAT_R8G8B8A8_SRGB;
        default:
            return compFmt;
    }
}

// static
VkFormat CompressedImageInfo::getCompressedMipmapsFormat(VkFormat compFmt) {
    if (isAstcHdr(compFmt)) return VK_FORMAT_R32G32B32A32_UINT;
    switch (compFmt) {
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK:
            return VK_FORMAT_R16G16B16A16_UINT;
        case VK_FORMAT_EAC_R11_UNORM_BLOCK:
        case VK_FORMAT_EAC_R11_SNORM_BLOCK:
            return VK_FORMAT_R32G32_UINT;
        case VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK:
        case VK_FORMAT_EAC_R11G11_UNORM_BLOCK:
        case VK_FORMAT_EAC_R11G11_SNORM_BLOCK:
        case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
        case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
        case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_6x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_8x8_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x5_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x6_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x8_UNORM_BLOCK:
        case VK_FORMAT_ASTC_10x10_UNORM_BLOCK:
        case VK_FORMAT_ASTC_12x10_UNORM_BLOCK:
        case VK_FORMAT_ASTC_12x12_UNORM_BLOCK:
        case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
        case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
        case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_6x6_SRGB_BLOCK:
        case VK_FORMAT_ASTC_8x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_8x6_SRGB_BLOCK:
        case VK_FORMAT_ASTC_8x8_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x5_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x6_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x8_SRGB_BLOCK:
        case VK_FORMAT_ASTC_10x10_SRGB_BLOCK:
        case VK_FORMAT_ASTC_12x10_SRGB_BLOCK:
        case VK_FORMAT_ASTC_12x12_SRGB_BLOCK:
            return VK_FORMAT_R32G32B32A32_UINT;
        default:
            return compFmt;
    }
}

// static
bool CompressedImageInfo::needEmulatedAlpha(VkFormat format) {
    switch (format) {
        case VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK:
        case VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK:
            return true;
        default:
            return false;
    }
}

bool CompressedImageInfo::isEtc2() const {
    return gfxstream::host::vk::isEtc2(mCompressedFormat);
}

bool CompressedImageInfo::isAstc() const {
    return gfxstream::host::vk::isAstc(mCompressedFormat);
}

VkImageCreateInfo CompressedImageInfo::getOutputCreateInfo(
    const VkImageCreateInfo& createInfo, gfxstream::base::BumpPool* pool) const {
    VkImageCreateInfo result;
    deepcopy_VkImageCreateInfo(pool, VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &createInfo, &result);
    // Guest ASTC view formats do not describe the decoded or payload images. Allow
    // compatible host views, including the integer view used by the compute shader.
    if (auto* formats = vk_find_struct<VkImageFormatListCreateInfo>(&result)) {
        formats->viewFormatCount = 0;
        formats->pViewFormats = nullptr;
    }
    result.format = mOutputFormat;

    result.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                    // Needed for ASTC->BC3 transcoding so that we can create a BC3 image with
                    // VK_IMAGE_USAGE_STORAGE_BIT
                    VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;

    if (!isCompressedFormat(mOutputFormat)) {
        // Need to clear this flag since the application might have specified it, but it's invalid
        // on non-compressed formats
        result.flags &= ~VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
    } else {
        // Need to set this flag so that we can cast the output image into a non-compressed format
        // so that the decompression shader can write to it.
        result.flags |= VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT;
    }

    result.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    return result;
}

void CompressedImageInfo::createCompressedMipmapImages(VulkanDispatch* vk,
                                                       const VkImageCreateInfo& createInfo) {
    if (!mCompressedMipmaps.empty()) {
        return;
    }

    VkImageCreateInfo createInfoCopy = createInfo;
    createInfoCopy.format = mCompressedMipmapsFormat;
    // Note: if you change the flags here, you must also change both versions of
    // on_vkGetPhysicalDeviceImageFormatProperties in VkDecoderGlobalState
    // TODO(gregschlom): Remove duplicated logic.
    createInfoCopy.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    createInfoCopy.flags &= ~(VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT |
                              VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);
    createInfoCopy.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    createInfoCopy.mipLevels = 1;

    mCompressedMipmaps.resize(mMipLevels);
    for (uint32_t i = 0; i < mMipLevels; ++i) {
        createInfoCopy.extent = compressedMipmapExtent(i);
        vk->vkCreateImage(mDevice, &createInfoCopy, nullptr, &mCompressedMipmaps[i]);
    }

    // Compute the memory requirements for all the images (output image + compressed mipmaps)

    vk->vkGetImageMemoryRequirements(mDevice, mOutputImage, &mMemoryRequirements);
    checkValidAlignment(mMemoryRequirements.alignment);
    std::vector<VkMemoryRequirements> mipmapsMemReqs(mMipLevels);
    for (size_t i = 0; i < mMipLevels; ++i) {
        vk->vkGetImageMemoryRequirements(mDevice, mCompressedMipmaps[i], &mipmapsMemReqs[i]);
        checkValidAlignment(mipmapsMemReqs[i].alignment);
    }

    for (const auto& r : mipmapsMemReqs) {
        // What we want here is the least common multiple of all the alignments. However, since
        // alignments are always powers of 2, the lcm is simply the largest value.
        if (r.alignment > mMemoryRequirements.alignment) {
            mMemoryRequirements.alignment = r.alignment;
        }
        mMemoryRequirements.memoryTypeBits &= r.memoryTypeBits;
    }

    // At this point, we have the following:
    //   - mMemoryRequirements.size is the size of the output image
    //   - mMemoryRequirements.alignment is the least common multiple of all alignments
    //   - mMemoryRequirements.memoryTypeBits is the intersection of all the memoryTypeBits
    // Now, compute the offsets of each mipmap image as well as the total memory size we need.
    mMipmapOffsets.resize(mMipLevels);
    for (size_t i = 0; i < mMipLevels; ++i) {
        // This works because the alignment we request is the lcm of all alignments
        mMipmapOffsets[i] =
            nextAlignedOffset(mMemoryRequirements.size, mipmapsMemReqs[i].alignment);
        mMemoryRequirements.size = mMipmapOffsets[i] + mipmapsMemReqs[i].size;
    }
}

void CompressedImageInfo::initAstcCpuDecompression(VulkanDispatch* vk,
                                                   VkPhysicalDevice physicalDevice) {
    mAstcTexture = std::make_unique<AstcTexture>(vk, mDevice, physicalDevice, mExtent, mBlock.width,
                                                 mBlock.height, &AstcCpuDecompressor::get());
}

bool CompressedImageInfo::decompressIfNeeded(VulkanDispatch* vk, VkCommandBuffer commandBuffer,
                                             VkQueueFlags queueFlags,
                                             VkPipelineStageFlags srcStageMask,
                                             VkPipelineStageFlags dstStageMask,
                                             const VkImageMemoryBarrier& targetBarrier,
                                             std::vector<VkImageMemoryBarrier>& outputBarriers) {
    std::vector<VkImageMemoryBarrier> imageBarriers = getImageBarriers(targetBarrier);

    if (!(queueFlags & VK_QUEUE_COMPUTE_BIT) ||
        !imageWillBecomeReadable(targetBarrier, dstStageMask)) {
        // Decoding is needed only for shader reads on a compute-capable queue.
        // Apply the target barrier to the compressed mipmaps and the decompressed image.
        outputBarriers.insert(outputBarriers.end(), imageBarriers.begin(), imageBarriers.end());
        return false;
    }

    VkResult result = initializeDecompressionPipeline(vk, mDevice);
    if (result != VK_SUCCESS) {
        GFXSTREAM_WARNING("Failed to initialize pipeline for texture decompression");
        return false;
    }

    // Transition the layout of all the compressed mipmaps so that the shader can read from them.
    for (auto& barrier : imageBarriers) {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    // Transition the layout of the output image so that we can write to it.
    imageBarriers.back().srcAccessMask = 0;
    imageBarriers.back().oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarriers.back().dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imageBarriers.back().newLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Do the layout transitions
    vk->vkCmdPipelineBarrier(commandBuffer, srcStageMask, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 0, nullptr, imageBarriers.size(), imageBarriers.data());

    // Run the decompression shader
    decompress(vk, commandBuffer, getImageSubresourceRange(targetBarrier.subresourceRange));

    // Finally, transition the layout of all images to match the target barrier.
    for (auto& barrier : imageBarriers) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.dstAccessMask = targetBarrier.dstAccessMask;
        barrier.newLayout = targetBarrier.newLayout;
    }
    // (adjust the last barrier since it's for the output image)
    imageBarriers.back().srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    // Do the layout transitions
    vk->vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, dstStageMask, 0,
                             0, nullptr, 0, nullptr, imageBarriers.size(), imageBarriers.data());

    return true;
}

void CompressedImageInfo::decompressOnCpu(VkCommandBuffer commandBuffer, uint8_t* srcAstcData,
                                          size_t astcDataSize, VkImage dstImage,
                                          VkImageLayout dstImageLayout, uint32_t regionCount,
                                          const VkBufferImageCopy* pRegions,
                                          const VkDecoderContext& context) {
    mAstcTexture->on_vkCmdCopyBufferToImage(commandBuffer, srcAstcData, astcDataSize, dstImage,
                                            dstImageLayout, regionCount, pRegions, context);
}

void CompressedImageInfo::decompressOnCpu(VkCommandBuffer commandBuffer, uint8_t* srcAstcData, size_t astcDataSize,
                         const VkCopyBufferToImageInfo2* pCopyBufferToImageInfo, const VkDecoderContext& context) {
    mAstcTexture->on_vkCmdCopyBufferToImage2(commandBuffer, srcAstcData, astcDataSize, pCopyBufferToImageInfo, context);
}

VkMemoryRequirements CompressedImageInfo::getMemoryRequirements() const {
    return mMemoryRequirements;
}

VkResult CompressedImageInfo::bindCompressedMipmapsMemory(VulkanDispatch* vk, VkDeviceMemory memory,
                                                          VkDeviceSize memoryOffset) {
    VkResult result = VK_SUCCESS;
    for (size_t i = 0; i < mCompressedMipmaps.size(); i++) {
        VkResult res = vk->vkBindImageMemory(mDevice, mCompressedMipmaps[i], memory,
                                             memoryOffset + mMipmapOffsets[i]);
        if (res != VK_SUCCESS) result = res;
    }
    return result;
}

VkBufferImageCopy CompressedImageInfo::getBufferImageCopy(
    const VkBufferImageCopy& origRegion) const {
    VkBufferImageCopy region = origRegion;
    uint32_t mipLevel = region.imageSubresource.mipLevel;
    region.imageSubresource.mipLevel = 0;
    region.bufferRowLength /= mBlock.width;
    region.bufferImageHeight /= mBlock.height;
    region.imageOffset.x /= mBlock.width;
    region.imageOffset.y /= mBlock.height;
    region.imageExtent = compressedMipmapPortion(region.imageExtent, mipLevel);
    return region;
}

VkBufferImageCopy2 CompressedImageInfo::getBufferImageCopy(
    const VkBufferImageCopy2& origRegion) const {
    VkBufferImageCopy2 region = origRegion;
    uint32_t mipLevel = region.imageSubresource.mipLevel;
    region.imageSubresource.mipLevel = 0;
    region.bufferRowLength /= mBlock.width;
    region.bufferImageHeight /= mBlock.height;
    region.imageOffset.x /= mBlock.width;
    region.imageOffset.y /= mBlock.height;
    region.imageExtent = compressedMipmapPortion(region.imageExtent, mipLevel);
    return region;
}

// static
VkImageCopy CompressedImageInfo::getCompressedMipmapsImageCopy(const VkImageCopy& origRegion,
                                                               const CompressedImageInfo* srcImg,
                                                               const CompressedImageInfo* dstImg) {
    const bool needEmulatedSrc = (srcImg != nullptr);
    const bool needEmulatedDst = (dstImg != nullptr);
    VkImageCopy region = origRegion;
    if (needEmulatedSrc) {
        uint32_t mipLevel = region.srcSubresource.mipLevel;
        region.srcSubresource.mipLevel = 0;
        region.srcOffset.x /= srcImg->mBlock.width;
        region.srcOffset.y /= srcImg->mBlock.height;
        region.extent = srcImg->compressedMipmapPortion(region.extent, mipLevel);
    }
    if (needEmulatedDst) {
        region.dstSubresource.mipLevel = 0;
        region.dstOffset.x /= dstImg->mBlock.width;
        region.dstOffset.y /= dstImg->mBlock.height;
    }
    return region;
}

VkImageCopy2 CompressedImageInfo::getCompressedMipmapsImageCopy(const VkImageCopy2& origRegion,
                                                                const CompressedImageInfo* srcImg,
                                                                const CompressedImageInfo* dstImg) {
    const bool needEmulatedSrc = (srcImg != nullptr);
    const bool needEmulatedDst = (dstImg != nullptr);
    VkImageCopy2 region = origRegion;
    if (needEmulatedSrc) {
        uint32_t mipLevel = region.srcSubresource.mipLevel;
        region.srcSubresource.mipLevel = 0;
        region.srcOffset.x /= srcImg->mBlock.width;
        region.srcOffset.y /= srcImg->mBlock.height;
        region.extent = srcImg->compressedMipmapPortion(region.extent, mipLevel);
    }
    if (needEmulatedDst) {
        region.dstSubresource.mipLevel = 0;
        region.dstOffset.x /= dstImg->mBlock.width;
        region.dstOffset.y /= dstImg->mBlock.height;
    }
    return region;
}

void CompressedImageInfo::destroy(VulkanDispatch* vk) {
    for (const auto& image : mCompressedMipmaps) {
        vk->vkDestroyImage(mDevice, image, nullptr);
    }
    vk->vkDestroyDescriptorPool(mDevice, mDecompDescriptorPool, nullptr);
    for (const auto& imageView : mCompressedMipmapsImageViews) {
        vk->vkDestroyImageView(mDevice, imageView, nullptr);
    }
    for (const auto& imageView : mOutputImageViews) {
        vk->vkDestroyImageView(mDevice, imageView, nullptr);
    }
    vk->vkDestroyImage(mDevice, mOutputImage, nullptr);
}

std::vector<VkImageMemoryBarrier> CompressedImageInfo::getImageBarriers(
    const VkImageMemoryBarrier& srcBarrier) {
    const VkImageSubresourceRange range = getImageSubresourceRange(srcBarrier.subresourceRange);

    std::vector<VkImageMemoryBarrier> imageBarriers;
    imageBarriers.reserve(range.levelCount + 1);

    // Add the barriers for the compressed mipmaps
    VkImageMemoryBarrier mipmapBarrier = srcBarrier;
    mipmapBarrier.subresourceRange.baseMipLevel = 0;
    mipmapBarrier.subresourceRange.levelCount = 1;
    imageBarriers.insert(imageBarriers.begin(), range.levelCount, mipmapBarrier);
    for (uint32_t j = 0; j < range.levelCount; j++) {
        imageBarriers[j].image = mCompressedMipmaps[range.baseMipLevel + j];
    }

    // Add a barrier for the output image
    imageBarriers.push_back(srcBarrier);
    imageBarriers.back().image = mOutputImage;

    return imageBarriers;
}

VkImageSubresourceRange CompressedImageInfo::getImageSubresourceRange(
    const VkImageSubresourceRange& range) const {
    VkImageSubresourceRange result = range;
    if (result.levelCount == VK_REMAINING_MIP_LEVELS) {
        result.levelCount = mMipLevels - range.baseMipLevel;
    }
    if (result.layerCount == VK_REMAINING_ARRAY_LAYERS) {
        result.layerCount = mLayerCount - range.baseArrayLayer;
    }
    return result;
}

VkResult CompressedImageInfo::initializeDecompressionPipeline(VulkanDispatch* vk, VkDevice device) {
    if (mDecompPipelineInitialized) {
        return VK_SUCCESS;
    }

    mDecompPipeline = mPipelineManager->get(mCompressedFormat, mImageType);
    if (mDecompPipeline == nullptr) {
        GFXSTREAM_ERROR("Failed to initialize GPU decompression pipeline");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const bool mesaAstc = isAstc() &&
                          GpuDecompressionPipelineManager::astcDecoder() == AstcDecoder::Mesa;
    const uint32_t viewLayers = mesaAstc ? mLayerCount : 1;
    const uint32_t descriptorCount = mMipLevels * viewLayers;
    const VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * descriptorCount},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 6 * descriptorCount},
    };
    VkDescriptorPoolCreateInfo dsPoolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = descriptorCount,
        .poolSizeCount = mesaAstc ? 2u : 1u,
        .pPoolSizes = poolSizes,
    };
    VkResult result =
        vk->vkCreateDescriptorPool(device, &dsPoolInfo, nullptr, &mDecompDescriptorPool);
    if (result != VK_SUCCESS) {
        GFXSTREAM_ERROR("GPU decompression error. vkCreateDescriptorPool failed: %d", result);
        return result;
    }

    std::vector<VkDescriptorSetLayout> layouts(descriptorCount, mDecompPipeline->descriptorSetLayout());

    VkDescriptorSetAllocateInfo dsInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = mDecompDescriptorPool,
        .descriptorSetCount = descriptorCount,
        .pSetLayouts = layouts.data(),
    };
    mDecompDescriptorSets.resize(descriptorCount);
    result = vk->vkAllocateDescriptorSets(device, &dsInfo, mDecompDescriptorSets.data());
    if (result != VK_SUCCESS) {
        GFXSTREAM_ERROR("GPU decompression error. vkAllocateDescriptorSets failed: %d", result);
        return result;
    }

    VkFormat shaderFormat = getShaderFormat(mOutputFormat);
    mCompressedMipmapsImageViews.resize(descriptorCount);
    mOutputImageViews.resize(descriptorCount);

    VkDescriptorImageInfo compressedMipmapsDescriptorImageInfo = {.imageLayout =
                                                                      VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo mDecompDescriptorImageInfo = {.imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet writeDescriptorSets[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &compressedMipmapsDescriptorImageInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &mDecompDescriptorImageInfo,
        }};

    // A view must not expose layers whose layouts differ from this dispatch's layout.
    for (uint32_t i = 0; i < descriptorCount; i++) {
        const uint32_t mip = i / viewLayers;
        const uint32_t layer = i % viewLayers;
        const uint32_t layers = mesaAstc ? 1 : mLayerCount;
        mCompressedMipmapsImageViews[i] =
            createDefaultImageView(vk, device, mCompressedMipmaps[mip], mCompressedMipmapsFormat,
                                   mImageType, 0, layers, layer);
        mOutputImageViews[i] = createDefaultImageView(vk, device, mOutputImage, shaderFormat,
                                                      mImageType, mip, layers, layer);
        if (!mCompressedMipmapsImageViews[i] || !mOutputImageViews[i]) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        if (mesaAstc) mPipelineManager->writeAstcDescriptors(mDecompDescriptorSets[i], mCompressedFormat);
        compressedMipmapsDescriptorImageInfo.imageView = mCompressedMipmapsImageViews[i];
        mDecompDescriptorImageInfo.imageView = mOutputImageViews[i];
        writeDescriptorSets[0].dstSet = mDecompDescriptorSets[i];
        writeDescriptorSets[1].dstSet = mDecompDescriptorSets[i];
        vk->vkUpdateDescriptorSets(device, 2, writeDescriptorSets, 0, nullptr);
    }

    mDecompPipelineInitialized = true;
    return VK_SUCCESS;
}

void CompressedImageInfo::decompress(VulkanDispatch* vk, VkCommandBuffer commandBuffer,
                                     const VkImageSubresourceRange& range) {
    vk->vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          mDecompPipeline->pipeline());
    if (isAstc() && GpuDecompressionPipelineManager::astcDecoder() == AstcDecoder::Mesa) {
        for (uint32_t mip = range.baseMipLevel; mip < range.baseMipLevel + range.levelCount; ++mip) {
            const auto extent = mipmapExtent(mip);
            const MesaAstcPushConstant constants = {
                {0, 0}, {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height)},
                0, mBlock.width, mBlock.height,
            };
            vk->vkCmdPushConstants(commandBuffer, mDecompPipeline->pipelineLayout(),
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
            for (uint32_t layer = range.baseArrayLayer;
                 layer < range.baseArrayLayer + range.layerCount; ++layer) {
                vk->vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                           mDecompPipeline->pipelineLayout(), 0, 1,
                                           &mDecompDescriptorSets[mip * mLayerCount + layer], 0, nullptr);
                vk->vkCmdDispatch(commandBuffer, ceil_div(extent.width, 2 * mBlock.width),
                                  ceil_div(extent.height, 2 * mBlock.height),
                                  mImageType == VK_IMAGE_TYPE_3D ? extent.depth : 1);
            }
        }
        return;
    }
    uint32_t dispatchZ = mExtent.depth == 1 ? range.layerCount : mExtent.depth;
    bool perPixel = false;  // Whether the shader operates per compressed block or per pixel
    if (isEtc2()) {
        const Etc2PushConstant pushConstant = {
            .compFormat = (uint32_t)mCompressedFormat,
            .baseLayer = mExtent.depth == 1 ? range.baseArrayLayer : 0};
        vk->vkCmdPushConstants(commandBuffer, mDecompPipeline->pipelineLayout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstant), &pushConstant);
    } else if (isAstc()) {
        uint32_t smallBlock = false;
        switch (mCompressedFormat) {
            case VK_FORMAT_ASTC_4x4_UNORM_BLOCK:
            case VK_FORMAT_ASTC_5x4_UNORM_BLOCK:
            case VK_FORMAT_ASTC_5x5_UNORM_BLOCK:
            case VK_FORMAT_ASTC_6x5_UNORM_BLOCK:
            case VK_FORMAT_ASTC_4x4_SRGB_BLOCK:
            case VK_FORMAT_ASTC_5x4_SRGB_BLOCK:
            case VK_FORMAT_ASTC_5x5_SRGB_BLOCK:
            case VK_FORMAT_ASTC_6x5_SRGB_BLOCK:
                smallBlock = true;
                break;
            default:
                break;
        }
        const AstcPushConstant pushConstant = {
            .blockSize = {mBlock.width, mBlock.height},
            .baseLayer = mExtent.depth == 1 ? range.baseArrayLayer : 0,
            .smallBlock = smallBlock};
        vk->vkCmdPushConstants(commandBuffer, mDecompPipeline->pipelineLayout(),
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstant), &pushConstant);
        // The old shader is per-block, the new shaders are per-pixel
        perPixel = GpuDecompressionPipelineManager::astcDecoder() != AstcDecoder::Old;
    }
    for (uint32_t i = range.baseMipLevel; i < range.baseMipLevel + range.levelCount; i++) {
        vk->vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    mDecompPipeline->pipelineLayout(), 0, 1,
                                    mDecompDescriptorSets.data() + i, 0, nullptr);
        VkExtent3D extent = perPixel ? mipmapExtent(i) : compressedMipmapExtent(i);
        vk->vkCmdDispatch(commandBuffer, ceil_div(extent.width, 8), ceil_div(extent.height, 8),
                          dispatchZ);
    }
}

VkExtent3D CompressedImageInfo::mipmapExtent(uint32_t level) const {
    return {
        .width = std::max<uint32_t>(mExtent.width >> level, 1),
        .height = std::max<uint32_t>(mExtent.height >> level, 1),
        .depth = std::max<uint32_t>(mExtent.depth >> level, 1),
    };
}

VkExtent3D CompressedImageInfo::compressedMipmapExtent(uint32_t level) const {
    VkExtent3D result = mipmapExtent(level);
    result.width = ceil_div(result.width, mBlock.width);
    result.height = ceil_div(result.height, mBlock.height);
    return result;
}

VkExtent3D CompressedImageInfo::compressedMipmapPortion(const VkExtent3D& origExtent,
                                                        uint32_t level) const {
    VkExtent3D maxExtent = compressedMipmapExtent(level);
    return {
        .width = std::min(ceil_div(origExtent.width, mBlock.width), maxExtent.width),
        .height = std::min(ceil_div(origExtent.height, mBlock.height), maxExtent.height),
        .depth = std::min(origExtent.depth, maxExtent.depth),
    };
}

}  // namespace vk
}  // namespace host
}  // namespace gfxstream
