// Copyright 2026 The Android Open Source Project
// SPDX-License-Identifier: Apache-2.0
#include "mesa_astc_luts.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "mesa/texcompress_astc_luts.h"
#include "vk_format_utils.h"

namespace gfxstream::host::vk {
namespace {
constexpr VkExtent2D kBlocks[] = {
    {4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6}, {8, 5}, {8, 6},
    {8, 8}, {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12},
};
}

bool MesaAstcLuts::initialize(VulkanDispatch* vk, VkPhysicalDevice physicalDevice,
                            VkDevice device) {
    if (mBuffer) return true;
    auto& luts = Granite::get_astc_luts();
    VkPhysicalDeviceProperties properties;
    vk->vkGetPhysicalDeviceProperties(physicalDevice, &properties);
    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        properties.limits.minTexelBufferOffsetAlignment, 1);
    struct Table {
        const void* data;
        size_t size;
        VkFormat format;
        VkDeviceSize offset = 0;
    };
    std::vector<Table> tables = {
        {luts.color_endpoint.lut, sizeof(luts.color_endpoint.lut), VK_FORMAT_R16G16B16A16_UINT},
        {luts.color_endpoint.unquant_lut, luts.color_endpoint.unquant_offset, VK_FORMAT_R8_UINT},
        {luts.weights.lut, sizeof(luts.weights.lut), VK_FORMAT_R8G8B8A8_UINT},
        {luts.weights.unquant_lut, luts.weights.unquant_offset, VK_FORMAT_R8_UINT},
        {luts.integer.trits_quints, sizeof(luts.integer.trits_quints), VK_FORMAT_R16_UINT},
    };
    for (const auto& block : kBlocks) {
        auto& table = luts.get_partition_table(block.width, block.height);
        tables.push_back({table.lut_buffer.data(), table.lut_buffer.size(), VK_FORMAT_R8_UINT});
    }
    VkDeviceSize size = 0;
    for (auto& table : tables) {
        table.offset = ((size + alignment - 1) / alignment) * alignment;
        size = table.offset + table.size;
    }
    const VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vk->vkCreateBuffer(device, &bufferInfo, nullptr, &mBuffer) != VK_SUCCESS) return false;
    VkMemoryRequirements requirements;
    vk->vkGetBufferMemoryRequirements(device, mBuffer, &requirements);
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vk->vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    uint32_t memoryType = UINT32_MAX;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const auto flags = memoryProperties.memoryTypes[i].propertyFlags;
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            memoryType = i;
            if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) break;
        }
    }
    if (memoryType == UINT32_MAX) {
        destroy(vk, device);
        return false;
    }
    const VkMemoryAllocateInfo allocation = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = memoryType,
    };
    if (vk->vkAllocateMemory(device, &allocation, nullptr, &mMemory) != VK_SUCCESS ||
        vk->vkBindBufferMemory(device, mBuffer, mMemory, 0) != VK_SUCCESS) {
        destroy(vk, device);
        return false;
    }
    void* data = nullptr;
    if (vk->vkMapMemory(device, mMemory, 0, VK_WHOLE_SIZE, 0, &data) != VK_SUCCESS) {
        destroy(vk, device);
        return false;
    }
    for (const auto& table : tables) {
        std::memcpy(static_cast<uint8_t*>(data) + table.offset, table.data, table.size);
    }
    VkResult result = VK_SUCCESS;
    if (!(memoryProperties.memoryTypes[memoryType].propertyFlags &
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        const VkMappedMemoryRange range = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = mMemory,
            .size = VK_WHOLE_SIZE,
        };
        result = vk->vkFlushMappedMemoryRanges(device, 1, &range);
    }
    vk->vkUnmapMemory(device, mMemory);
    if (result != VK_SUCCESS) {
        destroy(vk, device);
        return false;
    }
    for (size_t i = 0; i < tables.size(); ++i) {
        const VkBufferViewCreateInfo view = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            .buffer = mBuffer,
            .format = tables[i].format,
            .offset = tables[i].offset,
            .range = tables[i].size,
        };
        if (vk->vkCreateBufferView(device, &view, nullptr, &mViews[i]) != VK_SUCCESS) {
            destroy(vk, device);
            return false;
        }
    }
    return true;
}

void MesaAstcLuts::writeDescriptors(VulkanDispatch* vk, VkDevice device, VkDescriptorSet set,
                                   VkFormat format) const {
    const auto block = astcBlockSize(format);
    size_t partition = 0;
    for (size_t i = 0; i < std::size(kBlocks); ++i) {
        if (kBlocks[i].width == block.width && kBlocks[i].height == block.height) partition = i;
    }
    VkWriteDescriptorSet writes[6] = {};
    for (uint32_t i = 0; i < 6; ++i) {
        writes[i] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = i + 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
            .pTexelBufferView = &mViews[i == 5 ? 5 + partition : i],
        };
    }
    vk->vkUpdateDescriptorSets(device, 6, writes, 0, nullptr);
}

void MesaAstcLuts::destroy(VulkanDispatch* vk, VkDevice device) {
    for (auto& view : mViews) {
        vk->vkDestroyBufferView(device, view, nullptr);
        view = VK_NULL_HANDLE;
    }
    vk->vkDestroyBuffer(device, mBuffer, nullptr);
    vk->vkFreeMemory(device, mMemory, nullptr);
    mBuffer = VK_NULL_HANDLE;
    mMemory = VK_NULL_HANDLE;
}
}  // namespace gfxstream::host::vk
