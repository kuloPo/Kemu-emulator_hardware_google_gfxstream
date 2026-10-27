// Copyright 2026 The Android Open Source Project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include "goldfish_vk_dispatch.h"

namespace gfxstream::host::vk {

// Immutable lookup buffers shared by all Mesa ASTC pipelines on a device.
class MesaAstcLuts {
public:
    bool initialize(VulkanDispatch* vk, VkPhysicalDevice physicalDevice, VkDevice device);
    void writeDescriptors(VulkanDispatch* vk, VkDevice device, VkDescriptorSet set,
                          VkFormat format) const;
    void destroy(VulkanDispatch* vk, VkDevice device);

private:
    VkBuffer mBuffer = VK_NULL_HANDLE;
    VkDeviceMemory mMemory = VK_NULL_HANDLE;
    std::array<VkBufferView, 19> mViews = {};
};

}  // namespace gfxstream::host::vk
