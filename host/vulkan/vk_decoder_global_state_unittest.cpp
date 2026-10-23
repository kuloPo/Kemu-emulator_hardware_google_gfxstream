// copyright (c) 2022 the android open source project
//
// licensed under the apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// you may obtain a copy of the license at
//
// http://www.apache.org/licenses/license-2.0
//
// unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// see the license for the specific language governing permissions and
// limitations under the license.

#include "vk_decoder_global_state.cpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gfxstream/testing/TestUtils.h"

namespace gfxstream {
namespace host {
namespace vk {
namespace {
using ::testing::_;
using ::testing::InSequence;
using ::testing::MockFunction;
using ::testing::Return;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

template <typename Attachment>
struct RenderPassTypes;

template <>
struct RenderPassTypes<VkAttachmentDescription> {
    using CreateInfo = VkRenderPassCreateInfo;
    using Subpass = VkSubpassDescription;
    using Reference = VkAttachmentReference;
};

template <>
struct RenderPassTypes<VkAttachmentDescription2> {
    using CreateInfo = VkRenderPassCreateInfo2;
    using Subpass = VkSubpassDescription2;
    using Reference = VkAttachmentReference2;
};

template <typename Attachment>
class AttachmentOpsTest : public Test {
   protected:
    using Types = RenderPassTypes<Attachment>;
    using CreateInfo = typename Types::CreateInfo;
    using Subpass = typename Types::Subpass;
    using Reference = typename Types::Reference;

    AttachmentOpsTest() {
        source.format = VK_FORMAT_D24_UNORM_S8_UINT;
        source.loadOp = source.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        source.storeOp = source.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        reference.layout = VK_IMAGE_LAYOUT_GENERAL;
        subpass.pDepthStencilAttachment = &reference;
        info.attachmentCount = 1;
        info.pAttachments = &source;
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
    }

    bool normalize() {
        adjusted.clear();
        clears.clear();
        return normalizeRenderPassAttachments(info, adjusted, clears);
    }

    Attachment source{};
    Reference reference{};
    Subpass subpass{};
    CreateInfo info{};
    std::vector<Attachment> adjusted;
    std::vector<AttachmentClear> clears;
};

using RenderPassAttachmentTypes =
    ::testing::Types<VkAttachmentDescription, VkAttachmentDescription2>;
TYPED_TEST_SUITE(AttachmentOpsTest, RenderPassAttachmentTypes);

TYPED_TEST(AttachmentOpsTest, ConvertsAllFormatsWithoutChangingGuestData) {
    for (auto format : {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, VK_FORMAT_S8_UINT,
                        VK_FORMAT_D24_UNORM_S8_UINT}) {
        this->source.format = format;
        const auto aspects = getFormatAspects(format);
        this->subpass.pDepthStencilAttachment =
            (aspects & VK_IMAGE_ASPECT_COLOR_BIT) ? nullptr : &this->reference;
        this->subpass.colorAttachmentCount = (aspects & VK_IMAGE_ASPECT_COLOR_BIT) ? 1 : 0;
        this->subpass.pColorAttachments =
            (aspects & VK_IMAGE_ASPECT_COLOR_BIT) ? &this->reference : nullptr;
        ASSERT_TRUE(this->normalize());
        ASSERT_EQ(this->adjusted.size(), 1);
        ASSERT_EQ(this->clears.size(), 1);
        EXPECT_EQ(this->clears[0].attachment, 0);
        EXPECT_EQ(this->clears[0].aspects, aspects);
        const auto& result = this->adjusted[0];
        if (aspects & ~VK_IMAGE_ASPECT_STENCIL_BIT) {
            EXPECT_EQ(result.loadOp, VK_ATTACHMENT_LOAD_OP_CLEAR);
            EXPECT_EQ(result.storeOp, VK_ATTACHMENT_STORE_OP_STORE);
        } else {
            EXPECT_EQ(result.loadOp, this->source.loadOp);
            EXPECT_EQ(result.storeOp, this->source.storeOp);
        }
        if (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) {
            EXPECT_EQ(result.stencilLoadOp, VK_ATTACHMENT_LOAD_OP_CLEAR);
            EXPECT_EQ(result.stencilStoreOp, VK_ATTACHMENT_STORE_OP_STORE);
        } else {
            EXPECT_EQ(result.stencilLoadOp, this->source.stencilLoadOp);
            EXPECT_EQ(result.stencilStoreOp, this->source.stencilStoreOp);
        }
        EXPECT_EQ(this->source.loadOp, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
        EXPECT_EQ(this->source.stencilStoreOp, VK_ATTACHMENT_STORE_OP_DONT_CARE);
    }
}

TYPED_TEST(AttachmentOpsTest, PreservesExplicitLoadClearAndNone) {
    for (auto load : {VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR,
                      VK_ATTACHMENT_LOAD_OP_NONE_EXT}) {
        for (auto store : {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_NONE}) {
            this->source.loadOp = this->source.stencilLoadOp = load;
            this->source.storeOp = this->source.stencilStoreOp = store;
            EXPECT_FALSE(this->normalize());
            EXPECT_TRUE(this->adjusted.empty());
            EXPECT_TRUE(this->clears.empty());
        }
        this->source.storeOp = this->source.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ASSERT_TRUE(this->normalize());
        EXPECT_EQ(this->adjusted[0].loadOp, load);
        EXPECT_EQ(this->adjusted[0].stencilLoadOp, load);
        EXPECT_EQ(this->adjusted[0].storeOp, VK_ATTACHMENT_STORE_OP_STORE);
        EXPECT_EQ(this->adjusted[0].stencilStoreOp, VK_ATTACHMENT_STORE_OP_STORE);
        EXPECT_TRUE(this->clears.empty());
    }
}

TYPED_TEST(AttachmentOpsTest, KeepsReadOnlyAspectsAndClearsWritableAspects) {
    for (auto layout :
         {VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
          VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
          VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL}) {
        this->reference.layout = layout;
        ASSERT_TRUE(this->normalize());
        const auto& result = this->adjusted[0];
        EXPECT_EQ(result.loadOp,
                  layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL
                      ? VK_ATTACHMENT_LOAD_OP_CLEAR
                      : VK_ATTACHMENT_LOAD_OP_DONT_CARE);
        EXPECT_EQ(result.stencilLoadOp,
                  layout == VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
                      ? VK_ATTACHMENT_LOAD_OP_CLEAR
                      : VK_ATTACHMENT_LOAD_OP_DONT_CARE);
        EXPECT_EQ(result.storeOp, VK_ATTACHMENT_STORE_OP_STORE);
        EXPECT_EQ(result.stencilStoreOp, VK_ATTACHMENT_STORE_OP_STORE);
    }
}

TYPED_TEST(AttachmentOpsTest, DoesNotClearAnInputOnlyFirstUse) {
    typename TestFixture::Subpass subpasses[2] = {this->subpass, this->subpass};
    subpasses[0].pDepthStencilAttachment = nullptr;
    subpasses[0].inputAttachmentCount = 1;
    subpasses[0].pInputAttachments = &this->reference;
    this->info.subpassCount = 2;
    this->info.pSubpasses = subpasses;
    ASSERT_TRUE(this->normalize());
    EXPECT_TRUE(this->clears.empty());
    EXPECT_EQ(this->adjusted[0].loadOp, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
    EXPECT_EQ(this->adjusted[0].storeOp, VK_ATTACHMENT_STORE_OP_STORE);
}

TYPED_TEST(AttachmentOpsTest, HandlesEmptyRenderPass) {
    this->info.attachmentCount = 0;
    this->info.pAttachments = nullptr;
    EXPECT_FALSE(this->normalize());
    EXPECT_TRUE(this->clears.empty());
}

TEST(AttachmentOps, HonorsSeparateStencilLayout) {
    VkAttachmentReferenceStencilLayout stencilLayout{};
    stencilLayout.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_STENCIL_LAYOUT;
    stencilLayout.stencilLayout = VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL;
    VkAttachmentReference2 reference{};
    reference.pNext = &stencilLayout;
    reference.layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    EXPECT_EQ(
        clearableAspects(reference) & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT),
        VK_IMAGE_ASPECT_DEPTH_BIT);
    stencilLayout.stencilLayout = VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL;
    reference.layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    EXPECT_EQ(
        clearableAspects(reference) & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT),
        VK_IMAGE_ASPECT_STENCIL_BIT);
}

TEST(AttachmentOps, ExtendsClearArrayAndPreservesExplicitAspects) {
    VkClearValue source[2]{};
    source[0].color.float32[0] = 0.75f;
    source[1].depthStencil = {0.25f, 27};
    VkRenderPassBeginInfo begin{};
    begin.clearValueCount = 2;
    begin.pClearValues = source;
    std::vector<VkClearValue> values;
    prepareRenderPassClearValues({{1, VK_IMAGE_ASPECT_DEPTH_BIT}, {3, VK_IMAGE_ASPECT_COLOR_BIT}},
                                 begin, values);
    ASSERT_EQ(begin.clearValueCount, 4);
    EXPECT_EQ(begin.pClearValues, values.data());
    EXPECT_EQ(values[0].color.float32[0], 0.75f);
    EXPECT_EQ(values[1].depthStencil.depth, 1.0f);
    EXPECT_EQ(values[1].depthStencil.stencil, 27);
    for (auto channel : values[3].color.uint32) EXPECT_EQ(channel, 0);
    EXPECT_EQ(source[1].depthStencil.depth, 0.25f);
}

TEST(AttachmentOps, CreatesClearValuesWhenGuestSuppliedNone) {
    VkRenderPassBeginInfo begin{};
    std::vector<VkClearValue> values;
    prepareRenderPassClearValues({{0, VK_IMAGE_ASPECT_COLOR_BIT},
                                  {2, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT}},
                                 begin, values);
    ASSERT_EQ(begin.clearValueCount, 3);
    EXPECT_EQ(values[0].color.uint32[0], 0);
    EXPECT_EQ(values[2].depthStencil.depth, 1.0f);
    EXPECT_EQ(values[2].depthStencil.stencil, 0);
}

TEST(AttachmentOps, PreservesExplicitDepthWhenOnlyStencilBecomesClear) {
    VkClearValue source{};
    source.depthStencil = {0.125f, 99};
    VkRenderPassBeginInfo begin{};
    begin.clearValueCount = 1;
    begin.pClearValues = &source;
    std::vector<VkClearValue> values;
    prepareRenderPassClearValues({{0, VK_IMAGE_ASPECT_STENCIL_BIT}}, begin, values);
    EXPECT_EQ(values[0].depthStencil.depth, 0.125f);
    EXPECT_EQ(values[0].depthStencil.stencil, 0);
    EXPECT_EQ(source.depthStencil.stencil, 99);
    values.clear();
    begin.pClearValues = &source;
    prepareRenderPassClearValues({}, begin, values);
    EXPECT_EQ(begin.pClearValues, &source);
    EXPECT_TRUE(values.empty());
}

class VkDecoderGlobalStateExternalFenceTest : public Test {
protected:
    class MockDispatch {
       public:
        MOCK_METHOD(VkResult, vkGetFenceStatus, (VkDevice device, VkFence fence), ());
        MOCK_METHOD(VkResult,
                    vkResetFences,
                    (VkDevice device, uint32_t numFences, const VkFence* fence),
                    ());
    };

    VkDecoderGlobalStateExternalFenceTest()
        : mDevice(reinterpret_cast<VkDevice>(0x2222'0000)), mPool(&mMockDispatch, mDevice) {}

    ~VkDecoderGlobalStateExternalFenceTest() {
        mPool.popAll();
    }

    MockDispatch mMockDispatch;
    VkDevice mDevice;
    ExternalFencePool<MockDispatch> mPool;
};

using VkDecoderGlobalStateExternalFenceDeathTest = VkDecoderGlobalStateExternalFenceTest;

TEST_F(VkDecoderGlobalStateExternalFenceTest, poolNoDeviceFences) {
    VkFenceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = 0,
        .flags = 0,
    };
    ASSERT_EQ(VK_NULL_HANDLE, mPool.pop(&createInfo));
}

TEST_F(VkDecoderGlobalStateExternalFenceTest, poolReuseSignalledFence) {
    {
        InSequence s;
        EXPECT_CALL(mMockDispatch, vkGetFenceStatus(_, _)).WillOnce(Return(VK_SUCCESS));
        EXPECT_CALL(mMockDispatch, vkResetFences(_, _, _)).WillOnce(Return(VK_SUCCESS));
    }

    VkFence fence = reinterpret_cast<VkFence>(0x1234'0000);
    VkFenceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = 0,
        .flags = 0,
    };
    mPool.add(fence);
    VkFence reusedFence = mPool.pop(&createInfo);

    ASSERT_EQ(fence, reusedFence);
}

TEST_F(VkDecoderGlobalStateExternalFenceTest, poolReuseSignalledFenceAsSignaled) {
    {
        InSequence s;
        EXPECT_CALL(mMockDispatch, vkGetFenceStatus(_, _)).WillOnce(Return(VK_SUCCESS));
        EXPECT_CALL(mMockDispatch, vkResetFences(_, _, _)).Times(0);
    }

    VkFence fence = reinterpret_cast<VkFence>(0x1234'0000);
    VkFenceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = 0,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    mPool.add(fence);
    VkFence reusedFence = mPool.pop(&createInfo);

    ASSERT_EQ(fence, reusedFence);
}

TEST_F(VkDecoderGlobalStateExternalFenceTest, poolUnsignalledFence) {
    {
        InSequence s;
        EXPECT_CALL(mMockDispatch, vkGetFenceStatus(_, _)).WillOnce(Return(VK_NOT_READY));
        EXPECT_CALL(mMockDispatch, vkResetFences(_, _, _)).Times(0);
    }

    VkFence fence = reinterpret_cast<VkFence>(0x1234'0000);
    VkFenceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = 0,
        .flags = 0,
    };
    mPool.add(fence);

    ASSERT_EQ(VK_NULL_HANDLE, mPool.pop(&createInfo));
}

TEST_F(VkDecoderGlobalStateExternalFenceTest, poolPopAll) {
    VkFence fence1 = reinterpret_cast<VkFence>(0x1234'0000);
    VkFence fence2 = reinterpret_cast<VkFence>(0x2234'0000);
    VkFence fence3 = reinterpret_cast<VkFence>(0x3234'0000);
    mPool.add(fence1);
    mPool.add(fence2);
    mPool.add(fence3);

    std::vector<VkFence> result = mPool.popAll();
    ASSERT_THAT(result, UnorderedElementsAre(fence1, fence2, fence3));
}

TEST_F(VkDecoderGlobalStateExternalFenceDeathTest, undestroyedFences) {
    ASSERT_DEATH(
        {
            ExternalFencePool<MockDispatch> pool(&mMockDispatch, mDevice);
            VkFence fence = reinterpret_cast<VkFence>(0x1234'0000);
            pool.add(fence);
        },
        MatchesStdRegex(
            "External fence pool for VkDevice:0000000022220000|0x22220000 destroyed but 1 "
            "fences still not destroyed."));
}

}  // namespace
}  // namespace vk
}  // namespace host
}  // namespace gfxstream
