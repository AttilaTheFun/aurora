#include "pipeline.hpp"

#include "../gfx/encoding.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/resource_cache.hpp"

#include "gx_fmt.hpp"
#include "shader_info.hpp"

#include <tracy/Tracy.hpp>

#ifdef __EMSCRIPTEN__
extern "C" void melee_browser_bind_group(WGPURenderPassEncoder, uint32_t, WGPUBindGroup);
extern "C" void melee_browser_bind_uniforms(WGPURenderPassEncoder, WGPUBindGroup, uint32_t, uint32_t);
#endif

namespace aurora::gx {

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  ZoneScoped;
  const auto shader = build_shader(config.shaderConfig);
  const auto label =
      fmt::format("GX Pipeline {:x} shader {:x}", xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX)),
                  xxh3_hash(config.shaderConfig));
  return build_pipeline(config, {}, shader, label.c_str());
}

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  if (!gfx::bind_pipeline(data.pipeline, pass)) {
    return;
  }

  const auto& resources = gfx::detail::resources();
#ifdef __EMSCRIPTEN__
  // Copy both offsets into the proxy queue; no temporary array survives this call.
  melee_browser_bind_uniforms(pass.Get(), resources.uniformBindGroup.Get(),
                             data.uniformRange.offset, data.immediateRange.offset);
#else
  pass.SetImmediates(0, &data.immediateData, sizeof(data.immediateData));
  const std::array offsets{data.uniformRange.offset};
  pass.SetBindGroup(1, resources.uniformBindGroup, offsets.size(), offsets.data());
#endif
  if (data.bindGroups.textureBindGroup) {
#ifdef __EMSCRIPTEN__
    const auto textures = gfx::find_bind_group(data.bindGroups.textureBindGroup);
    melee_browser_bind_group(pass.Get(), 2, textures.Get());
#else
    pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
#endif
  }
  pass.SetIndexBuffer(resources.indexBuffer, wgpu::IndexFormat::Uint16, data.idxRange.offset, data.idxRange.size);
  if (data.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  if (data.indexCount == 0) {
    pass.Draw(data.vtxCount, data.instanceCount);
  } else {
    pass.DrawIndexed(data.indexCount, data.instanceCount);
  }
}

} // namespace aurora::gx
