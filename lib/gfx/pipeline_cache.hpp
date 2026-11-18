#pragma once

#include "types.hpp"

#include <functional>

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

using NewPipelineCallback = std::function<wgpu::RenderPipeline()>;

// Observer is installed before initialization and must not call back into gfx.
// Called from recording/compiler threads; observers must not perform file I/O.
struct PipelineEvent {
  enum Kind { Discovered, Compiled, Waited, WarmupComplete } kind;
  uint64_t hash, durationNs;
  uint32_t frame;
  bool cached;
};
using PipelineObserver = void (*)(const PipelineEvent&);
void set_pipeline_observer(PipelineObserver observer);
struct PipelinePreparation { uint64_t total, remaining; bool failed, cacheFailed; };
PipelinePreparation pipeline_preparation();
// Native-only startup barrier; call before recording any game frames.
bool prepare_known_pipelines();

struct PipelineTiming { uint64_t waits, waitNs, builds, buildNs; };
PipelineTiming take_pipeline_timing();
void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();

template <typename Config>
PipelineRef find_pipeline(ShaderType type, const Config& config, NewPipelineCallback&& cb);

bool get_pipeline(PipelineRef ref, wgpu::RenderPipeline& pipeline);

} // namespace aurora::gfx
