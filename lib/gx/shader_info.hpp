#pragma once

#include "gx.hpp"

namespace aurora::gx {
// Game-thread-only presentation aspect correction. GX register values remain unchanged.
void set_display_widescreen(bool enabled) noexcept;
void update_display_projection() noexcept;
ShaderInfo build_shader_info(const ShaderConfig& config) noexcept;
gfx::Range build_uniform(const ShaderInfo& info) noexcept;
u8 color_channel(GXChannelID id) noexcept;
}; // namespace aurora::gx
