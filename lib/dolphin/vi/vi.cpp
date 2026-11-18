#include <dolphin/vi.h>

#include "../../window.hpp"
#include "aurora/math.hpp"
#include "vi_internal.hpp"

#include <optional>
#include <mutex>

namespace aurora::vi {
static std::optional<GXRenderModeObj> g_renderMode;
static std::mutex sRenderModeMutex;

static Vec2<uint32_t> render_mode_size_unlocked() noexcept {
  if (!g_renderMode) {
    return {640, 480};
  }
  return {g_renderMode->fbWidth, g_renderMode->efbHeight};
}

void configure(const GXRenderModeObj* rm) noexcept {
  bool resized;
  {
    // HSD may configure VI inside the native retrace callback while the
    // rendering thread reads its dimensions.
    std::lock_guard lock{sRenderModeMutex};
    const auto oldSize = render_mode_size_unlocked();
    if (rm == nullptr) {
      g_renderMode.reset();
    } else {
      g_renderMode = *rm;
    }
    resized = render_mode_size_unlocked() != oldSize;
  }
  if (resized) {
    window::request_frame_buffer_resize();
  }
}

Vec2<uint32_t> configured_fb_size() noexcept {
  std::lock_guard lock{sRenderModeMutex};
  return render_mode_size_unlocked();
}
} // namespace aurora::vi

extern "C" {
void melee_vi_init();
void melee_vi_flush();
void melee_vi_configure(uint32_t mode, uint16_t y);
void VIInit() { melee_vi_init(); }
void VIConfigure(const GXRenderModeObj* rm) {
  aurora::vi::configure(rm);
  if (rm) melee_vi_configure(rm->viTVmode, rm->viYOrigin);
}
void VIConfigurePan(u16 xOrg, u16 yOrg, u16 width, u16 height) {
  (void)xOrg;
  (void)yOrg;
  (void)width;
  (void)height;
}
u32 VIGetTvFormat() { return 0; }
void VIFlush() { melee_vi_flush(); }

void VISetWindowTitle(const char* title) { aurora::window::set_title(title); }
void VISetWindowFullscreen(bool fullscreen) { aurora::window::set_fullscreen(fullscreen); }
bool VIGetWindowFullscreen() { return aurora::window::get_fullscreen(); }
void VISetWindowSize(uint32_t width, uint32_t height) { aurora::window::set_window_size(width, height); }
void VISetWindowPosition(uint32_t x, uint32_t y) { aurora::window::set_window_position(x, y); }
void VICenterWindow() { aurora::window::center_window(); }
void VISetFrameBufferScale(float scale) { aurora::window::set_frame_buffer_scale(scale); }
}
