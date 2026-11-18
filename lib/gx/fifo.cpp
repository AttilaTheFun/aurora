#include "fifo.hpp"

#include "../thread.hpp"
#include "command_processor.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../webgpu/gpu.hpp"
#include <condition_variable>
#include <deque>
#include <memory>
#include <thread>
#include <chrono>

extern "C" int OSDisableInterrupts();
extern "C" int OSRestoreInterrupts(int);

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>

#include <tracy/Tracy.hpp>

namespace aurora::gx::fifo {
namespace detail {
uint8_t* sBufferData = nullptr;
uint32_t sBufferSize = 0;
uint32_t sBufferCapacity = 0;
bool sInDisplayList = false;
uint8_t* sDlBuffer = nullptr;
uint32_t sDlSize = 0;
uint32_t sDlWritePos = 0;
} // namespace detail

namespace {
constexpr Module Log{"aurora::gx::fifo"};
constexpr auto kProcessingMode = ProcessingMode::Thread;
constexpr uint32_t kDrawBatchSize = 1;

bool sFrameActive = false;
uint32_t sPendingDraws = 0;
std::atomic<uint64_t> sPublished{0};
std::atomic<uint64_t> sProcessed{0};
uint64_t sStreamBase = 0;
std::mutex sBufferMutex;
std::atomic<uint32_t> sWorkerWake{0};
thread::Thread sWorkerThread;
std::atomic<DrawDoneCallback> sDrawDoneCallback{nullptr};
std::atomic<DrawDoneSubmitter> sDrawDoneSubmitter{nullptr};
std::atomic<uint64_t> sDrawDoneIssued{0};
uint64_t sDrawDoneCompleted = 0; // sCompletionMutex
thread_local bool sInsideDrawDone = false;

std::mutex sCompletionMutex;
std::condition_variable sCompletionWake;
struct Completion {
  wgpu::Future future;
  std::shared_ptr<std::atomic<bool>> success;
};
std::deque<Completion> sCompletions;
std::thread sCompletionWorker;
bool sCompletionStop = false;
gfx::EncoderTaskId sDrawDoneTask = gfx::InvalidEncoderTask;

void dispatch_draw_done() noexcept {
  const int enabled = OSDisableInterrupts();
  if (const auto callback = sDrawDoneCallback.load(std::memory_order_acquire); callback != nullptr) {
    sInsideDrawDone = true;
    callback();
    sInsideDrawDone = false;
  }
  OSRestoreInterrupts(enabled);
}

void completion_main() {
  for (;;) {
    Completion completion;
    {
      std::unique_lock lock{sCompletionMutex};
      sCompletionWake.wait(lock, [] { return sCompletionStop || !sCompletions.empty(); });
      if (sCompletions.empty()) return;
      completion = std::move(sCompletions.front());
      sCompletions.pop_front();
    }
#ifdef __EMSCRIPTEN__
    // JS promises resolve on the browser thread; never block that event loop
    // or require Asyncify for every game/render worker. Poll the C++ future.
    auto result = wgpu::WaitStatus::TimedOut;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    do {
      result = webgpu::g_instance.WaitAny(completion.future, 0);
      if (result != wgpu::WaitStatus::TimedOut) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
#else
    const auto result = webgpu::g_instance.WaitAny(completion.future, 10000000000ULL);
#endif
    AURORA_ASSERT(result == wgpu::WaitStatus::Success &&
                  completion.success->load(std::memory_order_acquire),
                  "GX draw-done GPU completion failed");
    dispatch_draw_done();
    {
      std::lock_guard lock{sCompletionMutex};
      ++sDrawDoneCompleted;
    }
    sCompletionWake.notify_all();
  }
}

void submit_draw_done(const gfx::EncoderTaskCompletionContext& ctx, const void*, size_t, void*) {
  auto success = std::make_shared<std::atomic<bool>>(false);
  auto future = ctx.queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
    [success](wgpu::QueueWorkDoneStatus status, wgpu::StringView) {
      success->store(status == wgpu::QueueWorkDoneStatus::Success, std::memory_order_release);
    });
  {
    std::lock_guard lock{sCompletionMutex};
    AURORA_ASSERT(!sCompletionStop, "GX completion queued after shutdown");
    sCompletions.push_back({future, std::move(success)});
  }
  sCompletionWake.notify_all();
}

void wake_worker() noexcept {
  sWorkerWake.fetch_add(1, std::memory_order_release);
  sWorkerWake.notify_all();
}

void process_to(uint64_t target, std::memory_order order) noexcept {
  uint64_t processed = sProcessed.load(std::memory_order_relaxed);
  while (processed < target) {
    ProcessResult result{};
    {
      std::lock_guard lock{sBufferMutex};
      AURORA_ASSERT(processed >= sStreamBase && target <= sStreamBase + detail::sBufferSize,
                    "FIFO processing range [{}, {}) is outside buffered range [{}, {})", processed, target, sStreamBase,
                    sStreamBase + detail::sBufferSize);
      const auto start = static_cast<uint32_t>(processed - sStreamBase);
      const auto size = static_cast<uint32_t>(target - processed);
      result = process(detail::sBufferData + start, size);
    }
    AURORA_ASSERT(result.bytesProcessed > 0 && result.bytesProcessed <= target - processed,
                  "FIFO processor made invalid progress: processed {} of {} remaining bytes", result.bytesProcessed,
                  target - processed);
    if (result.drawDone) {
      // CPU parsing is not GPU completion. Record an ordered task and fence
      // only after the containing frame's command buffer has been submitted.
      sDrawDoneIssued.fetch_add(1, std::memory_order_release);
      AURORA_ASSERT(gfx::push_encoder_task_from_fifo(sDrawDoneTask),
                    "GX draw-done requires an active render frame");
    }
    processed += result.bytesProcessed;
    sProcessed.store(processed, order);
    sProcessed.notify_all();
  }
}

void worker_main(std::stop_token token) noexcept {
  std::stop_callback wakeOnStop{token, wake_worker};
  while (true) {
    const uint32_t event = sWorkerWake.load(std::memory_order_acquire);
    const uint64_t processed = sProcessed.load(std::memory_order_relaxed);
    const uint64_t published = sPublished.load(std::memory_order_acquire);
    if (published != processed) {
      process_to(published, std::memory_order_release);
      continue;
    }

    if (token.stop_requested()) {
      break;
    }
    sWorkerWake.wait(event, std::memory_order_acquire);
  }
}

void start_worker() {
  if (kProcessingMode != ProcessingMode::Thread || sWorkerThread.joinable()) {
    return;
  }
  sWorkerThread = thread::Thread{{
                                     .name = "Aurora FIFO processor",
                                     .affinity = thread::Affinity::SharedCache,
                                 },
                                 worker_main};
}

void stop_worker() {
  if (!sWorkerThread.joinable()) {
    return;
  }
  sWorkerThread.request_stop();
  sWorkerThread.join();
}
} // namespace

ProcessingMode processing_mode() noexcept { return kProcessingMode; }

void init() {
  AURORA_ASSERT(!sCompletionWorker.joinable(), "GX FIFO reinitialization requires shutdown");
  stop_worker();

  constexpr uint32_t initialCapacity = 64 * 1024;
  free(detail::sBufferData);
  detail::sBufferData = static_cast<uint8_t*>(malloc(initialCapacity));
  AURORA_ASSERT(detail::sBufferData != nullptr, "fifo::init: failed to allocate {} bytes", initialCapacity);
  detail::sBufferSize = 0;
  detail::sBufferCapacity = initialCapacity;
  detail::sInDisplayList = false;
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;

  sFrameActive = false;
  sPendingDraws = 0;
  sStreamBase = 0;
  sPublished.store(0, std::memory_order_relaxed);
  sProcessed.store(0, std::memory_order_relaxed);
  sWorkerWake.store(0, std::memory_order_relaxed);

  sDrawDoneTask = gfx::register_encoder_task_type({
    .label = "GX draw-done fence",
    .callback = [](const gfx::EncoderTaskContext&, const wgpu::CommandEncoder&,
                   const void*, size_t, void*) {},
    .afterSubmit = submit_draw_done,
  });
  sCompletionStop = false;
  sDrawDoneIssued.store(0, std::memory_order_relaxed);
  sDrawDoneCompleted = 0;
  sCompletionWorker = std::thread{completion_main};
  start_worker();
}

void shutdown() {
  const int enabled = OSDisableInterrupts();
  AURORA_ASSERT(enabled, "GX FIFO shutdown requires interrupts enabled");
  OSRestoreInterrupts(enabled);
  stop_worker();
  gfx::gpu_synchronize(); // All submitted-frame completion tasks must be queued.
  {
    std::lock_guard lock{sCompletionMutex};
    sCompletionStop = true;
  }
  sCompletionWake.notify_all();
  if (sCompletionWorker.joinable()) sCompletionWorker.join();
  gfx::unregister_encoder_task_type(sDrawDoneTask);
  sDrawDoneTask = gfx::InvalidEncoderTask;
  sDrawDoneCallback.store(nullptr, std::memory_order_release);
  sDrawDoneSubmitter.store(nullptr, std::memory_order_release);
}

void begin_frame() noexcept { sFrameActive = true; }

void end_frame() noexcept {
  sFrameActive = false;
  clear_draw_cache(); // command_processor
}

void write_data_grow(const void* data, uint32_t length) {
  const uint64_t needed64 = static_cast<uint64_t>(detail::sBufferSize) + length;
  AURORA_ASSERT(needed64 <= std::numeric_limits<uint32_t>::max(), "fifo::write_data: buffer size overflow");
  const auto needed = static_cast<uint32_t>(needed64);
  const auto doubledCapacity = static_cast<uint64_t>(detail::sBufferCapacity) * 2;
  const auto newCapacity = static_cast<uint32_t>(
      std::min<uint64_t>(std::max(doubledCapacity, needed64), std::numeric_limits<uint32_t>::max()));
  const auto grow = [newCapacity] {
    auto* resized = static_cast<uint8_t*>(realloc(detail::sBufferData, newCapacity));
    AURORA_ASSERT(resized != nullptr, "fifo::write_data: failed to allocate {} bytes", newCapacity);
    detail::sBufferData = resized;
  };
  if (sWorkerThread.joinable()) {
    std::lock_guard lock{sBufferMutex};
    grow();
  } else {
    grow();
  }
  std::memcpy(detail::sBufferData + detail::sBufferSize, data, length);
  detail::sBufferSize = needed;
  detail::sBufferCapacity = newCapacity;
}

void publish() noexcept {
  if (!sFrameActive || kProcessingMode == ProcessingMode::Drain || detail::sInDisplayList) {
    return;
  }

  const uint64_t target = sStreamBase + detail::sBufferSize;
  if (target > sPublished.load(std::memory_order_relaxed)) {
    sPendingDraws = 0;
    sPublished.store(target, std::memory_order_release);
    if (kProcessingMode == ProcessingMode::Thread) {
      wake_worker();
    } else {
      process_to(target, std::memory_order_relaxed);
    }
  }
}

DrawDoneCallback set_draw_done_callback(DrawDoneCallback callback) noexcept {
  return sDrawDoneCallback.exchange(callback, std::memory_order_acq_rel);
}

DrawDoneSubmitter set_draw_done_submitter(DrawDoneSubmitter submitter) noexcept {
  return sDrawDoneSubmitter.exchange(submitter, std::memory_order_acq_rel);
}

void submit_draw_done_frame() {
  if (auto submitter = sDrawDoneSubmitter.load(std::memory_order_acquire)) submitter();
}

void wait_draw_done() {
  AURORA_ASSERT(!sInsideDrawDone, "GXWaitDrawDone cannot wait inside its own callback");
  drain(); // Publish/parse all preceding markers before taking the watermark.
  const uint64_t target = sDrawDoneIssued.load(std::memory_order_acquire);
  const int enabled = OSDisableInterrupts();
  // The completion callback must acquire the gate before it wakes us.
  OSRestoreInterrupts(1);
  bool complete;
  {
    std::unique_lock lock{sCompletionMutex};
    complete = sCompletionWake.wait_for(lock, std::chrono::seconds(10), [target] {
      return sDrawDoneCompleted >= target;
    });
  }
  OSDisableInterrupts();
  OSRestoreInterrupts(enabled);
  AURORA_ASSERT(complete, "GXWaitDrawDone timed out; the frame owner must submit the marked frame");
}

void finish_draw() noexcept {
  if (!sFrameActive || kProcessingMode == ProcessingMode::Drain || detail::sInDisplayList) {
    return;
  }
  if (++sPendingDraws >= kDrawBatchSize) {
    publish();
  }
}

void patch_u32(uint32_t offset, uint32_t val) {
  AURORA_ASSERT(!detail::sInDisplayList && offset <= detail::sBufferSize &&
                    sizeof(uint32_t) <= detail::sBufferSize - offset,
                "fifo::patch_u32: invalid patch offset {} (buffer size {})", offset, detail::sBufferSize);
  AURORA_ASSERT(sStreamBase + offset >= sPublished.load(std::memory_order_relaxed),
                "fifo::patch_u32: offset {} is below the published watermark", offset);
  const auto out = bswap(val);
  std::memcpy(detail::sBufferData + offset, &out, sizeof(out));
}

void begin_display_list(uint8_t* buf, uint32_t size) {
  detail::sInDisplayList = true;
  detail::sDlBuffer = buf;
  detail::sDlSize = size;
  detail::sDlWritePos = 0;
}

uint32_t end_display_list() {
  detail::sInDisplayList = false;
  const uint32_t bytesWritten = detail::sDlWritePos;
  const uint32_t padded = (bytesWritten + 31) & ~31u;
  while (detail::sDlWritePos < padded && detail::sDlWritePos < detail::sDlSize) {
    detail::sDlBuffer[detail::sDlWritePos++] = 0;
  }
  detail::sDlBuffer = nullptr;
  detail::sDlSize = 0;
  detail::sDlWritePos = 0;
  return padded;
}

bool in_display_list() { return detail::sInDisplayList; }

void drain() {
  if (detail::sBufferSize == 0) {
    return;
  }

  ZoneScoped;
  const uint64_t target = sStreamBase + detail::sBufferSize;

  switch (kProcessingMode) {
  case ProcessingMode::Drain:
  case ProcessingMode::Inline:
    sPublished.store(target, std::memory_order_relaxed);
    process_to(target, std::memory_order_relaxed);
    break;
  case ProcessingMode::Thread: {
    sPublished.store(target, std::memory_order_release);
    wake_worker();

    uint64_t processed = sProcessed.load(std::memory_order_acquire);
    if (processed < target) {
      do {
        sProcessed.wait(processed, std::memory_order_acquire);
        processed = sProcessed.load(std::memory_order_acquire);
      } while (processed < target);
    }
    break;
  }
  }

  {
    std::lock_guard lock{sBufferMutex};
    sStreamBase = target;
    detail::sBufferSize = 0;
  }
  sPendingDraws = 0;
}

const uint8_t* get_buffer_data() { return detail::sBufferData; }
uint32_t get_buffer_size() { return detail::sBufferSize; }

void clear_buffer() {
  const uint64_t processed = sProcessed.load(std::memory_order_acquire);
  AURORA_ASSERT(sPublished.load(std::memory_order_acquire) == processed,
                "fifo::clear_buffer: published commands are still pending");
  std::lock_guard lock{sBufferMutex};
  sStreamBase = processed;
  detail::sBufferSize = 0;
  sPendingDraws = 0;
}

} // namespace aurora::gx::fifo
