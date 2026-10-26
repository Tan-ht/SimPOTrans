class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.totalFrames = 0;
    this.intervalFrames = 0;
    this.intervalCallbacks = 0;
    this.intervalEmptyInputs = 0;
    this.intervalMinFrames = null;
    this.intervalMaxFrames = 0;
    this.intervalMaxChannels = 0;
    this.lastReportFrame = 0;
    this.reportIntervalFrames = sampleRate;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0] || input[0].length === 0) {
      this.intervalEmptyInputs += 1;
      this._maybeReportStats();
      return true;
    }

    const frameCount = input[0].length;
    this._recordInput(frameCount, input.length);

    if (input.length >= 2) {
      const left = input[0];
      const right = input[1];
      const mono = new Float32Array(left.length);
      for (let i = 0; i < left.length; i += 1) {
        mono[i] = 0.5 * (left[i] + right[i]);
      }
      this.port.postMessage(mono);
    } else {
      this.port.postMessage(input[0]);
    }
    this._maybeReportStats();
    return true;
  }

  _recordInput(frameCount, channelCount) {
    this.totalFrames += frameCount;
    this.intervalFrames += frameCount;
    this.intervalCallbacks += 1;
    this.intervalMinFrames = this.intervalMinFrames == null
      ? frameCount
      : Math.min(this.intervalMinFrames, frameCount);
    this.intervalMaxFrames = Math.max(this.intervalMaxFrames, frameCount);
    this.intervalMaxChannels = Math.max(this.intervalMaxChannels, channelCount);
  }

  _maybeReportStats() {
    if (this.totalFrames - this.lastReportFrame < this.reportIntervalFrames) {
      return;
    }
    this.port.postMessage({
      type: "worklet_stats",
      sample_rate: sampleRate,
      current_frame: currentFrame,
      current_time: currentTime,
      total_frames: this.totalFrames,
      total_audio_seconds: this.totalFrames / sampleRate,
      interval_frames: this.intervalFrames,
      interval_audio_seconds: this.intervalFrames / sampleRate,
      interval_callbacks: this.intervalCallbacks,
      interval_empty_inputs: this.intervalEmptyInputs,
      interval_min_frame_samples: this.intervalMinFrames,
      interval_max_frame_samples: this.intervalMaxFrames,
      interval_max_channels: this.intervalMaxChannels,
    });
    this.lastReportFrame = this.totalFrames;
    this.intervalFrames = 0;
    this.intervalCallbacks = 0;
    this.intervalEmptyInputs = 0;
    this.intervalMinFrames = null;
    this.intervalMaxFrames = 0;
    this.intervalMaxChannels = 0;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
