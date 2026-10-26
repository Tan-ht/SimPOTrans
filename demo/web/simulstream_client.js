function float32ToInt16(float32) {
  const int16 = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, float32[i]));
    int16[i] = sample < 0 ? sample * 32768 : sample * 32767;
  }
  return int16;
}

export class SimulStreamClient {
  constructor(url, options = {}) {
    this.url = url;
    this.socket = null;
    this.audioContext = null;
    this.streamSource = null;
    this.processorNode = null;
    this.stream = null;
    this.onopen = null;
    this.onerror = null;
    this.onclose = null;
    this.onservermessage = null;
    this.metricQueue = [];
    this.abortRequested = false;
    this.pingInterval = null;
    this.pingSequence = 0;
    this.audioSendStats = null;
    this.batchAudio100ms = Boolean(options.batchAudio100ms);
    this.audioBatch = [];
    this.audioBatchSamples = 0;
    this.audioBatchTargetSamples = 0;
  }

  async start(
    tgtLang,
    srcLang,
    enableTts = false,
    ttsSpeaker = "default",
    ttsSessionId = "",
    beforeConnect = null,
  ) {
    this.abortRequested = false;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const [audioTrack] = this.stream.getAudioTracks();
    const audioTrackSettings = audioTrack ? audioTrack.getSettings() : null;
    if (audioTrack) {
      console.info("Microphone settings:", audioTrackSettings);
    }
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.audioBatchTargetSamples = this.batchAudio100ms
      ? Math.max(1, Math.round(this.audioContext.sampleRate * 0.1))
      : 0;
    this.resetAudioBatch();
    this.resetAudioSendStats();
    this.sendFrontendMetric({
      event: "audio_capture_start",
      audio_context_sample_rate: this.audioContext.sampleRate,
      audio_context_base_latency: this.audioContext.baseLatency ?? null,
      audio_context_output_latency: this.audioContext.outputLatency ?? null,
      audio_context_state: this.audioContext.state,
      batch_audio_100ms: this.batchAudio100ms,
      audio_batch_target_samples: this.audioBatchTargetSamples || null,
      track_sample_rate: audioTrackSettings?.sampleRate ?? null,
      track_channel_count: audioTrackSettings?.channelCount ?? null,
      track_latency: audioTrackSettings?.latency ?? null,
      track_device_id: audioTrackSettings?.deviceId ?? null,
      document_hidden: document.hidden,
      visibility_state: document.visibilityState,
    });
    this.streamSource = this.audioContext.createMediaStreamSource(this.stream);
    this.socket = new WebSocket(this.url);

    this.socket.onerror = (event) => {
      if (this.onerror) this.onerror(event);
    };
    this.socket.onclose = (event) => {
      this.stopNetworkPings();
      if (this.onclose) this.onclose(event);
    };
    this.socket.onopen = () => {
      if (this.onopen) this.onopen();
      this.socket.send(JSON.stringify({
        target_lang: tgtLang,
        source_lang: srcLang,
        sample_rate: this.audioContext.sampleRate,
        enable_tts: Boolean(enableTts),
        tts_speaker: ttsSpeaker || "default",
        tts_session_id: ttsSessionId || "",
      }));
      this.flushFrontendMetrics();
      this.startNetworkPings();
    };
    this.socket.onmessage = (event) => {
      if (typeof event.data !== "string") return;
      const browserReceiveWallTimeMs = Date.now();
      const browserReceivePerfMs = performance.now();
      const serverMessage = JSON.parse(event.data);
      if (serverMessage.client_pong) {
        this.reportNetworkPongMetric(serverMessage.client_pong, browserReceiveWallTimeMs, browserReceivePerfMs);
        return;
      }
      serverMessage.browser_receive_wall_time_ms = browserReceiveWallTimeMs;
      serverMessage.browser_receive_perf_ms = browserReceivePerfMs;
      this.reportTextReceiveMetric(serverMessage, browserReceiveWallTimeMs, browserReceivePerfMs);
      if (this.onservermessage) {
        this.onservermessage(serverMessage);
      }
    };

    await this.audioContext.audioWorklet.addModule("processor.js");
    if (beforeConnect) {
      beforeConnect(this.audioContext);
    }
    this.processorNode = new AudioWorkletNode(this.audioContext, "pcm-processor");
    this.streamSource.connect(this.processorNode);

    this.processorNode.port.onmessage = (event) => {
      const data = event.data;
      if (data && data.type === "worklet_stats") {
        this.sendFrontendMetric({
          event: "audio_worklet_stats",
          ...data,
          audio_context_current_time: this.audioContext?.currentTime ?? null,
          document_hidden: document.hidden,
          visibility_state: document.visibilityState,
        });
        return;
      }
      if (data && data.length > 0) {
        this.sendAudioFrame(data);
      }
    };
    this.sendFrontendMetric({ event: "microphone_audio_started" });
  }

  resetAudioSendStats() {
    const now = performance.now();
    this.audioSendStats = {
      startPerfMs: now,
      lastReportPerfMs: now,
      totalSamples: 0,
      totalBytes: 0,
      totalSendCalls: 0,
      intervalSamples: 0,
      intervalBytes: 0,
      intervalSendCalls: 0,
      intervalMaxFrameSamples: 0,
      intervalMinFrameSamples: null,
      intervalMaxBufferedAmount: 0,
      intervalMaxSendMs: 0,
      droppedFrames: 0,
    };
  }

  sendAudioFrame(float32Frame) {
    if (!this.audioSendStats) {
      this.resetAudioSendStats();
    }
    const stats = this.audioSendStats;
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      stats.droppedFrames += 1;
      return;
    }

    if (this.batchAudio100ms) {
      this.enqueueAudioFrame(float32Frame);
      return;
    }

    this.sendAudioPacket(float32Frame);
  }

  resetAudioBatch() {
    this.audioBatch = [];
    this.audioBatchSamples = 0;
  }

  enqueueAudioFrame(float32Frame) {
    this.audioBatch.push(float32Frame);
    this.audioBatchSamples += float32Frame.length;
    if (this.audioBatchSamples >= this.audioBatchTargetSamples) {
      this.flushAudioBatch();
    }
  }

  flushAudioBatch() {
    if (!this.audioBatchSamples) return;
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.resetAudioBatch();
      return;
    }
    const packet = new Float32Array(this.audioBatchSamples);
    let offset = 0;
    for (const frame of this.audioBatch) {
      packet.set(frame, offset);
      offset += frame.length;
    }
    this.resetAudioBatch();
    this.sendAudioPacket(packet);
  }

  sendAudioPacket(float32Frame) {
    const stats = this.audioSendStats;
    const frameSamples = float32Frame.length;
    const beforeBufferedAmount = this.socket.bufferedAmount || 0;
    const start = performance.now();
    const int16Frame = float32ToInt16(float32Frame);
    this.socket.send(int16Frame.buffer);
    const sendMs = performance.now() - start;
    const afterBufferedAmount = this.socket.bufferedAmount || 0;

    stats.totalSamples += frameSamples;
    stats.totalBytes += int16Frame.byteLength;
    stats.totalSendCalls += 1;
    stats.intervalSamples += frameSamples;
    stats.intervalBytes += int16Frame.byteLength;
    stats.intervalSendCalls += 1;
    stats.intervalMaxFrameSamples = Math.max(stats.intervalMaxFrameSamples, frameSamples);
    stats.intervalMinFrameSamples = stats.intervalMinFrameSamples == null
      ? frameSamples
      : Math.min(stats.intervalMinFrameSamples, frameSamples);
    stats.intervalMaxBufferedAmount = Math.max(
      stats.intervalMaxBufferedAmount,
      beforeBufferedAmount,
      afterBufferedAmount,
    );
    stats.intervalMaxSendMs = Math.max(stats.intervalMaxSendMs, sendMs);
    this.maybeReportAudioSendStats();
  }

  maybeReportAudioSendStats() {
    const stats = this.audioSendStats;
    if (!stats || !this.audioContext) return;
    const now = performance.now();
    const intervalMs = now - stats.lastReportPerfMs;
    if (intervalMs < 1000) return;

    const elapsedSeconds = (now - stats.startPerfMs) / 1000;
    const intervalSeconds = intervalMs / 1000;
    const sampleRate = this.audioContext.sampleRate;
    const totalAudioSeconds = stats.totalSamples / sampleRate;
    const intervalAudioSeconds = stats.intervalSamples / sampleRate;
    this.sendFrontendMetric({
      event: "audio_send_pace",
      audio_context_sample_rate: sampleRate,
      audio_context_current_time: this.audioContext.currentTime,
      audio_context_state: this.audioContext.state,
      elapsed_seconds: elapsedSeconds,
      total_sent_audio_seconds: totalAudioSeconds,
      sent_minus_elapsed: totalAudioSeconds - elapsedSeconds,
      interval_seconds: intervalSeconds,
      interval_sent_audio_seconds: intervalAudioSeconds,
      interval_send_rate: intervalSeconds ? intervalAudioSeconds / intervalSeconds : null,
      total_send_calls: stats.totalSendCalls,
      interval_send_calls: stats.intervalSendCalls,
      interval_bytes: stats.intervalBytes,
      interval_min_frame_samples: stats.intervalMinFrameSamples,
      interval_max_frame_samples: stats.intervalMaxFrameSamples,
      batch_audio_100ms: this.batchAudio100ms,
      audio_batch_target_samples: this.audioBatchTargetSamples || null,
      pending_batch_samples: this.audioBatchSamples,
      socket_buffered_amount: this.socket?.bufferedAmount ?? null,
      interval_max_buffered_amount: stats.intervalMaxBufferedAmount,
      interval_max_send_ms: stats.intervalMaxSendMs,
      dropped_audio_frames: stats.droppedFrames,
      document_hidden: document.hidden,
      visibility_state: document.visibilityState,
    });

    stats.lastReportPerfMs = now;
    stats.intervalSamples = 0;
    stats.intervalBytes = 0;
    stats.intervalSendCalls = 0;
    stats.intervalMaxFrameSamples = 0;
    stats.intervalMinFrameSamples = null;
    stats.intervalMaxBufferedAmount = 0;
    stats.intervalMaxSendMs = 0;
  }

  reportTextReceiveMetric(serverMessage, browserReceiveWallTimeMs, browserReceivePerfMs) {
    const added = serverMessage.new || "";
    const deleted = serverMessage.deleted || "";
    if (!added && !deleted && !serverMessage.segment_end) {
      return;
    }
    const serverSendWallTimeMs = Number(serverMessage.server_send_wall_time_ms || 0);
    this.sendFrontendMetric({
      event: "text_message_received",
      text_message_id: serverMessage.text_message_id || null,
      new_chars: added.length,
      deleted_chars: deleted.length,
      segment_end: Boolean(serverMessage.segment_end),
      processed_audio_seconds: serverMessage.processed_audio_seconds ?? null,
      server_send_wall_time_ms: serverSendWallTimeMs || null,
      server_send_perf_ms: serverMessage.server_send_perf_ms || null,
      browser_receive_wall_time_ms: browserReceiveWallTimeMs,
      browser_receive_perf_ms: browserReceivePerfMs,
      server_to_browser_wall_ms: serverSendWallTimeMs ? browserReceiveWallTimeMs - serverSendWallTimeMs : null,
    });
  }

  startNetworkPings() {
    this.stopNetworkPings();
    this.sendNetworkPing();
    this.pingInterval = window.setInterval(() => this.sendNetworkPing(), 2000);
  }

  stopNetworkPings() {
    if (this.pingInterval) {
      window.clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  sendNetworkPing() {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
    this.pingSequence += 1;
    this.socket.send(JSON.stringify({
      client_ping: {
        ping_id: `${Date.now()}-${this.pingSequence}`,
        client_send_wall_time_ms: Date.now(),
        client_send_perf_ms: performance.now(),
      },
    }));
  }

  reportNetworkPongMetric(pong, browserReceiveWallTimeMs, browserReceivePerfMs) {
    const clientSendPerfMs = Number(pong.client_send_perf_ms || 0);
    const clientSendWallTimeMs = Number(pong.client_send_wall_time_ms || 0);
    const rttMs = clientSendPerfMs ? browserReceivePerfMs - clientSendPerfMs : null;
    const estimatedOneWayMs = rttMs == null ? null : rttMs / 2;
    const serverReceiveWallTimeMs = Number(pong.server_receive_wall_time_ms || 0);
    const clockOffsetMs = clientSendWallTimeMs && serverReceiveWallTimeMs && rttMs != null
      ? serverReceiveWallTimeMs - (clientSendWallTimeMs + estimatedOneWayMs)
      : null;
    this.sendFrontendMetric({
      event: "network_rtt",
      ping_id: pong.ping_id || null,
      client_send_wall_time_ms: clientSendWallTimeMs || null,
      client_send_perf_ms: clientSendPerfMs || null,
      browser_receive_wall_time_ms: browserReceiveWallTimeMs,
      browser_receive_perf_ms: browserReceivePerfMs,
      server_receive_wall_time_ms: pong.server_receive_wall_time_ms || null,
      server_send_wall_time_ms: pong.server_send_wall_time_ms || null,
      rtt_ms: rttMs,
      estimated_one_way_ms: estimatedOneWayMs,
      estimated_client_server_clock_offset_ms: clockOffsetMs,
    });
  }

  sendFrontendMetric(metric) {
    const payload = {
      ...metric,
      perf_now_ms: performance.now(),
      wall_time_ms: Date.now(),
    };
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ frontend_metrics: payload }));
    } else {
      this.metricQueue.push(payload);
    }
  }

  flushFrontendMetrics() {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN || this.metricQueue.length === 0) {
      return;
    }
    const metrics = this.metricQueue.splice(0, this.metricQueue.length);
    this.socket.send(JSON.stringify({ frontend_metrics: metrics }));
  }

  async stop() {
    this.abortRequested = true;
    this.stopNetworkPings();
    this.flushAudioBatch();
    this.maybeReportAudioSendStats();
    this.disconnectAudio();
    this.sendFrontendMetric({ event: "abort_stream_requested" });
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ abort_stream: true }));
    }
    if (this.audioContext) {
      await this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
  }

  disconnectAudio() {
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    if (this.streamSource) {
      this.streamSource.disconnect();
      this.streamSource = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
  }

  shutdown() {
    this.abortRequested = true;
    this.stopNetworkPings();
    this.flushAudioBatch();
    this.maybeReportAudioSendStats();
    this.disconnectAudio();
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ abort_stream: true }));
      this.socket.close(1001, "Page closed");
    } else if (this.socket) {
      this.socket.close();
    }
    this.socket = null;
    if (this.audioContext) {
      this.audioContext.close().catch(() => {});
      this.audioContext = null;
    }
  }

  clearContext() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ clear_context: true }));
    }
  }

  closeSocket() {
    this.stopNetworkPings();
    this.flushAudioBatch();
    if (this.socket) {
      if (this.abortRequested && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify({ abort_stream: true }));
      }
      this.socket.close();
      this.socket = null;
    }
  }

  socketOpen() {
    return this.socket !== null && this.socket.readyState === WebSocket.OPEN;
  }
}
