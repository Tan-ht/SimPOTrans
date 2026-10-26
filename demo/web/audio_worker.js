let socket = null;
let playerPort = null;
let targetSampleRate = 48000;

function postMetric(metric) {
  self.postMessage({ type: "metric", metric: {
    ...metric,
    worker_perf_now_ms: performance.now(),
  } });
}

function closeSocket() {
  if (socket) {
    socket.close();
    socket = null;
  }
}

function clearPlayer() {
  if (playerPort) {
    playerPort.postMessage({ type: "clear" });
  }
}

function readString(view, offset, length) {
  let text = "";
  for (let index = 0; index < length; index += 1) {
    text += String.fromCharCode(view.getUint8(offset + index));
  }
  return text;
}

function parseWav(buffer) {
  const view = new DataView(buffer);
  if (readString(view, 0, 4) !== "RIFF" || readString(view, 8, 4) !== "WAVE") {
    throw new Error("Unsupported WAV container");
  }

  let offset = 12;
  let format = null;
  let dataOffset = -1;
  let dataSize = 0;

  while (offset + 8 <= view.byteLength) {
    const chunkId = readString(view, offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataOffset = offset + 8;

    if (chunkId === "fmt ") {
      format = {
        audioFormat: view.getUint16(chunkDataOffset, true),
        channels: view.getUint16(chunkDataOffset + 2, true),
        sampleRate: view.getUint32(chunkDataOffset + 4, true),
        bitsPerSample: view.getUint16(chunkDataOffset + 14, true),
      };
    } else if (chunkId === "data") {
      dataOffset = chunkDataOffset;
      dataSize = chunkSize;
    }

    offset = chunkDataOffset + chunkSize + (chunkSize % 2);
  }

  if (!format || dataOffset < 0) {
    throw new Error("Invalid WAV data");
  }

  return { ...format, dataOffset, dataSize, buffer };
}

function wavToMonoFloat32(wav) {
  const view = new DataView(wav.buffer, wav.dataOffset, wav.dataSize);
  const bytesPerSample = wav.bitsPerSample / 8;
  const frameCount = Math.floor(wav.dataSize / (bytesPerSample * wav.channels));
  const pcm = new Float32Array(frameCount);

  for (let frame = 0; frame < frameCount; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < wav.channels; channel += 1) {
      const offset = (frame * wav.channels + channel) * bytesPerSample;
      let sample = 0;
      if (wav.audioFormat === 1 && wav.bitsPerSample === 16) {
        sample = view.getInt16(offset, true) / 32768;
      } else if (wav.audioFormat === 1 && wav.bitsPerSample === 24) {
        const b0 = view.getUint8(offset);
        const b1 = view.getUint8(offset + 1);
        const b2 = view.getUint8(offset + 2);
        let value = b0 | (b1 << 8) | (b2 << 16);
        if (value & 0x800000) value |= 0xff000000;
        sample = value / 8388608;
      } else if (wav.audioFormat === 1 && wav.bitsPerSample === 32) {
        sample = view.getInt32(offset, true) / 2147483648;
      } else if (wav.audioFormat === 3 && wav.bitsPerSample === 32) {
        sample = view.getFloat32(offset, true);
      } else {
        throw new Error(`Unsupported WAV format ${wav.audioFormat}/${wav.bitsPerSample}`);
      }
      sum += sample;
    }
    pcm[frame] = Math.max(-1, Math.min(1, sum / wav.channels));
  }
  return pcm;
}

function resampleLinear(input, sourceRate, targetRate) {
  if (!input.length || sourceRate === targetRate) return input;
  const outputLength = Math.max(1, Math.round(input.length * targetRate / sourceRate));
  const output = new Float32Array(outputLength);
  const ratio = sourceRate / targetRate;

  for (let index = 0; index < outputLength; index += 1) {
    const sourceIndex = index * ratio;
    const left = Math.floor(sourceIndex);
    const right = Math.min(left + 1, input.length - 1);
    const fraction = sourceIndex - left;
    output[index] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

function decodeWavToPcm(buffer) {
  const wav = parseWav(buffer);
  const pcm = wavToMonoFloat32(wav);
  return {
    pcm: resampleLinear(pcm, wav.sampleRate, targetSampleRate || wav.sampleRate),
    sourceSampleRate: wav.sampleRate,
    sourceSamples: pcm.length,
    metric: { legacy_wav: true },
  };
}

function parseTtsAudioFrame(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 4) return null;
  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);
  if (headerLength <= 0 || headerLength > 65536 || 4 + headerLength > buffer.byteLength) {
    return null;
  }

  try {
    const headerBytes = new Uint8Array(buffer, 4, headerLength);
    const headerText = new TextDecoder("utf-8", { fatal: false }).decode(headerBytes);
    const header = JSON.parse(headerText);
    if (
      header.type !== "tts_audio_frame" ||
      header.version !== 1 ||
      header.encoding !== "pcm_s16le" ||
      header.channels !== 1
    ) {
      return null;
    }
    return {
      header,
      payloadOffset: 4 + headerLength,
      payloadBytes: buffer.byteLength - 4 - headerLength,
    };
  } catch (_error) {
    return null;
  }
}

function pcm16PayloadToFloat32(buffer, offset, byteLength) {
  const sampleCount = Math.floor(byteLength / 2);
  const view = new DataView(buffer, offset, sampleCount * 2);
  const pcm = new Float32Array(sampleCount);
  for (let index = 0; index < sampleCount; index += 1) {
    pcm[index] = Math.max(-1, Math.min(1, view.getInt16(index * 2, true) / 32768));
  }
  return pcm;
}

function frameMetricFields(frame, sourceSamples) {
  const sampleRate = Number(frame.header.sample_rate) || targetSampleRate;
  return {
    tts_audio_sequence_id: frame.header.sequence_id,
    tts_audio_frame_index: frame.header.frame_index,
    tts_audio_frame_ms: sampleRate > 0 ? sourceSamples * 1000 / sampleRate : frame.header.frame_ms,
    tts_audio_is_final: Boolean(frame.header.is_final),
    source_sample_rate: sampleRate,
  };
}

function decodeAudioToPcm(buffer) {
  const frame = parseTtsAudioFrame(buffer);
  if (frame) {
    const sourceSampleRate = Number(frame.header.sample_rate) || targetSampleRate;
    const pcm = pcm16PayloadToFloat32(buffer, frame.payloadOffset, frame.payloadBytes);
    return {
      pcm: resampleLinear(pcm, sourceSampleRate, targetSampleRate || sourceSampleRate),
      sourceSampleRate,
      sourceSamples: pcm.length,
      metric: frameMetricFields(frame, pcm.length),
    };
  }
  return decodeWavToPcm(buffer);
}

function sendPcmToPlayer(buffer) {
  if (!playerPort) return;
  const start = performance.now();
  const inputBytes = buffer.byteLength || 0;
  const decoded = decodeAudioToPcm(buffer);
  const decodeMs = performance.now() - start;
  postMetric({
    event: "tts_audio_pcm_queued",
    input_bytes: inputBytes,
    pcm_samples: decoded.pcm.length,
    target_sample_rate: targetSampleRate,
    decode_ms: decodeMs,
    ...decoded.metric,
  });
  playerPort.postMessage({ type: "pcm", pcm: decoded.pcm }, [decoded.pcm.buffer]);
}

function connectAudioSocket(url, sessionId) {
  closeSocket();
  socket = new WebSocket(url);
  socket.binaryType = "arraybuffer";
  socket.onopen = () => {
    socket.send(JSON.stringify({ tts_session_id: sessionId }));
    postMetric({ event: "tts_audio_ws_connected" });
    self.postMessage({ type: "connected" });
  };
  socket.onerror = () => {
    self.postMessage({ type: "error", error: "TTS audio websocket error" });
  };
  socket.onclose = () => {
    socket = null;
    postMetric({ event: "tts_audio_ws_closed" });
    self.postMessage({ type: "closed" });
  };
  socket.onmessage = (event) => {
    try {
      if (event.data instanceof ArrayBuffer) {
        const frame = parseTtsAudioFrame(event.data);
        let metric = { legacy_wav: true };
        if (frame) {
          const sampleRate = Number(frame.header.sample_rate) || targetSampleRate;
          metric = frameMetricFields(frame, Math.floor(frame.payloadBytes / 2));
          metric.tts_audio_frame_ms = sampleRate > 0
            ? (Math.floor(frame.payloadBytes / 2) * 1000 / sampleRate)
            : frame.header.frame_ms;
          metric.legacy_wav = false;
        }
        postMetric({
          event: "tts_audio_frame_received",
          audio_bytes: event.data.byteLength,
          ...metric,
        });
        sendPcmToPlayer(event.data);
      }
    } catch (error) {
      self.postMessage({ type: "error", error: String(error) });
    }
  };
}

self.onmessage = (event) => {
  const message = event.data || {};
  try {
    if (message.type === "connect") {
      playerPort = message.playerPort;
      targetSampleRate = message.targetSampleRate || targetSampleRate;
      clearPlayer();
      connectAudioSocket(message.url, message.sessionId);
    } else if (message.type === "reset") {
      closeSocket();
      clearPlayer();
    }
  } catch (error) {
    self.postMessage({ type: "error", error: String(error) });
  }
};
