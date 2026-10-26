class TTSPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.current = null;
    this.offset = 0;
    this.inputPort = null;

    this.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === "bind" && message.port) {
        this.inputPort = message.port;
        this.inputPort.onmessage = (portEvent) => this.handleMessage(portEvent.data || {});
        this.inputPort.start();
      } else {
        this.handleMessage(message);
      }
    };
  }

  handleMessage(message) {
    if (message.type === "pcm" && message.pcm) {
      this.queue.push(message.pcm);
    } else if (message.type === "clear") {
      this.queue = [];
      this.current = null;
      this.offset = 0;
    }
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const channel = output[0];
    let index = 0;
    while (index < channel.length) {
      if (!this.current || this.offset >= this.current.length) {
        this.current = this.queue.shift() || null;
        this.offset = 0;
        if (!this.current) break;
      }

      const available = this.current.length - this.offset;
      const needed = channel.length - index;
      const count = Math.min(available, needed);
      channel.set(this.current.subarray(this.offset, this.offset + count), index);
      this.offset += count;
      index += count;
    }

    for (; index < channel.length; index += 1) {
      channel[index] = 0;
    }
    for (let channelIndex = 1; channelIndex < output.length; channelIndex += 1) {
      output[channelIndex].set(channel);
    }
    return true;
  }
}

registerProcessor("tts-player", TTSPlayerProcessor);
