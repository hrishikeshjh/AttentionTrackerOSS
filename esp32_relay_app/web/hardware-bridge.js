/**
 * CognEn Hardware Bridge
 * Multi-Transport Dispatcher:
 * 1. Native Python Server Bridge (/api/all?s=xxxx) - 100% bypasses Snap/Sandbox
 * 2. Wired Web Serial API (Direct in Chrome/Edge)
 * 3. Wireless (OTA) Wi-Fi HTTP REST API
 * 4. Loopback / Simulation Mode
 */

export class HardwareBridge {
  constructor(onLogCallback) {
    this.log = onLogCallback || console.log;

    // Native Python Server Bridge
    this.isBridgeActive = false;
    this.bridgePort = "";

    // Wired (Web Serial)
    this.serialPort = null;
    this.serialWriter = null;
    this.isSerialConnected = false;

    // Wireless (Wi-Fi)
    this.esp32Ip = localStorage.getItem("cognen_esp32_ip") || "192.168.0.109";
    this.isWifiActive = false;

    // Dispatch Mode: "bridge", "serial", "wifi", "simulation"
    this.transportMode = "simulation";

    // Current hardware relay cache [ch1, ch2, ch3, ch4]
    this.currentStates = [0, 0, 0, 0];
    this.lastSentPayload = "";

    // Auto-probe Native Bridge on startup
    this.probeNativeBridge();
  }

  async probeNativeBridge() {
    try {
      const resp = await fetch("/api/status", { cache: "no-store" });
      if (resp.ok) {
        const data = await resp.json();
        if (data.connected) {
          this.isBridgeActive = true;
          this.bridgePort = data.port;
          this.transportMode = "bridge";
          this.log(`[HARDWARE] Native Bridge active on ${data.port}. Direct hardware control ready!`, "success");
          return true;
        } else if (data.error) {
          this.log(`[HARDWARE ERROR] ${data.error}`, "error");
        } else {
          this.log("[HARDWARE] Server bridge is running, but ESP32 is not detected on USB. Plug in ESP32 via USB.", "error");
        }
      }
    } catch {
      // Running plain static server without API
    }
    return false;
  }

  /**
   * Connect to ESP32 via Web Serial (USB)
   */
  async connectSerial() {
    if (!("serial" in navigator)) {
      this.log("Web Serial API not supported in this browser.", "error");
      return false;
    }

    try {
      this.serialPort = await navigator.serial.requestPort();
      await this.serialPort.open({ baudRate: 115200 });

      const textEncoder = new TextEncoderStream();
      textEncoder.readable.pipeTo(this.serialPort.writable);
      this.serialWriter = textEncoder.writable.getWriter();

      this.isSerialConnected = true;
      this.transportMode = "serial";
      this.log("Connected to ESP32 via Web Serial (115200 baud).", "success");
      await this.writeSerial("PING\n");
      return true;
    } catch (err) {
      this.log(`Serial connection failed: ${err.message}`, "error");
      this.isSerialConnected = false;
      return false;
    }
  }

  async disconnectSerial() {
    try {
      if (this.serialWriter) {
        await this.serialWriter.close();
        this.serialWriter = null;
      }
      if (this.serialPort) {
        await this.serialPort.close();
        this.serialPort = null;
      }
      this.isSerialConnected = false;
      this.transportMode = this.isBridgeActive ? "bridge" : "simulation";
      this.log("Web Serial disconnected.", "info");
    } catch (err) {
      this.log(`Disconnect error: ${err.message}`, "error");
    }
  }

  async writeSerial(text) {
    if (!this.serialWriter) return;
    try {
      await this.serialWriter.write(text);
    } catch (err) {
      this.log(`Serial write error: ${err.message}`, "error");
    }
  }

  /**
   * Configure and Ping Wi-Fi Endpoint
   */
  async configureWifi(ipAddress) {
    this.esp32Ip = ipAddress.replace(/^https?:\/\//, "").replace(/\/.*$/, "").trim();
    localStorage.setItem("cognen_esp32_ip", this.esp32Ip);

    this.log(`Pinging ESP32 at http://${this.esp32Ip}/ping ...`, "info");
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2500);

      const resp = await fetch(`http://${this.esp32Ip}/ping`, {
        method: "GET",
        mode: "cors",
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (resp.ok) {
        this.isWifiActive = true;
        this.transportMode = "wifi";
        this.log(`ESP32 Wi-Fi connected at ${this.esp32Ip}`, "success");
        return true;
      }
    } catch (err) {
      this.log(`Wi-Fi connection failed: ${err.message}`, "error");
    }
    this.isWifiActive = false;
    return false;
  }

  /**
   * Dispatch 4-Channel States [s1, s2, s3, s4] (e.g. [1, 0, 0, 1])
   */
  async dispatchRelays(states) {
    const payload = states.map(s => (s ? "1" : "0")).join("");

    if (payload === this.lastSentPayload) {
      return;
    }
    this.lastSentPayload = payload;
    this.currentStates = [...states];

    if (this.transportMode === "bridge" || (this.isBridgeActive && !this.isSerialConnected && !this.isWifiActive)) {
      // Dispatch via Native Python Bridge
      fetch(`/api/all?s=${payload}`, { cache: "no-store" }).catch(() => {});
      this.log(`[Bridge TX] ALL:${payload} -> ${this.bridgePort}`, "tx");
    } else if (this.transportMode === "serial" && this.isSerialConnected) {
      await this.writeSerial(`ALL:${payload}\n`);
      this.log(`[Wired TX] ALL:${payload}`, "tx");
    } else if (this.transportMode === "wifi" && this.isWifiActive) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 1200);
        fetch(`http://${this.esp32Ip}/all?s=${payload}`, {
          method: "GET",
          mode: "cors",
          signal: controller.signal
        }).catch(() => {});
        clearTimeout(timeoutId);
        this.log(`[Wi-Fi TX] /all?s=${payload}`, "tx");
      } catch (e) {
        this.log(`Wi-Fi dispatch error: ${e.message}`, "error");
      }
    } else {
      // Simulation mode
      this.log(`[Simulated] Relays -> ${payload}`, "sim");
    }
  }

  /**
   * Manual Single Channel Toggle
   */
  async setChannelManual(chIndex, state) {
    const newStates = [...this.currentStates];
    newStates[chIndex] = state ? 1 : 0;
    await this.dispatchRelays(newStates);
  }
}
