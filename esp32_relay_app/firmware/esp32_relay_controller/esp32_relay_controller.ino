#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>

// ================= CONFIGURATION =================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Relay Pins (Channels 1 through 4)
const int RELAY_PINS[4] = {18, 19, 21, 22};

// Relay Logic: Most 5V optocoupled relay modules are ACTIVE LOW.
const bool RELAY_ACTIVE_LOW = true;
const int RELAY_ON  = RELAY_ACTIVE_LOW ? LOW : HIGH;
const int RELAY_OFF = RELAY_ACTIVE_LOW ? HIGH : LOW;

WebServer server(80);

int relayStates[4] = {0, 0, 0, 0};
unsigned long pulseEndTime[4] = {0, 0, 0, 0};
String serialBuffer = "";

// ================= HARDWARE FUNCTIONS =================
void setRelay(int chIndex, int state) {
  if (chIndex < 0 || chIndex >= 4) return;
  relayStates[chIndex] = (state != 0) ? 1 : 0;
  digitalWrite(RELAY_PINS[chIndex], relayStates[chIndex] ? RELAY_ON : RELAY_OFF);
  pulseEndTime[chIndex] = 0;
}

void pulseRelay(int chIndex, unsigned long durationMs) {
  if (chIndex < 0 || chIndex >= 4) return;
  setRelay(chIndex, 1);
  pulseEndTime[chIndex] = millis() + durationMs;
}

void setAllRelays(int s1, int s2, int s3, int s4) {
  setRelay(0, s1);
  setRelay(1, s2);
  setRelay(2, s3);
  setRelay(3, s4);
}

// ================= HTTP REST HANDLERS =================
void sendCorsHeader() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void handleOptions() {
  sendCorsHeader();
  server.send(204);
}

void handleStatus() {
  sendCorsHeader();
  String json = "{";
  json += "\"status\":\"ok\",";
  json += "\"ch1\":" + String(relayStates[0]) + ",";
  json += "\"ch2\":" + String(relayStates[1]) + ",";
  json += "\"ch3\":" + String(relayStates[2]) + ",";
  json += "\"ch4\":" + String(relayStates[3]) + ",";
  json += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
  json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
  json += "\"uptime_ms\":" + String(millis());
  json += "}";
  server.send(200, "application/json", json);
}

void handleRelay() {
  sendCorsHeader();
  if (server.hasArg("ch") && server.hasArg("state")) {
    int ch = server.arg("ch").toInt();
    int state = server.arg("state").toInt();
    if (ch >= 1 && ch <= 4) {
      setRelay(ch - 1, state);
      Serial.printf("[HTTP] CH%d -> %d\n", ch, state);
      handleStatus();
      return;
    }
  }
  server.send(400, "application/json", "{\"error\":\"Invalid ch (1-4) or state (0-1)\"}");
}

void handlePulse() {
  sendCorsHeader();
  if (server.hasArg("ch")) {
    int ch = server.arg("ch").toInt();
    unsigned long ms = server.hasArg("ms") ? server.arg("ms").toInt() : 2000;
    if (ch >= 1 && ch <= 4) {
      pulseRelay(ch - 1, ms);
      Serial.printf("[HTTP] PULSE CH%d for %lu ms\n", ch, ms);
      handleStatus();
      return;
    }
  }
  server.send(400, "application/json", "{\"error\":\"Invalid ch parameter\"}");
}

void handleAll() {
  sendCorsHeader();
  if (server.hasArg("s") && server.arg("s").length() == 4) {
    String s = server.arg("s");
    setAllRelays(s[0] == '1' ? 1 : 0,
                 s[1] == '1' ? 1 : 0,
                 s[2] == '1' ? 1 : 0,
                 s[3] == '1' ? 1 : 0);
    Serial.printf("[HTTP] ALL -> %s\n", s.c_str());
    handleStatus();
    return;
  }
  server.send(400, "application/json", "{\"error\":\"Requires s=xxxx parameter\"}");
}

void handlePing() {
  sendCorsHeader();
  server.send(200, "application/json", "{\"pong\":true}");
}

// ================= SERIAL COMMAND PARSER =================
void processSerialCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) return;

  if (cmd == "PING") {
    Serial.println("PONG");
  } else if (cmd == "STATUS") {
    Serial.printf("STATE:%d%d%d%d\n", relayStates[0], relayStates[1], relayStates[2], relayStates[3]);
  } else if (cmd.startsWith("CH") && cmd.length() >= 5 && cmd.charAt(3) == ':') {
    int ch = cmd.charAt(2) - '0';
    int state = cmd.charAt(4) - '0';
    if (ch >= 1 && ch <= 4) {
      setRelay(ch - 1, state);
      Serial.printf("ACK:CH%d=%d\n", ch, state);
    }
  } else if (cmd.startsWith("ALL:") && cmd.length() == 8) {
    int s1 = cmd.charAt(4) == '1' ? 1 : 0;
    int s2 = cmd.charAt(5) == '1' ? 1 : 0;
    int s3 = cmd.charAt(6) == '1' ? 1 : 0;
    int s4 = cmd.charAt(7) == '1' ? 1 : 0;
    setAllRelays(s1, s2, s3, s4);
    Serial.printf("ACK:ALL=%d%d%d%d\n", s1, s2, s3, s4);
  } else if (cmd.startsWith("PULSE:")) {
    int firstColon = cmd.indexOf(':');
    int secondColon = cmd.indexOf(':', firstColon + 1);
    if (secondColon > firstColon) {
      int ch = cmd.substring(firstColon + 1, secondColon).toInt();
      unsigned long ms = cmd.substring(secondColon + 1).toInt();
      if (ch >= 1 && ch <= 4 && ms > 0) {
        pulseRelay(ch - 1, ms);
        Serial.printf("ACK:PULSE:%d:%lu\n", ch, ms);
      }
    }
  } else {
    Serial.println("ERR:UNKNOWN_COMMAND");
  }
}

// ================= SETUP & LOOP =================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[CognEn-ESP32] Initializing Relay Controller...");

  for (int i = 0; i < 4; i++) {
    pinMode(RELAY_PINS[i], OUTPUT);
    digitalWrite(RELAY_PINS[i], RELAY_OFF);
    relayStates[i] = 0;
  }
  Serial.println("[CognEn-ESP32] Relays initialized (Active-Low SAFE OFF).");

  if (String(WIFI_SSID) != "YOUR_WIFI_SSID") {
    Serial.printf("[CognEn-ESP32] Connecting to Wi-Fi: %s\n", WIFI_SSID);
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
      delay(500);
      Serial.print(".");
      attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n[CognEn-ESP32] Wi-Fi Connected!");
      Serial.print("[CognEn-ESP32] IP Address: ");
      Serial.println(WiFi.localIP());

      server.on("/status", HTTP_GET, handleStatus);
      server.on("/relay", HTTP_GET, handleRelay);
      server.on("/pulse", HTTP_GET, handlePulse);
      server.on("/all", HTTP_GET, handleAll);
      server.on("/ping", HTTP_GET, handlePing);
      server.onNotFound(handleOptions);

      server.begin();
      Serial.println("[CognEn-ESP32] HTTP REST Server active on port 80.");
    } else {
      Serial.println("\n[CognEn-ESP32] Wi-Fi timed out. Continuing in Wired Serial mode.");
    }
  } else {
    Serial.println("[CognEn-ESP32] No Wi-Fi SSID configured. Running exclusively in Wired Serial mode.");
  }

  Serial.println("[CognEn-ESP32] Ready for commands. Ready baud: 115200.\n");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    server.handleClient();
  }

  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        processSerialCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
    }
  }

  unsigned long now = millis();
  for (int i = 0; i < 4; i++) {
    if (pulseEndTime[i] > 0 && now >= pulseEndTime[i]) {
      setRelay(i, 0);
      Serial.printf("[AUTO] PULSE EXPIRED: CH%d -> OFF\n", i + 1);
    }
  }
}
