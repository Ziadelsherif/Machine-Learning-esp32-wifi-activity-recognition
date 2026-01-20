/*
 * ESP32 Transmitter (Tx) for HAR Project
 * 
 * ROLE: Access Point (AP)
 * This device creates a dedicated Wi-Fi network that the Receiver will listen to.
 * This ensures a stable, high-frequency signal source independent of your home router.
 */

#include <WiFi.h>

// Configuration
const char* ssid     = "HAR_PROJECT_TX";  // The name of the network we are creating
const char* password = "password123";     // Detailed password not needed for this exp

void setup() {
  Serial.begin(115200);
  
  // Set Wi-Fi to Access Point Mode
  Serial.println("Setting up Access Point...");
  WiFi.softAP(ssid, password);
  
  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);
  
  Serial.println("Transmitter Ready. Broadcasting beacon frames...");
}

void loop() {
  // The ESP32 automatically broadcasts "Beacon Frames" ~10 times per second
  // just by being an Access Point. We don't need to manually send packets 
  // for RSSI sensing to work. The Receiver just measures the strength 
  // of this "Hidden" background signal.
  
  // Optional: distinct blink to show it's alive
  delay(1000);
}
