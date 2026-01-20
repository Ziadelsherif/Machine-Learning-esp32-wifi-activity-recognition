/*
 * ESP32 Wi-Fi RSSI Logger for Human Activity Recognition (HAR)
 * 
 * Instructions:
 * 1. Modify 'ssid' and 'password' to match your Router/Hotspot.
 * 2. Upload to ESP32.
 * 3. Open Serial Monitor (Baud Rate 115200).
 * 4. Data format is: Timestamp(ms), RSSI(dBm)
 * 5. Copy the output to a text file -> save as CSV.
 */

#include <WiFi.h>

// =======================
// CONFIGURATION
// =======================
// Replace with your Router's credentials
const char* ssid     = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

// Sampling frequency (Delay in ms)
// 20ms = 50Hz (Ideal for HAR)
const int SAMPLING_DELAY_MS = 20; 

void setup() {
  // 1. Initialize Serial Communication
  Serial.begin(115200);
  delay(1000);

  // 2. Initialize Wi-Fi in Station Mode
  Serial.println("\n[Info] Initializing ESP32 Station Mode...");
  WiFi.mode(WIFI_STA);
  
  // Disable power saving to ensure maximum performance/responsiveness
  WiFi.setSleep(false);

  // 3. Connect to Access Point
  Serial.printf("[Info] Connecting to %s ", ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("[Info] Connected!");
  Serial.printf("[Info] IP Address: %s\n", WiFi.localIP().toString().c_str());
  
  // Header for CSV
  Serial.println("Timestamp,RSSI,Activity"); 
  Serial.println("START_LOGGING_NOW");
}

void loop() {
  // Check connection status
  if (WiFi.status() == WL_CONNECTED) {
    
    // Get RSSI (Received Signal Strength Indicator)
    long rssi = WiFi.RSSI();
    
    // Get Timestamp (Milliseconds since boot)
    unsigned long timestamp = millis();

    // Print to Serial Monitor
    // Format: Timestamp, RSSI, Placeholder_Label
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(rssi);
    Serial.println(",Unknown"); // 'Unknown' is a placeholder. You will label this manually in Excel later (e.g., 'Walking')

  } else {
    Serial.println("[Error] WiFi Disconnected. Reconnecting...");
    WiFi.reconnect();
    delay(1000);
  }

  delay(SAMPLING_DELAY_MS);
}
