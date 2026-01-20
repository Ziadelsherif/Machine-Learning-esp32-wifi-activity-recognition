import serial
import time
import csv
import os
import sys

def get_unique_filename(activity):
    """Generates a unique filename like 'data/Walking_01.csv'"""
    os.makedirs("data", exist_ok=True)
    i = 1
    while True:
        filename = f"data/{activity}_{i:02d}.csv"
        if not os.path.exists(filename):
            return filename
        i += 1

def collect_data():
    print("--- Wi-Fi HAR Data Collector ---")
    
    # 1. Get COM Port
    default_port = 'COM7'
    port_input = input(f"Enter ESP32 COM Port [default {default_port}]: ").strip()
    serial_port = port_input if port_input else default_port
    
    # 2. Get Activity Label
    print("\nSelect Activity:")
    print("1. Standing")
    print("2. Walking")
    print("3. Jumping")
    choice = input("Enter number (1-3): ").strip()
    
    if choice == '1':
        label = "Standing"
    elif choice == '2':
        label = "Walking"
    elif choice == '3':
        label = "Jumping"
    else:
        print("Invalid choice. Defaulting to 'Unknown'")
        label = "Unknown"

    # 3. Setup File
    output_file = get_unique_filename(label)
    baud_rate = 115200

    # 4. Connect
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"\nSuccessfully connected to {serial_port}!")
        print(f"Saving data to: {output_file}")
        print(f"Labeling data as: {label}")
        print("\n[IMPORTANT] Close this window or press Ctrl+C to stop recording.")
        print("-" * 40)
    except Exception as e:
        print(f"\nError: Could not open {serial_port}.")
        print("Tip: Make sure the Arduino Serial Monitor is CLOSED.")
        print(f"Details: {e}")
        return

    # 5. Record Loop
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "RSSI", "Activity"]) # Header
        
        try:
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Expecting format: timestamp,rssi,extra...
                    # We only care if it's a valid CSV line
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 2:
                            # Parse data
                            timestamp_ms = parts[0]
                            rssi_val = parts[1]
                            
                            # Write to file
                            writer.writerow([timestamp_ms, rssi_val, label])
                            f.flush()
                            
                            print(f"[{label}] Time: {timestamp_ms} | RSSI: {rssi_val}")

        except KeyboardInterrupt:
            print(f"\n\nRecording Stopped. Saved to {output_file}")
            print("To record the next activity, run this script again.")
            ser.close()

if __name__ == "__main__":
    collect_data()
