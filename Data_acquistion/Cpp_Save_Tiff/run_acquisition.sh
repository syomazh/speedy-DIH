#!/bin/bash

# Hardware-Triggered Acquisition Script
# This script runs the camera acquisition program with proper setup

echo "=============================================="
echo "Hardware-Triggered Acquisition for DIH"
echo "=============================================="
echo ""

# Check if camera is connected
echo "Checking for camera..."
if ! ip link show eth0 &> /dev/null; then
    echo "Warning: eth0 network interface not found"
    echo "Make sure camera is connected via Ethernet"
    echo ""
fi

# Navigate to correct directory
cd "$(dirname "$0")"

# Check if executable exists
if [ ! -f "./Cpp_Save_Tiff" ]; then
    echo "Error: Cpp_Save_Tiff executable not found"
    echo "Please compile first with: make clean && make"
    exit 1
fi

# Check if settings file exists
if [ ! -f "./BergatronSensorSettings_HardwareTrigger.txt" ]; then
    echo "Error: Settings file not found"
    echo "Expected: BergatronSensorSettings_HardwareTrigger.txt"
    exit 1
fi

# Check if output directory exists
if [ ! -d "./Images/test_save" ]; then
    echo "Creating output directory..."
    mkdir -p ./Images/test_save
fi

echo "Pre-flight checks complete"
echo ""
echo "IMPORTANT REMINDERS:"
echo "  1. Power cycle the camera before running"
echo "  2. Verify hardware trigger is connected to Line0"
echo "  3. Trigger should be generating 3 pulses/second"
echo "  4. Laser must be synchronized with trigger"
echo ""
read -p "Press Enter to start acquisition (Ctrl+C to abort)..."
echo ""

# Run the acquisition program
./Cpp_Save_Tiff

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "Acquisition completed successfully!"
    echo "Images saved to: ./Images/test_save/"
    echo ""
    echo "Captured images:"
    ls -lh ./Images/test_save/image_*.tiff 2>/dev/null | tail -n 10
else
    echo ""
    echo "Acquisition failed or was interrupted"
    echo "Check error messages above"
fi

echo ""
echo "Done."
