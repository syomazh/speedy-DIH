# Hardware-Triggered Acquisition for Digital Inline Holography

## Overview
This program captures 24 images from a Lucid Vision Phoenix camera using hardware triggering via GPIO Line0. The trigger synchronizes image capture with laser pulses for digital inline holography.

## Key Features
- **Hardware Triggered**: Images are captured only when Line0 receives a rising edge trigger (3 Hz)
- **Global Reset Shutter**: Ensures all pixels expose simultaneously (critical for holography)
- **Settings File**: Applies all camera settings from `BergatronSensorSettings_HardwareTrigger.txt`
- **Sequential Saving**: Images saved as `image_0001.tiff` through `image_0024.tiff`
- **Graceful Shutdown**: Press Ctrl+C to stop acquisition cleanly
- **Progress Updates**: Shows progress every 6 images (every 2 seconds)

## Setup

### 1. Hardware Connections
- Camera connected to Jetson Orin Nano via Ethernet
- Hardware trigger connected to GPIO Line0 on camera
- Trigger generates 3 pulses per second (rising edge)

### 2. Power Cycle Camera
**Important**: Power cycle the camera before each run to ensure clean state.

### 3. Compile
```bash
cd /home/berg/Documents/git/speedy-DIH/Data_acquistion/Cpp_Save_Tiff
make clean
make
```

### 4. Run
```bash
./Cpp_Save_Tiff
```

## Critical Settings Applied

The program automatically applies settings from `BergatronSensorSettings_HardwareTrigger.txt`:

### Trigger Configuration
- `TriggerSelector`: FrameStart
- `TriggerMode`: On
- `TriggerSource`: Line0 (GPIO input)
- `TriggerActivation`: RisingEdge
- `TriggerDelay`: 0

### Sensor Settings
- `ShutterMode`: GlobalReset (all pixels expose simultaneously)
- `PixelFormat`: Mono16
- `ADCBitDepth`: Bits12
- `ExposureTime`: 266.528 µs
- `Width`: 4024 pixels
- `Height`: 3036 pixels

### Network Settings
- `GevSCPSPacketSize`: 9000 (jumbo frames)
- `GevSCPD`: 80 µs (inter-packet delay)

## Output

Images are saved to: `Images/test_save/`
- Filename format: `image_0001.tiff`, `image_0002.tiff`, ..., `image_0024.tiff`
- Format: 16-bit TIFF, uncompressed
- Resolution: 4024 x 3036 pixels
- **No persistence**: Images are overwritten on each run

## Expected Behavior

1. Program reads settings file
2. Connects to camera
3. Applies all settings
4. Verifies critical trigger settings
5. Starts acquisition and waits for hardware triggers
6. Captures 24 images (takes ~8 seconds at 3 Hz)
7. Saves each image immediately after capture
8. Displays progress every 6 images
9. Stops automatically after 24 images

## Timing

- **Trigger Rate**: 3 Hz (one trigger every ~333 ms)
- **Expected Duration**: ~8 seconds for 24 images
- **Timeout per Image**: 3 seconds (if no trigger received)

## Troubleshooting

### No Images Captured (Timeout)
- **Check**: Is the hardware trigger connected and active?
- **Check**: Is Line0 configured as input?
- **Check**: Is trigger generating 3 pulses per second?

### Black Images
- **Problem**: Images not synchronized with laser pulse
- **Check**: Hardware trigger timing relative to laser pulse
- **Check**: ExposureTime setting (266.528 µs)
- **Check**: TriggerDelay setting (should be 0)

### Camera Not Found
- **Check**: Camera is powered on
- **Check**: Ethernet cable is connected
- **Check**: Network interface is up and configured
- **Check**: Can ping camera IP address

### Compilation Errors
- **Check**: Arena SDK installed at `/home/berg/ArenaSDK_Linux_ARM64`
- **Check**: `ARENA_SDK_PATH` in `common.mk` is correct
- **Check**: All SDK libraries are present

## Modifying Settings

### Change Number of Images
Edit `Cpp_Save_Tiff.cpp`:
```cpp
#define NUM_IMAGES 24  // Change to desired number
```

### Change Save Path
Edit `Cpp_Save_Tiff.cpp`:
```cpp
#define SAVE_PATH "Images/test_save/"  // Change to desired path
```

### Adjust Camera Settings
Edit `BergatronSensorSettings_HardwareTrigger.txt`:
- Modify any camera parameter (tab-separated: parameter name, value)
- Program automatically applies all valid settings

### Buffer Count
Edit `Cpp_Save_Tiff.cpp`:
```cpp
#define NUM_BUFFERS 50  // Increase for more buffering
```

## Notes

- Program uses ~50 buffers to prevent dropped frames
- Each 12MP Mono16 image is ~24 MB
- Total memory usage: ~1.2 GB for image buffers
- Images are saved with no compression for maximum speed
- Progress updates print every 6 images to avoid console spam

## API vs ArenaView Setting Names

Some settings have different names in the API vs ArenaView:
- `SensorShutterMode` (ArenaView) → `ShutterMode` (API) ✓ Fixed in settings file
- All other settings use API names directly
