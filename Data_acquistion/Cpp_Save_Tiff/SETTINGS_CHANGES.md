# Camera Configuration Summary

## Problem Solved
The original `BergatronSensorSettings.txt` had 256+ settings including very dark exposure (266µs) and zero gain, causing black images.

## Solution
Created a **minimal settings file** that only sets essential parameters.

---

## New Settings File: `BergatronSensorSettings_minimal.txt`

This file now contains **ONLY 4 settings**:

```
ShutterMode         GlobalReset      ← Global reset shutter mode
TriggerSelector     FrameStart       ← Select frame start trigger
TriggerMode         On               ← Enable hardware trigger
TriggerSource       Line0            ← Use Line0 as trigger input
TriggerActivation   RisingEdge       ← Trigger on rising edge
```

**All other camera parameters use their default values** (exposure, gain, resolution, etc.)

---

## Program Configuration

In `Cpp_Save_Tiff.cpp` (lines ~33-39):

```cpp
#define USE_HARDWARE_TRIGGER true        // Hardware trigger mode
#define OVERRIDE_EXPOSURE_GAIN true      // Override exposure/gain
#define OVERRIDE_EXPOSURE_TIME 100000.0  // 100ms exposure (adjustable)
#define OVERRIDE_GAIN 15.0               // 15 dB gain (adjustable)
```

---

## Current Setup

✅ **Minimal Settings**: Only sets shutter mode and trigger configuration  
✅ **Hardware Trigger**: Enabled, waits for Line0 rising edge  
✅ **Exposure**: 100ms (100,000 µs) - much brighter than original 266µs  
✅ **Gain**: 15 dB - amplifies the signal  
✅ **Resolution**: Camera default (4024x3036)  
✅ **All other parameters**: Camera defaults  

---

## Adjusting Brightness

If image is still too dark or too bright, edit these values in `Cpp_Save_Tiff.cpp`:

### For Brighter Images:
```cpp
#define OVERRIDE_EXPOSURE_TIME 200000.0  // 200ms (longer exposure)
#define OVERRIDE_GAIN 20.0               // 20 dB (more gain)
```

### For Darker Images (if too bright):
```cpp
#define OVERRIDE_EXPOSURE_TIME 50000.0   // 50ms (shorter exposure)
#define OVERRIDE_GAIN 10.0               // 10 dB (less gain)
```

### To Use Camera Auto-Exposure:
```cpp
#define OVERRIDE_EXPOSURE_GAIN false     // Use camera defaults
```

Then run: `make && ./Cpp_Save_Tiff`

---

## Files

- **Minimal settings**: `BergatronSensorSettings_minimal.txt` (4 settings)
- **Original settings**: `BergatronSensorSettings.txt` (256+ settings, preserved)
- **Source code**: `Cpp_Save_Tiff.cpp`
- **Output**: `Images/Cpp_Save/image_N.tiff`

---

## Testing

1. **Software trigger test** (no hardware trigger needed):
   ```cpp
   #define USE_HARDWARE_TRIGGER false
   ```
   Then: `make && ./Cpp_Save_Tiff`

2. **Hardware trigger mode** (requires trigger on Line0):
   ```cpp
   #define USE_HARDWARE_TRIGGER true
   ```
   Then: `make && ./Cpp_Save_Tiff`

---

## Summary of Changes

| Setting | Old File | New File |
|---------|----------|----------|
| Total settings | 256+ | 4 |
| Exposure | 266µs (too dark) | 100ms (override) |
| Gain | 0 dB | 15 dB (override) |
| Shutter | GlobalReset ✓ | GlobalReset ✓ |
| Trigger | Hardware ✓ | Hardware ✓ |
| Other params | All specified | Camera defaults |

The camera should now produce properly exposed images while maintaining the required shutter mode and hardware trigger functionality.
