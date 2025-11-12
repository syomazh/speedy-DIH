/***************************************************************************************
 ***                                                                                 ***
 ***  Copyright (c) 2024, Lucid Vision Labs, Inc.                                    ***
 ***                                                                                 ***
 ***  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     ***
 ***  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       ***
 ***  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    ***
 ***  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         ***
 ***  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  ***
 ***  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  ***
 ***  SOFTWARE.                                                                      ***
 ***                                                                                 ***
 ***************************************************************************************/

#include "stdafx.h"
#include "ArenaApi.h"
#include "SaveApi.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <thread>
#include <chrono>

#define TAB1 "  "
#define SETTINGS_FILE "BergatronSensorSettings_HardwareTrigger.txt"
#define SAVE_PATH "Images/test_save/"

// Load settings from file and apply to device
void LoadAndApplySettings(Arena::IDevice* pDevice)
{
    std::cout << TAB1 << "Loading settings from " << SETTINGS_FILE << "\n";
    
    GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();
    std::ifstream file(SETTINGS_FILE);
    
    if (!file.is_open())
    {
        std::cout << TAB1 << "ERROR: Could not open settings file!\n";
        throw std::runtime_error("Settings file not found");
    }
    
    int settingsApplied = 0;
    int settingsFailed = 0;
    std::string line;
    
    while (std::getline(file, line))
    {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;
            
        // Parse line (format: NodeName\tValue)
        size_t tabPos = line.find('\t');
        if (tabPos == std::string::npos)
            continue;
            
        std::string nodeName = line.substr(0, tabPos);
        std::string value = line.substr(tabPos + 1);
        
        try
        {
            GenApi::INode* pNode = pNodeMap->GetNode(nodeName.c_str());
            if (!pNode || !GenApi::IsWritable(pNode))
            {
                settingsFailed++;
                continue;
            }
            
            GenApi::EInterfaceType nodeType = pNode->GetPrincipalInterfaceType();
            
            switch (nodeType)
            {
                case GenApi::intfIInteger:
                {
                    GenApi::CIntegerPtr pInteger(pNode);
                    pInteger->SetValue(std::stoll(value));
                    settingsApplied++;
                    break;
                }
                case GenApi::intfIFloat:
                {
                    GenApi::CFloatPtr pFloat(pNode);
                    pFloat->SetValue(std::stod(value));
                    settingsApplied++;
                    break;
                }
                case GenApi::intfIBoolean:
                {
                    GenApi::CBooleanPtr pBoolean(pNode);
                    bool boolValue = (value == "1" || value == "true" || value == "True");
                    pBoolean->SetValue(boolValue);
                    settingsApplied++;
                    break;
                }
                case GenApi::intfIEnumeration:
                {
                    GenApi::CEnumerationPtr pEnumeration(pNode);
                    pEnumeration->FromString(value.c_str());
                    settingsApplied++;
                    break;
                }
                case GenApi::intfIString:
                {
                    GenApi::CStringPtr pString(pNode);
                    pString->SetValue(value.c_str());
                    settingsApplied++;
                    break;
                }
                default:
                    settingsFailed++;
                    break;
            }
        }
        catch (...)
        {
            settingsFailed++;
        }
    }
    
    file.close();
    
    std::cout << TAB1 << "Settings applied: " << settingsApplied << "\n";
    if (settingsFailed > 0)
        std::cout << TAB1 << "Settings failed: " << settingsFailed << "\n";
}

// Configure stream for optimal performance
void ConfigureStream(Arena::IDevice* pDevice)
{
    GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();
    GenApi::INodeMap* pStreamNodeMap = pDevice->GetTLStreamNodeMap();
    
    std::cout << TAB1 << "Configuring stream...\n";
    
    // Auto-negotiate packet size
    try
    {
        Arena::SetNodeValue<bool>(pNodeMap, "GevSCPSDoNotFragment", false);
        Arena::ExecuteNode(pStreamNodeMap, "StreamAutoNegotiatePacketSize");
        int64_t packetSize = Arena::GetNodeValue<int64_t>(pNodeMap, "GevSCPSPacketSize");
        std::cout << TAB1 << "Packet size: " << packetSize << " bytes\n";
    }
    catch (...)
    {
        std::cout << TAB1 << "Warning: Could not auto-negotiate packet size\n";
    }
    
    // Set inter-packet delay
    try
    {
        Arena::SetNodeValue<int64_t>(pNodeMap, "GevSCPD", 1000);
    }
    catch (...) {}
    
    // Set buffer handling mode to newest only
    try
    {
        Arena::SetNodeValue<GenICam::gcstring>(
            pStreamNodeMap,
            "StreamBufferHandlingMode",
            "NewestOnly");
    }
    catch (...) {}
    
    std::cout << TAB1 << "Stream configured\n";
}

int main()
{
    Arena::ISystem* pSystem = nullptr;
    Arena::IDevice* pDevice = nullptr;

    try
    {
        // Create system and find device
        std::cout << "Initializing system...\n";
        pSystem = Arena::OpenSystem();
        pSystem->UpdateDevices(100);
        
        std::vector<Arena::DeviceInfo> deviceInfos = pSystem->GetDevices();
        if (deviceInfos.size() == 0)
        {
            std::cout << "ERROR: No devices found\n";
            Arena::CloseSystem(pSystem);
            return -1;
        }
        
        std::cout << "Found " << deviceInfos.size() << " device(s)\n";
        pDevice = pSystem->CreateDevice(deviceInfos[0]);
        
        // Load settings from file
        LoadAndApplySettings(pDevice);
        
        // Get node map BEFORE configuring stream
        GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();
        
        // CRITICAL: Disable ExposureActive trigger BEFORE stream configuration
        // This must be done early, while camera is not streaming
        std::cout << "\nConfiguring Trigger Settings (before stream):\n";
        try {
            std::cout << TAB1 << "Disabling ExposureActive trigger\n";
            Arena::SetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerSelector", "ExposureActive");
            Arena::SetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerMode", "Off");
            std::cout << TAB1 << "✓ ExposureActive trigger disabled\n";
        } catch (GenICam::GenericException& e) {
            std::cout << TAB1 << "✗ Warning: Could not disable ExposureActive: " << e.what() << "\n";
        }
        
        // Configure FrameStart trigger
        std::cout << TAB1 << "Configuring FrameStart trigger\n";
        Arena::SetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerSelector", "FrameStart");
        Arena::SetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerMode", "On");
        Arena::SetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerSource", "Line0");
        Arena::SetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerActivation", "RisingEdge");
        std::cout << TAB1 << "✓ FrameStart trigger: Line0, RisingEdge\n";
        
        // Configure stream
        ConfigureStream(pDevice);
        
        // Verify settings
        GenApi::CIntegerPtr pWidth = pNodeMap->GetNode("Width");
        GenApi::CIntegerPtr pHeight = pNodeMap->GetNode("Height");
        
        std::cout << "\nCamera Configuration:\n";
        std::cout << TAB1 << "Trigger Mode: On\n";
        std::cout << TAB1 << "Trigger Source: Line0 (GPIO)\n";
        std::cout << TAB1 << "Trigger Activation: RisingEdge\n";
        std::cout << TAB1 << "Resolution: " << pWidth->GetValue() << "x" << pHeight->GetValue() << "\n";
        
        // Print Exposure/Shutter Settings
        std::cout << "\nExposure/Shutter Settings:\n";
        try {
            double exposureTime = Arena::GetNodeValue<double>(pNodeMap, "ExposureTime");
            std::cout << TAB1 << "ExposureTime: " << exposureTime << " µs\n";
        } catch (...) {
            std::cout << TAB1 << "ExposureTime: Unable to read\n";
        }
        
        try {
            GenICam::gcstring exposureAuto = Arena::GetNodeValue<GenICam::gcstring>(pNodeMap, "ExposureAuto");
            std::cout << TAB1 << "ExposureAuto: " << exposureAuto << "\n";
        } catch (...) {}
        
        try {
            GenICam::gcstring exposureMode = Arena::GetNodeValue<GenICam::gcstring>(pNodeMap, "ExposureMode");
            std::cout << TAB1 << "ExposureMode: " << exposureMode << "\n";
        } catch (...) {}
        
        try {
            GenICam::gcstring shutterMode = Arena::GetNodeValue<GenICam::gcstring>(pNodeMap, "ShutterMode");
            std::cout << TAB1 << "ShutterMode: " << shutterMode << "\n";
        } catch (...) {
            // Try alternative parameter name
            try {
                GenICam::gcstring sensorShutterMode = Arena::GetNodeValue<GenICam::gcstring>(pNodeMap, "SensorShutterMode");
                std::cout << TAB1 << "SensorShutterMode: " << sensorShutterMode << "\n";
            } catch (...) {
                std::cout << TAB1 << "ShutterMode: Unable to read\n";
            }
        }
        
        // Print ExposureActive Trigger Status
        std::cout << "\nExposureActive Trigger Status:\n";
        try {
            Arena::SetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerSelector", "ExposureActive");
            GenICam::gcstring expActiveTriggerMode = Arena::GetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerMode");
            std::cout << TAB1 << "ExposureActive TriggerMode: " << expActiveTriggerMode << "\n";
            
            if (expActiveTriggerMode == "On") {
                GenICam::gcstring expActiveTriggerSource = Arena::GetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerSource");
                GenICam::gcstring expActiveTriggerActivation = Arena::GetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerActivation");
                std::cout << TAB1 << "ExposureActive TriggerSource: " << expActiveTriggerSource << "\n";
                std::cout << TAB1 << "ExposureActive TriggerActivation: " << expActiveTriggerActivation << "\n";
                std::cout << TAB1 << "⚠️  WARNING: ExposureActive trigger is ON - this may cause timing issues!\n";
            }
            
            // Set selector back to FrameStart
            Arena::SetNodeValue<GenICam::gcstring>(pNodeMap, "TriggerSelector", "FrameStart");
        } catch (...) {
            std::cout << TAB1 << "Unable to read ExposureActive trigger status\n";
        }
        
        // Start stream
        std::cout << "\nStarting stream...\n";
        pDevice->StartStream();
        
        std::cout << "\n==============================================\n";
        std::cout << "Waiting for hardware triggers on GPIO Line0...\n";
        std::cout << "Press 'q' + Enter to quit\n";
        std::cout << "==============================================\n\n";
        
        int imageCount = 0;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Main acquisition loop - wait for hardware triggers
        bool running = true;
        while (running)
        {
            try
            {
                // Wait for hardware-triggered image (5 second timeout)
                // This will block until a GPIO trigger pulse is received
                Arena::IImage* pImage = pDevice->GetImage(5000);
                
                imageCount++;
                
                // Generate filename with timestamp
                auto now = std::chrono::system_clock::now();
                auto timestamp = std::chrono::system_clock::to_time_t(now);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()) % 1000;
                
                std::stringstream filename;
                filename << SAVE_PATH << "image_" 
                         << std::setfill('0') << std::setw(4) << imageCount 
                         << "_" << timestamp << "_" << std::setw(3) << ms.count() 
                         << ".tiff";
                
                // Save image
                Save::ImageParams params(
                    pImage->GetWidth(),
                    pImage->GetHeight(),
                    pImage->GetBitsPerPixel()
                );
                Save::ImageWriter writer(params, filename.str().c_str());
                writer.Save(pImage->GetData());
                
                // Print status
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - startTime).count();
                
                std::cout << "Saved! (" << pImage->GetWidth() << "x" << pImage->GetHeight() 
                          << ") | Total: " << imageCount << " | Runtime: " << elapsed << "s\n";
                std::cout << "Saved to: " << filename.str() << "\n\n";
                
                // Requeue buffer
                pDevice->RequeueBuffer(pImage);
            }
            catch (GenICam::TimeoutException&)
            {
                std::cout << "Timeout waiting for image\n\n";
            }
            catch (GenICam::GenericException& ge)
            {
                std::cout << "Error acquiring image: " << ge.what() << "\n\n";
            }
        }
        
        std::cout << "\nStopping acquisition...\n";
        
        // Stop stream
        pDevice->StopStream();
        
        std::cout << "\nTotal images captured: " << imageCount << "\n";
        std::cout << "Images saved to: " << SAVE_PATH << "\n";
        
        // Cleanup
        pSystem->DestroyDevice(pDevice);
        Arena::CloseSystem(pSystem);
    }
    catch (GenICam::GenericException& ge)
    {
        std::cout << "\nGenICam exception: " << ge.what() << "\n";
        
        if (pDevice)
            pSystem->DestroyDevice(pDevice);
        if (pSystem)
            Arena::CloseSystem(pSystem);
        
        return -1;
    }
    catch (std::exception& ex)
    {
        std::cout << "\nException: " << ex.what() << "\n";
        
        if (pDevice)
            pSystem->DestroyDevice(pDevice);
        if (pSystem)
            Arena::CloseSystem(pSystem);
        
        return -1;
    }
    
    std::cout << "\nPress enter to exit\n";
    std::cin.get();
    
    return 0;
}

