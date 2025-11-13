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
#include <csignal>
#include <map>
#include <vector>

#define TAB1 "  "
#define TAB2 "    "
#define SETTINGS_FILE "BergatronSensorSettings_HardwareTrigger.txt"
#define SAVE_PATH "Images/test_save/"
#define NUM_IMAGES 24
#define NUM_BUFFERS 50

// Global flag for graceful shutdown
std::atomic<bool> g_stopAcquisition(false);

// Signal handler for Ctrl+C
void SignalHandler(int signal)
{
	if (signal == SIGINT)
	{
		std::cout << "\n" << TAB1 << "Interrupt signal received. Stopping acquisition...\n";
		g_stopAcquisition = true;
	}
}

// =-=-=-=-=-=-=-=-=-
// =-=- FUNCTIONS -=-
// =-=-=-=-=-=-=-=-=-

// Read settings from file
std::map<std::string, std::string> ReadSettingsFile(const std::string& filename)
{
	std::map<std::string, std::string> settings;
	std::ifstream file(filename);
	
	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open settings file: " + filename);
	}
	
	std::string line;
	while (std::getline(file, line))
	{
		// Skip empty lines and comments
		if (line.empty() || line[0] == '#')
			continue;
		
		// Parse tab-separated key-value pairs
		std::istringstream iss(line);
		std::string key, value;
		
		if (std::getline(iss, key, '\t') && std::getline(iss, value))
		{
			// Trim whitespace
			key.erase(0, key.find_first_not_of(" \t\r\n"));
			key.erase(key.find_last_not_of(" \t\r\n") + 1);
			value.erase(0, value.find_first_not_of(" \t\r\n"));
			value.erase(value.find_last_not_of(" \t\r\n") + 1);
			
			if (!key.empty() && !value.empty())
			{
				settings[key] = value;
			}
		}
	}
	
	file.close();
	return settings;
}

// Apply a single setting to the camera
bool ApplySetting(GenApi::INodeMap* pNodeMap, const std::string& nodeName, const std::string& value)
{
	try
	{
		GenApi::INode* pNode = pNodeMap->GetNode(nodeName.c_str());
		if (!pNode)
		{
			std::cout << TAB2 << "Warning: Node '" << nodeName << "' not found. Skipping.\n";
			return false;
		}
		
		if (!GenApi::IsAvailable(pNode) || !GenApi::IsWritable(pNode))
		{
			std::cout << TAB2 << "Warning: Node '" << nodeName << "' not available or writable. Skipping.\n";
			return false;
		}
		
		// Determine node type and apply value
		switch (pNode->GetPrincipalInterfaceType())
		{
			case GenApi::intfIInteger:
			{
				GenApi::CIntegerPtr pInteger(pNode);
				int64_t intValue = std::stoll(value);
				pInteger->SetValue(intValue);
				break;
			}
			case GenApi::intfIFloat:
			{
				GenApi::CFloatPtr pFloat(pNode);
				double floatValue = std::stod(value);
				pFloat->SetValue(floatValue);
				break;
			}
			case GenApi::intfIBoolean:
			{
				GenApi::CBooleanPtr pBoolean(pNode);
				bool boolValue = (value == "1" || value == "true" || value == "True" || value == "On");
				pBoolean->SetValue(boolValue);
				break;
			}
			case GenApi::intfIEnumeration:
			{
				GenApi::CEnumerationPtr pEnumeration(pNode);
				GenApi::CEnumEntryPtr pEntry = pEnumeration->GetEntryByName(value.c_str());
				if (pEntry && GenApi::IsAvailable(pEntry))
				{
					pEnumeration->SetIntValue(pEntry->GetValue());
				}
				else
				{
					std::cout << TAB2 << "Warning: Enum value '" << value << "' not available for '" << nodeName << "'. Skipping.\n";
					return false;
				}
				break;
			}
			case GenApi::intfIString:
			{
				GenApi::CStringPtr pString(pNode);
				pString->SetValue(value.c_str());
				break;
			}
			case GenApi::intfICommand:
			{
				GenApi::CCommandPtr pCommand(pNode);
				pCommand->Execute();
				break;
			}
			default:
				std::cout << TAB2 << "Warning: Unsupported node type for '" << nodeName << "'. Skipping.\n";
				return false;
		}
		
		return true;
	}
	catch (const GenICam::GenericException& e)
	{
		std::cout << TAB2 << "Error applying setting '" << nodeName << "': " << e.what() << "\n";
		return false;
	}
}

// Apply all settings from the file to the camera
void ApplySettingsToCamera(Arena::IDevice* pDevice, const std::map<std::string, std::string>& settings)
{
	std::cout << TAB1 << "Applying " << settings.size() << " settings from file...\n";
	
	GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();
	int successCount = 0;
	int failCount = 0;
	
	for (const auto& pair : settings)
	{
		if (ApplySetting(pNodeMap, pair.first, pair.second))
		{
			successCount++;
		}
		else
		{
			failCount++;
		}
	}
	
	std::cout << TAB1 << "Settings applied: " << successCount << " successful, " << failCount << " skipped/failed\n";
}

// Verify critical hardware trigger settings
void VerifyTriggerSettings(Arena::IDevice* pDevice)
{
	GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();
	
	std::cout << TAB1 << "Verifying critical trigger settings:\n";
	
	// Check TriggerMode
	GenApi::CEnumerationPtr pTriggerMode = pNodeMap->GetNode("TriggerMode");
	if (pTriggerMode)
	{
		std::cout << TAB2 << "TriggerMode: " << pTriggerMode->ToString() << "\n";
	}
	
	// Check TriggerSource
	GenApi::CEnumerationPtr pTriggerSource = pNodeMap->GetNode("TriggerSource");
	if (pTriggerSource)
	{
		std::cout << TAB2 << "TriggerSource: " << pTriggerSource->ToString() << "\n";
	}
	
	// Check TriggerActivation
	GenApi::CEnumerationPtr pTriggerActivation = pNodeMap->GetNode("TriggerActivation");
	if (pTriggerActivation)
	{
		std::cout << TAB2 << "TriggerActivation: " << pTriggerActivation->ToString() << "\n";
	}
	
	// Check SensorShutterMode
	GenApi::CEnumerationPtr pShutterMode = pNodeMap->GetNode("SensorShutterMode");
	if (pShutterMode)
	{
		std::cout << TAB2 << "SensorShutterMode: " << pShutterMode->ToString() << "\n";
	}
	
	// Check ExposureTime
	GenApi::CFloatPtr pExposureTime = pNodeMap->GetNode("ExposureTime");
	if (pExposureTime)
	{
		std::cout << TAB2 << "ExposureTime: " << pExposureTime->GetValue() << " µs\n";
	}
}

// Save image to file with sequential numbering
void SaveImage(Arena::IImage* pImage, int imageNumber)
{
	try
	{
		// Create filename with zero-padded number
		std::ostringstream filename;
		filename << SAVE_PATH << "image_" << std::setw(4) << std::setfill('0') << imageNumber << ".tiff";
		
		// Convert to Mono16 if needed (should already be Mono16 from camera)
		Arena::IImage* pImageToSave = pImage;
		
		// Prepare image parameters
		Save::ImageParams params(
			pImageToSave->GetWidth(),
			pImageToSave->GetHeight(),
			pImageToSave->GetBitsPerPixel());
		
		// Prepare image writer
		Save::ImageWriter writer(params, filename.str().c_str());
		
		// Set to TIFF format with no compression
		writer.SetTiff(".tiff", Save::NoCompression, false);
		
		// Save image
		writer << pImageToSave->GetData();
	}
	catch (const std::exception& e)
	{
		std::cout << TAB2 << "Error saving image " << imageNumber << ": " << e.what() << "\n";
	}
}

// Main acquisition function
void RunHardwareTriggeredAcquisition(Arena::IDevice* pDevice)
{
	std::cout << "\n" << TAB1 << "Starting hardware-triggered acquisition...\n";
	std::cout << TAB1 << "Waiting for " << NUM_IMAGES << " trigger events from Line0...\n";
	std::cout << TAB1 << "(Press Ctrl+C to stop early)\n\n";
	
	// Start stream
	pDevice->StartStream();
	
	int imagesReceived = 0;
	int lastReportedCount = 0;
	
	auto startTime = std::chrono::steady_clock::now();
	
	while (imagesReceived < NUM_IMAGES && !g_stopAcquisition)
	{
		try
		{
			// Wait for image with timeout (3 seconds - longer than trigger period)
			Arena::IImage* pImage = pDevice->GetImage(3000);
			
			imagesReceived++;
			
			// Save image
			SaveImage(pImage, imagesReceived);
			
			// Print progress every 6 images (every 2 seconds at 3 fps)
			if (imagesReceived % 6 == 0 || imagesReceived == NUM_IMAGES)
			{
				auto currentTime = std::chrono::steady_clock::now();
				auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
				std::cout << TAB1 << "Progress: " << imagesReceived << "/" << NUM_IMAGES 
				          << " images captured and saved (Time: " << elapsed << "s)\n";
			}
			
			// Requeue buffer
			pDevice->RequeueBuffer(pImage);
		}
		catch (const GenICam::TimeoutException&)
		{
			if (g_stopAcquisition)
				break;
			
			std::cout << TAB1 << "Warning: Timeout waiting for triggered image. ";
			std::cout << "Check that hardware trigger is active.\n";
			std::cout << TAB1 << "Images received so far: " << imagesReceived << "/" << NUM_IMAGES << "\n";
			
			// Continue waiting unless we want to abort
			continue;
		}
	}
	
	// Stop stream
	pDevice->StopStream();
	
	auto endTime = std::chrono::steady_clock::now();
	auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
	
	std::cout << "\n" << TAB1 << "Acquisition complete!\n";
	std::cout << TAB1 << "Total images captured: " << imagesReceived << "/" << NUM_IMAGES << "\n";
	std::cout << TAB1 << "Total time: " << totalTime << " seconds\n";
	if (imagesReceived > 0 && totalTime > 0)
	{
		std::cout << TAB1 << "Average rate: " << (double)imagesReceived / totalTime << " fps\n";
	}
}

// =-=-=-=-=-=-=-=-=-
// =- PREPARATION -=-
// =- & CLEAN UP =-=-
// =-=-=-=-=-=-=-=-=-

int main()
{
	// Register signal handler for graceful shutdown
	std::signal(SIGINT, SignalHandler);
	
	// Flag to track when an exception has been thrown
	bool exceptionThrown = false;
	
	std::cout << "===================================================\n";
	std::cout << "Hardware-Triggered Acquisition for Digital Inline Holography\n";
	std::cout << "===================================================\n\n";
	
	try
	{
		// Read settings from file
		std::cout << TAB1 << "Reading settings from: " << SETTINGS_FILE << "\n";
		std::map<std::string, std::string> settings = ReadSettingsFile(SETTINGS_FILE);
		std::cout << TAB1 << "Loaded " << settings.size() << " settings from file\n\n";
		
		// Initialize Arena SDK
		std::cout << TAB1 << "Initializing Arena SDK...\n";
		Arena::ISystem* pSystem = Arena::OpenSystem();
		
		// Discover devices
		std::cout << TAB1 << "Discovering devices...\n";
		pSystem->UpdateDevices(100);
		std::vector<Arena::DeviceInfo> devices = pSystem->GetDevices();
		
		if (devices.size() == 0)
		{
			std::cout << "\n" << TAB1 << "No camera connected!\n";
			std::cout << TAB1 << "Please check:\n";
			std::cout << TAB2 << "1. Camera is powered on\n";
			std::cout << TAB2 << "2. Ethernet cable is connected\n";
			std::cout << TAB2 << "3. Network interface is configured correctly\n";
			Arena::CloseSystem(pSystem);
			std::cout << "\nPress enter to exit\n";
			std::getchar();
			return -1;
		}
		
		// Display and select device
		std::cout << TAB1 << "Found " << devices.size() << " device(s)\n";
		for (size_t i = 0; i < devices.size(); i++)
		{
			std::cout << TAB2 << i + 1 << ". " << devices[i].ModelName() 
			          << " (SN: " << devices[i].SerialNumber() 
			          << ", IP: " << devices[i].IpAddressStr() << ")\n";
		}
		
		// Auto-select first device
		Arena::IDevice* pDevice = pSystem->CreateDevice(devices[0]);
		std::cout << TAB1 << "Selected device: " << devices[0].ModelName() << "\n\n";
		
		// Configure stream settings
		std::cout << TAB1 << "Configuring stream settings...\n";
		GenApi::INodeMap* pTLStreamNodeMap = pDevice->GetTLStreamNodeMap();
		
		// Enable auto packet size negotiation
		Arena::SetNodeValue<bool>(pTLStreamNodeMap, "StreamAutoNegotiatePacketSize", true);
		
		// Enable packet resend
		Arena::SetNodeValue<bool>(pTLStreamNodeMap, "StreamPacketResendEnable", true);
		
		std::cout << TAB1 << "Stream configuration complete\n\n";
		
		// Apply settings from file
		ApplySettingsToCamera(pDevice, settings);
		std::cout << "\n";
		
		// Verify critical settings
		VerifyTriggerSettings(pDevice);
		std::cout << "\n";
		
		// Run acquisition
		RunHardwareTriggeredAcquisition(pDevice);
		
		// Clean up
		std::cout << "\n" << TAB1 << "Cleaning up...\n";
		pSystem->DestroyDevice(pDevice);
		Arena::CloseSystem(pSystem);
		
		std::cout << TAB1 << "Done!\n";
	}
	catch (GenICam::GenericException& ge)
	{
		std::cout << "\nGenICam exception thrown: " << ge.what() << "\n";
		exceptionThrown = true;
	}
	catch (std::exception& ex)
	{
		std::cout << "\nStandard exception thrown: " << ex.what() << "\n";
		exceptionThrown = true;
	}
	catch (...)
	{
		std::cout << "\nUnexpected exception thrown\n";
		exceptionThrown = true;
	}
	
	std::cout << "\nPress enter to exit\n";
	std::cin.ignore();
	std::getchar();
	
	if (exceptionThrown)
		return -1;
	else
		return 0;
}
