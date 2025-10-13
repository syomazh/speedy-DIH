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
#include <sstream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>

#define TAB1 "  "
#define SETTINGS_FILE "BergatronSensorSettings.txt"
#define SAVE_PATH "Images/test_save/"

// Function to load settings from file and apply to device
void LoadAndApplySettings(Arena::IDevice* pDevice)
{
	std::cout << TAB1 << "Loading settings from " << SETTINGS_FILE << "\n";
	
	GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();
	
	// Read settings file
	std::ifstream file(SETTINGS_FILE);
	if (!file.is_open())
	{
		std::cout << TAB1 << "Warning: Could not open settings file\n";
		return;
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
			if (!pNode)
			{
				std::cout << TAB1 << "  Failed (node not found): " << nodeName << "\n";
				settingsFailed++;
				continue;
			}
			
			if (!GenApi::IsWritable(pNode))
			{
				std::cout << TAB1 << "  Failed (not writable): " << nodeName << "\n";
				settingsFailed++;
				continue;
			}
				
			// Try to set the value based on node type
			GenApi::EInterfaceType nodeType = pNode->GetPrincipalInterfaceType();
			
			if (nodeType == GenApi::intfIInteger)
			{
				GenApi::CIntegerPtr pInteger(pNode);
				pInteger->SetValue(std::stoll(value));
				settingsApplied++;
			}
			else if (nodeType == GenApi::intfIFloat)
			{
				GenApi::CFloatPtr pFloat(pNode);
				pFloat->SetValue(std::stod(value));
				settingsApplied++;
			}
			else if (nodeType == GenApi::intfIBoolean)
			{
				GenApi::CBooleanPtr pBoolean(pNode);
				bool boolValue = (value == "1" || value == "true" || value == "True");
				pBoolean->SetValue(boolValue);
				settingsApplied++;
			}
			else if (nodeType == GenApi::intfIEnumeration)
			{
				GenApi::CEnumerationPtr pEnumeration(pNode);
				pEnumeration->FromString(value.c_str());
				settingsApplied++;
			}
			else if (nodeType == GenApi::intfIString)
			{
				GenApi::CStringPtr pString(pNode);
				pString->SetValue(value.c_str());
				settingsApplied++;
			}
		}
		catch (GenICam::GenericException& ge)
		{
			std::cout << TAB1 << "  Failed (exception): " << nodeName << " = " << value << " - " << ge.what() << "\n";
			settingsFailed++;
		}
		catch (...)
		{
			std::cout << TAB1 << "  Failed (unknown error): " << nodeName << " = " << value << "\n";
			settingsFailed++;
		}
	}
	
	file.close();
	
	std::cout << TAB1 << "Settings applied: " << settingsApplied << "\n";
	if (settingsFailed > 0)
		std::cout << TAB1 << "Settings failed: " << settingsFailed << " (this is often normal)\n";
}

// Function to configure stream settings
void ConfigureStream(Arena::IDevice* pDevice)
{
	GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();
	GenApi::INodeMap* pStreamNodeMap = pDevice->GetTLStreamNodeMap();
	
	// Enable packet size auto-negotiation
	try
	{
		std::cout << TAB1 << "Attempting to auto-negotiate packet size...\n";
		Arena::SetNodeValue<bool>(pNodeMap, "GevSCPSDoNotFragment", false);
		Arena::ExecuteNode(pStreamNodeMap, "StreamAutoNegotiatePacketSize");
		std::cout << TAB1 << "Packet size auto-negotiation completed\n";
		
		// Check the negotiated packet size
		int64_t packetSize = Arena::GetNodeValue<int64_t>(pNodeMap, "GevSCPSPacketSize");
		std::cout << TAB1 << "Negotiated packet size: " << packetSize << " bytes\n";
	}
	catch (GenICam::GenericException& ge)
	{
		std::cout << TAB1 << "Warning: Could not auto-negotiate packet size: " << ge.what() << "\n";
		std::cout << TAB1 << "Attempting to use jumbo frame packet size...\n";
		
		// Try jumbo frames first (MTU 9000 is enabled on interface)
		try
		{
			Arena::SetNodeValue<int64_t>(pNodeMap, "GevSCPSPacketSize", 9000);
			std::cout << TAB1 << "Set packet size to 9000 bytes (jumbo frames)\n";
		}
		catch (...)
		{
			// Fall back to standard Ethernet if jumbo frames don't work
			try
			{
				Arena::SetNodeValue<int64_t>(pNodeMap, "GevSCPSPacketSize", 1500);
				std::cout << TAB1 << "Set packet size to 1500 bytes (standard)\n";
			}
			catch (...)
			{
				std::cout << TAB1 << "Warning: Could not set packet size\n";
			}
		}
	}
	
	// Set inter-packet delay to reduce network congestion (helps prevent black lines)
	try
	{
		// Set delay between packets (in ticks, where 1 tick = 8ns for GigE)
		// 1000 ticks = 8 microseconds delay
		Arena::SetNodeValue<int64_t>(pNodeMap, "GevSCPD", 1000);
		std::cout << TAB1 << "Set inter-packet delay to 1000 ticks (~8us)\n";
	}
	catch (GenICam::GenericException& ge)
	{
		std::cout << TAB1 << "Warning: Could not set inter-packet delay: " << ge.what() << "\n";
	}
	
	// Increase stream buffer count to handle bursts better
	try
	{
		Arena::SetNodeValue<int64_t>(pStreamNodeMap, "StreamBufferCountMode", 1); // Manual mode
		Arena::SetNodeValue<int64_t>(pStreamNodeMap, "StreamBufferCountManual", 10);
		std::cout << TAB1 << "Set stream buffer count to 10\n";
	}
	catch (GenICam::GenericException& ge)
	{
		std::cout << TAB1 << "Warning: Could not set buffer count: " << ge.what() << "\n";
	}
	
	// Set buffer handling mode
	Arena::SetNodeValue<GenICam::gcstring>(
		pStreamNodeMap,
		"StreamBufferHandlingMode",
		"NewestOnly");
		
	std::cout << TAB1 << "Stream configured\n";
}

int main()
{
	// Flag to track if a system is created
	bool isSystemCreated = false;
	Arena::ISystem* pSystem = nullptr;
	Arena::IDevice* pDevice = nullptr;

	try
	{
		// Create system
		std::cout << "Creating system\n";
		pSystem = Arena::OpenSystem();
		isSystemCreated = true;
		
		// Update and get device list
		std::cout << "Updating device list\n";
		pSystem->UpdateDevices(100);
		std::vector<Arena::DeviceInfo> deviceInfos = pSystem->GetDevices();
		
		if (deviceInfos.size() == 0)
		{
			std::cout << "No devices found\n";
			return -1;
		}
		
		std::cout << "Found " << deviceInfos.size() << " device(s)\n";
		
		// Create device
		std::cout << "Creating device\n";
		pDevice = pSystem->CreateDevice(deviceInfos[0]);
		
		// Load and apply settings
		LoadAndApplySettings(pDevice);
		
		// Configure stream
		ConfigureStream(pDevice);
		
		// Get device stream nodemap
		GenApi::INodeMap* pNodeMap = pDevice->GetNodeMap();
		
		// Get image parameters
		GenApi::CIntegerPtr pWidth = pNodeMap->GetNode("Width");
		GenApi::CIntegerPtr pHeight = pNodeMap->GetNode("Height");
		GenApi::CEnumerationPtr pPixelFormat = pNodeMap->GetNode("PixelFormat");
		
		std::cout << "\nImage settings:\n";
		std::cout << TAB1 << "Width: " << pWidth->GetValue() << "\n";
		std::cout << TAB1 << "Height: " << pHeight->GetValue() << "\n";
		std::cout << TAB1 << "Pixel Format: " << pPixelFormat->GetCurrentEntry()->GetSymbolic() << "\n";
		
		// Get trigger settings
		GenApi::CEnumerationPtr pTriggerMode = pNodeMap->GetNode("TriggerMode");
		GenApi::CEnumerationPtr pTriggerSource = pNodeMap->GetNode("TriggerSource");
		
		std::cout << "\nTrigger settings:\n";
		std::cout << TAB1 << "Trigger Mode: " << pTriggerMode->GetCurrentEntry()->GetSymbolic() << "\n";
		std::cout << TAB1 << "Trigger Source: " << pTriggerSource->GetCurrentEntry()->GetSymbolic() << "\n";
		
		// Start stream
		std::cout << "\nStarting stream...\n";
		pDevice->StartStream();
		
		std::cout << "\nAcquiring images (Press Enter to stop)...\n\n";
		
		// Create a separate thread to check for user input
		std::atomic<bool> stopAcquisition(false);
		std::thread inputThread([&stopAcquisition]() {
			std::cin.get();
			stopAcquisition = true;
		});
		
		int imageCount = 0;
		int savedCount = 0;
		auto startTime = std::chrono::high_resolution_clock::now();
		
		// Acquisition loop
		while (!stopAcquisition)
		{
			try
			{
				// Get image with timeout (1000ms)
				Arena::IImage* pImage = pDevice->GetImage(1000);
				
				imageCount++;
				
				// Save image to disk
				try
				{
					std::stringstream filename;
					filename << SAVE_PATH << "image_" << imageCount << ".tiff";
					
					Save::ImageParams params(
						pImage->GetWidth(),
						pImage->GetHeight(),
						pImage->GetBitsPerPixel()
					);
					Save::ImageWriter writer(params, filename.str().c_str());
					writer.Save(pImage->GetData());
					savedCount++;
				}
				catch (GenICam::GenericException& ge)
				{
					std::cout << "\nWarning: Could not save image: " << ge.what() << "\n";
				}
				
				// Calculate FPS every second
				auto currentTime = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
				
				if (duration >= 1)
				{
					double fps = imageCount / static_cast<double>(duration);
					std::cout << "\rImages: " << imageCount 
							  << " | Saved: " << savedCount
							  << " | FPS: " << std::fixed << std::setprecision(2) << fps 
							  << " | Size: " << pImage->GetWidth() << "x" << pImage->GetHeight()
							  << " | Timestamp: " << pImage->GetTimestamp()
							  << std::flush;
				}
				
				// Requeue image buffer
				pDevice->RequeueBuffer(pImage);
			}
			catch (GenICam::TimeoutException&)
			{
				// Timeout waiting for image - this is expected with trigger mode
				// Continue waiting
			}
		}
		
		std::cout << "\n\nStopping acquisition...\n";
		
		// Stop stream
		pDevice->StopStream();
		
		// Wait for input thread to finish
		if (inputThread.joinable())
			inputThread.join();
		
		std::cout << "\nTotal images acquired: " << imageCount << "\n";
		std::cout << "Total images saved: " << savedCount << "\n";
		std::cout << "Images saved to: " << SAVE_PATH << "\n";
		
		// Destroy device
		std::cout << "Destroying device\n";
		pSystem->DestroyDevice(pDevice);
		
		// Close system
		std::cout << "Closing system\n";
		Arena::CloseSystem(pSystem);
	}
	catch (GenICam::GenericException& ge)
	{
		std::cout << "\nGenICam exception thrown: " << ge.what() << "\n";
		
		if (pDevice)
		{
			pSystem->DestroyDevice(pDevice);
		}
		
		if (isSystemCreated)
		{
			Arena::CloseSystem(pSystem);
		}
		
		return -1;
	}
	catch (std::exception& ex)
	{
		std::cout << "\nStandard exception thrown: " << ex.what() << "\n";
		
		if (pDevice)
		{
			pSystem->DestroyDevice(pDevice);
		}
		
		if (isSystemCreated)
		{
			Arena::CloseSystem(pSystem);
		}
		
		return -1;
	}
	
	std::cout << "\nPress enter to complete\n";
	std::cin.get();
	
	return 0;
}

