#!/bin/bash
# Script to configure network interface for Lucid Phoenix GigE camera
# This script sets up the ethernet interface with proper settings for camera operation

INTERFACE="enP8p1s0"
IP_ADDRESS="169.254.0.1"
NETMASK="255.255.0.0"
MTU_SIZE="9000"

echo "Configuring network interface for Lucid Phoenix camera..."

# Configure the ethernet interface with static IP
echo "Setting IP address ${IP_ADDRESS} on ${INTERFACE}..."
sudo ifconfig ${INTERFACE} ${IP_ADDRESS} netmask ${NETMASK} up

# Set MTU to 9000 for jumbo frames (required for high-speed camera)
echo "Setting MTU to ${MTU_SIZE} for jumbo frames..."
sudo ifconfig ${INTERFACE} mtu ${MTU_SIZE}

# Increase network buffer sizes for GigE camera
echo "Increasing network receive buffers..."
sudo sysctl -w net.core.rmem_max=33554432 > /dev/null
sudo sysctl -w net.core.rmem_default=33554432 > /dev/null

# Verify configuration
echo ""
echo "Configuration complete! Status:"
ip addr show ${INTERFACE} | grep "inet "
ip link show ${INTERFACE} | grep mtu

echo ""
echo "Camera network is ready!"
