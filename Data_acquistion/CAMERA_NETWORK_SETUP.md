# Lucid Phoenix Camera Network Setup

This directory contains scripts to automatically configure your network interface for the Lucid Phoenix GigE camera after system reboots.

## Files

- **setup_camera_network.sh** - Script that configures the network interface
- **lucid-camera-network.service** - Systemd service for automatic startup

## Quick Start (Manual)

If you just want to run the setup manually after a reboot:

```bash
cd /home/berg/Documents/git/speedy-DIH/Data_acquistion
./setup_camera_network.sh
```

## Automatic Setup on Boot (Recommended)

To have the network automatically configured every time you boot:

### 1. Install the systemd service:

```bash
sudo cp /home/berg/Documents/git/speedy-DIH/Data_acquistion/lucid-camera-network.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lucid-camera-network.service
```

### 2. Test the service:

```bash
sudo systemctl start lucid-camera-network.service
sudo systemctl status lucid-camera-network.service
```

### 3. Verify it works:

```bash
# Check the network interface configuration
ip addr show enP8p1s0
ip link show enP8p1s0 | grep mtu

# The IP should be 169.254.0.1 and MTU should be 9000
```

## What the script does:

1. Sets the ethernet interface `enP8p1s0` to IP `169.254.0.1` (GigE camera subnet)
2. Sets MTU to 9000 bytes for jumbo frames (required for high-speed imaging)
3. Increases network receive buffers to 32MB for smooth data transfer

## Troubleshooting

### Camera not detected after running script?

```bash
# Check if the service ran
sudo systemctl status lucid-camera-network.service

# Manually verify network settings
ip addr show enP8p1s0
```

### Disable the automatic startup:

```bash
sudo systemctl disable lucid-camera-network.service
```

### Remove the service completely:

```bash
sudo systemctl disable lucid-camera-network.service
sudo rm /etc/systemd/system/lucid-camera-network.service
sudo systemctl daemon-reload
```

## Manual Configuration Commands

If you need to configure manually:

```bash
# Set IP and bring interface up
sudo ifconfig enP8p1s0 169.254.0.1 netmask 255.255.0.0 up

# Set MTU for jumbo frames
sudo ifconfig enP8p1s0 mtu 9000

# Increase network buffers
sudo sysctl -w net.core.rmem_max=33554432
sudo sysctl -w net.core.rmem_default=33554432
```
