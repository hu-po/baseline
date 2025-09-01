# System Hardware Information - NVIDIA Jetson AGX Orin Developer Kit

## Overview
- **Hostname:** ojo
- **Hardware Model:** NVIDIA Jetson AGX Orin Developer Kit
- **Architecture:** ARM64 (aarch64)
- **Operating System:** Ubuntu 22.04.5 LTS (Jammy)
- **Kernel:** Linux 5.15.148-tegra

## CPU Information
- **Model:** ARM Cortex-A78AE
- **Architecture:** ARMv8 64-bit
- **Total Cores:** 12
- **Clusters:** 3 (4 cores per cluster)
- **Frequency Range:** 115.2 MHz - 2201.6 MHz
- **Cache:**
  - L1 Data: 768 KiB (12 instances)
  - L1 Instruction: 768 KiB (12 instances)
  - L2: 3 MiB (12 instances)
  - L3: 6 MiB (3 instances)
- **Features:** fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm lrcpc dcpop asimddp uscat ilrcpc flagm paca pacg

## Memory
- **Total RAM:** 29.97 GB (31,433,768 KB)
- **Available RAM:** 24.51 GB
- **Swap:** 16 GB total (mostly unused)
- **Current Usage:**
  - Used: 5.0 GB
  - Free: 16 GB
  - Buffers/Cache: 8.6 GB

## Storage
### Primary Storage (eMMC)
- **Device:** /dev/mmcblk0 (59.3 GB total)
- **Root Partition:** /dev/mmcblk0p1 - 57.8 GB (89% used, 6.1 GB available)
- **Boot Partition:** /dev/mmcblk0p10 - 64 MB mounted at /boot/efi

### Secondary Storage
- **NVMe SSD:** /dev/nvme0n1 - 931.5 GB (20% used, 698 GB available, mounted at /mnt)
- **SD Card:** /dev/mmcblk1 - 476.7 GB (not mounted)

## GPU
- **Model:** NVIDIA Orin (nvgpu)
- **Driver Version:** 540.4.0
- **CUDA Version:** 12.6
- **Architecture:** Integrated GPU (part of Jetson AGX Orin SoC)
- **Status:** Active with no running GPU processes

## Network Interfaces
### Wireless
- **Interface:** wlP1p1s0
- **Status:** UP and connected
- **IP Address:** 10.61.230.149/24
- **MAC Address:** 90:e8:68:bc:80:79

### Ethernet
- **Interface:** eno1
- **Status:** Available but not connected
- **MAC Address:** 48:b0:2d:a5:08:c6

### Other Interfaces
- **CAN Interfaces:** can0, can1 (both DOWN)
- **USB Network:** usb0, usb1 (configured but DOWN)
- **Docker Bridge:** docker0 (172.17.0.1/16)
- **L4T Bridge:** l4tbr0 (configured)

## USB Devices
- Realtek 4-Port USB 3.0 Hub
- Realtek 4-Port USB 2.0 Hub
- IMC Networks Bluetooth Radio

## NVIDIA Jetson Platform Details
- **Tegra Release:** R36 (release), REVISION: 4.3
- **Build Date:** Wed Jan 8 01:49:37 UTC 2025
- **Board Type:** Generic
- **Target Architecture:** aarch64
- **Kernel Variant:** Out-of-tree (OOT)

## Key Features
This is a high-performance edge AI development platform featuring:
- 12-core ARM Cortex-A78AE CPU
- Integrated NVIDIA GPU with CUDA 12.6 support
- 30 GB RAM for intensive AI workloads
- Multiple storage options (eMMC, NVMe, SD card)
- Comprehensive connectivity (WiFi, Ethernet, CAN, USB)
- Ubuntu 22.04 LTS with NVIDIA's specialized Tegra kernel