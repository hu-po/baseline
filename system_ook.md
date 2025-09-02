# System Hardware Information - Intel Core i7-13620H Laptop

## Overview
- **Hostname:** ook
- **Hardware Model:** Intel Core i7-13620H based system
- **Architecture:** x86_64
- **Operating System:** Ubuntu 24.04.2 LTS (Noble)
- **Kernel:** Linux 6.14.0-29-generic

## CPU Information
- **Model:** 13th Gen Intel(R) Core(TM) i7-13620H
- **Architecture:** x86_64 (Intel Family 6, Model 186)
- **Total Cores:** 16 (10 physical cores, 16 threads)
- **Threads per Core:** 2
- **Frequency Range:** 400 MHz - 4900 MHz
- **Current Scaling:** 43%
- **Cache:**
  - L1 Data: 416 KiB (10 instances)
  - L1 Instruction: 448 KiB (10 instances)
  - L2: 9.5 MiB (7 instances)
  - L3: 24 MiB (1 instance)
- **Features:** AVX2, AES-NI, SHA-NI, FMA, SSE4.2, VMX virtualization
- **BogoMIPS:** 5836.80

## Memory
- **Total RAM:** 15.9 GB (15,695 MB)
- **Swap:** 4.0 GB
- **Memory Configuration:**
  - Block Size: 128 MB
  - Total Online: 15.9 GB
  - Memory Range: 896 MB + 15 GB across blocks

## Storage
### Primary Storage (NVMe)
- **Device:** /dev/nvme0n1 - 953.9 GB total
- **Root Partition:** /dev/nvme0n1p2 - 952.8 GB (10% used, 804 GB available)
- **Boot Partition:** /dev/nvme0n1p1 - 1 GB mounted at /boot/efi

### Filesystem Usage
- **Root (/):** 937 GB total, 86 GB used, 804 GB available (10% usage)
- **Boot (/boot/efi):** 1.1 GB total, 6.2 MB used
- **Temporary Filesystems:** Multiple tmpfs mounts for /run, /dev/shm, etc.

## GPU
### Integrated Graphics
- **Model:** Intel Raptor Lake-P [UHD Graphics]
- **Bus ID:** 00:02.0
- **Revision:** 04

### Discrete Graphics
- **Model:** NVIDIA GeForce RTX 4050 Max-Q / Mobile
- **Driver Version:** 575.57.08
- **CUDA Version:** 12.9
- **Bus ID:** 01:00.0
- **Memory:** 6141 MiB total (18 MiB used)
- **Power:** 3W current / 35W max
- **Temperature:** 40°C
- **Performance State:** P8 (idle)
- **Persistence Mode:** On

## Network Interfaces
### Wireless
- **Interface:** wlp62s0
- **Chipset:** MEDIATEK MT7921 802.11ax
- **Status:** UP and connected
- **IP Address:** 192.168.1.90/24
- **MAC Address:** 70:08:94:ef:7a:f9

### Ethernet
- **Interface:** enp63s0
- **Chipset:** Realtek RTL8111/8168/8211/8411 Gigabit Ethernet
- **Status:** DOWN (not connected)
- **MAC Address:** 74:d4:dd:c0:e1:39

## PCI Devices
### Major Components
- **Host Bridge:** Intel Raptor Lake (Device a715)
- **Graphics:** Intel UHD Graphics + NVIDIA RTX 4050
- **Storage Controller:** Intel NVMe RAID Controller
- **Audio:** Intel Raptor Lake-P/U/H cAVS + NVIDIA Audio
- **Network:** MEDIATEK WiFi + Realtek Ethernet
- **Thunderbolt:** Raptor Lake-P Thunderbolt 4 Controllers
- **USB Controllers:** Multiple xHCI controllers for USB 3.2 support

## USB Devices
- **Webcam:** Quanta ACER HD User Facing (04:08:403d)
- **Wireless Device:** Lite-On Technology Corp. (04ca:3802)
- **USB Root Hubs:** Multiple USB 2.0 and 3.0 root hubs

## Security Mitigations
- **Spectre v1:** Mitigated (usercopy/swapgs barriers)
- **Spectre v2:** Mitigated (Enhanced/Automatic IBRS)
- **Spec Store Bypass:** Mitigated (disabled via prctl)
- **L1TF, Meltdown, MDS:** Not affected
- **Reg File Data Sampling:** Mitigated (Clear Register File)

## Key Features
This is a high-performance laptop system featuring:
- 13th Gen Intel Core i7 with 10 cores/16 threads
- Hybrid P-core/E-core architecture for efficiency
- NVIDIA RTX 4050 discrete GPU with CUDA 12.9
- 16 GB RAM for multitasking
- Fast NVMe storage with ~954 GB capacity
- Thunderbolt 4 support
- WiFi 6 (802.11ax) connectivity
- Ubuntu 24.04 LTS