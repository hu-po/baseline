Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           39 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  16
On-line CPU(s) list:                     0-15
Vendor ID:                               GenuineIntel
Model name:                              13th Gen Intel(R) Core(TM) i7-13620H
CPU family:                              6
Model:                                   186
Thread(s) per core:                      2
Core(s) per socket:                      10
Socket(s):                               1
Stepping:                                2
CPU(s) scaling MHz:                      43%
CPU max MHz:                             4900.0000
CPU min MHz:                             400.0000
BogoMIPS:                                5836.80
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb intel_pt sha_ni xsaveopt xsavec xgetbv1 xsaves split_lock_detect user_shstk avx_vnni dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req hfi vnmi umip pku ospke waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize arch_lbr ibt flush_l1d arch_capabilities
Virtualization:                          VT-x
L1d cache:                               416 KiB (10 instances)
L1i cache:                               448 KiB (10 instances)
L2 cache:                                9.5 MiB (7 instances)
L3 cache:                                24 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-15
Vulnerability Gather data sampling:      Not affected
Vulnerability Ghostwrite:                Not affected
Vulnerability Indirect target selection: Not affected
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Not affected
Vulnerability Reg file data sampling:    Mitigation; Clear Register File
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S
Vulnerability Srbds:                     Not affected
Vulnerability Tsx async abort:           Not affected

--- lsmem ---
RANGE                                  SIZE  STATE REMOVABLE  BLOCK
0x0000000000000000-0x0000000037ffffff  896M online       yes    0-6
0x0000000100000000-0x00000004bfffffff   15G online       yes 32-151

Memory block size:       128M
Total online memory:    15.9G
Total offline memory:      0B

--- lspci ---
0000:00:00.0 Host bridge: Intel Corporation Device a715
0000:00:02.0 VGA compatible controller: Intel Corporation Raptor Lake-P [UHD Graphics] (rev 04)
0000:00:04.0 Signal processing controller: Intel Corporation Raptor Lake Dynamic Platform and Thermal Framework Processor Participant
0000:00:06.0 PCI bridge: Intel Corporation Raptor Lake PCIe 4.0 Graphics Port
0000:00:07.0 PCI bridge: Intel Corporation Raptor Lake-P Thunderbolt 4 PCI Express Root Port #0
0000:00:08.0 System peripheral: Intel Corporation GNA Scoring Accelerator module
0000:00:0a.0 Signal processing controller: Intel Corporation Raptor Lake Crashlog and Telemetry (rev 01)
0000:00:0d.0 USB controller: Intel Corporation Raptor Lake-P Thunderbolt 4 USB Controller
0000:00:0d.2 USB controller: Intel Corporation Raptor Lake-P Thunderbolt 4 NHI #0
0000:00:0e.0 RAID bus controller: Intel Corporation Volume Management Device NVMe RAID Controller Intel Corporation
0000:00:14.0 USB controller: Intel Corporation Alder Lake PCH USB 3.2 xHCI Host Controller (rev 01)
0000:00:14.2 RAM memory: Intel Corporation Alder Lake PCH Shared SRAM (rev 01)
0000:00:15.0 Serial bus controller: Intel Corporation Alder Lake PCH Serial IO I2C Controller #0 (rev 01)
0000:00:16.0 Communication controller: Intel Corporation Alder Lake PCH HECI Controller (rev 01)
0000:00:1c.0 PCI bridge: Intel Corporation Alder Lake-P PCH PCIe Root Port #4 (rev 01)
0000:00:1c.6 PCI bridge: Intel Corporation Device 51be (rev 01)
0000:00:1f.0 ISA bridge: Intel Corporation Raptor Lake LPC/eSPI Controller (rev 01)
0000:00:1f.3 Multimedia audio controller: Intel Corporation Raptor Lake-P/U/H cAVS (rev 01)
0000:00:1f.4 SMBus: Intel Corporation Alder Lake PCH-P SMBus Host Controller (rev 01)
0000:00:1f.5 Serial bus controller: Intel Corporation Alder Lake-P PCH SPI Controller (rev 01)
0000:01:00.0 VGA compatible controller: NVIDIA Corporation AD107M [GeForce RTX 4050 Max-Q / Mobile] (rev a1)
0000:01:00.1 Audio device: NVIDIA Corporation Device 22be (rev a1)
0000:3e:00.0 Network controller: MEDIATEK Corp. MT7921 802.11ax PCI Express Wireless Network Adapter
0000:3f:00.0 Ethernet controller: Realtek Semiconductor Co., Ltd. RTL8111/8168/8211/8411 PCI Express Gigabit Ethernet Controller (rev 15)
10000:e0:06.0 System peripheral: Intel Corporation RST VMD Managed Controller
10000:e0:06.2 PCI bridge: Intel Corporation Device a73d
10000:e1:00.0 Non-Volatile memory controller: Sandisk Corp WD Black SN770 / PC SN740 256GB / PC SN560 (DRAM-less) NVMe SSD (rev 01)

--- lsusb ---
Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
Bus 003 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 003 Device 002: ID 0408:403d Quanta Computer, Inc. ACER HD User Facing
Bus 003 Device 003: ID 04ca:3802 Lite-On Technology Corp. Wireless_Device
Bus 004 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub

--- lsblk ---
NAME        MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
loop0         7:0    0     4K  1 loop /snap/bare/5
loop1         7:1    0  73.9M  1 loop /snap/core22/2082
loop2         7:2    0  73.9M  1 loop /snap/core22/2111
loop3         7:3    0   516M  1 loop /snap/gnome-42-2204/202
loop4         7:4    0  91.7M  1 loop /snap/gtk-common-themes/1535
loop5         7:5    0  11.1M  1 loop /snap/firmware-updater/167
loop6         7:6    0  10.8M  1 loop /snap/snap-store/1248
loop7         7:7    0  10.8M  1 loop /snap/snap-store/1270
loop8         7:8    0  49.3M  1 loop /snap/snapd/24792
loop9         7:9    0  50.8M  1 loop /snap/snapd/25202
loop10        7:10   0   568K  1 loop /snap/snapd-desktop-integration/253
loop11        7:11   0   576K  1 loop /snap/snapd-desktop-integration/315
nvme0n1     259:0    0 953.9G  0 disk 
├─nvme0n1p1 259:1    0     1G  0 part /boot/efi
└─nvme0n1p2 259:2    0 952.8G  0 part /

--- df -h ---
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           1.6G  2.5M  1.6G   1% /run
/dev/nvme0n1p2  937G   86G  804G  10% /
tmpfs           7.7G  136M  7.6G   2% /dev/shm
tmpfs           5.0M   12K  5.0M   1% /run/lock
efivarfs        268K  192K   71K  73% /sys/firmware/efi/efivars
/dev/nvme0n1p1  1.1G  6.2M  1.1G   1% /boot/efi
tmpfs           1.6G  164K  1.6G   1% /run/user/1000

--- nvidia-smi ---
Mon Sep  1 13:03:45 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4050 ...    On  |   00000000:01:00.0 Off |                  N/A |
| N/A   40C    P8              3W /   35W |      18MiB /   6141MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2638      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
