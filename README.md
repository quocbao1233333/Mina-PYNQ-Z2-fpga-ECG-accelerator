# MINA PYNQ-Z2 FPGA ECG Accelerator

> RTL-based educational FPGA implementation of a MINA-style 1-D CNN accelerator for ECG beat classification on the **PYNQ-Z2** board.

This repository documents and implements a student/research version of a **Mini-InceptionNet Accelerator (MINA)-style ECG classifier** on FPGA. The project is inspired by the paper:

**“MINA: A Hardware-Efficient and Flexible Mini-InceptionNet Accelerator for ECG Classification in Wearable Devices”**  
Hoai Luan Pham, Thi Diem Tran, Vu Trung Duong Le, Yasuhiko Nakashima, IEEE TCAS-I, 2024.

The original paper proposes a lightweight **1-D CNN model** named **Mini InceptionNet** and a hardware accelerator architecture named **MINA** for ECG beat classification in wearable devices. This project rebuilds the core idea on **PYNQ-Z2 / Zynq-7000** using RTL Verilog, Vivado Block Design, AXI BRAM, AXI GPIO, and Jupyter/PYNQ MMIO testing.

---

## 1. Project Status

### Current implemented status

The current hardware project has successfully reached the following stage:

- Vivado project rebuilt successfully.
- `mina_system_wrapper.bit` and `mina_system.hwh` generated.
- Files renamed for PYNQ:
  - `mina.bit`
  - `mina.hwh`
- Overlay loaded successfully on PYNQ-Z2 through Jupyter.
- AXI GPIO control/status interface works.
- AXI BRAM bias/weight write-read test works.
- `mina_top` can be started from Jupyter.
- Hardware reaches final layer state.
- Zero-feature hardware test returns:

```text
predicted_class = 3
best_logit      = 0x00001000
```

### Important technical note

The current hardware test is a **zero-feature hardware verification test**, not yet full ECG inference.

In the current test:

```text
feature input = 0
weight        = 0
bias class 3  = 1.0 in Q4.12 fixed-point = 0x00001000
```

Therefore the final logits are expected to be:

```text
logit[0] = 0
logit[1] = 0
logit[2] = 0
logit[3] = 1.0
logit[4] = 0
```

This forces:

```text
predicted_class = 3
```

This confirms that the FPGA control path, BRAM access, bias read path, linear/argmax output path, and Jupyter/PYNQ MMIO interface are working. It does **not yet prove full real ECG classification**, because the shared feature memory and real trained weight/bias loading flow still need to be completed.

---

## 2. Background: Why MINA?

ECG classification is important for wearable healthcare devices. Traditional hospital ECG systems can provide high-quality signals, but they are not convenient for continuous long-term monitoring. Wearable ECG devices need compact, fast, and low-power hardware.

The paper argues that ECG signals are low-frequency time-series signals. Instead of using a heavy 2-D CNN that requires transforming ECG into images such as spectrograms or scalograms, a **1-D CNN** can directly process the ECG waveform.

### Main reasons for using 1-D CNN

- ECG is naturally a 1-D time-domain signal.
- No image conversion is required.
- Hardware is smaller than 2-D CNN accelerators.
- Memory requirement is lower.
- It is more suitable for wearable devices.
- It can still achieve high ECG beat classification accuracy.

---

## 3. ECG Input Format

The target ECG task is **beat classification**.

Each ECG beat is extracted around the **R-peak**:

```text
159 samples before R-peak
1 sample at R-peak
160 samples after R-peak
```

Therefore, each input beat has:

```text
159 + 1 + 160 = 320 samples
```

The input tensor shape is:

```text
ECG input = 320 × 1
```

This means:

```text
X = [x0, x1, x2, ..., x319]
```

Each `xi` is one ECG amplitude sample. In the hardware/CNN context, the word **pixel** means one ECG sample or one intermediate feature-map value. It does not mean an image pixel.

---

## 4. ECG Classes

The project follows the five-class ECG beat classification setup:

| Class ID | Label | Meaning |
|---:|---|---|
| 0 | NOR | Normal beat |
| 1 | LBBB | Left Bundle Branch Block beat |
| 2 | RBBB | Right Bundle Branch Block beat |
| 3 | PVC | Premature Ventricular Contraction |
| 4 | APB | Atrial Premature Beat |

In the current zero-feature hardware test:

```text
predicted_class = 3
```

means the selected class is:

```text
PVC = Premature Ventricular Contraction
```

Again, in the current test this is produced by manually setting the class-3 bias, not by real ECG inference yet.

---

## 5. Mini InceptionNet Model Overview

The model is a lightweight 1-D CNN designed for ECG beat classification.

The high-level architecture used in this project is:

```text
Input ECG 320×1
→ Conv1D J=7, stride=2, 8 channels
→ Inception Block
→ Inception Block
→ Residual Add

→ Conv1D J=5, stride=2, 16 channels
→ Inception Block
→ Inception Block
→ Residual Add

→ Conv1D J=3, stride=2, 32 channels
→ Inception Block
→ Inception Block
→ Residual Add

→ Global Average Pooling
→ Linear / Dense 5 classes
→ predicted_class
```

### Why Inception Block?

An Inception Block uses several convolution branches with different kernel sizes:

```text
J = 1
J = 3
J = 5
J = 7
```

This allows the network to extract ECG features at multiple time scales:

| Kernel | Interpretation in ECG |
|---:|---|
| J=1 | Very local feature / channel mixing |
| J=3 | Short local pattern |
| J=5 | Medium ECG morphology |
| J=7 | Wider waveform context around QRS/T wave |

This is useful because ECG abnormalities may appear as changes in waveform width, slope, peak shape, and local morphology.

---

## 6. Convolution Rule Used in This Project

The 1-D convolution formula is:

```text
Z[n, y] = Σ_k Σ_j W[n, k, j] × X[k, y × stride + j] + b[n]
```

Where:

| Symbol | Meaning |
|---|---|
| `X` | input ECG or feature map |
| `W` | convolution weight |
| `b` | bias |
| `n` | output channel index |
| `k` | input channel index |
| `j` | kernel position index |
| `y` | output position index |
| `stride` | kernel sliding step |
| `Z` | output feature map |

### Simple numeric example

Input ECG segment:

```text
X = [2, 1, 3, 4, 6, 5]
```

Kernel:

```text
W = [1, 0, -1]
```

Bias:

```text
b = 0
```

At output position 0:

```text
Z[0] = 2×1 + 1×0 + 3×(-1)
     = 2 + 0 - 3
     = -1
```

At output position 1:

```text
Z[1] = 1×1 + 3×0 + 4×(-1)
     = 1 + 0 - 4
     = -3
```

So convolution means:

1. Select an input window.
2. Multiply each input value by its corresponding weight.
3. Accumulate the products.
4. Add bias.
5. Slide the window according to stride.
6. Repeat until the end of the signal.

In hardware, this operation is implemented by a **MAC** unit:

```text
MAC = Multiply–Accumulate
```

---

## 7. Fixed-Point Format

The project uses fixed-point arithmetic for hardware-friendly computation.

Current test convention:

```text
Q4.12
```

In Q4.12:

```text
1.0 = 4096 = 0x00001000
```

Example:

```text
0.5  = 0x0800
0.25 = 0x0400
```

When multiplying two Q4.12 values:

```text
Q4.12 × Q4.12 = Q8.24
```

To return to Q4.12, the product must be shifted right by 12 bits:

```text
product_q4_12 = product_q8_24 >> 12
```

This scaling rule is important for Verilog RTL modules such as:

```text
mina_mac.v
mina_alu.v
mina_linear_layer.v
```

---

## 8. RTL Module List

The current RTL hierarchy includes the following main modules:

```text
mina_top.v
mina_controller.v
mina_context_rom.v
mina_conv1d_layer.v
mina_inception_block.v
mina_maxpool_layer.v
mina_residual_add.v
mina_global_pool.v
mina_linear_layer.v

mina_pea.v
mina_pe.v
mina_alu.v
mina_mac.v
mina_relu.v
mina_maxpool.v
mina_sba.v
mina_ldm.v

mina_weight_reader.v
mina_bias_reader.v
mina_output_writer.v
```

### Module roles

| Module | Role |
|---|---|
| `mina_top.v` | Top-level accelerator module |
| `mina_controller.v` | Controls layer sequence and accelerator state |
| `mina_context_rom.v` | Stores layer configuration context |
| `mina_conv1d_layer.v` | 1-D convolution layer |
| `mina_inception_block.v` | Inception-style multi-branch block |
| `mina_residual_add.v` | Residual addition |
| `mina_global_pool.v` | Global average pooling |
| `mina_linear_layer.v` | Final dense/linear classification |
| `mina_mac.v` | Multiply-accumulate unit |
| `mina_relu.v` | ReLU activation |
| `mina_maxpool.v` | Max pooling primitive |
| `mina_alu.v` | Selects MAC/ReLU/MaxPool/Add-type operations |
| `mina_pe.v` | Processing element |
| `mina_pea.v` | Processing element array |
| `mina_sba.v` | Sharing buffer allocator |
| `mina_ldm.v` | Local data memory |
| `mina_weight_reader.v` | Weight memory read helper |
| `mina_bias_reader.v` | Bias memory read helper |
| `mina_output_writer.v` | Output write helper |

---

## 9. Controller Layer ID Mapping

The current controller is organized with `debug_layer_id` from 0 to 13:

| Layer ID | Function |
|---:|---|
| 0 | CONV1D_0 |
| 1 | INCEPTION_1 |
| 2 | INCEPTION_2 |
| 3 | RESIDUAL_ADD_0 |
| 4 | CONV1D_1 |
| 5 | INCEPTION_3 |
| 6 | INCEPTION_4 |
| 7 | RESIDUAL_ADD_1 |
| 8 | CONV1D_2 |
| 9 | INCEPTION_5 |
| 10 | INCEPTION_6 |
| 11 | RESIDUAL_ADD_2 |
| 12 | GLOBAL_POOL |
| 13 | LINEAR |

In the current Jupyter test, the hardware reaches:

```text
layer = 13
state = 0
busy  = 0
class = 3
```

This indicates that the accelerator reached the final linear layer and returned to idle state.

---

## 10. Vivado Block Design Overview

The PYNQ-Z2 system uses the Zynq Processing System and AXI-connected peripheral blocks.

Current working overlay includes:

```text
processing_system7_0
smartconnect_0
proc_sys_reset_0

axi_gpio_ctrl
axi_gpio_status
axi_gpio_best_logit

axi_bram_ctrl_bias_0
axi_bram_ctrl_weight_0
axi_bram_ctrl_input_0
axi_bram_ctrl_output_0

blk_mem_gen_bias_0
blk_mem_gen_weight_0
blk_mem_gen_input_0
blk_mem_gen_output_0

mina_top_0
```

### AXI Memory Map

| Peripheral | Base Address | Range | Purpose |
|---|---:|---:|---|
| `axi_bram_ctrl_bias_0` | `0x40000000` | `0x2000` | Bias memory |
| `axi_bram_ctrl_input_0` | `0x42000000` | `0x2000` | Input/feature memory candidate |
| `axi_bram_ctrl_output_0` | `0x44000000` | `0x2000` | Output memory candidate |
| `axi_bram_ctrl_weight_0` | `0x46000000` | `0x2000` | Weight memory |
| `axi_gpio_ctrl` | `0x41200000` | `0x10000` | Start / soft reset control |
| `axi_gpio_status` | `0x41210000` | `0x10000` | Status and predicted class |
| `axi_gpio_best_logit` | `0x41220000` | `0x10000` | Best logit output |

---

## 11. AXI GPIO Control and Status

### Control GPIO

`axi_gpio_ctrl` is used to send control signals from Jupyter/Python to the FPGA accelerator.

| Bit | Signal | Meaning |
|---:|---|---|
| 0 | `start` | Start accelerator |
| 1 | `soft_reset` | Software reset |

Example:

```python
ctrl.write(0x00, 0b01)  # start pulse
ctrl.write(0x00, 0b10)  # soft reset pulse
```

### Status GPIO

`axi_gpio_status` returns accelerator state to Jupyter.

| Bit range | Signal |
|---:|---|
| bit 0 | `busy` |
| bit 1 | `done` |
| bit 2 | `error` |
| bit 3 | `result_valid` |
| bits 6:4 | `predicted_class` |
| bits 11:7 | `debug_layer_id` |
| bits 15:12 | `debug_controller_state` |

Current note:

```text
done/result_valid may be short pulses.
Python polling can miss them.
```

Therefore, the Jupyter test also accepts this final-state condition:

```text
busy = 0
debug_layer_id = 13
debug_controller_state = 0
predicted_class = 3
```

---

## 12. Data Flow in Current Hardware Test

Current test flow:

```text
Jupyter
  ↓
Load mina.bit / mina.hwh
  ↓
Map AXI GPIO and AXI BRAM using MMIO
  ↓
Clear bias BRAM and weight BRAM
  ↓
Write bias class 3 = 0x00001000
  ↓
Send soft_reset
  ↓
Send start
  ↓
mina_top runs through layer sequence
  ↓
Jupyter reads predicted_class and best_logit
```

Expected output:

```text
TEST PASS ON FPGA
predicted_class = 3
best_logit      = 0x1000
```

---

## 13. Jupyter Test Code Summary

The Jupyter notebook performs the following steps:

1. Check `mina.bit` and `mina.hwh`.
2. Load overlay using `Overlay("mina.bit")`.
3. Print detected IPs from `mina.hwh`.
4. Map MMIO regions.
5. Configure GPIO directions.
6. Clear bias and weight BRAM.
7. Write linear bias class 3.
8. Send soft reset.
9. Send start pulse.
10. Poll status.
11. Read `best_logit`.
12. Print PASS/FAIL.

Core overlay load:

```python
from pynq import Overlay, MMIO

ol = Overlay("mina.bit")
ol.download()
```

Core test condition:

```python
expected_class = 3
expected_best_logit = 0x00001000
```

---

## 14. Training Script

The repository includes a Python training script:

```text
mina_ecg_train.py
```

The script follows the paper's ECG beat extraction idea:

```text
159 samples before R-peak
1 sample at R-peak
160 samples after R-peak
Total = 320 samples
```

It maps MIT-BIH annotation labels:

| MIT-BIH Symbol | Class ID | Label |
|---|---:|---|
| `N` | 0 | NOR |
| `L` | 1 | LBBB |
| `R` | 2 | RBBB |
| `V` | 3 | PVC |
| `A` | 4 | APB |

The script builds a Mini InceptionNet-like 1-D CNN:

```text
Input 320×1
Conv1D J=7 stride=2 channels=8
2× Inception + Residual
Conv1D J=5 stride=2 channels=16
2× Inception + Residual
Conv1D J=3 stride=2 channels=32
2× Inception + Residual
GlobalAveragePooling1D
Dense 5 classes
```

Training setup in the script:

```text
Batch size = 20
Initial learning rate = 1e-4
Reduced learning rate = 1e-5
Optimizer = Adam
Loss = sparse categorical crossentropy
```

The script saves:

```text
data/processed/*.npy
models/best_mina_mini_inceptionnet.keras
models/mina_mini_inceptionnet.h5
models/weights_biases.npz
results/training_curve.png
results/confusion_matrix.png
results/classification_report.txt
results/metrics_ACC_SEN_SPEC_PPV.csv
```

---

## 15. Suggested Repository Structure

Recommended GitHub structure:

```text
mina-pynqz2-fpga-ecg-accelerator/
├── README.md
├── LICENSE
├── docs/
│   ├── main.pdf
│   ├── convolution_rules.md
│   ├── tcl_axi_bram_analysis.md
│   └── jupyter_code_analysis.md
├── rtl/
│   ├── mina_top.v
│   ├── mina_controller.v
│   ├── mina_context_rom.v
│   ├── mina_conv1d_layer.v
│   ├── mina_inception_block.v
│   ├── mina_residual_add.v
│   ├── mina_global_pool.v
│   ├── mina_linear_layer.v
│   ├── mina_mac.v
│   ├── mina_relu.v
│   ├── mina_maxpool.v
│   ├── mina_alu.v
│   ├── mina_pe.v
│   ├── mina_pea.v
│   ├── mina_sba.v
│   ├── mina_ldm.v
│   ├── mina_weight_reader.v
│   ├── mina_bias_reader.v
│   └── mina_output_writer.v
├── tcl/
│   └── mina_system_bd.tcl
├── notebooks/
│   └── test_mina_fpga.ipynb
├── python/
│   └── mina_ecg_train.py
├── export/
│   ├── mina.bit
│   └── mina.hwh
└── dataset/
    └── MIT-BIH files are not included by default
```

Recommended note:

```text
Large datasets, generated Vivado cache folders, and build outputs should not be committed unless required.
```

---

## 16. How to Rebuild the Vivado Project

Example Tcl build flow:

```tcl
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

source C:/VIVADO/MINA_REBUILD/tcl/mina_system_bd.tcl

open_bd_design C:/VIVADO/MINA_REBUILD/MINA_REBUILD.srcs/sources_1/bd/mina_system/mina_system.bd

validate_bd_design -force
save_bd_design

generate_target all [get_files C:/VIVADO/MINA_REBUILD/MINA_REBUILD.srcs/sources_1/bd/mina_system/mina_system.bd]

make_wrapper -files [get_files C:/VIVADO/MINA_REBUILD/MINA_REBUILD.srcs/sources_1/bd/mina_system/mina_system.bd] -top

add_files -norecurse C:/VIVADO/MINA_REBUILD/MINA_REBUILD.gen/sources_1/bd/mina_system/hdl/mina_system_wrapper.v

set_property top mina_system_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
```

Copy output files:

```tcl
file mkdir C:/VIVADO/MINA_REBUILD/export

file copy -force C:/VIVADO/MINA_REBUILD/MINA_REBUILD.runs/impl_1/mina_system_wrapper.bit C:/VIVADO/MINA_REBUILD/export/mina.bit

file copy -force C:/VIVADO/MINA_REBUILD/MINA_REBUILD.gen/sources_1/bd/mina_system/hw_handoff/mina_system.hwh C:/VIVADO/MINA_REBUILD/export/mina.hwh
```

---

## 17. How to Run on PYNQ-Z2

1. Connect PC directly to PYNQ-Z2 through Ethernet.
2. Set PC Ethernet static IP:

```text
IP address: 192.168.2.1
Subnet mask: 255.255.255.0
Gateway: empty
```

3. Open Jupyter:

```text
http://192.168.2.99
```

4. Create folder:

```text
/home/xilinx/jupyter_notebooks/mina_overlay/
```

5. Upload:

```text
mina.bit
mina.hwh
test_mina_fpga.ipynb
```

6. Run the first overlay test:

```python
from pynq import Overlay

ol = Overlay("mina.bit")
ol.download()

print("Overlay loaded OK")
print(ol.ip_dict.keys())
```

---

## 18. Current Limitations

This repository is not yet a complete reproduction of the full paper accelerator.

Current limitations:

- Real ECG data path is not fully integrated.
- Shared feature memory is not completed.
- Real trained weights and biases are not fully mapped into hardware BRAM.
- Current memory ranges are small for full model execution.
- `done` and `result_valid` may be short pulses and should be latched for robust PS polling.
- Current PYNQ test verifies zero-feature control/data path only.

---

## 19. Future Work

Planned improvements:

- Add shared feature memory wrapper.
- Support real ECG input loading into feature memory.
- Export trained weights and biases into fixed-point BRAM binary format.
- Implement full end-to-end ECG inference.
- Compare FPGA inference with TensorFlow software inference.
- Add complete testbench with real ECG samples.
- Improve `done` and `result_valid` latching.
- Optimize memory layout for `FM_IN_ECG`, `FM_A`, `FM_B`, `FM_C`, `FM_D`, `FM_GAP`, and `FM_CLASS`.
- Add performance/resource utilization report for PYNQ-Z2.
- Add diagrams for data flow, AXI map, and hardware hierarchy.

---

## 20. Reference Paper Key Numbers

Important numbers from the reference MINA paper:

| Item | Value |
|---|---:|
| Input beat length | 320 samples |
| Classes | NOR, LBBB, RBBB, PVC, APB |
| Model type | 1-D CNN / Mini InceptionNet |
| Parameters | 6,457 |
| Parameter reduction | 41.6% |
| Accuracy | 99.37% |
| Sensitivity | 99.37% |
| Specificity | 98.83% |
| PPV | 99.38% |
| FPGA platform in paper | ZCU102 / ZC706 |
| Selected PE count | 40 PEs |
| Frequency | 250 MHz on ZCU102 |
| Inference time | 34.45 µs on ZCU102 |
| Power | 0.827 W |
| Precision | 16-bit fixed-point |

---

## 21. Disclaimer

This repository is an educational and research-oriented FPGA implementation study inspired by the MINA paper. It is not an official release from the original authors. The current PYNQ-Z2 implementation is focused on rebuilding, understanding, and verifying the hardware data/control path before completing real ECG inference.

For medical use, this project is not a certified diagnostic system.

---

## 22. Suggested Citation

If this repository helps your study, please cite the original paper:

```bibtex
@article{pham2024mina,
  title={MINA: A Hardware-Efficient and Flexible Mini-InceptionNet Accelerator for ECG Classification in Wearable Devices},
  author={Pham, Hoai Luan and Tran, Thi Diem and Le, Vu Trung Duong and Nakashima, Yasuhiko},
  journal={IEEE Transactions on Circuits and Systems I: Regular Papers},
  year={2024}
}
```

---

## 23. Short Project Description

```text
RTL-based MINA-style 1-D CNN ECG accelerator on PYNQ-Z2 FPGA, with Vivado block design, AXI BRAM/GPIO interfaces, and Jupyter MMIO hardware testing.
```
