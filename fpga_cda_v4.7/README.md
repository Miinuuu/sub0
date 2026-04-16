# CDA-FPGA v4.7 Evaluation

This repository contains the evaluation binary and minimal host code for the Compressed-Domain Attention (CDA) FPGA acceleration (version 4.7) presented in our paper. 

To maintain blind review and protect proprietary RTL implementations of the novel hardware architecture, we provide pre-compiled FPGA bitstreams (`.xclbin`) and minimal host execution codes (C++) for functional and performance reproduction.

## Requirements
- Xilinx Alveo U200 FPGA
- Xilinx Runtime (XRT) installed and sourced (e.g. `/opt/xilinx/xrt/setup.sh`)
- `g++` supporting C++17

## Download 
- https://drive.google.com/drive/folders/1qx60cXTm7nw-Ek9C5pOsrzwgLk4ZTpKq?usp=sharing
- weights.bin
- tokenizer.bin
- forwar.xclbin


## Build Host Code

Compile the minimal host driver:
```bash
make
```

## Run Evaluation

Run the generated inference executable to verify the E2E latency and attention generation process:
```bash
make run
```
Or execute it manually:
```bash
source /opt/xilinx/xrt/setup.sh
./run_cda_fpga weights.bin -k forward.xclbin -z tokenizer.bin -n 10 -i "Hello"
```

## Files Provided
- `forward.xclbin`: Pre-compiled Alveo U200 bitstream for CDA v4.7 attention.
- `weights.bin` / `tokenizer.bin`: Model weights and vocabulary.
- `src/`: Minimal host C++ headers and driver logic to setup XRT buffers and launch the `forward` kernel.
