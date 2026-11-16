# Roadmap

Fusing depthwise and pointwise convolutions in hardware: a term project for *Architectures for Deep Learning Acceleration*, Fall 2026.

## Project in brief

The project asks how much off-chip memory traffic hardware fusion of depthwise and pointwise convolutions can remove in depthwise-separable CNNs, and what it costs in on-chip memory and logic. It has two halves:

- **Software.** A fusion pass in Python captures a PyTorch model, folds batch normalization, quantizes to int8, decides which layers to fuse, and emits job descriptors and weight images. An analytical model predicts traffic and buffer sizes and selects fusion groups under a BRAM budget.
- **Hardware.** An accelerator in SystemVerilog with an 8×8 systolic array for pointwise layers, a depthwise engine, and a fusion buffer between them. It runs a depthwise→pointwise pair either fused (the intermediate stays on chip) or unfused (a round trip through DRAM).

The main workload is MobileNetV2 and the target device is a Xilinx Zynq-7020 (XC7Z020). The design is verified in simulation and synthesized; running on a physical board is out of scope. The full proposal is kept outside this repository.

## Status

| Step | Title | Phase | Status |
|---|---|---|---|
| 0 | Environment and repo | Setup | Done |
| 1 | Model capture and BN folding | Software foundation | Done |
| 2 | Quantization and integer reference model | Software foundation | Done |
| 3 | Analytical model, first version | Software foundation | Done |
| 4 | Fusion pass back end | Software foundation | Done |
| 5 | PE, 8×8 systolic array, accumulators | Pointwise path | Next |
| 6 | Post-processing unit | Pointwise path | Planned |
| 7 | Unfused pointwise layers end to end | Pointwise path | Planned |
| 8 | Depthwise engine | Depthwise and fusion | Planned |
| 9 | Fusion buffer | Depthwise and fusion | Planned |
| 10 | Fused depthwise→pointwise mode | Depthwise and fusion | Planned |
| 11 | Block runs and experiments | Evaluation and report | Planned |
| 12 | Vivado synthesis | Evaluation and report | Planned |
| 13 | Model validation and whole-network study | Evaluation and report | Planned |
| 14 | Report and presentation | Evaluation and report | Planned |

## Decisions made

- **Repository layout.** `sw/` holds the fusion pass, quantization, reference model and analytical model; `rtl/` the SystemVerilog; `tb/` the cocotb tests; `syn/` the Vivado scripts; `results/` the experiment outputs. `build/` holds the fusion pass's output and is not committed. Proposals and the final report live outside the repository.
- **Simulation host.** All Python and simulation work runs in Ubuntu 26.04 under WSL2. Vivado runs on Windows.
- **Tool versions.** Python 3.14, Verilator 5.050 built from source, cocotb 2.1.0, PyTorch 2.9.0 (CPU), Vivado 2026.1. The Python packages are pinned in `requirements.txt`.
- **Test flow.** pytest launches the simulations through cocotb's Python runner, so there are no Makefiles.
- **Quantization scheme.** Weights are int8, symmetric, per output channel. Activations are int8, asymmetric (with a zero-point), per tensor. The input zero-point is folded into the bias, so every MAC stays int8 × int8, and depthwise padding uses the zero-point value. Asymmetric activations let ReLU6 outputs use the full 8-bit range at almost no hardware cost.
- **Integer arithmetic.** `sw/reference.py` is the contract that the RTL must match bit for bit. For each output value: sum the int8 × int8 products, add the int32 bias (the result fits in 32 bits), multiply by a per-channel multiplier below 2^15, shift right by a per-channel amount with ties rounded up, add the output zero-point, and saturate to int8. ReLU6 needs no logic of its own: a ReLU6 output has zero-point −128 and a range of at most 6, so the saturation performs it. A residual layer rescales the skip tensor with one multiplier and shift and adds it before the saturation. The stem convolution and the classifier stay in float.
- **Calibration data.** 100 training images from Imagenette (320 px version), a public 10-class subset of ImageNet. The dataset lives in a `data/` folder next to the repository, not inside it; `python -m sw.quantize` downloads it there on first use (about 340 MB). Accuracy studies are out of scope; the images only set the activation ranges.
- **Analytical model.** `sw/analytical.py` counts int8 activations and weights at one byte each, and one 8-byte parameter record (bias, requantization multiplier and shift) per output channel, kept on chip or re-fetched together with the weights. A group is a run of consecutive layers executed one full-width row tile at a time. Line buffers keep the overlapping rows, so no input is read twice. A group's weights are either kept on chip and read once, or re-fetched for every tile. Group selection has three settings: single layers only, depthwise→pointwise pairs as the RTL supports, or any run of layers. Memory budgets are counted in BRAM36 blocks of 4,096 bytes, of which the XC7Z020 has 140.
- **RTL sizing.** The tile height is one row (T = 1): with the weights on chip, traffic does not depend on T, and the buffers grow with it. The buffers are sized for the largest of the three blocks run in RTL (56×56, 14×14 and 7×7): a three-row depthwise line buffer of 24,192 bytes, one row of the skip tensor (1,344 bytes), two fusion-buffer banks of 8,192 bytes each, an output buffer of 8,192 bytes, accumulators for 56 pixels × 8 channels in int32 (1,792 bytes), a weight buffer of 309,760 bytes for the 7×7 block's pointwise layer (307,200 bytes of weights and 2,560 of parameter records), which alone is 76 BRAM36 blocks, and a separate weight memory of 16,320 bytes in the depthwise engine for the same block's depthwise layer (8,640 bytes of weights and 7,680 of parameter records, 4 BRAM36 blocks). In a fused job both engines read their weights at the same time, so they do not share one memory. All depths are RTL parameters, so Step 11 can sweep the tile height.
- **Fusion pass output.** `python -m sw.fusion` writes one folder per block to `build/`: the weight image, the parameter image, the block's input, the expected output of each layer, and the unfused and fused job lists. Everything in DRAM is a 64-bit word and addresses count words; the docstring of `sw/fusion.py` is the reference for the formats. An activation word holds 8 channels of one pixel, ordered by row, pixel and channel group. Pointwise weights are 8×8 tiles in the array's loop order (output group, then input group), one word per array row. Depthwise weights are nine words per group of 8 channels, one per tap. Each output channel has one parameter word: int32 bias, 16-bit multiplier, 8-bit shift. A job descriptor is five words and goes to the controller's job queue, not through DRAM; it has no activation field, because the saturation performs ReLU6. Each image file starts with its DRAM address, so `$readmemh` places it. The tensors come from the first Imagenette validation image.

## Steps

### Phase 0: Setup

#### Step 0. Environment and repo (done)

- **What:** a repository where pytest runs the Python side and a cocotb + Verilator simulation passes.
- **How:** the commands below reproduce the environment in Ubuntu 26.04 under WSL2.
- **Done when:** torch.fx traces MobileNetV2, and a cocotb test on a small SystemVerilog module passes under Verilator. Both were checked on 14 Nov 2026. Vivado is installed separately on Windows and is first used for the early synthesis pass after Step 7; its Zynq-7000 device support still has to be added before then.

```bash
# System packages
sudo apt update
sudo apt install -y build-essential python3-venv python3-dev \
    autoconf flex bison help2man libfl-dev

# Verilator 5.050 from source, installed to ~/.local (no sudo)
git clone https://github.com/verilator/verilator.git ~/src/verilator
cd ~/src/verilator && git checkout v5.050
autoconf && ./configure --prefix="$HOME/.local"
make -j12 && make install
# Open a new terminal afterwards so ~/.local/bin is on the PATH.

# Python environment, kept outside the repository
python3 -m venv ~/.venvs/dwpw
source ~/.venvs/dwpw/bin/activate
pip install -r requirements.txt
```

### Phase 1: Software foundation

#### Step 1. Model capture and BN folding

- **What:** a folded MobileNetV2 graph and a layer table (type, shapes, stride, channels, activation, residual).
- **How:** trace the torchvision model with torch.fx, run shape propagation, and fold each batch normalization into the preceding convolution.
- **Done when:** the folded model's float output matches the original on random input, and the layer table lists every convolution.

#### Step 2. Quantization and integer reference model

- **What:** the bit-exact arithmetic contract that the RTL must match.
- **How:** apply the quantization scheme above, with int32 biases and requantization by a per-channel multiplier and shift under one fixed rounding rule. Define ReLU6 and the residual addition in integer arithmetic. Write the reference model in numpy using integer operations only.
- **Done when:** the reference model runs MobileNetV2's depthwise and pointwise layers, and its top-1 agreement with the float model has been checked on a few held-out images. Checked on 16 Nov 2026 with `python -m sw.quantize`: the two models agree on 90 of 100 held-out validation images.

#### Step 3. Analytical model, first version

- **What:** off-chip traffic and on-chip buffer size per layer and per fusion group as a function of tile height, plus group selection under a memory budget.
- **How:** a Python script over the Step 1 layer table, with the group selection as a dynamic program. Run it on MobileNetV2 and ResNet-18.
- **Done when:** unit tests reproduce the proposal's worked example for activation traffic (56×56 block, 24→144→24 channels: 1,956,864 bytes unfused, 1,053,696 with the pair fused, 150,528 with the whole block fused), and the tile height and buffer sizes for the RTL are chosen. Checked on 16 Nov 2026 with `python -m sw.analytical`: the tests reproduce the three numbers, and with every depthwise→pointwise pair fused the model predicts that MobileNetV2's activation traffic falls from 13.66 MB to 9.06 MB.

#### Step 4. Fusion pass back end

- **What:** pattern matching, job descriptors, weight images and expected outputs.
- **How:** match depthwise→pointwise pairs, conv–ReLU6 chains and residual additions. Define the job descriptor as a Python dataclass packed to hex. Lay out the weight images in the array's 8×8 tile order. All MobileNetV2 depthwise and pointwise channel counts are multiples of 8, so the pass asserts this instead of padding.
- **Done when:** descriptors, weight images, input images and expected outputs are emitted for an early, a middle and a late MobileNetV2 block. Checked on 16 Nov 2026 with `python -m sw.fusion`: each block gets three unfused jobs and two fused ones (the expand layer, then the depthwise→pointwise pair), and the image sizes equal the analytical model's traffic.

### Phase 2: Pointwise path

#### Step 5. PE, 8×8 systolic array, accumulators

- **What:** the weight-stationary array with skew and de-skew registers, and the int32 accumulator memory.
- **How:** SystemVerilog, with cocotb tests against numpy matrix multiplication on random data.
- **Done when:** the tests pass for multi-pass accumulation, back-to-back pixel streams and weight reload.

#### Step 6. Post-processing unit

- **What:** bias, requantization, residual addition, saturation to int8 (which also performs ReLU6).
- **How:** a small pipeline tested against the Step 2 reference functions.
- **Done when:** random and corner-case inputs (saturation, negative rounding) match the reference bit for bit.

#### Step 7. Unfused pointwise layers end to end

- **What:** the controller FSM, address generators, input, weight and output buffers, and a DRAM model with byte counters.
- **How:** a simple valid/ready memory interface, not AXI. The testbench loads the Step 4 images and descriptors.
- **Done when (milestone M1, the minimum deliverable):** real MobileNetV2 pointwise layers are bit-exact against the reference model, and the byte counters match the analytical model.

### Phase 3: Depthwise and fusion

#### Step 8. Depthwise engine

- **What:** a three-row line buffer, 8 lanes of 9 MACs, its own requantization, stride 1 and 2.
- **How:** unit tests first, then an unfused depthwise job run through the controller to DRAM.
- **Done when:** the unfused depthwise job is bit-exact against the reference model.

#### Step 9. Fusion buffer

- **What:** banks of T rows × W pixels × C channels in 64-bit words, write and read address generators, bank swap, mode select.
- **How:** the number of banks is a parameter (1 or 2), which the one-versus-two-banks experiment needs.
- **Done when:** unit tests with random stalls on both sides pass for both bank counts.

#### Step 10. Fused depthwise→pointwise mode

- **What:** the controller runs a pair one row tile at a time with both engines active.
- **How:** extend the FSM, and add the array-input multiplexer and the residual input path.
- **Done when (milestone M2, the target):** the fused output equals the unfused output and the reference, and the byte counters show the intermediate never reached DRAM.

### Phase 4: Evaluation and report

#### Step 11. Block runs and experiments

- **What:** early, middle and late MobileNetV2 blocks, fused and unfused, swept over tile height and bank count.
- **How:** one script per experiment, writing CSV files to `results/`: bytes split into activations and weights, cycles, PE utilization.
- **Done when:** every planned experiment can be regenerated from its script.

#### Step 12. Vivado synthesis

- **What:** LUTs, flip-flops, DSP slices, BRAM and clock frequency for the XC7Z020, with and without the fusion hardware.
- **How:** an out-of-context Tcl batch script. A first pass runs right after Step 7 to catch BRAM and DSP inference problems early.
- **Done when:** the resource and timing table for both configurations is in `results/`.

#### Step 13. Model validation and whole-network study

- **What:** the analytical model's error against the RTL byte counters, fusion groups under several BRAM budgets, pair versus whole-block fusion, and the equal-memory unfused baseline.
- **How:** calibrate the model's re-fetch factors against the measurements, then generate the plots for MobileNetV2 and ResNet-18. Also check the model's skip-tensor buffer rule (one tile per bypassed layer), which is an estimate, against the RTL.
- **Done when:** the model error is reported and the whole-network figures are generated.

#### Step 14. Report and presentation

- **What:** the final report and slides.
- **How:** all figures are regenerated from the scripts.
- **Done when:** both are submitted.

## Fallbacks

- **Depthwise engine or fusion buffer runs late.** The systolic array runs the depthwise and pointwise tiles in turn through the fusion buffer. This is slower, but the intermediate still stays on chip, so the off-chip traffic result is unchanged.
- **Hardware falls behind overall.** Steps 1–7 already form the minimum deliverable: the fusion pass, the analytical results for both networks, and unfused pointwise layers bit-exact in simulation.

## Stretch goals

Only if time is left after Step 13:

- Fusion of the whole inverted residual block (expand→depthwise→project) in RTL.
- im2col support, so the array can also run standard convolutions.
