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
| 1 | Model capture and BN folding | Software foundation | Next |
| 2 | Quantization and integer reference model | Software foundation | Planned |
| 3 | Analytical model, first version | Software foundation | Planned |
| 4 | Fusion pass back end | Software foundation | Planned |
| 5 | PE, 8×8 systolic array, accumulators | Pointwise path | Planned |
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

- **Repository layout.** `sw/` holds the fusion pass, quantization, reference model and analytical model; `rtl/` the SystemVerilog; `tb/` the cocotb tests; `syn/` the Vivado scripts; `results/` the experiment outputs. Proposals and the final report live outside the repository.
- **Simulation host.** All Python and simulation work runs in Ubuntu 26.04 under WSL2. Vivado runs on Windows.
- **Tool versions.** Python 3.14, Verilator 5.050 built from source, cocotb 2.1.0, PyTorch 2.9.0 (CPU). The Python packages are pinned in `requirements.txt`.
- **Test flow.** pytest launches the simulations through cocotb's Python runner, so there are no Makefiles.
- **Quantization scheme.** Weights are int8, symmetric, per output channel. Activations are int8, asymmetric (with a zero-point), per tensor. The input zero-point is folded into the bias, so every MAC stays int8 × int8, and depthwise padding uses the zero-point value. Asymmetric activations let ReLU6 outputs use the full 8-bit range at almost no hardware cost.
- **Calibration data.** About 100 images from Imagenette, a public 10-class subset of ImageNet, kept outside the repository. Accuracy studies are out of scope; the images only set the activation ranges.

## Steps

### Phase 0: Setup

#### Step 0. Environment and repo (done)

- **What:** a repository where pytest runs the Python side and a cocotb + Verilator simulation passes.
- **How:** the commands below reproduce the environment in Ubuntu 26.04 under WSL2.
- **Done when:** torch.fx traces MobileNetV2, and a cocotb test on a small SystemVerilog module passes under Verilator. Both were checked on 14 Nov 2026. Vivado is installed separately on Windows and is first used for the early synthesis pass after Step 7.

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
- **Done when:** the reference model runs MobileNetV2's depthwise and pointwise layers, and its top-1 agreement with the float model has been checked on a few held-out images.

#### Step 3. Analytical model, first version

- **What:** off-chip traffic and on-chip buffer size per layer and per fusion group as a function of tile height, plus group selection under a memory budget.
- **How:** a Python script over the Step 1 layer table, with the group selection as a dynamic program. Run it on MobileNetV2 and ResNet-18.
- **Done when:** unit tests reproduce the proposal's worked example for activation traffic (56×56 block, 24→144→24 channels: 1,956,864 bytes unfused, 1,053,696 with the pair fused, 150,528 with the whole block fused), and the tile height and buffer sizes for the RTL are chosen.

#### Step 4. Fusion pass back end

- **What:** pattern matching, job descriptors, weight images and expected outputs.
- **How:** match depthwise→pointwise pairs, conv–ReLU6 chains and residual additions. Define the job descriptor as a Python dataclass packed to hex. Lay out the weight images in the array's 8×8 tile order. All MobileNetV2 depthwise and pointwise channel counts are multiples of 8, so the pass asserts this instead of padding.
- **Done when:** descriptors, weight images, input images and expected outputs are emitted for an early, a middle and a late MobileNetV2 block.

### Phase 2: Pointwise path

#### Step 5. PE, 8×8 systolic array, accumulators

- **What:** the weight-stationary array with skew and de-skew registers, and the int32 accumulator memory.
- **How:** SystemVerilog, with cocotb tests against numpy matrix multiplication on random data.
- **Done when:** the tests pass for multi-pass accumulation, back-to-back pixel streams and weight reload.

#### Step 6. Post-processing unit

- **What:** bias, ReLU6 or none, requantization, residual addition.
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
- **How:** calibrate the model's re-fetch factors against the measurements, then generate the plots for MobileNetV2 and ResNet-18.
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
