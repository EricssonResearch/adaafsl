# AdaAFSL: Adaptive Federated Split Learning

A framework for network-aware distributed machine learning that combines **Split Learning (SL)** with **Federated Learning (FL)** and adapts communication overhead based on real-time network conditions. The system monitors link utilization via SNMP and dynamically reduces the volume of intermediate activations and gradients exchanged between clients and the server using stochastic masking and quantization.

## What It Does

AFSL addresses the communication bottleneck in distributed learning over bandwidth-constrained networks (e.g., edge/IoT deployments). It:

1. **Splits a neural network** between clients (early layers) and a server (later layers), so clients only transmit intermediate representations rather than raw data or full model updates.
2. **Monitors network utilization** in real time using SNMP polling of switch interfaces to detect congestion.
3. **Adaptively compresses** the exchanged activations and gradients using Bernoulli masking and FP16 quantization, with the compression ratio driven by current link utilization.
4. **Aggregates models asynchronously** on the server using a weighted averaging scheme (exponentially decaying client contribution), enabling clients with different computational speeds to participate without synchronization barriers.

## How It Works

### Architecture

```
┌──────────────┐         Socket (TCP)         ┌──────────────────────┐
│   Client(s)  │ ───── activations/grads ────▶ │       Server         │
│  (c_LeNet /  │ ◀──── gradients/model ─────── │  (s_LeNet / s_ResNet │
│  c_ResNet)   │                               │   + aggregation)     │
└──────────────┘                               └──────────────────────┘
       │                                                │
       │              SNMP Monitoring                    │
       └──────────────── Switch ────────────────────────┘
```

- **Client side**: Runs the first few layers of the model, encodes intermediate activations (with adaptive masking), and sends them to the server. Receives compressed gradients back for local backpropagation.
- **Server side**: Runs the remaining layers, computes the loss, performs backpropagation on its portion, and returns gradients to the client. Periodically aggregates model weights from all connected clients.
- **Network monitor**: Continuously polls SNMP counters on the network switch to compute per-link utilization. This drives the reduction factor `p` (probability of zeroing out activations).

### Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| **AFSL**  | Adaptive Federated Split Learning (network-aware compression) |
| **FSL**   | Federated Split Learning (fixed split, no adaptive compression) |
| **SL**    | Standard Split Learning |
| **PSL**   | Parallel Split Learning |

### Supported Datasets

- **MNIST** — Handwritten digit classification
- **CIFAR-100** — 100-class image classification (uses ResNet)
- **5G** — 5G network traffic classification
- **IOT** — IoT device traffic regression/classification
- **VOD** — Video-on-Demand QoE prediction

### Network Stress Testing

The `Stresser/` module uses `iperf3` to generate controlled background traffic on the network, simulating congestion scenarios for evaluation purposes.

## Project Structure

```
├── server.py                  # Entry point to launch the server
├── server_threshold.py        # Server launcher for threshold-finding experiments
├── eval_server.py             # Entry point for evaluation
├── requirements.txt           # Python dependencies
├── AFSL/                      # Server-side algorithm implementations
│   ├── server_model_AFSL_*.py # AFSL server for each dataset
│   ├── server_model_FSL_*.py  # FSL server for each dataset
│   ├── server_model_SL_*.py   # SL server for each dataset
│   ├── models.py              # Neural network architectures (LeNet, ResNet, MLP, CNN)
│   ├── utils.py               # Communication, encoding/decoding, SNMP utilities
│   ├── args.py                # Configuration and hyperparameters
│   └── test.py                # Test/evaluation functions
├── Client/                    # Client-side implementations
│   ├── client_model_*.py      # Client training scripts per dataset
│   ├── models.py              # Client-side model architectures
│   ├── utils.py               # Client communication utilities
│   ├── preprocess_data.py     # Data loading and preprocessing
│   └── args.py                # Client configuration
├── Stresser/                  # Network stress-testing tools (iperf3-based)
├── RantopK/                   # RandTopK sparsification baseline
├── RTT/                       # Round-trip time measurement utilities
└── batch_scripts/             # PowerShell scripts for experiment orchestration
```

## Installation

### Prerequisites

- Python 3.8+
- A network environment with SNMP-enabled switches (for network monitoring)
- `iperf3` (for stress testing experiments)
- `scapy` — used as an external process for packet capture but not provided by this repository; install separately (`pip install scapy`)
- CUDA-capable GPU (optional, falls back to CPU)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd distributedml

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit the configuration files to match your network topology:

- `AFSL/args.py` — Server-side settings (IP addresses, switch ports, SNMP credentials, hyperparameters)
- `Client/args.py` — Client-side settings

Key parameters in `AFSL/args.py`:
- `switch_port`: Mapping of device IPs to switch interface indices
- `snmp_host` / `snmp_community`: SNMP agent address and community string
- `reduction_threshold`: Per-dataset maximum compression ratio
- `version`: `0` = no compression, `1` = uplink only, `2` = uplink + downlink

## Usage

### Running the Server

```bash
python3 server.py
```

The server script launches the selected algorithm (configured inside `server.py`):
- `algorithm`: one of `AFSL`, `FSL`, `PSL`, `SL`
- `dataset`: one of `MNIST`, `CIFAR100`, `5G`, `IOT`, `VOD`

### Running a Client

On each client device:

```bash
python3 Client/client_model_CIFAR100.py <server_ip> <num_local_clients> <local_id> <delay> <seed> <batch_size> <port> <input_size> <start_delay>
```

### Finding Compression Thresholds

```bash
python3 server_threshold.py <probability>
```

This runs the training with a constant reduction probability to determine the accuracy-compression trade-off.

### Evaluation

```bash
python3 eval_server.py
```

## Results

Experiment results are saved under `results/Experiment_<id>/` with the following structure:
- `accuracy/` — Per-epoch accuracy and loss
- `data_transfer/` — Bytes sent/received per client, reduction factors
- `Network_Monitoring/` — Per-interface utilization logs
- `resources/` — Model sizes, aggregation contributions
- `epoch_time/` — Training timing breakdowns

## Citation

If you use this code in your research, please cite:

```bibtex
@article{author2025afsl,
  title     = {AFSL: Adaptive Federated Split Learning with Network-Aware Communication Reduction},
  author    = {TODO: Author Names},
  journal   = {TODO: Journal/Conference Name},
  year      = {TODO: Year},
  volume    = {TODO},
  pages     = {TODO},
  doi       = {TODO}
}
```

## License

TODO: Specify license.
