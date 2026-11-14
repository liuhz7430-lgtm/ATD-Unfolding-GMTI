# ATD-Unfolding-GMTI

Reference implementation for **Joint Spatial and Temporal Unfolding for Moving-Target State Estimation and Imaging in Multichannel SAR** by Haozong Liu, College of Future Information Technology, Fudan University.

[Read the manuscript (PDF)](manuscript.pdf)

The model estimates the two-dimensional ground position and velocity of one moving point target from three ideally registered along-track SAR channels. It combines unfolded static/moving echo separation, coherent per-frame state fitting, and sliding-window refinement. The estimated velocity is used to refocus the target with local backprojection. The temporal refiner is offline and can use later frames.

## Setup

The paper configuration was run on Linux with Python 3.10 and three NVIDIA RTX 3090 GPUs. The CUDA installation below matches the PyTorch 2.5.1 / CUDA 11.8 environment used for those runs.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

## Evaluate the released checkpoint

The [epoch-20 checkpoint](checkpoints/best.pt) is included at the default path used by `test.py`. It contains the trained model used for the reported experiments. The public copy replaces an internal path in the training metadata and preserves the tensor values. Its SHA-256 is `a5b82138f68af27d519130a8775ee4dc9a6950d6f106a57be4b8210855dee81b`.

Both commands default to four independent trajectories, SCRs of −5, 0, and 10 dB, and a 15 dB target-to-noise ratio.

```bash
python test.py --gpus 0,1,2 --dataset simulated
python test.py --gpus 0,1,2 --dataset semireal
```

For a single GPU, use `--gpus 0`. For the MiniSAR-derived experiment, first follow the [data setup instructions](real_scene/README.md). It derives the background from a measured complex image and generates echoes, target motion, and noise under the paper's acquisition model.

Results go into `runs/` by default and include mean Euclidean errors, state predictions, and figures. The [reference CSVs](results) contain the values used for the manuscript. They compare DPCA/ATI + Gauss–Newton, untrained unfolding, per-frame outputs from the trained model, and the final jointly trained model. The three SCRs reuse the same four independent trajectories within each background type. At 0 dB, joint unfolding yields mean position / velocity errors of 0.582 m / 0.378 m/s with simulated clutter and 0.551 m / 0.300 m/s with MiniSAR-derived clutter.

## Train and check

Use the same training directory and GPU count when resuming. The two commands mirror the initial 12 epochs and continuation to 20.

```bash
python train.py --gpus 0,1,2 --epochs 12 --run-dir runs/paper-training
python train.py --gpus 0,1,2 --epochs 20 --run-dir runs/paper-training --resume
python test.py --gpus 0,1,2 --checkpoint runs/paper-training/checkpoints/best.pt \
  --dataset simulated
```

Run the automated tests with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m pytest tests -q
```

- `gmti/` contains the signal model, estimators, data generation, and reporting.
- `configs/current.yaml` specifies the acquisition and model configuration.
- `tests/` contains numerical and pipeline tests.

The project code and documentation are available under the [MIT License](LICENSE).
