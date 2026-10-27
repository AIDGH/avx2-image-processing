# AVX2 Image Processing

An assembly programming course project exploring 3×3 image convolution in C and x86-64 AVX2 assembly, followed by shape detection, MNIST digit recognition, and a separate PyTorch brain MRI classification demo.

The native convolution kernel processes eight single-precision values at a time with AVX2/FMA instructions. C code handles image I/O, RGB channel conversion, reference convolution, and application logic.

## Project structure

| Directory | Contents |
| --- | --- |
| `part1/` | RGB image filtering and C versus assembly timing benchmark |
| `part2/simple shape/` | Thresholding, connected components, geometric shape classification, and bounding boxes |
| `part2/train/` | MNIST softmax regression in C, with optional Sobel X/Y and Laplacian features using C or AVX2 |
| `part3/native/` | Native convolution sources and a saved MRI CNN checkpoint |
| `part3/py/` | PyTorch CNN training/inference and an independent NumPy/ctypes convolution wrapper |

Original source files, Persian comments, sample images, example outputs, and saved models are preserved. Compiled programs, object files, shared libraries, editor files, and Python caches are excluded.

## Requirements

- Native assembly examples: **Linux x86-64**, a CPU with **AVX2 and FMA**, and GCC with C11 support.
- Image conversion examples use ImageMagick (`convert`; use `magick` on installations that require it).
- Python examples require Python 3 and the packages in `part3/requirements.txt`. `infer_one.py` uses Pillow's `get_flattened_data()` API, so use a Pillow release that provides it.
- The assembly uses GNU/ELF directives and the System V AMD64 calling convention; it is not ready to build directly on macOS ARM or Windows.

Commands below start from the repository root unless stated otherwise.

## Part 1: image filtering and benchmark

```sh
cd part1
gcc conv_float_rgb.c conv_utils.c conv_avx2.S -O3 -mavx2 -mfma -std=c11 -Wall -Wextra -o program
convert photo.jpg -resize 512x512! -depth 8 -define ppm:format=binary -strip input.ppm
./program input.ppm output.ppm
convert output.ppm output.png
```

The program prints C and assembly timings and their ratio. Edit `k` in `conv_float_rgb.c` to select a filter; commented examples include sharpening, Gaussian blur, Sobel, and Laplacian kernels. The active kernel is a custom example.

The benchmark uses the AVX2 wrapper: assembly handles complete eight-pixel blocks and C computes the remaining interior columns. Both paths process the complete interior; image borders are zeroed by the caller. Timings are measured locally, and no fixed speedup is claimed.

## Part 2: simple shapes

```sh
cd "part2/simple shape"
gcc main.c conv_utils.c conv_avx2.S -O3 -std=c11 -Wall -Wextra -mavx2 -mfma -lm -o detect
./detect square.ppm 128
convert edges.ppm edges.png
convert boxed.ppm boxed.png
```

The threshold is an **integer from 1 to 254**, with a default of 128. The original `run.txt` uses `0.30`, but the current code parses an integer. The detector uses thresholding and geometric heuristics, with labels for circles, triangles, pentagons, squares, rectangles, and diamonds. Its current `main.c` does not call the convolution routines; the native files remain alongside it from the course project.

Example output included in the archive:

![Shape bounding box](part2/simple%20shape/boxed.png)

## Part 2: MNIST digits

Place the four decompressed MNIST IDX files in `part2/train/mnist_idx/`:

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

The original download commands are in `part2/train/run.txt`; ensure the files end up in `mnist_idx/`.

```sh
cd part2/train
gcc main.c conv_utils.c conv_avx2.S -O3 -std=c11 -Wall -Wextra -mavx2 -mfma -lm -o mnist_lr
# Arguments: epochs train_limit learning_rate use_features use_avx2
./mnist_lr 10 60000 0.05 0 0
python3 png_to_raw784.py digit.png my.raw
./mnist_lr predict model.bin my.raw 0 0
```

Training saves `model.bin`. The included model has 784 input features (raw pixels); use `0 0` for its prediction flags. To train with four feature channels instead:

```sh
./mnist_lr 15 60000 0.005 1 1
./mnist_lr predict model.bin my.raw 1 1
```

Training overwrites `model.bin`; keep prediction flags consistent with the model you trained. Dataset files are not bundled. The PNG conversion script needs Pillow and expects a digit with MNIST-compatible foreground/background polarity.

## Part 3: MRI classification

A small CNN classifies resized 128×128 RGB images into `glioma`, `meningioma`, `notumor`, and `pituitary`. Prepare this folder structure for training:

```text
part3/data/
  Training/{glioma,meningioma,notumor,pituitary}/
  Testing/{glioma,meningioma,notumor,pituitary}/
```

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r part3/requirements.txt
cd part3/py
python3 train_mri.py
python3 infer_one.py ../native/mri_cnn_best.pt /path/to/image.jpg
```

The archived checkpoint is included; the MRI dataset is not. Training uses CUDA when available and otherwise CPU. It reports accuracy, a classification report, and a confusion matrix, and saves the checkpoint with the best test accuracy. This is a course demo; no independently verified accuracy results are included. The script selects checkpoints using the test set, so its reported best test accuracy is not an independent held-out evaluation.

The CNN uses PyTorch convolution layers. `convwrap.py` is a separate native convolution interface and is **not integrated into the CNN training or inference path**. To build its library on the supported Linux machine:

```sh
cd part3/native
gcc -O3 -std=c11 -mavx2 -mfma -fPIC -shared conv_utils.c conv_avx2.S -o libconv.so
```

## Archive limitations

The source is preserved as a course submission, including its original `run.txt` notes. Convolution functions leave borders unchanged; callers must initialize them. The automated regression checks compare all four AVX2 wrappers against scalar C over widths 1–42, heights 1–8, and three kernels, covering every eight-pixel remainder and narrow images. Existing model files and images are archived examples; benchmark speedups and model accuracy have not been reproduced as part of preparing this repository.

## Preparation checks

Python source syntax was checked, and the standalone shape detector was compiled and run on the bundled `square.ppm` sample on macOS ARM. This confirms that the detector runs, not the accuracy of every shape label. The Linux AVX2 binaries and MRI training/inference were not executed during repository preparation.
