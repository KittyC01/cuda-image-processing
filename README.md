**GPU-Accelerated Image Processing with CUDA
Overview**
This project demonstrates GPU-accelerated image processing using CUDA C++.
The program generates and processes a large collection of small grayscale images. A CUDA kernel applies a 3x3 box blur filter to each image.

The default configuration processes 100 images at 256x256 pixels. The program was also tested with 500 images.

The project was developed and executed in Google Colab using an NVIDIA Tesla T4 GPU.

**Requirements**
-  NVIDIA GPU
- CUDA Toolkit
- nvcc
- GNU Make
- Build
- Compile the program using:
- make

**Run**
Run the project using:
./run.sh

The executable also accepts command-line arguments:
./image_filter [image_count] [width] [height] [output_directory]

For example:
./image_filter 500 256 256 output

**CUDA Algorithm**
The program uses a CUDA kernel called **BlurKernel**.
Each CUDA thread processes one image pixel. The kernel reads the surrounding 3x3 pixel neighborhood and calculates the average intensity.

A 16x16 CUDA thread block is used. For a 256x256 image, the program uses a 16x16 grid.

**GPU Processing**
For each image:
- An input image is generated.
- The image is copied from CPU memory to GPU memory.
- The CUDA blur kernel is executed.
- GPU threads process the pixels in parallel.
- The result is copied back to CPU memory.
- The result is saved for verification.

The implementation uses cudaMalloc, cudaMemcpy, CUDA kernel execution, cudaDeviceSynchronize, and cudaFree.
Dataset
The project generates reproducible grayscale test images programmatically. This allows hundreds of small inputs to be processed without requiring a large external dataset download.
Results
The program successfully processed 100 images in one execution and was additionally tested with 500 images.
The proof artifacts contain execution logs, GPU information, and before/after images.

**Lessons Learned**
This project demonstrates how image-processing operations can be divided into independent pixel operations and executed in parallel using CUDA.
It also demonstrates the relationship between CUDA threads, blocks, and grids.

One consideration is that transferring data between CPU and GPU memory has overhead. For small images, this overhead can become significant compared with the computation itself.

**Conclusion**
This project demonstrates GPU-based image processing using CUDA C++. A CUDA kernel performs the actual image filtering operation, and the program processes a large number of small image inputs in a single execution.
