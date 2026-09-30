
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)             \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
      return 1;                                                            \
    }                                                                      \
  } while (0)

__global__ void BlurKernel(const unsigned char* input,
                           unsigned char* output,
                           int width,
                           int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    output[y * width + x] = input[y * width + x];
    return;
  }

  int sum = 0;

  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      sum += input[(y + dy) * width + (x + dx)];
    }
  }

  output[y * width + x] =
      static_cast<unsigned char>(sum / 9);
}

void GenerateImage(std::vector<unsigned char>& image,
                   int width,
                   int height,
                   int index) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int value = (x + y + index * 10) % 256;

      int square_x = (index * 7) % (width - 20);
      int square_y = (index * 5) % (height - 20);

      if (x >= square_x && x < square_x + 20 &&
          y >= square_y && y < square_y + 20) {
        value = 255;
      }

      image[y * width + x] =
          static_cast<unsigned char>(value);
    }
  }
}

void SavePgm(const std::string& filename,
             const std::vector<unsigned char>& image,
             int width,
             int height) {
  FILE* file = fopen(filename.c_str(), "wb");

  if (!file) {
    return;
  }

  fprintf(file, "P5\n%d %d\n255\n", width, height);
  fwrite(image.data(), 1, image.size(), file);

  fclose(file);
}

int main(int argc, char* argv[]) {
  int image_count = 100;
  int width = 256;
  int height = 256;
  std::string output_directory = "output";

  if (argc > 1) {
    image_count = std::atoi(argv[1]);
  }

  if (argc > 2) {
    width = std::atoi(argv[2]);
  }

  if (argc > 3) {
    height = std::atoi(argv[3]);
  }

  if (argc > 4) {
    output_directory = argv[4];
  }

  std::cout << "CUDA Image Processing Project\n";
  std::cout << "Images: " << image_count << "\n";
  std::cout << "Resolution: " << width << "x" << height << "\n";

  size_t image_size =
      static_cast<size_t>(width) * static_cast<size_t>(height);

  unsigned char* d_input = nullptr;
  unsigned char* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, image_size));
  CUDA_CHECK(cudaMalloc(&d_output, image_size));

  dim3 block_size(16, 16);

  dim3 grid_size(
      (width + block_size.x - 1) / block_size.x,
      (height + block_size.y - 1) / block_size.y);

  std::cout << "GPU block size: "
            << block_size.x << "x" << block_size.y << "\n";

  std::cout << "GPU grid size: "
            << grid_size.x << "x" << grid_size.y << "\n";

  std::vector<unsigned char> input(image_size);
  std::vector<unsigned char> output(image_size);

  for (int i = 0; i < image_count; ++i) {

    GenerateImage(input, width, height, i);

    if (i == 0) {
      SavePgm(output_directory + "/before.pgm",
              input,
              width,
              height);
    }

    CUDA_CHECK(cudaMemcpy(
        d_input,
        input.data(),
        image_size,
        cudaMemcpyHostToDevice));

    BlurKernel<<<grid_size, block_size>>>(
        d_input,
        d_output,
        width,
        height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        output.data(),
        d_output,
        image_size,
        cudaMemcpyDeviceToHost));

    if (i == 0) {
      SavePgm(output_directory + "/after.pgm",
              output,
              width,
              height);
    }

    if ((i + 1) % 10 == 0) {
      std::cout << "Processed "
                << (i + 1)
                << "/"
                << image_count
                << " images\n";
    }
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  std::cout << "Processing completed successfully.\n";

  return 0;
}
