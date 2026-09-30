
NVCC = nvcc
CXXFLAGS = -O2

TARGET = image_filter
SOURCE = src/image_filter.cu

all:
	$(NVCC) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

run:
	./$(TARGET) 100 256 256 output
