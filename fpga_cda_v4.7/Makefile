CXX = g++
CXXFLAGS = -std=c++17 -O2 -I$(XILINX_XRT)/include -I./src
LDFLAGS = -L$(XILINX_XRT)/lib -lxrt_coreutil -lxrt_core -lpthread

TARGET = run_cda_fpga
SRCS = src/llama2_k4v2_v2_6port.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	bash -c "source /opt/xilinx/xrt/setup.sh && ./$(TARGET) weights.bin -k forward.xclbin -z tokenizer.bin -n 10 -i 'Hello'"
