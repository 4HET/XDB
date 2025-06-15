CXX = g++
CXXFLAGS = -O3 -march=native -fopenmp -mavx2 -mfma -std=c++17 -Wall -Wextra
LDFLAGS = -fopenmp

SRCDIR = src
OBJDIR = obj
SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
TARGET = gemm_test

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET)

test: $(TARGET)
	./$(TARGET)

benchmark: $(TARGET)
	@echo "运行性能基准测试..."
	@export OMP_NUM_THREADS=32 && ./$(TARGET)

profile: $(TARGET)
	@echo "运行性能分析..."
	perf record -g ./$(TARGET)
	perf report

install-deps:
	@echo "检查依赖..."
	@which g++ || (echo "请安装 g++" && exit 1)
	@echo "所有依赖已满足"

help:
	@echo "可用目标:"
	@echo "  all        - 编译程序"
	@echo "  test       - 运行测试"
	@echo "  benchmark  - 运行基准测试"
	@echo "  profile    - 运行性能分析"
	@echo "  clean      - 清理编译文件"
	@echo "  help       - 显示帮助"