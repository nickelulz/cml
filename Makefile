BUILD_DIR := build
EXECUTABLE := $(BUILD_DIR)/executable/main

.PHONY: all clean run

# Default target: build the project
debug: all
	@echo "[MAKE] Debugging $(EXECUTABLE).."
	@valgrind $(EXECUTABLE) --track-origins=yes --leak-check=full

all:
	@echo "[MAKE] Configuring and building project..."
	cmake -S . -B $(BUILD_DIR)
	cmake --build $(BUILD_DIR)

# Run the built executable
run: all
	@echo "[MAKE] Running $(EXECUTABLE)..."
	@$(EXECUTABLE)

# Clean build artifacts
clean:
	@echo "[MAKE] Cleaning build directory..."
	rm -rf $(BUILD_DIR)
	rm gnuplot_tmp*
