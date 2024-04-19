debug ?= 0

NAME := seam-carving
SOURCE_DIR := .
TARGET_DIR := target

CHECK_FLAGS = -strict-style -vet-semicolon -vet-style
ifeq ($(debug), 1)
	CHECK_FLAGS := $(CHECK_FLAGS) -debug
	BUILD_FLAGS := $(CHECK_FLAGS)
else
	BUILD_FLAGS := $(CHECK_FLAGS) -o:speed
endif

run: prepare $(SOURCE_FILES)
	@mkdir -p $(TARGET_DIR)/$@
	odin run $(SOURCE_DIR) $(BUILD_FLAGS) -out:$(TARGET_DIR)/$@/$(NAME)

test: prepare $(SOURCE_FILES)
	@mkdir -p $(TARGET_DIR)/$@
	odin test $(SOURCE_DIR) $(BUILD_FLAGS) -out:$(TARGET_DIR)/$@/$(NAME)

build: prepare $(SOURCE_FILES)
	@mkdir -p $(TARGET_DIR)/$@
	odin build $(SOURCE_DIR) $(BUILD_FLAGS) -out:$(TARGET_DIR)/$@/$(NAME)

check:
	odin check $(SOURCE_DIR) $(CHECK_FLAGS)

prepare:
	@mkdir -p $(TARGET_DIR)

clean:
	@rm -rf $(TARGET_DIR)

.PHONY: run test prepare clean
