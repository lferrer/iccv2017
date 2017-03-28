/* Sources directory */
#define SOURCE_FOLDER "/home/ferrer/c3d/C3D-v1.1"

/* Binaries directory */
#define BINARY_FOLDER "/home/ferrer/c3d/C3D-v1.1/build"

/* Test device */
#define CUDA_TEST_DEVICE -1

/* Temporary (TODO: remove) */
#if 1
  #define CMAKE_SOURCE_DIR SOURCE_FOLDER "/src/"
  #define EXAMPLES_SOURCE_DIR BINARY_FOLDER "/examples/"
  #define CMAKE_EXT ".gen.cmake"
#else
  #define CMAKE_SOURCE_DIR "src/"
  #define EXAMPLES_SOURCE_DIR "examples/"
  #define CMAKE_EXT ""
#endif
