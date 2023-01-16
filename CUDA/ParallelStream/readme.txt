並行編譯
nvcc ./stream_test.cu -o stream_legacy

並行編譯
nvcc --default-stream per-thread ./stream_test.cu -o stream_per-thread