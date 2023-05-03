add_library(catch STATIC catch/catch.cpp)
target_include_directories(catch PUBLIC catch)
target_compile_features(catch PUBLIC cxx_std_11)
