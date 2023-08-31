include(CMakeFindDependencyMacro)

# use the same version for metatensor-core as the main CMakeLists.txt
set(REQUIRED_METATENSOR_VERSION @REQUIRED_METATENSOR_VERSION@)
find_package(metatensor ${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)

# We can only load metatensor_torch with the exact same version of Torch that
# was used to compile it (and is stored in BUILD_TORCH_VERSION)
set(BUILD_TORCH_VERSION @Torch_VERSION@)

find_package(Torch ${BUILD_TORCH_VERSION} REQUIRED EXACT)


include(${CMAKE_CURRENT_LIST_DIR}/metatensor_torch-targets.cmake)
