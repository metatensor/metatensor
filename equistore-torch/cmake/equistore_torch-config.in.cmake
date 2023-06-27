include(CMakeFindDependencyMacro)

# use the same version for equistore-core as the main CMakeLists.txt
set(REQUIRED_EQUISTORE_VERSION @REQUIRED_EQUISTORE_VERSION@)
find_package(equistore ${REQUIRED_EQUISTORE_VERSION} CONFIG REQUIRED)

# We can only load equistore_torch with the exact same version of Torch that
# was used to compile it (and is stored in BUILD_TORCH_VERSION)
set(BUILD_TORCH_VERSION @Torch_VERSION@)

find_package(Torch ${BUILD_TORCH_VERSION} REQUIRED EXACT)


include(${CMAKE_CURRENT_LIST_DIR}/equistore_torch-targets.cmake)
