add_library(LGC SHARED LGC.cpp)
target_include_directories(LGC PUBLIC ../../include)
target_link_libraries(LGC Eigen3::Eigen)