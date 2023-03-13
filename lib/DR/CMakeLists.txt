add_library(DR SHARED DR.cpp DR_interface.cpp)
target_include_directories(DR PUBLIC ../../include)
target_link_libraries(DR Eigen3::Eigen)