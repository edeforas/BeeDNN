include_directories(../src)

add_executable(test_layer_convolution test_layer_convolution.cpp  )
target_link_libraries(test_layer_convolution libBeeDNN)

add_executable(test_matrix test_matrix.cpp  )
target_link_libraries(test_matrix libBeeDNN)

add_executable(test_jsonfile test_jsonfile.cpp  )
target_link_libraries(test_jsonfile libBeeDNN)

#add_executable(test_matrix_GEMM test_matrix_GEMM.cpp  )
#target_link_libraries(test_matrix_GEMM libBeeDNN)