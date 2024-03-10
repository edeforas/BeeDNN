include_directories(../src)

add_executable(test_layer_convolution test_layer_convolution.cpp  )
target_link_libraries(test_layer_convolution libBeeDNN)

add_executable(test_matrix test_matrix.cpp  )
target_link_libraries(test_matrix libBeeDNN)

add_executable(test_regression_sin test_regression_sin.cpp  )
target_link_libraries(test_regression_sin libBeeDNN)

add_executable(test_classification_xor test_classification_xor.cpp  )
target_link_libraries(test_classification_xor libBeeDNN)

#add_executable(test_matrix_GEMM test_matrix_GEMM.cpp  )
#target_link_libraries(test_matrix_GEMM libBeeDNN)

add_executable(test_metrics test_metrics.cpp  )
target_link_libraries(test_metrics libBeeDNN)


add_test(test_regression_sin test_regression_sin)
add_test(test_classification_xor test_classification_xor)
add_test(test_metrics test_metrics)
add_test(test_matrix_GEMM test_matrix_GEMM)