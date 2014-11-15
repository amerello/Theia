ADD_EXECUTABLE(match_sift_descriptors match_sift_descriptors.cc)
TARGET_LINK_LIBRARIES(match_sift_descriptors theia ${GFLAGS_LIBRARIES})

ADD_EXECUTABLE(extract_features extract_features.cc)
TARGET_LINK_LIBRARIES(extract_features theia ${GFLAGS_LIBRARIES})
