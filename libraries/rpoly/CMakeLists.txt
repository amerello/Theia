project (rpoly)

INCLUDE_DIRECTORIES(./rpoly)
SET (rpoly_sources
  rpoly.cc)

ADD_LIBRARY(rpoly ${rpoly_sources})
INSTALL(TARGETS rpoly
  EXPORT  TheiaExport
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib${LIB_SUFFIX}
  ARCHIVE DESTINATION lib${LIB_SUFFIX})
