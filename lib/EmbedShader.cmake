# Turns a shader source file into a C++ header holding it as a raw string literal, for
# compilers without C23 #embed. Run in script mode with -DIN_FILE=... -DOUT_FILE=...
#
# A raw string is used rather than a byte array so the generated header stays readable and
# compiles fast. The delimiter is deliberately obscure: the payload is GLSL, which cannot
# contain )BM3DVKSHADER" unless someone puts it there on purpose, and the check below turns
# that into a configuration error rather than a mangled build.

if(NOT IN_FILE OR NOT OUT_FILE)
    message(FATAL_ERROR "EmbedShader.cmake requires -DIN_FILE= and -DOUT_FILE=")
endif()

file(READ "${IN_FILE}" _contents)

string(FIND "${_contents}" ")BM3DVKSHADER\"" _clash)
if(NOT _clash EQUAL -1)
    message(FATAL_ERROR "${IN_FILE} contains the raw string delimiter )BM3DVKSHADER\" -- change the delimiter in EmbedShader.cmake.")
endif()

get_filename_component(_name "${IN_FILE}" NAME)
set(_out "// Generated from ${_name} by EmbedShader.cmake. Do not edit.\n")
string(APPEND _out "const char bm3dGlsl[] = R\"BM3DVKSHADER(\n")
string(APPEND _out "${_contents}")
string(APPEND _out ")BM3DVKSHADER\";\n")

# Only rewrite on a real change, so an unchanged shader does not force a rebuild.
set(_existing "")
if(EXISTS "${OUT_FILE}")
    file(READ "${OUT_FILE}" _existing)
endif()
if(NOT _existing STREQUAL _out)
    file(WRITE "${OUT_FILE}" "${_out}")
endif()
