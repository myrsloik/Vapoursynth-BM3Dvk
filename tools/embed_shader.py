#!/usr/bin/env python3
# Turns the kernel source into a C++ header holding it as a raw string literal, so it ships
# inside the binary and nothing has to be found or read at runtime.
#
# Usage: embed_shader.py <input.comp> <output.h>
#
# A raw string rather than a byte array keeps the generated header readable and quick to
# compile. The delimiter is deliberately obscure: GLSL cannot contain )BM3DVKSHADER" unless
# someone puts it there on purpose, and that is refused here rather than mangled.

import sys

DELIMITER = "BM3DVKSHADER"


def main(argv):
    if len(argv) != 3:
        print("usage: embed_shader.py <input.comp> <output.h>", file=sys.stderr)
        return 2
    in_path, out_path = argv[1], argv[2]

    with open(in_path, "rb") as f:
        source = f.read().decode("utf-8")
    if (")" + DELIMITER + '"') in source:
        print("%s contains the raw string delimiter )%s\" -- change DELIMITER in embed_shader.py"
              % (in_path, DELIMITER), file=sys.stderr)
        return 1

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("// Generated from %s by tools/embed_shader.py. Do not edit.\n" % in_path.replace("\\", "/"))
        f.write('const char bm3dGlsl[] = R"%s(\n' % DELIMITER)
        f.write(source)
        f.write(')%s";\n' % DELIMITER)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
