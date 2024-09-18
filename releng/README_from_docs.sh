#! /bin/bash

sub_docs_from_directory() {
    template="$1"
    directory="$2"

    for file in "$directory"/*; do
        if [ -f "$file" ]; then
            placeholder="{{ $(echo "${file^^}" | tr ./ _) }}"
            echo "Saving $file to $placeholder with expanded environment variables" >&2
            # Export file contents to text following a pattern in template.md, e.g.:
            # section/test_file.md would replace "{{ SECTION_TEST_FILE_MD }}""
            template="${template//"$placeholder"/"$(<"$file")"}"
        fi
    done
    echo "$template"
}

REPO_DIR="$(realpath -- "$(dirname -- "$(dirname -- "${BASH_SOURCE[0]}")")")"
cd "$REPO_DIR/docs/" || exit 1
# Fills from docs/sections sections, then docs/help (note "working-directory")
sub_docs_from_directory "$(sub_docs_from_directory "$(<template.md)" "sections")" "help" >../README.md
