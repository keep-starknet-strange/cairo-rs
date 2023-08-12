#!/usr/bin/env sh

# move to the directory where the script is located
cd $(dirname "$0")

tests_path="../../../cairo_programs"
proof_tests_path="${tests_path}/proof_programs"
exit_code=0
trace=false
memory=false
passed_tests=0
failed_tests=0

for i in $@; do
    case $i in
        "trace") trace=true
        echo "Requested trace comparison"
        ;;
        "memory") memory=true
        echo "Requested memory comparison"
        ;;
        "proof_mode") tests_path=$proof_tests_path
        echo "Requested proof mode usage"
        ;;
        *)
        ;;
    esac
done

files=$(ls $tests_path)
EXIT_CODE=$?
if [ ${EXIT_CODE} != 0 ]; then
    exit ${EXIT_CODE}
fi

for file in $(ls $tests_path | grep .cairo$ | sed -E 's/\.cairo$//'); do
    path_file="$tests_path/$file"

    if $trace; then
        if ! diff -q $path_file.trace $path_file.rs.trace; then
            echo "Traces for $file differ"
            exit_code=1
            failed_tests=$((failed_tests + 1))
        else
            passed_tests=$((passed_tests + 1))
        fi
    fi

    if $memory; then
        if ! ./memory_comparator.py $path_file.memory $path_file.rs.memory; then
            echo "Memory differs for $file"
            exit_code=1
            failed_tests=$((failed_tests + 1))
        else
            passed_tests=$((passed_tests + 1))
        fi
    fi
done

if test $failed_tests != 0; then
    echo "Comparisons: $failed_tests failed, $passed_tests passed, $((failed_tests + passed_tests)) total"
elif test $passed_tests = 0; then
    echo "No tests ran!"
    exit_code=2
else
    echo "All $passed_tests tests passed; no discrepancies found"
fi

exit "${exit_code}"
