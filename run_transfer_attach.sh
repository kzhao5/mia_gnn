# # shellcheck disable=SC2006
# py_path=`which python`

# run() {
#     number=$1
#     shift
#     for i in $(seq $number); do
#       # shellcheck disable=SC2068
#       $@
#       $py_path transfer_based_attack.py
#     done
# }

# # shellcheck disable=SC2046
# # shellcheck disable=SC2006
# #echo $epoch
# run "$1"
#!/bin/bash

py_path=$(which python)

run() {
    number=$1
    for i in $(seq 1 "$number"); do
        $py_path transfer_based_attack.py
    done
}

if [ "$1" == "--number" ]; then
    run "$2"
else
    echo "Usage: $0 --number <number_of_runs>"
    exit 1
fi