#!/bin/bash

display_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --method METHOD      Specify the training regime (default: simclr, options: byol)"
    echo "  --mode MODE          Specify the mode (default: poisoned, options: clean)"
    echo "  --defense DEFENSE    Specify the defense (default: None, options: blur)"
    echo "  --dataset DATASET    Specify the dataset (default: cifar10)"
    echo "  --suffix SUFFIX      Specify a suffix for the job name (default: empty)"
    echo "  --threat_model MODEL Specify the threat model (default: ctrl, options: fiba, htba)"
    echo "  --ctype CTYPE        Specify the colorspace to evaluate model (default: RGB, options: YUV, Y, U, V, HLS, HSV, LUV, LAB, YCbCr)"
    echo "  -h, --help           Display this help and exit"
    exit 1
}

method="simclr"
mode="poisoned"
defense=""
dataset="cifar10"
threat_model="ctrl"
ctype="RGB"
suffix=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) display_help; exit 0 ;;
    esac 

    key="$1"

    case $key in
        --method)
            method="$2"
            shift
            shift
            ;;
        --mode)
            mode="$2"
            shift
            shift
            ;;
        --defense)
            defense="$2"
            shift
            shift
            ;;
        --dataset)
            dataset="$2"
            shift
            shift
            ;;
        --suffix)
            suffix="$2"
            shift
            shift
            ;;
        --threat_model)
            threat_model="$2"
            shift
            shift
            ;;
        --ctype)
            ctype="$2"
            shift
            shift
            ;;
        *)
            # Unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# # Check if required options are provided
# if [ -z "$mode" ] || [ -z "$defense" ]; then
#     echo "Usage: $0 --mode <mode> --defense <defense>"
#     exit 1
# fi

# Generate a unique job name
if [ "$mode" != "poisoned" ]; then
    job_name="${method}_${mode}_${defense}_${dataset}${suffix}"
else
    job_name="${method}_${defense}_${dataset}_${threat_model}${suffix}"
fi

poisoned="--mode normal --trial clean"
if [ "$mode" == "poisoned" ]; then
    poisoned="--mode frequency --trial test"
fi

augmentation=""
if [ "$defense" == "blur" ]; then
    augmentation="--blur"
elif [ "$defense" == "value" ]; then
    augmentation="--value_channel"
elif [ "$defense" == "both" ]; then
    augmentation="--blur --value_channel"
fi

poison_ratio=0.01
if [ "$dataset" != "cifar10" ]; then
    poison_ratio=$(awk "BEGIN {print $poison_ratio / 5; exit}")
fi

# source your virtual environmenet, if any
# source ../ctrl/bin/activate

MYPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

python classifier.py $poisoned --freqPatch --dataset $dataset $augmentation --ctype $ctype --method $method --threat_model ${threat_model} --channel 1 2 --trigger_position 15 31 --poison_ratio $poison_ratio --lr 0.06 --wd 0.0005 --magnitude 100.0 --poisoning --epochs 20 --gpu 0 --window_size 32 --saved_path "${MYPATH}/saves/${job_name}/"