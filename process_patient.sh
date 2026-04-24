#!/bin/bash
# Master script to run full ASOCA inference and extract individual NIfTI masks

if [ "$#" -ne 2 ]; then
    echo "Usage: bash process_patient.sh <input_nrrd_file> <output_directory>"
    echo "Example: bash process_patient.sh ./ASOCA/ASOCA/Normal/CTCA/Normal_1.nrrd ./Outputs"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_DIR=$2
BASENAME=$(basename "$INPUT_FILE" .nrrd)
OUTPUT_BASE="$OUTPUT_DIR/$BASENAME"

mkdir -p "$OUTPUT_DIR"

echo -e "\n======================================================="
echo "🚀 Running AI Inference Pipeline"
echo -e "=======================================================\n"
export nnUNet_results=/workspace/Project/nnUNet_results
python /workspace/Project/run_inference_pipeline.py \
    --input "$INPUT_FILE" \
    --output "${OUTPUT_BASE}.nii.gz" \
    --dataset_id 100 \
    --fold 0

echo -e "\n✅ All done! Individual binary masks saved to: $OUTPUT_DIR"