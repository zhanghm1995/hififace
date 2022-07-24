set -x

python hififace_inference.py --gpus 0 --model_config config/model.yaml \
                          --model_checkpoint_path hififace_opensouce_299999.ckpt \
                          --source_image_path assets/inference_samples/01_source.png \
                          --target_image_path assets/inference_samples/01_target.png \
                          --output_image_path ./01_result.png