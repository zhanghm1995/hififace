set -x

## The offical script
# python hififace_trainer.py --model_config config/model.yaml --train_config config/trainer.yaml -n hififace

python hififace_trainer.py --model_config config/model_HDTF.yaml \
                           --train_config config/trainer_HDTF.yaml \
                           -n hififace_expression
