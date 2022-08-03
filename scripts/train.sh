## The offical script
# python hififace_trainer.py --model_config config/model.yaml --train_config config/trainer.yaml -n hififace

ZHM_PATH="/221019051/"
YZH_PATH="/221019041/"

PYTHON=python
if [ -d "$ZHM_PATH" -o -d "$YZH_PATH" ]; then
    echo "In AIStation platform"
    PYTHON=/root/miniconda3/envs/py36-torch100-cu11/bin/python
    if [ -f "$PYTHON" ]; then
        PYTHON=/root/miniconda3/envs/py36-torch100-cu11/bin/python
    else
        PYTHON=python
    fi
    cd /221019051/Research/Face/hififace
fi

set -x
${PYTHON} hififace_trainer.py --model_config config/model_HDTF.yaml \
                              --train_config config/trainer_HDTF.yaml \
                              -n hififace_expression
