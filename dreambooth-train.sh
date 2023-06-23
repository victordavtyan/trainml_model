#!/bin/bash

for i in "$@"; do
  case $i in
    -s=*|--steps=*)
      STEPS="${i#*=}"
      shift # optional
      ;;
    -s=*|--uid=*)
      U_UID="${i#*=}"
      shift # optional
      ;;
    -i=*|--images=*)
      IMAGES="${i#*=}"
      shift # optional
      ;;
    -t=*|--token=*)
      TOKEN="${i#*=}"
      shift # optional
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done


#cp -r ${TRAINML_DATA_PATH}/* .

cd diffusers/examples/dreambooth

## Disable accelerate logging and increase verbosity
sed -i '/import os/aimport sys' train_dreambooth.py
# sed -i '/import os/aimport logging' train_dreambooth.py
sed -i 's/logger = get_logger(__name__)/logger = logging.getLogger()/' train_dreambooth.py
sed -i 's/logger\./logging./' train_dreambooth.py
sed -i '/logger = logging.getLogger()/alogger.setLevel(logging.INFO)' train_dreambooth.py
sed -i '/logger = logging.getLogger()/alogger.addHandler(logging.StreamHandler(sys.stderr))' train_dreambooth.py
sed -i 's/if accelerator.is_local_main_process:/if True:/' train_dreambooth.py
sed -i 's/disable=not accelerator.is_local_main_process/disable=False/' train_dreambooth.py
sed -i 's/accelerator.state, main_process_only=False/accelerator.state/' train_dreambooth.py


ls -al ${TRAINML_CHECKPOINT_PATH}/
## Run training

#--enable_xformers_memory_efficient_attention \

python train_dreambooth.py \
--pretrained_model_name_or_path=${TRAINML_CHECKPOINT_PATH} \
--instance_data_dir=${TRAINML_DATA_PATH}/instance-data-${U_UID} \
--class_data_dir=${TRAINML_DATA_PATH}/regularization-data-men \
--output_dir=${TRAINML_OUTPUT_PATH} \
--with_prior_preservation --prior_loss_weight=1 \
--instance_prompt="photo of ${TOKEN} person" \
--class_prompt="photo of a person" \
--resolution=512  \
--train_batch_size=2 \
--sample_batch_size=1 \
--gradient_accumulation_steps=2 \
--use_8bit_adam  \
--train_text_encoder \
--learning_rate=0.75e-06 \
--seed=123456789 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--num_class_images=196 \
--max_train_steps=350 \
--checkpointing_steps=250 \
--mixed_precision=fp16 \
--prior_generation_precision=fp16 \
--allow_tf32