DATA_TYPE=$1
SCENE=$2
STYLE=$3

ckpt_gs=output/ckpt_gs/${DATA_TYPE}/${SCENE}
ckpt_stylegs=output/ckpt_stylegs/${DATA_TYPE}/${SCENE}_${STYLE}
data_dir=/data/storage/users/msabater/datasets/${DATA_TYPE}/${SCENE}
style_img=/data/storage/users/msabater/datasets/styles/${STYLE}.jpg



CUDA_VISIBLE_DEVICES=2 python palette_baseline.py -s ${data_dir} \
                -m ${ckpt_stylegs} \
                --point_cloud ${ckpt_gs}/point_cloud/iteration_30000/point_cloud.ply \
                --style ${style_img} \
                
