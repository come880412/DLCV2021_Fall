CUDA_VISIBLE_DEVICE=0 python3 train.py --n_epochs 100 \
                                    --batch_size 64 \
                                    --image_size 400 \
                                    --model resnest50 \
                                    --data_path ../dataset \
                                    --saved_name resnest50 \
                                    --val --lr 1e-3 --reweight

CUDA_VISIBLE_DEVICE=0 python3 train.py --n_epochs 100 \
                                    --batch_size 64 \
                                    --image_size 400 \
                                    --model SEresnet50 \
                                    --data_path ../dataset \
                                    --saved_name SEresnet50 \
                                    --val --lr 1e-3 --reweight


CUDA_VISIBLE_DEVICE=0 python3 train.py --n_epochs 100 \
                                    --batch_size 32 \
                                    --image_size 400 \
                                    --model resnest101 \
                                    --data_path ../dataset \
                                    --saved_name resnest101 \
                                    --val --lr 1e-3 --reweight

CUDA_VISIBLE_DEVICE=0 python3 train.py --n_epochs 100 \
                                    --batch_size 32 \
                                    --image_size 400 \
                                    --model SEresnet101 \
                                    --data_path ../dataset \
                                    --saved_name SEresnet101 \
                                    --val --lr 1e-3 --reweight

CUDA_VISIBLE_DEVICE=0 python3 train.py --n_epochs 100 \
                                    --batch_size 32 \
                                    --image_size 384 \
                                    --model pvt_v2_b2 \
                                    --data_path ../dataset \
                                    --saved_name pvt_v2_b2 \
                                    --val --lr 1e-3 --reweight

CUDA_VISIBLE_DEVICE=0 python3 train.py --n_epochs 100 \
                                    --batch_size 32 \
                                    --image_size 384 \
                                    --model pvt_v2_b3 \
                                    --data_path ../dataset \
                                    --saved_name pvt_v2_b3 \
                                    --val --lr 1e-3 --reweight

CUDA_VISIBLE_DEVICE=0 python3 train_arc.py --n_epochs 200 \
                                    --batch_size 64 \
                                    --image_size 384 \
                                    --model ArcSEResnet50 \
                                    --data_path ../dataset \
                                    --saved_name ArcSEresnet50 \
                                    --val --lr 1e-3 --reweight --arc 


