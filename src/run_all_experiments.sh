python3 src/eval_bias_fairface_utk.py --clip_backend=RN50 --dataset=utk_faces
python3 src/eval_bias_fairface_utk.py --clip_backend=ViT-L/14 --dataset=utk_faces
python3 src/eval_bias_fairface_utk.py --clip_backend=RN50x64 --dataset=utk_faces

python3 src/eval_bias_fairface_utk.py --clip_backend=RN50 --dataset=fair_face_padding_025
python3 src/eval_bias_fairface_utk.py --clip_backend=ViT-L/14 --dataset=fair_face_padding_025
python3 src/eval_bias_fairface_utk.py --clip_backend=RN50x64 --dataset=fair_face_padding_025

python3 src/eval_bias_fairface_utk.py --clip_backend=RN50 --dataset=fair_face_padding_125
python3 src/eval_bias_fairface_utk.py --clip_backend=ViT-L/14 --dataset=fair_face_padding_125
python3 src/eval_bias_fairface_utk.py --clip_backend=RN50x64 --dataset=fair_face_padding_125