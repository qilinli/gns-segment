# Cmd to run
# Train
CUDA_VISIBLE_DEVICES=1 python -m gns.train --mode=train --data_path=/home/jovyan/work/data_temp/segment_beam/ --model_path=./models/segment_beam/ --output_path=./rollouts/segment_beam/ --batch_size=1 --noise_std=0.1 --connection_radius=15 --layers=5 --hidden_dim=64 --lr_init=0.001 --ntraining_steps=30000 --lr_decay_steps=9000 --dim=3 --project_name=Segment-3D --run_name=nsNS1_R15_L5N64 --nsave_steps=1000 --log=True

# Rollout
CUDA_VISIBLE_DEVICES=3 python -m gns.train --mode=rollout --data_path=/home/jovyan/work/data_temp/segment_beam/ --model_path=./models/segment_beam/ --model_file=NS0.1_R20_L5N96_BoundD/model-013000.pt --output_path=./rollouts/segment_beam/ --batch_size=1 --noise_std=0.1 --connection_radius=15 --layers=5 --hidden_dim=96 --dim=3

# Visualisation
python -m gns.render_rollout_1d --rollout_path=rollouts/Concrete1D/rollout_0.pkl --output_path=rollouts/Concrete1D/rollout_0.gif

CUDA_VISIBLE_DEVICES=7 python -m gns.train -mode=rollout --data_path=./data/Concrete1D/ --model_path=./models/Concrete1D/ --model_file=noise6.7e-4_R0.04_bs2_lr1e-3_step500k-model-326000.pt --output_path=./rollouts/Concrete1D-new/ --noise_std=0.00067 --dim=1d --connection_radius=0.03

# Notes
- If net config changed before evaluation, load weights may fail
- If subtle config changed, evaluation may have low results
- Train loss (acc) and val loss (pos) are not comparable currently
- wandb step increase by default everytime wandb.log() is called
- For quasi-static simulation, many particles have no acceleartion in many timesteps. Hence, the sampled training steps might have many zero ground truth or not, resulting in
    a large difference between training iterations, as shown by the training loss. This might be the reason that the training loss stucks quickly at some point
- Adding noise significantly decreases the training loss but the GNN is probably fitting the Gaussian noise. This is evidenced by the relative constant rollout (all particles move
    the same as the learning is on noise)
- pytorch geometric caps the knn in radius_graph to be <=32
- The original domain is x (-165, 165) and y (-10, 85). Normalise it to (0,1) and (0,1) will change the w/h ratio. 
- Be careful with the simulation domain, the Bullet in impact loading has made y too large unnessarily
