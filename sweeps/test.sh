WANDB="+logging_params.wandb.project=polar-express"
COMMON="+training_data=shakespeare"
MUON="optimizer_params.name=muon +optimizer_params.args.ns_steps=5"

#./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.polar_method=Keller $WANDB
#./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.polar_method=PolarExpress $WANDB


for ns_steps in 1; do
./submit_hydra.sh test_hydra $COMMON optimizer_params.name=muon +optimizer_params.args.ns_steps=ns_steps +optimizer_params.args.polar_method=MachPolar17 $WANDB
done
