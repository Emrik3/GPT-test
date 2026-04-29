WANDB="+logging_params.wandb.project=polar-express"
COMMON="+training_data=shakespeare hydra.job.name=\"test_run\""
MUON="optimizer_params.name=muon +optimizer_params.args.ns_steps=5"

for wd in 0; do

for lr in 0.001 0.002 0.005 0.01 0.02 0.05 0.1; do
for steps in 5 9 17; do
./submit_hydra.sh test_hydra $COMMON optimizer_params.name=muon +optimizer_params.args.ns_steps=$steps +optimizer_params.args.polar_method=$polar_method optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=$wd $WANDB
done
done

for lr in 0.001 0.002 0.005 0.01 0.02 0.05 0.1; do
for polar_method in Keller polarexpress machpolar; do
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.polar_method=$polar_method optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=$wd $WANDB
done
done


done
