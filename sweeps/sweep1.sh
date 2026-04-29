WANDB="+logging_params.wandb.project=polar-express"
COMMON="+training_data=fineweb gpt_model=gpt-small hydra.job.name=\"main_run\""
MUON="optimizer_params.name=muon +optimizer_params.args.ns_steps=5"

for wd in 0.0; do

for lr in 0.002 0.005; do
for polar_method in Keller; do
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.polar_method=$polar_method optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=$wd $WANDB
done
done

done
