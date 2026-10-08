WANDB="+logging_params.wandb.project=polar-express"
COMMON="+training_data=fineweb gpt_model=gpt-small hydra.job.name=\"main_run_wdoff\""
MUON="optimizer_params.name=muon +optimizer_params.args.ns_steps=5"

#Do now run with weight decay on
for wd in 0.0; do

for lr in 0.002 0.005 0.01; do
./submit_hydra.sh test_hydra $COMMON optimizer_params.name=muon +optimizer_params.args.ns_steps=2 +optimizer_params.args.polar_method=machpolar17 optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=$wd $WANDB
./submit_hydra.sh test_hydra $COMMON $MUON +optimizer_params.args.polar_method=polarexpress optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=$wd $WANDB
done

#for lr in 0.0001 0.0002 0.0005 0.001 0.002 0.005; do
#./submit_hydra.sh test_hydra $COMMON optimizer_params.name=adamw optimizer_params.args.lr=$lr optimizer_params.args.weight_decay=$wd $WANDB
#done

done
