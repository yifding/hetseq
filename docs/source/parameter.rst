**********
Parameters
**********


usage:  [-h] [--no-progress-bar] [--seed N] [--cpu] [--log-interval N][--log-format {none,simple}] 
        [--num-workers N] [--max-tokens N] [--max-sentences N] [--required-batch-size-multiple N] 
        [--train-subset SPLIT] [--valid-subset SPLIT] [--validate-interval N] [--disable-validation] 
        [--max-tokens-valid N] [--max-sentences-valid N] [--curriculum N] [--task TASK] [--data DATA] 
        [--dict PATH of a file] [--config_file PATH of a file] [--max_pred_length MAX_PRED_LENGTH] 
        [--num_file NUM_FILE] [--distributed-world-size N] [--distributed-rank DISTRIBUTED_RANK] 
        [--distributed-gpus DISTRIBUTED_GPUS] [--distributed-backend DISTRIBUTED_BACKEND] 
        [--distributed-init-method DISTRIBUTED_INIT_METHOD] [--device-id DEVICE_ID] 
        [--distributed-no-spawn] [--ddp-backend {c10d}] [--bucket-cap-mb MB] [--fix-batches-to-gpus] 
        [--find-unused-parameters] [--fast-stat-sync] [--max-epoch N] [--max-update N] 
        [--clip-norm NORM] [--update-freq N1,N2,...,N_K] [--lr LR_1,LR_2,...,LR_N] [--min-lr LR] 
        [--use-bmuf] [--optimizer OPTIMIZER] [--adam-betas B] [--adam-eps D] [--weight-decay WD] 
        [--lr_scheduler LR_SCHEDULER] [--force-anneal N] [--warmup-updates N] 
        [--end-learning-rate END_LEARNING_RATE] [--power POWER] [--total-num-update TOTAL_NUM_UPDATE] 
        [--save-dir DIR] [--restore-file RESTORE_FILE] [--reset-dataloader] [--reset-lr-scheduler] 
        [--reset-meters] [--reset-optimizer] [--optimizer-overrides DICT] [--save-interval N]
        [--save-interval-updates N] [--keep-interval-updates N] [--keep-last-epochs N] [--no-save] 
        [--no-epoch-checkpoints] [--no-last-checkpoints] [--no-save-optimizer-state] 
        [--best-checkpoint-metric BEST_CHECKPOINT_METRIC] [--maximize-best-checkpoint-metric]
