#1
task = tasks.setup_task(args)
## 1. pass through the data file, load dictionary




#2
task.load_dataset(valid_sub_split, combine=False, epoch=0)
#3
model = task.build_model(args)
#4
criterion = task.build_criterion(args)
#5
trainer = Trainer(args, task, model, criterion)
#6
train(args, trainer, task, epoch_itr)
#7
train(args, trainer, task, epoch_itr)