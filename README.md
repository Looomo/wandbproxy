# wandbproxy
* 用于将wandb log同步保存至其他平台，如mlflow.
* 目前仅支持mlflow的log_metrics，暂不支持`{k: v}`之外的log, 如图片等.
* v0.2.0: 
  * 支持从wandb run中自动获取project， run name, config
  * 同步于wandb.init()的mlflow.start_run(), 不再需要手动管理mlflow生命周期。
* v0.1.0:
  * 初始化提交。 
# Usage
```
pip install https://github.com/Looomo/wandbproxy
# when you import wandb:
import wandbproxy # must import wandbproxy first

# Common setting of wandb
import wandb
wandb.init(
  project="test",
  config={},
  name="test",
)

# Common setting of mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.start_run()

# Training with logging
epochs = args.epoch
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    # this will log to wandb and mlflow at the same time
    wandb.log({"acc": acc, "loss": loss}, step = epoch)

mlflow.end_run()

```