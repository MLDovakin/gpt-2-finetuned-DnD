
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

from torch.optim import AdamW

accelerator  = Accelerator()
device = accelerator.device
optimizer = AdamW(model.parameters(), lr=1e-3)
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)


num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_linear_schedule_with_warmup(
   optimizer=optimizer, num_warmup_steps=4000, num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
       
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
       
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    print({'epoch': epoch, 'loss': loss.item()})
