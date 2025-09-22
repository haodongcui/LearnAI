import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np

num_tasks = 5

inner_epochs = 10
inner_lr = 0.01
outer_lr = 0.001

def generate_one_task():
    x = np.random.rand(10, 1).astype(np.float32)
    y = np.sin(x) + 0.1 * np.random.randn(10, 1).astype(np.float32)

    x_q = np.random.rand(10, 1).astype(np.float32)
    y_q = np.sin(x_q) + 0.1 * np.random.randn(10, 1).astype(np.float32)
    return torch.tensor(x), torch.tensor(y), torch.tensor(x_q), torch.tensor(y_q)

def generate_tasks(num_tasks = 5):
    tasks = []
    for _ in range(num_tasks):
        task = generate_one_task()
        tasks.append(task)
    return tasks

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        out = self.linear3(x)
        return out

model = Model()
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=outer_lr)

tasks = generate_tasks(num_tasks=num_tasks)
losses = []

original_params = [p.data.clone() for p in model.parameters()]

losses_by_task = []
grads_by_task = []

for i, task in enumerate(tasks):
    x, y, x_q, y_q = task

    # fast_model = Model()
    # fast_model.load_state_dict(model.state_dict())
    fast_model = deepcopy(model) # 深度拷贝模型
    fast_optimizer = optim.Adam(fast_model.parameters(), lr=inner_lr)

    # 内循环（在支持集上训练）
    for epoch in range(inner_epochs):
        fast_optimizer.zero_grad()
        y_pred = fast_model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        fast_optimizer.step()
        print(f"Task {i}, Inner Epoch {epoch}/{inner_epochs}, Loss: {loss.item()}")

    # 外循环1（在查询集上计算每个任务的loss）
    y_q_pred = fast_model(x_q)
    loss = criterion(y_q_pred, y_q)
    losses_by_task.append(loss.item())
    grads = torch.autograd.grad(outputs=loss, inputs=fast_model.parameters())  # 因变量是loss， 自变量是模型参数
    # grads = [grad.view(-1) for grad in grads]
    # grads_by_task.append(grads) # 因变量是loss， 自变量是模型参数
    # print(grads)

    # 手动更新主模型参数
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            param -= outer_lr * grad



# 检查参数是否变化
updated_params = [p.data.clone() for p in model.parameters()]
for p_old, p_new in zip(original_params, updated_params):
    print(torch.equal(p_old, p_new))  # 应输出 False