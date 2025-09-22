import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np

num_tasks = 5

inner_epochs = 5
outer_epochs = 10
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
tasks_test = generate_tasks(num_tasks=num_tasks)

# MAML 算法
for outer_epoch in range(outer_epochs):
    for i, task in enumerate(tasks):
        x, y, x_q, y_q = task

        fast_model = deepcopy(model) # 深度拷贝模型
        fast_optimizer = optim.Adam(fast_model.parameters(), lr=inner_lr)

        # 内循环（在支持集上训练）
        for inner_epoch in range(inner_epochs):
            fast_optimizer.zero_grad()
            y_pred = fast_model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            fast_optimizer.step()
            # print(f"Task {i}, Inner Epoch {inner_epoch}/{inner_epochs}, Loss: {loss.item()}")

        # 外循环预备（在查询集上计算每个任务的loss）
        y_q_pred = fast_model(x_q)
        loss = criterion(y_q_pred, y_q)
        grads = torch.autograd.grad(outputs=loss, inputs=fast_model.parameters())  # 因变量是loss， 自变量是模型参数

        # 手动更新模型全局参数（梯度下降）
        with torch.no_grad():
            for param, grad in zip(model.parameters(), grads):
                param -= outer_lr * grad

    # 外循环（在测试集上计算每个任务的loss，显示一下效果）
    losses_test = []
    for i, task in enumerate(tasks_test):
        x, y, x_q, y_q = task

        fast_model = deepcopy(model)  # 深度拷贝模型
        fast_optimizer = optim.Adam(fast_model.parameters(), lr=inner_lr)
        for inner_epoch in range(inner_epochs):
            fast_optimizer.zero_grad()
            y_pred = fast_model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            fast_optimizer.step()
        y_q_pred = fast_model(x_q)
        loss = criterion(y_q_pred, y_q) # 一个任务的loss
        losses_test.append(loss.item())
    loss_test = np.mean(losses_test)
    print(f"Outer Epoch {outer_epoch}/{outer_epochs}, Loss: {loss_test}")

