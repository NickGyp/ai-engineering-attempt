import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0])
w = torch.tensor(0.5, requires_grad=True)

optimizer = torch.optim.SGD([w], lr=0.01)

learning_rate = 0.01

for epoch in range(20):
    predictions = w * x 
    loss = ((predictions - y)**2).mean()

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch+1}: weight = {w.item():.4f}, loss = {loss.item():.4f}")