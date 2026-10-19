import torch
from torch import nn

# Δημιουργούμε το μοντέλο
model = nn.Linear(1, 1)

# Δεδομένα εκπαίδευσης
x_train = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y_train = torch.tensor([
    [5.0],
    [8.0],
    [11.0],
    [14.0]
])

# Loss function και optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()

    predictions = model(x_train)
    loss = loss_fn(predictions, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())
#########################################################
new_x = torch.tensor([[6.0]])

model.eval()

with torch.no_grad():
    prediction = model(new_x)

print("Prediction for x=6:", prediction.item())