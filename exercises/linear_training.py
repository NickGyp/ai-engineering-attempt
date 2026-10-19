import torch

x = torch.tensor(4.0)
y = torch.tensor(8.0)

w = torch.tensor(0.5, requires_grad=True)

learning_rate = 0.01

for epoch in range(20):
    prediction = w*x
    loss = (prediction-y)**2

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad

    w.grad.zero_()

    print(f"Epoch {epoch+1}: weight = {w.item():.4f}, loss = {loss.item():.4f}")


new_x = torch.tensor(7.0)

with torch.no_grad():
    new_prediction = w*new_x

print("New prediction:", new_prediction.item())

new_x = torch.tensor([1.0, 2.0, 3.0, 4.0])

with torch.no_grad():
    predictions = w* new_x

print ("predictionsss:", predictions)


targets = torch.tensor([2.0, 4.0, 6.0, 8.0])

squared_errors = (predictions - targets) ** 2

mse = squared_errors.mean()

print("Squared errors:", squared_errors)
print("MSE:", mse.item())