import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)
print(a * b)
print(torch.dot(a, b))
print(a.sum())

inputs = torch.tensor([2.0, 3.0, 1.0])
weights = torch.tensor([0.5, 0.2, 0.8])
bias = -3.0

z = torch.dot(inputs, weights) + bias

print(z)

activated = torch.relu(z)
print(activated)


w = torch.tensor(2.0, requires_grad=True)

y = 3*w

y.backward()

print(w.grad)

w = torch.tensor(2.0, requires_grad=True)

loss = w ** 2
loss.backward()

with torch.no_grad():
    w -= 0.1 * w.grad

print(w)

w = torch.tensor(2.0, requires_grad=True)

for step in range(20):
    loss = w ** 2
    loss.backward()

    with torch.no_grad():
        w -= 0.1 * w.grad

    w.grad.zero_()

    print(f"Step {step + 1}: weight = {w.item():.4f}")

    x = torch.tensor(4.0)
y = torch.tensor(8.0)

w = torch.tensor(0.5, requires_grad=True)

prediction = w * x
loss = (prediction - y) ** 2

print("Prediction:", prediction.item())
print("Loss:", loss.item())

loss.backward()

print("Gradient:", w.grad.item())

with torch.no_grad():
    w -= 0.01 * w.grad
print()
print("New weight:", w.item())