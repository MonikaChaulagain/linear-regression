#linear REgression from Scratch
import torch

x=torch.randn(100,1)
y=3*x+2+0.1*torch.randn(100,1)

w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True)

print(f"initially:\nweight={w}\n bias={b}")

lr=0.1

for epoch in range(100):
    y_pred=w*x+b

    loss=((y_pred-y)**2).mean()

    loss.backward()

    #updatde weights manually
    with torch.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad

        w.grad.zero_()
        b.grad.zero_()

    if (epoch+1)%20==0:
        print(f"Epoch{epoch+1}:loss={loss.item()},weight={w.item()},bias={b.item()}")