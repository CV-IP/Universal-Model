### Fine-tune Universal Model on your own target task

```python
base_model = unet3d.UNet3D()

#Load pre-trained model
model_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
checkpoint = torch.load(model_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
base_model.load_state_dict(unParalled_state_dict)
```

