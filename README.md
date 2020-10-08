### Fine-tune Universal Model on your own target task

Universal Model is a transferable and generalizable pre-trained model for 3D medical image analysis, which can be utilized to initialize the encoder for the target classification tasks and to initialize the encoder-decoder for the target segmentation tasks.  
The 3D deep model can be initialized with the pre-trained model as following:

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

