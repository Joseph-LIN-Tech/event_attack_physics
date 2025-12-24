import torch

# # 加载两个模型的checkpoint
model1_params = torch.load('/home/lgx/code/AAAI25/Attack/submission/event_20/checkpoints/gen_rvt-b.ckpt')
model1_params = model1_params['state_dict']
model2_params = torch.load('/home/lgx/code/AAAI25/Attack/submission/event_20/output/saved_checkpoints/11500_G.pth')
# 打印checkpoint的键
#print(checkpoint1.keys())
# 假设模型的state_dict包含参数
# model1_params = checkpoint1['model_state_dict']
# model2_params = checkpoint2['model_state_dict']

# 比较参数
different_params = {}

for name, param1 in model1_params.items():
    #print(name)
    name_fix = "rvt_model."+name
    if name_fix in model2_params.keys():
        param2 = model2_params[name_fix].to(param1.device)
        
        if not torch.equal(param1, param2):
            # 记录差异
            different_params[name] = (param1, param2)
            print(f"Parameter {name} is different.")
        else:
            print("same", name)

print(f"Total different parameters: {len(different_params)}")

