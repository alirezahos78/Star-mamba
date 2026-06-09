import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import sys
import numpy as np
import random
import math
from tqdm import tqdm
from models.vmamba_efficient import EfficientVSSM

try:
    import timm
    from timm.data import RepeatedAugmentSampler
except ImportError:
    RepeatedAugmentSampler = None
    print("⚠️ timm not installed or RepeatedAugmentSampler not available. Repeated Augmentation will be disabled.")

# ==========================================
# 0.5. Logger Class
# ==========================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if not os.path.exists('logs'): os.makedirs('logs')
sys.stdout = Logger(f'logs/log_efficient_vmamba_cifar10_scratch.txt')

# ==========================================
# 0. Seed Setting
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'⚙️ Global Seed set to {seed}')

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
CONFIG = {
    'seed': 42,
    'batch_size': 128,
    'epochs': 300,
    'lr': 1e-3,
    'weight_decay': 0.05,
    'num_classes': 10,
    'label_smoothing': 0.1,
    'mixup_alpha': 1.0,
    'warmup_epochs': 10,
    'repeated_aug': True,
    'img_size': 224, # Resizing to 224 to match Vim experiment
    'patch_size': 16,
    'workers': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Configuration: {CONFIG}")
set_seed(CONFIG['seed'])

# ==========================================
# 1.5. Augmentation Helpers
# ==========================================
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask).expand_as(img)
        img = img * mask
        return img

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = random.betavariate(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 2. Model Setup (EfficientVMamba-Tiny)
# ==========================================
print("Initializing EfficientVMamba-Tiny model from scratch...")
# Using Tiny configuration adapted for 6M parameters
model = EfficientVSSM(
    patch_size=CONFIG['patch_size'],
    in_chans=3,
    num_classes=CONFIG['num_classes'],
    depths=[2, 2, 4, 2], 
    dims=48,
    ssm_d_state=16,
    ssm_dt_rank="auto",
    ssm_ratio=2.0,
    ssm_conv=3,
    ssm_conv_bias=False,
    forward_type="v2",
    mlp_ratio=0.0, # Efficient version has no MLP
    downsample_version="v3",
    patchembed_version="v2",
    drop_path_rate=0.2,
)
model = model.to(CONFIG['device'])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Total Parameters: {total_params/1e6:.2f}M")

# Calculate FLOPs (try `thop` first, then fall back to `fvcore`)
try:
    from thop import profile
    input_dummy = torch.randn(1, 3, CONFIG['img_size'], CONFIG['img_size']).to(CONFIG['device'])
    macs, params = profile(model, inputs=(input_dummy,), verbose=False)
    # `macs` is multiply-adds (MACs). FLOPs ≈ 2 * MACs. Many tools report MACs as FLOPs directly;
    # here we print both to be explicit.
    print(f"💻 MACs: {macs/1e9:.2f}G, FLOPs (≈2*MACs): {2*macs/1e9:.2f}G")
except ImportError:
    try:
        from fvcore.nn import FlopCountAnalysis
        input_dummy = torch.randn(1, 3, CONFIG['img_size'], CONFIG['img_size']).to(CONFIG['device'])
        flops = FlopCountAnalysis(model, input_dummy).total()
        print(f"💻 Total FLOPs (fvcore): {flops/1e9:.2f}G")
    except ImportError:
        print("⚠️ Neither `thop` nor `fvcore` installed. Install `thop` (recommended) via `pip install thop` to calculate FLOPs.")
    except Exception as e:
        print(f"⚠️ Error calculating FLOPs with fvcore: {e}")
except Exception as e:
    print(f"⚠️ Error calculating FLOPs with thop: {e}")

# ==========================================
# 3. Data Preparation
# ==========================================
print("\n🔄 Preparing CIFAR-10 dataset...")
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

# Scaled augmentations for 224x224
transform_train = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])), 
    transforms.RandomCrop(CONFIG['img_size'], padding=CONFIG['img_size']//8),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9), 
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
    transforms.RandomErasing(p=0.25), 
    Cutout(n_holes=1, length=CONFIG['img_size']//2) 
])

transform_test = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

# Ensure data directory exists
os.makedirs('./data', exist_ok=True)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

if CONFIG['repeated_aug'] and RepeatedAugmentSampler is not None:
    print('🔥 Repeated Augmentation Enabled (timm)!')
    ra_sampler = RepeatedAugmentSampler(trainset, num_repeats=5)
    train_loader = DataLoader(trainset, batch_size=CONFIG['batch_size'], sampler=ra_sampler, 
                             num_workers=CONFIG['workers'], pin_memory=True)
else:
    train_loader = DataLoader(trainset, batch_size=CONFIG['batch_size'], shuffle=True, 
                             num_workers=CONFIG['workers'], pin_memory=True)

test_loader = DataLoader(testset, batch_size=CONFIG['batch_size'], shuffle=False, 
                        num_workers=CONFIG['workers'], pin_memory=True)

print(f"✅ Train samples: {len(trainset)}, Test samples: {len(testset)}")

# ==========================================
# 4. Optimizer & Scheduler
# ==========================================
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
test_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

# Warmup Scheduler Logic (per step)
total_steps = CONFIG['epochs'] * len(train_loader)
warmup_steps = CONFIG['warmup_epochs'] * len(train_loader)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    T_max = total_steps - warmup_steps
    T_cur = step - warmup_steps
    return 0.5 * (1. + math.cos(math.pi * T_cur / T_max))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==========================================
# 5. Training Loop
# ==========================================
print("\n" + "="*60)
print("🚀 Starting Training of EfficientVMamba on CIFAR-10")
print("="*60 + "\n")

best_acc = 0.0

def save_checkpoint(model, acc, epoch, filename="efficient_vmamba_cifar10_best.pth"):
    if not os.path.exists("checkpoints_scratch"):
        os.makedirs("checkpoints_scratch")
    filepath = f"checkpoints_scratch/{filename}"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': acc,
    }, filepath)
    print(f"💾 Checkpoint saved: {filepath}")

for epoch in range(CONFIG['epochs']):
    start_time = time.time()
    
    # --- Train ---
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    
    for inputs, targets in loop:
        inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
        
        optimizer.zero_grad()
        
        # Mixup
        inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, targets, CONFIG['mixup_alpha'], CONFIG['device'])
        outputs = model(inputs_mixed)
        
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total_train += targets.size(0)
        
        # Approx accuracy for mixup
        correct_train += (predicted.eq(targets_a) * lam + predicted.eq(targets_b) * (1 - lam)).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100. * correct_train / total_train
    
    # --- Evaluate ---
    model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            outputs = model(inputs)
            
            loss = test_criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += targets.size(0)
            correct_test += predicted.eq(targets).sum().item()
            
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct_test / total_test
    
    epoch_time = time.time() - start_time
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1:03d} | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
    print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Test Loss:  {avg_test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
    
    if test_acc > best_acc:
        print(f"🎉 New Best Accuracy! {best_acc:.2f}% → {test_acc:.2f}%")
        best_acc = test_acc
        save_checkpoint(model, best_acc, epoch+1, f"efficient_vmamba_cifar10_best_acc{best_acc:.2f}.pth")
    
    print("-" * 60)

print("\n" + "="*60)
print("🏆 TRAINING COMPLETED!")
print(f"✅ Best Test Accuracy: {best_acc:.2f}%")
print("="*60)
