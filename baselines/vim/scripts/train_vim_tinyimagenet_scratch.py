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
import requests
import zipfile
import shutil

try:
    import timm
    from timm.data import RepeatedAugmentSampler
except ImportError:
    RepeatedAugmentSampler = None
    print("⚠️ timm not installed or RepeatedAugmentSampler not available. Repeated Augmentation will be disabled.")

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

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
sys.stdout = Logger(f'logs/log_vim_tinyimagenet_scratch.txt')

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
    'num_classes': 200, # Tiny ImageNet has 200 classes
    'label_smoothing': 0.1,
    'mixup_alpha': 1.0,
    'warmup_epochs': 10,
    'repeated_aug': True,
    'img_size': 224, # Resizing 64x64 to 224x224
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
# 2. Model Setup (Vim-tiny)
# ==========================================
# Add the cloned Vim repository to the python path
vim_repo_path = os.path.join(os.getcwd(), 'Vim')
vim_module_path = os.path.join(vim_repo_path, 'vim')

if not os.path.exists(vim_module_path):
    print(f"⚠️ Vim repository not found at {vim_repo_path}. Cloning now...")
    os.system(f"git clone https://github.com/hustvl/Vim.git {vim_repo_path}")

if vim_module_path not in sys.path:
    sys.path.append(vim_module_path)

try:
    from models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
    print("✅ Successfully imported Vision Mamba model definition")
except ImportError as e:
    print(f"❌ Failed to import Vision Mamba: {e}")
    sys.exit(1)

print("Initializing Vim-tiny model from scratch...")
model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
    pretrained=False,
    num_classes=CONFIG['num_classes']
)
model = model.to(CONFIG['device'])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"📊 Total Parameters: {total_params/1e6:.2f}M")

# ==========================================
# 3. Data Preparation (Tiny ImageNet)
# ==========================================
def download_and_prepare_tiny_imagenet(root='./data'):
    dataset_dir = os.path.join(root, 'tiny-imagenet-200')
    if os.path.exists(dataset_dir):
        print(f"✅ Tiny ImageNet found at {dataset_dir}")
        return dataset_dir

    print("⚠️ Tiny ImageNet not found. Downloading...")
    os.makedirs(root, exist_ok=True)
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(root, 'tiny-imagenet-200.zip')
    
    # Download
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        sys.exit(1)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    
    # Format Validation Set
    print("Formatting validation set...")
    val_dir = os.path.join(dataset_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_anno_file = os.path.join(val_dir, 'val_annotations.txt')
    
    if os.path.exists(val_anno_file):
        with open(val_anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_id = parts[1]
                
                class_dir = os.path.join(val_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                
                src = os.path.join(val_img_dir, img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.exists(src):
                    shutil.move(src, dst)
        
        # Clean up images folder
        if os.path.exists(val_img_dir):
            shutil.rmtree(val_img_dir)
    
    print("✅ Tiny ImageNet prepared.")
    return dataset_dir

print("\n🔄 Preparing Tiny ImageNet dataset...")
data_dir = download_and_prepare_tiny_imagenet(root='./data')

# ImageNet Mean/Std
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)), # Resize 64x64 -> 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9), 
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
    transforms.RandomErasing(p=0.25), 
    Cutout(n_holes=1, length=112) 
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)

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
print("🚀 Starting Training of Vision Mamba on Tiny ImageNet")
print("="*60 + "\n")

best_acc = 0.0

def save_checkpoint(model, acc, epoch, filename="vim_tinyimagenet_best.pth"):
    if not os.path.exists("checkpoints_tinyimagenet"):
        os.makedirs("checkpoints_tinyimagenet")
    filepath = f"checkpoints_tinyimagenet/{filename}"
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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        
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
        save_checkpoint(model, best_acc, epoch+1, f"vim_tinyimagenet_best_acc{best_acc:.2f}.pth")
    
    print("-" * 60)

print("\n" + "="*60)
print("🏆 TRAINING COMPLETED!")
print(f"✅ Best Test Accuracy: {best_acc:.2f}%")
print("="*60)
