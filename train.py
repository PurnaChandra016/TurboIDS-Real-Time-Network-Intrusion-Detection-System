import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle, os

DATA_DIR   = r'C:\Users\purna\seai_project\data'
MODELS_DIR = r'C:\Users\purna\seai_project\models'
os.makedirs(MODELS_DIR, exist_ok=True)

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

print("Loading NSL-KDD dataset...")
train_df = pd.read_csv(f'{DATA_DIR}/KDDTrain+.txt', names=COLUMNS)
test_df  = pd.read_csv(f'{DATA_DIR}/KDDTest+.txt',  names=COLUMNS)

train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty',  axis=1, inplace=True)
print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['label']  = test_df['label'].apply( lambda x: 0 if x == 'normal' else 1)
print(f"Label distribution (train): {train_df['label'].value_counts().to_dict()}")

cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], test_df[col]]))
    train_df[col] = le.transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])

X_train = train_df.drop('label', axis=1).values.astype(np.float32)
y_train = train_df['label'].values.astype(np.float32)
X_test  = test_df.drop('label',  axis=1).values.astype(np.float32)
y_test  = test_df['label'].values.astype(np.float32)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

with open(f'{MODELS_DIR}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved.")

num_normal = np.sum(y_train == 0)
num_attack = np.sum(y_train == 1)
pos_weight = torch.tensor([num_normal / num_attack], dtype=torch.float32)
print(f"Class weight (pos_weight): {pos_weight.item():.4f}")

def make_loader(X, y, batch_size=256, shuffle=False):
    X_t = torch.tensor(X)
    y_t = torch.tensor(y).unsqueeze(1)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
test_loader  = make_loader(X_test,  y_test)

class IntrusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining on: {device}")

model     = IntrusionMLP(X_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=4, factor=0.5
)

EPOCHS = 50
best_acc = 0
print("\nEpoch | Loss   | Accuracy | LR")
print("-" * 45)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            prob = torch.sigmoid(out)
            preds.extend((prob.cpu().numpy() > 0.5).astype(int))
            labels.extend(yb.numpy().astype(int))

    acc = accuracy_score(labels, preds)
    scheduler.step(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f'{MODELS_DIR}/mlp_ids_best.pth')

    current_lr = optimizer.param_groups[0]['lr']
    print(f"  {epoch+1:02d}  | {total_loss/len(train_loader):.4f} | {acc*100:.2f}%    | {current_lr:.6f}")

print(f"\nBest Accuracy: {best_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=['Normal', 'Attack']))
print(f"\nModel saved: {MODELS_DIR}/mlp_ids_best.pth")