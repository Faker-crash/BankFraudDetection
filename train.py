import torch
import torch.nn as nn
import numpy as np  
import pandas as pd
from torch.utils.data import Dataset, DataLoader,random_split,ConcatDataset
import torch.nn.functional as F
import sys
import os
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from ctgan import CTGAN
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
from imblearn.over_sampling import RandomOverSampler, SMOTE,ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve, auc
torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(True)    
class CreditData(Dataset):
    def __init__(self, data):# data为tensor
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx,:-1]
        labels = self.data[idx,-1]
        return data, labels
    
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,n_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim,n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        h0 = torch.zeros(self.n_layers,x.size(0),self.hidden_dim).to(x.device)
        out,_ = self.gru(x.unsqueeze(1),h0)
        out = self.fc(out[:,-1,:])
        return out
    
class LSTMNet(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,n_layers):
        super(LSTMNet,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim,hidden_dim,n_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim,output_dim)
    def forward(self,x):
        h0 = torch.zeros(self.n_layers,x.size(0),self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers,x.size(0),self.hidden_dim).to(x.device)
        out,_ = self.lstm(x.unsqueeze(1),(h0,c0))
        out = self.fc(out[:,-1,:])
        return out
    
class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )
    def forward(self,x):
        x = self.encoder(x)
        return x
    
class CNN(nn.Module):
    def __init__(self,input_channel,output_dim):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv1d(1,32,kernel_size=3,stride=1,padding=1)# 一维卷积提取特征
        self.conv2 = nn.Conv1d(32,64,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2,padding=0)
        self.KAN_hidden_layer = [64, 1]

        self.KAN = KAN(self.KAN_hidden_layer,grid_size=5,spline_order=3,scale_noise=0.1,scale_base=1.0,scale_spline=1.0,
        base_activation=torch.nn.SiLU,grid_eps=0.02,grid_range=[-1, 1])#B-spline
        self.fc1 = nn.Linear(4096,1024)
        self.fc2 = nn.Linear(1024,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,16)
        self.fc5 = nn.Linear(16,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x = self.KAN(x)

        # x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        # x = self.fc5(x)

        return x
    
class CreditCardClassifier(nn.Module):
    def __init__(self,input_dim,LSTM_hidden_dim,LSTM_layers,LSTM_output_dim,
                 GRU_hidden_dim,GRU_output_dim,GRU_layers,Encoder_input_dim,Encoder_output_dim,
                 Encoder_hidden_dim,CNN_input_channel,CNN_output_dim):
        super(CreditCardClassifier,self).__init__()
        self.LSTM = LSTMNet(input_dim,LSTM_hidden_dim,LSTM_output_dim,LSTM_layers)
        self.GRU = GRUNet(input_dim,GRU_hidden_dim,GRU_output_dim,GRU_layers)
        self.AE = Encoder(Encoder_input_dim,Encoder_output_dim,Encoder_hidden_dim)
        self.CNN = CNN(CNN_input_channel,CNN_output_dim)
        self.KAN_hidden_layer = [CNN_output_dim, 128, 256, 128, 64, Encoder_output_dim]# 定义KAN模型的隐藏层结构，每层的输入和输出维度
        self.KAN = KAN(self.KAN_hidden_layer,grid_size=5,spline_order=3,scale_noise=0.1,scale_base=1.0,scale_spline=1.0,
        base_activation=torch.nn.SiLU,grid_eps=0.02,grid_range=[-1, 1])
        self.LSTM_weight = 0.5
        self.GRU_weight = 1 - self.LSTM_weight
        self.Sigmoid = nn.Sigmoid()
    def forward(self,x):
        # print(x)
        lstm_output = self.LSTM(x)
        # print(f"lstm:{lstm_output}")
        gru_output = self.GRU(x)
        # print(f"gru:{gru_output}")
        combined_output = lstm_output * self.LSTM_weight + gru_output * self.GRU_weight
        # print(f"combined:{combined_output}")
        # encoder_output = self.AE(combined_output)
        # print(f"encoder:{encoder_output}")
        # print(combined_output.shape)
        cnn_output = self.CNN(combined_output)
        # kan_output = self.KAN(cnn_output)
        # print(f"cnn:{cnn_output}")
        # kan_output = self.KAN(cnn_output)
        res = self.Sigmoid(cnn_output)
        return res
# def augementation(credit_data_tensor,percentage,batch_siz,mean,std):# 对不平衡的数据进行增强
#      generator = Generator(latent_dim, img_shape)
#      generator.load_state_dict(torch.load('generator.pth'))#读取训练好的生成器
#      generator.eval() # 设置为评估模式
#      nums = int((len(credit_data_tensor)*percentage)//batch_size)
#      for _ in range(nums):
#         label_vector = torch.ones((batch_size,1),requires_grad=False)
#         z = torch.randn((batch_size, latent_dim))
#         out = generator(z)
#         out = (out-mean)/std
#         new_data = torch.concatenate((out,label_vector),dim=1)
#         credit_data_tensor = torch.concatenate((credit_data_tensor,new_data))
#      return credit_data_tensor
def ctganAugementation(credit_card_df,target):# 使用CTGAN进行数据增强
    '''
    credit_card_df: 原本的少数类数据集
    target: 需要生成数据的数量
    '''
    columns=list(credit_card_df.columns)
    ctgan = CTGAN(epochs=100)
    ctgan.fit(credit_card_df, columns)
    synthetic_data = ctgan.sample(target)# 生成的数据
    synthetic_data_tensor = torch.tensor(synthetic_data.values, dtype=torch.float32)
    return synthetic_data_tensor#直接将生成的数据进行返回
def smoteAugemntation(credit_card_df):# 使用SMOTE进行数据增强
    smote = SMOTE(random_state=42)
    df1 = pd.DataFrame(credit_card_df[:,:-1].numpy(), columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'])
    df2 = pd.DataFrame(credit_card_df[:,-1].numpy(),columns=['Class'])
    X_resampled, y_resampled = smote.fit_resample(df1, df2)
    df_resampled_smote = pd.concat([X_resampled, y_resampled], axis=1)
    df_resampled_smote_tensor = torch.tensor(df_resampled_smote.values, dtype=torch.float32)
    return df_resampled_smote_tensor
def randomAugementation(credit_card_df):#使用随机过采样进行数据增强
    ros = RandomOverSampler(random_state=42)
    df1 = pd.DataFrame(credit_card_df[:,:-1].numpy(), columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'])
    df2 = pd.DataFrame(credit_card_df[:,-1].numpy(),columns=['Class'])
    X_resampled, y_resampled = ros.fit_resample(df1, df2)
    df_resampled_smote = pd.concat([X_resampled, y_resampled], axis=1)
    df_resampled_smote_tensor = torch.tensor(df_resampled_smote.values, dtype=torch.float32)
    return df_resampled_smote_tensor
def adasynAugementation(credit_card_df):#使用ADASYN进行数据增强
    adasyn = ADASYN(random_state=42)
    df1 = pd.DataFrame(credit_card_df[:,:-1].numpy(), columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'])
    df2 = pd.DataFrame(credit_card_df[:,-1].numpy(),columns=['Class'])
    X_resampled, y_resampled = adasyn.fit_resample(df1, df2)
    df_resampled_smote = pd.concat([X_resampled, y_resampled], axis=1)
    df_resampled_smote_tensor = torch.tensor(df_resampled_smote.values, dtype=torch.float32)
    return df_resampled_smote_tensor

def train_model(model,train_dataloader,criterion,optimizer,n_epochs,device):
    model.train()
    losses = []
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for i,(inputs,labels) in enumerate(train_dataloader):
            inputs,labels = inputs.to(device),labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs,labels)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

            running_loss += loss.item()
            # 计算准确率
            predicted = (outputs > 0.5).float()  # Sigmoid 输出 > 0.5 则为类别 1
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            print(f"Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        # scheduler.step(running_loss)
        avg_loss = running_loss / len(train_dataloader)
        accuracy = correct_preds / total_preds
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    torch.save(model.state_dict(), 'classifier_model_ctgan_b.pth')
    # plt.figure(figsize=(10, 5))
    # plt.title("Classification Loss During Training")
    # plt.plot(losses, label="Loss")
    # plt.xlabel("Iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.savefig('classification_loss.png')
                
def evaluate_model(model,test_dataloader,device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    with torch.no_grad():
        for inputs,labels in test_dataloader:
            inputs,labels = inputs.to(device),labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            preditions = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preditions.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # np.savetxt('labels.txt', all_labels, delimiter=',', fmt='%d')
    # np.savetxt('predictions.txt', all_predictions, delimiter=',', fmt='%d')

    fpr1, tpr1, _ = roc_curve(all_labels, all_probabilities)
    roc_auc1 = auc(fpr1, tpr1)
    
    acc_score = accuracy_score(all_labels,all_predictions)
    prec_score = precision_score(all_labels,all_predictions, zero_division=1)
    rec_score = recall_score(all_labels,all_predictions, zero_division=1)
    f_score = f1_score(all_labels,all_predictions, zero_division=1)
    print(f"Accuracy: {acc_score:.4f}, Precision: {prec_score:.4f}, Recall: {rec_score:.4f}, F1 Score: {f_score:.4f}")
    # with open('result.txt','w') as f:
    #     f.write(f"Accuracy: {acc_score:.4f}, Precision: {prec_score:.4f}, Recall: {rec_score:.4f}, F1 Score: {f_score:.4f}")

    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'CTGAN ROC curve (area = {roc_auc1})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"CTGAN.png")
    
if __name__ == '__main__':
    # 超参数
    n_epochs = 100
    learning_rate = 1e-5   
    latent_dim = 28 # 输入维度
    img_shape = 28*28
    percentage = 1 # 需要生成数据的比例
    batch_size = 8192
    LSTM_hidden_dim = 128
    LSTM_layers = 6
    LSTM_output_dim = 256
    GRU_hidden_dim = 128
    GRU_output_dim = LSTM_output_dim # 与LSTM_output_dim相同
    GRU_layers = 10
    Encoder_input_dim = GRU_output_dim # 与LSTM_output_dim相同
    Encoder_output_dim = 32
    Encoder_hidden_dim = 128
    CNN_input_channel = GRU_output_dim # 上一层输出的特征数
    CNN_output_dim = 16
    optimizer_name = "adamw"  # 可选 "adam", "adamw", "sgd", "rmsprop"
    
    

    #先划分训练集和测试集 -> 训练集拿去做数据增强和训练 -> 测试集拿去测试
    creditDataSet = pd.read_csv('Base_cleaned.csv',skiprows=1)
    # creditDataSet = pd.read_csv('creditcard_2023.csv',skiprows=1)
    
    minority_creditDataSet = pd.read_csv('Base_cleaned_minority.csv')

    credit_data_tensor = torch.tensor(creditDataSet.values, dtype=torch.float32)
    minority_num  = 0

    train_size = int(0.8 * len(credit_data_tensor))
    # val_size = int(0.1 * len(new_data_tensor))
    test_size = len(credit_data_tensor) - train_size

    creadit_data = CreditData(credit_data_tensor)
    train_dataset,test_dataset = random_split(creadit_data, [train_size, test_size])
    dataloader = DataLoader(train_dataset,batch_size=len(train_dataset))
    for (input,lable) in dataloader:
        lable = lable.unsqueeze(1)
        all_data = torch.concat([input,lable],dim=1)
    for dataset in train_dataset:
        _,label = dataset
        if label.item() ==1:minority_num+=1
    majority_num = len(train_dataset)-minority_num

    # all_data 是原本数据集的数据

    print("Start augementation...")
    print(majority_num-minority_num)
    new_data_tensor = ctganAugementation(minority_creditDataSet,majority_num-minority_num)#少数类进行样本增强
    
    # new_data_tensor = credit_data_tensor# 不使用数据增强
    # new_data_tensor = smoteAugemntation(all_data)#使用SMOTE进行数据增强
    # new_data_tensor = randomAugementation(all_data)#使用随机过采样进行数据增强
    # new_data_tensor = adasynAugementation(all_data)#使用ADASYN过采样进行数据增强
    print("Finish augementation...")
    new_train_data_tensor = torch.concat([all_data,new_data_tensor],dim=0)#CTGAN
    # new_train_data_tensor = new_data_tensor#SMOTE,ADASYN
    # new_train_data_tensor = new_data_tensor # 2023数据集
    train_dataset = CreditData(new_train_data_tensor)#经过数据增强后的数据集

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model = CreditCardClassifier(input_dim=latent_dim,LSTM_hidden_dim=LSTM_hidden_dim,LSTM_layers=LSTM_layers,LSTM_output_dim=LSTM_output_dim,
                                 GRU_hidden_dim=GRU_hidden_dim,GRU_output_dim=GRU_output_dim,GRU_layers=GRU_layers,
                                 Encoder_input_dim=Encoder_input_dim,Encoder_output_dim=Encoder_output_dim,Encoder_hidden_dim=Encoder_hidden_dim,
                                 CNN_input_channel=CNN_input_channel,CNN_output_dim=CNN_output_dim).to(device)
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99,weight_decay=0.001)

    criterion = nn.BCEWithLogitsLoss()
    # summary(model, input_size=(batch_size, 28))
    # criterion = nn.FocalLoss(gamma=2, alpha=0.25)

    # 选择一个学习率调度器
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)

    # summary(model, input_size=(batch_size,latent_dim))

    print("Start training...")
    train_model(model,train_dataloader,criterion,optimizer,n_epochs,device)
    print("Finish training...")

    # # model.load_state_dict(torch.load('classifier_model.pth'))

    print("Start evaluating...")
    evaluate_model(model,test_dataloader,device)
    print("Finish evaluating...")