import os
import pandas as pd
import re
from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np
import logging
import warnings

# Beta-Gamma tarzı uyarılar çıkıyordu araştırdığım kadarıyla bir sıkıntı yaratmıyor
warnings.filterwarnings("ignore", category=UserWarning, message=r".*beta.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*gamma.*")

# diğer uyarıları bastırmak için araştırdığım kadarıyla sıkıntı olmuyor
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

# Proje klasöründeki logs dizini
logs_dir = os.path.join(os.getcwd(), 'logs')

# Log dosyasının yüklenmesi ve işlenmesi
log_file_path = os.path.join(logs_dir, 'weblog.log')

# Log dosyasını satır satır okuma ve regex ile ayrıştırma
pattern = r'(?P<IP>\S+) \S+ \S+ \[(?P<Timestamp>[^\]]+)\] "(?P<Request>[^"]+)" (?P<Status>\d{3}) (?P<Size>\S+) "(?P<Referrer>[^"]*)" "(?P<UserAgent>[^"]*)"'
data = []

with open(log_file_path, 'r') as file:
    for line in file:
        match = re.match(pattern, line)
        if match:
            data.append(match.groupdict())

# Elde edilen verileri DataFrame'e dönüştürme
log_df = pd.DataFrame(data)

# Timestamp sütununu datetime formatına dönüştürme
log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')

# Eksik ve hatalı satırları temizleme
log_df.dropna(inplace=True)

# BERT model ve tokenizer'ın yüklenmesi
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# BERT işlemlerini Cuda var ise onda çalıştırmak için yoksa cpu da çalışıyor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Vektörleştirmek istediğimiz sütunlar
text_columns = ['IP', 'Timestamp', 'Request']

# Tüm sütunları str türüne dönüştürme
log_df[text_columns] = log_df[text_columns].astype(str)

# Sütunları birleştirerek tek bir metin sütunu oluşturma
log_df['Combined_Text'] = log_df[text_columns].agg(' '.join, axis=1)

# BERT ile vektörleştirme fonksiyonu
def bert_encode(texts, batch_size=32):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            embeddings = outputs[0][:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)
    return all_embeddings

# Tüm metinleri BERT ile vektörleştirme
log_df['Combined_BERT'] = bert_encode(log_df['Combined_Text'].tolist())

# Vektörleştirilmiş logları kaydetme
encoded_logs_path = os.path.join(logs_dir, 'bert_encoded_logs.pkl')
log_df.to_pickle(encoded_logs_path)
print(f"BERT vektörleştirme işlemi tamamlandı ve sonuçlar {encoded_logs_path} olarak kaydedildi.")

# Vektörleri FAISS index'e ekleme ve kaydetme
dimension = log_df['Combined_BERT'][0].shape[0]
index = faiss.IndexFlatL2(dimension)

vectors = np.vstack(log_df['Combined_BERT'].values).astype('float32')
index.add(vectors)

# FAISS index'i kaydetme
faiss_index_path = os.path.join(logs_dir, 'faiss_index.index')
faiss.write_index(index, faiss_index_path)
print(f"FAISS index oluşturuldu ve {faiss_index_path} olarak kaydedildi.")
