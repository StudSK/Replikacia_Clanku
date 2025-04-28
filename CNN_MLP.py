import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tree_sitter import Language, Parser
import os
# --- 1. Инициализация AST парсера ---
def init_ast_parser():
    try:
        PYTHON_LANGUAGE = Language(r'C:\Users\turok\PycharmProjects\PythonProject\Clanok\tree-sitter-languages\build\my-languages.so', 'python')

        parser = Parser()
        parser.set_language(PYTHON_LANGUAGE)
        return parser
    except Exception as e:
        print(f"Ошибка при инициализации парсера AST: {e}")
        print("Продолжаем без AST")
        return None

# --- 2. Извлечение AST токенов ---
def extract_ast_tokens(code, parser):
    if parser is None or not isinstance(code, str):
        return []
    try:
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens = []

        def traverse(node, depth=0):
            tokens.append(f"{node.type}_{depth}")  # <-- добавляем глубину в токен
            for child in node.children:
                traverse(child, depth + 1)
                traverse(child, depth + 1)

        traverse(root_node)
        return tokens
    except:
        return []


# --- 3. Построение словаря токенов ---
def build_vocab(token_lists, min_freq=1):
    from collections import Counter
    counter = Counter(token for tokens in token_lists for token in tokens)
    vocab = {token: idx+1 for idx, (token, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab["<PAD>"] = 0
    return vocab

def encode_tokens(tokens, vocab, max_len=100):
    ids = [vocab.get(token, 0) for token in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

# --- 4. Загрузка данных ---
def load_data(csv_path, parser, vocab=None, max_len=100):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if 'defects' not in df.columns:
        raise ValueError("Столбец 'defects' не найден")

    # AST обработка
    if 'lOCode' in df.columns:
        token_lists = df['lOCode'].apply(lambda x: extract_ast_tokens(x, parser)).tolist()

        if vocab is None:
            vocab = build_vocab(token_lists)

        ast_encoded = [encode_tokens(tokens, vocab, max_len) for tokens in token_lists]
        ast_encoded = np.array(ast_encoded)
    else:
        ast_encoded = np.zeros((len(df), max_len))  # Заполняем нулями

    # Метрики
    metrics_cols = [col for col in df.columns if col not in ['defects', 'lOCode']]
    metrics = df[metrics_cols].values
    labels = df['defects'].values

    return ast_encoded, metrics, labels, vocab

# --- 5. Модель ---
# --- Улучшенная модель CNN + BiLSTM + Attention + Fusion ---
class CNN_BiLSTM_Attention_Fusion(nn.Module):
    def __init__(self, vocab_size, metrics_size, embedding_dim=64, cnn_out=64, lstm_hidden=64):
        super(CNN_BiLSTM_Attention_Fusion, self).__init__()

        # 1. AST ветка
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_out, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=cnn_out, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)

        # Attention слой
        self.attention = nn.Linear(lstm_hidden * 2, 1)

        # 2. MLP ветка для метрик
        self.mlp_metrics = nn.Sequential(
            nn.Linear(metrics_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 3. Fusion слой
        self.fusion = nn.Sequential(
            nn.Linear((lstm_hidden * 2) + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, ast, metrics):
        # AST путь
        x_ast = self.embedding(ast)  # (batch_size, seq_len, embedding_dim)
        x_ast = x_ast.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x_ast = self.pool(torch.relu(self.conv1(x_ast)))  # (batch_size, cnn_out, seq_len/2)
        x_ast = x_ast.permute(0, 2, 1)  # (batch_size, seq_len/2, cnn_out)

        lstm_out, _ = self.lstm(x_ast)  # (batch_size, seq_len/2, lstm_hidden*2)

        # Attention механизм
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_len/2, 1)
        x_ast = (lstm_out * attn_weights).sum(dim=1)  # (batch_size, lstm_hidden*2)

        # Metrics путь
        x_metrics = self.mlp_metrics(metrics)  # (batch_size, 32)

        # Объединение
        fusion_input = torch.cat([x_ast, x_metrics], dim=1)  # (batch_size, lstm_hidden*2 + 32)
        output = self.fusion(fusion_input)
        return output


# --- 6. Тренировка и оценка ---
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for ast, metrics, y in train_loader:
        ast, metrics, y = ast.to(device), metrics.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(ast, metrics)
        loss = criterion(outputs.view(-1), y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for ast, metrics, y in test_loader:
            ast, metrics, y = ast.to(device), metrics.to(device), y.to(device)
            output = model(ast, metrics).cpu().numpy()
            preds.extend(output)
            labels.extend(y.cpu().numpy())

    preds_bin = [1 if p >= 0.5 else 0 for p in preds]
    return f1_score(labels, preds_bin, zero_division=1), roc_auc_score(labels, preds), labels, preds_bin


# --- 7. Графики ---
def plot_metrics(losses, f1s, aucs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Loss")
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(f1s, label="F1")
    plt.plot(aucs, label="AUC")
    plt.title("Metrics")
    plt.legend()
    plt.show()

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Defect", "Defect"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# --- 8. Основной запуск на одном датасете ---
def main(csv_path):
    parser = init_ast_parser()
    ast_data, metrics_data, labels, vocab = load_data(csv_path, parser)

    X_ast_train, X_ast_test, X_metrics_train, X_metrics_test, y_train, y_test = train_test_split(
        ast_data, metrics_data, labels, test_size=0.3, random_state=42, stratify=labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Преобразуем в тензоры
    X_ast_train = torch.tensor(X_ast_train, dtype=torch.long)
    X_ast_test = torch.tensor(X_ast_test, dtype=torch.long)
    X_metrics_train = torch.tensor(X_metrics_train, dtype=torch.float32)
    X_metrics_test = torch.tensor(X_metrics_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Создаем датасеты
    train_dataset = TensorDataset(X_ast_train, X_metrics_train, y_train)
    test_dataset = TensorDataset(X_ast_test, X_metrics_test, y_test)

    # Создаем веса для классов (балансировка)
    class_sample_count = np.array([len(np.where(y_train.numpy() == t)[0]) for t in np.unique(y_train.numpy())])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in y_train.numpy()])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type(torch.DoubleTensor), len(samples_weight))

    # Лоадеры
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Модель
    model = CNN_BiLSTM_Attention_Fusion(vocab_size=len(vocab), metrics_size=X_metrics_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Для логов
    losses, f1s, aucs = [], [], []

    # Тренировка
    for epoch in range(100):
        loss = train_model(model, train_loader, criterion, optimizer, device)
        f1, auc, _, _ = evaluate_model(model, test_loader, device)
        losses.append(loss)
        f1s.append(f1)
        aucs.append(auc)
        print(f"Epoch {epoch+1:02d}: Loss={loss:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # Финальная оценка
    f1, auc, labels_pred, preds_bin = evaluate_model(model, test_loader, device)
    print(f"\nFinal F1: {f1:.4f}, AUC: {auc:.4f}")

    # Визуализация
    plot_metrics(losses, f1s, aucs)
    plot_confusion_matrix(labels_pred, preds_bin)

    return {"dataset": csv_path, "final_f1": f1, "final_auc": auc}


# --- 9. Запуск на всех датасетах ---
if __name__ == "__main__":
    datasets = ["cm1.csv", "kc1.csv", "jm1.csv"]  # Замени здесь на свои датасеты
    all_results = []

    for dataset in datasets:
        print(f"\n==== Running on {dataset} ====")
        result = main(dataset)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    print("\n==== Final Results ====")
    print(results_df)

    results_df.to_csv("final_results.csv", index=False)
