# Detector de SPAM com KNN usando TF-IDF (dados em CSV)

# 1. Importar bibliotecas
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Ler dataset de um arquivo CSV
df = pd.read_csv("emails.csv")

textos = df["texto"]
classes = df["classe"]

# 3. Transformar texto em features numéricas (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)
y = classes

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

# 5. Treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 6. Avaliar o modelo
y_pred = knn.predict(X_test)
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

# 7. Testar com novas frases
novas_frases = [
    "Ganhe dinheiro fácil agora",
    "Reunião de status do projeto",
    "Promoção exclusiva só hoje",
    "Vamos sair para jantar?",
    "reunião hj mais tarde meu amigo",
    "reunião",
    "clique aqui para sua viagem gratis"
]

X_novas = vectorizer.transform(novas_frases)
predicoes = knn.predict(X_novas)

print("\nTestando novas frases:")
for frase, pred in zip(novas_frases, predicoes):
    print(f"'{frase}' -> {pred}")
