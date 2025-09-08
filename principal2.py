# Detector de SPAM com KNN usando TF-IDF

# 1. Importar bibliotecas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Criar dataset maior
textos = [
    "Compre já, promoção exclusiva",
    "Reunião amanhã às 10h",
    "Você ganhou um prêmio, clique aqui",
    "Favor revisar o relatório anexo",
    "Oferta grátis só hoje",
    "Vamos almoçar mais tarde?",
    "Clique agora para ganhar dinheiro fácil",
    "Encontro marcado para amanhã cedo",
    "Parabéns, você foi sorteado!",
    "Preciso da sua ajuda com o projeto",
    "Ganhe prêmios incríveis agora mesmo",
    "Relatório final entregue ao cliente",
    "Aproveite esta promoção imperdível",
    "O professor marcou prova para segunda",
    "Clique aqui e resgate seu cupom grátis",
    "Vamos viajar no próximo feriado?",
    "Promoção de tempo limitado, não perca",
    "Reunião adiada para sexta-feira",
    "Você foi selecionado para ganhar um prêmio",
    "Favor enviar a planilha atualizada"
]

classes = [
    "spam",  # Compre já...
    "ham",   # Reunião...
    "spam",  # Você ganhou...
    "ham",   # Favor revisar...
    "spam",  # Oferta grátis...
    "ham",   # Vamos almoçar...
    "spam",  # Clique agora...
    "ham",   # Encontro marcado...
    "spam",  # Parabéns...
    "ham",   # Preciso da sua ajuda...
    "spam",  # Ganhe prêmios...
    "ham",   # Relatório final...
    "spam",  # Aproveite promoção...
    "ham",   # Prova segunda...
    "spam",  # Cupom grátis...
    "ham",   # Viajar feriado...
    "spam",  # Promoção limitada...
    "ham",   # Reunião sexta...
    "spam",  # Selecionado prêmio...
    "ham"    # Planilha atualizada...
]

# 3. Transformar texto em features numéricas (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)
y = classes

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
    "Vamos sair para jantar?"
]

X_novas = vectorizer.transform(novas_frases)
predicoes = knn.predict(X_novas)

print("\nTestando novas frases:")
for frase, pred in zip(novas_frases, predicoes):
    print(f"'{frase}' -> {pred}")
