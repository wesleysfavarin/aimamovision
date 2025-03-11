
# script help you install :D hehehe

#!/bin/bash

echo "🚀 Preparando o ambiente para o projeto de Mamografia..."

# Verifica se o Python 3 está instalado
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 não encontrado! Instale o Python 3 e tente novamente."
    exit 1
fi

# Cria o ambiente virtual se ainda não existir
if [ ! -d "venv" ]; then
    echo "📦 Criando ambiente virtual..."
    python3 -m venv venv
fi

# Ativa o ambiente virtual
echo "✅ Ativando o ambiente virtual..."
source venv/bin/activate

# Atualiza o pip
echo "⬆️ Atualizando o pip..."
pip install --upgrade pip

# Instala as dependências necessárias
echo "📦 Instalando dependências..."
pip install \
    Pillow \
    matplotlib \
    numpy \
    pandas \
    scikit-learn \
    tensorflow \
    keras \
    opencv-python \
    flask \
    joblib \
    pathlib \
    tk \
    seaborn

# Exibe a mensagem final de sucesso
echo "🎉 Ambiente configurado com sucesso! Agora você pode rodar o seu projeto com:"
echo "   source venv/bin/activate"
echo "   python mamografia_predictor.py"
