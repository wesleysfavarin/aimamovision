@echo off
echo ========================================
echo Configurando o ambiente Python...
echo ========================================

REM Verifica se o Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Python não está instalado! Baixe e instale em https://www.python.org/downloads/
    pause
    exit /b
)

echo Criando ambiente virtual...
python -m venv venv

echo Ativando ambiente virtual...
call venv\Scripts\activate

echo Atualizando o pip...
pip install --upgrade pip

echo Instalando dependências...
pip install contourpy==1.3.1 ^
    cycler==0.12.1 ^
    fonttools==4.56.0 ^
    imageio==2.37.0 ^
    joblib==1.4.2 ^
    kiwisolver==1.4.8 ^
    lazy_loader==0.4 ^
    matplotlib==3.10.1 ^
    networkx==3.4.2 ^
    numpy==2.2.3 ^
    packaging==24.2 ^
    pandas==2.2.3 ^
    pillow==11.1.0 ^
    pyparsing==3.2.1 ^
    python-dateutil==2.9.0.post0 ^
    pytz==2025.1 ^
    scikit-image==0.25.2 ^
    scikit-learn==1.6.1 ^
    scipy==1.15.2 ^
    six==1.17.0 ^
    threadpoolctl==3.5.0 ^
    tifffile==2025.2.18 ^
    tk==0.1.0 ^
    tzdata==2025.1

echo ========================================
echo Instalacao concluida!
echo Para ativar o ambiente virtual, use: venv\Scripts\activate
echo Para rodar seu script, use: python seu_script.py
echo ========================================
pause
