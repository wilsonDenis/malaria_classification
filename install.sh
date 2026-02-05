#!/bin/bash

echo "======================================"
echo "VÉRIFICATION DE L'INSTALLATION"
echo "======================================"
echo ""

echo "[1] Vérification de Python..."
python3 --version

echo ""
echo "[2] Création de l'environnement virtuel..."
python3 -m venv env

echo ""
echo "[3] Activation de l'environnement..."
source env/bin/activate

echo ""
echo "[4] Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[5] Vérification de TensorFlow..."
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

echo ""
echo "======================================"
echo "INSTALLATION TERMINÉE!"
echo "======================================"
echo ""
echo "Pour lancer le projet:"
echo "  1. Activez l'environnement: source env/bin/activate"
echo "  2. Lancez l'entraînement: python3 main.py"
echo ""
