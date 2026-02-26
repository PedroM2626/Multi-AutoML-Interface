import sys
import importlib

# Forçar reload do módulo h2o_utils
if 'src.h2o_utils' in sys.modules:
    importlib.reload(sys.modules['src.h2o_utils'])

# Testar a função
from src.h2o_utils import train_h2o_model
import inspect

print("Assinatura da função:")
print(inspect.signature(train_h2o_model))

# Verificar se o parâmetro seed existe
sig = inspect.signature(train_h2o_model)
if 'seed' in sig.parameters:
    print("OK: Parametro 'seed' encontrado!")
else:
    print("ERRO: Parametro 'seed' NAO encontrado!")
