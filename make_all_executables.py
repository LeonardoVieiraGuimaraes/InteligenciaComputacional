import os
import glob
import shutil
import subprocess

# Caminho base onde estão os notebooks
base_dir = "Trabalho03/src"
# Pasta central para todos os executáveis
output_dir = "executaveis_py"
os.makedirs(output_dir, exist_ok=True)

# Procura todos os notebooks .ipynb recursivamente
notebooks = glob.glob(os.path.join(base_dir, "**", "*.ipynb"), recursive=True)

for nb_path in notebooks:
    # Converte notebook para script .py
    cmd_convert = f'jupyter nbconvert --to script "{nb_path}"'
    print(f"Convertendo: {nb_path}")
    subprocess.run(cmd_convert, shell=True, check=True)

    # Nome do script gerado
    py_path = nb_path.replace(".ipynb", ".py")
    if os.path.exists(py_path):
        # Move o script .py para a pasta centralizada
        dest_path = os.path.join(output_dir, os.path.basename(py_path))
        shutil.move(py_path, dest_path)
        print(f"Script movido para: {dest_path}")
    else:
        print(f"Script não encontrado: {py_path}")

print("Processo concluído! Todos os scripts .py estão em:", output_dir)
