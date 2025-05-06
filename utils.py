import json
import os
import re

# Script para montar o JSON correto a partir dos dois arquivos gerados pelo pdfplucker

def alter_sections(source: str = "'D:/Users/B19943781742/Desktop/resultados"):
    """Alterar a seção de textos para o modelo mais funcional, enquanto o pdfplucker não é atualizado."""

    os.makedirs("jsons", exist_ok=True)

    for root, dirs, files in os.walk(source):
        for file in files:
            if file.endswith('.md'):
                md_file = os.path.join(root, file)
            elif file.endswith('.json'):
                json_file = os.path.join(root, file)

        # Read the markdown file
        md_content = ""
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Read the json
        json_content = ""
        with open(json_file, 'r', encoding='utf-8') as f:
            json_content = f.read()
        json_content = json.loads(json_content)

        pattern = r"!\[.*?\]\(data:image\/[a-zA-Z]+;base64,[^\)]+\)"

        # Check and remove base64 images
        for i in range(len(json_content['images'])):
            try:
                md_content = re.sub(pattern, f"<{json_content['images'][i]["ref"]}>", md_content)
            except:
                md_content = re.sub(pattern, "<!image>", md_content)

        json_content['sections'] = md_content

        filename = os.path.basename(json_file)
        filepath = os.path.join('teste', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_content, f, ensure_ascii=False, indent=4)

def reference_images(source: str = "'D:/Users/B19943781742/Desktop/resultados"):
    """Removendo as imagens de base64 para referenciar elas externamente"""
    for root, dirs, files in os.walk(source):
        for folder in dirs:
            if folder == 'images':
                # copy all images and paste do data/images
                folder_path = os.path.join(root, folder)
                for image in os.listdir(folder_path):
                    if image.endswith('.jpg') or image.endswith('.png'):
                        src = os.path.join(folder_path, image)
                        dst = os.path.join('data', 'images', image)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                            fdst.write(fsrc.read())
                        print(f"Copied {src} to {dst}")

                    