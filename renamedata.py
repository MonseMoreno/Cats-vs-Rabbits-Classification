import os

def renombrar_imagenes(carpeta, prefijo, max_numero):
    imagenes = [f for f in os.listdir(carpeta) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    imagenes = sorted(imagenes)[:max_numero]

    for i, nombre_original in enumerate(imagenes, start=1):
        extension = os.path.splitext(nombre_original)[1]
        nuevo_nombre = f"{prefijo}.{i}{extension}"
        ruta_original = os.path.join(carpeta, nombre_original)
        nueva_ruta = os.path.join(carpeta, nuevo_nombre)

        if os.path.exists(nueva_ruta):
            os.remove(nueva_ruta)

        os.rename(ruta_original, nueva_ruta)
        print(f"{nombre_original} => {nuevo_nombre}")


carpeta = r"C:\Users\casas\Desktop\Cats-vs-Rabbits-Classification\data\train-images\dog"
prefijo = "dog"
max_numero = 1000

renombrar_imagenes(carpeta, prefijo, max_numero)
