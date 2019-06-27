# El práctico

Esta competencia ha sido creada a partir de la competencia de adopción de mascotas de PetFinder.my en Kaggle: https://www.kaggle.com/c/petfinder-adoption-prediction/overview. Se ha creado una versión más básica (sin imágenes por ejemplo) para usarla como práctica de la materia de *Aprendizaje Supervisado* de la *DiploDatos*.

## Kaggle

La competencia se puede acceder mediante [este link](https://www.kaggle.com/t/0bb80a8296854cfab77d9d9ec1d60fe1).

## Requerimientos del práctico

1. Superar el puntaje del baseline en la competencia kaggle.
1. Usar al menos un modelo más que los árboles de decisión.
1. Entregar el código fuente que generó el último archivo que se subió a kaggle. Tal código debe ser fácilmente ejecutable (virtualenv?) y debe estar documentado.

## Pasos a seguir

1. Crear una cuenta en kaggle.com. 
1. Sumarse a la competencia [acá](https://www.kaggle.com/t/0bb80a8296854cfab77d9d9ec1d60fe1).
    * Hacer click en "Join Competition".
    * Aceptar las reglas.
1. Pueden crear un equipo (Team). *Recomendamos hacerlo en equipos de dos alumnos*.
1. Pueden descargar los datos (Data), aunque también están incluidos en este repo.
1. Una vez realizada una predicción (ver ejemplo abajo), subir los resultados a kaggle haciendo click en "Submit Predictions" en la página principal de la competencia. Ahí deberán subir el archivo csv generado y describir (para sus registros) qué están subiendo.

## Un ejemplo

Adjuntamos una implementación que tiene por objetivo:

* Levantar los datos que usaremos.
* Crear un *baseline* para la competencia.
* Generar el archivo que se subirá a kaggle para su evaluación (*submission.csv*).
