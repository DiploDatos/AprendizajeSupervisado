# El práctico

Esta competencia ha sido creada a partir de la competencia de clasificación de visitas a Walmart en Kaggle: https://www.kaggle.com/c/walmart-recruiting-trip-type-classification/overview. Se ha creado una versión simplificada para usarla como práctica de la materia de *Aprendizaje Supervisado* de la *DiploDatos*.

## Requerimientos del práctico

1. Superar el puntaje del baseline en la competencia kaggle.
1. Usar al menos un modelo más que los árboles de decisión que se entrega.
1. Entregar el código fuente que generó el último archivo que se subió a kaggle. Tal código debe ser fácilmente ejecutable (virtualenv?) y debe estar documentado.

## Pasos a seguir

1. Crear una cuenta en kaggle.com. 
1. Sumarse a la competencia [acá](https://www.kaggle.com/t/f445b42d310243318bed5cb76c2db2df).
    * Hacer click en "Join Competition".
    * Aceptar las reglas.
1. Crear un equipo (Team): *hacer el práctico en equipos de tres o cuatro alumnos*.
1. Pueden descargar los datos (Data), aunque también están incluidos en este repo.
1. Una vez realizada una predicción (ver ejemplo abajo), subir los resultados a kaggle haciendo click en "Submit Predictions" en la página principal de la competencia. Ahí deberán subir el archivo csv generado y describir (para sus registros) qué están subiendo.

## Un ejemplo

Adjuntamos una implementación que tiene por objetivo:

* Levantar los datos que usaremos.
* Analizar de una manera simple los datos.
* Preparar los datos para procesarlos con un modelo en particular.
* Crear un *baseline* para la competencia.
* Generar el archivo que se subirá a kaggle para su evaluación (*submission.csv*).

## Subir una predicción a Kaggle

En el ejemplo de baseline que se entrega, se genera un archivo en el path *data/submission.csv*. Tal archivo es un csv con un formato en particular, que asigna a números de visita en el conjunto de test, una predicción del tipo de viaje. 
Ese archivo debe ser subido a kaggle como lo explicamos arriba: haciendo click en "Submit Predictions" en la página principal de la competencia.

## Algunas consideraciones

* Dividimos los datos de entrenamiento en "train" y "validation". Pero hicimos cross-validation (CV) sólo sobre "train".
* Estamos usando grid search, sólo porque se nos ocurrió. Sirven otros métodos para determinar hiperparámetros como random search?
* Usamos one-hot encoding para lidear con las variables categóricas. Para ello, usamos la función "get_dummies" de pandas. Hicimos esto último sobre todo el universo de datos, asumiendo el riesgo que esto conlleva (puede darse el caso en que creemos muchas nuevas features para valores que no son vistos en los datos de "train").
* No estamos usando los datos de Upc y FinelineNumber. Estos datos parecen ser muy relevantes, quizás los puedan agregar.
