# El práctico


Esta competencia ha sido creada a partir de un datasets sobre una recopilación de datos médicos y demográficos de los pacientes, junto con su estado de diabetes (positivo o negativo). Los datos incluyen características como la edad, el sexo, el índice de masa corporal (IMC), la hipertensión, las enfermedades cardíacas, el historial de tabaquismo, el nivel de HbA1c y el nivel de glucosa en sangre. Este conjunto de datos se puede usar para construir modelos de aprendizaje automático para predecir la diabetes en pacientes en función de su historial médico e información demográfica. Esto puede ser útil para los profesionales de la salud en la identificación de pacientes que pueden estar en riesgo de desarrollar diabetes y en el desarrollo de planes de tratamiento personalizados. Además, los investigadores pueden utilizar el conjunto de datos para explorar las relaciones entre varios factores médicos y demográficos y la probabilidad de desarrollar diabetes.


## Atributos del dataset:
 * **Gender**: Gender refers to the biological sex of the individual, which can have an impact on their susceptibility to diabetes. There are three categories in it male ,female and other.
 * **age**: Age is an important factor as diabetes is more commonly diagnosed in older adults.Age ranges from 0-80 in our dataset.
 * **hypertension**: Hypertension is a medical condition in which the blood pressure in the arteries is persistently elevated. It has values a 0 or 1 where 0 indicates they don’t have hypertension and for 1 it means they have hypertension.
 * **heart_disease**: Heart disease is another medical condition that is associated with an increased risk of developing diabetes. It has values a 0 or 1 where 0 indicates they don’t have heart disease and for 1 it means they have heart disease.
 * **smoking_history**: Smoking history is also considered a risk factor for diabetes and can exacerbate the complications associated with diabetes.In our dataset we have 5 categories i.e not current,former,No Info,current,never and ever.
 * **bmi**: BMI (Body Mass Index) is a measure of body fat based on weight and height. Higher BMI values are linked to a higher risk of diabetes. The range of BMI in the dataset is from 10.16 to 71.55. BMI less than 18.5 is underweight, 18.5-24.9 is normal, 25-29.9 is overweight, and 30 or more is obese.
 * **HbA1c_level**: HbA1c (Hemoglobin A1c) level is a measure of a person's average blood sugar level over the past 2-3 months. Higher levels indicate a greater risk of developing diabetes. Mostly more than 6.5% of HbA1c Level indicates diabetes.
 * **blood_glucose_level**: Blood glucose level refers to the amount of glucose in the bloodstream at a given time. High blood glucose levels are a key indicator of diabetes.
 * **diabetes**: Diabetes is the target variable being predicted, with values of 1 indicating the presence of diabetes and 0 indicating the absence of diabetes.


## Requerimientos del práctico

1. Realizar un análisis exploratorio de datos, y utilizar lo aprendido para generar su mejor modelo.
1. Superar el puntaje del baseline (público) en la competencia kaggle.
1. Usar al menos 3 modelos distintos al del árbol de decisión que se explora como baseline (se puede explorar más profundamente el árbol de decisión, pero aún así deberán explorar otros modelos).
1. Entregar un notebook con el análisis exploratorio de datos y el código con los 3 mejores modelos entregados en la competencia (de acuerdo a los resultados obtenidos).

## Pasos a seguir

1. Crear una cuenta en kaggle.com.
1. Sumarse a la competencia [acá](https://www.kaggle.com/t/ffbd02f08a14443194f84dff7b8c2f1f).
    * Hacer click en "Join Competition".
    * Aceptar las reglas.
1. Crear un equipo (Team): El trabajo se evaluará en los grupos asignados.
1. Pueden descargar los datos (Data), aunque también están incluidos en este repo.
1. Una vez realizada una predicción (ver ejemplo abajo), subir los resultados a kaggle haciendo click en "Submit Predictions" en la página principal de la competencia. Ahí deberán subir el archivo csv generado y describir (para sus registros) qué están subiendo.

## Un ejemplo

Adjuntamos una implementación que tiene por objetivo:

* Levantar los datos que usaremos.
* Analizar de una manera simple los datos.
* Preparar los datos para procesarlos con un modelo en particular.
* Crear un *baseline* para la competencia.
* Generar el archivo que se subirá a kaggle para su evaluación.

## Subir una predicción a Kaggle

En el ejemplo de baseline que se entrega, se genera un archivo en el path *data/submission.csv*. Tal archivo es un csv con un formato en particular, que asigna a números de visita en el conjunto de test, una predicción de si fue teletransportado o no a otra dimensión alternativa.
Ese archivo debe ser subido a kaggle como lo explicamos arriba: haciendo click en "Submit Predictions" en la página principal de la competencia.

## Algunas consideraciones

* En el baseline solo se utiliza cross-validation (mediante *GridSearchCV*) para evaluar el modelo, son libres de generar un subconjunto de validación aparte del conjunto de entrenamiento si así lo desean.
* Los *features* escogidos no tienen ningún análisis y son casi por defecto. Parte del proceso de encontrar un buen modelo es ver como trabajar dichos features.
* La métrica a optimizar será el accuracy score.
