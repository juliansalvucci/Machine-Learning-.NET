namespace QuienEstaHablando
{
    //Este algoritmo se encarga de predecir a partir de una determinada frase, si la dijo o no
    //el presidente, tomando como 71.000 registro de frases registradas en "quien_esta_hablando.csv"

    using Microsoft.ML; //Se debe importar el nuget
    using System;

    class Program
    {
        static void Main(string[] args)
        {
            var contexto = new MLContext(); //Se debe crear un contexto para ML

            var todosLosDatos = contexto.Data.LoadFromTextFile<Dialogo>(  //cargar información del .csv en la clase Dialogo
                "quien_esta_hablando.csv",
                separatorChar: ',', //Indicar caracter de separación .csv usa comas.
                hasHeader: true //El archivo posee header.
            );

            var conjuntos = contexto.Data.TrainTestSplit( //Conjunto de entrenamiento, Data especifica la fuente de datos.
                data: todosLosDatos,
                testFraction: 0.2 //20% de registros de pureba, varía de problema en problema, mientras mayor sea es mejor.
            );

            //Profundizar otras formas de convertir palabras a números
            var transformacionTexto = contexto.Transforms.Text.FeaturizeText( //Transformo texto a números, para poder analizarlos.
                outputColumnName: "Features", //Nombre de columna de salida con los datos ya convertidos a números.
                inputColumnName: nameof(Dialogo.Mensaje) //Toma la columna mensaje para convertir a números.
            );

            var mapaBooleano = contexto.Data.LoadFromEnumerable(new[] { //Creo un mapa booleano
                new { Entrada = "amlo", Valor = true },  //El mapa consta de un arreglo de objetos.
                new { Entrada = "lopez_gatell", Valor = false },
            });

            var transformacionEtiqueta = contexto.Transforms.Conversion.MapValue( //Conversión a mapa
                outputColumnName: "Label", //Indico nombre de la columna de salida.
                lookupMap: mapaBooleano,//Diccionario mediante el cual realiza la transformación.
                keyColumn: mapaBooleano.Schema["Entrada"], //Llave con valor de entrada
                valueColumn: mapaBooleano.Schema["Valor"], //Valor con valor de valor(valga la redundancia jaja)
                inputColumnName: nameof(Dialogo.Hablante) 
            );

            //Clasificador
            var clasificador = contexto.BinaryClassification.Trainers.SdcaLogisticRegression(); //clasificación binaria entre dos clases. Regresión logística sencilla.

            //Crear pipeline para poder entrenar el modelo
            var pipeline = transformacionTexto  //a la transformación a texto le añade la etiqueta y el clasificador.
                .Append(transformacionEtiqueta)
                .Append(clasificador);

            //Entrenar pipeline, le paso el conjunto de datos de entrenamiento.
            var modeloEntrenado = pipeline.Fit(conjuntos.TrainSet);

            //Evaluación de desempeño del algoritmo.
            var evaluacion = contexto.BinaryClassification.Evaluate(
                modeloEntrenado.Transform(conjuntos.TestSet) //Tranform, hace transformaciones de texto, aplica el clasificador, usa el valor real de las etiquetas, sus valores en booleanos y accede a las predicciones del clasificador.
            );
            
            //Mostrar la exactitud del algoritmo.
            Console.WriteLine($"Tu algoritmo es {evaluacion.Accuracy * 100}% correcto");

            // En lugar de estar entrenando el modelo constantemente, podemos leer el que ya entrenamos
            // y realizar predicciones con él: 

            //Guardo modelo entrenado para poder usarlo luego
            contexto.Model.Save(modeloEntrenado, todosLosDatos.Schema, "modelo.zip")

            //Esto es para comenzar a usar en producción el modelo entrenado.
            var modeloGuardado = contexto.Model.Load("modelo.zip", out var schema);

            //Motor de predicción que solicita clases de entrada y de salida. y finalmente el modelo entrenado.
            var motorPrediccion = contexto.Model.CreatePredictionEngine<Dialogo, ResultadoPrediccion>(modeloGuardado);

            //Se realiza la predicción.
            ResultadoPrediccion resultado = motorPrediccion.Predict(new Dialogo 
            { 
                //Le paso el mensaje que se desea predecir.
                Mensaje = args[0]
            });

            Console.WriteLine($"Lo dijo el presidente {resultado.PredictedLabel} {resultado.Probability}");
            //Output "Lo dijo el presidente True 0.84911805" esto es como una probabilidad del 85%
            
        }
    }
}
