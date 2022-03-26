
# ¿Quién está hablando?

A quick simple demo to showcase how to use ML.NET to perform text classification.

## Extra resources (in spanish)

 - [Machine Learning en .NET](https://thatcsharpguy.com/post/machine-learning-en-.net)
 - [Machine Learning con .NET y C# (YouTube)](https://youtu.be/AAAkgHiJNf4)
 - [Más Machine Learning con .NET y C# (YouTube)](https://youtu.be/eQj7vN0vIaQ)
 - [Twitter @io_exception](https://twitter.com/io_exception)

## Step-by-step  

### 1. Add `Microsoft.ML` package  

```shell
dotnet add package Microsoft.ML
```

### 2. Everything starts with the `MLContext`  

```chsarp
using Microsoft.ML;
// ... 
var context = new MLContext();
```

### 3. Create an input class  

```csharp
using Microsoft.ML.Data;

namespace QuienEstaHablando
{
    public class Dialogo
    {
        [LoadColumn(0)]
        public string Hablante { get; set; }

        [LoadColumn(1)]
        public string Mensaje { get; set; }
    }
}
```

### 4. Read data  

```csharp
IDataView allData = context.Data.LoadFromTextFile<Dialogo>(
    path: "quien_esta_hablando.csv",
    separatorChar: ',',
    hasHeader: true
);
```

### 5. Train-Test split

```csharp
var splits = context.Data.TrainTestSplit(
    data: allData,
    testFraction: 0.2
);
```

### 6. Add Label transformation code  

```csharp
var booleanMap = context.Data.LoadFromEnumerable(new[]
{
    new { InputValue = "amlo", Value = true },
    new { InputValue = "lopez_gatell", Value = false }
});

var transformLabel = context.Transforms.Conversion.MapValue(
    outputColumnName: "Label",
    lookupMap: booleanMap,
    keyColumn: booleanMap.Schema["InputValue"],
    valueColumn: booleanMap.Schema["Value"],
    inputColumnName: nameof(Dialogo.Hablante)
);
```

### 7. Add Features transformation code  

```chsarp
var textTransformation = context.Transforms.Text.FeaturizeText(
    outputColumnName: "Features",
    inputColumnName: nameof(Dialogo.Mensaje)
);
```

### 8. Add classifier

```csharp
var classifier = context.BinaryClassification.Trainers.SdcaLogisticRegression();
```

### 9. Build pipeline

```csharp
var pipeline = transformLabel
    .Append(textTransformation)
    .Append(classifier);
```

### 10. Train the model  

```csharp
 var trainedModel = pipeline.Fit(splits.TrainSet);
```

### 11. Create a prediction class

```csharp
public class PrediccionDialogo
{
    public bool PredictedLabel { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
```

### 12. Create a prediction engine from the trained model

```csharp
var predictionEngine = context.Model.CreatePredictionEngine<Dialogo, PrediccionDialogo>(trainedModel);
```

Now we can predict values!

```csharp
var prediction = predictionEngine.Predict(new Dialogo { Mensaje = "El cubrebocas no sirve para la pandemia, la secretaría de salud, quedate en casa" });

Console.WriteLine($"{prediction.PredictedLabel} {prediction.Score} {prediction.Probability}");
```

### 13. Evaluate our model

```csharp
var evaluation = context.BinaryClassification.Evaluate(
    trainedModel.Transform(splits.TestSet)
    );

Console.WriteLine(evaluation.Accuracy);
```

### 14. Save trained model

```csharp
context.Model.Save(trainedModel, allData.Schema, "model.zip");
```

### 15. Load saved model

```csharp
var savedModel = context.Model.Load("model.zip", out var schema);
```

### 16. Make more predictions

```csharp
var predictionEngine = context.Model.CreatePredictionEngine<Dialogo, PrediccionDialogo>(savedModel);

// var mensaje = "Un negocio jugoso ilícito siempre lleva el visto bueno del presidente";
var mensaje = "Quiero recordarles cómo funciona el Sistema Nacional de Vigilancia Epidemiológica";
var prediction = predictionEngine.Predict(new Dialogo { Mensaje = mensaje });

var mensajero = prediction.PredictedLabel ? "AMLO" : "López-Gatell";
Console.WriteLine($"{mensajero}: {mensaje}");
```# Machine-Learning-.NET
