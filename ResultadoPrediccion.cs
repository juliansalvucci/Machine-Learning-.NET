namespace QuienEstaHablando
{
    //Clase para guardar predicciones
    public class ResultadoPrediccion
    {
        public bool PredictedLabel { get; set; } //va a ser verdadero o falso dependiendo si lo dijo un presidente o no.
        public float Probability { get; set; } //número entre 0 y 1 de que la persona haya dicho el diálogo.
        public float Score { get; set; } //Métrica que nos dice que tan seguro esta el algoritmo para realizar predicciones.
    }
}
