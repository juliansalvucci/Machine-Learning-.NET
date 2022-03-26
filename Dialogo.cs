namespace QuienEstaHablando
{
    using Microsoft.ML.Data;

    public class Dialogo //Esta clase nos permite conocer que datos analizar.
    {
        [LoadColumn(0)] //Indicar numero de columna del archivo .csv
        public string Hablante { get; set; }


        [LoadColumn(1)]
        public string Mensaje { get; set; }
    }
}
